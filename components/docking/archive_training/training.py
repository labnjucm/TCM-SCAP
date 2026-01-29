import copy
import numpy as np
from rdkit.Chem import RemoveAllHs
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import torch

from confidence.dataset import ListDataset
from utils import so3, torus
from utils.molecules_utils import get_symmetry_rmsd
from utils.sampling import randomize_position, sampling
from utils.diffusion_utils import get_t_schedule

"""
loss_function 用于计算不同预测值（如平移、旋转、扭转等）的损失。这些损失加权后合并为总损失。支持多个模块的损失计算，包括：

平移分量(Translation Component)
旋转分量(Rotation Component)
扭转分量(Torsion Component)
主链分量(Backbone Component)
侧链分量(Sidechain Component)
该函数还允许通过权重调整不同损失对总损失的贡献。
"""
def loss_function(tr_pred, rot_pred, tor_pred, sidechain_pred, data, t_to_sigma, device, tr_weight=1, rot_weight=1,
                  tor_weight=1, backbone_weight=0, sidechain_weight=0, apply_mean=True, no_torsion=False):
    """
    tr_pred, rot_pred, tor_pred, sidechain_pred: 模型对平移、旋转、扭转和侧链的预测值。
    data: 包含真实值和相关信息的数据。
    t_to_sigma: 将时间步长 t 转换为噪声标准差的函数。
    device: 计算设备（如 cuda 或 cpu）。
    tr_weight, rot_weight, tor_weight: 分别表示平移、旋转和扭转损失的权重。
    backbone_weight, sidechain_weight: 分别表示主链和侧链损失的权重。
    apply_mean: 是否对损失取平均值。
    no_torsion: 是否禁用扭转损失。
    """
    
    #转换时间步长为标准差
    tr_sigma, rot_sigma, tor_sigma = t_to_sigma(
        *[torch.cat([d.complex_t[noise_type] for d in data]) if device.type == 'cuda' else data.complex_t[noise_type]
          for noise_type in ['tr', 'rot', 'tor']])
    mean_dims = (0, 1) if apply_mean else 1

    # translation component平移损失
    tr_score = torch.cat([d.tr_score for d in data], dim=0) if device.type == 'cuda' else data.tr_score
    tr_sigma = tr_sigma.unsqueeze(-1)
    tr_loss = ((tr_pred.cpu() - tr_score.cpu()) ** 2 * tr_sigma.cpu() ** 2).mean(dim=mean_dims)
    tr_base_loss = (tr_score ** 2 * tr_sigma ** 2).mean(dim=mean_dims).detach()

    # rotation component旋转损失
    rot_score = torch.cat([d.rot_score for d in data], dim=0) if device.type == 'cuda' else data.rot_score
    rot_score_norm = so3.score_norm(rot_sigma.cpu()).unsqueeze(-1)
    rot_loss = (((rot_pred.cpu() - rot_score.cpu()) / rot_score_norm) ** 2).mean(dim=mean_dims)
    rot_base_loss = ((rot_score.cpu() / rot_score_norm) ** 2).mean(dim=mean_dims).detach()

    # torsion component扭转损失
    if not no_torsion:
        edge_tor_sigma = torch.from_numpy(
            np.concatenate([d.tor_sigma_edge for d in data] if device.type == 'cuda' else data.tor_sigma_edge))
        tor_score = torch.cat([d.tor_score for d in data], dim=0) if device.type == 'cuda' else data.tor_score
        tor_score_norm2 = torch.tensor(torus.score_norm(edge_tor_sigma.cpu().numpy())).float()
        tor_loss = ((tor_pred.cpu() - tor_score.cpu()) ** 2 / tor_score_norm2)
        tor_base_loss = ((tor_score.cpu() ** 2 / tor_score_norm2)).detach()
        # 如果需要取平均值
        if apply_mean:
            tor_loss, tor_base_loss = tor_loss.mean() * torch.ones(1, dtype=torch.float), tor_base_loss.mean() * torch.ones(1, dtype=torch.float)
        else:
            index = torch.cat([torch.ones(d['ligand'].edge_mask.sum()) * i for i, d in
                               enumerate(data)]).long() if device.type == 'cuda' else data['ligand'].batch[
                data['ligand', 'ligand'].edge_index[0][data['ligand'].edge_mask]]
            num_graphs = len(data) if device.type == 'cuda' else data.num_graphs
            t_l, t_b_l, c = torch.zeros(num_graphs), torch.zeros(num_graphs), torch.zeros(num_graphs)
            c.index_add_(0, index, torch.ones(tor_loss.shape))
            c = c + 0.0001
            t_l.index_add_(0, index, tor_loss)
            t_b_l.index_add_(0, index, tor_base_loss)
            tor_loss, tor_base_loss = t_l / c, t_b_l / c
    else:
        if apply_mean:
            tor_loss, tor_base_loss = torch.zeros(1, dtype=torch.float), torch.zeros(1, dtype=torch.float)
        else:
            tor_loss, tor_base_loss = torch.zeros(len(rot_loss), dtype=torch.float), torch.zeros(len(rot_loss), dtype=torch.float)

    #主链损失
    if backbone_weight > 0:
        backbone_vecs = torch.cat([d['receptor'].side_chain_vecs.cpu() for d in data], dim=0) if device.type == 'cuda' else data['receptor'].side_chain_vecs
        backbone_vecs = backbone_vecs[:, 4:]
        backbone_pred = sidechain_pred[:, 4:]

        backbone_base_loss = (backbone_vecs ** 2).detach().mean(dim=1) + 0.0001
        backbone_loss = ((backbone_pred.cpu() - backbone_vecs) ** 2).mean(dim=1) / backbone_base_loss.mean()
        backbone_base_loss = backbone_base_loss / backbone_base_loss.mean()
        if apply_mean:
            backbone_loss, backbone_base_loss = backbone_loss.mean() * torch.ones(1, dtype=torch.float), backbone_base_loss.mean() * torch.ones(1, dtype=torch.float)
        else:
            index = torch.cat([torch.ones((d['receptor'].pos.shape[0])) * i for i, d in enumerate(data)], dim=0).long() if device.type == 'cuda' else data['receptor'].batch
            num_graphs = len(data) if device.type == 'cuda' else data.num_graphs
            s_l, s_b_l, c = torch.zeros(num_graphs), torch.zeros(num_graphs), torch.zeros(num_graphs)
            c.index_add_(0, index, torch.ones(backbone_loss.shape[0]))
            c = c + 0.0001
            s_l.index_add_(0, index, backbone_loss)
            s_b_l.index_add_(0, index, backbone_base_loss)
            backbone_loss, backbone_base_loss = s_l / c, s_b_l / c
    else:
        if apply_mean:
            backbone_loss, backbone_base_loss = torch.zeros(1, dtype=torch.float), torch.zeros(1, dtype=torch.float)
        else:
            backbone_loss, backbone_base_loss = torch.zeros(len(rot_loss), dtype=torch.float), torch.zeros(len(rot_loss), dtype=torch.float)

    #侧链损失
    if sidechain_weight > 0:
        sidechain_vecs = torch.cat([d['receptor'].side_chain_vecs.cpu() for d in data],
                                  dim=0) if device.type == 'cuda' else data['receptor'].side_chain_vecs
        chi_angles = sidechain_vecs[:, :4].to(device)
        chi_pred = sidechain_pred[:, :4].to(device)

        chi_pred = torch.where(torch.isnan(chi_angles), torch.zeros_like(chi_angles, device=device), chi_pred)
        chi_angles = torch.where(torch.isnan(chi_angles), torch.zeros_like(chi_angles, device=device), chi_angles)

        difference = torch.abs(chi_pred - chi_angles)
        difference = torch.min(difference, 1 - difference) # angles are circular and 360 degrees = 1

        sidechain_base_loss = (chi_angles ** 2).detach().mean(dim=1) + 0.0001
        sidechain_loss = (difference ** 2).mean(dim=1) / sidechain_base_loss.mean()
        sidechain_base_loss = sidechain_base_loss / sidechain_base_loss.mean()
        if apply_mean:
            sidechain_loss, sidechain_base_loss = \
                sidechain_loss.mean().cpu() * torch.ones(1, dtype=torch.float), \
                sidechain_base_loss.mean().cpu() * torch.ones(1, dtype=torch.float)
        else:
            index = torch.cat([torch.ones((d['receptor'].pos.shape[0])) * i for i, d in enumerate(data)],
                              dim=0).long() if device.type == 'cuda' else data['receptor'].batch
            num_graphs = len(data) if device.type == 'cuda' else data.num_graphs
            s_l, s_b_l, c = torch.zeros(num_graphs), torch.zeros(num_graphs), torch.zeros(num_graphs)
            c.index_add_(0, index, torch.ones(sidechain_loss.shape[0]))
            c = c + 0.0001
            s_l.index_add_(0, index, sidechain_loss.cpu())
            s_b_l.index_add_(0, index, sidechain_base_loss.cpu())
            sidechain_loss, sidechain_base_loss = s_l / c, s_b_l / c
    else:
        if apply_mean:
            sidechain_loss, sidechain_base_loss = torch.zeros(1, dtype=torch.float), torch.zeros(1, dtype=torch.float)
        else:
            sidechain_loss, sidechain_base_loss = torch.zeros(len(rot_loss), dtype=torch.float), torch.zeros(
                len(rot_loss), dtype=torch.float)

    #合并损失
    loss = tr_loss * tr_weight + rot_loss * rot_weight + tor_loss * tor_weight + sidechain_loss * sidechain_weight + backbone_loss * backbone_weight
    return loss, tr_loss.detach(), rot_loss.detach(), tor_loss.detach(), backbone_loss.detach(), sidechain_loss.detach(), \
           tr_base_loss, rot_base_loss, tor_base_loss, backbone_base_loss, sidechain_base_loss
"""
多模块支持：函数可计算平移、旋转、扭转、主链和侧链的损失。
可扩展性：通过调整权重，可以灵活地在训练中强调特定的损失分量。
容错性：处理了角度的循环问题，并针对 NaN 数据进行了特殊处理。
"""

#AverageMeter 是一个工具类，用于累积和计算不同类型指标的平均值。这在训练过程中监控损失和其他度量值非常有用。
class AverageMeter():
    def __init__(self, types, unpooled_metrics=False, intervals=1):
        self.types = types
        self.intervals = intervals
        self.count = 0 if intervals == 1 else torch.zeros(len(types), intervals)
        self.acc = {t: torch.zeros(intervals) for t in types}
        self.unpooled_metrics = unpooled_metrics

    #添加新的指标值到累积器。
    def add(self, vals, interval_idx=None):
        if self.intervals == 1:
            self.count += 1 if vals[0].dim() == 0 else len(vals[0])
            for type_idx, v in enumerate(vals):
                self.acc[self.types[type_idx]] += v.sum().cpu() if self.unpooled_metrics else v.cpu()
        else:
            for type_idx, v in enumerate(vals):
                self.count[type_idx].index_add_(0, interval_idx[type_idx], torch.ones(len(v)))
                if not torch.allclose(v, torch.tensor(0.0)):
                    self.acc[self.types[type_idx]].index_add_(0, interval_idx[type_idx], v)

    #计算并返回所有指标的平均值。
    def summary(self):
        if self.intervals == 1:
            out = {k: v.item() / self.count for k, v in self.acc.items()}
            return out
        else:
            out = {}
            for i in range(self.intervals):
                for type_idx, k in enumerate(self.types):
                    out['int' + str(i) + '_' + k] = (
                            list(self.acc.values())[type_idx][i] / self.count[type_idx][i]).item()
            return out

#train_epoch 是模型训练一个完整周期（epoch）的逻辑实现。它处理数据加载、前向传播、损失计算、反向传播以及指标记录。
def train_epoch(model, loader, optimizer, device, t_to_sigma, loss_fn, ema_weights):
    model.train()#模型设置为训练模式:
    meter = AverageMeter(['loss', 'tr_loss', 'rot_loss', 'tor_loss', 'backbone_loss', 'sidechain_loss',
                          'tr_base_loss', 'rot_base_loss', 'tor_base_loss', 'backbone_base_loss', 'sidechain_base_loss'])#创建指标累积器:

    #数据加载循环:
    for data in tqdm(loader, total=len(loader)):
        #跳过单样本批次:
        if device.type == 'cuda' and len(data) == 1 or device.type == 'cpu' and data.num_graphs == 1:
            print("Skipping batch of size 1 since otherwise batchnorm would not work.")
            continue
        optimizer.zero_grad()
        data = [d.to(device) for d in data] if device.type == 'cuda' else data
        try:
            #前向传播:
            tr_pred, rot_pred, tor_pred, sidechain_pred = model(data)
            loss_tuple = loss_fn(tr_pred, rot_pred, tor_pred, sidechain_pred, data=data, t_to_sigma=t_to_sigma, device=device)
            #检查损失有效性
            if loss_tuple is None:
                print("None loss tuple, skipping")
                continue
            loss = loss_tuple[0]

            if torch.any(torch.isnan(loss)):
                names = data.name if device.type == 'cpu' else [d.name for d in data]
                print("Nan loss, skipping batch with complexes", names)
                continue
            #反向传播和优化:
            loss.backward()
            optimizer.step()
            #EMA 更新:
            if ema_weights is not None: ema_weights.update(model.parameters())
            meter.add([loss.cpu().detach(), *loss_tuple[1:]])
            
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                continue
            elif 'Input mismatch' in str(e):
                print('| WARNING: weird torch_cluster error, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                continue
            else:
                #raise e
                print(e)
                continue
            
    return meter.summary()

#实现模型在验证或测试数据集上的一个完整周期的推理过程。此函数计算模型在所有批次上的各种损失，并支持基于不同噪声级别的间隔计算指标
def test_epoch(model, loader, device, t_to_sigma, loss_fn, test_sigma_intervals=False):
    model.eval()#设置模型为评估模式:
    meter = AverageMeter(['loss', 'tr_loss', 'rot_loss', 'tor_loss', 'backbone_loss', 'sidechain_loss',
                          'tr_base_loss', 'rot_base_loss', 'tor_base_loss', 'backbone_base_loss', 'sidechain_base_loss'],
                         unpooled_metrics=True)#创建指标累积器:

    if test_sigma_intervals:#支持间隔指标的累积器:
        meter_all = AverageMeter(
            ['loss', 'tr_loss', 'rot_loss', 'tor_loss', 'backbone_loss', 'sidechain_loss',
             'tr_base_loss', 'rot_base_loss', 'tor_base_loss', 'backbone_base_loss', 'sidechain_base_loss'],
            unpooled_metrics=True, intervals=10)

    #数据加载循环:
    for data in tqdm(loader, total=len(loader)):
        try:
            #前向传播和损失计算:
            with torch.no_grad():
                tr_pred, rot_pred, tor_pred, sidechain_pred = model(data)
            loss_tuple = loss_fn(tr_pred, rot_pred, tor_pred, sidechain_pred, data=data, t_to_sigma=t_to_sigma, apply_mean=False, device=device)
            if loss_tuple is None: continue
            #更新累积器:
            meter.add([loss_tuple[0].cpu().detach(), *loss_tuple[1:]])

            #间隔指标计算
            if test_sigma_intervals > 0:
                complex_t_tr, complex_t_rot, complex_t_tor = [torch.cat([data[i].complex_t[noise_type] for i in range(len(data))]) for
                                                              noise_type in ['tr', 'rot', 'tor']]
                sigma_index_tr = torch.round(complex_t_tr.cpu() * (10 - 1)).long()
                sigma_index_rot = torch.round(complex_t_rot.cpu() * (10 - 1)).long()
                sigma_index_tor = torch.round(complex_t_tor.cpu() * (10 - 1)).long()
                meter_all.add([loss_tuple[0].cpu().detach(), *loss_tuple[1:]],
                    [sigma_index_tr, sigma_index_tr, sigma_index_rot, sigma_index_tor, sigma_index_tr, sigma_index_tr,
                     sigma_index_tr, sigma_index_rot, sigma_index_tor, sigma_index_tr, sigma_index_tr])

        #异常处理
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                continue
            elif 'Input mismatch' in str(e):
                print('| WARNING: weird torch_cluster error, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                continue
            else:
                raise e
                print(e)
                continue

    #总结指标
    out = meter.summary()
    if test_sigma_intervals > 0: out.update(meter_all.summary())
    return out

#实现分子对接任务中针对一批配体-受体复合物的完整推理过程，计算 RMSD（根均方偏差）指标以评估配体的预测精度。
def inference_epoch_fix(model, complex_graphs, device, t_to_sigma, args):
    #生成推理调度表
    t_schedule = get_t_schedule(sigma_schedule='expbeta', inference_steps=args.inference_steps,
                                inf_sched_alpha=1, inf_sched_beta=1)
    tr_schedule, rot_schedule, tor_schedule = t_schedule, t_schedule, t_schedule

    # 创建数据集和加载器
    dataset = ListDataset(complex_graphs)
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    #初始化指标
    rmsds, min_rmsds = [], []

    #数据处理和推理
    for orig_complex_graph in tqdm(loader):
        #深度复制数据并随机化初始位置
        data_list = [copy.deepcopy(orig_complex_graph) for _ in range(args.inference_samples)]
        randomize_position(data_list, args.no_torsion, False, args.tr_sigma_max)

        # 执行采样过程
        predictions_list = None
        failed_convergence_counter = 0
        while predictions_list == None:
            try:
                # 处理过滤条件
                predictions_list, confidences = sampling(data_list=data_list, model=model.module if device.type == 'cuda' else model,
                                                         inference_steps=args.inference_steps,
                                                         tr_schedule=tr_schedule, rot_schedule=rot_schedule,
                                                         tor_schedule=tor_schedule,
                                                         device=device, t_to_sigma=t_to_sigma, model_args=args,
                                                         t_schedule=t_schedule)
            except Exception as e:
                failed_convergence_counter += 1
                if failed_convergence_counter > 5:
                    print('failed 5 times - skipping the complex')
                    break
                print("Exception while running inference on complex:", e)
        if failed_convergence_counter > 5:
            rmsds.extend([100] * args.inference_samples)
            min_rmsds.append(100)
            continue

        if args.no_torsion:
            orig_complex_graph['ligand'].orig_pos = (orig_complex_graph[
                                                         'ligand'].pos.cpu().numpy() + orig_complex_graph.original_center.cpu().numpy())

        filterHs = torch.not_equal(predictions_list[0]['ligand'].x[:, 0], 0).cpu().numpy()

        if isinstance(orig_complex_graph['ligand'].orig_pos, list):
            orig_complex_graph['ligand'].orig_pos = orig_complex_graph['ligand'].orig_pos[0]
        # if len(orig_complex_graph['ligand'].orig_pos.shape) == 3:
        #     orig_complex_graph['ligand'].orig_pos = orig_complex_graph['ligand'].orig_pos[0]

        #RMSD 计算
        ligand_pos = np.asarray(
            [complex_graph['ligand'].pos.cpu().numpy()[filterHs] for complex_graph in predictions_list])
        if len(orig_complex_graph['ligand'].orig_pos.shape) == 2:
            orig_complex_graph['ligand'].orig_pos = orig_complex_graph['ligand'].orig_pos[None, :, :]
        try:
            orig_ligand_pos = orig_complex_graph['ligand'].orig_pos[:, filterHs] - orig_complex_graph.original_center.cpu().numpy()
        except Exception as e:
            print("problem with orig_pos which is of shape:", orig_complex_graph['ligand'].orig_pos.shape, e)
            continue
        mol = RemoveAllHs(orig_complex_graph.mol[0])
        complex_rmsds = []
        for i in range(len(orig_ligand_pos)):
            try:
                rmsd = get_symmetry_rmsd(mol, orig_ligand_pos[i], [l for l in ligand_pos])
            except Exception as e:
                print("Using non corrected RMSD because of the error:", e)
                rmsd = np.sqrt(((ligand_pos - orig_ligand_pos[i]) ** 2).sum(axis=2).mean(axis=1))
            complex_rmsds.append(rmsd)
        complex_rmsds = np.asarray(complex_rmsds)
        rmsd = np.min(complex_rmsds, axis=0)
        
        rmsds.extend([r for r in rmsd])
        min_rmsds.append(rmsd.min(axis=0))

    rmsds = np.array(rmsds)
    min_rmsds = np.array(min_rmsds)
    #统计指标
    losses = {'rmsds_lt2': (100 * (rmsds < 2).sum() / len(rmsds)),
              'rmsds_lt5': (100 * (rmsds < 5).sum() / len(rmsds)),
              'min_rmsds_lt2': (100 * (min_rmsds < 2).sum() / len(min_rmsds)),
              'min_rmsds_lt5': (100 * (min_rmsds < 5).sum() / len(min_rmsds)),}
    return losses

import gc
import math
import os

import shutil

from argparse import Namespace, ArgumentParser, FileType
import torch.nn.functional as F

import wandb
import torch
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import DataListLoader, DataLoader
from tqdm import tqdm

from confidence.dataset import ConfidenceDataset
from utils.training import AverageMeter

torch.multiprocessing.set_sharing_strategy('file_system')

import yaml
from utils.utils import save_yaml_file, get_optimizer_and_scheduler, get_model

# 创建ArgumentParser对象，用于处理命令行参数
parser = ArgumentParser()
# 添加命令行参数
# 配置文件路径，默认为None
parser.add_argument('--config', type=FileType(mode='r'), default=None)
# 原始模型目录路径，默认为'workdir'，包含已训练模型及超参数
parser.add_argument('--original_model_dir', type=str, default='workdir', help='Path to folder with trained model and hyperparameters')
# 重启训练的目录，默认为None
parser.add_argument('--restart_dir', type=str, default=None, help='')
# 是否使用原始模型的缓存数据集，默认为False
parser.add_argument('--use_original_model_cache', action='store_true', default=False, help='If this is true, the same dataset as in the original model will be used. Otherwise, the dataset parameters are used.')
# 数据集路径，默认为'./data/PDBBind_processed/'，包含原始结构数据
parser.add_argument('--data_dir', type=str, default='data/PDBBind_processed/', help='Folder containing original structures')
# 模型检查点，默认为'best_model.pt'
parser.add_argument('--ckpt', type=str, default='best_model.pt', help='Checkpoint to use inside the folder')
# 模型保存频率，默认为0，表示只有在满足早期停止准则时保存最好的模型
parser.add_argument('--model_save_frequency', type=int, default=0, help='Frequency with which to save the last model. If 0, then only the early stopping criterion best model is saved and overwritten.')
# 最佳模型保存频率，默认为0
parser.add_argument('--best_model_save_frequency', type=int, default=0, help='Frequency with which to save the best model. If 0, then only the early stopping criterion best model is saved and overwritten.')
# 运行名称，默认为'test_confidence'
parser.add_argument('--run_name', type=str, default='test_confidence', help='')
# 项目名称，默认为'herbdock_confidence'
parser.add_argument('--project', type=str, default='herbdock_confidence', help='')
# 训练集分割路径，默认为'data/splits/timesplit_no_lig_overlap_train'
parser.add_argument('--split_train', type=str, default='data/splits/timesplit_no_lig_overlap_train', help='Path of file defining the split')
# 验证集分割路径，默认为'data/splits/timesplit_no_lig_overlap_val'
parser.add_argument('--split_val', type=str, default='data/splits/timesplit_no_lig_overlap_val', help='Path of file defining the split')
# 测试集分割路径，默认为'data/splits/timesplit_test'
parser.add_argument('--split_test', type=str, default='data/splits/timesplit_test', help='Path of file defining the split')

# Inference parameters for creating the positions and rmsds that the confidence predictor will be trained on.
# 推理参数，用于创建信心预测器的位置信息和RMSD值
parser.add_argument('--cache_path', type=str, default='data/cacheNew', help='Folder from where to load/restore cached dataset')
# 是否将多个缓存合并，默认为None
parser.add_argument('--cache_ids_to_combine', nargs='+', type=str, default=None, help='RMSD value below which a prediction is considered a postitive. This can also be multiple cutoffs.')
# 缓存创建ID，用于标识推理次数
parser.add_argument('--cache_creation_id', type=int, default=None, help='number of times that inference is run on the full dataset before concatenating it and coming up with the full confidence dataset')
# 是否使用WandB进行实验管理，默认为False
parser.add_argument('--wandb', action='store_true', default=False, help='')
# 推理步骤的数量，默认为2
parser.add_argument('--inference_steps', type=int, default=2, help='Number of denoising steps')
# 每个复合体的样本数量，默认为3
parser.add_argument('--samples_per_complex', type=int, default=3, help='')
# 是否平衡正负样本数量，默认为False
parser.add_argument('--balance', action='store_true', default=False, help='If this is true than we do not force the samples seen during training to be the same amount of negatives as positives')
# 是否进行RMSD预测，默认为False
parser.add_argument('--rmsd_prediction', action='store_true', default=False, help='')
# RMSD分类阈值，默认为2
parser.add_argument('--rmsd_classification_cutoff', nargs='+', type=float, default=2, help='RMSD value below which a prediction is considered a postitive. This can also be multiple cutoffs.')

# 日志目录路径，默认为'workdir'
parser.add_argument('--log_dir', type=str, default='workdir', help='')
# 主评估指标，默认为'accuracy'
parser.add_argument('--main_metric', type=str, default='accuracy', help='Metric to track for early stopping. Mostly [loss, accuracy, ROC AUC]')
# 主评估目标，'max'或'min'，默认为'max'
parser.add_argument('--main_metric_goal', type=str, default='max', help='Can be [min, max]')
# 是否传递权重，默认为False
parser.add_argument('--transfer_weights', action='store_true', default=False, help='')
# 批大小，默认为5
parser.add_argument('--batch_size', type=int, default=5, help='')
# 学习率，默认为1e-3
parser.add_argument('--lr', type=float, default=1e-3, help='')
# 权重衰减，默认为0
parser.add_argument('--w_decay', type=float, default=0.0, help='')
# 学习率调度器，默认为'plateau'
parser.add_argument('--scheduler', type=str, default='plateau', help='')
# 学习率调度器的耐心值，默认为20
parser.add_argument('--scheduler_patience', type=int, default=20, help='')
# 训练周期数，默认为5
parser.add_argument('--n_epochs', type=int, default=5, help='')

# Dataset
# 限制复合体数量，默认为0
parser.add_argument('--limit_complexes', type=int, default=0, help='')
# 是否使用所有原子，默认为True
parser.add_argument('--all_atoms', action='store_true', default=True, help='')
# 配对多重性，默认为1
parser.add_argument('--multiplicity', type=int, default=1, help='')
# 链切割阈值，默认为10
parser.add_argument('--chain_cutoff', type=float, default=10, help='')
# 受体半径，默认为30
parser.add_argument('--receptor_radius', type=float, default=30, help='')
# 最大邻居数，默认为10
parser.add_argument('--c_alpha_max_neighbors', type=int, default=10, help='')
# 原子半径，默认为5
parser.add_argument('--atom_radius', type=float, default=5, help='')
# 最大原子邻居数，默认为8
parser.add_argument('--atom_max_neighbors', type=int, default=8, help='')
# 配对最大迭代次数，默认为20
parser.add_argument('--matching_popsize', type=int, default=20, help='')
# 配对最大迭代次数，默认为20
parser.add_argument('--matching_maxiter', type=int, default=20, help='')
# 最大配体大小，默认为None
parser.add_argument('--max_lig_size', type=int, default=None, help='Maximum number of heavy atoms')
# 是否去除氢原子，默认为False
parser.add_argument('--remove_hs', action='store_true', default=False, help='remove Hs')
# 配体构象数，默认为1
parser.add_argument('--num_conformers', type=int, default=1, help='')
# 如果设置了此路径，使用该路径下的LM嵌入用于受体特征
parser.add_argument('--esm_embeddings_path', type=str, default=None,help='If this is set then the LM embeddings at that path will be used for the receptor features')
# 是否禁用扭转角度，默认为False
parser.add_argument('--no_torsion', action='store_true', default=False, help='')

# Model
# 交互层数量，默认为2
parser.add_argument('--num_conv_layers', type=int, default=2, help='Number of interaction layers')
# 几何图的半径截断，默认为5.0
parser.add_argument('--max_radius', type=float, default=5.0, help='Radius cutoff for geometric graph')
# 是否按sigma值缩放，默认为True
parser.add_argument('--scale_by_sigma', action='store_true', default=True, help='Whether to normalise the score')
# 每个节点的隐藏特征数（阶0），默认为16
parser.add_argument('--ns', type=int, default=16, help='Number of hidden features per node of order 0')
# 每个节点的隐藏特征数（阶>0），默认为4
parser.add_argument('--nv', type=int, default=4, help='Number of hidden features per node of order >0')
#距离嵌入维度，默认为32
parser.add_argument('--distance_embed_dim', type=int, default=32, help='')
#跨距离嵌入维度，默认为32
parser.add_argument('--cross_distance_embed_dim', type=int, default=32, help='')
#是否移除批归一化，默认为False
parser.add_argument('--no_batch_norm', action='store_true', default=False, help='If set, it removes the batch norm')
#是否使用二阶表示，默认为False
parser.add_argument('--use_second_order_repr', action='store_true', default=False, help='Whether to use only up to first order representations or also second')
#最大交叉距离，默认为80
parser.add_argument('--cross_max_distance', type=float, default=80, help='')
#是否动态调整最大交叉距离，默认为False
parser.add_argument('--dynamic_max_cross', action='store_true', default=False, help='')
#MLP的dropout值，默认为0.0
parser.add_argument('--dropout', type=float, default=0.0, help='MLP dropout')
#嵌入类型，默认为"sinusoidal"
parser.add_argument('--embedding_type', type=str, default="sinusoidal", help='')
#sigma嵌入维度，默认为32
parser.add_argument('--sigma_embed_dim', type=int, default=32, help='')
#嵌入缩放因子，默认为10000
parser.add_argument('--embedding_scale', type=int, default=10000, help='')
#是否在信心读取器中移除批归一化，默认为False
parser.add_argument('--confidence_no_batchnorm', action='store_true', default=False, help='')
#信心读取器的MLP dropout值，默认为0.0
parser.add_argument('--confidence_dropout', type=float, default=0.0, help='MLP dropout in confidence readout')
#解析命令行参数
args = parser.parse_args()
if args.config:#如果提供了配置文件，将其加载并覆盖默认参数
    config_dict = yaml.load(args.config, Loader=yaml.FullLoader)
    arg_dict = args.__dict__
    for key, value in config_dict.items():
        if isinstance(value, list):
            for v in value:
                arg_dict[key].append(v)
        else:
            arg_dict[key] = value
    args.config = args.config.name
assert(args.main_metric_goal == 'max' or args.main_metric_goal == 'min')#确保主评估指标的目标为'max'或'min'

## 训练一个epoch
def train_epoch(model, loader, optimizer, rmsd_prediction):
    model.train()# 设置模型为训练模式
    meter = AverageMeter(['confidence_loss'])# 用于统计损失的平均值

    for data in tqdm(loader, total=len(loader)):# 遍历训练数据加载器
        # 如果批大小为1，跳过以避免batchnorm问题
        if device.type == 'cuda' and len(data) % torch.cuda.device_count() == 1 or device.type == 'cpu' and data.num_graphs == 1:
            print("Skipping batch of size 1 since otherwise batchnorm would not work.")
        optimizer.zero_grad() # 清空梯度
        try:
            pred = model(data)# 模型前向传播，得到预测值
            if rmsd_prediction:# 如果是RMSD预测任务
                labels = torch.cat([graph.rmsd for graph in data]).to(device) if isinstance(data, list) else data.rmsd
                confidence_loss = F.mse_loss(pred, labels)# 使用均方误差损失
            else:# 如果是分类任务
                if isinstance(args.rmsd_classification_cutoff, list): # 多分类
                    labels = torch.cat([graph.y_binned for graph in data]).to(device) if isinstance(data, list) else data.y_binned
                    confidence_loss = F.cross_entropy(pred, labels)# 使用交叉熵损失
                else:
                    labels = torch.cat([graph.y for graph in data]).to(device) if isinstance(data, list) else data.y
                    confidence_loss = F.binary_cross_entropy_with_logits(pred, labels)# 使用二分类损失
            confidence_loss.backward()# 反向传播计算梯度
            optimizer.step()# 更新模型参数
            meter.add([confidence_loss.cpu().detach()]) # 添加损失到计量器
        except RuntimeError as e: # 捕捉运行时错误
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                gc.collect()
                continue
            else:
                raise e

    return meter.summary()
# 测试一个epoch
def test_epoch(model, loader, rmsd_prediction):
    model.eval()# 设置模型为评估模式
    # 根据任务类型选择不同的计量器
    meter = AverageMeter(['loss'], unpooled_metrics=True) if rmsd_prediction else AverageMeter(['confidence_loss', 'accuracy', 'ROC AUC'], unpooled_metrics=True)
    all_labels = []# 保存所有标签
    for data in tqdm(loader, total=len(loader)):
        try:
            with torch.no_grad():# 禁用梯度计算
                pred = model(data)# 模型前向传播，得到预测值
            affinity_loss = torch.tensor(0.0, dtype=torch.float, device=pred[0].device)
            accuracy = torch.tensor(0.0, dtype=torch.float, device=pred[0].device)
            if rmsd_prediction:# 如果是RMSD预测任务
                labels = torch.cat([graph.rmsd for graph in data]).to(device) if isinstance(data, list) else data.rmsd
                confidence_loss = F.mse_loss(pred, labels)
                meter.add([confidence_loss.cpu().detach()])
            else:# 如果是分类任务
                if isinstance(args.rmsd_classification_cutoff, list):# 多分类
                    labels = torch.cat([graph.y_binned for graph in data]).to(device) if isinstance(data,list) else data.y_binned
                    confidence_loss = F.cross_entropy(pred, labels)
                else:# 二分类
                    labels = torch.cat([graph.y for graph in data]).to(device) if isinstance(data, list) else data.y
                    confidence_loss = F.binary_cross_entropy_with_logits(pred, labels)
                    accuracy = torch.mean((labels == (pred > 0).float()).float())
                try:
                    roc_auc = roc_auc_score(labels.detach().cpu().numpy(), pred.detach().cpu().numpy())
                except ValueError as e:# 如果无法计算ROC AUC
                    if 'Only one class present in y_true. ROC AUC score is not defined in that case.' in str(e):
                        roc_auc = 0
                    else:
                        raise e
            meter.add([confidence_loss.cpu().detach(), accuracy.cpu().detach(), torch.tensor(roc_auc)])
            all_labels.append(labels)

        except RuntimeError as e: # 捕捉运行时错误
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                continue
            else:
                raise e

    all_labels = torch.cat(all_labels)# 合并所有标签

     # 计算基线指标
    if rmsd_prediction:
        baseline_metric = ((all_labels - all_labels.mean()).abs()).mean()
    else:
        baseline_metric = all_labels.sum() / len(all_labels)
    results = meter.summary()
    results.update({'baseline_metric': baseline_metric})
    return meter.summary(), baseline_metric

# 训练函数
def train(args, model, optimizer, scheduler, train_loader, val_loader, run_dir):
    # 根据目标选择初始的最佳指标
    best_val_metric = math.inf if args.main_metric_goal == 'min' else 0
    best_epoch = 0

    print("Starting training...")
    for epoch in range(args.n_epochs):# 遍历所有训练epoch
        logs = {}
        train_metrics = train_epoch(model, train_loader, optimizer, args.rmsd_prediction)
        print("Epoch {}: Training loss {:.4f}".format(epoch, train_metrics['confidence_loss']))

        val_metrics, baseline_metric = test_epoch(model, val_loader, args.rmsd_prediction)
        if args.rmsd_prediction:
            print("Epoch {}: Validation loss {:.4f}".format(epoch, val_metrics['confidence_loss']))
        else:
            print("Epoch {}: Validation loss {:.4f}  accuracy {:.4f}".format(epoch, val_metrics['confidence_loss'], val_metrics['accuracy']))

        if args.wandb: # 如果启用了WandB日志记录
            logs.update({'valinf_' + k: v for k, v in val_metrics.items()}, step=epoch + 1)
            logs.update({'train_' + k: v for k, v in train_metrics.items()}, step=epoch + 1)
            logs.update({'mean_rmsd' if args.rmsd_prediction else 'fraction_positives': baseline_metric,
                         'current_lr': optimizer.param_groups[0]['lr']})
            wandb.log(logs, step=epoch + 1)

        if scheduler: # 调整学习率
            scheduler.step(val_metrics[args.main_metric])

        # 保存模型
        state_dict = model.module.state_dict() if device.type == 'cuda' else model.state_dict()

        # 更新最佳模型
        if args.main_metric_goal == 'min' and val_metrics[args.main_metric] < best_val_metric or \
                args.main_metric_goal == 'max' and val_metrics[args.main_metric] > best_val_metric:
            best_val_metric = val_metrics[args.main_metric]
            best_epoch = epoch
            torch.save(state_dict, os.path.join(run_dir, 'best_model.pt'))
        # 根据保存频率保存最新模型
        if args.model_save_frequency > 0 and (epoch + 1) % args.model_save_frequency == 0:
            torch.save(state_dict, os.path.join(run_dir, f'model_epoch{epoch+1}.pt'))
        if args.best_model_save_frequency > 0 and (epoch + 1) % args.best_model_save_frequency == 0:
            shutil.copyfile(os.path.join(run_dir, 'best_model.pt'), os.path.join(run_dir, f'best_model_epoch{epoch+1}.pt'))

        # 保存最后的模型
        torch.save({
            'epoch': epoch,
            'model': state_dict,
            'optimizer': optimizer.state_dict(),
        }, os.path.join(run_dir, 'last_model.pt'))

    print("Best Validation accuracy {} on Epoch {}".format(best_val_metric, best_epoch))

# 构建训练和验证数据加载器
def construct_loader_confidence(args, device):
     # 公共参数，用于传递给 ConfidenceDataset
    common_args = {'cache_path': args.cache_path, 'original_model_dir': args.original_model_dir, 'device': device,
                   'inference_steps': args.inference_steps, 'samples_per_complex': args.samples_per_complex,
                   'limit_complexes': args.limit_complexes, 'all_atoms': args.all_atoms, 'balance': args.balance,
                   'rmsd_classification_cutoff': args.rmsd_classification_cutoff, 'use_original_model_cache': args.use_original_model_cache,
                   'cache_creation_id': args.cache_creation_id, "cache_ids_to_combine": args.cache_ids_to_combine,
                   "model_ckpt": args.ckpt}
    # 根据是否使用GPU选择数据加载器类
    loader_class = DataListLoader if torch.cuda.is_available() else DataLoader

    exception_flag = False# 用于标记是否在加载过程中发生异常
    try:
        # 构造训练数据集和加载器
        train_dataset = ConfidenceDataset(split="train", args=args, **common_args)
        train_loader = loader_class(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    except Exception as e:
        # 如果加载训练数据时出现特定异常，则继续尝试加载验证数据
        if 'The generated ligand positions with cache_id do not exist:' in str(e):
            print("HAPPENING | Encountered the following exception when loading the confidence train dataset:")
            print(str(e))
            print("HAPPENING | We are still continuing because we want to try to generate the validation dataset if it has not been created yet:")
            exception_flag = True
        else: raise e

    # 构造验证数据集和加载器
    val_dataset = ConfidenceDataset(split="val", args=args, **common_args)
    val_loader = loader_class(dataset=val_dataset, batch_size=args.batch_size, shuffle=True)

      # 如果在加载训练数据时发生异常，则抛出异常
    if exception_flag: raise Exception('We encountered the exception during train dataset loading: ', e)
    return train_loader, val_loader


if __name__ == '__main__':
     # 确定设备（GPU或CPU）
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # 加载模型参数
    with open(f'{args.original_model_dir}/model_parameters.yml') as f:
        score_model_args = Namespace(**yaml.full_load(f))

    # construct loader
    # 构建训练和验证数据加载器
    train_loader, val_loader = construct_loader_confidence(args, device)
    # 初始化模型
    model = get_model(score_model_args if args.transfer_weights else args, device, t_to_sigma=None, confidence_mode=True)
     # 获取优化器和学习率调度器
    optimizer, scheduler = get_optimizer_and_scheduler(args, model, scheduler_mode=args.main_metric_goal)

    if args.transfer_weights:
        # 如果启用了权重迁移，则从原始模型加载权重
        print("HAPPENING | Transferring weights from original_model_dir to the new model after using original_model_dir's arguments to construct the new model.")
        checkpoint = torch.load(os.path.join(args.original_model_dir,args.ckpt), map_location=device)
        model_state_dict = model.state_dict()
        transfer_weights_dict = {k: v for k, v in checkpoint.items() if k in list(model_state_dict.keys())}
        model_state_dict.update(transfer_weights_dict)  # update the layers with the pretrained weights
        model.load_state_dict(model_state_dict)

    elif args.restart_dir:
        # 如果启用了从断点重启，则加载断点模型
        dict = torch.load(f'{args.restart_dir}/last_model.pt', map_location=torch.device('cpu'))
        model.module.load_state_dict(dict['model'], strict=True)
        optimizer.load_state_dict(dict['optimizer'])
        print("Restarting from epoch", dict['epoch'])

    # 计算模型参数数量
    numel = sum([p.numel() for p in model.parameters()])
    print('Model with', numel, 'parameters')

    if args.wandb:
        wandb.init(
            entity='entity',
            settings=wandb.Settings(start_method="fork"),
            project=args.project,
            name=args.run_name,
            config=args
        )
        wandb.log({'numel': numel})

    # record parameters
    # 记录模型参数并保存到指定路径
    run_dir = os.path.join(args.log_dir, args.run_name)
    yaml_file_name = os.path.join(run_dir, 'model_parameters.yml')
    save_yaml_file(yaml_file_name, args.__dict__)
    args.device = device

    train(args, model, optimizer, scheduler, train_loader, val_loader, run_dir)

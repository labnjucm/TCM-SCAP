import os
import subprocess
import warnings
from datetime import datetime
from typing import List

import numpy
import numpy as np
import torch
import yaml
from rdkit import Chem
from rdkit.Chem import RemoveHs, MolToPDBFile
from torch import nn, Tensor
from torch_geometric.nn.data_parallel import DataParallel
from torch_geometric.utils import degree, subgraph

from models.aa_model import AAModel
from models.cg_model import CGModel
from models.old_aa_model import AAOldModel
from models.old_cg_model import CGOldModel
from utils.diffusion_utils import get_timestep_embedding

#计算两个分子结构之间的 RMSD（根均方偏差），通过调用 Open Babel 的 obrms 工具实现。
def get_obrmsd(mol1_path, mol2_path, cache_name=None):
    cache_name = datetime.now().strftime('date%d-%m_time%H-%M-%S.%f') if cache_name is None else cache_name#如果未指定 cache_name，自动生成带有时间戳的文件名。
    os.makedirs(".openbabel_cache", exist_ok=True)
    if not isinstance(mol1_path, str):
        MolToPDBFile(mol1_path, '.openbabel_cache/obrmsd_mol1_cache.pdb')
        mol1_path = '.openbabel_cache/obrmsd_mol1_cache.pdb'
    if not isinstance(mol2_path, str):
        MolToPDBFile(mol2_path, '.openbabel_cache/obrmsd_mol2_cache.pdb')
        mol2_path = '.openbabel_cache/obrmsd_mol2_cache.pdb'
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return_code = subprocess.run(f"obrms {mol1_path} {mol2_path} > .openbabel_cache/obrmsd_{cache_name}.rmsd",
                                     shell=True)
        print(return_code)
    obrms_output = read_strings_from_txt(f".openbabel_cache/obrmsd_{cache_name}.rmsd")
    rmsds = [line.split(" ")[-1] for line in obrms_output]
    return np.array(rmsds, dtype=np.float)

#移除分子中的所有氢原子（H）
def remove_all_hs(mol):
    params = Chem.RemoveHsParameters()
    params.removeAndTrackIsotopes = True
    params.removeDefiningBondStereo = True
    params.removeDegreeZero = True
    params.removeDummyNeighbors = True
    params.removeHigherDegrees = True
    params.removeHydrides = True
    params.removeInSGroups = True
    params.removeIsotopes = True
    params.removeMapped = True
    params.removeNonimplicit = True
    params.removeOnlyHNeighbors = True
    params.removeWithQuery = True
    params.removeWithWedgedBond = True
    return RemoveHs(mol, params)

#从文本文件中逐行读取内容，返回列表
def read_strings_from_txt(path):
    # every line will be one element of the returned list
    with open(path) as file:
        lines = file.readlines()
        return [line.rstrip() for line in lines]

#根据批次信息将输入张量分解为多个子张量
def unbatch(src, batch: Tensor, dim: int = 0) -> List[Tensor]:
    r"""Splits :obj:`src` according to a :obj:`batch` vector along dimension
    :obj:`dim`.

    Args:
        src (Tensor): The source tensor.
        batch (LongTensor): The batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            entry in :obj:`src` to a specific example. Must be ordered.
        dim (int, optional): The dimension along which to split the :obj:`src`
            tensor. (default: :obj:`0`)

    :rtype: :class:`List[Tensor]`
    """
    sizes = degree(batch, dtype=torch.long).tolist()
    if isinstance(src, numpy.ndarray):
        return np.split(src, np.array(sizes).cumsum()[:-1], axis=dim)
    else:
        return src.split(sizes, dim)

#根据批次信息分解图的边索引
def unbatch_edge_index(edge_index: Tensor, batch: Tensor) -> List[Tensor]:
    r"""Splits the :obj:`edge_index` according to a :obj:`batch` vector.

    Args:
        edge_index (Tensor): The edge_index tensor. Must be ordered.
        batch (LongTensor): The batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. Must be ordered.

    :rtype: :class:`List[Tensor]`
    """
    deg = degree(batch, dtype=torch.int64)
    ptr = torch.cat([deg.new_zeros(1), deg.cumsum(dim=0)[:-1]], dim=0)

    edge_batch = batch[edge_index[0]]
    edge_index = edge_index - ptr[edge_batch]
    sizes = degree(edge_batch, dtype=torch.int64).cpu().tolist()
    return edge_index.split(sizes, dim=1)

#将批量边属性按批次分割为多个张量列表
def unbatch_edge_attributes(edge_attributes, edge_index: Tensor, batch: Tensor) -> List[Tensor]:
    edge_batch = batch[edge_index[0]]
    sizes = degree(edge_batch, dtype=torch.int64).cpu().tolist()
    return edge_attributes.split(sizes, dim=0)

#将内容保存为 YAML 文件
def save_yaml_file(path, content):
    assert isinstance(path, str), f'path must be a string, got {path} which is a {type(path)}'
    content = yaml.dump(data=content)
    if '/' in path and os.path.dirname(path) and not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, 'w') as f:
        f.write(content)

#解冻模型中某一层的参数，使其可训练
def unfreeze_layer(model):
    for name, child in (model.named_children()):
        #print(name, child.parameters())
        for param in child.parameters():
            param.requires_grad = True

#根据指定的调度器和优化器策略，创建优化器和学习率调度器
def get_optimizer_and_scheduler(args, model, scheduler_mode='min', step=0, optimizer=None):
    if args.scheduler == 'layer_linear_warmup':
        if step == 0:
            for name, child in (model.named_children()):
                if name.find('batch_norm') == -1:
                    for name, param in child.named_parameters():
                        if name.find('batch_norm') == -1:
                            param.requires_grad = False

            for l in [model.center_edge_embedding, model.final_conv, model.tr_final_layer, model.rot_final_layer,
                      model.final_edge_embedding, model.final_tp_tor, model.tor_bond_conv, model.tor_final_layer]:
                unfreeze_layer(l)

        elif 0 < step <= args.num_conv_layers:
            unfreeze_layer(model.conv_layers[-step])

        elif step == args.num_conv_layers + 1:
            for l in [model.lig_node_embedding, model.lig_edge_embedding, model.rec_node_embedding, model.rec_edge_embedding,
                      model.rec_sigma_embedding, model.cross_edge_embedding, model.rec_emb_layers, model.lig_emb_layers]:
                unfreeze_layer(l)

    if step == 0 or args.scheduler == 'layer_linear_warmup':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.w_decay)

    #创建优化器，仅优化 requires_grad=True 的参数
    scheduler_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=scheduler_mode, factor=0.7, patience=args.scheduler_patience, min_lr=args.lr / 100)
    #创建基于验证集性能的 ReduceLROnPlateau 调度器
    if args.scheduler == 'plateau':
        scheduler = scheduler_plateau
    elif args.scheduler == 'linear_warmup' or args.scheduler == 'layer_linear_warmup':
        if (args.scheduler == 'linear_warmup' and step < 1) or \
                (args.scheduler == 'layer_linear_warmup' and step <= args.num_conv_layers + 1):
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=args.lr_start_factor, end_factor=1.0,
                                                       total_iters=args.warmup_dur)
        else:
            scheduler = scheduler_plateau
    else:
        print('No scheduler')
        scheduler = None

    return optimizer, scheduler

#根据给定的参数 args 构建模型实例，同时支持不同类型的模型、嵌入方式和配置
def get_model(args, device, t_to_sigma, no_parallel=False, confidence_mode=False, old=False):

    #时间步嵌入的构建
    timestep_emb_func = get_timestep_embedding(
        embedding_type=args.embedding_type if 'embedding_type' in args else 'sinusoidal',
        embedding_dim=args.sigma_embed_dim,
        embedding_scale=args.embedding_scale if 'embedding_type' in args else 10000)

    #模型选择和构造
    if old:
        if 'all_atoms' in args and args.all_atoms:
            model_class = AAOldModel
        else:
            model_class = CGOldModel

        #语言模型嵌入类型：
        lm_embedding_type = None
        if args.esm_embeddings_path is not None: lm_embedding_type = 'esm'

        #构建旧版模型
        model = model_class(t_to_sigma=t_to_sigma,
                            device=device,
                            no_torsion=args.no_torsion,
                            timestep_emb_func=timestep_emb_func,
                            num_conv_layers=args.num_conv_layers,
                            lig_max_radius=args.max_radius,
                            scale_by_sigma=args.scale_by_sigma,
                            sigma_embed_dim=args.sigma_embed_dim,
                            norm_by_sigma='norm_by_sigma' in args and args.norm_by_sigma,
                            ns=args.ns, nv=args.nv,
                            distance_embed_dim=args.distance_embed_dim,
                            cross_distance_embed_dim=args.cross_distance_embed_dim,
                            batch_norm=not args.no_batch_norm,
                            dropout=args.dropout,
                            use_second_order_repr=args.use_second_order_repr,
                            cross_max_distance=args.cross_max_distance,
                            dynamic_max_cross=args.dynamic_max_cross,
                            smooth_edges=args.smooth_edges if "smooth_edges" in args else False,
                            odd_parity=args.odd_parity if "odd_parity" in args else False,
                            lm_embedding_type=lm_embedding_type,
                            confidence_mode=confidence_mode,
                            affinity_prediction=args.affinity_prediction if 'affinity_prediction' in args else False,
                            parallel=args.parallel if "parallel" in args else 1,
                            num_confidence_outputs=len(
                                args.rmsd_classification_cutoff) + 1 if 'rmsd_classification_cutoff' in args and isinstance(
                                args.rmsd_classification_cutoff, list) else 1,
                            parallel_aggregators=args.parallel_aggregators if "parallel_aggregators" in args else "",
                            fixed_center_conv=not args.not_fixed_center_conv if "not_fixed_center_conv" in args else False,
                            no_aminoacid_identities=args.no_aminoacid_identities if "no_aminoacid_identities" in args else False,
                            include_miscellaneous_atoms=args.include_miscellaneous_atoms if hasattr(args, 'include_miscellaneous_atoms') else False,
                            use_old_atom_encoder=args.use_old_atom_encoder if hasattr(args, 'use_old_atom_encoder') else True)

    #新版模型
    else:
        if 'all_atoms' in args and args.all_atoms:
            model_class = AAModel
        else:
            model_class = CGModel

        #语言模型嵌入类型
        lm_embedding_type = None
        if ('moad_esm_embeddings_path' in args and args.moad_esm_embeddings_path is not None) or \
            ('pdbbind_esm_embeddings_path' in args and args.pdbbind_esm_embeddings_path is not None) or \
            ('pdbsidechain_esm_embeddings_path' in args and args.pdbsidechain_esm_embeddings_path is not None) or \
            ('esm_embeddings_path' in args and args.esm_embeddings_path is not None):
            lm_embedding_type = 'precomputed'
        if 'esm_embeddings_model' in args and args.esm_embeddings_model is not None: lm_embedding_type = args.esm_embeddings_model

        #构建新版模型
        model = model_class(t_to_sigma=t_to_sigma,
                            device=device,
                            no_torsion=args.no_torsion,
                            timestep_emb_func=timestep_emb_func,
                            num_conv_layers=args.num_conv_layers,
                            lig_max_radius=args.max_radius,
                            scale_by_sigma=args.scale_by_sigma,
                            sigma_embed_dim=args.sigma_embed_dim,
                            norm_by_sigma='norm_by_sigma' in args and args.norm_by_sigma,
                            ns=args.ns, nv=args.nv,
                            distance_embed_dim=args.distance_embed_dim,
                            cross_distance_embed_dim=args.cross_distance_embed_dim,
                            batch_norm=not args.no_batch_norm,
                            dropout=args.dropout,
                            use_second_order_repr=args.use_second_order_repr,
                            cross_max_distance=args.cross_max_distance,
                            dynamic_max_cross=args.dynamic_max_cross,
                            smooth_edges=args.smooth_edges if "smooth_edges" in args else False,
                            odd_parity=args.odd_parity if "odd_parity" in args else False,
                            lm_embedding_type=lm_embedding_type,
                            confidence_mode=confidence_mode,
                            affinity_prediction=args.affinity_prediction if 'affinity_prediction' in args else False,
                            parallel=args.parallel if "parallel" in args else 1,
                            num_confidence_outputs=len(
                                args.rmsd_classification_cutoff) + 1 if 'rmsd_classification_cutoff' in args and isinstance(
                                args.rmsd_classification_cutoff, list) else 1,
                            atom_num_confidence_outputs=len(
                                args.atom_rmsd_classification_cutoff) + 1 if 'atom_rmsd_classification_cutoff' in args and isinstance(
                                args.atom_rmsd_classification_cutoff, list) else 1,
                            parallel_aggregators=args.parallel_aggregators if "parallel_aggregators" in args else "",
                            fixed_center_conv=not args.not_fixed_center_conv if "not_fixed_center_conv" in args else False,
                            no_aminoacid_identities=args.no_aminoacid_identities if "no_aminoacid_identities" in args else False,
                            include_miscellaneous_atoms=args.include_miscellaneous_atoms if hasattr(args, 'include_miscellaneous_atoms') else False,
                            sh_lmax=args.sh_lmax if 'sh_lmax' in args else 2,
                            differentiate_convolutions=not args.no_differentiate_convolutions if "no_differentiate_convolutions" in args else True,
                            tp_weights_layers=args.tp_weights_layers if "tp_weights_layers" in args else 2,
                            num_prot_emb_layers=args.num_prot_emb_layers if "num_prot_emb_layers" in args else 0,
                            reduce_pseudoscalars=args.reduce_pseudoscalars if "reduce_pseudoscalars" in args else False,
                            embed_also_ligand=args.embed_also_ligand if "embed_also_ligand" in args else False,
                            atom_confidence=args.atom_confidence_loss_weight > 0.0 if "atom_confidence_loss_weight" in args else False,
                            sidechain_pred=(hasattr(args, 'sidechain_loss_weight') and args.sidechain_loss_weight > 0) or
                                           (hasattr(args, 'backbone_loss_weight') and args.backbone_loss_weight > 0),
                            depthwise_convolution=args.depthwise_convolution if hasattr(args, 'depthwise_convolution') else False)

    #多 GPU 并行化
    if device.type == 'cuda' and not no_parallel and ('dataset' not in args or not args.dataset == 'torsional'):
        model = DataParallel(model)
    model.to(device)
    return model

import signal
from contextlib import contextmanager


class TimeoutException(Exception): pass


@contextmanager
#限制代码块的执行时间，如果超时则抛出 TimeoutException
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


#维护一组参数的指数移动平均值，用于平滑模型参数
class ExponentialMovingAverage:
    """ from https://github.com/yang-song/score_sde_pytorch/blob/main/models/ema.py
    Maintains (exponential) moving average of a set of parameters. """

    def __init__(self, parameters, decay, use_num_updates=True):
        """
        Args:
          parameters: Iterable of `torch.nn.Parameter`; usually the result of
            `model.parameters()`.
          decay: The exponential decay.
          use_num_updates: Whether to use number of updates when computing
            averages.
        """
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')
        self.decay = decay
        self.num_updates = 0 if use_num_updates else None
        self.shadow_params = [p.clone().detach()
                              for p in parameters if p.requires_grad]
        self.collected_params = []

    #更新 EMA 参数，每次参数更新后调用
    def update(self, parameters):
        """
        Update currently maintained parameters.
        Call this every time the parameters are updated, such as the result of
        the `optimizer.step()` call.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; usually the same set of
            parameters used to initialize this object.
        """
        decay = self.decay
        if self.num_updates is not None:
            self.num_updates += 1
            decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
        one_minus_decay = 1.0 - decay
        with torch.no_grad():
            parameters = [p for p in parameters if p.requires_grad]
            for s_param, param in zip(self.shadow_params, parameters):
                s_param.sub_(one_minus_decay * (s_param - param))

    #将当前 EMA 参数复制到目标参数
    def copy_to(self, parameters):
        """
        Copy current parameters into given collection of parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored moving averages.
        """
        parameters = [p for p in parameters if p.requires_grad]
        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                param.data.copy_(s_param.data)

    #暂存并恢复原始参数，用于测试和验证
    def store(self, parameters):
        """
        Save the current parameters for restoring later.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            temporarily stored.
        """
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored parameters.
        """
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)

    #用于保存和加载 EMA 的状态
    def state_dict(self):
        return dict(decay=self.decay, num_updates=self.num_updates,
                    shadow_params=self.shadow_params)

    def load_state_dict(self, state_dict, device):
        self.decay = state_dict['decay']
        self.num_updates = state_dict['num_updates']
        self.shadow_params = [tensor.to(device) for tensor in state_dict['shadow_params']]

#裁剪受体（receptor）和原子（atom）信息，仅保留与配体（ligand）距离在指定阈值内的部分
def crop_beyond(complex_graph, cutoff, all_atoms):
    ligand_pos = complex_graph['ligand'].pos
    receptor_pos = complex_graph['receptor'].pos
    residues_to_keep = torch.any(torch.sum((ligand_pos.unsqueeze(0) - receptor_pos.unsqueeze(1)) ** 2, -1) < cutoff ** 2, dim=1)

    if all_atoms:
        #print(complex_graph['atom'].x.shape, complex_graph['atom'].pos.shape, complex_graph['atom', 'atom_rec_contact', 'receptor'].edge_index.shape)
        atom_to_res_mapping = complex_graph['atom', 'atom_rec_contact', 'receptor'].edge_index[1]
        atoms_to_keep = residues_to_keep[atom_to_res_mapping]
        rec_remapper = (torch.cumsum(residues_to_keep.long(), dim=0) - 1)
        atom_to_res_new_mapping = rec_remapper[atom_to_res_mapping][atoms_to_keep]
        atom_res_edge_index = torch.stack([torch.arange(len(atom_to_res_new_mapping), device=atom_to_res_new_mapping.device), atom_to_res_new_mapping])

    complex_graph['receptor'].pos = complex_graph['receptor'].pos[residues_to_keep]
    complex_graph['receptor'].x = complex_graph['receptor'].x[residues_to_keep]
    complex_graph['receptor'].side_chain_vecs = complex_graph['receptor'].side_chain_vecs[residues_to_keep]
    complex_graph['receptor', 'rec_contact', 'receptor'].edge_index = \
        subgraph(residues_to_keep, complex_graph['receptor', 'rec_contact', 'receptor'].edge_index, relabel_nodes=True)[0]

    if all_atoms:
        complex_graph['atom'].x = complex_graph['atom'].x[atoms_to_keep]
        complex_graph['atom'].pos = complex_graph['atom'].pos[atoms_to_keep]
        complex_graph['atom', 'atom_contact', 'atom'].edge_index = subgraph(atoms_to_keep, complex_graph['atom', 'atom_contact', 'atom'].edge_index, relabel_nodes=True)[0]
        complex_graph['atom', 'atom_rec_contact', 'receptor'].edge_index = atom_res_edge_index

    #print("cropped", 1-torch.mean(residues_to_keep.float()), 'residues', 1-torch.mean(atoms_to_keep.float()), 'atoms')

import math

from e3nn import o3
import torch
from e3nn.o3 import Linear
from esm.pretrained import load_model_and_alphabet
from torch import nn
from torch.nn import functional as F
from torch_cluster import radius, radius_graph
from torch_scatter import scatter, scatter_mean
import numpy as np

from models.layers import GaussianSmearing, AtomEncoder
from models.tensor_layers import TensorProductConvLayer, get_irrep_seq
from utils import so3, torus
from datasets.process_mols import lig_feature_dims, rec_residue_feature_dims, rec_atom_feature_dims


class CGModel(torch.nn.Module):
    def __init__(self, t_to_sigma, device, timestep_emb_func, in_lig_edge_features=4, sigma_embed_dim=32, sh_lmax=2,
                 ns=16, nv=4, num_conv_layers=2, lig_max_radius=5, rec_max_radius=30, cross_max_distance=250,
                 center_max_distance=30, distance_embed_dim=32, cross_distance_embed_dim=32, no_torsion=False,
                 scale_by_sigma=True, norm_by_sigma=True, use_second_order_repr=False, batch_norm=True,
                 dynamic_max_cross=False, dropout=0.0, smooth_edges=False, odd_parity=False,
                 separate_noise_schedule=False, lm_embedding_type=None, confidence_mode=False,
                 confidence_dropout=0, confidence_no_batchnorm=False,
                 asyncronous_noise_schedule=False, affinity_prediction=False, parallel=1,
                 parallel_aggregators="mean max min std", num_confidence_outputs=1, atom_num_confidence_outputs=1, fixed_center_conv=False,
                 no_aminoacid_identities=False, include_miscellaneous_atoms=False,
                 differentiate_convolutions=True, tp_weights_layers=2, num_prot_emb_layers=0, reduce_pseudoscalars=False,
                 embed_also_ligand=False, atom_confidence=False, sidechain_pred=False, depthwise_convolution=False):
        """
        CGModel类的初始化函数，包含多个超参数用于模型的配置。
        
        参数：
        - t_to_sigma：时间步长到sigma的映射
        - device：模型运行设备（CPU/GPU）
        - timestep_emb_func：时间步长嵌入函数
        - in_lig_edge_features：配体边缘特征数
        - sigma_embed_dim：sigma嵌入的维度
        - sh_lmax：球面谐波最大阶数
        - ns：节点特征的维度
        - nv：节点的数量
        - num_conv_layers：卷积层的数量
        - lig_max_radius：配体最大半径
        - rec_max_radius：受体最大半径
        - cross_max_distance：跨分子最大距离
        - center_max_distance：中心最大距离
        - distance_embed_dim：距离嵌入维度
        - cross_distance_embed_dim：交叉距离嵌入维度
        - no_torsion：是否不使用扭转角度
        - scale_by_sigma：是否通过sigma进行缩放
        - norm_by_sigma：是否通过sigma进行归一化
        - use_second_order_repr：是否使用二阶表示
        - batch_norm：是否使用批量归一化
        - dropout：丢弃率
        - smooth_edges：是否平滑边缘
        - odd_parity：是否使用奇偶性
        - separate_noise_schedule：是否使用单独的噪声调度
        - lm_embedding_type：语言模型嵌入类型
        - confidence_mode：是否使用置信度模式
        - confidence_dropout：置信度丢弃率
        - confidence_no_batchnorm：是否禁用批量归一化
        - asyncronous_noise_schedule：是否使用异步噪声调度
        - affinity_prediction：是否进行亲和力预测
        - parallel：并行化参数
        - parallel_aggregators：并行聚合方法
        - num_confidence_outputs：置信度输出数量
        - atom_num_confidence_outputs：原子置信度输出数量
        - fixed_center_conv：是否固定中心卷积
        - no_aminoacid_identities：是否禁用氨基酸标识
        - include_miscellaneous_atoms：是否包含杂项原子
        - differentiate_convolutions：是否区分卷积操作
        - tp_weights_layers：TensorProduct卷积权重层数
        - num_prot_emb_layers：蛋白质嵌入层数
        - reduce_pseudoscalars：是否减少伪标量
        - embed_also_ligand：是否也嵌入配体
        - atom_confidence：是否进行原子置信度预测
        - sidechain_pred：是否进行侧链预测
        - depthwise_convolution：是否使用深度卷积
        """
        super(CGModel, self).__init__()
        # 参数检查
        assert parallel == 1, "not implemented"# 暂不支持并行
        assert (not no_aminoacid_identities) or (lm_embedding_type is None), "no language model emb without identities"
        # 保存输入的配置参数
        self.t_to_sigma = t_to_sigma
        self.in_lig_edge_features = in_lig_edge_features
        sigma_embed_dim *= (3 if separate_noise_schedule else 1)# sigma嵌入维度
        self.sigma_embed_dim = sigma_embed_dim
        self.lig_max_radius = lig_max_radius
        self.rec_max_radius = rec_max_radius
        self.include_miscellaneous_atoms = include_miscellaneous_atoms
        self.cross_max_distance = cross_max_distance
        self.dynamic_max_cross = dynamic_max_cross
        self.center_max_distance = center_max_distance
        self.distance_embed_dim = distance_embed_dim
        self.cross_distance_embed_dim = cross_distance_embed_dim
        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax=sh_lmax) # 球面谐波表示
        self.ns, self.nv = ns, nv# 节点的数量和特征维度
        self.scale_by_sigma = scale_by_sigma
        self.norm_by_sigma = norm_by_sigma
        self.device = device
        self.no_torsion = no_torsion
        self.smooth_edges = smooth_edges
        self.odd_parity = odd_parity
        self.timestep_emb_func = timestep_emb_func
        self.separate_noise_schedule = separate_noise_schedule
        self.confidence_mode = confidence_mode
        self.num_conv_layers = num_conv_layers
        self.num_prot_emb_layers = num_prot_emb_layers
        self.asyncronous_noise_schedule = asyncronous_noise_schedule
        self.affinity_prediction = affinity_prediction
        self.fixed_center_conv = fixed_center_conv
        self.no_aminoacid_identities = no_aminoacid_identities
        self.differentiate_convolutions = differentiate_convolutions
        self.reduce_pseudoscalars = reduce_pseudoscalars
        self.atom_confidence = atom_confidence
        self.atom_num_confidence_outputs = atom_num_confidence_outputs
        self.sidechain_pred = sidechain_pred

        self.lm_embedding_type = lm_embedding_type
        if lm_embedding_type is None:# 语言模型嵌入（如果适用）
            lm_embedding_dim = 0
        elif lm_embedding_type == "precomputed":
            lm_embedding_dim=1280
        else:
            lm, alphabet = load_model_and_alphabet(lm_embedding_type)# 加载语言模型
            self.batch_converter = alphabet.get_batch_converter()
            lm.lm_head = torch.nn.Identity()# 去除模型头
            lm.contact_head = torch.nn.Identity()
            lm_embedding_dim = lm.embed_dim
            self.lm = lm

         # 配体节点和边特征嵌入
        atom_encoder_class = AtomEncoder

        self.lig_node_embedding = atom_encoder_class(emb_dim=ns, feature_dims=lig_feature_dims, sigma_embed_dim=sigma_embed_dim)
        self.lig_edge_embedding = nn.Sequential(nn.Linear(in_lig_edge_features + sigma_embed_dim + distance_embed_dim, ns),nn.ReLU(),nn.Dropout(dropout),nn.Linear(ns, ns))

        # 受体节点和边特征嵌入
        self.rec_node_embedding = atom_encoder_class(emb_dim=ns, feature_dims=rec_residue_feature_dims, sigma_embed_dim=0, lm_embedding_dim=lm_embedding_dim)
        self.rec_edge_embedding = nn.Sequential(nn.Linear(distance_embed_dim, ns), nn.ReLU(), nn.Dropout(dropout), nn.Linear(ns, ns))
        self.rec_sigma_embedding = nn.Sequential(nn.Linear(sigma_embed_dim, ns), nn.ReLU(), nn.Dropout(dropout), nn.Linear(ns, ns))

        # 额外的原子类型处理（如果需要）
        if self.include_miscellaneous_atoms:
            self.misc_atom_node_embedding = atom_encoder_class(emb_dim=ns, feature_dims=rec_atom_feature_dims, sigma_embed_dim=sigma_embed_dim)
            self.misc_atom_edge_embedding = nn.Sequential(nn.Linear(sigma_embed_dim + distance_embed_dim, ns), nn.ReLU(),nn.Dropout(dropout), nn.Linear(ns, ns))
            self.ar_edge_embedding = nn.Sequential(nn.Linear(sigma_embed_dim + distance_embed_dim, ns), nn.ReLU(),nn.Dropout(dropout), nn.Linear(ns, ns))
            self.la_edge_embedding = nn.Sequential(nn.Linear(sigma_embed_dim + cross_distance_embed_dim, ns), nn.ReLU(),nn.Dropout(dropout), nn.Linear(ns, ns))

         # 跨分子边的嵌入
        self.cross_edge_embedding = nn.Sequential(nn.Linear(sigma_embed_dim + cross_distance_embed_dim, ns), nn.ReLU(), nn.Dropout(dropout),nn.Linear(ns, ns))

        # 高斯扩展，用于距离编码
        self.lig_distance_expansion = GaussianSmearing(0.0, lig_max_radius, distance_embed_dim)
        self.rec_distance_expansion = GaussianSmearing(0.0, rec_max_radius, distance_embed_dim)
        self.cross_distance_expansion = GaussianSmearing(0.0, cross_max_distance, cross_distance_embed_dim)

        # 获取表示的irreps序列
        irrep_seq = get_irrep_seq(ns, nv, use_second_order_repr, reduce_pseudoscalars)

        assert not self.include_miscellaneous_atoms, "currently not supported"

        # 配体和受体的卷积层（例如蛋白质结构嵌入层）
        rec_emb_layers = []
        for i in range(num_prot_emb_layers):
            in_irreps = irrep_seq[min(i, len(irrep_seq) - 1)]
            out_irreps = irrep_seq[min(i + 1, len(irrep_seq) - 1)]
            layer = TensorProductConvLayer(
                in_irreps=in_irreps,
                sh_irreps=self.sh_irreps,
                out_irreps=out_irreps,
                n_edge_features=3 * ns,
                hidden_features=3 * ns,
                residual=True,
                batch_norm=batch_norm,
                dropout=dropout,
                faster=sh_lmax == 1 and not use_second_order_repr,
                tp_weights_layers=tp_weights_layers,
                edge_groups=1,
                depthwise=depthwise_convolution
            )
            rec_emb_layers.append(layer)
        self.rec_emb_layers = nn.ModuleList(rec_emb_layers)

         # 配体嵌入层（如果需要）
        self.embed_also_ligand = embed_also_ligand
        if embed_also_ligand:
            lig_emb_layers = []
            for i in range(num_prot_emb_layers):
                in_irreps = irrep_seq[min(i, len(irrep_seq) - 1)]
                out_irreps = irrep_seq[min(i + 1, len(irrep_seq) - 1)]
                layer = TensorProductConvLayer(
                    in_irreps=in_irreps,
                    sh_irreps=self.sh_irreps,
                    out_irreps=out_irreps,
                    n_edge_features=3 * ns,
                    hidden_features=3 * ns,
                    residual=True,
                    batch_norm=batch_norm,
                    dropout=dropout,
                    faster=sh_lmax == 1 and not use_second_order_repr,
                    tp_weights_layers=tp_weights_layers,
                    edge_groups=1,
                    depthwise=depthwise_convolution
                )
                lig_emb_layers.append(layer)
            self.lig_emb_layers = nn.ModuleList(lig_emb_layers)

        # 额外的卷积层（根据需要添加）
        conv_layers = []
        for i in range(num_prot_emb_layers, num_prot_emb_layers + num_conv_layers):
            in_irreps = irrep_seq[min(i, len(irrep_seq) - 1)]
            out_irreps = irrep_seq[min(i + 1, len(irrep_seq) - 1)]
            layer = TensorProductConvLayer(
                in_irreps=in_irreps,
                sh_irreps=self.sh_irreps,
                out_irreps=out_irreps,
                n_edge_features=3 * ns,
                hidden_features=3 * ns,
                residual=True,
                batch_norm=batch_norm,
                dropout=dropout,
                faster=sh_lmax == 1 and not use_second_order_repr,
                tp_weights_layers=tp_weights_layers,
                edge_groups=1 if not differentiate_convolutions else (2 if i == num_prot_emb_layers + num_conv_layers - 1 else 4),
                depthwise=depthwise_convolution
            )
            conv_layers.append(layer)
        self.conv_layers = nn.ModuleList(conv_layers)

         # 侧链预测（如果需要）
        if sidechain_pred:
            self.sidechain_predictor = Linear(
                irreps_in=irrep_seq[min(num_prot_emb_layers + num_conv_layers, len(irrep_seq) - 1)],
                irreps_out='4x0e + 2x1e + 4x0o + 2x1o',
                internal_weights=True,
                shared_weights=True,
            )

         # 置信度模式（如果启用）
        if self.confidence_mode:
            input_size = ns + (nv if reduce_pseudoscalars else ns) if num_conv_layers + num_prot_emb_layers >= 3 else ns

            if self.atom_confidence:
                self.atom_confidence_predictor = nn.Sequential(
                    nn.Linear(input_size, ns),
                    nn.BatchNorm1d(ns) if not confidence_no_batchnorm else nn.Identity(),
                    nn.ReLU(),
                    nn.Dropout(confidence_dropout),
                    nn.Linear(ns, ns),
                    nn.BatchNorm1d(ns) if not confidence_no_batchnorm else nn.Identity(),
                    nn.ReLU(),
                    nn.Dropout(confidence_dropout),
                    nn.Linear(ns, atom_num_confidence_outputs + ns)
                )
                input_size = ns

            self.confidence_predictor = nn.Sequential(
                nn.Linear(input_size, ns),
                nn.BatchNorm1d(ns) if not confidence_no_batchnorm else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(confidence_dropout),
                nn.Linear(ns, ns),
                nn.BatchNorm1d(ns) if not confidence_no_batchnorm else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(confidence_dropout),
                nn.Linear(ns, num_confidence_outputs + (1 if self.affinity_prediction else 0))
            )

        else:
            # center of mass translation and rotation components# 旋转和翻译组件的处理（例如物体中心计算）
            self.center_distance_expansion = GaussianSmearing(0.0, center_max_distance, distance_embed_dim)
            self.center_edge_embedding = nn.Sequential(
                nn.Linear(distance_embed_dim + sigma_embed_dim, ns),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(ns, ns)
            )

            self.final_conv = TensorProductConvLayer(
                in_irreps=self.conv_layers[-1].out_irreps,
                sh_irreps=self.sh_irreps,
                out_irreps=f'2x1o + 2x1e' if not self.odd_parity else '1x1o + 1x1e',
                n_edge_features=2 * ns,
                residual=False,
                dropout=dropout,
                batch_norm=batch_norm
            )
             # 旋转和质心计算层
            self.tr_final_layer = nn.Sequential(nn.Linear(1 + sigma_embed_dim, ns),nn.Dropout(dropout), nn.ReLU(), nn.Linear(ns, 1))
            self.rot_final_layer = nn.Sequential(nn.Linear(1 + sigma_embed_dim, ns),nn.Dropout(dropout), nn.ReLU(), nn.Linear(ns, 1))

            if not no_torsion: # 扭转角度组件（如果启用）
                # torsion angles components
                self.final_edge_embedding = nn.Sequential(
                    nn.Linear(distance_embed_dim, ns),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(ns, ns)
                )
                self.final_tp_tor = o3.FullTensorProduct(self.sh_irreps, "2e")
                self.tor_bond_conv = TensorProductConvLayer(
                    in_irreps=self.conv_layers[-1].out_irreps,
                    sh_irreps=self.final_tp_tor.irreps_out,
                    out_irreps=f'{ns}x0o + {ns}x0e' if not self.odd_parity else f'{ns}x0o',
                    n_edge_features=3 * ns,
                    residual=False,
                    dropout=dropout,
                    batch_norm=batch_norm
                )
                self.tor_final_layer = nn.Sequential(
                    nn.Linear(2 * ns if not self.odd_parity else ns, ns, bias=False),
                    nn.Tanh(),
                    nn.Dropout(dropout),
                    nn.Linear(ns, 1, bias=False)
                )

    """
    此函数处理配体的嵌入。它通过调用build_lig_conv_graph()构建配体图，然后通过不同的嵌入层进行处理。

    lig_node_attr、lig_edge_index、lig_edge_attr等是从build_lig_conv_graph方法中获取的配体图相关属性。
    lig_node_attr是配体节点的属性，lig_edge_index是配体图的边索引，lig_edge_attr是配体图的边属性。
    嵌入通过多个层（lig_emb_layers）进行，每一层都会更新lig_node_attr。
    """
    def ligand_embedding(self, data):
        # ligand embedding
        lig_node_attr, lig_edge_index, lig_edge_attr, lig_edge_sh, lig_edge_weight = self.build_lig_conv_graph(data)
        lig_node_attr = self.lig_node_embedding(lig_node_attr)
        lig_edge_attr = self.lig_edge_embedding(lig_edge_attr)

        assert self.embed_also_ligand, "otherwise reimplement padding"
        for l in range(len(self.lig_emb_layers)):
            edge_attr_ = torch.cat([lig_edge_attr, lig_node_attr[lig_edge_index[0], :self.ns],
                                    lig_node_attr[lig_edge_index[1], :self.ns]], -1)
            lig_node_attr = self.lig_emb_layers[l](lig_node_attr, lig_edge_index, edge_attr_, lig_edge_sh,
                                                   edge_weight=lig_edge_weight)

        return lig_node_attr, lig_edge_index, lig_edge_attr, lig_edge_sh, lig_edge_weight

    """
    该方法完成配体和受体的嵌入操作。

    配体嵌入：首先调用ligand_embedding来计算配体嵌入。
    受体嵌入：如果data['receptor']没有预计算的rec_node_attr，则使用语言模型（如BERT）对受体序列进行嵌入，并将其添加到data['receptor'].x中。
    使用build_rec_conv_graph构建受体的卷积图，并对受体的节点属性进行嵌入。
    更新图数据中的receptor属性。
    """
    def embedding(self, data):
        if not hasattr(data['receptor'], "rec_node_attr"):
            if self.lm_embedding_type not in [None, 'precomputed']:
                sequences = [s for l in data['receptor'].sequence for s in l]
                if isinstance(sequences[0], list):
                    sequences = [s for l in sequences for s in l]
                sequences = [(i, s) for i, s in enumerate(sequences)]
                batch_labels, batch_strs, batch_tokens = self.batch_converter(sequences)
                out = self.lm(batch_tokens.to(data['receptor'].x.device), repr_layers=[self.lm.num_layers], return_contacts=False)
                rec_lm_emb = torch.cat([t[:len(sequences[i][1])] for i, t in enumerate(out['representations'][self.lm.num_layers])], dim=0)
                data['receptor'].x = torch.cat([data['receptor'].x, rec_lm_emb], dim=-1)

            rec_node_attr, rec_edge_attr, rec_edge_sh, rec_edge_weight = self.build_rec_conv_graph(data)
            rec_node_attr = self.rec_node_embedding(rec_node_attr)
            rec_edge_attr = self.rec_edge_embedding(rec_edge_attr)

            for l in range(len(self.rec_emb_layers)):
                edge_attr_ = torch.cat([rec_edge_attr, rec_node_attr[data['receptor', 'receptor'].edge_index[0], :self.ns], rec_node_attr[data['receptor', 'receptor'].edge_index[1], :self.ns]], -1)
                rec_node_attr = self.rec_emb_layers[l](rec_node_attr, data['receptor', 'receptor'].edge_index, edge_attr_, rec_edge_sh, edge_weight=rec_edge_weight)

            data['receptor'].rec_node_attr = rec_node_attr
            data['receptor', 'receptor'].rec_edge_attr = rec_edge_attr
            data['receptor', 'receptor'].edge_sh = rec_edge_sh
            data['receptor', 'receptor'].edge_weight = rec_edge_weight

        # receptor embedding
        rec_sigma_emb = self.rec_sigma_embedding(self.timestep_emb_func(data.complex_t['tr']))
        rec_node_attr = data['receptor'].rec_node_attr + 0
        rec_node_attr[:, :self.ns] = rec_node_attr[:, :self.ns] + rec_sigma_emb[data['receptor'].batch]
        rec_edge_attr = data['receptor', 'receptor'].rec_edge_attr + rec_sigma_emb[data['receptor'].batch[data['receptor', 'receptor'].edge_index[0]]]

        lig_node_attr, lig_edge_index, lig_edge_attr, lig_edge_sh, lig_edge_weight = self.ligand_embedding(data)

        return lig_node_attr, lig_edge_index, lig_edge_attr, lig_edge_sh, lig_edge_weight, \
               rec_node_attr, data['receptor', 'receptor'].edge_index, rec_edge_attr, data['receptor', 'receptor'].edge_sh, data['receptor', 'receptor'].edge_weight

    """
    这是模型的前向传播方法，用于计算最终的转动、平移、扭转等几何预测。

    平移、旋转、扭转噪声：根据confidence_mode模式来处理不同的噪声。
    构建交叉图：通过build_cross_conv_graph方法构建配体和受体之间的交叉图。
    卷积层处理：通过conv_layers进行卷积处理，处理配体、受体和交叉图的节点特征和边特征。
    计算平移和旋转向量：通过卷积层输出的global_pred来得到平移（tr_pred）和旋转（rot_pred）的预测值。
    信心得分：在confidence_mode下，计算分子对接的信心得分。
    预测侧链：如果sidechain_pred启用，预测侧链的方向。
    扭转预测：使用build_bond_conv_graph构建与扭转相关的卷积图，并预测扭转组件。
    """
    def forward(self, data):
        if self.no_aminoacid_identities:
            data['receptor'].x = data['receptor'].x * 0

        if not self.confidence_mode:
            tr_sigma, rot_sigma, tor_sigma = self.t_to_sigma(*[data.complex_t[noise_type] for noise_type in ['tr', 'rot', 'tor']])
        else:
            tr_sigma, rot_sigma, tor_sigma = [data.complex_t[noise_type] for noise_type in ['tr', 'rot', 'tor']]

        lig_node_attr, lig_edge_index, lig_edge_attr, lig_edge_sh, lig_edge_weight, rec_node_attr, \
            rec_edge_index, rec_edge_attr, rec_edge_sh, rec_edge_weight = self.embedding(data)

        # build cross graph
        if self.dynamic_max_cross:
            cross_cutoff = (tr_sigma * 3 + 20).unsqueeze(1)
        else:
            cross_cutoff = self.cross_max_distance

        lr_edge_index, lr_edge_attr, lr_edge_sh, rev_lr_edge_sh, lr_edge_weight = self.build_cross_conv_graph(data, cross_cutoff)
        lr_edge_attr = self.cross_edge_embedding(lr_edge_attr)

        node_attr = torch.cat([lig_node_attr, rec_node_attr], dim=0)
        lr_edge_index[1] = lr_edge_index[1] + len(lig_node_attr)
        edge_index = torch.cat([lig_edge_index, lr_edge_index, rec_edge_index + len(lig_node_attr),
                                torch.flip(lr_edge_index, dims=[0])], dim=1)
        edge_attr = torch.cat([lig_edge_attr, lr_edge_attr, rec_edge_attr, lr_edge_attr], dim=0)
        edge_sh = torch.cat([lig_edge_sh, lr_edge_sh, rec_edge_sh, rev_lr_edge_sh], dim=0)
        edge_weight = torch.cat([lig_edge_weight, lr_edge_weight, rec_edge_weight, lr_edge_weight],
                                dim=0) if torch.is_tensor(lig_edge_weight) else torch.ones((len(edge_index[0]), 1),
                                                                                           device=edge_index.device)
        s1, s2, s3 = len(lig_edge_index[0]), len(lig_edge_index[0]) + len(lr_edge_index[0]), len(lig_edge_index[0]) + len(lr_edge_index[0]) + len(rec_edge_index[0])

        for l in range(len(self.conv_layers)):
            if l < len(self.conv_layers) - 1:
                edge_attr_ = torch.cat(
                    [edge_attr, node_attr[edge_index[0], :self.ns], node_attr[edge_index[1], :self.ns]], -1)
                if self.differentiate_convolutions: edge_attr_ = [edge_attr_[:s1], edge_attr_[s1:s2], edge_attr_[s2:s3], edge_attr_[s3:]]
                node_attr = self.conv_layers[l](node_attr, edge_index, edge_attr_, edge_sh, edge_weight=edge_weight)
            else:
                edge_attr_ = torch.cat([edge_attr[:s2], node_attr[edge_index[0, :s2], :self.ns], node_attr[edge_index[1, :s2], :self.ns]], -1)
                if self.differentiate_convolutions: edge_attr_ = [edge_attr_[:s1], edge_attr_[s1:s2]]
                node_attr = self.conv_layers[l](node_attr, edge_index[:, :s2], edge_attr_, edge_sh[:s2], edge_weight=edge_weight[:s2])

        lig_node_attr = node_attr[:len(lig_node_attr)]

        # compute confidence score
        if self.confidence_mode:
            scalar_lig_attr = torch.cat([lig_node_attr[:,:self.ns], lig_node_attr[:,-(self.nv if self.reduce_pseudoscalars else self.ns):] ], dim=1) \
                if self.num_conv_layers + self.num_prot_emb_layers >= 3 else lig_node_attr[:,:self.ns]

            if self.atom_confidence:
                scalar_lig_attr = self.atom_confidence_predictor(scalar_lig_attr)
                atom_confidence = scalar_lig_attr[:, :self.atom_num_confidence_outputs]
                scalar_lig_attr = scalar_lig_attr[:, self.atom_num_confidence_outputs:]
            else:
                atom_confidence = torch.zeros((len(lig_node_attr),), device=lig_node_attr.device)

            confidence = self.confidence_predictor(scatter_mean(scalar_lig_attr, data['ligand'].batch, dim=0)).squeeze(dim=-1)
            return confidence, atom_confidence

        # compute translational and rotational score vectors
        center_edge_index, center_edge_attr, center_edge_sh = self.build_center_conv_graph(data)
        center_edge_attr = self.center_edge_embedding(center_edge_attr)
        if self.fixed_center_conv:
            center_edge_attr = torch.cat([center_edge_attr, lig_node_attr[center_edge_index[1], :self.ns]], -1)
        else:
            center_edge_attr = torch.cat([center_edge_attr, lig_node_attr[center_edge_index[0], :self.ns]], -1)
        global_pred = self.final_conv(lig_node_attr, center_edge_index, center_edge_attr, center_edge_sh, out_nodes=data.num_graphs)

        tr_pred = global_pred[:, :3] + (global_pred[:, 6:9] if not self.odd_parity else 0)
        rot_pred = global_pred[:, 3:6] + (global_pred[:, 9:] if not self.odd_parity else 0)

        if self.separate_noise_schedule:
            data.graph_sigma_emb = torch.cat([self.timestep_emb_func(data.complex_t[noise_type]) for noise_type in ['tr','rot','tor']], dim=1)
        elif self.asyncronous_noise_schedule:
            data.graph_sigma_emb = self.timestep_emb_func(data.complex_t['t'])
        else: # tr rot and tor noise is all the same in this case
            data.graph_sigma_emb = self.timestep_emb_func(data.complex_t['tr'])

        # fix the magnitude of translational and rotational score vectors
        tr_norm = torch.linalg.vector_norm(tr_pred, dim=1).unsqueeze(1)
        tr_pred = tr_pred / tr_norm * self.tr_final_layer(torch.cat([tr_norm, data.graph_sigma_emb], dim=1))
        rot_norm = torch.linalg.vector_norm(rot_pred, dim=1).unsqueeze(1)
        rot_pred = rot_pred / rot_norm * self.rot_final_layer(torch.cat([rot_norm, data.graph_sigma_emb], dim=1))

        if self.scale_by_sigma:
            tr_pred = tr_pred / tr_sigma.unsqueeze(1)
            rot_pred = rot_pred * so3.score_norm(rot_sigma.cpu()).unsqueeze(1).to(data['ligand'].x.device)

        # predict sidechain orientation
        sidechain_pred = None
        if self.sidechain_pred:
            rec_node_attr = node_attr[len(lig_node_attr):]
            sidechain_pred = self.sidechain_predictor(rec_node_attr)
            sidechain_pred = sidechain_pred[:, :10] + sidechain_pred[:, 10:] # sum even and odd components

        if self.no_torsion or data['ligand'].edge_mask.sum() == 0: return tr_pred, rot_pred, torch.empty(0, device=self.device), sidechain_pred

        # torsional components
        tor_bonds, tor_edge_index, tor_edge_attr, tor_edge_sh, tor_edge_weight = self.build_bond_conv_graph(data)
        tor_bond_vec = data['ligand'].pos[tor_bonds[1]] - data['ligand'].pos[tor_bonds[0]]
        tor_bond_attr = lig_node_attr[tor_bonds[0]] + lig_node_attr[tor_bonds[1]]

        tor_bonds_sh = o3.spherical_harmonics("2e", tor_bond_vec, normalize=True, normalization='component')
        tor_edge_sh = self.final_tp_tor(tor_edge_sh, tor_bonds_sh[tor_edge_index[0]])

        tor_edge_attr = torch.cat([tor_edge_attr, lig_node_attr[tor_edge_index[1], :self.ns],
                                   tor_bond_attr[tor_edge_index[0], :self.ns]], -1)
        tor_pred = self.tor_bond_conv(lig_node_attr, tor_edge_index, tor_edge_attr, tor_edge_sh,
                                  out_nodes=data['ligand'].edge_mask.sum(), reduce='mean', edge_weight=tor_edge_weight)
        tor_pred = self.tor_final_layer(tor_pred).squeeze(1)
        edge_sigma = tor_sigma[data['ligand'].batch][data['ligand', 'ligand'].edge_index[0]][data['ligand'].edge_mask]

        if self.scale_by_sigma:
            tor_pred = tor_pred * torch.sqrt(torch.tensor(torus.score_norm(edge_sigma.cpu().numpy())).float()
                                             .to(data['ligand'].x.device))
        return tr_pred, rot_pred, tor_pred, sidechain_pred

    """
    此方法计算与配体扭转（torsion）相关的组件：

    扭转图构建：与forward方法中的扭转部分类似，torsional_forward使用build_bond_conv_graph构建与扭转相关的图。
    计算扭转预测：通过tor_bond_conv计算扭转的预测值，并根据噪声对其进行缩放。

    """
    def torsional_forward(self, data):
        tor_sigma = self.t_to_sigma(data.complex_t['tor'])

        # build ligand graph
        lig_node_attr, lig_edge_index, lig_edge_attr, lig_edge_sh, lig_edge_weight = self.ligand_embedding(data)

        if self.separate_noise_schedule:
            data.graph_sigma_emb = torch.cat([self.timestep_emb_func(data.complex_t[noise_type]) for noise_type in ['tr','rot','tor']], dim=1)
        elif self.asyncronous_noise_schedule:
            data.graph_sigma_emb = self.timestep_emb_func(data.complex_t['t'])
        else: # tr rot and tor noise is all the same in this case
            data.graph_sigma_emb = self.timestep_emb_func(data.complex_t['tr'])

        # torsional components
        tor_bonds, tor_edge_index, tor_edge_attr, tor_edge_sh, tor_edge_weight = self.build_bond_conv_graph(data)
        tor_bond_vec = data['ligand'].pos[tor_bonds[1]] - data['ligand'].pos[tor_bonds[0]]
        tor_bond_attr = lig_node_attr[tor_bonds[0]] + lig_node_attr[tor_bonds[1]]

        tor_bonds_sh = o3.spherical_harmonics("2e", tor_bond_vec, normalize=True, normalization='component')
        tor_edge_sh = self.final_tp_tor(tor_edge_sh, tor_bonds_sh[tor_edge_index[0]])

        tor_edge_attr = torch.cat([tor_edge_attr, lig_node_attr[tor_edge_index[1], :self.ns],
                                   tor_bond_attr[tor_edge_index[0], :self.ns]], -1)
        tor_pred = self.tor_bond_conv(lig_node_attr, tor_edge_index, tor_edge_attr, tor_edge_sh,
                                  out_nodes=data['ligand'].edge_mask.sum(), reduce='mean', edge_weight=tor_edge_weight)
        tor_pred = self.tor_final_layer(tor_pred).squeeze(1)
        edge_sigma = tor_sigma[data['ligand'].batch][data['ligand', 'ligand'].edge_index[0]][data['ligand'].edge_mask]

        if self.scale_by_sigma:
            tor_pred = tor_pred * torch.sqrt(torch.tensor(torus.score_norm(edge_sigma.cpu().numpy())).float()
                                             .to(data['ligand'].x.device))
        return 0, 0, tor_pred, 0

    def get_edge_weight(self, edge_vec, max_norm):
        # computes weights for edges that are decreasing with the distance
        # it has an effect only if smooth edges is true
        # 计算边的权重，权重随着距离递减
        # 只有在平滑边缘 (smooth_edges) 为True时才有影响
        if self.smooth_edges:
            normalised_norm = torch.clip(edge_vec.norm(dim=-1) * np.pi / max_norm, max=np.pi)# 将边向量的范数归一化并限制最大值为pi
            return 0.5 * (torch.cos(normalised_norm) + 1.0).unsqueeze(-1)# 计算基于余弦的权重，范围从0到1
        return 1.0

    def build_lig_conv_graph(self, data): # 构建配体的图结构，包括节点和边的初始特征
        # builds the ligand graph edges and initial node and edge features
        if self.separate_noise_schedule:# 如果存在分开的噪声调度（tr, rot, tor），则为每种噪声类型生成时间步嵌入
            data['ligand'].node_sigma_emb = torch.cat([self.timestep_emb_func(data['ligand'].node_t[noise_type]) for noise_type in ['tr','rot','tor']], dim=1)
        elif self.asyncronous_noise_schedule:# 如果是异步噪声调度，使用时间步嵌入
            data['ligand'].node_sigma_emb = self.timestep_emb_func(data['ligand'].node_t['t'])
        else: # 否则，假设tr, rot和tor噪声是相同的
            data['ligand'].node_sigma_emb = self.timestep_emb_func(data['ligand'].node_t['tr']) # tr rot and tor noise is all the same

        # compute edges # 计算配体节点之间的距离边
        radius_edges = radius_graph(data['ligand'].pos, self.lig_max_radius, data['ligand'].batch)
        edge_index = torch.cat([data['ligand', 'ligand'].edge_index, radius_edges], 1).long()
        # 初始化边特征，添加零特征
        edge_attr = torch.cat([
            data['ligand', 'ligand'].edge_attr,
            torch.zeros(radius_edges.shape[-1], self.in_lig_edge_features, device=data['ligand'].x.device)
        ], 0)

        # compute initial features
        # 计算初始特征
        edge_sigma_emb = data['ligand'].node_sigma_emb[edge_index[0].long()]
        edge_attr = torch.cat([edge_attr, edge_sigma_emb], 1)
        node_attr = torch.cat([data['ligand'].x, data['ligand'].node_sigma_emb], 1)

        # 计算边向量和长度嵌入
        src, dst = edge_index
        edge_vec = data['ligand'].pos[dst.long()] - data['ligand'].pos[src.long()]
        edge_length_emb = self.lig_distance_expansion(edge_vec.norm(dim=-1))

         # 将长度嵌入添加到边特征中
        edge_attr = torch.cat([edge_attr, edge_length_emb], 1)
         # 计算球面谐波
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')
         # 计算边权重
        edge_weight = self.get_edge_weight(edge_vec, self.lig_max_radius)

        return node_attr, edge_index, edge_attr, edge_sh, edge_weight

    def build_rec_conv_graph(self, data):# 构建受体的初始图节点和边的嵌入
        # builds the receptor initial node and edge embeddings
        assert not self.separate_noise_schedule or self.asyncronous_noise_schedule, "removed support in this function"
        node_attr = data['receptor'].x

        # this assumes the edges were already created in preprocessing since protein's structure is fixed
          # 假设受体的边已经在预处理过程中创建
        edge_index = data['receptor', 'receptor'].edge_index
        src, dst = edge_index
        edge_vec = data['receptor'].pos[dst.long()] - data['receptor'].pos[src.long()]

        # 计算边长度嵌入
        edge_length_emb = self.rec_distance_expansion(edge_vec.norm(dim=-1))
        edge_attr = edge_length_emb
        # 计算球面谐波
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')
        # 计算边权重
        edge_weight = self.get_edge_weight(edge_vec, self.rec_max_radius)

        return node_attr, edge_attr, edge_sh, edge_weight

    def build_misc_atom_conv_graph(self, data):# 构建受体的杂质原子之间的图
        # build the graph between receptor misc_atoms
        if self.separate_noise_schedule:
            data['misc_atom'].node_sigma_emb = torch.cat([self.timestep_emb_func(data['misc_atom'].node_t[noise_type]) for noise_type in ['tr', 'rot', 'tor']],dim=1)
        elif self.asyncronous_noise_schedule:
            data['misc_atom'].node_sigma_emb = self.timestep_emb_func(data['misc_atom'].node_t['t'])
        else:
            data['misc_atom'].node_sigma_emb = self.timestep_emb_func(data['misc_atom'].node_t['tr'])  # tr rot and tor noise is all the same
        node_attr = torch.cat([data['misc_atom'].x, data['misc_atom'].node_sigma_emb], 1)

        # this assumes the edges were already created in preprocessing since protein's structure is fixed
        # 假设受体的杂质原子边已经在预处理过程中创建
        edge_index = data['misc_atom', 'misc_atom'].edge_index
        src, dst = edge_index
        edge_vec = data['misc_atom'].pos[dst.long()] - data['misc_atom'].pos[src.long()]

         # 计算边长度嵌入
        edge_length_emb = self.lig_distance_expansion(edge_vec.norm(dim=-1))
        edge_sigma_emb = data['misc_atom'].node_sigma_emb[edge_index[0].long()]
        edge_attr = torch.cat([edge_sigma_emb, edge_length_emb], 1)
         # 计算球面谐波
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')
         # 计算边权重
        edge_weight = self.get_edge_weight(edge_vec, self.lig_max_radius)

        return node_attr, edge_index, edge_attr, edge_sh, edge_weight

    def build_cross_conv_graph(self, data, cross_distance_cutoff):
        # builds the cross edges between ligand and receptor
         # 构建配体和受体之间的交叉边
        if torch.is_tensor(cross_distance_cutoff):
            # different cutoff for every graph (depends on the diffusion time)
            # 对每个图使用不同的cutoff（依赖于扩散时间）
            edge_index = radius(data['receptor'].pos / cross_distance_cutoff[data['receptor'].batch],
                                data['ligand'].pos / cross_distance_cutoff[data['ligand'].batch], 1,
                                data['receptor'].batch, data['ligand'].batch, max_num_neighbors=10000)
        else: # 使用统一的cutoff值
            edge_index = radius(data['receptor'].pos, data['ligand'].pos, cross_distance_cutoff,
                            data['receptor'].batch, data['ligand'].batch, max_num_neighbors=10000)

        src, dst = edge_index
        edge_vec = data['receptor'].pos[dst.long()] - data['ligand'].pos[src.long()]

         # 计算交叉边的长度嵌入
        edge_length_emb = self.cross_distance_expansion(edge_vec.norm(dim=-1))
        edge_sigma_emb = data['ligand'].node_sigma_emb[src.long()]
        edge_attr = torch.cat([edge_sigma_emb, edge_length_emb], 1)
        # 计算球面谐波
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')
        rev_edge_sh = o3.spherical_harmonics(self.sh_irreps, -edge_vec, normalize=True, normalization='component')

         # 计算边权重
        cutoff_d = cross_distance_cutoff[data['ligand'].batch[src]].squeeze() if torch.is_tensor(cross_distance_cutoff) else cross_distance_cutoff
        edge_weight = self.get_edge_weight(edge_vec, cutoff_d)

        return edge_index, edge_attr, edge_sh, rev_edge_sh, edge_weight

    def build_misc_cross_conv_graph(self, data, lr_cross_distance_cutoff):
        # build the cross edges between ligan atoms, receptor residues and receptor atoms

        # LIGAND to RECEPTOR
        # 构建配体原子、受体残基和受体原子之间的交叉边
        # LIGAND 到 RECEPTOR（配体到受体）
        if torch.is_tensor(lr_cross_distance_cutoff):
            # different cutoff for every graph
            lr_edge_index = radius(data['receptor'].pos / lr_cross_distance_cutoff[data['receptor'].batch],
                                data['ligand'].pos / lr_cross_distance_cutoff[data['ligand'].batch], 1,
                                data['receptor'].batch, data['ligand'].batch, max_num_neighbors=10000)
        else:# 如果是常量的截止值
            lr_edge_index = radius(data['receptor'].pos, data['ligand'].pos, lr_cross_distance_cutoff,
                            data['receptor'].batch, data['ligand'].batch, max_num_neighbors=10000)

         # 计算配体和受体之间的边向量
        lr_edge_vec = data['receptor'].pos[lr_edge_index[1].long()] - data['ligand'].pos[lr_edge_index[0].long()]
         # 计算交叉边的距离嵌入
        lr_edge_length_emb = self.cross_distance_expansion(lr_edge_vec.norm(dim=-1))
         # 获取配体节点的噪声嵌入（sigma embedding）
        lr_edge_sigma_emb = data['ligand'].node_sigma_emb[lr_edge_index[0].long()]
         # 拼接边的特征（sigma 嵌入 + 距离嵌入）
        lr_edge_attr = torch.cat([lr_edge_sigma_emb, lr_edge_length_emb], 1)
         # 计算球面谐波
        lr_edge_sh = o3.spherical_harmonics(self.sh_irreps, lr_edge_vec, normalize=True, normalization='component')

         # 计算边的权重
        cutoff_d = lr_cross_distance_cutoff[data['ligand'].batch[lr_edge_index[0]]].squeeze() \
            if torch.is_tensor(lr_cross_distance_cutoff) else lr_cross_distance_cutoff
        lr_edge_weight = self.get_edge_weight(lr_edge_vec, cutoff_d)

        # LIGAND to ATOM
        # LIGAND 到 ATOM（配体到原子）
        la_edge_index = radius(data['misc_atom'].pos, data['ligand'].pos, self.lig_max_radius,
                               data['misc_atom'].batch, data['ligand'].batch, max_num_neighbors=10000)

         # 计算配体和原子之间的边向量
        la_edge_vec = data['misc_atom'].pos[la_edge_index[1].long()] - data['ligand'].pos[la_edge_index[0].long()]
        # 计算交叉边的距离嵌入
        la_edge_length_emb = self.cross_distance_expansion(la_edge_vec.norm(dim=-1))
        # 获取配体节点的噪声嵌入（sigma embedding）
        la_edge_sigma_emb = data['ligand'].node_sigma_emb[la_edge_index[0].long()]
        # 拼接边的特征（sigma 嵌入 + 距离嵌入）
        la_edge_attr = torch.cat([la_edge_sigma_emb, la_edge_length_emb], 1)
        # 计算球面谐波
        la_edge_sh = o3.spherical_harmonics(self.sh_irreps, la_edge_vec, normalize=True, normalization='component')
        # 计算边的权重
        la_edge_weight = self.get_edge_weight(la_edge_vec, self.lig_max_radius)

        # ATOM to RECEPTOR
         # ATOM 到 RECEPTOR（原子到受体）
        ar_edge_index = data['misc_atom', 'receptor'].edge_index
        ar_edge_vec = data['receptor'].pos[ar_edge_index[1].long()] - data['misc_atom'].pos[ar_edge_index[0].long()]
         # 计算原子到受体之间的距离嵌入
        ar_edge_length_emb = self.rec_distance_expansion(ar_edge_vec.norm(dim=-1))
         # 获取原子节点的噪声嵌入（sigma embedding）
        ar_edge_sigma_emb = data['misc_atom'].node_sigma_emb[ar_edge_index[0].long()]
        # 拼接边的特征（sigma 嵌入 + 距离嵌入）
        ar_edge_attr = torch.cat([ar_edge_sigma_emb, ar_edge_length_emb], 1)
        # 计算球面谐波
        ar_edge_sh = o3.spherical_harmonics(self.sh_irreps, ar_edge_vec, normalize=True, normalization='component')
        # 原子到受体的边权重设置为 1（不变的边权重）
        ar_edge_weight = 1

        return lr_edge_index, lr_edge_attr, lr_edge_sh, lr_edge_weight, la_edge_index, la_edge_attr, \
               la_edge_sh, la_edge_weight, ar_edge_index, ar_edge_attr, ar_edge_sh, ar_edge_weight

    def build_center_conv_graph(self, data):
        # builds the filter and edges for the convolution generating translational and rotational scores
         # 构建卷积图，生成平移和旋转得分
         # 创建一个边索引：表示配体batch中每个节点与其对应节点的关系
        edge_index = torch.cat([data['ligand'].batch.unsqueeze(0), torch.arange(len(data['ligand'].batch)).to(data['ligand'].x.device).unsqueeze(0)], dim=0)

         # 初始化用于存储中心位置的张量
        center_pos, count = torch.zeros((data.num_graphs, 3)).to(data['ligand'].x.device), torch.zeros((data.num_graphs, 3)).to(data['ligand'].x.device)
        # 累加每个batch中的配体位置
        center_pos.index_add_(0, index=data['ligand'].batch, source=data['ligand'].pos)
        # 计算每个batch的配体位置的平均值（中心位置）
        center_pos = center_pos / torch.bincount(data['ligand'].batch).unsqueeze(1)

         # 计算边向量：配体每个节点和中心位置之间的向量
        edge_vec = data['ligand'].pos[edge_index[1]] - center_pos[edge_index[0]]
         # 计算中心距离的嵌入
        edge_attr = self.center_distance_expansion(edge_vec.norm(dim=-1))
        # 获取配体节点的噪声嵌入（sigma embedding）
        edge_sigma_emb = data['ligand'].node_sigma_emb[edge_index[1].long()]
         # 拼接边的特征（距离嵌入 + sigma 嵌入）
        edge_attr = torch.cat([edge_attr, edge_sigma_emb], 1)
        # 计算球面谐波
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')
        return edge_index, edge_attr, edge_sh

    def build_bond_conv_graph(self, data):
        # builds the graph for the convolution between the center of the rotatable bonds and the neighbouring nodes
        # 构建卷积图，表示旋转键的中心和邻近节点之间的关系
        # 获取配体中旋转键（bond）的边索引
        bonds = data['ligand', 'ligand'].edge_index[:, data['ligand'].edge_mask].long()
        # 计算每个旋转键的中心位置（两个端点的平均位置）
        bond_pos = (data['ligand'].pos[bonds[0]] + data['ligand'].pos[bonds[1]]) / 2
        # 获取旋转键所属的batch
        bond_batch = data['ligand'].batch[bonds[0]]
         # 计算旋转键中心到配体其他节点的距离，得到边索引
        edge_index = radius(data['ligand'].pos, bond_pos, self.lig_max_radius, batch_x=data['ligand'].batch, batch_y=bond_batch)

         # 计算旋转键中心到其他节点的边向量
        edge_vec = data['ligand'].pos[edge_index[1]] - bond_pos[edge_index[0]]
         # 计算距离嵌入
        edge_attr = self.lig_distance_expansion(edge_vec.norm(dim=-1))

         # 对距离嵌入进行最终嵌入
        edge_attr = self.final_edge_embedding(edge_attr)
         # 计算球面谐波
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')
        # 计算边的权重
        edge_weight = self.get_edge_weight(edge_vec, self.lig_max_radius)

        return bonds, edge_index, edge_attr, edge_sh, edge_weight


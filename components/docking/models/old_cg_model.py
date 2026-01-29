import math

from e3nn import o3
import torch
from torch import nn
from torch.nn import functional as F
from torch_cluster import radius, radius_graph
from torch_scatter import scatter, scatter_mean
import numpy as np
from e3nn.nn import BatchNorm

from models.layers import OldAtomEncoder, AtomEncoder, GaussianSmearing
from models.tensor_layers import OldTensorProductConvLayer
from utils import so3, torus
from datasets.process_mols import lig_feature_dims, rec_residue_feature_dims, rec_atom_feature_dims


class CGOldModel(torch.nn.Module):
    def __init__(self, t_to_sigma, device, timestep_emb_func, in_lig_edge_features=4, sigma_embed_dim=32, sh_lmax=2,
                 ns=16, nv=4, num_conv_layers=2, lig_max_radius=5, rec_max_radius=30, cross_max_distance=250,
                 center_max_distance=30, distance_embed_dim=32, cross_distance_embed_dim=32, no_torsion=False,
                 scale_by_sigma=True, norm_by_sigma=True, use_second_order_repr=False, batch_norm=True,
                 dynamic_max_cross=False, dropout=0.0, smooth_edges=False, odd_parity=False,
                 separate_noise_schedule=False, lm_embedding_type=None, confidence_mode=False,
                 confidence_dropout=0, confidence_no_batchnorm=False,
                 asyncronous_noise_schedule=False, affinity_prediction=False, parallel=1,
                 parallel_aggregators="mean max min std", num_confidence_outputs=1, fixed_center_conv=False,
                 no_aminoacid_identities=False, include_miscellaneous_atoms=False, use_old_atom_encoder=False):
        # 初始化父类
        super(CGOldModel, self).__init__()
        # 检查一些参数是否符合要求
        assert parallel == 1, "not implemented"# 暂时不支持并行化
        assert (not no_aminoacid_identities) or (lm_embedding_type is None), "no language model emb without identities"
        # 保存超参数
        self.t_to_sigma = t_to_sigma
        self.in_lig_edge_features = in_lig_edge_features
        sigma_embed_dim *= (3 if separate_noise_schedule else 1)# 如果分开噪声调度，则sigma维度乘3
        self.sigma_embed_dim = sigma_embed_dim
        self.lig_max_radius = lig_max_radius # 配置药物分子最大半径
        self.rec_max_radius = rec_max_radius# 配置受体最大半径
        self.include_miscellaneous_atoms = include_miscellaneous_atoms# 是否包含其他原子
        self.cross_max_distance = cross_max_distance # 配置交叉距离
        self.dynamic_max_cross = dynamic_max_cross# 是否动态调整最大交叉距离
        self.center_max_distance = center_max_distance# 配置中心最大距离
        self.distance_embed_dim = distance_embed_dim# 距离嵌入维度
        self.cross_distance_embed_dim = cross_distance_embed_dim# 交叉距离嵌入维度
        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax=sh_lmax)# 球面谐波表示，lmax为最高阶数
        self.ns, self.nv = ns, nv# 相关的神经网络尺寸
        self.scale_by_sigma = scale_by_sigma# 是否按sigma缩放
        self.norm_by_sigma = norm_by_sigma # 是否按sigma归一化
        self.device = device # 设备配置
        self.no_torsion = no_torsion # 是否忽略扭转角
        self.smooth_edges = smooth_edges # 是否平滑边缘
        self.odd_parity = odd_parity # 奇偶性
        self.timestep_emb_func = timestep_emb_func# 时间步长嵌入函数
        self.separate_noise_schedule = separate_noise_schedule# 是否分开噪声调度
        self.confidence_mode = confidence_mode# 是否启用置信度模式
        self.num_conv_layers = num_conv_layers# 卷积层的数量
        self.asyncronous_noise_schedule = asyncronous_noise_schedule# 是否使用异步噪声调度
        self.affinity_prediction = affinity_prediction# 是否进行亲和力预测
        self.fixed_center_conv = fixed_center_conv   # 是否固定中心卷积
        self.no_aminoacid_identities = no_aminoacid_identities # 是否不使用氨基酸标识符

         # 根据是否使用旧的原子编码器来选择使用的原子编码器类
        atom_encoder_class = OldAtomEncoder if use_old_atom_encoder else AtomEncoder

         # 药物分子节点嵌入层（原子编码器）
        self.lig_node_embedding = atom_encoder_class(emb_dim=ns, feature_dims=lig_feature_dims, sigma_embed_dim=sigma_embed_dim)
         # 药物分子边缘嵌入层
        self.lig_edge_embedding = nn.Sequential(nn.Linear(in_lig_edge_features + sigma_embed_dim + distance_embed_dim, ns),nn.ReLU(),nn.Dropout(dropout),nn.Linear(ns, ns))

         # 受体节点嵌入层
        self.rec_node_embedding = atom_encoder_class(emb_dim=ns, feature_dims=rec_residue_feature_dims, sigma_embed_dim=sigma_embed_dim, lm_embedding_type=lm_embedding_type)
        # 受体边缘嵌入层
        self.rec_edge_embedding = nn.Sequential(nn.Linear(sigma_embed_dim + distance_embed_dim, ns), nn.ReLU(), nn.Dropout(dropout),nn.Linear(ns, ns))

        if self.include_miscellaneous_atoms:# 如果包含其他原子，则创建相应的节点和边缘嵌入层
            self.misc_atom_node_embedding = atom_encoder_class(emb_dim=ns, feature_dims=rec_atom_feature_dims, sigma_embed_dim=sigma_embed_dim)
            self.misc_atom_edge_embedding = nn.Sequential(nn.Linear(sigma_embed_dim + distance_embed_dim, ns), nn.ReLU(),nn.Dropout(dropout), nn.Linear(ns, ns))
            self.ar_edge_embedding = nn.Sequential(nn.Linear(sigma_embed_dim + distance_embed_dim, ns), nn.ReLU(),nn.Dropout(dropout), nn.Linear(ns, ns))
            self.la_edge_embedding = nn.Sequential(nn.Linear(sigma_embed_dim + cross_distance_embed_dim, ns), nn.ReLU(),nn.Dropout(dropout), nn.Linear(ns, ns))

        # 配置交叉边缘嵌入层
        self.cross_edge_embedding = nn.Sequential(nn.Linear(sigma_embed_dim + cross_distance_embed_dim, ns), nn.ReLU(), nn.Dropout(dropout),nn.Linear(ns, ns))

         # 配置距离扩展层
        self.lig_distance_expansion = GaussianSmearing(0.0, lig_max_radius, distance_embed_dim)
        self.rec_distance_expansion = GaussianSmearing(0.0, rec_max_radius, distance_embed_dim)
        self.cross_distance_expansion = GaussianSmearing(0.0, cross_max_distance, cross_distance_embed_dim)

        if use_second_order_repr: # 根据是否使用二阶表示来配置表示序列
            irrep_seq = [
                f'{ns}x0e',
                f'{ns}x0e + {nv}x1o + {nv}x2e',
                f'{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o',
                f'{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o + {ns}x0o'
            ]
        else:
            irrep_seq = [
                f'{ns}x0e',
                f'{ns}x0e + {nv}x1o',
                f'{ns}x0e + {nv}x1o + {nv}x1e',
                f'{ns}x0e + {nv}x1o + {nv}x1e + {ns}x0o'
            ]

        # 配置卷积层
        lig_conv_layers, rec_conv_layers, lig_to_rec_conv_layers, rec_to_lig_conv_layers = [], [], [], []
        if self.include_miscellaneous_atoms:
            misc_conv_layers, la_conv_layers, ra_conv_layers, al_conv_layers, ar_conv_layers  = [], [], [], [], []
        for i in range(num_conv_layers): # 遍历每一层卷积，配置卷积层
            in_irreps = irrep_seq[min(i, len(irrep_seq) - 1)]
            out_irreps = irrep_seq[min(i + 1, len(irrep_seq) - 1)]
            parameters = {
                'in_irreps': in_irreps,
                'sh_irreps': self.sh_irreps,
                'out_irreps': out_irreps,
                'n_edge_features': 3 * ns,# 边缘特征
                'hidden_features': 3 * ns,# 隐藏层特征
                'residual': False,
                'batch_norm': batch_norm,# 是否使用批量归一化
                'dropout': dropout# Dropout概率
            }

            # 配置每一层卷积
            lig_layer = OldTensorProductConvLayer(**parameters)
            lig_conv_layers.append(lig_layer)
            rec_layer = OldTensorProductConvLayer(**parameters)
            rec_conv_layers.append(rec_layer)
            lig_to_rec_layer = OldTensorProductConvLayer(**parameters)
            lig_to_rec_conv_layers.append(lig_to_rec_layer)
            rec_to_lig_layer = OldTensorProductConvLayer(**parameters)
            rec_to_lig_conv_layers.append(rec_to_lig_layer)
            if self.include_miscellaneous_atoms:
                misc_conv_layer = OldTensorProductConvLayer(**parameters)
                la_conv_layer = OldTensorProductConvLayer(**parameters)
                ra_conv_layer = OldTensorProductConvLayer(**parameters)
                al_conv_layer = OldTensorProductConvLayer(**parameters)
                ar_conv_layer = OldTensorProductConvLayer(**parameters)
                misc_conv_layers.append(misc_conv_layer)
                la_conv_layers.append(la_conv_layer)
                ra_conv_layers.append(ra_conv_layer)
                al_conv_layers.append(al_conv_layer)
                ar_conv_layers.append(ar_conv_layer)

        # 将所有卷积层包装成ModuleList
        self.lig_conv_layers = nn.ModuleList(lig_conv_layers)
        self.rec_conv_layers = nn.ModuleList(rec_conv_layers)
        self.lig_to_rec_conv_layers = nn.ModuleList(lig_to_rec_conv_layers)
        self.rec_to_lig_conv_layers = nn.ModuleList(rec_to_lig_conv_layers)
         
        if self.include_miscellaneous_atoms:
            self.misc_conv_layers = nn.ModuleList(misc_conv_layers)
            self.la_conv_layers = nn.ModuleList(la_conv_layers)
            self.ra_conv_layers = nn.ModuleList(ra_conv_layers)
            self.al_conv_layers = nn.ModuleList(al_conv_layers)
            self.ar_conv_layers = nn.ModuleList(ar_conv_layers)

        if self.confidence_mode:# 如果启用了置信度模式，则添加置信度预测层
            self.confidence_predictor = nn.Sequential(
                nn.Linear(2*self.ns if num_conv_layers >= 3 else self.ns,ns),
                nn.BatchNorm1d(ns) if not confidence_no_batchnorm else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(confidence_dropout),
                nn.Linear(ns, ns),
                nn.BatchNorm1d(ns) if not confidence_no_batchnorm else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(confidence_dropout),
                nn.Linear(ns, 2 if self.affinity_prediction else 1)
            )
        else: # 如果不启用置信度模式，则创建目标层（旋转、中心和扭转角度预测层）
            # center of mass translation and rotation components
            self.center_distance_expansion = GaussianSmearing(0.0, center_max_distance, distance_embed_dim)
            self.center_edge_embedding = nn.Sequential(
                nn.Linear(distance_embed_dim + sigma_embed_dim, ns),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(ns, ns)
            )

            # 最终卷积层，基于sigma维度
            self.final_conv = OldTensorProductConvLayer(
                in_irreps=self.lig_conv_layers[-1].out_irreps,
                sh_irreps=self.sh_irreps,
                out_irreps=f'2x1o + 2x1e' if not self.odd_parity else '1x1o + 1x1e',
                n_edge_features=2 * ns,
                residual=False,
                dropout=dropout,
                batch_norm=batch_norm
            )
            self.tr_final_layer = nn.Sequential(nn.Linear(1 + sigma_embed_dim, ns),nn.Dropout(dropout), nn.ReLU(), nn.Linear(ns, 1))
            self.rot_final_layer = nn.Sequential(nn.Linear(1 + sigma_embed_dim, ns),nn.Dropout(dropout), nn.ReLU(), nn.Linear(ns, 1))

            if not no_torsion:# 如果没有禁用扭转角度，则添加扭转角度层
                # torsion angles components
                self.final_edge_embedding = nn.Sequential(
                    nn.Linear(distance_embed_dim, ns),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(ns, ns)
                )
                self.final_tp_tor = o3.FullTensorProduct(self.sh_irreps, "2e")# 二阶张量积
                self.tor_bond_conv = OldTensorProductConvLayer(
                    in_irreps=self.lig_conv_layers[-1].out_irreps,
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

    def forward(self, data):
        if self.no_aminoacid_identities: # 如果禁用了氨基酸身份标识，将受体的节点特征清零
            data['receptor'].x = data['receptor'].x * 0

        if not self.confidence_mode:# 根据是否启用置信度模式，决定如何计算 sigma
            tr_sigma, rot_sigma, tor_sigma = self.t_to_sigma(*[data.complex_t[noise_type] for noise_type in ['tr', 'rot', 'tor']])
        else:
            tr_sigma, rot_sigma, tor_sigma = [data.complex_t[noise_type] for noise_type in ['tr', 'rot', 'tor']]

        # build ligand graph # 构建配体图（Ligand Graph）
        lig_node_attr, lig_edge_index, lig_edge_attr, lig_edge_sh, lig_edge_weight = self.build_lig_conv_graph(data)
        lig_src, lig_dst = lig_edge_index
        lig_node_attr = self.lig_node_embedding(lig_node_attr)# 对配体节点特征进行嵌入
        lig_edge_attr = self.lig_edge_embedding(lig_edge_attr)# 对配体边缘特征进行嵌入

        # build receptor graph# 构建受体图（Receptor Graph）
        rec_node_attr, rec_edge_index, rec_edge_attr, rec_edge_sh, rec_edge_weight = self.build_rec_conv_graph(data)
        rec_src, rec_dst = rec_edge_index
        rec_node_attr = self.rec_node_embedding(rec_node_attr) # 对受体节点特征进行嵌入
        rec_edge_attr = self.rec_edge_embedding(rec_edge_attr)# 对受体边缘特征进行嵌入
         # 如果包含其他原子，构建这些原子的图
        if self.include_miscellaneous_atoms:
            # build misc_atom graph
            atom_node_attr, atom_edge_index, atom_edge_attr, atom_edge_sh, atom_edge_weight = self.build_misc_atom_conv_graph(data)
            atom_node_attr = self.misc_atom_node_embedding(atom_node_attr)# 嵌入其他原子的节点特征
            atom_edge_attr = self.misc_atom_edge_embedding(atom_edge_attr)# 嵌入其他原子的边缘特征

        # build cross graph
        # 构建交叉图（Cross Graph）
        if self.dynamic_max_cross:
            cross_cutoff = (tr_sigma * 3 + 20).unsqueeze(1)# 动态最大交叉距离
        else:
            cross_cutoff = self.cross_max_distance  # 静态最大交叉距离
        if self.include_miscellaneous_atoms: # 构建多个交叉边缘图：配体-受体、配体-其他原子、受体-其他原子等
            lr_edge_index, lr_edge_attr, lr_edge_sh, lr_edge_weight, la_edge_index, la_edge_attr, \
            la_edge_sh, la_edge_weight, ar_edge_index, ar_edge_attr, ar_edge_sh, ar_edge_weight = \
                self.build_misc_cross_conv_graph(data, cross_cutoff)
            lr_edge_attr = self.cross_edge_embedding(lr_edge_attr) # 对交叉边缘进行嵌入
            la_edge_attr = self.la_edge_embedding(la_edge_attr)# 对配体-其他原子边缘进行嵌入
            ar_edge_attr = self.ar_edge_embedding(ar_edge_attr)# 对受体-其他原子边缘进行嵌入
            cross_lig, cross_rec = lr_edge_index # 配体和受体之间的交叉边缘 
        else: # 如果没有包含其他原子，仅构建配体-受体的交叉图
            lr_edge_index, lr_edge_attr, lr_edge_sh, lr_edge_weight = self.build_cross_conv_graph(data, cross_cutoff)
            cross_lig, cross_rec = lr_edge_index
            lr_edge_attr = self.cross_edge_embedding(lr_edge_attr)

         # 进行多层卷积更新
        for l in range(len(self.lig_conv_layers)):
            # intra graph message passing
            # 处理配体图的内部信息传递
            lig_edge_attr_ = torch.cat([lig_edge_attr, lig_node_attr[lig_src, :self.ns], lig_node_attr[lig_dst, :self.ns]], -1)
            lig_intra_update = self.lig_conv_layers[l](lig_node_attr, lig_edge_index, lig_edge_attr_, lig_edge_sh, edge_weight=lig_edge_weight)

            # inter graph message passing
            # 处理配体与受体之间的交互信息传递
            rec_to_lig_edge_attr_ = torch.cat([lr_edge_attr, lig_node_attr[cross_lig, :self.ns], rec_node_attr[cross_rec, :self.ns]], -1)
            lig_inter_update = self.rec_to_lig_conv_layers[l](rec_node_attr, lr_edge_index, rec_to_lig_edge_attr_, lr_edge_sh,
                                                              out_nodes=lig_node_attr.shape[0], edge_weight=lr_edge_weight)
            if self.include_miscellaneous_atoms:
                # 如果包含其他原子，处理与配体和其他原子之间的交互
                la_edge_attr_ = torch.cat([la_edge_attr, lig_node_attr[la_edge_index[0], :self.ns],atom_node_attr[la_edge_index[1], :self.ns]], -1)
                la_update = self.la_conv_layers[l](atom_node_attr, la_edge_index, la_edge_attr_, la_edge_sh,out_nodes=lig_node_attr.shape[0], edge_weight=la_edge_weight)

            if l != len(self.lig_conv_layers) - 1:# 处理受体图的内部信息传递
                rec_edge_attr_ = torch.cat([rec_edge_attr, rec_node_attr[rec_src, :self.ns], rec_node_attr[rec_dst, :self.ns]], -1)
                rec_intra_update = self.rec_conv_layers[l](rec_node_attr, rec_edge_index, rec_edge_attr_, rec_edge_sh, edge_weight=rec_edge_weight)

                # 受体到配体的交互更新
                lig_to_rec_edge_attr_ = torch.cat([lr_edge_attr, lig_node_attr[cross_lig, :self.ns], rec_node_attr[cross_rec, :self.ns]], -1)
                rl_update = self.lig_to_rec_conv_layers[l](lig_node_attr, torch.flip(lr_edge_index, dims=[0]),lig_to_rec_edge_attr_,lr_edge_sh, out_nodes=rec_node_attr.shape[0],edge_weight=lr_edge_weight)
                if self.include_miscellaneous_atoms:
                    # ATOM UPDATES
                    # 处理其他原子的卷积更新
                    atom_edge_attr_ = torch.cat([atom_edge_attr, atom_node_attr[atom_edge_index[0], :self.ns],atom_node_attr[atom_edge_index[1], :self.ns]], -1)
                    atom_update = self.misc_conv_layers[l](atom_node_attr, atom_edge_index, atom_edge_attr_,atom_edge_sh, edge_weight=atom_edge_weight)

                    # 处理原子到配体、受体之间的交互
                    al_edge_attr_ = torch.cat([la_edge_attr, atom_node_attr[la_edge_index[1], :self.ns],lig_node_attr[la_edge_index[0], :self.ns]], -1)
                    al_update = self.al_conv_layers[l](lig_node_attr, torch.flip(la_edge_index, dims=[0]),al_edge_attr_,la_edge_sh, out_nodes=atom_node_attr.shape[0],edge_weight=la_edge_weight)

                    ar_edge_attr_ = torch.cat([ar_edge_attr, atom_node_attr[ar_edge_index[0], :self.ns],rec_node_attr[ar_edge_index[1], :self.ns]], -1)
                    ar_update = self.ar_conv_layers[l](rec_node_attr, ar_edge_index, ar_edge_attr_, ar_edge_sh,out_nodes=atom_node_attr.shape[0],edge_weight=ar_edge_weight)

                     # 处理原子到受体的交互
                    ra_edge_attr_ = torch.cat([ar_edge_attr, rec_node_attr[ar_edge_index[1], :self.ns],atom_node_attr[ar_edge_index[0], :self.ns]], -1)
                    ra_update = self.ra_conv_layers[l](atom_node_attr, torch.flip(ar_edge_index, dims=[0]), ra_edge_attr_, ar_edge_sh, out_nodes=rec_node_attr.shape[0], edge_weight=ar_edge_weight)

            # padding original features
            # 填充原始特征
            lig_node_attr = F.pad(lig_node_attr, (0, lig_intra_update.shape[-1] - lig_node_attr.shape[-1]))

            # update features with residual updates
            # 使用残差更新特征
            lig_node_attr = lig_node_attr + lig_intra_update + lig_inter_update
            if self.include_miscellaneous_atoms:
                lig_node_attr += la_update

             # 更新受体节点特征
            if l != len(self.lig_conv_layers) - 1:
                rec_node_attr = F.pad(rec_node_attr, (0, rec_intra_update.shape[-1] - rec_node_attr.shape[-1]))
                rec_node_attr = rec_node_attr + rec_intra_update + rl_update
                if self.include_miscellaneous_atoms:
                    rec_node_attr += ra_update
                    atom_node_attr = F.pad(atom_node_attr, (0, atom_update.shape[-1] - atom_node_attr.shape[-1]))
                    atom_node_attr = atom_node_attr + atom_update + al_update + ar_update

        # compute confidence score
        if self.confidence_mode:# 如果启用置信度模式，计算置信度得分
            scalar_lig_attr = torch.cat([lig_node_attr[:,:self.ns],lig_node_attr[:,-self.ns:] ], dim=1) if self.num_conv_layers >= 3 else lig_node_attr[:,:self.ns]
            confidence = self.confidence_predictor(scatter_mean(scalar_lig_attr, data['ligand'].batch, dim=0)).squeeze(dim=-1)
            return confidence

        # compute translational and rotational score vectors  # 计算平移和旋转得分向量
        center_edge_index, center_edge_attr, center_edge_sh = self.build_center_conv_graph(data)
        center_edge_attr = self.center_edge_embedding(center_edge_attr)
        if self.fixed_center_conv:
            center_edge_attr = torch.cat([center_edge_attr, lig_node_attr[center_edge_index[1], :self.ns]], -1)
        else:
            center_edge_attr = torch.cat([center_edge_attr, lig_node_attr[center_edge_index[0], :self.ns]], -1)
        global_pred = self.final_conv(lig_node_attr, center_edge_index, center_edge_attr, center_edge_sh, out_nodes=data.num_graphs)

        tr_pred = global_pred[:, :3] + (global_pred[:, 6:9] if not self.odd_parity else 0)
        rot_pred = global_pred[:, 3:6] + (global_pred[:, 9:] if not self.odd_parity else 0)

         # 如果启用分离的噪声调度，计算每个噪声的时间步嵌入
        if self.separate_noise_schedule:
            data.graph_sigma_emb = torch.cat([self.timestep_emb_func(data.complex_t[noise_type]) for noise_type in ['tr','rot','tor']], dim=1)
        elif self.asyncronous_noise_schedule:
            data.graph_sigma_emb = self.timestep_emb_func(data.complex_t['t'])
        else: # tr rot and tor noise is all the same in this case
            data.graph_sigma_emb = self.timestep_emb_func(data.complex_t['tr'])

        # fix the magnitude of translational and rotational score vectors
         # 固定平移和旋转得分向量的幅度
        tr_norm = torch.linalg.vector_norm(tr_pred, dim=1).unsqueeze(1)
        tr_pred = tr_pred / tr_norm * self.tr_final_layer(torch.cat([tr_norm, data.graph_sigma_emb], dim=1))
        rot_norm = torch.linalg.vector_norm(rot_pred, dim=1).unsqueeze(1)
        rot_pred = rot_pred / rot_norm * self.rot_final_layer(torch.cat([rot_norm, data.graph_sigma_emb], dim=1))

        if self.scale_by_sigma: # 如果启用了 sigma 缩放，根据 sigma 调整得分
            tr_pred = tr_pred / tr_sigma.unsqueeze(1)
            rot_pred = rot_pred * so3.score_norm(rot_sigma.cpu()).unsqueeze(1).to(data['ligand'].x.device)

         # 如果禁用了扭转或者配体没有边缘，返回平移和旋转得分
        if self.no_torsion or data['ligand'].edge_mask.sum() == 0: return tr_pred, rot_pred, torch.empty(0, device=self.device)

        # torsional components
        # 处理扭转分量（Torsional components）
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

        if self.scale_by_sigma: # 如果启用了 sigma 缩放，根据 sigma 调整扭转得分
            tor_pred = tor_pred * torch.sqrt(torch.tensor(torus.score_norm(edge_sigma.cpu().numpy())).float()
                                             .to(data['ligand'].x.device))
        return tr_pred, rot_pred, tor_pred

    def get_edge_weight(self, edge_vec, max_norm):
        # computes weights for edges that are decreasing with the distance
        # it has an effect only if smooth edges is true
        # 计算边缘的权重，这些权重与距离成反比
        if self.smooth_edges:
            normalised_norm = torch.clip(edge_vec.norm(dim=-1) * np.pi / max_norm, max=np.pi)
            return 0.5 * (torch.cos(normalised_norm) + 1.0).unsqueeze(-1)
        return 1.0

    def build_lig_conv_graph(self, data):
        # builds the ligand graph edges and initial node and edge features
        # 构建配体图的边缘和初始节点、边特征
        if self.separate_noise_schedule: # 如果启用分离噪声调度，分别计算不同噪声的嵌入
            data['ligand'].node_sigma_emb = torch.cat([self.timestep_emb_func(data['ligand'].node_t[noise_type]) for noise_type in ['tr','rot','tor']], dim=1)
        elif self.asyncronous_noise_schedule: # 如果启用异步噪声调度，计算相应的嵌入
            data['ligand'].node_sigma_emb = self.timestep_emb_func(data['ligand'].node_t['t'])
        else: # 默认情况下，平移、旋转和扭转噪声使用相同的嵌入
            data['ligand'].node_sigma_emb = self.timestep_emb_func(data['ligand'].node_t['tr']) # tr rot and tor noise is all the same

        # compute edges
        # 计算配体图的边缘
        radius_edges = radius_graph(data['ligand'].pos, self.lig_max_radius, data['ligand'].batch)
        edge_index = torch.cat([data['ligand', 'ligand'].edge_index, radius_edges], 1).long()
        edge_attr = torch.cat([
            data['ligand', 'ligand'].edge_attr,
            torch.zeros(radius_edges.shape[-1], self.in_lig_edge_features, device=data['ligand'].x.device)
        ], 0)# 为新加入的边缘初始化零特征

        # compute initial features
         # 计算节点和边缘的初始特征
        edge_sigma_emb = data['ligand'].node_sigma_emb[edge_index[0].long()]# 获取每条边的节点sigma嵌入
        edge_attr = torch.cat([edge_attr, edge_sigma_emb], 1)# 将边缘特征和sigma嵌入拼接
        node_attr = torch.cat([data['ligand'].x, data['ligand'].node_sigma_emb], 1)# 将节点特征和sigma嵌入拼接

         # 计算边向量和长度嵌入
        src, dst = edge_index
        edge_vec = data['ligand'].pos[dst.long()] - data['ligand'].pos[src.long()]# 计算边缘的向量
        edge_length_emb = self.lig_distance_expansion(edge_vec.norm(dim=-1))# 计算边长嵌入

        # 将长度嵌入、sigma嵌入与边缘特征拼接
        edge_attr = torch.cat([edge_attr, edge_length_emb], 1)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component') # 计算边缘的球面谐波特征
        edge_weight = self.get_edge_weight(edge_vec, self.lig_max_radius) # 计算边权重

        return node_attr, edge_index, edge_attr, edge_sh, edge_weight

    def build_rec_conv_graph(self, data):
        # builds the receptor initial node and edge embeddings
        # 构建受体图的初始节点和边缘嵌入
        if self.separate_noise_schedule:
            data['receptor'].node_sigma_emb = torch.cat([self.timestep_emb_func(data['receptor'].node_t[noise_type]) for noise_type in ['tr', 'rot', 'tor']], dim=1)
        elif self.asyncronous_noise_schedule:
            data['receptor'].node_sigma_emb = self.timestep_emb_func(data['receptor'].node_t['t'])
        else:
            data['receptor'].node_sigma_emb = self.timestep_emb_func(data['receptor'].node_t['tr']) # tr rot and tor noise is all the same
        # 将节点特征与sigma嵌入拼接
        node_attr = torch.cat([data['receptor'].x, data['receptor'].node_sigma_emb], 1)

        # this assumes the edges were already created in preprocessing since protein's structure is fixed
         # 假设边已经在预处理步骤中构建，受体的结构是固定的
        edge_index = data['receptor', 'receptor'].edge_index# 获取受体图的边索引
        src, dst = edge_index
        edge_vec = data['receptor'].pos[dst.long()] - data['receptor'].pos[src.long()]

        # 计算边缘长度嵌入和sigma嵌入
        edge_length_emb = self.rec_distance_expansion(edge_vec.norm(dim=-1))
        edge_sigma_emb = data['receptor'].node_sigma_emb[edge_index[0].long()]
        edge_attr = torch.cat([edge_sigma_emb, edge_length_emb], 1)# 拼接边缘特征
        # 计算球面谐波和边权重
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')
        edge_weight = self.get_edge_weight(edge_vec, self.rec_max_radius)

        return node_attr, edge_index, edge_attr, edge_sh, edge_weight

    def build_misc_atom_conv_graph(self, data):
        # build the graph between receptor misc_atoms
         # 构建受体杂项原子图
        if self.separate_noise_schedule:
            data['misc_atom'].node_sigma_emb = torch.cat([self.timestep_emb_func(data['misc_atom'].node_t[noise_type]) for noise_type in ['tr', 'rot', 'tor']],dim=1)
        elif self.asyncronous_noise_schedule:
            data['misc_atom'].node_sigma_emb = self.timestep_emb_func(data['misc_atom'].node_t['t'])
        else:
            data['misc_atom'].node_sigma_emb = self.timestep_emb_func(data['misc_atom'].node_t['tr'])  # tr rot and tor noise is all the same
          # 将节点特征与sigma嵌入拼接
        node_attr = torch.cat([data['misc_atom'].x, data['misc_atom'].node_sigma_emb], 1)

        # this assumes the edges were already created in preprocessing since protein's structure is fixed
          # 计算边缘索引
        edge_index = data['misc_atom', 'misc_atom'].edge_index
        src, dst = edge_index
        edge_vec = data['misc_atom'].pos[dst.long()] - data['misc_atom'].pos[src.long()]

        # 计算边缘长度和sigma嵌入
        edge_length_emb = self.lig_distance_expansion(edge_vec.norm(dim=-1))
        edge_sigma_emb = data['misc_atom'].node_sigma_emb[edge_index[0].long()]
        edge_attr = torch.cat([edge_sigma_emb, edge_length_emb], 1)
         # 计算球面谐波和边权重
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')
        edge_weight = self.get_edge_weight(edge_vec, self.lig_max_radius)

        return node_attr, edge_index, edge_attr, edge_sh, edge_weight

    def build_cross_conv_graph(self, data, cross_distance_cutoff):
        # builds the cross edges between ligand and receptor
        # 构建配体与受体之间的交叉边缘
        if torch.is_tensor(cross_distance_cutoff):
            # different cutoff for every graph (depends on the diffusion time)
            # 如果每个图有不同的cutoff，按批次计算
            edge_index = radius(data['receptor'].pos / cross_distance_cutoff[data['receptor'].batch],
                                data['ligand'].pos / cross_distance_cutoff[data['ligand'].batch], 1,
                                data['receptor'].batch, data['ligand'].batch, max_num_neighbors=10000)
        else:# 使用统一的cutoff计算边缘
            edge_index = radius(data['receptor'].pos, data['ligand'].pos, cross_distance_cutoff,
                            data['receptor'].batch, data['ligand'].batch, max_num_neighbors=10000)

        # 计算边缘向量和边缘长度嵌入
        src, dst = edge_index
        edge_vec = data['receptor'].pos[dst.long()] - data['ligand'].pos[src.long()]

        edge_length_emb = self.cross_distance_expansion(edge_vec.norm(dim=-1))
         # 计算边缘sigma嵌入
        edge_sigma_emb = data['ligand'].node_sigma_emb[src.long()]
        edge_attr = torch.cat([edge_sigma_emb, edge_length_emb], 1)
        # 计算球面谐波和边权重
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')

        cutoff_d = cross_distance_cutoff[data['ligand'].batch[src]].squeeze() if torch.is_tensor(cross_distance_cutoff) else cross_distance_cutoff
        edge_weight = self.get_edge_weight(edge_vec, cutoff_d)

        return edge_index, edge_attr, edge_sh, edge_weight

    def build_misc_cross_conv_graph(self, data, lr_cross_distance_cutoff):
        # build the cross edges between ligan atoms, receptor residues and receptor atoms

        # LIGAND to RECEPTOR
        # 构建配体与受体原子之间的交叉边缘
        # 配体到受体
        if torch.is_tensor(lr_cross_distance_cutoff):# 每个图有不同的cutoff，按批次计算
            # different cutoff for every graph
            lr_edge_index = radius(data['receptor'].pos / lr_cross_distance_cutoff[data['receptor'].batch],
                                data['ligand'].pos / lr_cross_distance_cutoff[data['ligand'].batch], 1,
                                data['receptor'].batch, data['ligand'].batch, max_num_neighbors=10000)
        else: # 使用统一的cutoff
            lr_edge_index = radius(data['receptor'].pos, data['ligand'].pos, lr_cross_distance_cutoff,
                            data['receptor'].batch, data['ligand'].batch, max_num_neighbors=10000)

        # 计算边缘特征
        lr_edge_vec = data['receptor'].pos[lr_edge_index[1].long()] - data['ligand'].pos[lr_edge_index[0].long()]
        lr_edge_length_emb = self.cross_distance_expansion(lr_edge_vec.norm(dim=-1))
        lr_edge_sigma_emb = data['ligand'].node_sigma_emb[lr_edge_index[0].long()]
        lr_edge_attr = torch.cat([lr_edge_sigma_emb, lr_edge_length_emb], 1)
        lr_edge_sh = o3.spherical_harmonics(self.sh_irreps, lr_edge_vec, normalize=True, normalization='component')

        cutoff_d = lr_cross_distance_cutoff[data['ligand'].batch[lr_edge_index[0]]].squeeze() \
            if torch.is_tensor(lr_cross_distance_cutoff) else lr_cross_distance_cutoff
        lr_edge_weight = self.get_edge_weight(lr_edge_vec, cutoff_d)

        # LIGAND to ATOM
        # 配体到杂项原子
        la_edge_index = radius(data['misc_atom'].pos, data['ligand'].pos, self.lig_max_radius,
                               data['misc_atom'].batch, data['ligand'].batch, max_num_neighbors=10000)

        la_edge_vec = data['misc_atom'].pos[la_edge_index[1].long()] - data['ligand'].pos[la_edge_index[0].long()]
        la_edge_length_emb = self.cross_distance_expansion(la_edge_vec.norm(dim=-1))
        la_edge_sigma_emb = data['ligand'].node_sigma_emb[la_edge_index[0].long()]
        la_edge_attr = torch.cat([la_edge_sigma_emb, la_edge_length_emb], 1)
        la_edge_sh = o3.spherical_harmonics(self.sh_irreps, la_edge_vec, normalize=True, normalization='component')
        la_edge_weight = self.get_edge_weight(la_edge_vec, self.lig_max_radius)

        # ATOM to RECEPTOR
         # 杂项原子到受体
        ar_edge_index = data['misc_atom', 'receptor'].edge_index
        ar_edge_vec = data['receptor'].pos[ar_edge_index[1].long()] - data['misc_atom'].pos[ar_edge_index[0].long()]
        ar_edge_length_emb = self.rec_distance_expansion(ar_edge_vec.norm(dim=-1))
        ar_edge_sigma_emb = data['misc_atom'].node_sigma_emb[ar_edge_index[0].long()]
        ar_edge_attr = torch.cat([ar_edge_sigma_emb, ar_edge_length_emb], 1)
        ar_edge_sh = o3.spherical_harmonics(self.sh_irreps, ar_edge_vec, normalize=True, normalization='component')
        ar_edge_weight = 1

        return lr_edge_index, lr_edge_attr, lr_edge_sh, lr_edge_weight, la_edge_index, la_edge_attr, \
               la_edge_sh, la_edge_weight, ar_edge_index, ar_edge_attr, ar_edge_sh, ar_edge_weight

    def build_center_conv_graph(self, data):
        # builds the filter and edges for the convolution generating translational and rotational scores
        # 构建用于卷积的滤波器和边缘，生成平移和旋转评分
        edge_index = torch.cat([data['ligand'].batch.unsqueeze(0), torch.arange(len(data['ligand'].batch)).to(data['ligand'].x.device).unsqueeze(0)], dim=0)

        
        # 计算配体的质心位置
        center_pos, count = torch.zeros((data.num_graphs, 3)).to(data['ligand'].x.device), torch.zeros((data.num_graphs, 3)).to(data['ligand'].x.device)
        center_pos.index_add_(0, index=data['ligand'].batch, source=data['ligand'].pos)
        center_pos = center_pos / torch.bincount(data['ligand'].batch).unsqueeze(1)

        # 计算边缘向量和距离嵌入
        edge_vec = data['ligand'].pos[edge_index[1]] - center_pos[edge_index[0]]
        edge_attr = self.center_distance_expansion(edge_vec.norm(dim=-1))
        edge_sigma_emb = data['ligand'].node_sigma_emb[edge_index[1].long()]
        edge_attr = torch.cat([edge_attr, edge_sigma_emb], 1)
        # 计算球面谐波
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')
        return edge_index, edge_attr, edge_sh

    def build_bond_conv_graph(self, data):
        # builds the graph for the convolution between the center of the rotatable bonds and the neighbouring nodes
        # 构建旋转键和邻近节点之间的卷积图
        bonds = data['ligand', 'ligand'].edge_index[:, data['ligand'].edge_mask].long()
        bond_pos = (data['ligand'].pos[bonds[0]] + data['ligand'].pos[bonds[1]]) / 2
        bond_batch = data['ligand'].batch[bonds[0]]
        # 根据键的中点位置和配体的所有节点，构建半径图
        edge_index = radius(data['ligand'].pos, bond_pos, self.lig_max_radius, batch_x=data['ligand'].batch, batch_y=bond_batch)

         # 计算边缘向量和边缘特征
        edge_vec = data['ligand'].pos[edge_index[1]] - bond_pos[edge_index[0]]
        # 获取最终边缘特征
        edge_attr = self.lig_distance_expansion(edge_vec.norm(dim=-1))

        edge_attr = self.final_edge_embedding(edge_attr)
        
        # 计算球面谐波和边权重
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')
        edge_weight = self.get_edge_weight(edge_vec, self.lig_max_radius)

        return bonds, edge_index, edge_attr, edge_sh, edge_weight

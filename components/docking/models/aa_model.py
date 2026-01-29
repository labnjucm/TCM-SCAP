from e3nn import o3
import torch
from esm.pretrained import load_model_and_alphabet
from torch import nn
from torch.nn import functional as F
from torch_cluster import radius, radius_graph
from torch_geometric.utils import subgraph
from torch_scatter import scatter_mean
import numpy as np

from models.layers import GaussianSmearing, AtomEncoder
from models.tensor_layers import get_irrep_seq, TensorProductConvLayer
from utils import so3, torus
from datasets.process_mols import lig_feature_dims, rec_residue_feature_dims, rec_atom_feature_dims

# 定义聚合函数（mean, max, min, std）
AGGREGATORS = {"mean": lambda x: torch.mean(x, dim=1),
               "max": lambda x: torch.max(x, dim=1)[0],
               "min": lambda x: torch.min(x, dim=1)[0],
               "std": lambda x: torch.std(x, dim=1)}



# 定义一个继承自torch.nn.Module的模型类AAModel
class AAModel(torch.nn.Module):
    def __init__(self, t_to_sigma, device, timestep_emb_func, in_lig_edge_features=4, sigma_embed_dim=32, sh_lmax=2,
                 ns=16, nv=4, num_conv_layers=2, lig_max_radius=5, rec_max_radius=30, cross_max_distance=250,
                 center_max_distance=30, distance_embed_dim=32, cross_distance_embed_dim=32, no_torsion=False,
                 scale_by_sigma=True, norm_by_sigma=True, use_second_order_repr=False, batch_norm=True,
                 dynamic_max_cross=False, dropout=0.0, smooth_edges=False, odd_parity=False,
                 separate_noise_schedule=False, lm_embedding_type=False, confidence_mode=False,
                 confidence_dropout=0, confidence_no_batchnorm = False,
                 asyncronous_noise_schedule=False, affinity_prediction=False, parallel=1,
                 parallel_aggregators="mean max min std", num_confidence_outputs=1, atom_num_confidence_outputs=1, fixed_center_conv=False,
                 no_aminoacid_identities=False, include_miscellaneous_atoms=False,
                 differentiate_convolutions=True, tp_weights_layers=2, num_prot_emb_layers=0,
                 reduce_pseudoscalars=False, embed_also_ligand=False, atom_confidence=False, sidechain_pred=False,
                 depthwise_convolution=False, crop_beyond=None):
        super(AAModel, self).__init__()
        # 参数检查
        assert (not no_aminoacid_identities) or (lm_embedding_type is None), "no language model emb without identities"
        assert not sidechain_pred, "sidechain prediction not implemented/makes sense for all atom model"
        assert not depthwise_convolution, "depthwise convolution not implemented for all atom model"
        if parallel > 1: assert affinity_prediction

        # 模型的主要配置参数
        self.t_to_sigma = t_to_sigma
        self.in_lig_edge_features = in_lig_edge_features
        sigma_embed_dim *= (3 if separate_noise_schedule else 1)# 根据是否分离噪声调度修改sigma嵌入维度
        self.sigma_embed_dim = sigma_embed_dim
        self.sigma_embed_dim = sigma_embed_dim
        self.lig_max_radius = lig_max_radius# 配置配体的最大半径
        self.rec_max_radius = rec_max_radius# 配置受体的最大半径
        self.cross_max_distance = cross_max_distance # 配体与受体之间的最大距离
        self.dynamic_max_cross = dynamic_max_cross# 是否使用动态最大跨距
        self.center_max_distance = center_max_distance# 中心的最大距离
        self.distance_embed_dim = distance_embed_dim # 距离嵌入维度
        self.cross_distance_embed_dim = cross_distance_embed_dim # 配体与受体之间的距离嵌入维度
        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax=sh_lmax) # 球面谐波的不可约表示
        self.ns, self.nv = ns, nv
        self.scale_by_sigma = scale_by_sigma# 是否按sigma缩放
        self.norm_by_sigma = norm_by_sigma# 是否按sigma归一化
        self.device = device
        self.no_torsion = no_torsion# 是否启用扭转角
        self.smooth_edges = smooth_edges # 是否平滑边缘
        self.odd_parity = odd_parity# 是否启用奇偶性
        self.num_conv_layers = num_conv_layers# 卷积层数量
        self.timestep_emb_func = timestep_emb_func # 时间步嵌入函数
        self.separate_noise_schedule = separate_noise_schedule# 是否分离噪声调度
        self.confidence_mode = confidence_mode# 是否启用置信度模式
        self.num_conv_layers = num_conv_layers# 蛋白质嵌入层的数量
        self.num_prot_emb_layers = num_prot_emb_layers# 蛋白质嵌入层的数量
        self.asyncronous_noise_schedule = asyncronous_noise_schedule# 是否使用异步噪声调度
        self.affinity_prediction = affinity_prediction # 是否进行亲和力预测
        self.parallel, self.parallel_aggregators = parallel, parallel_aggregators.split(' ') # 并行配置
        self.fixed_center_conv = fixed_center_conv# 是否固定中心卷积
        self.no_aminoacid_identities = no_aminoacid_identities# 是否没有氨基酸身份
        self.differentiate_convolutions = differentiate_convolutions# 是否区分卷积
        self.reduce_pseudoscalars = reduce_pseudoscalars# 是否减少伪标量
        self.atom_confidence = atom_confidence# 是否启用原子置信度
        self.atom_num_confidence_outputs = atom_num_confidence_outputs# 原子置信度输出数量
        self.crop_beyond = crop_beyond# 是否裁剪超出边界的部分

        # 语言模型嵌入类型的配置
        self.lm_embedding_type = lm_embedding_type
        if lm_embedding_type is None:
            lm_embedding_dim = 0
        elif lm_embedding_type == "precomputed":
            lm_embedding_dim=1280# 预计算的嵌入维度
        else:
            lm, alphabet = load_model_and_alphabet(lm_embedding_type)
            self.batch_converter = alphabet.get_batch_converter()
            lm.lm_head = torch.nn.Identity()# 禁用语言模型头
            lm.contact_head = torch.nn.Identity()# 禁用接触头
            lm_embedding_dim = lm.embed_dim
            self.lm = lm# 将加载的语言模型赋值给self.lm

        # embedding layers
        # 嵌入层
        atom_encoder_class = AtomEncoder# 原子编码器类
        # 配体的节点嵌入层
        self.lig_node_embedding = atom_encoder_class(emb_dim=ns, feature_dims=lig_feature_dims, sigma_embed_dim=sigma_embed_dim)
        # 配体的边缘嵌入层
        self.lig_edge_embedding = nn.Sequential(nn.Linear(in_lig_edge_features + sigma_embed_dim + distance_embed_dim, ns),nn.ReLU(),nn.Dropout(dropout),nn.Linear(ns, ns))

        # 受体的节点嵌入层
        self.rec_sigma_embedding = nn.Sequential(nn.Linear(sigma_embed_dim, ns), nn.ReLU(), nn.Dropout(dropout), nn.Linear(ns, ns))
        self.rec_node_embedding = atom_encoder_class(emb_dim=ns, feature_dims=rec_residue_feature_dims, sigma_embed_dim=0, lm_embedding_dim=lm_embedding_dim)
        self.rec_edge_embedding = nn.Sequential(nn.Linear(distance_embed_dim, ns), nn.ReLU(), nn.Dropout(dropout), nn.Linear(ns, ns))
         # 原子节点嵌入层
        self.atom_node_embedding = atom_encoder_class(emb_dim=ns, feature_dims=rec_atom_feature_dims, sigma_embed_dim=0)
        self.atom_edge_embedding = nn.Sequential(nn.Linear(distance_embed_dim, ns), nn.ReLU(), nn.Dropout(dropout), nn.Linear(ns, ns))

        self.lr_edge_embedding = nn.Sequential(nn.Linear(sigma_embed_dim + cross_distance_embed_dim, ns), nn.ReLU(), nn.Dropout(dropout),nn.Linear(ns, ns))
        self.ar_edge_embedding = nn.Sequential(nn.Linear(distance_embed_dim, ns), nn.ReLU(), nn.Dropout(dropout),nn.Linear(ns, ns))
        self.la_edge_embedding = nn.Sequential(nn.Linear(sigma_embed_dim + cross_distance_embed_dim, ns), nn.ReLU(), nn.Dropout(dropout),nn.Linear(ns, ns))

        # 配体、受体的距离扩展层
        self.lig_distance_expansion = GaussianSmearing(0.0, lig_max_radius, distance_embed_dim)
        self.rec_distance_expansion = GaussianSmearing(0.0, rec_max_radius, distance_embed_dim)
        self.cross_distance_expansion = GaussianSmearing(0.0, cross_max_distance, cross_distance_embed_dim)

        # 获取不可约表示序列
        irrep_seq = get_irrep_seq(ns, nv, use_second_order_repr, reduce_pseudoscalars)
        assert not include_miscellaneous_atoms, "currently not supported"

         # 受体嵌入层的定义
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
                edge_groups=1 if not differentiate_convolutions else 4,
            )
            rec_emb_layers.append(layer)
        self.rec_emb_layers = nn.ModuleList(rec_emb_layers)# 将所有受体嵌入层加入到ModuleList中

        # 配体嵌入层
        self.embed_also_ligand = embed_also_ligand # 是否也进行配体嵌入
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
                )
                lig_emb_layers.append(layer)
            self.lig_emb_layers = nn.ModuleList(lig_emb_layers)

        # convolutional layers
        # 卷积层
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
                edge_groups=1 if not differentiate_convolutions else (3 if i == num_prot_emb_layers + num_conv_layers - 1 else 9),
            )
            conv_layers.append(layer)
        self.conv_layers = nn.ModuleList(conv_layers) # 将所有卷积层加入到ModuleList中

        # confidence and affinity prediction layers
        if self.confidence_mode:# 如果启用置信度模式
            if self.affinity_prediction:# 如果启用亲和力预测
                if self.parallel > 1:# 如果启用并行
                    output_confidence_dim = 1 + ns
                else:
                    output_confidence_dim = num_confidence_outputs + 1# 否则，输出维度为num_confidence_outputs + 1
            else:
                output_confidence_dim = num_confidence_outputs# 如果不进行亲和力预测，输出维度为num_confidence_outputs

            # 根据卷积层和嵌入层的数量调整输入大小
            input_size = ns + (nv if reduce_pseudoscalars else ns) if num_conv_layers + num_prot_emb_layers >= 3 else ns

            if self.atom_confidence:# 如果启用了原子置信度预测
                # 原子置信度预测器
                self.atom_confidence_predictor = nn.Sequential(
                    nn.Linear(input_size, ns),# 第一个线性层
                    nn.BatchNorm1d(ns) if not confidence_no_batchnorm else nn.Identity(),
                    nn.ReLU(),# 激活函数
                    nn.Dropout(confidence_dropout),# Dropout层
                    nn.Linear(ns, ns),# 第二个线性层
                    nn.BatchNorm1d(ns) if not confidence_no_batchnorm else nn.Identity(),
                    nn.ReLU(),
                    nn.Dropout(confidence_dropout),
                    nn.Linear(ns, atom_num_confidence_outputs + ns)# 输出层
                )
                input_size = ns# 更新输入大小为ns

             # 置信度预测器
            self.confidence_predictor = nn.Sequential(
                nn.Linear(input_size, ns),# 第一个线性层
                nn.BatchNorm1d(ns) if not confidence_no_batchnorm else nn.Identity(),# 是否使用批量归一化
                nn.ReLU(),# 激活函数
                nn.Dropout(confidence_dropout),# Dropout层
                nn.Linear(ns, ns),# 第二个线性层
                nn.BatchNorm1d(ns) if not confidence_no_batchnorm else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(confidence_dropout),
                nn.Linear(ns, output_confidence_dim)# 输出层，维度由output_confidence_dim决定
            )

            if self.parallel > 1: # 如果启用了并行
                # 亲和力预测器
                self.affinity_predictor = nn.Sequential(
                    nn.Linear(len(self.parallel_aggregators) * ns, ns),# 第一个线性层，输入维度是并行聚合器数量乘以ns
                    nn.BatchNorm1d(ns) if not confidence_no_batchnorm else nn.Identity(),
                    nn.ReLU(),
                    nn.Dropout(confidence_dropout),# Dropout层
                    nn.Linear(ns, ns),# 第二个线性层
                    nn.BatchNorm1d(ns) if not confidence_no_batchnorm else nn.Identity(),
                    nn.ReLU(),
                    nn.Dropout(confidence_dropout),
                    nn.Linear(ns, 1)# 输出层，预测亲和力
                )

        else:# 如果没有启用置信度模式
            # convolution for translational and rotational scores
            # 处理平移和旋转分数的卷积层
            self.center_distance_expansion = GaussianSmearing(0.0, center_max_distance, distance_embed_dim)# 中心距离的高斯扩展
            self.center_edge_embedding = nn.Sequential(
                nn.Linear(distance_embed_dim + sigma_embed_dim, ns),# 第一个线性层，输入维度是距离和sigma嵌入维度的总和
                nn.ReLU(),
                nn.Dropout(dropout),# Dropout层
                nn.Linear(ns, ns)
            )

            # 最终卷积层，用于处理卷积特征
            self.final_conv = TensorProductConvLayer(
                in_irreps=self.conv_layers[-1].out_irreps, # 上一层的输出不可约表示
                sh_irreps=self.sh_irreps,# 球面谐波表示
                out_irreps=f'2x1o + 2x1e' if not self.odd_parity else '1x1o + 1x1e', # 输出的不可约表示
                n_edge_features=2 * ns,# 边缘特征数量
                residual=False,# 是否使用残差连接
                dropout=dropout,# Dropout层
                batch_norm=batch_norm # 是否使用批量归一化
            )

            # 平移分数的最终层
            self.tr_final_layer = nn.Sequential(nn.Linear(1 + sigma_embed_dim, ns),nn.Dropout(dropout), nn.ReLU(), nn.Linear(ns, 1))# 线性层，输入是sigma嵌入和1
            # 旋转分数的最终层
            self.rot_final_layer = nn.Sequential(nn.Linear(1 + sigma_embed_dim, ns),nn.Dropout(dropout), nn.ReLU(), nn.Linear(ns, 1))

            # 如果没有禁用扭转角度
            if not no_torsion:
                # convolution for torsional score
                # 处理扭转分数的卷积层
                self.final_edge_embedding = nn.Sequential(
                    nn.Linear(distance_embed_dim, ns),# 线性层，输入是距离嵌入维度
                    nn.ReLU(),# 激活函数
                    nn.Dropout(dropout), # Dropout层
                    nn.Linear(ns, ns) # 输出层
                )
                # 扭转的完整张量积
                self.final_tp_tor = o3.FullTensorProduct(self.sh_irreps, "2e")
                self.tor_bond_conv = TensorProductConvLayer(
                    in_irreps=self.conv_layers[-1].out_irreps,# 上一层的输出不可约表示
                    sh_irreps=self.final_tp_tor.irreps_out,# 扭转的输出不可约表示
                    out_irreps=f'{ns}x0o + {ns}x0e' if not self.odd_parity else f'{ns}x0o', # 输出的不可约表示
                    n_edge_features=3 * ns, # 边缘特征数量
                    residual=False, # 是否使用残差连接
                    dropout=dropout,# Dropout层
                    batch_norm=batch_norm # 是否使用批量归一化
                )
                # 扭转分数的最终层
                self.tor_final_layer = nn.Sequential(
                    nn.Linear(2 * ns if not self.odd_parity else ns, ns, bias=False),# 线性层，输入维度为2 * ns或ns
                    nn.Tanh(), # 激活函数
                    nn.Dropout(dropout),
                    nn.Linear(ns, 1, bias=False)
                )

    def embedding(self, data):
        # 判断受体（receptor）是否包含 'rec_node_attr' 属性
        if not hasattr(data['receptor'], "rec_node_attr"):
             # 如果未启用预计算的嵌入，且未选择“None”作为嵌入类型
            if self.lm_embedding_type not in [None, 'precomputed']:
                # 扁平化序列数据，提取受体序列
                sequences = [s for l in data['receptor'].sequence for s in l]
                if isinstance(sequences[0], list):
                    sequences = [s for l in sequences for s in l]
                sequences = [(i, s) for i, s in enumerate(sequences)]
                # 将序列转换为批次标签、字符串和令牌
                batch_labels, batch_strs, batch_tokens = self.batch_converter(sequences)
                # 使用语言模型（lm）进行嵌入处理
                out = self.lm(batch_tokens.to(data['receptor'].x.device), repr_layers=[self.lm.num_layers], return_contacts=False)
                # 拼接受体的语言模型嵌入
                rec_lm_emb = torch.cat([t[:len(sequences[i][1])] for i, t in enumerate(out['representations'][self.lm.num_layers])], dim=0)
                 # 将受体的原始特征与语言模型嵌入拼接在一起
                data['receptor'].x = torch.cat([data['receptor'].x, rec_lm_emb], dim=-1)

             # 构建受体、原子和交叉受体原子图的卷积图
            rec_node_attr, rec_edge_attr, rec_edge_sh, rec_edge_weight = self.build_rec_conv_graph(data)
            rec_node_attr = self.rec_node_embedding(rec_node_attr)
            rec_edge_attr = self.rec_edge_embedding(rec_edge_attr)

            atom_node_attr, atom_edge_attr, atom_edge_sh, atom_edge_weight = self.build_atom_conv_graph(data)
            atom_node_attr = self.atom_node_embedding(atom_node_attr)
            atom_edge_attr = self.atom_edge_embedding(atom_edge_attr)

            ar_edge_attr, ar_edge_sh, ar_edge_weight = self.build_cross_rec_conv_graph(data)
            ar_edge_attr = self.ar_edge_embedding(ar_edge_attr)

            # 获取受体、原子和交叉受体原子的边缘索引
            rec_edge_index = data['receptor', 'receptor'].edge_index.clone()
            atom_edge_index = data['atom', 'atom'].edge_index.clone()
            ar_edge_index = data['atom', 'receptor'].edge_index.clone()

            # 将受体和原子的节点属性合并
            node_attr = torch.cat([rec_node_attr, atom_node_attr], dim=0)
             # 调整交叉边缘索引的节点偏移
            ar_edge_index[0] = ar_edge_index[0] + len(rec_node_attr)
            # 合并所有边缘索引
            edge_index = torch.cat([rec_edge_index, ar_edge_index, atom_edge_index + len(rec_node_attr), torch.flip(ar_edge_index, dims=[0])], dim=1)
             # 合并所有边缘属性
            edge_attr = torch.cat([rec_edge_attr, ar_edge_attr, atom_edge_attr, ar_edge_attr], dim=0)
            edge_sh = torch.cat([rec_edge_sh, ar_edge_sh, atom_edge_sh, ar_edge_sh], dim=0)
            # 合并所有边缘权重（如果没有指定，默认为全1）
            edge_weight = torch.cat([rec_edge_weight, ar_edge_weight, atom_edge_weight, ar_edge_weight], dim=0) \
                if torch.is_tensor(rec_edge_weight) else torch.ones((len(edge_index[0]), 1), device=edge_index.device)
            # 计算不同类型边缘的拆分点
            s1, s2, s3 = len(rec_edge_index[0]), len(rec_edge_index[0]) + len(ar_edge_index[0]), len(rec_edge_index[0]) + len(ar_edge_index[0]) + len(atom_edge_index[0])

            # 通过受体嵌入层计算节点属性
            for l in range(len(self.rec_emb_layers)):
                # 合并边缘属性、节点属性并通过卷积层处理
                edge_attr_ = torch.cat(
                    [edge_attr, node_attr[edge_index[0], :self.ns], node_attr[edge_index[1], :self.ns]], -1)
                # 如果启用了不同的卷积操作，则拆分边缘属性
                if self.differentiate_convolutions: edge_attr_ = [edge_attr_[:s1], edge_attr_[s1:s2], edge_attr_[s2:s3], edge_attr_[s3:]]
                node_attr = self.rec_emb_layers[l](node_attr, edge_index, edge_attr_, edge_sh, edge_weight=edge_weight)


            # 更新受体的节点和边缘属性
            data['receptor'].rec_node_attr = node_attr[:len(rec_node_attr)]
            data['receptor', 'receptor'].rec_edge_attr = rec_edge_attr
            data['receptor', 'receptor'].edge_sh = rec_edge_sh
            data['receptor', 'receptor'].edge_weight = rec_edge_weight

             # 更新原子的节点和边缘属性
            data['atom'].atom_node_attr = node_attr[len(rec_node_attr):]
            data['atom', 'atom'].atom_edge_attr = atom_edge_attr
            data['atom', 'atom'].edge_sh = atom_edge_sh
            data['atom', 'atom'].edge_weight = atom_edge_weight

            # 更新交叉受体原子的边缘属性
            data['atom', 'receptor'].edge_attr = ar_edge_attr
            data['atom', 'receptor'].edge_sh = ar_edge_sh
            data['atom', 'receptor'].edge_weight = ar_edge_weight

        # receptor embedding
        # 受体嵌入处理
        rec_sigma_emb = self.rec_sigma_embedding(self.timestep_emb_func(data.complex_t['tr']))
        rec_node_attr = data['receptor'].rec_node_attr + 0
        rec_node_attr[:, :self.ns] = rec_node_attr[:, :self.ns] + rec_sigma_emb[data['receptor'].batch]
        rec_edge_attr = data['receptor', 'receptor'].rec_edge_attr + rec_sigma_emb[data['receptor'].batch[data['receptor', 'receptor'].edge_index[0]]]

        # atom embedding
        # 原子嵌入处理
        atom_node_attr = data['atom'].atom_node_attr + 0
        atom_node_attr[:, :self.ns] = atom_node_attr[:, :self.ns] + rec_sigma_emb[data['atom'].batch]
        atom_edge_attr = data['atom', 'atom'].atom_edge_attr + rec_sigma_emb[data['atom'].batch[data['atom', 'atom'].edge_index[0]]]

        # atom-receptor embedding
         # 交叉受体原子嵌入处理
        ar_edge_attr = data['atom', 'receptor'].edge_attr + rec_sigma_emb[data['atom'].batch[data['atom', 'receptor'].edge_index[0]]]

        # ligand embedding
        # 配体嵌入处理
        lig_node_attr, lig_edge_index, lig_edge_attr, lig_edge_sh, lig_edge_weight = self.build_lig_conv_graph(data)
        lig_node_attr = self.lig_node_embedding(lig_node_attr)
        lig_edge_attr = self.lig_edge_embedding(lig_edge_attr)

        # 如果需要还包括配体嵌入，则通过配体嵌入层计算节点属性
        if self.embed_also_ligand:
            for l in range(len(self.lig_emb_layers)):
                edge_attr_ = torch.cat([lig_edge_attr, lig_node_attr[lig_edge_index[0], :self.ns], lig_node_attr[lig_edge_index[1], :self.ns]], -1)
                lig_node_attr = self.lig_emb_layers[l](lig_node_attr, lig_edge_index, edge_attr_, lig_edge_sh, edge_weight=lig_edge_weight)

        else: # 否则，填充配体的节点属性，以便其与受体的节点属性维度匹配
            lig_node_attr = F.pad(lig_node_attr, (0, rec_node_attr.shape[-1] - lig_node_attr.shape[-1]))

         # 返回各类嵌入结果，包括配体、受体和原子的节点属性、边缘属性等
        return lig_node_attr, lig_edge_index, lig_edge_attr, lig_edge_sh, lig_edge_weight, \
               rec_node_attr, data['receptor', 'receptor'].edge_index, rec_edge_attr, data['receptor', 'receptor'].edge_sh, data['receptor', 'receptor'].edge_weight, \
               atom_node_attr, data['atom', 'atom'].edge_index, atom_edge_attr, data['atom', 'atom'].edge_sh, data['atom', 'atom'].edge_weight, \
               data['atom', 'receptor'].edge_index, ar_edge_attr, data['atom', 'receptor'].edge_sh, data['atom', 'receptor'].edge_weight

    def forward(self, data):
        if self.crop_beyond is not None:# 如果设置了 crop_beyond，进行裁剪处理
            # TODO missing filtering atoms目前缺少过滤原子功能
            raise NotImplementedError
             # 获取配体和受体的坐标位置
            ligand_pos = data['ligand'].pos
            receptor_pos = data['receptor'].pos
            # 计算配体和受体之间的距离并确定需要保留的残基
            residues_to_keep = torch.any(torch.sum((ligand_pos.unsqueeze(0) - receptor_pos.unsqueeze(1)) ** 2, -1) < self.crop_beyond ** 2, dim=1)

            # 根据保留的残基索引裁剪受体数据
            data['receptor'].pos = data['receptor'].pos[residues_to_keep]
            data['receptor'].x = data['receptor'].x[residues_to_keep]
            data['receptor'].side_chain_vecs = data['receptor'].side_chain_vecs[residues_to_keep]
            # 根据裁剪后的索引更新受体之间的边缘信息
            data['receptor', 'rec_contact', 'receptor'].edge_index = subgraph(residues_to_keep, data['receptor', 'rec_contact', 'receptor'].edge_index, relabel_nodes=True)[0]

        if self.no_aminoacid_identities:# 如果设置了 no_aminoacid_identities，清空受体的 x 属性
            data['receptor'].x = data['receptor'].x * 0

        if not self.confidence_mode:# 如果没有开启 confidence_mode，则根据复杂体数据计算噪声
            tr_sigma, rot_sigma, tor_sigma = self.t_to_sigma(*[data.complex_t[noise_type] for noise_type in ['tr', 'rot', 'tor']])
        else:
            tr_sigma, rot_sigma, tor_sigma = [data.complex_t[noise_type] for noise_type in ['tr', 'rot', 'tor']]

        # 获取嵌入表示，包括配体、受体、原子等的节点和边的特征
        lig_node_attr, lig_edge_index, lig_edge_attr, lig_edge_sh, lig_edge_weight, rec_node_attr, \
            rec_edge_index, rec_edge_attr, rec_edge_sh, rec_edge_weight,\
            atom_node_attr, atom_edge_index, atom_edge_attr, atom_edge_sh, atom_edge_weight, \
            ar_edge_index, ar_edge_attr, ar_edge_sh, ar_edge_weight = self.embedding(data)

        # build lig cross graph
        # 构建配体交叉图
        cross_cutoff = (tr_sigma * 3 + 20).unsqueeze(1) if self.dynamic_max_cross else self.cross_max_distance
        lr_edge_index, lr_edge_attr, lr_edge_sh, lr_edge_weight, la_edge_index, la_edge_attr, \
            la_edge_sh, la_edge_weight = self.build_cross_lig_conv_graph(data, cross_cutoff)
         # 嵌入配体交叉图的边特征
        lr_edge_attr= self.lr_edge_embedding(lr_edge_attr)
        la_edge_attr = self.la_edge_embedding(la_edge_attr)

        # 获取配体和受体的节点数量
        n_lig, n_rec = len(lig_node_attr), len(rec_node_attr)

        # 将配体、受体和原子的节点特征拼接在一起
        node_attr = torch.cat([lig_node_attr, rec_node_attr, atom_node_attr], dim=0)
        # 复制和调整受体和原子边的索引，确保所有图形的一致性
        rec_edge_index, atom_edge_index, lr_edge_index, la_edge_index, ar_edge_index = rec_edge_index.clone(), atom_edge_index.clone(), lr_edge_index.clone(), la_edge_index.clone(), ar_edge_index.clone()
        rec_edge_index[0], rec_edge_index[1] = rec_edge_index[0] + n_lig, rec_edge_index[1] + n_lig
        atom_edge_index[0], atom_edge_index[1] = atom_edge_index[0] + n_lig + n_rec, atom_edge_index[1] + n_lig + n_rec
        lr_edge_index[1] = lr_edge_index[1] + n_lig
        la_edge_index[1] = la_edge_index[1] + n_lig + n_rec
        ar_edge_index[0], ar_edge_index[1] = ar_edge_index[0] + n_lig + n_rec, ar_edge_index[1] + n_lig

        # 拼接所有的边信息
        edge_index = torch.cat([lig_edge_index, lr_edge_index, la_edge_index, rec_edge_index,
                                torch.flip(lr_edge_index, dims=[0]), torch.flip(ar_edge_index, dims=[0]),
                                atom_edge_index, torch.flip(la_edge_index, dims=[0]), ar_edge_index], dim=1)
         # 拼接所有边的特征、属性和权重
        edge_attr = torch.cat([lig_edge_attr, lr_edge_attr, la_edge_attr, rec_edge_attr, lr_edge_attr,
                               ar_edge_attr, atom_edge_attr, la_edge_attr, ar_edge_attr], dim=0)
        edge_sh = torch.cat([lig_edge_sh, lr_edge_sh, la_edge_sh, rec_edge_sh, lr_edge_sh, ar_edge_sh,
                             atom_edge_sh, la_edge_sh, ar_edge_sh], dim=0)
        edge_weight = torch.cat([lig_edge_weight, lr_edge_weight, la_edge_weight, rec_edge_weight, lr_edge_weight,
                                 ar_edge_weight, atom_edge_weight, la_edge_weight, ar_edge_weight],
                                dim=0) if torch.is_tensor(lig_edge_weight) else torch.ones((len(edge_index[0]), 1),
                                                                                           device=edge_index.device)
        # 使用累加和计算不同类型的边
        s1, s2, s3, s4, s5, s6, s7, s8, _ = tuple(np.cumsum(list(map(len, [lig_edge_attr, lr_edge_attr, la_edge_attr,
                                                                           rec_edge_attr, lr_edge_attr, ar_edge_attr, atom_edge_attr, la_edge_attr, ar_edge_attr]))).tolist())

        # 使用卷积层进行特征学习
        for l in range(len(self.conv_layers)):
            if l < len(self.conv_layers) - 1:
                edge_attr_ = torch.cat([edge_attr, node_attr[edge_index[0], :self.ns], node_attr[edge_index[1], :self.ns]], -1)
                if self.differentiate_convolutions: edge_attr_ = [edge_attr_[:s1], edge_attr_[s1:s2], edge_attr_[s2:s3], edge_attr_[s3:s4],
                                                                  edge_attr_[s4:s5], edge_attr_[s5:s6], edge_attr_[s6:s7], edge_attr_[s7:s8], edge_attr_[s8:]]
                node_attr = self.conv_layers[l](node_attr, edge_index, edge_attr_, edge_sh, edge_weight=edge_weight)
            else:
                edge_attr_ = torch.cat([edge_attr[:s3], node_attr[edge_index[0, :s3], :self.ns], node_attr[edge_index[1, :s3], :self.ns]], -1)
                if self.differentiate_convolutions: edge_attr_ = [edge_attr_[:s1], edge_attr_[s1:s2], edge_attr_[s2:s3]]
                node_attr = self.conv_layers[l](node_attr, edge_index[:, :s3], edge_attr_, edge_sh[:s3], edge_weight=edge_weight[:s3])

        # 获取最终配体节点特征
        lig_node_attr = node_attr[:len(lig_node_attr)]

        # confidence and affinity prediction
         # 如果是信心模式，则进行信心和亲和力预测
        if self.confidence_mode:
            scalar_lig_attr = torch.cat([lig_node_attr[:,:self.ns], lig_node_attr[:,-(self.nv if self.reduce_pseudoscalars else self.ns):] ], dim=1) \
                if self.num_conv_layers + self.num_prot_emb_layers >= 3 else lig_node_attr[:,:self.ns]

            # 预测原子的信心值
            if self.atom_confidence:
                scalar_lig_attr = self.atom_confidence_predictor(scalar_lig_attr)
                atom_confidence = scalar_lig_attr[:, :self.atom_num_confidence_outputs]
                scalar_lig_attr = scalar_lig_attr[:, self.atom_num_confidence_outputs:]
            else:
                atom_confidence = torch.zeros((len(lig_node_attr),), device=lig_node_attr.device)

             # 计算信心值
            confidence = self.confidence_predictor(scatter_mean(scalar_lig_attr, data['ligand'].batch, dim=0)).squeeze(dim=-1)

            if self.parallel > 1:
                confidence, affinity = confidence[:, 0], confidence[:, 1:]
                confidence = confidence.reshape(data.num_graphs, self.parallel)
                affinity = affinity.reshape(data.num_graphs, self.parallel, -1)
                affinity = torch.cat([AGGREGATORS[agg](affinity) for agg in self.parallel_aggregators], dim=-1)
                affinity = self.affinity_predictor(affinity).squeeze(dim=-1)
                confidence = confidence, affinity
            return confidence, atom_confidence
        # 如果并行为1，则继续执行
        assert self.parallel == 1

        # compute translational and rotational score vectors
        # 计算平移和旋转分数向量
        center_edge_index, center_edge_attr, center_edge_sh = self.build_center_conv_graph(data)
        center_edge_attr = self.center_edge_embedding(center_edge_attr)
        if self.fixed_center_conv:
            center_edge_attr = torch.cat([center_edge_attr, lig_node_attr[center_edge_index[1], :self.ns]], -1)
        else:
            center_edge_attr = torch.cat([center_edge_attr, lig_node_attr[center_edge_index[0], :self.ns]], -1)
         # 进行最终的卷积操作
        global_pred = self.final_conv(lig_node_attr, center_edge_index, center_edge_attr, center_edge_sh, out_nodes=data.num_graphs)

         # 提取平移和旋转的预测结果
        tr_pred = global_pred[:, :3] + (global_pred[:, 6:9] if not self.odd_parity else 0)
        rot_pred = global_pred[:, 3:6] + (global_pred[:, 9:] if not self.odd_parity else 0)

        # 使用噪声嵌入和最终层进行规范化
        if self.separate_noise_schedule:
            data.graph_sigma_emb = torch.cat([self.timestep_emb_func(data.complex_t[noise_type]) for noise_type in ['tr', 'rot', 'tor']], dim=1)
        elif self.asyncronous_noise_schedule:
            data.graph_sigma_emb = self.timestep_emb_func(data.complex_t['t'])
        else:  # tr rot and tor noise is all the same in this case # 平移、旋转和扭转噪声都相同
            data.graph_sigma_emb = self.timestep_emb_func(data.complex_t['tr'])

        # adjust the magniture of the score vectors
        # 调整分数向量的大小
        tr_norm = torch.linalg.vector_norm(tr_pred, dim=1).unsqueeze(1)
        tr_pred = tr_pred / tr_norm * self.tr_final_layer(torch.cat([tr_norm, data.graph_sigma_emb], dim=1))

        rot_norm = torch.linalg.vector_norm(rot_pred, dim=1).unsqueeze(1)
        rot_pred = rot_pred / rot_norm * self.rot_final_layer(torch.cat([rot_norm, data.graph_sigma_emb], dim=1))

        # 如果开启了按sigma缩放，进行缩放操作
        if self.scale_by_sigma:
            tr_pred = tr_pred / tr_sigma.unsqueeze(1)
            rot_pred = rot_pred * so3.score_norm(rot_sigma.cpu()).unsqueeze(1).to(data['ligand'].x.device)

        # 如果不需要扭转或配体的边没有连接，则返回平移和旋转预测
        if self.no_torsion or data['ligand'].edge_mask.sum() == 0: return tr_pred, rot_pred, torch.empty(0,device=self.device), None

        # torsional components
         # 扭转成分处理
        tor_bonds, tor_edge_index, tor_edge_attr, tor_edge_sh, tor_edge_weight = self.build_bond_conv_graph(data)
        tor_bond_vec = data['ligand'].pos[tor_bonds[1]] - data['ligand'].pos[tor_bonds[0]]
        tor_bond_attr = lig_node_attr[tor_bonds[0]] + lig_node_attr[tor_bonds[1]]

         # 计算扭转边的球谐特征
        tor_bonds_sh = o3.spherical_harmonics("2e", tor_bond_vec, normalize=True, normalization='component')
        tor_edge_sh = self.final_tp_tor(tor_edge_sh, tor_bonds_sh[tor_edge_index[0]])

        # 拼接扭转边的特征
        tor_edge_attr = torch.cat([tor_edge_attr, lig_node_attr[tor_edge_index[1], :self.ns],
                                   tor_bond_attr[tor_edge_index[0], :self.ns]], -1)
         # 执行扭转卷积操作
        tor_pred = self.tor_bond_conv(lig_node_attr, tor_edge_index, tor_edge_attr, tor_edge_sh,
                                  out_nodes=data['ligand'].edge_mask.sum(), reduce='mean', edge_weight=tor_edge_weight)
        tor_pred = self.tor_final_layer(tor_pred).squeeze(1)
        # 使用边的sigma进行扭转分数的规范化
        edge_sigma = tor_sigma[data['ligand'].batch][data['ligand', 'ligand'].edge_index[0]][data['ligand'].edge_mask]

        if self.scale_by_sigma:
            tor_pred = tor_pred * torch.sqrt(torch.tensor(torus.score_norm(edge_sigma.cpu().numpy())).float()
                                             .to(data['ligand'].x.device))
        return tr_pred, rot_pred, tor_pred, None

    def get_edge_weight(self, edge_vec, max_norm):# 计算边的权重，基于边的向量和最大范数
        if self.smooth_edges:# 如果开启了平滑边的功能，则根据边的范数归一化并使用cosine函数
            normalised_norm = torch.clip(edge_vec.norm(dim=-1) * np.pi / max_norm, max=np.pi)
            return 0.5 * (torch.cos(normalised_norm) + 1.0).unsqueeze(-1)
        return 1.0

    def build_lig_conv_graph(self, data):# 构建配体原子之间的图
        # build the graph between ligand atoms
        if self.separate_noise_schedule:# 如果开启了分离噪声调度，则为每种噪声类型生成sigma嵌入
            data['ligand'].node_sigma_emb = torch.cat(
                [self.timestep_emb_func(data['ligand'].node_t[noise_type]) for noise_type in ['tr', 'rot', 'tor']],
                dim=1)
        elif self.asyncronous_noise_schedule:# 如果开启了异步噪声调度，则使用统一的噪声类型
            data['ligand'].node_sigma_emb = self.timestep_emb_func(data['ligand'].node_t['t'])
        else:# 默认使用平移噪声类型
            data['ligand'].node_sigma_emb = self.timestep_emb_func(
                data['ligand'].node_t['tr'])  # tr rot and tor noise is all the same

        if self.parallel == 1:# 如果并行数为1，构建配体原子之间的半径图
            radius_edges = radius_graph(data['ligand'].pos, self.lig_max_radius, data['ligand'].batch)
        else: # 如果并行数不为1，生成配体图的并行批次
            batches = torch.zeros(data.num_graphs, device=data['ligand'].x.device).long()
            batches = batches.index_add(0, data['ligand'].batch, torch.ones(len(data['ligand'].batch), device=data['ligand'].x.device).long())
            outer_batches = data.num_graphs
            b = [torch.ones(batches[i].item()//self.parallel, device=data['ligand'].x.device).long() * (self.parallel * i + j)
                 for i in range(outer_batches) for j in range(self.parallel)]
            data['ligand'].batch_parallel = torch.cat(b)
            radius_edges = radius_graph(data['ligand'].pos, self.lig_max_radius, data['ligand'].batch_parallel)
        # 合并配体的边
        edge_index = torch.cat([data['ligand', 'ligand'].edge_index, radius_edges], 1).long()
        edge_attr = torch.cat([
            data['ligand', 'ligand'].edge_attr,
            torch.zeros(radius_edges.shape[-1], self.in_lig_edge_features, device=data['ligand'].x.device)
        ], 0)

        # 为边加入sigma嵌入
        edge_sigma_emb = data['ligand'].node_sigma_emb[edge_index[0].long()]
        edge_attr = torch.cat([edge_attr, edge_sigma_emb], 1)
         # 构建节点属性
        node_attr = torch.cat([data['ligand'].x, data['ligand'].node_sigma_emb], 1)

        # 计算边向量
        src, dst = edge_index
        edge_vec = data['ligand'].pos[dst.long()] - data['ligand'].pos[src.long()]
        # 计算边的长度嵌入
        edge_length_emb = self.lig_distance_expansion(edge_vec.norm(dim=-1))

         # 合并边的属性
        edge_attr = torch.cat([edge_attr, edge_length_emb], 1)
        # 计算边的球谐特征
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')
        # 获取边的权重
        edge_weight = self.get_edge_weight(edge_vec, self.lig_max_radius)

        return node_attr, edge_index, edge_attr, edge_sh, edge_weight

    def build_rec_conv_graph(self, data):# 构建受体残基之间的图
        # build the graph between receptor residues
        node_attr = data['receptor'].x

        # this assumes the edges were already created in preprocessing since protein's structure is fixed
        # 假设在预处理过程中已经创建了受体的边
        edge_index = data['receptor', 'receptor'].edge_index
        src, dst = edge_index
        edge_vec = data['receptor'].pos[dst.long()] - data['receptor'].pos[src.long()]

        # 计算边的属性和球谐特征
        edge_attr = self.rec_distance_expansion(edge_vec.norm(dim=-1))
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')
        # 获取边的权重
        edge_weight = self.get_edge_weight(edge_vec, self.rec_max_radius)

        return node_attr, edge_attr, edge_sh, edge_weight

    def build_atom_conv_graph(self, data):# 构建受体原子之间的图
        # build the graph between receptor atoms
        node_attr = data['atom'].x

        # this assumes the edges were already created in preprocessing since protein's structure is fixed
        edge_index = data['atom', 'atom'].edge_index # 假设在预处理过程中已经创建了原子之间的边
        src, dst = edge_index
        edge_vec = data['atom'].pos[dst.long()] - data['atom'].pos[src.long()]

        # 计算边的属性和球谐特征
        edge_attr = self.lig_distance_expansion(edge_vec.norm(dim=-1))
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')
        # 获取边的权重
        edge_weight = self.get_edge_weight(edge_vec, self.lig_max_radius)

        return node_attr, edge_attr, edge_sh, edge_weight

    def build_cross_lig_conv_graph(self, data, lr_cross_distance_cutoff):# 构建配体与受体之间的交叉边图
        # build the cross edges between ligand atoms and receptor residues + atoms

        # LIGAND to RECEPTOR
        # 配体到受体
        if torch.is_tensor(lr_cross_distance_cutoff):# 如果使用每个图独立的截断距离
            # different cutoff for every graph
            lr_edge_index = radius(data['receptor'].pos / lr_cross_distance_cutoff[data['receptor'].batch],
                                data['ligand'].pos / lr_cross_distance_cutoff[data['ligand'].batch], 1,
                                data['receptor'].batch, data['ligand'].batch, max_num_neighbors=10000)
        else: # 使用统一的截断距离
            lr_edge_index = radius(data['receptor'].pos, data['ligand'].pos, lr_cross_distance_cutoff,
                            data['receptor'].batch, data['ligand'].batch, max_num_neighbors=10000)

        # 计算交叉边的向量和长度嵌入
        lr_edge_vec = data['receptor'].pos[lr_edge_index[1].long()] - data['ligand'].pos[lr_edge_index[0].long()]
        lr_edge_length_emb = self.cross_distance_expansion(lr_edge_vec.norm(dim=-1))
        # 获取配体sigma嵌入
        lr_edge_sigma_emb = data['ligand'].node_sigma_emb[lr_edge_index[0].long()]
        # 合并属性
        lr_edge_attr = torch.cat([lr_edge_sigma_emb, lr_edge_length_emb], 1)
        lr_edge_sh = o3.spherical_harmonics(self.sh_irreps, lr_edge_vec, normalize=True, normalization='component')

        # 获取交叉边的权重
        cutoff_d = lr_cross_distance_cutoff[data['ligand'].batch[lr_edge_index[0]]].squeeze() \
            if torch.is_tensor(lr_cross_distance_cutoff) else lr_cross_distance_cutoff
        lr_edge_weight = self.get_edge_weight(lr_edge_vec, cutoff_d)

        # LIGAND to ATOM
        # 配体到原子
        la_edge_index = radius(data['atom'].pos, data['ligand'].pos, self.lig_max_radius,
                               data['atom'].batch, data['ligand'].batch, max_num_neighbors=10000)

        la_edge_vec = data['atom'].pos[la_edge_index[1].long()] - data['ligand'].pos[la_edge_index[0].long()]
        la_edge_length_emb = self.lig_distance_expansion(la_edge_vec.norm(dim=-1))
        la_edge_sigma_emb = data['ligand'].node_sigma_emb[la_edge_index[0].long()]
        la_edge_attr = torch.cat([la_edge_sigma_emb, la_edge_length_emb], 1)
        la_edge_sh = o3.spherical_harmonics(self.sh_irreps, la_edge_vec, normalize=True, normalization='component')
        la_edge_weight = self.get_edge_weight(la_edge_vec, self.lig_max_radius)

        return lr_edge_index, lr_edge_attr, lr_edge_sh, lr_edge_weight, la_edge_index, la_edge_attr, \
               la_edge_sh, la_edge_weight

    def build_cross_rec_conv_graph(self, data):# 构建配体原子、受体残基和受体原子之间的交叉边图
        # build the cross edges between ligan atoms, receptor residues and receptor atoms

        # ATOM to RECEPTOR
        ar_edge_index = data['atom', 'receptor'].edge_index
        ar_edge_vec = data['receptor'].pos[ar_edge_index[1].long()] - data['atom'].pos[ar_edge_index[0].long()]
        ar_edge_attr = self.rec_distance_expansion(ar_edge_vec.norm(dim=-1))
        ar_edge_sh = o3.spherical_harmonics(self.sh_irreps, ar_edge_vec, normalize=True, normalization='component')
        ar_edge_weight = 1 # 受体原子边的权重为1

        return ar_edge_attr, ar_edge_sh, ar_edge_weight

    def build_center_conv_graph(self, data): # 构建用于配体原子和中心的卷积过滤器
        # build the filter for the convolution of the center with the ligand atoms
        # for translational and rotational score
        edge_index = torch.cat([data['ligand'].batch.unsqueeze(0), torch.arange(len(data['ligand'].batch)).to(data['ligand'].x.device).unsqueeze(0)], dim=0)

        center_pos, count = torch.zeros((data.num_graphs, 3)).to(data['ligand'].x.device), torch.zeros((data.num_graphs, 3)).to(data['ligand'].x.device)
        center_pos.index_add_(0, index=data['ligand'].batch, source=data['ligand'].pos)
        center_pos = center_pos / torch.bincount(data['ligand'].batch).unsqueeze(1)

         # 计算中心到配体原子的边向量
        edge_vec = data['ligand'].pos[edge_index[1]] - center_pos[edge_index[0]]
        edge_attr = self.center_distance_expansion(edge_vec.norm(dim=-1))
         # 计算中心到配体原子的边向量
        edge_sigma_emb = data['ligand'].node_sigma_emb[edge_index[1].long()]
        edge_attr = torch.cat([edge_attr, edge_sigma_emb], 1)
        # 计算边的球谐特征
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')
        return edge_index, edge_attr, edge_sh

    def build_bond_conv_graph(self, data): # 构建用于伪扭矩层的配体之间的键图
        # build graph for the pseudotorque layer
        bonds = data['ligand', 'ligand'].edge_index[:, data['ligand'].edge_mask].long()
        bond_pos = (data['ligand'].pos[bonds[0]] + data['ligand'].pos[bonds[1]]) / 2
        bond_batch = data['ligand'].batch[bonds[0]]
         # 构建半径图
        edge_index = radius(data['ligand'].pos, bond_pos, self.lig_max_radius, batch_x=data['ligand'].batch, batch_y=bond_batch)

        edge_vec = data['ligand'].pos[edge_index[1]] - bond_pos[edge_index[0]]
        edge_attr = self.lig_distance_expansion(edge_vec.norm(dim=-1))

        # 完成边的最终嵌入
        edge_attr = self.final_edge_embedding(edge_attr)
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')
        edge_weight = self.get_edge_weight(edge_vec, self.lig_max_radius)

        return bonds, edge_index, edge_attr, edge_sh, edge_weight

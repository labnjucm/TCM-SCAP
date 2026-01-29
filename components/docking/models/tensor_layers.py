import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import o3
from e3nn.nn import BatchNorm
from e3nn.o3 import TensorProduct, Linear
from torch_scatter import scatter, scatter_mean

from models.layers import FCBlock

def get_irrep_seq(ns, nv, use_second_order_repr, reduce_pseudoscalars):
    """
    根据给定的输入参数，返回表示对称表示（irreps）序列的列表。
    
    参数:
    ns: 代表基态对称表示的系数
    nv: 代表激发态对称表示的系数
    use_second_order_repr: 是否使用二阶表示
    reduce_pseudoscalars: 是否减少伪标量表示
    
    返回:
    irrep_seq: 对称表示序列的字符串列表
    """
    if use_second_order_repr:
        irrep_seq = [
            f'{ns}x0e',# 基态对称表示
            f'{ns}x0e + {nv}x1o + {nv}x2e',# 一阶激发态表示
            f'{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o',# 二阶激发态表示
            f'{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o + {nv if reduce_pseudoscalars else ns}x0o'# 包括伪标量
        ]
    else:
        irrep_seq = [
            f'{ns}x0e',
            f'{ns}x0e + {nv}x1o',
            f'{ns}x0e + {nv}x1o + {nv}x1e',
            f'{ns}x0e + {nv}x1o + {nv}x1e + {nv if reduce_pseudoscalars else ns}x0o'
        ]
    return irrep_seq


def irrep_to_size(irrep):
    """
    将表示对称表示（irreps）序列转换为其大小。
    
    参数:
    irrep: 表示对称表示的字符串，如 "1x0e + 2x1o"
    
    返回:
    size: 对应的大小
    """
    irreps = irrep.split(' + ')
    size = 0
    for ir in irreps:
        m, (l, p) = ir.split('x')# m 是系数，(l, p) 是表示类型
        size += int(m) * (2 * int(l) + 1)# 根据表示的维度计算大小
    return size


class FasterTensorProduct(torch.nn.Module):
    # Implemented by Bowen Jing
    """
    通过加速的张量积操作实现的模块，用于基于对称表示的计算。
    该实现适用于处理二维或三维球面谐波表示的输入和输出。
    """
    def __init__(self, in_irreps, sh_irreps, out_irreps, **kwargs):
        """
        初始化 FasterTensorProduct 模块
        
        参数:
        in_irreps: 输入的对称表示
        sh_irreps: 球面谐波的对称表示
        out_irreps: 输出的对称表示
        """
        super().__init__()
         # 确保 sh_irreps 是有效的球面谐波表示
        #for ir in in_irreps:
        #    m, (l, p) = ir
        #    assert l in [0, 1], "Higher order in irreps are not supported"
        #for ir in out_irreps:
        #    m, (l, p) = ir
        #    assert l in [0, 1], "Higher order out irreps are not supported"
        assert o3.Irreps(sh_irreps) == o3.Irreps('1x0e+1x1o'), "sh_irreps don't look like 1st order spherical harmonics"
        self.in_irreps = o3.Irreps(in_irreps)
        self.out_irreps = o3.Irreps(out_irreps)

        # 初始化表示的乘法（multiplicities）
        in_muls = {'0e': 0, '1o': 0, '1e': 0, '0o': 0}
        out_muls = {'0e': 0, '1o': 0, '1e': 0, '0o': 0}
        for (m, ir) in self.in_irreps: in_muls[str(ir)] = m
        for (m, ir) in self.out_irreps: out_muls[str(ir)] = m

         # 计算权重形状
        self.weight_shapes = {
            '0e': (in_muls['0e'] + in_muls['1o'], out_muls['0e']),
            '1o': (in_muls['0e'] + in_muls['1o'] + in_muls['1e'], out_muls['1o']),
            '1e': (in_muls['1o'] + in_muls['1e'] + in_muls['0o'], out_muls['1e']),
            '0o': (in_muls['1e'] + in_muls['0o'], out_muls['0o'])
        }
        self.weight_numel = sum(a * b for (a, b) in self.weight_shapes.values())

    def forward(self, in_, sh, weight):
        """
        前向传播函数
        
        参数:
        in_: 输入张量
        sh: 球面谐波张量
        weight: 权重张量
        
        返回:
        out: 输出张量
        """
         # 存储输入和输出张量的字典
        in_dict, out_dict = {}, {'0e': [], '1o': [], '1e': [], '0o': []}
        for (m, ir), sl in zip(self.in_irreps, self.in_irreps.slices()):
            in_dict[str(ir)] = in_[..., sl]# 填充输入张量
            if ir[0] == 1: in_dict[str(ir)] = in_dict[str(ir)].reshape(list(in_dict[str(ir)].shape)[:-1] + [-1, 3])# 处理1阶表示（如果有的话）
        sh_0e, sh_1o = sh[..., 0], sh[..., 1:]# 提取球面谐波的分量
         # 计算不同类型表示的输出
        if '0e' in in_dict:
            out_dict['0e'].append(in_dict['0e'] * sh_0e.unsqueeze(-1))
            out_dict['1o'].append(in_dict['0e'].unsqueeze(-1) * sh_1o.unsqueeze(-2))
        if '1o' in in_dict:
            out_dict['0e'].append((in_dict['1o'] * sh_1o.unsqueeze(-2)).sum(-1) / np.sqrt(3))
            out_dict['1o'].append(in_dict['1o'] * sh_0e.unsqueeze(-1).unsqueeze(-1))
            out_dict['1e'].append(torch.linalg.cross(in_dict['1o'], sh_1o.unsqueeze(-2), dim=-1) / np.sqrt(2))
        if '1e' in in_dict:
            out_dict['1o'].append(torch.linalg.cross(in_dict['1e'], sh_1o.unsqueeze(-2), dim=-1) / np.sqrt(2))
            out_dict['1e'].append(in_dict['1e'] * sh_0e.unsqueeze(-1).unsqueeze(-1))
            out_dict['0o'].append((in_dict['1e'] * sh_1o.unsqueeze(-2)).sum(-1) / np.sqrt(3))
        if '0o' in in_dict:
            out_dict['1e'].append(in_dict['0o'].unsqueeze(-1) * sh_1o.unsqueeze(-2))
            out_dict['0o'].append(in_dict['0o'] * sh_0e.unsqueeze(-1))

        # 使用权重进行张量乘法
        weight_dict = {}
        start = 0
        for key in self.weight_shapes:
            in_, out = self.weight_shapes[key]
            weight_dict[key] = weight[..., start:start + in_ * out].reshape(
                list(weight.shape)[:-1] + [in_, out]) / np.sqrt(in_)
            start += in_ * out

         # 处理不同类型的输出
        if out_dict['0e']:
            out_dict['0e'] = torch.cat(out_dict['0e'], dim=-1)
            out_dict['0e'] = torch.matmul(out_dict['0e'].unsqueeze(-2), weight_dict['0e']).squeeze(-2)

        if out_dict['1o']:
            out_dict['1o'] = torch.cat(out_dict['1o'], dim=-2)
            out_dict['1o'] = (out_dict['1o'].unsqueeze(-2) * weight_dict['1o'].unsqueeze(-1)).sum(-3)
            out_dict['1o'] = out_dict['1o'].reshape(list(out_dict['1o'].shape)[:-2] + [-1])

        if out_dict['1e']:
            out_dict['1e'] = torch.cat(out_dict['1e'], dim=-2)
            out_dict['1e'] = (out_dict['1e'].unsqueeze(-2) * weight_dict['1e'].unsqueeze(-1)).sum(-3)
            out_dict['1e'] = out_dict['1e'].reshape(list(out_dict['1e'].shape)[:-2] + [-1])

        if out_dict['0o']:
            out_dict['0o'] = torch.cat(out_dict['0o'], dim=-1)
            # out_dict['0o'] = (out_dict['0o'].unsqueeze(-1) * weight_dict['0o']).sum(-2)
            out_dict['0o'] = torch.matmul(out_dict['0o'].unsqueeze(-2), weight_dict['0o']).squeeze(-2)

        out = [] # 合并输出张量
        for _, ir in self.out_irreps:
            out.append(out_dict[str(ir)])
        return torch.cat(out, dim=-1)


class TensorProductConvLayer(torch.nn.Module):
    """
    这是一个实现张量积卷积层（Tensor Product Convolution Layer）的类，支持深度卷积和加速的张量积计算。
    """
    def __init__(self, in_irreps, sh_irreps, out_irreps, n_edge_features, residual=True, batch_norm=True, dropout=0.0,
                 hidden_features=None, faster=False, edge_groups=1, tp_weights_layers=2, activation='relu', depthwise=False):
        """
        初始化 TensorProductConvLayer 类
        
        参数：
        in_irreps: 输入对称表示（irreps）
        sh_irreps: 球面谐波的对称表示（irreps）
        out_irreps: 输出对称表示（irreps）
        n_edge_features: 边特征的数量
        residual: 是否使用残差连接（默认：True）
        batch_norm: 是否使用批量归一化（默认：True）
        dropout: dropout比率（默认：0.0）
        hidden_features: 隐藏层特征的数量（默认：None）
        faster: 是否使用加速的张量积计算（默认：False）
        edge_groups: 边特征分组数（默认：1）
        tp_weights_layers: 张量积权重层的数量（默认：2）
        activation: 激活函数类型（默认：'relu'）
        depthwise: 是否使用深度卷积（默认：False）
        """
        super(TensorProductConvLayer, self).__init__()
         # 初始化对称表示、残差连接等参数
        self.in_irreps = in_irreps
        self.out_irreps = out_irreps
        self.sh_irreps = sh_irreps
        self.residual = residual
        self.edge_groups = edge_groups
        self.out_size = irrep_to_size(out_irreps)
        self.depthwise = depthwise
        if hidden_features is None:
            hidden_features = n_edge_features # 默认隐藏层特征等于边特征数量

        # 如果使用深度卷积
        if depthwise:
            # 将输入、球面谐波和输出表示转换为o3.Irreps格式
            in_irreps = o3.Irreps(in_irreps)
            sh_irreps = o3.Irreps(sh_irreps)
            out_irreps = o3.Irreps(out_irreps)

            # 初始化中间表示的列表和指令列表
            irreps_mid = []
            instructions = []
             # 遍历输入和球面谐波表示进行张量积计算
            for i, (mul, ir_in) in enumerate(in_irreps):
                for j, (_, ir_edge) in enumerate(sh_irreps):
                    for ir_out in ir_in * ir_edge:
                        if ir_out in out_irreps:
                            k = len(irreps_mid)
                            irreps_mid.append((mul, ir_out))
                            instructions.append((i, j, k, "uvu", True))

            # We sort the output irreps of the tensor product so that we can simplify them
            # when they are provided to the second o3.Linear
             # 对输出的表示进行排序
            irreps_mid = o3.Irreps(irreps_mid)
            irreps_mid, p, _ = irreps_mid.sort()

            # Permute the output indexes of the instructions to match the sorted irreps:
             # 调整指令的输出索引，使其与排序后的表示匹配
            instructions = [
                (i_in1, i_in2, p[i_out], mode, train)
                for i_in1, i_in2, i_out, mode, train in instructions
            ]

             # 初始化张量积（Tensor Product）模块
            self.tp = TensorProduct(
                in_irreps,
                sh_irreps,
                irreps_mid,
                instructions,
                shared_weights=False,
                internal_weights=False,
            )

            # 初始化线性层
            self.linear_2 = Linear(
                # irreps_mid has uncoallesed irreps because of the uvu instructions,
                # but there's no reason to treat them seperately for the Linear
                # Note that normalization of o3.Linear changes if irreps are coallesed
                # (likely for the better)
                irreps_in=irreps_mid.simplify(),# 对简化后的中间表示进行线性变换
                irreps_out=out_irreps,
                internal_weights=True,
                shared_weights=True,
            )

        else:# 如果不使用深度卷积，根据是否加速选择不同的张量积实现
            if faster:
                print("Faster Tensor Product")
                self.tp = FasterTensorProduct(in_irreps, sh_irreps, out_irreps)
            else:
                self.tp = o3.FullyConnectedTensorProduct(in_irreps, sh_irreps, out_irreps, shared_weights=False)

        if edge_groups == 1:# 初始化全连接层（FCBlock），用于处理边特征
            self.fc = FCBlock(n_edge_features, hidden_features, self.tp.weight_numel, tp_weights_layers, dropout, activation)
        else:# 如果有多个边特征组，为每个组初始化一个FCBlock
            self.fc = [FCBlock(n_edge_features, hidden_features, self.tp.weight_numel, tp_weights_layers, dropout, activation) for _ in range(edge_groups)]
            self.fc = nn.ModuleList(self.fc)

        self.batch_norm = BatchNorm(out_irreps) if batch_norm else None# 如果需要，初始化批量归一化层

    def forward(self, node_attr, edge_index, edge_attr, edge_sh, out_nodes=None, reduce='mean', edge_weight=1.0):

        """
        前向传播函数
        
        参数：
        node_attr: 节点特征
        edge_index: 边索引
        edge_attr: 边特征
        edge_sh: 边的球面谐波特征
        out_nodes: 输出节点数量（默认：None）
        reduce: 汇聚操作（默认：'mean'）
        edge_weight: 边的权重（默认：1.0）
        
        返回：
        out: 输出特征
        """
        if edge_index.shape[1] == 0:# 如果没有边索引，直接返回零张量
            out = torch.zeros((node_attr.shape[0], self.out_size), dtype=node_attr.dtype, device=node_attr.device)
        else: # 获取边的源节点和目标节点索引
            edge_src, edge_dst = edge_index
             # 处理边特征，如果有多个边特征组，则为每个组分别计算特征
            edge_attr_ = self.fc(edge_attr) if self.edge_groups == 1 else torch.cat(
                [self.fc[i](edge_attr[i]) for i in range(self.edge_groups)], dim=0).to(node_attr.device)
            # 计算张量积
            tp = self.tp(node_attr[edge_dst], edge_sh, edge_attr_ * edge_weight)

            # 默认使用“mean”进行汇聚
            out_nodes = out_nodes or node_attr.shape[0]
            out = scatter(tp, edge_src, dim=0, dim_size=out_nodes, reduce=reduce)

            # 如果使用深度卷积，应用线性层
            if self.depthwise:
                out = self.linear_2(out)

            # 如果使用批量归一化，进行批量归一化
            if self.batch_norm:
                out = self.batch_norm(out)

        # 如果使用残差连接，将输入特征与输出特征相加
        if self.residual:
            padded = F.pad(node_attr, (0, out.shape[-1] - node_attr.shape[-1]))
            out = out + padded
        return out


class OldTensorProductConvLayer(torch.nn.Module):
    """
    这是旧版的张量积卷积层实现类。
    """
    def __init__(self, in_irreps, sh_irreps, out_irreps, n_edge_features, residual=True, batch_norm=True, dropout=0.0,
                 hidden_features=None):
        """
        初始化 OldTensorProductConvLayer 类
        
        参数：
        in_irreps: 输入对称表示（irreps）
        sh_irreps: 球面谐波的对称表示（irreps）
        out_irreps: 输出对称表示（irreps）
        n_edge_features: 边特征的数量
        residual: 是否使用残差连接（默认：True）
        batch_norm: 是否使用批量归一化（默认：True）
        dropout: dropout比率（默认：0.0）
        hidden_features: 隐藏层特征的数量（默认：None）
        """
        super(OldTensorProductConvLayer, self).__init__()
        self.in_irreps = in_irreps
        self.out_irreps = out_irreps
        self.sh_irreps = sh_irreps
        self.residual = residual
        if hidden_features is None:
            hidden_features = n_edge_features # 默认隐藏层特征等于边特征数量

         # 初始化张量积模块
        self.tp = tp = o3.FullyConnectedTensorProduct(in_irreps, sh_irreps, out_irreps, shared_weights=False)

         # 初始化全连接层
        self.fc = nn.Sequential(
            nn.Linear(n_edge_features, hidden_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, tp.weight_numel)# 输出张量积权重的数量
        )
        self.batch_norm = BatchNorm(out_irreps) if batch_norm else None

    def forward(self, node_attr, edge_index, edge_attr, edge_sh, out_nodes=None, reduce='mean', edge_weight=1.0):

        """
        前向传播函数
        
        参数：
        node_attr: 节点特征
        edge_index: 边索引
        edge_attr: 边特征
        edge_sh: 边的球面谐波特征
        out_nodes: 输出节点数量（默认：None）
        reduce: 汇聚操作（默认：'mean'）
        edge_weight: 边的权重（默认：1.0）
        
        返回：
        out: 输出特征
        """
        # 获取边的源节点和目标节点索引
        edge_src, edge_dst = edge_index
         # 计算张量积
        tp = self.tp(node_attr[edge_dst], edge_sh, self.fc(edge_attr) * edge_weight)

         # 汇聚操作
        out_nodes = out_nodes or node_attr.shape[0]
        out = scatter(tp, edge_src, dim=0, dim_size=out_nodes, reduce=reduce)

        if self.residual:# 如果使用残差连接，将输入特征与输出特征相加
            padded = F.pad(node_attr, (0, out.shape[-1] - node_attr.shape[-1]))# 对齐维度
            out = out + padded # 残差连接

        # 如果使用批量归一化，进行批量归一化
        if self.batch_norm:
            out = self.batch_norm(out)
        return out

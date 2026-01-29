import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from scipy.stats import beta

from utils.geometry import axis_angle_to_matrix, rigid_transform_Kabsch_3D_torch, rigid_transform_Kabsch_3D_torch_batch
from utils.torsion import modify_conformer_torsion_angles, modify_conformer_torsion_angles_batch


#实现Sigmoid函数，输入为实数 t，返回值为 1 / (1 + exp(-t))，常用于将实数映射到 (0, 1) 区间。
def sigmoid(t):
    return 1 / (1 + np.e**(-t))

#这是一个Sigmoid调度函数，通过调整参数 k 和 m 来平滑控制 t（通常用于学习率调整或参数调度）。函数返回一个归一化的Sigmoid值，使其范围为 [0, 1]。
def sigmoid_schedule(t, k=10, m=0.5):
    s = lambda t: sigmoid(k*(t-m))
    return (s(t)-s(0))/(s(1)-s(0))

#根据给定的时间 t，类型 schedule_type，最小值 sigma_min 和最大值 sigma_max，计算一个平滑的参数。可以使用指数平滑或Sigmoid平滑控制，结果值会在 sigma_min 到 sigma_max 范围内变化。
def t_to_sigma_individual(t, schedule_type, sigma_min, sigma_max, schedule_k=10, schedule_m=0.4):
    if schedule_type == "exponential":
        return sigma_min ** (1 - t) * sigma_max ** t
    elif schedule_type == 'sigmoid':
        return sigmoid_schedule(t, k=schedule_k, m=schedule_m) * (sigma_max - sigma_min) + sigma_min

#根据传入的 t_tr、t_rot 和 t_tor（分别代表平移、旋转和扭转的时间或进度），根据相应的最小和最大值计算出对应的平移、旋转和扭转的平滑因子。
def t_to_sigma(t_tr, t_rot, t_tor, args):
    tr_sigma = args.tr_sigma_min ** (1-t_tr) * args.tr_sigma_max ** t_tr
    rot_sigma = args.rot_sigma_min ** (1-t_rot) * args.rot_sigma_max ** t_rot
    tor_sigma = args.tor_sigma_min ** (1-t_tor) * args.tor_sigma_max ** t_tor
    return tr_sigma, rot_sigma, tor_sigma

#更新配体的位置。首先根据平移 tr_update 和旋转 rot_update 更新配体的位置。如果有扭转更新 torsion_updates，则根据扭转角度进一步调整配体的灵活部分。若提供了 pivot，则使用Kabsch算法进行更精确的对齐。
def modify_conformer(data, tr_update, rot_update, torsion_updates, pivot=None):
    lig_center = torch.mean(data['ligand'].pos, dim=0, keepdim=True)# 计算配体的中心位置
    rot_mat = axis_angle_to_matrix(rot_update.squeeze())# 将旋转角度转换为旋转矩阵
    rigid_new_pos = (data['ligand'].pos - lig_center) @ rot_mat.T + tr_update + lig_center# 计算旋转和平移后的新位置

    if torsion_updates is not None:
        # 如果有扭转更新，应用扭转角度更新
        flexible_new_pos = modify_conformer_torsion_angles(rigid_new_pos,
                                                           data['ligand', 'ligand'].edge_index.T[data['ligand'].edge_mask],
                                                           data['ligand'].mask_rotate if isinstance(data['ligand'].mask_rotate, np.ndarray) else data['ligand'].mask_rotate[0],
                                                           torsion_updates).to(rigid_new_pos.device)
        if pivot is None:
            # 如果没有pivot点，使用Kabsch算法对灵活配体进行刚性对齐
            R, t = rigid_transform_Kabsch_3D_torch(flexible_new_pos.T, rigid_new_pos.T)
            aligned_flexible_pos = flexible_new_pos @ R.T + t.T
        else:
             # 使用pivot点进行配体对齐
            R1, t1 = rigid_transform_Kabsch_3D_torch(pivot.T, rigid_new_pos.T)
            R2, t2 = rigid_transform_Kabsch_3D_torch(flexible_new_pos.T, pivot.T)

            aligned_flexible_pos = (flexible_new_pos @ R2.T + t2.T) @ R1.T + t1.T

        data['ligand'].pos = aligned_flexible_pos # 更新配体的位置
    else:
        data['ligand'].pos = rigid_new_pos# 如果没有扭转更新，仅更新旋转和平移
    return data# 返回更新后的数据

#该函数与 modify_conformer 类似，但此处支持批量处理。函数接受一批数据，并根据每个配体的更新进行旋转、平移和扭转更新。最后，返回更新后的配体位置。
def modify_conformer_batch(orig_pos, data, tr_update, rot_update, torsion_updates, mask_rotate):
    B = data.num_graphs# 获取批次大小
    N, M, R = data['ligand'].num_nodes // B, data['ligand', 'ligand'].num_edges // B, data['ligand'].edge_mask.sum().item() // B# 获取节点、边和旋转掩码的数量

    pos, edge_index, edge_mask = orig_pos.reshape(B, N, 3) + 0, data['ligand', 'ligand'].edge_index[:, :M], data['ligand'].edge_mask[:M]
    torsion_updates = torsion_updates.reshape(B, -1) if torsion_updates is not None else None# 将扭转更新重塑为批次形式

    lig_center = torch.mean(pos, dim=1, keepdim=True)# 计算配体的中心位置
    rot_mat = axis_angle_to_matrix(rot_update) # 获取旋转矩阵
    rigid_new_pos = torch.bmm(pos - lig_center, rot_mat.permute(0, 2, 1)) + tr_update.unsqueeze(1) + lig_center# 更新旋转和平移后的配体位置

    if torsion_updates is not None:
        # 如果有扭转更新，应用扭转角度更新
        flexible_new_pos = modify_conformer_torsion_angles_batch(rigid_new_pos, edge_index.T[edge_mask], mask_rotate, torsion_updates)
        R, t = rigid_transform_Kabsch_3D_torch_batch(flexible_new_pos, rigid_new_pos)# 使用Kabsch算法对齐
        aligned_flexible_pos = torch.bmm(flexible_new_pos, R.transpose(1, 2)) + t.transpose(1, 2)
        final_pos = aligned_flexible_pos.reshape(-1, 3)# 最终位置
    else:
        final_pos = rigid_new_pos.reshape(-1, 3) # 如果没有扭转更新，仅返回旋转和平移后的配体位置
    return final_pos# 返回最终更新的配体位置

#更新配体的位置。首先根据平移 tr_update 和旋转 rot_update 更新配体的位置。如果有扭转更新 torsion_updates，则根据扭转角度进一步调整配体的灵活部分。如果有扭转更新，还会使用Kabsch算法进行刚性对齐。此函数没有像 modify_conformer 那样传递图形数据，因此避免了创建新的异构图。
def modify_conformer_coordinates(pos, tr_update, rot_update, torsion_updates, edge_mask, mask_rotate, edge_index):
    # Made this function which does the same as modify_conformer because passing a graph would require
    # creating a new heterograph for reach graph when unbatching a batch of graphs
    lig_center = torch.mean(pos, dim=0, keepdim=True)# 计算配体的中心位置
    rot_mat = axis_angle_to_matrix(rot_update.squeeze())# 将旋转角度转换为旋转矩阵
    rigid_new_pos = (pos - lig_center) @ rot_mat.T + tr_update + lig_center# 计算旋转和平移后的新位置

    if torsion_updates is not None:
        # 如果有扭转更新，应用扭转角度更新
        flexible_new_pos = modify_conformer_torsion_angles(rigid_new_pos,edge_index.T[edge_mask],mask_rotate \
            if isinstance(mask_rotate, np.ndarray) else mask_rotate[0], torsion_updates).to(rigid_new_pos.device)

        R, t = rigid_transform_Kabsch_3D_torch(flexible_new_pos.T, rigid_new_pos.T)# 使用Kabsch算法进行配体对齐
        aligned_flexible_pos = flexible_new_pos @ R.T + t.T# 应用刚性变换进行对齐
        return aligned_flexible_pos# 返回调整后的配体位置
    else:
        return rigid_new_pos # 如果没有扭转更新，直接返回旋转和平移后的位置

#生成一个正弦嵌入（Sinusoidal embedding），用于将时间步长 timesteps 映射到一个高维空间。通过正弦和余弦函数生成周期性特征，通常用于扩展时间或位置嵌入，以便模型能够捕捉时序模式。
def sinusoidal_embedding(timesteps, embedding_dim, max_positions=10000):
    """ from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py   """
    assert len(timesteps.shape) == 1# 确保timesteps是一维的
    half_dim = embedding_dim // 2# 计算embedding维度的一半
    emb = math.log(max_positions) / (half_dim - 1)# 计算指数衰减的步长
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)# 计算emb的指数衰减因子
    emb = timesteps.float()[:, None] * emb[None, :]# 根据时间步长与衰减因子计算embedding
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)# 将正弦和余弦结果连接
    if embedding_dim % 2 == 1:  # zero pad# 如果embedding维度是奇数，进行零填充
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)# 确保embedding的形状是正确的
    return emb

#该类使用高斯傅里叶投影将噪声级别映射到一个高维空间。通过正弦和余弦变换进行映射，常用于扩展噪声级别的表示，使得模型能更好地捕捉噪声信息。
class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for noise levels.
    from https://github.com/yang-song/score_sde_pytorch/blob/1618ddea340f3e4a2ed7852a0694a809775cf8d0/models/layerspp.py#L32
    """

    def __init__(self, embedding_size=256, scale=1.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embedding_size//2) * scale, requires_grad=False)# 初始化一个固定的随机参数W

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi # 将输入x与W相乘，得到投影
        emb = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1) # 计算正弦和余弦嵌入
        return emb# 返回嵌入结果

#根据 embedding_type 返回相应的时间步长嵌入函数。如果 embedding_type 为 'sinusoidal'，则返回正弦嵌入函数；如果是 'fourier'，则返回傅里叶嵌入函数。
def get_timestep_embedding(embedding_type, embedding_dim, embedding_scale=10000):
    if embedding_type == 'sinusoidal':
        emb_func = (lambda x : sinusoidal_embedding(embedding_scale * x, embedding_dim))# 如果是正弦嵌入
    elif embedding_type == 'fourier':
        emb_func = GaussianFourierProjection(embedding_size=embedding_dim, scale=embedding_scale)# 如果是傅里叶嵌入
    else:
        raise NotImplemented # 如果嵌入类型不支持，抛出异常
    return emb_func# 返回相应的嵌入函数

#根据给定的 sigma_schedule 类型返回相应的时间步长调度。如果选择 'expbeta'，则使用Beta分布生成一个随时间变化的调度。返回值是一个根据 t_max、inf_sched_alpha 和 inf_sched_beta 设置的时间步调度。
def get_t_schedule(sigma_schedule, inference_steps, inf_sched_alpha=1, inf_sched_beta=1, t_max=1):
    if sigma_schedule == 'expbeta':
        lin_max = beta.cdf(t_max, a=inf_sched_alpha, b=inf_sched_beta)# 计算Beta分布的累积分布函数
        c = np.linspace(lin_max, 0, inference_steps + 1)[:-1]# 创建一个线性空间，从lin_max到0
        return beta.ppf(c, a=inf_sched_alpha, b=inf_sched_beta)# 返回Beta分布的分位数
    raise Exception()# 其他情况抛出异常

#此函数设置了与不同节点（如配体、受体、原子节点等）相关的时间变量 t_tr（平移时间）、t_rot（旋转时间）和 t_tor（扭转时间）。它为输入的图结构（complex_graphs）中的各个部分（包括配体、受体、原子、杂项原子等）分配这些时间变量。
def set_time(complex_graphs, t, t_tr, t_rot, t_tor, batchsize, all_atoms, device, include_miscellaneous_atoms=False):
    # 为配体的每个节点设置时间相关的参数（平移、旋转、扭转）
    complex_graphs['ligand'].node_t = {
        'tr': t_tr * torch.ones(complex_graphs['ligand'].num_nodes).to(device),# 配体节点的平移时间t_tr
        'rot': t_rot * torch.ones(complex_graphs['ligand'].num_nodes).to(device), # 配体节点的旋转时间t_rot
        'tor': t_tor * torch.ones(complex_graphs['ligand'].num_nodes).to(device)}# 配体节点的扭转时间t_tor
    # 为受体的每个节点设置时间相关的参数（平移、旋转、扭转）
    complex_graphs['receptor'].node_t = {
        'tr': t_tr * torch.ones(complex_graphs['receptor'].num_nodes).to(device),# 受体节点的平移时间t_tr
        'rot': t_rot * torch.ones(complex_graphs['receptor'].num_nodes).to(device),# 受体节点的旋转时间t_rot
        'tor': t_tor * torch.ones(complex_graphs['receptor'].num_nodes).to(device)}# 受体节点的扭转时间t_tor
    # 为整个复合物设置时间相关的参数（平移、旋转、扭转）
    complex_graphs.complex_t = {'tr': t_tr * torch.ones(batchsize).to(device),# 复合物的平移时间t_tr
                               'rot': t_rot * torch.ones(batchsize).to(device),# 复合物的旋转时间t_rot
                               'tor': t_tor * torch.ones(batchsize).to(device)}# 复合物的扭转时间t_tor
    # 如果设置了`all_atoms`为True，则为所有原子节点设置时间参数
    if all_atoms:
        complex_graphs['atom'].node_t = {
            'tr': t_tr * torch.ones(complex_graphs['atom'].num_nodes).to(device),# 原子节点的平移时间t_tr
            'rot': t_rot * torch.ones(complex_graphs['atom'].num_nodes).to(device),# 原子节点的旋转时间t_rot
            'tor': t_tor * torch.ones(complex_graphs['atom'].num_nodes).to(device)} # 原子节点的扭转时间t_tor

    # 如果设置了`include_miscellaneous_atoms`且`all_atoms`为False，则为杂项原子节点设置时间参数
    if include_miscellaneous_atoms and not all_atoms:
        complex_graphs['misc_atom'].node_t = {
            'tr': t_tr * torch.ones(complex_graphs['misc_atom'].num_nodes).to(device), # 杂项原子节点的平移时间t_tr
            'rot': t_rot * torch.ones(complex_graphs['misc_atom'].num_nodes).to(device),# 杂项原子节点的旋转时间t_rot
            'tor': t_tor * torch.ones(complex_graphs['misc_atom'].num_nodes).to(device)}# 杂项原子节点的扭转时间t_tor

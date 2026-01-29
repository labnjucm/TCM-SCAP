import networkx as nx
import numpy as np
import torch, copy
from scipy.spatial.transform import Rotation as R
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data

from utils.geometry import rigid_transform_Kabsch_independent_torch, axis_angle_to_matrix

"""
    Preprocessing and computation for torsional updates to conformers
"""

"""
将分子图转换为 NetworkX 图：

使用 to_networkx 将 PyG 图转换为 NetworkX 图 G,以便操作图结构。
检查哪些边允许扭转：

遍历边列表 edges,移除一条边后检查图的连通性。
如果图仍然连通，则该边两侧没有独立部分，无法扭转。
如果图不连通，则提取扭转部分的节点列表 to_rotate。
生成掩码：

mask_edges:标记哪些边可扭转。
mask_rotate:标记在每条可扭转边上,哪些节点会被影响。
返回掩码：

返回 mask_edges(边的扭转掩码)和 mask_rotate(节点的扭转掩码)。
"""
#提取分子图中可以进行扭转操作的部分，并生成与扭转相关的掩码。
def get_transformation_mask(pyg_data):#PyTorch Geometric 数据对象，包含分子的图表示信息
    G = to_networkx(pyg_data.to_homogeneous(), to_undirected=False)
    to_rotate = []
    edges = pyg_data['ligand', 'ligand'].edge_index.T.numpy()
    for i in range(0, edges.shape[0], 2):
        assert edges[i, 0] == edges[i+1, 1]

        G2 = G.to_undirected()
        G2.remove_edge(*edges[i])
        if not nx.is_connected(G2):
            l = list(sorted(nx.connected_components(G2), key=len)[0])
            if len(l) > 1:
                if edges[i, 0] in l:
                    to_rotate.append([])
                    to_rotate.append(l)
                else:
                    to_rotate.append(l)
                    to_rotate.append([])
                continue
        to_rotate.append([])
        to_rotate.append([])

    mask_edges = np.asarray([0 if len(l) == 0 else 1 for l in to_rotate], dtype=bool)
    mask_rotate = np.zeros((np.sum(mask_edges), len(G.nodes())), dtype=bool)
    idx = 0
    for i in range(min(edges.shape[0], len(G.edges()))):
        if mask_edges[i]:
            mask_rotate[idx][np.asarray(to_rotate[i], dtype=int)] = True
            idx += 1

    return mask_edges, mask_rotate


"""
准备数据：

深拷贝 pos,以防止修改原始数据。
如果 pos 不是 NumPy 数组，转换为 NumPy 格式。
确保 mask_rotate 是数组格式。
遍历边并应用扭转：

遍历 edge_index 中的边，提取每条边的两端原子 u 和 v。
如果对应的 torsion_updates 为 0,则跳过。
检查 mask_rotate,确保边的方向符合扭转规则。
计算旋转向量 rot_vec 并生成旋转矩阵 rot_mat。
对受到扭转影响的原子应用旋转矩阵更新坐标。
返回结果：

如果 as_numpy 为 False,返回 PyTorch 张量格式的坐标。
"""
#根据扭转角度更新分子的三维坐标。
def modify_conformer_torsion_angles(pos, edge_index, mask_rotate, torsion_updates, as_numpy=False):
    pos = copy.deepcopy(pos)
    if type(pos) != np.ndarray: pos = pos.cpu().numpy()
    
    if type(mask_rotate) == list: mask_rotate = mask_rotate[0]
        
    for idx_edge, e in enumerate(edge_index.cpu().numpy()):
        if torsion_updates[idx_edge] == 0:
            continue
        u, v = e[0], e[1]

        # check if need to reverse the edge, v should be connected to the part that gets rotated
        if mask_rotate[idx_edge, u] or (not mask_rotate[idx_edge, v]):
            print("mask rotate exception")
        #assert not mask_rotate[idx_edge, u]
        #assert mask_rotate[idx_edge, v]

        rot_vec = pos[u] - pos[v]  # convention: positive rotation if pointing inwards
        rot_vec = rot_vec * torsion_updates[idx_edge] / np.linalg.norm(rot_vec) # idx_edge!
        rot_mat = R.from_rotvec(rot_vec).as_matrix()

        # Apply rotation to affected nodes
        pos[mask_rotate[idx_edge]] = (pos[mask_rotate[idx_edge]] - pos[v]) @ rot_mat.T + pos[v]

    if not as_numpy: pos = torch.from_numpy(pos.astype(np.float32))
    return pos

#批量对分子的原子坐标进行键扭转（torsion angle）更新。
def modify_conformer_torsion_angles_batch(pos, edge_index, mask_rotate, torsion_updates):
    pos = pos + 0# Create a copy of the positions tensor.
    for idx_edge, e in enumerate(edge_index):
        u, v = e[0], e[1]

        # check if need to reverse the edge, v should be connected to the part that gets rotated
        assert not mask_rotate[idx_edge, u]# Ensure that the rotation is applied to the correct part of the molecule.
        assert mask_rotate[idx_edge, v]

        # Calculate the rotation vector (torsional axis) based on the bond
        rot_vec = pos[:, u] - pos[:, v]  # convention: positive rotation if pointing inwards
        # Generate the rotation matrix for the torsional update.
        rot_mat = axis_angle_to_matrix(
            rot_vec / torch.linalg.norm(rot_vec, dim=-1, keepdims=True) * torsion_updates[:, idx_edge:idx_edge + 1])

        # Apply the rotation to the atoms affected by the torsion.
        pos[:, mask_rotate[idx_edge]] = torch.bmm(pos[:, mask_rotate[idx_edge]] - pos[:, v:v + 1], torch.transpose(rot_mat, 1, 2)) + pos[:, v:v + 1]

    return pos


#在批量模式下对分子进行扭转扰动，同时支持返回更新信息和分割输出选项。
def perturb_batch(data, torsion_updates, split=False, return_updates=False):
    if type(data) is Data:
        return modify_conformer_torsion_angles(data.pos,
                                               data.edge_index.T[data.edge_mask],
                                               data.mask_rotate, torsion_updates)
    pos_new = [] if split else copy.deepcopy(data.pos)
    edges_of_interest = data.edge_index.T[data.edge_mask]
    idx_node = 0
    idx_edges = 0
    torsion_update_list = []
    for i, mask_rotate in enumerate(data.mask_rotate):
        pos = data.pos[idx_node:idx_node + mask_rotate.shape[1]]
        edges = edges_of_interest[idx_edges:idx_edges + mask_rotate.shape[0]] - idx_node
        torsion_update = torsion_updates[idx_edges:idx_edges + mask_rotate.shape[0]]
        torsion_update_list.append(torsion_update)
        pos_new_ = modify_conformer_torsion_angles(pos, edges, mask_rotate, torsion_update)
        if split:
            pos_new.append(pos_new_)
        else:
            pos_new[idx_node:idx_node + mask_rotate.shape[1]] = pos_new_

        idx_node += mask_rotate.shape[1]
        idx_edges += mask_rotate.shape[0]
    if return_updates:
        return pos_new, torsion_update_list
    return pos_new

#从分子图中提取二面角（dihedral angles）的定义。
def get_dihedrals(data_list):
    edge_index, edge_mask = data_list[0]['ligand', 'ligand'].edge_index, data_list[0]['ligand'].edge_mask
    edge_list = [[] for _ in range(torch.max(edge_index) + 1)]

    for p in edge_index.T:
        edge_list[p[0]].append(p[1])

    rot_bonds = [(p[0], p[1]) for i, p in enumerate(edge_index.T) if edge_mask[i]]

    dihedral = []
    for a, b in rot_bonds:
        c = edge_list[a][0] if edge_list[a][0] != b else edge_list[a][1]
        d = edge_list[b][0] if edge_list[b][0] != a else edge_list[b][1]
        dihedral.append((c.item(), a.item(), b.item(), d.item()))
    # dihedral_numpy = np.asarray(dihedral)
    # print(dihedral_numpy.shape)
    dihedral = torch.tensor(dihedral)
    return dihedral

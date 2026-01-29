# From Nick Polizzi
import numpy as np
from collections import defaultdict
import prody as pr
import os

from datasets.constants import chi, atom_order, aa_long2short, aa_short2aa_idx, aa_idx2aa_short

#获取指定残基的二面角索引
def get_dihedral_indices(resname, chi_num):
    """Return the atom indices for the specified dihedral angle.
    """
    if resname not in chi:
        return np.array([np.nan]*4)
    if chi_num not in chi[resname]:
        return np.array([np.nan]*4)
    return np.array([atom_order[resname].index(x) for x in chi[resname][chi_num]])

#构建二面角索引字典
dihedral_indices = defaultdict(list)
for aa in atom_order.keys():
    for i in range(1, 5):
        inds = get_dihedral_indices(aa, i)
        dihedral_indices[aa].append(inds)
    dihedral_indices[aa] = np.array(dihedral_indices[aa])

#向量批处理操作
def vector_batch(a, b):
    return a - b

#对向量进行归一化
def unit_vector_batch(v):
    return v / np.linalg.norm(v, axis=1, keepdims=True)

#二面角批处理计算
def dihedral_angle_batch(p):
    b0 = vector_batch(p[:, 0], p[:, 1])# 计算向量 b0
    b1 = vector_batch(p[:, 1], p[:, 2])# 计算向量 b1
    b2 = vector_batch(p[:, 2], p[:, 3])# 计算向量 b2
    
    n1 = np.cross(b0, b1) # 计算平面法向量 n1
    n2 = np.cross(b1, b2)# 计算平面法向量 n2
    
    m1 = np.cross(n1, b1 / np.linalg.norm(b1, axis=1, keepdims=True)) # 计算辅助向量 m1
    
    x = np.sum(n1 * n2, axis=1)# 计算 n1 和 n2 的点积
    y = np.sum(m1 * n2, axis=1) # 计算 m1 和 n2 的点积
    
    deg = np.degrees(np.arctan2(y, x))# 计算角度（弧度转角度）

    deg[deg < 0] += 360# 确保角度为正

    return deg


def batch_compute_dihedral_angles(sidechains):
    sidechains_np = np.array(sidechains)
    dihedral_angles = dihedral_angle_batch(sidechains_np)
    return dihedral_angles

#从 ProDy 对象获取坐标
def get_coords(prody_pdb):
    resindices = sorted(set(prody_pdb.ca.getResindices()))# 获取残基索引
    coords = np.zeros((len(resindices), 14, 3)) # 初始化坐标数组
    for i, resind in enumerate(resindices):# 遍历残基
        sel = prody_pdb.select(f'resindex {resind}')# 选择当前残基
        resname = sel.getResnames()[0]
        for j, name in enumerate(atom_order[aa_long2short[resname] if resname in aa_long2short else 'X']):
            sel_resnum_name = sel.select(f'name {name}')
            if sel_resnum_name is not None:
                coords[i, j, :] = sel_resnum_name.getCoords()[0]
            else:
                coords[i, j, :] = [np.nan, np.nan, np.nan]
    return coords

#生成序列的 One-Hot 编码
def get_onehot_sequence(seq):
    onehot = np.zeros((len(seq), 20))
    for i, aa in enumerate(seq):
        idx = aa_short2aa_idx[aa] if aa in aa_short2aa_idx else 7 # 7 is the index for GLY
        onehot[i, idx] = 1
    return onehot


def get_dihedral_indices(onehot_sequence):
    return np.array([dihedral_indices[aa_idx2aa_short[aa_idx]] for aa_idx in np.where(onehot_sequence)[1]])


def _get_chi_angles(coords, indices):
    X = coords
    Y = indices.astype(int)
    N = coords.shape[0]
    mask = np.isnan(indices)
    Y[mask] = 0
    Z = X[np.arange(N)[:, None, None], Y, :]
    Z[mask] = np.nan
    chi_angles = batch_compute_dihedral_angles(Z.reshape(-1, 4, 3)).reshape(N, 4)
    return chi_angles

#获取 χ 角计算蛋白质的 χ 角。
def get_chi_angles(coords, seq, return_onehot=False):
    """

    Parameters
    ----------
    prody_pdb : prody.AtomGroup
        prody pdb object or selection
    return_coords : bool, optional
        return coordinates of prody_pdb in (N, 14, 3) array format, by default False
    return_onehot : bool, optional
        return one-hot sequence of prody_pdb, by default False

    Returns
    -------
    numpy array of shape (N, 4)
        Array contains chi angles of sidechains in row-order of residue indices in prody_pdb.
        If a chi angle is not defined for a residue, due to missing atoms or GLY / ALA, it is set to np.nan.
    """
    onehot = get_onehot_sequence(seq)# 获取序列的 One-Hot 编码
    dihedral_indices = get_dihedral_indices(onehot)# 获取二面角索引
    if return_onehot:
        return _get_chi_angles(coords, dihedral_indices), onehot
    return _get_chi_angles(coords, dihedral_indices)

#测试 χ 角计算
def test_get_chi_angles(print_chi_angles=False):
    # need internet connection of '6w70.pdb' in working directory
    pdb = pr.parsePDB('6w70')# 从 PDB 文件解析数据
    prody_pdb = pdb.select('chain A')# 选择链 A
    chi_angles = get_chi_angles(prody_pdb)# 计算 χ 角
    assert chi_angles.shape == (prody_pdb.ca.numAtoms(), 4)# 检查形状是否正确
    assert chi_angles[0,0] < 56.0 and chi_angles[0,0] > 55.0 # 验证角度范围
    print('test_get_chi_angles passed')
    try:
        os.remove('6w70.pdb.gz')
    except:
        pass
    if print_chi_angles:
        print(chi_angles)
    return True


if __name__ == '__main__':
    test_get_chi_angles(print_chi_angles=True)



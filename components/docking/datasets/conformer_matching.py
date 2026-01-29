import copy, time
import numpy as np
from collections import defaultdict
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdMolTransforms
from rdkit import Geometry
import networkx as nx
from scipy.optimize import differential_evolution

RDLogger.DisableLog('rdApp.*')#禁用 RDKit 日志

"""
    Conformer matching routines from Torsional Diffusion
"""
#获取和设置扭转角
def GetDihedral(conf, atom_idx):
    """
    获取分子中指定四个原子之间的扭转角（弧度）。

    参数:
    ----------
    conf: rdkit.Chem.Conformer
        分子的构象对象。
    atom_idx: list[int]
        指定的四个原子的索引。

    返回值:
    -------
    float
        四个原子之间的扭转角（弧度）。
    """
    return rdMolTransforms.GetDihedralRad(conf, atom_idx[0], atom_idx[1], atom_idx[2], atom_idx[3])


def SetDihedral(conf, atom_idx, new_vale):
    """
    设置分子中指定四个原子之间的扭转角（弧度）。

    参数:
    ----------
    conf: rdkit.Chem.Conformer
        分子的构象对象。
    atom_idx: list[int]
        指定的四个原子的索引。
    new_vale: float
        要设置的扭转角（弧度）。
    """
    rdMolTransforms.SetDihedralRad(conf, atom_idx[0], atom_idx[1], atom_idx[2], atom_idx[3], new_vale)

#应用扭转角变化
def apply_changes(mol, values, rotable_bonds, conf_id):
    """
    根据给定的扭转角值更新分子中的可旋转键。

    参数:
    ----------
    mol: rdkit.Chem.Mol
        要操作的分子。
    values: list[float]
        可旋转键的新扭转角值（弧度）。
    rotable_bonds: list[list[int]]
        分子中的可旋转键（四个原子索引的列表）。
    conf_id: int
        分子构象的 ID。

    返回值:
    -------
    rdkit.Chem.Mol
        更新后的分子对象。
    """
    opt_mol = copy.copy(mol)
    [SetDihedral(opt_mol.GetConformer(conf_id), rotable_bonds[r], values[r]) for r in range(len(rotable_bonds))]
    return opt_mol

#差分进化优化可旋转键
def optimize_rotatable_bonds(mol, true_mol, rotable_bonds, probe_id=-1, ref_id=-1, seed=0, popsize=15, maxiter=500,
                             mutation=(0.5, 1), recombination=0.8):
    """
    使用差分进化优化分子的可旋转键。

    参数:
    ----------
    mol: rdkit.Chem.Mol
        要优化的分子。
    true_mol: rdkit.Chem.Mol
        目标分子，用于对齐。
    rotable_bonds: list[list[int]]
        可旋转键的列表（每个键由四个原子索引定义）。
    probe_id: int
        分子构象的 ID。
    ref_id: int
        参考分子构象的 ID。
    seed: int
        随机种子。
    popsize: int
        差分进化的种群大小。
    maxiter: int
        差分进化的最大迭代次数。
    mutation: tuple[float, float]
        差分进化的变异参数。
    recombination: float
        差分进化的重组率。

    返回值:
    -------
    rdkit.Chem.Mol
        优化后的分子。
    """
    opt = OptimizeConformer(mol, true_mol, rotable_bonds, seed=seed, probe_id=probe_id, ref_id=ref_id)
    max_bound = [np.pi] * len(opt.rotable_bonds)
    min_bound = [-np.pi] * len(opt.rotable_bonds)
    bounds = (min_bound, max_bound)
    bounds = list(zip(bounds[0], bounds[1]))

    # Optimize conformations
    # 使用差分进化优化扭转角
    result = differential_evolution(opt.score_conformation, bounds,
                                    maxiter=maxiter, popsize=popsize,
                                    mutation=mutation, recombination=recombination, disp=False, seed=seed)
    opt_mol = apply_changes(opt.mol, result['x'], opt.rotable_bonds, conf_id=probe_id)

    return opt_mol

#优化类 (OptimizeConformer)
class OptimizeConformer:
    """
    一个优化分子构象的类，包含打分函数用于差分进化。

    参数:
    ----------
    mol: rdkit.Chem.Mol
        要优化的分子。
    true_mol: rdkit.Chem.Mol
        目标分子，用于对齐。
    rotable_bonds: list[list[int]]
        可旋转键列表。
    probe_id: int
        分子构象的 ID。
    ref_id: int
        参考分子构象的 ID。
    seed: int
        随机种子。
    """
    def __init__(self, mol, true_mol, rotable_bonds, probe_id=-1, ref_id=-1, seed=None):
        super(OptimizeConformer, self).__init__()
        if seed:
            np.random.seed(seed)
        self.rotable_bonds = rotable_bonds
        self.mol = mol
        self.true_mol = true_mol
        self.probe_id = probe_id
        self.ref_id = ref_id

    def score_conformation(self, values):
        """
        根据扭转角设置分子构象并返回与目标分子的对齐得分。

        参数:
        ----------
        values: list[float]
            可旋转键的新扭转角值。

        返回值:
        -------
        float
            与目标分子的对齐得分。
        """
        for i, r in enumerate(self.rotable_bonds):
            SetDihedral(self.mol.GetConformer(self.probe_id), r, values[i])
        return AllChem.AlignMol(self.mol, self.true_mol, self.probe_id, self.ref_id)

#获取分子中的扭转角
def get_torsion_angles(mol):
    """
    提取分子中的所有可旋转键及其相关的扭转角信息。

    参数:
    ----------
    mol: rdkit.Chem.Mol
        分子对象。

    返回值:
    -------
    list[tuple[int, int, int, int]]
        可旋转键的四个原子索引。
    """
    torsions_list = []
    G = nx.Graph()
    for i, atom in enumerate(mol.GetAtoms()):
        G.add_node(i)# 构建分子的图表示
    nodes = set(G.nodes())
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        G.add_edge(start, end)
     # 遍历图中的边以识别可旋转键
    for e in G.edges():
        G2 = copy.deepcopy(G)
        G2.remove_edge(*e)
        if nx.is_connected(G2): continue
        l = list(sorted(nx.connected_components(G2), key=len)[0])
        if len(l) < 2: continue
        n0 = list(G2.neighbors(e[0]))
        n1 = list(G2.neighbors(e[1]))
        torsions_list.append(
            (n0[0], e[0], e[1], n1[0]) # 添加扭转角的原子索引
        )
    return torsions_list


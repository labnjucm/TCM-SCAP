from spyrmsd import rmsd, molecule

#这个函数计算了两个分子在其原子坐标之间的 对称 RMSD（Root Mean Square Deviation），并且在一定条件下返回一个排列（permutation）。该函数的目的是比较两个分子的空间结构的相似性。
def get_symmetry_rmsd(mol, coords1, coords2, mol2=None, return_permutation=False):
    # 使用 time_limit 装饰器限制函数的最大执行时间
    with time_limit(10):# 限制时间为10秒
        # 将 RDKit 分子对象转换为自定义的 Molecule 对象
        mol = molecule.Molecule.from_rdkit(mol)
        # 如果提供了第二个分子，则将其也转换为 Molecule 对象，否则 mol2 默认为 None
        mol2 = molecule.Molecule.from_rdkit(mol2) if mol2 is not None else mol2
        # 获取分子原子序号（atomicnums）和邻接矩阵（adjacency_matrix）
        mol2_atomicnums = mol2.atomicnums if mol2 is not None else mol.atomicnums
        mol2_adjacency_matrix = mol2.adjacency_matrix if mol2 is not None else mol.adjacency_matrix
        # 使用 RMSD 库的 symmrmsd 函数计算 RMSD
        RMSD = rmsd.symmrmsd(
            coords1,# 第一个分子的原子坐标
            coords2,# 第二个分子的原子坐标
            mol.atomicnums,# 第一个分子的原子序号
            mol2_atomicnums,# 第二个分子的原子序号
            mol.adjacency_matrix,# 第一个分子的邻接矩阵
            mol2_adjacency_matrix,# 第二个分子的邻接矩阵
            return_permutation=return_permutation# 是否返回排列
        )
        return RMSD# 返回计算出的 RMSD


import signal
from contextlib import contextmanager


class TimeoutException(Exception): pass

#get_symmetry_rmsd 用于计算两个分子在其坐标上的对称 RMSD，能够衡量它们在空间中的相似性。它通过限制执行时间来避免长时间的计算阻塞，适合在需要进行大量计算时使用。
#time_limit 用于在给定的时间内执行代码块，确保代码不会因超时问题而被卡住或无响应，尤其在计算量较大时非常有用。
@contextmanager
def time_limit(seconds):
    # 定义超时信号的处理函数
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")# 超时后抛出 TimeoutException

    # 设置信号处理程序
    signal.signal(signal.SIGALRM, signal_handler)
    # 设置超时定时器
    signal.alarm(seconds)
    try:
        yield# 执行代码块
    finally:
        signal.alarm(0)# 在代码块执行完毕后取消定时器


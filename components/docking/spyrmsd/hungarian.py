import numpy as np
import scipy

from .due import Doi, due

#模块引入与方法引用
due.cite(
    Doi("10.1021/ci400534h"),
    path="spyrmsd.hungarian",
    description="Hungarian method",
)

#计算成本矩阵
def cost_mtx(A: np.ndarray, B: np.ndarray):
    """
    Compute the cost matrix for atom-atom assignment.

    Parameters
    ----------
    A: numpy.ndarray
        Atomic coordinates of molecule A
    B: numpy.ndarray
        Atomic coordinates of molecule B

    Returns
    -------
    np.ndarray
        Cost matrix of squared atomic distances between atoms of
        molecules A and B
    """
    """
    计算原子-原子分配问题的成本矩阵。

    参数
    ----------
    A: numpy.ndarray
        分子 A 的原子坐标
    B: numpy.ndarray
        分子 B 的原子坐标

    返回
    -------
    np.ndarray
        成本矩阵，表示分子 A 和 B 的原子之间的平方欧几里得距离
    """
    return scipy.spatial.distance.cdist(A, B, metric="sqeuclidean")

#求解最优分配,使用匈牙利算法（线性分配算法）为两个分子间的原子坐标找到最优分配
def optimal_assignment(A: np.ndarray, B: np.ndarray):
    """
    Solve the optimal assignment problems between atomic coordinates of
    molecules A and B.

    Parameters
    ----------
    A: numpy.ndarray
        Atomic coordinates of molecule A
    B: numpy.ndarray
        Atomic coordinates of molecule B

    Returns
    -------
    Tuple[float, nd.array, nd.array]
        Cost of the optimal assignment, together with the row and column
        indices of said assignment
    """

    """
    使用匈牙利算法求解分子 A 和 B 原子坐标之间的最优分配问题。

    参数
    ----------
    A: numpy.ndarray
        分子 A 的原子坐标
    B: numpy.ndarray
        分子 B 的原子坐标

    返回
    -------
    Tuple[float, nd.array, nd.array]
        最优分配的成本值，以及分配的行索引和列索引
    """
    C = cost_mtx(A, B)

    row_idx, col_idx = scipy.optimize.linear_sum_assignment(C)

    # Compute assignment cost
    cost = C[row_idx, col_idx].sum()

    return cost, row_idx, col_idx

#计算分子 A 和分子 B 之间的基于匈牙利方法的 RMSD
def hungarian_rmsd(
    A: np.ndarray, B: np.ndarray, apropsA: np.ndarray, apropsB: np.ndarray
) -> float:
    """
    Solve the optimal assignment problems between atomic coordinates of
    molecules A and B.

    Parameters
    ----------
    A: numpy.ndarray
        Atomic coordinates of molecule A
    B: numpy.ndarray
        Atomic coordinates of molecule B
    apropsA: numpy.ndarray
        Atomic properties of molecule A
    apropsB: numpy.ndarray
        Atomic properties of molecule B

    Returns
    -------
    float
        RMSD computed with the Hungarian method

    Notes
    -----
    The Hungarian algorithm is used to solve the linear assignment problem, which is
    a minimum weight matching of the molecular graphs (bipartite).

    The linear assignment problem is solved for every element separately. [1]_

    .. [1] W. J. Allen and R. C. Rizzo, *Implementation of the Hungarian Algorithm to
        Account for Ligand Symmetry and Similarity in Structure-Based Design*,
        J. Chem. Inf. Model. **54**, 518-529 (2014)
    """

    #检查输入有效性
    assert A.shape == B.shape
    assert apropsA.shape == apropsB.shape

    elements = set(apropsA)#获取元素集合

    #逐元素计算最优分配
    total_cost: float = 0.0
    for t in elements:
        apropsA_idx = apropsA == t
        apropsB_idx = apropsB == t

        assert apropsA_idx.shape == apropsB_idx.shape

        cost, row_idx, col_idx = optimal_assignment(
            A[apropsA_idx, :], B[apropsB_idx, :]
        )

        total_cost += cost

    N = A.shape[0]#计算 RMSD

    rmsd = np.sqrt(total_cost / N)

    return rmsd

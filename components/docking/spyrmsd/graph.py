#模块导入与依赖处理
try:
    from spyrmsd.graphs.gt import (
        cycle,
        graph_from_adjacency_matrix,
        lattice,
        match_graphs,
        num_edges,
        num_vertices,
        vertex_property,
    )

except ImportError:
    try:
        from spyrmsd.graphs.nx import (
            cycle,
            graph_from_adjacency_matrix,
            lattice,
            match_graphs,
            num_edges,
            num_vertices,
            vertex_property,
        )
    except ImportError:
        raise ImportError("graph_tool or NetworkX libraries not found.")

#公共接口定义
__all__ = [
    "graph_from_adjacency_matrix",
    "match_graphs",
    "vertex_property",
    "num_vertices",
    "num_edges",
    "lattice",
    "cycle",
    "adjacency_matrix_from_atomic_coordinates",
]

import numpy as np

from spyrmsd import constants

#邻接矩阵计算函数
def adjacency_matrix_from_atomic_coordinates(
    aprops: np.ndarray, coordinates: np.ndarray
) -> np.ndarray:
    """
    Compute adjacency matrix from atomic coordinates.

    Parameters
    ----------
    aprops: numpy.ndarray
        Atomic properties
    coordinates: numpy.ndarray
        Atomic coordinates

    Returns
    -------
    numpy.ndarray
        Adjacency matrix

    Notes
    -----

    This function is based on an automatic bond perception algorithm: two
    atoms are considered to be bonded when their distance is smaller than
    the sum of their covalent radii plus a tolerance value. [3]_

    .. warning::
        The automatic bond perceptron rule implemented in this functions
        is very simple and only depends on atomic coordinates. Use
        with care!

    .. [3] E. C. Meng and R. A. Lewis, *Determination of molecular topology and atomic
       hybridization states from heavy atom coordinates*, J. Comp. Chem. **12**, 891-898
       (1991).
    """

    #输入校验
    n = len(aprops)

    assert coordinates.shape == (n, 3)

    A = np.zeros((n, n))

    #遍历原子对并判断键连接
    for i in range(n):
        r_i = constants.anum_to_covalentradius[aprops[i]]

        for j in range(i + 1, n):
            r_j = constants.anum_to_covalentradius[aprops[j]]

            distance = np.sqrt(np.sum((coordinates[i] - coordinates[j]) ** 2))

            if distance < (r_i + r_j + constants.connectivity_tolerance):
                A[i, j] = A[j, i] = 1

    return A
"""
构建分子图：从原子坐标生成邻接矩阵，用于分子图建模。
分子分析：判断分子中原子间的键连接关系，用于后续图匹配或拓扑分析
"""
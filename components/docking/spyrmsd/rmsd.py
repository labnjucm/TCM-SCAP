from typing import Any, List, Optional, Tuple, Union

import numpy as np

from spyrmsd import graph, hungarian, molecule, qcp, utils

#计算两个分子结构的均方根偏差（RMSD）
def rmsd(
    coords1: np.ndarray,
    coords2: np.ndarray,
    atomicn1: np.ndarray,
    atomicn2: np.ndarray,
    center: bool = False,
    minimize: bool = False,
    atol: float = 1e-9,
) -> float:#函数签名和文档
    """
    Compute RMSD

    Parameters
    ----------
    coords1: np.ndarray
        Coordinate of molecule 1
    coords2: np.ndarray
        Coordinates of molecule 2
    atomicn1: np.ndarray
        Atomic numbers for molecule 1
    atomicn2: np.ndarray
        Atomic numbers for molecule 2
    center: bool
        Center molecules at origin
    minimize: bool
        Compute minimum RMSD (with QCP method)
    atol: float
        Absolute tolerance parameter for QCP method (see :func:`qcp_rmsd`)

    Returns
    -------
    float
        RMSD

    Notes
    -----
    When `minimize=True`, the QCP method is used. [1]_ The molecules are
    centred at the origin according to the center of geometry and superimposed
    in order to minimize the RMSD.

    .. [1] D. L. Theobald, *Rapid calculation of RMSDs using a quaternion-based
       characteristic polynomial*, Acta Crys. A **61**, 478-480 (2005).
    """
    """
    计算 RMSD。

    参数
    ----------
    coords1: np.ndarray
        分子 1 的坐标矩阵 (N, 3)。
    coords2: np.ndarray
        分子 2 的坐标矩阵 (N, 3)。
    atomicn1: np.ndarray
        分子 1 的原子编号。
    atomicn2: np.ndarray
        分子 2 的原子编号。
    center: bool
        是否将分子平移到几何中心。
    minimize: bool
        是否使用 QCP 方法计算最小 RMSD。
    atol: float
        QCP 方法的绝对容差。

    返回值
    -------
    float
        RMSD 值。

    注意
    -----
    当 `minimize=True` 时，使用 QCP 方法计算最小 RMSD，分子会对齐到几何中心以最小化 RMSD。
    """

    #验证输入数据的形状和一致性
    assert np.all(atomicn1 == atomicn2)
    assert coords1.shape == coords2.shape

    # Center coordinates if required
    #根据 center 和 minimize 参数选择是否对坐标进行平移
    c1 = utils.center(coords1) if center or minimize else coords1
    c2 = utils.center(coords2) if center or minimize else coords2

    #如果 minimize=True，使用 QCP 方法计算最小 RMSD
    if minimize:
        rmsd = qcp.qcp_rmsd(c1, c2, atol)
    else:#否则，直接使用标准公式计算 RMSD
        n = coords1.shape[0]

        rmsd = np.sqrt(np.sum((c1 - c2) ** 2) / n)

    return rmsd

#使用匈牙利算法计算最小 RMSD，用于分子对齐和比较
def hrmsd(
    coords1: np.ndarray,
    coords2: np.ndarray,
    atomicn1: np.ndarray,
    atomicn2: np.ndarray,
    center=False,
):
    """
    Compute minimum RMSD using the Hungarian method.

    Parameters
    ----------
    coords1: np.ndarray
        Coordinate of molecule 1
    coords2: np.ndarray
        Coordinates of molecule 2
    atomicn1: np.ndarray
        Atomic numbers for molecule 1
    atomicn2: np.ndarray
        Atomic numbers for molecule 2

    Returns
    -------
    float
        Minimum RMSD (after assignment)

    Notes
    -----
    The Hungarian algorithm is used to solve the linear assignment problem, which is
    a minimum weight matching of the molecular graphs (bipartite). [2]_

    The linear assignment problem is solved for every element separately.

    .. [2] W. J. Allen and R. C. Rizzo, *Implementation of the Hungarian Algorithm to
        Account for Ligand Symmetry and Similarity in Structure-Based Design*,
        J. Chem. Inf. Model. **54**, 518-529 (2014)
    """
    """
    使用匈牙利算法计算最小 RMSD。

    参数
    ----------
    coords1: np.ndarray
        分子 1 的坐标矩阵 (N, 3)。
    coords2: np.ndarray
        分子 2 的坐标矩阵 (N, 3)。
    atomicn1: np.ndarray
        分子 1 的原子编号。
    atomicn2: np.ndarray
        分子 2 的原子编号。
    center: bool
        是否将分子平移到几何中心。

    返回值
    -------
    float
        最小 RMSD 值。

    注意
    -----
    使用匈牙利算法解决线性分配问题，最小化分子之间的 RMSD。
    """

    assert atomicn1.shape == atomicn2.shape
    assert coords1.shape == coords2.shape

    # Center coordinates if required
    #根据 center 参数决定是否对坐标进行平移
    c1 = utils.center(coords1) if center else coords1
    c2 = utils.center(coords2) if center else coords2

    return hungarian.hungarian_rmsd(c1, c2, atomicn1, atomicn2)#使用匈牙利算法计算最小 RMSD

#使用图同构计算两个分子之间的 RMSD
def _rmsd_isomorphic_core(
    coords1: np.ndarray,
    coords2: np.ndarray,
    aprops1: np.ndarray,
    aprops2: np.ndarray,
    am1: np.ndarray,
    am2: np.ndarray,
    center: bool = False,
    minimize: bool = False,
    isomorphisms: Optional[List[Tuple[List[int], List[int]]]] = None,
    atol: float = 1e-9,
) -> Tuple[float, List[Tuple[List[int], List[int]]], Tuple[List[int], List[int]]]:
    """
    Compute RMSD using graph isomorphism.

    Parameters
    ----------
    coords1: np.ndarray
        Coordinate of molecule 1
    coords2: np.ndarray
        Coordinates of molecule 2
    aprops1: np.ndarray
        Atomic properties for molecule 1
    aprops2: np.ndarray
        Atomic properties for molecule 2
    am1: np.ndarray
        Adjacency matrix for molecule 1
    am2: np.ndarray
        Adjacency matrix for molecule 2
    center: bool
        Centering flag
    minimize: bool
        Compute minized RMSD
    isomorphisms: Optional[List[Dict[int,int]]]
        Previously computed graph isomorphism
    atol: float
        Absolute tolerance parameter for QCP (see :func:`qcp_rmsd`)

    Returns
    -------
    Tuple[float, List[Dict[int, int]]]
        RMSD (after graph matching) and graph isomorphisms
    """
    """
    使用图同构计算 RMSD。

    参数
    ----------
    coords1: np.ndarray
        分子 1 的坐标矩阵。
    coords2: np.ndarray
        分子 2 的坐标矩阵。
    aprops1: np.ndarray
        分子 1 的原子属性。
    aprops2: np.ndarray
        分子 2 的原子属性。
    am1: np.ndarray
        分子 1 的邻接矩阵。
    am2: np.ndarray
        分子 2 的邻接矩阵。
    center: bool
        是否将分子平移到几何中心。
    minimize: bool
        是否使用 QCP 方法计算最小 RMSD。
    isomorphisms: Optional[List[Tuple[List[int], List[int]]]]
        预计算的图同构结果。
    atol: float
        QCP 方法的绝对容差。

    返回值
    -------
    Tuple[float, List[Tuple[List[int], List[int]]], Tuple[List[int], List[int]]]
        - 计算的 RMSD 值。
        - 所有可能的图同构。
        - 最小 RMSD 的同构。
    """

     # 确保输入坐标的形状一致
    assert coords1.shape == coords2.shape

    n = coords1.shape[0]

    # Center coordinates if required
    #如果 center=True 或 minimize=True，将分子坐标平移到几何中心
    c1 = utils.center(coords1) if center or minimize else coords1
    c2 = utils.center(coords2) if center or minimize else coords2

    # No cached isomorphisms
    #如果未提供图同构，计算两个分子图的所有可能同构
    if isomorphisms is None:
        # Convert molecules to graphs
        G1 = graph.graph_from_adjacency_matrix(am1, aprops1)
        G2 = graph.graph_from_adjacency_matrix(am2, aprops2)

        # Get all the possible graph isomorphisms
        isomorphisms = graph.match_graphs(G1, G2)

    # Minimum result
    # Squared displacement (not minimize) or RMSD (minimize)
    min_result = np.inf
    min_isomorphisms = None

    # Loop over all graph isomorphisms to find the lowest RMSD
    #遍历所有图同构，计算 RMSD 或平方偏差，保留最小值及对应同构
    for idx1, idx2 in isomorphisms:
        # Use the isomorphism to shuffle coordinates around (from original order)
        c1i = c1[idx1, :]
        c2i = c2[idx2, :]

        if not minimize:
            # Compute square displacement
            # Avoid dividing by n and an expensive sqrt() operation
            result = np.sum((c1i - c2i) ** 2)
        else:
            # Compute minimized RMSD using QCP
            result = qcp.qcp_rmsd(c1i, c2i, atol)

        if result < min_result:
            min_result = result
            min_isomorphisms = (idx1, idx2)

    #如果未使用最小化，计算 RMSD 值
    if not minimize:
        # Compute actual RMSD from square displacement
        min_result = np.sqrt(min_result / n)

    # Return the actual RMSD
    return min_result, isomorphisms, min_isomorphisms

#使用图同构计算多个分子相对于参考分子的对称性修正 RMSD
def symmrmsd(
    coordsref: np.ndarray,
    coords: Union[np.ndarray, List[np.ndarray]],
    apropsref: np.ndarray,
    aprops: np.ndarray,
    amref: np.ndarray,
    am: np.ndarray,
    center: bool = False,
    minimize: bool = False,
    cache: bool = True,
    atol: float = 1e-9,
    return_permutation: bool = False,
) -> Any:
    """
    Compute RMSD using graph isomorphism for multiple coordinates.

    Parameters
    ----------
    coordsref: np.ndarray
        Coordinate of reference molecule
    coords: List[np.ndarray]
        Coordinates of other molecule
    apropsref: np.ndarray
        Atomic properties for reference
    aprops: np.ndarray
        Atomic properties for other molecule
    amref: np.ndarray
        Adjacency matrix for reference molecule
    am: np.ndarray
        Adjacency matrix for other molecule
    center: bool
        Centering flag
    minimize: bool
        Minimum RMSD
    cache: bool
        Cache graph isomorphisms
    atol: float
        Absolute tolerance parameter for QCP (see :func:`qcp_rmsd`)

    Returns
    -------
    float: Union[float, List[float]]
        Symmetry-corrected RMSD(s) and graph isomorphisms

    Notes
    -----

    Graph isomorphism is introduced for symmetry corrections. However, it is also
    useful when two molecules do not have the atoms in the same order since atom
    matching according to atomic numbers and the molecular connectivity is
    performed. If atoms are in the same order and there is no symmetry, use the
    `rmsd` function.
    """
    """
    使用图同构计算多个分子相对于参考分子的 RMSD。

    参数
    ----------
    coordsref: np.ndarray
        参考分子的坐标矩阵。
    coords: List[np.ndarray]
        其他分子的坐标矩阵。
    apropsref: np.ndarray
        参考分子的原子属性。
    aprops: np.ndarray
        其他分子的原子属性。
    amref: np.ndarray
        参考分子的邻接矩阵。
    am: np.ndarray
        其他分子的邻接矩阵。
    center: bool
        是否将分子平移到几何中心。
    minimize: bool
        是否使用最小化 RMSD。
    cache: bool
        是否缓存图同构结果。
    atol: float
        QCP 方法的绝对容差。
    return_permutation: bool
        是否返回最优同构。

    返回值
    -------
    float: Union[float, List[float]]
        计算的 RMSD 值及图同构结果。
    """

    #如果 coords 是列表，依次计算每个分子的 RMSD
    if isinstance(coords, list):  # Multiple RMSD calculations
        RMSD: Any = []
        isomorphism = None
        min_iso = []

        for c in coords:
            if not cache:
                # Reset isomorphism
                isomorphism = None

            srmsd, isomorphism, min_i = _rmsd_isomorphic_core(
                coordsref,
                c,
                apropsref,
                aprops,
                amref,
                am,
                center=center,
                minimize=minimize,
                isomorphisms=isomorphism,
                atol=atol,
            )
            min_iso.append(min_i)
            RMSD.append(srmsd)

    #如果 coords 是单个分子，直接调用 _rmsd_isomorphic_core
    else:  # Single RMSD calculation
        RMSD, isomorphism, min_iso = _rmsd_isomorphic_core(
            coordsref,
            coords,
            apropsref,
            aprops,
            amref,
            am,
            center=center,
            minimize=minimize,
            isomorphisms=None,
            atol=atol,
        )

    #根据 return_permutation 参数决定是否返回同构
    if return_permutation:
        return RMSD, min_iso
    return RMSD

#rmsdwrapper 是一个方便的包装函数，用于计算参考分子 (molref) 和一个或多个目标分子 (mols) 之间的 RMSD。它支持对称性修正、分子中心化、最小化 RMSD 计算，并可选择移除氢原子
def rmsdwrapper(
    molref: molecule.Molecule,
    mols: Union[molecule.Molecule, List[molecule.Molecule]],
    symmetry: bool = True,
    center: bool = False,
    minimize: bool = False,
    strip: bool = True,
    cache: bool = True,
) -> Any:
    """
    Compute RMSD between two molecule.

    Parameters
    ----------
    molref: molecule.Molecule
        Reference molecule
    mols: Union[molecule.Molecule, List[molecule.Molecule]]
        Molecules to compare to reference molecule
    symmetry: bool, optional
        Symmetry-corrected RMSD (using graph isomorphism)
    center: bool, optional
        Center molecules at origin
    minimize: bool, optional
        Minimised RMSD (using the quaternion polynomial method)
    strip: bool, optional
        Strip hydrogen atoms

    Returns
    -------
    List[float]
        RMSDs
    """
    """
    计算两个分子或多个分子之间的 RMSD。

    参数
    ----------
    molref: molecule.Molecule
        参考分子。
    mols: Union[molecule.Molecule, List[molecule.Molecule]]
        需要与参考分子比较的目标分子。
    symmetry: bool, optional
        是否使用对称性修正（基于图同构）。
    center: bool, optional
        是否将分子中心化到原点。
    minimize: bool, optional
        是否使用四元数多项式方法最小化 RMSD。
    strip: bool, optional
        是否移除氢原子。
    cache: bool, optional
        是否缓存图同构的结果。

    返回值
    -------
    List[float]
        计算的 RMSD 列表。
    """

    #如果 mols 是单个分子，将其包装为列表，方便后续统一处理
    if not isinstance(mols, list):
        mols = [mols]

    #如果 strip=True，调用 strip() 方法移除参考分子和目标分子中的氢原子
    if strip:
        molref.strip()

        for mol in mols:
            mol.strip()

    #如果 minimize=True，自动将 center 设置为 True，因为最小化 RMSD 需要分子在原点对齐
    if minimize:
        center = True

    #调用 coords_from_molecule 函数提取参考分子和目标分子的原子坐标，支持中心化操作
    cref = molecule.coords_from_molecule(molref, center)
    cmols = [molecule.coords_from_molecule(mol, center) for mol in mols]

    RMSDlist = []

    #如果 symmetry=True，使用 symmrmsd 函数计算 RMSD。这种方法使用图同构来处理分子对称性
    if symmetry:
        RMSDlist = symmrmsd(
            cref,
            cmols,
            molref.atomicnums,
            mols[0].atomicnums,
            molref.adjacency_matrix,
            mols[0].adjacency_matrix,
            center=center,
            minimize=minimize,
            cache=cache,
        )
    else:  # No symmetry如果 symmetry=False，直接逐个调用 rmsd 函数来计算 RMSD
        for c in cmols:
            RMSDlist.append(
                rmsd(
                    cref,
                    c,
                    molref.atomicnums,
                    mols[0].atomicnums,
                    center=center,
                    minimize=minimize,
                )
            )

    return RMSDlist

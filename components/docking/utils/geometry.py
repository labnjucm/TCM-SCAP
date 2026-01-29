import math
import torch.nn.functional as F
import numpy as np
import torch

#该函数将四元数转换为旋转矩阵，四元数的实部为 r，虚部为 i, j, k。通过数学公式计算出旋转矩阵。
def quaternion_to_matrix(quaternions):
    """
    From https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html
    Convert rotations given as quaternions to rotation matrices.将旋转以四元数形式表示的旋转转换为旋转矩阵。

    Args:
        quaternions: quaternions with real part first,四元数，实部在前，形状为 (..., 4) 的张量。
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).旋转矩阵，形状为 (..., 3, 3) 的张量。
    """
    r, i, j, k = torch.unbind(quaternions, -1)# 将四元数解包为 r, i, j, k，分别表示四元数的实部和虚部
    two_s = 2.0 / (quaternions * quaternions).sum(-1)# 计算常量 2 / (四元数的每个元素的平方和)

     # 计算旋转矩阵的各个元素
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1, # 将结果堆叠在最后一个维度上
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))# 返回旋转矩阵，形状为 (..., 3, 3)

#该函数将旋转表示从轴角形式转换为四元数。使用半角公式计算四元数的实部和虚部。
def axis_angle_to_quaternion(axis_angle):
    """
    From https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html
    Convert rotations given as axis/angle to quaternions.将旋转从轴/角度表示转换为四元数表示。

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.轴角形式的旋转，形状为 (..., 3) 的张量，表示绕轴方向旋转的角度（弧度）。


    Returns:
        quaternions with real part first, as tensor of shape (..., 4).四元数，实部在前，形状为 (..., 4) 的张量。
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)# 计算旋转轴的角度大小（范数）
    half_angles = 0.5 * angles# 计算角度的一半
    # 定义一个小角度的容忍值，用于避免除以 0
    eps = 1e-6
    small_angles = angles.abs() < eps
    # 计算 sin(half_angle) / angle 的值
    sin_half_angles_over_angles = torch.empty_like(angles)
     # 对于非小角度，直接计算 sin(half_angle) / angle
    sin_half_angles_over_angles[~small_angles] = (
            torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    # 对于小角度，使用近似公式计算 sin(half_angle) / angle
    sin_half_angles_over_angles[small_angles] = (
            0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    # 拼接实部和虚部，得到四元数
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions

#该函数首先调用 axis_angle_to_quaternion 将轴角转换为四元数，然后调用 quaternion_to_matrix 将四元数转换为旋转矩阵。
def axis_angle_to_matrix(axis_angle):
    """
    From https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html
    Convert rotations given as axis/angle to rotation matrices.将旋转从轴角度表示转换为旋转矩阵。

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.轴角形式的旋转，形状为 (..., 3) 的张量，表示绕轴方向旋转的角度（弧度）。

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).旋转矩阵，形状为 (..., 3, 3) 的张量。
    """
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))# 首先将轴角转换为四元数，再通过四元数转换为旋转矩阵

#这个函数计算张量中每个元素的平方根，但对于 x 中为零的元素，返回的梯度为零。它通过 torch.max(0, x) 先筛选出大于零的部分，再计算平方根。
def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.返回 torch.sqrt(torch.max(0, x))，但当 x 为 0 时具有零子梯度。这个函数的作用是计算一个张量中每个元素的平方根，只针对大于零的元素，对于零元素返回零，并且确保梯度在 x 为零时为零。
    """
    ret = torch.zeros_like(x)# 创建一个与 x 相同形状的零张量
    positive_mask = x > 0# 创建一个掩码，表示 x 中大于零的元素
    ret[positive_mask] = torch.sqrt(x[positive_mask])# 对于 x 中大于零的元素，计算它们的平方根
    return ret# 返回处理后的张量

#将旋转矩阵转换为四元数。旋转矩阵和四元数都是常用的表示三维旋转的方法，该函数通过从旋转矩阵的元素计算四元数。四元数是一种数学结构，常用于表示三维空间中的旋转。四元数由一个实部和三个虚部组成，通常表示为 (r, i, j, k)，其中 r 是实部，i, j, k 是虚部。
def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.将旋转矩阵转换为四元数。

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3). matrix: 旋转矩阵，形状为 (..., 3, 3) 的张量。

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).四元数，实部在前，形状为 (..., 4) 的张量。
    """
     # 检查输入矩阵的形状是否有效（应为 3x3 的矩阵）
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    # 提取矩阵的批次维度，除了最后两个维度（3x3 矩阵）外的所有维度
    batch_dim = matrix.shape[:-2]
    # 将矩阵展平为一维张量，并按元素拆分为 9 个变量
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    # 计算四元数的绝对值部分，这里用到了自定义的 `_sqrt_positive_part` 函数来确保只有正数参与平方根计算
    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,# 计算四元数实部
                1.0 + m00 - m11 - m22,# 计算四元数 i 部分
                1.0 - m00 + m11 - m22,# 计算四元数 j 部分
                1.0 - m00 - m11 + m22,# 计算四元数 k 部分
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
     # 使用 q_abs 和旋转矩阵的各个元素构建一个临时的四元数候选矩阵
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    # 定义一个阈值，用于防止数值过小造成的精度问题
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    # 计算四元数候选矩阵的正规化值
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    # 通过选择最大值的索引来选择最合适的四元数候选，通常会选择与最大绝对值对应的四元数
    # 这里我们使用 `torch.argmax` 找出每个批次中最大值的索引，并根据该索引选择对应的四元数候选
    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))
"""
四元数与轴/角度表示：四元数和轴/角度形式是两种常用的旋转表示方法。四元数是一种无奇异的旋转表示，而轴/角度表示则用一个单位向量表示旋转轴，角度表示旋转的大小。
Kabsch 算法：Kabsch 算法用于在三维空间中对齐两个点集，寻找最小的均方误差旋转。通过最小化两组点集之间的点到点的距离，计算出最优的刚性变换
"""
#将四元数表示的旋转转换为轴/角度形式（旋转轴和旋转角度）。
def quaternion_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to axis/angle.将四元数表示的旋转转换为轴/角度形式。

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).四元数，实部在前，形状为 (..., 4) 的张量。

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.返回轴/角度表示的旋转，作为形状为 (..., 3) 的张量，其中大小是围绕向量方向的逆时针旋转角度，单位为弧度。
    """
    # 计算四元数的虚部（即 i, j, k 部分）的范数，表示旋转的大小
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    # 计算旋转的半角度，通过 atan2 计算虚部和实部的比值
    half_angles = torch.atan2(norms, quaternions[..., :1])
    # 旋转角度是半角度的两倍
    angles = 2 * half_angles
    # 定义一个小角度阈值
    eps = 1e-6
    # 检测角度是否较小，若小于阈值则使用一个特殊处理
    small_angles = angles.abs() < eps
    # 为了处理小角度的情况，创建一个张量来存储 sin(half_angle) / angle 的值
    sin_half_angles_over_angles = torch.empty_like(angles)
    # 对于正常角度，直接计算 sin(half_angle) / angle
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    # 对于小角度，使用近似公式来计算 sin(half_angle) / angle
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles# 返回轴/角度表示的旋转，注意方向是通过虚部（i, j, k）除以 sin(half_angle) / angle 来计算的

#将旋转矩阵转换为轴/角度形式。首先将矩阵转换为四元数，然后调用 quaternion_to_axis_angle 进行转换。
def matrix_to_axis_angle(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to axis/angle.将旋转矩阵表示的旋转转换为轴/角度形式

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).旋转矩阵，形状为 (..., 3, 3) 的张量。

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.返回轴/角度表示的旋转，作为形状为 (..., 3) 的张量，其中大小是围绕向量方向的逆时针旋转角度，单位为弧度。
    """
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))# 首先将旋转矩阵转换为四元数，然后再转换为轴/角度形式

#使用 Kabsch 算法计算两个 3D 点集的刚性变换（包括旋转矩阵和位移向量）。该算法通过最小化点集之间的均方误差来求解最佳的旋转与平移。
def rigid_transform_Kabsch_3D_torch(A, B):
    # R = 3x3 rotation matrix, t = 3x1 column vector Kabsch 算法，用于在三维空间中对齐两个点集，找到最佳旋转矩阵和位移向量。
    # This already takes residue identity into account.该算法通过最小化两组点集之间的均方误差来确定最佳的刚性变换（包括旋转和位移）。

    """
    Kabsch 算法，用于在三维空间中对齐两个点集，找到最佳旋转矩阵和位移向量。
    
    该算法通过最小化两组点集之间的均方误差来确定最佳的刚性变换（包括旋转和位移）。

    参数:
        A: 形状为 (3, N) 的张量，表示第一个点集（3xN，3D 空间中的 N 个点）
        B: 形状为 (3, N) 的张量，表示第二个点集（3xN，3D 空间中的 N 个点）

    返回:
        R: 3x3 的旋转矩阵
        t: 3x1 的位移向量
    """
    # 验证输入矩阵 A 和 B 的列数是否相同
    assert A.shape[1] == B.shape[1]
    # 获取矩阵的行列数
    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")
    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise: 3 x 1
    # 计算 A 和 B 的质心，3x1 的向量
    centroid_A = torch.mean(A, axis=1, keepdims=True)
    centroid_B = torch.mean(B, axis=1, keepdims=True)

    # subtract mean
    # 将 A 和 B 的点集分别减去它们的质心
    Am = A - centroid_A
    Bm = B - centroid_B

     # 计算协方差矩阵 H
    H = Am @ Bm.T

    # find rotation
    # 使用 SVD 分解来计算旋转矩阵
    U, S, Vt = torch.linalg.svd(H)

    # 计算旋转矩阵 R
    R = Vt.T @ U.T
    # special reflection case
    # 处理特殊的反射情况：若 det(R) < 0，说明存在反射，必须修正
    if torch.linalg.det(R) < 0:
        # print("det(R) < R, reflection detected!, correcting for it ...")
        # 通过调整对角矩阵来消除反射
        SS = torch.diag(torch.tensor([1.,1.,-1.], device=A.device))
        R = (Vt.T @ SS) @ U.T
    # 确保旋转矩阵的行列式接近 1
    assert math.fabs(torch.linalg.det(R) - 1) < 3e-3  # note I had to change this error bound to be higher# 修改了误差阈值，以适应数值计算

    # 计算平移向量 t
    t = -R @ centroid_A + centroid_B
    return R, t# 返回旋转矩阵和位移向量


def rigid_transform_Kabsch_3D_torch_batch(A, B):
    # R = Bx3x3 rotation matrix, t = Bx3x1 column vector
    """
    Kabsch 算法的批处理版本，计算一组 3D 点集的旋转矩阵和位移向量。
    
    参数:
        A: 形状为 BxNx3 的张量，表示 B 个 3D 点集，每个点集有 N 个点
        B: 形状为 BxNx3 的张量，表示 B 个目标点集，每个点集有 N 个点

    返回:
        R: 形状为 Bx3x3 的旋转矩阵
        t: 形状为 Bx3x1 的位移向量
    """
     # 确保 A 和 B 的形状相同
    assert A.shape == B.shape
    _, N, M = A.shape
    # 验证点集的维度，确保每个点集包含三个坐标
    if M != 3:
        raise Exception(f"matrix A and B should be BxNx3")

    # 转置 A 和 B 以便进行矩阵乘法
    A, B = A.permute(0, 2, 1), B.permute(0, 2, 1)

    # find mean column wise: 3 x 1
    # 计算每个点集的质心 (3 x 1)
    centroid_A = torch.mean(A, axis=2, keepdims=True)
    centroid_B = torch.mean(B, axis=2, keepdims=True)

    # subtract mean
    # 将每个点集的点减去对应的质心
    Am = A - centroid_A
    Bm = B - centroid_B
    # 计算协方差矩阵 H
    H = torch.bmm(Am, Bm.transpose(1, 2))

    # find rotation
    # 使用 SVD 分解计算旋转矩阵
    U, S, Vt = torch.linalg.svd(H)
    R = torch.bmm(Vt.transpose(1, 2), U.transpose(1, 2))

    # reflection case
    # 处理反射的特殊情况
    SS = torch.diag(torch.tensor([1., 1., -1.], device=A.device))
    # 对旋转矩阵进行反射修正
    Rm = torch.bmm(Vt.transpose(1,2) @ SS, U.transpose(1, 2))
    # 如果旋转矩阵的行列式小于 0，说明发生了反射，选择修正后的矩阵
    R = torch.where(torch.linalg.det(R)[:, None, None] < 0, Rm, R)
    # 确保旋转矩阵的行列式接近 1
    assert torch.all(torch.abs(torch.linalg.det(R) - 1) < 3e-3)  # note I had to change this error bound to be higher

     # 计算平移向量 t
    t = torch.bmm(-R, centroid_A) + centroid_B
    # 返回旋转矩阵和位移向量
    return R, t


def rigid_transform_Kabsch_independent_torch(A, B):
    # R = 3x3 rotation matrix, t = 3x1 column vector
    # This already takes residue identity into account.
    """
    Kabsch 算法，计算单个 3D 点集的旋转矩阵和位移向量，适用于非批量模式。
    
    参数:
        A: 形状为 3xN 的张量，表示 3D 点集 A，N 是点的数量
        B: 形状为 3xN 的张量，表示目标 3D 点集 B，N 是点的数量

    返回:
        t: 3x1 的位移向量
        R_vec: 旋转矩阵转换为轴/角度表示的旋转向量
    """
     # 确保 A 和 B 的形状匹配
    assert A.shape[1] == B.shape[1]
    num_rows, num_cols = A.shape
    # 验证点集的维度，确保每个点集是 3xN
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")
    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise: 3 x 1
    # 计算每个点集的质心
    centroid_A = torch.mean(A, axis=1, keepdims=True)
    centroid_B = torch.mean(B, axis=1, keepdims=True)

    # subtract mean
    # 将点集 A 和 B 中的点减去对应的质心
    Am = A - centroid_A
    Bm = B - centroid_B

    # 计算协方差矩阵 H
    H = Am @ Bm.T

    # find rotation
    # 使用 SVD 分解来计算旋转矩阵
    U, S, Vt = torch.linalg.svd(H)

    R = Vt.T @ U.T
    # special reflection case
    # 处理反射的特殊情况
    if torch.linalg.det(R) < 0:
        # print("det(R) < R, reflection detected!, correcting for it ...")
        # 对旋转矩阵进行反射修正
        SS = torch.diag(torch.tensor([1.,1.,-1.], device=A.device))
        R = (Vt.T @ SS) @ U.T
    # 确保旋转矩阵的行列式接近 1
    assert math.fabs(torch.linalg.det(R) - 1) < 3e-3  # note I had to change this error bound to be higher

    # 计算平移向量 t
    t = - centroid_A + centroid_B # note does not change rotation
    R_vec = matrix_to_axis_angle(R)# 将旋转矩阵转换为轴/角度表示
    return t, R_vec# 返回平移向量和旋转向量





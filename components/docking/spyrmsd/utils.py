import os

import numpy as np

#提取文件的扩展名，忽略压缩文件的 .gz
def format(fname: str) -> str:
    """
    Extract format extension from file name.

    Parameters
    ----------
    fname : str
        File name

    Returns
    -------
    str
        File extension

    Notes
    -----
    The file extension is returned without the `.` character, i.e. for the file
    `path/filename.ext` the string `ext` is returned.

    If a file is compressed, the `.gz` extension is ignored.
    """
    """
    提取文件名中的扩展名。

    参数
    ----------
    fname : str
        文件名

    返回值
    -------
    str
        文件扩展名（去掉 `.`）

    备注
    -----
    - 返回的文件扩展名不包含 `.` 字符。
    - 如果文件是压缩的（如 `.gz`），则会忽略 `.gz` 扩展名并返回实际文件的扩展名。
    """
    name, ext = os.path.splitext(fname)

    if ext == ".gz":
        _, ext = os.path.splitext(name)

    return ext[1:]  # Remove "."

#从文件名中提取并转换为 OpenBabel 支持的分子文件格式
def molformat(fname: str) -> str:
    """
    Extract an OpenBabel-friendly format from file name.

    Parameters
    ----------
    fname : str
        File name

    Returns
    -------
    str
        File extension in an OpenBabel-friendly format

    Notes
    -----
    File types in OpenBabel do not always correspond to the file extension. This
    function converts the file extension to an OpenBabel file type.

    The following table shows the different conversions performed by this function:

    ========= =========
    Extension File Type
    --------- ---------
    xyz       XYZ
    ========= =========
    """
    """
    提取并转换为 OpenBabel 支持的分子文件格式。

    参数
    ----------
    fname : str
        文件名

    返回值
    -------
    str
        OpenBabel 支持的文件格式

    备注
    -----
    - OpenBabel 支持的文件类型不一定与文件扩展名完全一致。
    - 例如，对于 `.xyz` 文件，OpenBabel 格式为 `XYZ`。
    """

    ext = format(fname)

    if ext == "xyz":
        # xyz files in OpenBabel are called XYZ
        ext = "XYZ"

    return ext

#将角度从度数转换为弧度
def deg_to_rad(angle: float) -> float:
    """
    Convert angle in degrees to angle in radians.

    Parameters
    ----------
    angle : float
        Angle (in degrees)

    Returns
    -------
    float
        Angle (in radians)
    """
    """
    将角度从度数转换为弧度。

    参数
    ----------
    angle : float
        角度（以度为单位）

    返回值
    -------
    float
        角度（以弧度为单位）
    """
    # 使用公式：弧度 = 度数 × (π / 180)

    return angle * np.pi / 180.0

#将一个 3D 向量围绕指定轴旋转一定角度
def rotate(
    v: np.ndarray, angle: float, axis: np.ndarray, units: str = "rad"
) -> np.ndarray:
    """
    Rotate vector.

    Parameters
    ----------
    v: numpy.array
        3D vector to be rotated
    angle : float
        Angle of rotation (in `units`)
    axis : numpy.array
        3D axis of rotation
    units: {"rad", "deg"}
        Units of `angle` (in radians `rad` or degrees `deg`)

    Returns
    -------
    numpy.array
        Rotated vector

    Raises
    ------
    AssertionError
        If the axis of rotation is not a 3D vector
    ValueError
        If `units` is not `rad` or `deg`
    """
    """
    对向量 `v` 进行旋转。

    参数
    ----------
    v: numpy.array
        要旋转的 3D 向量
    angle : float
        旋转角度
    axis : numpy.array
        旋转的 3D 轴
    units: {"rad", "deg"}
        角度的单位（弧度 `rad` 或度数 `deg`）

    返回值
    -------
    numpy.array
        旋转后的向量

    异常
    ------
    AssertionError
        如果旋转轴不是一个 3D 向量
    ValueError
        如果 `units` 既不是 `rad` 也不是 `deg`
    """

    assert len(axis) == 3# 检查旋转轴是否为 3D 向量

    # Ensure rotation axis is normalised
    axis = axis / np.linalg.norm(axis)# 将旋转轴归一化

    # 检查角度单位并转换为弧度
    if units.lower() == "rad":
        pass# 弧度无需转换
    elif units.lower() == "deg":
        angle = deg_to_rad(angle) # 度数转换为弧度
    else:
        raise ValueError(
            f"Units {units} for angle is not supported. Use 'deg' or 'rad' instead."
        )

    # 使用旋转公式计算新坐标
    t1 = np.outer(axis, np.inner(axis, v)).T# 投影到旋转轴方向
    t2 = np.cos(angle) * np.cross(np.cross(axis, v), axis)# 垂直于旋转轴的分量（余弦项）
    t3 = np.sin(angle) * np.cross(axis, v) # 旋转的正弦项

    return t1 + t2 + t3

#计算一组坐标的几何中心
def center_of_geometry(coordinates: np.ndarray) -> np.ndarray:
    """
    Center of geometry.

    Parameters
    ----------
    coordinates: np.ndarray
        Coordinates

    Returns
    -------
    np.ndarray
        Center of geometry
    """
    """
    计算几何中心。

    参数
    ----------
    coordinates: np.ndarray
        坐标数组

    返回值
    -------
    np.ndarray
        几何中心的坐标
    """

    assert coordinates.shape[1] == 3

    return np.mean(coordinates, axis=0)# 计算坐标的均值

#将坐标平移，使其几何中心位于原点
def center(coordinates: np.ndarray) -> np.ndarray:
    """
    Center coordinates.

    Parameters
    ----------
    coordinates: np.ndarray
        Coordinates

    Returns
    -------
    np.ndarray
        Centred coordinates
    """
    """
    将坐标中心化（平移到几何中心位于原点）。

    参数
    ----------
    coordinates: np.ndarray
        坐标数组

    返回值
    -------
    np.ndarray
        平移后的坐标
    """

    return coordinates - center_of_geometry(coordinates)# 使用几何中心平移坐标

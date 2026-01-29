import os
import subprocess

import numpy as np
from rdkit.Chem import AllChem, RemoveHs, RemoveAllHs

from datasets.process_mols import write_mol_with_coords, read_molecule
import re

from utils.utils import remove_all_hs


def read_gnina_metrics(gnina_sdf_path):
    """
    从 gnina SDF 文件中读取指标，返回字典格式的键值对。

    参数:
        gnina_sdf_path: SDF 文件路径，gnina 输出的结果文件。

    返回:
        metrics: 包含所有指标的字典，键是指标名称，值是指标值。
    """
    # 打开 SDF 文件并读取内容
    with open(gnina_sdf_path, 'r') as f:
        pattern = re.compile(r'> <(.*?)>\n(.*?)\n')# 正则表达式匹配 SDF 文件中的指标名称和对应的值
        content = f.read()
        matches = pattern.findall(content)# 找到所有匹配的指标名称和值
        metrics = {k: v for k, v in matches}# 将结果转换为字典格式 {指标名称: 指标值}
    return metrics


def read_gnina_score(gnina_sdf_path):
    """
    从 gnina SDF 文件中读取 CNNscore（评分）。

    参数:
        gnina_sdf_path: SDF 文件路径。

    返回:
        score: 从文件中读取的评分（float 类型）。
    """
    # 打开 SDF 文件并读取内容
    with open(gnina_sdf_path, 'r') as f:
        pattern = re.compile(r'> <CNNscore>\n(.*?)\n')# 使用正则表达式提取 CNNscore 的值
        content = f.read()
        matches = pattern.findall(content)# 获取所有 CNNscore 的匹配项
    return float(matches[0]) # 返回第一个匹配的 CNNscore，转换为浮点型


def invert_permutation(p):
    """Return an array s with which np.array_equal(arr[p][s], arr) is True.
    The array_like argument p must be some permutation of 0, 1, ..., len(p)-1.
    """
    """
    返回一个数组 s，使得 np.array_equal(arr[p][s], arr) 为 True。
    
    参数:
        p: 一个包含从 0 到 len(p)-1 的排列的数组或类似对象。
    
    返回:
        s: 对应的反向排列数组。
    """
    p = np.asanyarray(p) # in case p is a tuple, etc.# 将 p 转换为 numpy 数组（如果 p 是元组等其他类型）
    s = np.empty_like(p)# 创建一个与 p 相同大小的空数组
    s[p] = np.arange(p.size)# 将 s[p] 赋值为 [0, 1, ..., p.size-1]，即反向排列
    return s


def get_gnina_poses(args, mol, pos, orig_center, name, folder, gnina_path, thread_id=0):
    #folder = "data/MOAD_new_test_processed" if args.split == 'test' else "data/MOAD_new_val_processed"
    """
    使用 gnina 软件预测分子的三维位置，执行最小化，并返回优化后的位置、分子对象和评分。

    参数:
        args: 参数对象，包含必要的配置信息（如输出目录、是否进行完全对接等）。
        mol: 输入的分子对象。
        pos: 输入的原始位点。
        orig_center: 原始中心位置，用于定位。
        name: 分子名称。
        folder: 存储数据的文件夹路径。
        gnina_path: gnina 可执行文件的路径。
        thread_id: 当前线程的 ID，用于标记输出文件。

    返回:
        gnina_ligand_pos: 最终的分子位置。
        gnina_mol: 最终的分子对象。
        gnina_score: 最终的评分。
    """
    out_dir = args.out_dir if hasattr(args, 'out_dir') else args.inference_out_dir# 确定输出目录，支持通过 args 提供不同的路径
    rec_path = os.path.join(folder, name[:6] + '_protein_chain_removed.pdb')# 构建受体蛋白和配体文件的路径
    pred_lig_path = os.path.join(out_dir, f'pred_{name}_tid{thread_id}_lig.sdf')

     # 如果目标路径不存在，创建它
    if not os.path.exists(os.path.dirname(pred_lig_path)):
        os.mkdir(os.path.dirname(pred_lig_path))
    
    # 将分子和位置写入 sdf 文件
    print(f'Ligand path {pred_lig_path}')
    write_mol_with_coords(mol, pos + orig_center, pred_lig_path)
    # 构建 gnina 输出路径
    gnina_pred_path = os.path.join(out_dir, f'gnina_{name}_tid{thread_id}_lig.sdf')
    
    # gnina 日志文件存放目录
    gnina_logs_dir = os.path.join(out_dir, "gnina_logs")
    
    # 执行 gnina 软件，生成最小化的配体文件，并将日志写入文件
    with open(os.path.join(gnina_logs_dir, f'{name}'), "w+") as f:
        if args.gnina_full_dock:
            return_code = subprocess.run(
                f'{gnina_path} -r {rec_path} -l "{pred_lig_path}" --autobox_ligand "{pred_lig_path}" -o "{gnina_pred_path}" --no_gpu --autobox_add {args.gnina_autobox_add}',
                shell=True, stdout=f, stderr=f)
        else:
            return_code = subprocess.run(
                f'{gnina_path} --receptor {rec_path} --ligand "{pred_lig_path}" --minimize -o "{gnina_pred_path}"',
                shell=True, stdout=f, stderr=f)

    # print(f'gnina return code: {return_code}')
    # 处理 gnina 执行结果，提取优化后的位点和评分
    try:
        gnina_mol = RemoveAllHs(read_molecule(gnina_pred_path, remove_hs=True, sanitize=True))
        gnina_minimized_ligand_pos = np.array(gnina_mol.GetConformer(0).GetPositions())
        gnina_atoms = np.array([atom.GetSymbol() for atom in gnina_mol.GetAtoms()])
        gnina_filter_Hs = np.where(gnina_atoms != 'H')
        gnina_ligand_pos = gnina_minimized_ligand_pos[gnina_filter_Hs] - orig_center

        try:
            # 获取 gnina 评分
            gnina_score = read_gnina_score(gnina_pred_path)
            if gnina_score is None:
                gnina_score = 0
        except Exception as e:
            print(f'Error reading gnina score: {e}')
            gnina_score = 0

    except Exception as e:
        # 如果 gnina 执行失败，使用原始位置和默认评分
        print(f'Error when running gnina with {name} to minimize energy')
        print('Error:', e)
        print('Using score model output pos instead.')
        gnina_ligand_pos = pos
        gnina_mol = RemoveAllHs(mol)
        gnina_score = 0

    # 返回最终的分子位置、分子对象和评分
    return gnina_ligand_pos, gnina_mol, gnina_score

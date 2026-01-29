"""
预处理模块 - 仅用于推理

处理输入数据：蛋白质、配体等
"""

import os
from typing import Optional, Tuple
from pathlib import Path


def validate_protein_input(
    protein_path: Optional[str],
    protein_sequence: Optional[str]
) -> Tuple[bool, str]:
    """
    验证蛋白质输入
    
    Args:
        protein_path: 蛋白质PDB文件路径
        protein_sequence: 蛋白质序列
    
    Returns:
        (是否有效, 错误消息)
    """
    if protein_path is None and protein_sequence is None:
        return False, "必须提供蛋白质PDB文件或蛋白质序列"
    
    if protein_path is not None:
        if not os.path.exists(protein_path):
            return False, f"蛋白质文件不存在: {protein_path}"
        
        # 检查文件扩展名
        ext = Path(protein_path).suffix.lower()
        if ext not in ['.pdb', '.ent']:
            return False, f"不支持的蛋白质文件格式: {ext}，请使用 .pdb 文件"
    
    return True, ""


def validate_ligand_input(ligand_description: Optional[str]) -> Tuple[bool, str]:
    """
    验证配体输入
    
    Args:
        ligand_description: 配体SMILES字符串或文件路径
    
    Returns:
        (是否有效, 错误消息)
    """
    if ligand_description is None or ligand_description.strip() == "":
        return False, "必须提供配体SMILES字符串或分子文件"
    
    # 如果是文件路径，检查文件是否存在
    if os.path.exists(ligand_description):
        ext = Path(ligand_description).suffix.lower()
        if ext not in ['.sdf', '.mol', '.mol2', '.pdb']:
            return False, f"不支持的配体文件格式: {ext}"
    else:
        # 假设是SMILES字符串，进行基本验证
        if len(ligand_description) < 3:
            return False, "SMILES字符串过短，请检查输入"
        
        # TODO: 可以添加更严格的SMILES验证
    
    return True, ""


def prepare_input_summary(
    protein_path: Optional[str],
    ligand_description: Optional[str],
    protein_sequence: Optional[str],
    complex_name: Optional[str]
) -> str:
    """
    准备输入摘要信息
    
    Returns:
        输入摘要字符串
    """
    lines = []
    lines.append(f"复合物名称: {complex_name or '未指定'}")
    
    if protein_path:
        lines.append(f"蛋白质文件: {Path(protein_path).name}")
    elif protein_sequence:
        seq_preview = protein_sequence[:50] + "..." if len(protein_sequence) > 50 else protein_sequence
        lines.append(f"蛋白质序列: {seq_preview}")
    
    if ligand_description:
        if os.path.exists(ligand_description):
            lines.append(f"配体文件: {Path(ligand_description).name}")
        else:
            lig_preview = ligand_description[:50] + "..." if len(ligand_description) > 50 else ligand_description
            lines.append(f"配体SMILES: {lig_preview}")
    
    return "\n".join(lines)


def normalize_pdb_id(pdb_id: str) -> str:
    """
    规范化PDB ID（4个字符，小写）
    
    Args:
        pdb_id: PDB ID字符串
    
    Returns:
        规范化的PDB ID
    """
    return pdb_id.strip().lower()[:4]


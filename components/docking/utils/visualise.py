from rdkit.Chem.rdmolfiles import MolToPDBBlock, MolToPDBFile
import rdkit.Chem 
from rdkit import Geometry
from collections import defaultdict
import copy
import numpy as np
import torch

#该类用于处理分子对象并生成 PDB 文件，支持添加原子坐标、重复分子部分以及写入多模型的 PDB 文件。   
class PDBFile:
    def __init__(self, mol):
        self.parts = defaultdict(dict)# 存储分子部分及其排列顺序的字典
        self.mol = copy.deepcopy(mol)# 深拷贝 RDKit 分子对象
        [self.mol.RemoveConformer(j) for j in range(mol.GetNumConformers()) if j]  # 移除除第一个以外的所有构象      
    def add(self, coords, order, part=0, repeat=1):#向 PDBFile 中添加分子的坐标信息或 RDKit 分子对象。
        if type(coords) in [rdkit.Chem.Mol, rdkit.Chem.RWMol]:
            block = MolToPDBBlock(coords).split('\n')[:-2]
            self.parts[part][order] = {'block': block, 'repeat': repeat}
            return
        elif type(coords) is np.ndarray:
            coords = coords.astype(np.float64)
        elif type(coords) is torch.Tensor:
            coords = coords.double().numpy()
        for i in range(coords.shape[0]):
            self.mol.GetConformer(0).SetAtomPosition(i, Geometry.Point3D(coords[i, 0], coords[i, 1], coords[i, 2]))
        block = MolToPDBBlock(self.mol).split('\n')[:-2]
        self.parts[part][order] = {'block': block, 'repeat': repeat}
        
    def write(self, path=None, limit_parts=None):#将 self.parts 中的分子信息写入指定的路径，或返回 PDB 文件内容字符串。
        is_first = True
        str_ = ''
        for part in sorted(self.parts.keys()):
            if limit_parts and part >= limit_parts:
                break
            part = self.parts[part]
            keys_positive = sorted(filter(lambda x: x >=0, part.keys()))
            keys_negative = sorted(filter(lambda x: x < 0, part.keys()))
            keys = list(keys_positive) + list(keys_negative)
            for key in keys:
                block = part[key]['block']
                times = part[key]['repeat']
                for _ in range(times):
                    if not is_first:
                        block = [line for line in block if 'CONECT' not in line]
                    is_first = False
                    str_ += 'MODEL\n'
                    str_ += '\n'.join(block)
                    str_ += '\nENDMDL\n'
        if not path:
            return str_
        with open(path, 'w') as f:
            f.write(str_)
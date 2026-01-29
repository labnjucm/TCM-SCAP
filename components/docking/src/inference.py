"""
统一推理 API

提供一个简洁、标准化的接口用于分子对接推理。
完全独立于训练代码。
"""

import copy
import os
import sys
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from argparse import Namespace

import numpy as np
import torch
from rdkit import RDLogger
from rdkit.Chem import RemoveAllHs
from torch_geometric.loader import DataLoader
from tqdm import tqdm

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 导入必要的工具（仅推理相关）
from datasets.process_mols import write_mol_with_coords
from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl, get_t_schedule
from utils.inference_utils import InferenceDataset, set_nones
from utils.sampling import randomize_position, sampling
from utils.utils import get_model
from utils.visualise import PDBFile
from utils.download import download_and_extract

# 禁用RDKit警告
RDLogger.DisableLog('rdApp.*')

# 模型下载配置
REPOSITORY_URL = os.environ.get("REPOSITORY_URL", "https://github.com/gcorso/DiffDock")


class DiffDockRuntime:
    """
    推理运行时
    
    提供标准化的推理接口，支持：
    - 蛋白质-配体对接
    - 批量推理
    - 置信度评分
    - 可视化输出
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化推理运行时
        
        Args:
            config: 配置字典，应包含：
                - model_dir: 模型目录路径
                - ckpt: 模型检查点文件名
                - confidence_model_dir: (可选) 置信度模型目录
                - confidence_ckpt: (可选) 置信度模型检查点
                - device: 'cuda' 或 'cpu' 或 'auto'
                - samples_per_complex: 每个复合物生成的样本数
                - inference_steps: 推理步骤数
                - batch_size: 批量大小
                - 其他推理参数...
        """
        self.config = config
        self.device = self._setup_device()
        
        # 将配置转为 Namespace（兼容原代码）
        self.args = self._dict_to_namespace(config)
        
        # 模型相关
        self.model = None
        self.confidence_model = None
        self.score_model_args = None
        self.confidence_args = None
        self.t_to_sigma = None
        self.tr_schedule = None
        
        print(f"Runtime 初始化完成，使用设备: {self.device}")
    
    def _setup_device(self) -> torch.device:
        """设置计算设备"""
        device_cfg = self.config.get('device', 'auto')
        if device_cfg == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device_cfg)
        return device
    
    def _dict_to_namespace(self, d: Dict) -> Namespace:
        """将字典转换为Namespace"""
        args = Namespace()
        for key, value in d.items():
            setattr(args, key, value)
        return args
    
    def load(self):
        """加载模型和权重"""
        print("正在加载模型...")
        
        # 检查模型目录
        model_dir = self.config.get('model_dir')
        if not model_dir:
            raise ValueError("配置中未指定 model_dir")
        
        # 如果模型目录不存在，尝试自动下载
        if not os.path.exists(model_dir):
            print(f"模型目录不存在: {model_dir}")
            print("正在尝试自动下载模型...")
            
            try:
                self._download_models(model_dir)
            except Exception as e:
                raise ValueError(
                    f"模型目录不存在且自动下载失败: {model_dir}\n"
                    f"错误: {e}\n"
                    f"请手动下载模型并放置到 {model_dir} 目录"
                )
        
        # 加载评分模型的超参数
        with open(f'{model_dir}/model_parameters.yml') as f:
            self.score_model_args = Namespace(**yaml.full_load(f))
        
        # 设置 t_to_sigma 函数
        from functools import partial
        self.t_to_sigma = partial(t_to_sigma_compl, args=self.score_model_args)
        
        # 加载评分模型
        old_score = self.config.get('old_score_model', False)
        self.model = get_model(
            self.score_model_args,
            self.device,
            t_to_sigma=self.t_to_sigma,
            no_parallel=True,
            old=old_score
        )
        
        ckpt_path = f"{model_dir}/{self.config.get('ckpt', 'best_ema_inference_epoch_model.pt')}"
        if not os.path.exists(ckpt_path):
            raise ValueError(f"检查点文件不存在: {ckpt_path}")
        
        state_dict = torch.load(ckpt_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(state_dict, strict=True)
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"✓ 评分模型已加载: {ckpt_path}")
        
        # 加载置信度模型（如果提供）
        confidence_dir = self.config.get('confidence_model_dir')
        if confidence_dir and os.path.exists(confidence_dir):
            with open(f'{confidence_dir}/model_parameters.yml') as f:
                self.confidence_args = Namespace(**yaml.full_load(f))
            
            old_confidence = self.config.get('old_confidence_model', True)
            self.confidence_model = get_model(
                self.confidence_args,
                self.device,
                t_to_sigma=self.t_to_sigma,
                no_parallel=True,
                confidence_mode=True,
                old=old_confidence
            )
            
            conf_ckpt = f"{confidence_dir}/{self.config.get('confidence_ckpt', 'best_model.pt')}"
            state_dict = torch.load(conf_ckpt, map_location=torch.device('cpu'))
            self.confidence_model.load_state_dict(state_dict, strict=True)
            self.confidence_model = self.confidence_model.to(self.device)
            self.confidence_model.eval()
            print(f"✓ 置信度模型已加载: {conf_ckpt}")
        
        # 设置推理调度
        inference_steps = self.config.get('inference_steps', 20)
        self.tr_schedule = get_t_schedule(
            inference_steps=inference_steps,
            sigma_schedule='expbeta'
        )
        
        print("✓ 所有模型加载完成")
    
    def predict(
        self,
        protein_path: Optional[str] = None,
        ligand_description: Optional[str] = None,
        protein_sequence: Optional[str] = None,
        complex_name: Optional[str] = None,
        out_dir: Optional[str] = None,
        save_visualisation: bool = False
    ) -> Dict[str, Any]:
        """
        执行单个蛋白质-配体对接推理
        
        Args:
            protein_path: 蛋白质PDB文件路径
            ligand_description: 配体SMILES字符串或分子文件路径
            protein_sequence: 蛋白质序列（如果没有PDB文件）
            complex_name: 复合物名称
            out_dir: 输出目录
            save_visualisation: 是否保存可视化
        
        Returns:
            推理结果字典，包含：
            - success: 是否成功
            - output_dir: 输出目录
            - ligand_positions: 配体位置列表
            - confidences: 置信度列表
            - files: 生成的文件列表
        """
        if self.model is None:
            raise RuntimeError("模型未加载，请先调用 load() 方法")
        
        # 设置默认值
        if complex_name is None:
            complex_name = "complex_0"
        if out_dir is None:
            out_dir = self.config.get('out_dir', 'results/inference_output')
        
        os.makedirs(out_dir, exist_ok=True)
        
        # 准备数据
        complex_name_list = [complex_name]
        protein_path_list = [protein_path]
        ligand_description_list = [ligand_description]
        protein_sequence_list = [protein_sequence]
        
        # 创建输出目录
        write_dir = f'{out_dir}/{complex_name}'
        os.makedirs(write_dir, exist_ok=True)
        
        # 构建数据集
        try:
            test_dataset = InferenceDataset(
                out_dir=out_dir,
                complex_names=complex_name_list,
                protein_files=protein_path_list,
                ligand_descriptions=ligand_description_list,
                protein_sequences=protein_sequence_list,
                lm_embeddings=True,
                receptor_radius=self.score_model_args.receptor_radius,
                remove_hs=self.score_model_args.remove_hs,
                c_alpha_max_neighbors=self.score_model_args.c_alpha_max_neighbors,
                all_atoms=self.score_model_args.all_atoms,
                atom_radius=self.score_model_args.atom_radius,
                atom_max_neighbors=self.score_model_args.atom_max_neighbors,
                knn_only_graph=False if not hasattr(self.score_model_args, 'not_knn_only_graph') 
                               else not self.score_model_args.not_knn_only_graph
            )
            
            test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
            
        except Exception as e:
            return {
                'success': False,
                'error': f"数据准备失败: {str(e)}",
                'output_dir': write_dir
            }
        
        # 执行推理
        try:
            orig_complex_graph = next(iter(test_loader))
            
            if not orig_complex_graph.success[0]:
                return {
                    'success': False,
                    'error': "复合物图构建失败",
                    'output_dir': write_dir
                }
            
            N = self.config.get('samples_per_complex', 10)
            data_list = [copy.deepcopy(orig_complex_graph) for _ in range(N)]
            
            # 随机化位置
            randomize_position(
                data_list,
                self.score_model_args.no_torsion,
                False,
                self.score_model_args.tr_sigma_max,
                initial_noise_std_proportion=self.config.get('initial_noise_std_proportion', -1.0),
                choose_residue=self.config.get('choose_residue', False)
            )
            
            lig = orig_complex_graph.mol[0]
            
            # 可视化设置
            visualization_list = None
            if save_visualisation:
                visualization_list = []
                for graph in data_list:
                    pdb = PDBFile(lig)
                    pdb.add(lig, 0, 0)
                    pdb.add((orig_complex_graph['ligand'].pos + orig_complex_graph.original_center).detach().cpu(), 1, 0)
                    pdb.add((graph['ligand'].pos + graph.original_center).detach().cpu(), part=1, order=1)
                    visualization_list.append(pdb)
            
            # 执行采样
            actual_steps = self.config.get('actual_steps')
            if actual_steps is None:
                actual_steps = self.config.get('inference_steps', 20)
            # 准备置信度数据集（如果有置信度模型）
            confidence_data_list = None
            if self.confidence_model is not None:
                # 为置信度模型创建数据副本
                confidence_data_list = [copy.deepcopy(orig_complex_graph) for _ in range(N)]
            
            data_list, confidence = sampling(
                data_list=data_list,
                model=self.model,
                inference_steps=actual_steps,
                tr_schedule=self.tr_schedule,
                rot_schedule=self.tr_schedule,
                tor_schedule=self.tr_schedule,
                device=self.device,
                t_to_sigma=self.t_to_sigma,
                model_args=self.score_model_args,
                visualization_list=visualization_list,
                confidence_model=self.confidence_model,
                confidence_data_list=confidence_data_list,
                confidence_model_args=self.confidence_args,
                batch_size=self.config.get('batch_size', 10),
                no_final_step_noise=self.config.get('no_final_step_noise', True),
                temp_sampling=[
                    self.config.get('temp_sampling_tr', 1.0),
                    self.config.get('temp_sampling_rot', 1.0),
                    self.config.get('temp_sampling_tor', 1.0)
                ],
                temp_psi=[
                    self.config.get('temp_psi_tr', 0.0),
                    self.config.get('temp_psi_rot', 0.0),
                    self.config.get('temp_psi_tor', 0.0)
                ],
                temp_sigma_data=[
                    self.config.get('temp_sigma_data_tr', 0.5),
                    self.config.get('temp_sigma_data_rot', 0.5),
                    self.config.get('temp_sigma_data_tor', 0.5)
                ]
            )
            
            # 提取配体位置
            ligand_pos = np.asarray([
                complex_graph['ligand'].pos.cpu().numpy() + orig_complex_graph.original_center.cpu().numpy()
                for complex_graph in data_list
            ])
            
            # 按置信度排序
            if confidence is not None and isinstance(
                self.confidence_args.rmsd_classification_cutoff if self.confidence_args else None, list
            ):
                confidence = confidence[:, 0]
            
            if confidence is not None:
                confidence = confidence.cpu().numpy()
                re_order = np.argsort(confidence)[::-1]
                confidence = confidence[re_order]
                ligand_pos = ligand_pos[re_order]
            else:
                confidence = [0.0] * len(ligand_pos)
            
            # 保存结果
            output_files = []
            for rank, pos in enumerate(ligand_pos):
                mol_pred = copy.deepcopy(lig)
                if self.score_model_args.remove_hs:
                    mol_pred = RemoveAllHs(mol_pred)
                
                # 保存文件
                output_file = os.path.join(write_dir, f'rank{rank+1}_confidence{confidence[rank]:.2f}.sdf')
                write_mol_with_coords(mol_pred, pos, output_file)
                output_files.append(output_file)
                
                # rank1 也保存一份不带置信度的版本
                if rank == 0:
                    rank1_file = os.path.join(write_dir, f'rank1.sdf')
                    write_mol_with_coords(mol_pred, pos, rank1_file)
            
            # 保存可视化
            if save_visualisation and visualization_list:
                for rank, batch_idx in enumerate(re_order if confidence is not None else range(len(ligand_pos))):
                    vis_file = os.path.join(write_dir, f'rank{rank+1}_reverseprocess.pdb')
                    visualization_list[batch_idx].write(vis_file)
                    output_files.append(vis_file)
            
            return {
                'success': True,
                'output_dir': write_dir,
                'ligand_positions': ligand_pos.tolist(),
                'confidences': confidence.tolist() if isinstance(confidence, np.ndarray) else confidence,
                'files': output_files,
                'complex_name': complex_name
            }
            
        except Exception as e:
            import traceback
            return {
                'success': False,
                'error': f"推理失败: {str(e)}",
                'traceback': traceback.format_exc(),
                'output_dir': write_dir
            }
    
    def batch_predict(
        self,
        protein_ligand_pairs: List[Dict[str, str]],
        out_dir: Optional[str] = None,
        save_visualisation: bool = False
    ) -> List[Dict[str, Any]]:
        """
        批量推理
        
        Args:
            protein_ligand_pairs: 列表，每个元素是一个字典，包含：
                - protein_path: 蛋白质文件路径
                - ligand_description: 配体描述
                - complex_name: (可选) 复合物名称
                - protein_sequence: (可选) 蛋白质序列
            out_dir: 输出目录
            save_visualisation: 是否保存可视化
        
        Returns:
            推理结果列表
        """
        results = []
        for i, pair in enumerate(tqdm(protein_ligand_pairs, desc="批量推理")):
            result = self.predict(
                protein_path=pair.get('protein_path'),
                ligand_description=pair.get('ligand_description'),
                protein_sequence=pair.get('protein_sequence'),
                complex_name=pair.get('complex_name', f'complex_{i}'),
                out_dir=out_dir,
                save_visualisation=save_visualisation
            )
            results.append(result)
        
        return results


# 便捷函数
def create_runtime_from_yaml(config_path: str) -> DiffDockRuntime:
    """从YAML配置文件创建Runtime"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return DiffDockRuntime(config)


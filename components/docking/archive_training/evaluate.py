import copy
import os
import torch
from datasets.moad import MOAD
from utils.gnina_utils import get_gnina_poses
from utils.molecules_utils import get_symmetry_rmsd

torch.multiprocessing.set_sharing_strategy('file_system')

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (64000, rlimit[1]))

import time
from argparse import ArgumentParser, Namespace, FileType
from datetime import datetime
from functools import partial
import numpy as np
import wandb
from rdkit import RDLogger
from torch_geometric.loader import DataLoader
from rdkit.Chem import RemoveAllHs

from datasets.pdbbind import PDBBind
from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl, get_t_schedule
from utils.sampling import randomize_position, sampling
from utils.utils import get_model, ExponentialMovingAverage
from utils.visualise import PDBFile
from tqdm import tqdm

RDLogger.DisableLog('rdApp.*')
import yaml
import pickle


def get_dataset(args, model_args, confidence=False):
    if args.dataset != 'moad':
        dataset = PDBBind(transform=None, root=args.data_dir, limit_complexes=args.limit_complexes, dataset=args.dataset,
                        chain_cutoff=args.chain_cutoff,
                        receptor_radius=model_args.receptor_radius,
                        cache_path=args.cache_path, split_path=args.split_path,
                        remove_hs=model_args.remove_hs, max_lig_size=None,
                        c_alpha_max_neighbors=model_args.c_alpha_max_neighbors,
                        matching=not model_args.no_torsion, keep_original=True,
                        popsize=args.matching_popsize,
                        maxiter=args.matching_maxiter,
                        all_atoms=model_args.all_atoms if 'all_atoms' in model_args else False,
                        atom_radius=model_args.atom_radius if 'all_atoms' in model_args else None,
                        atom_max_neighbors=model_args.atom_max_neighbors if 'all_atoms' in model_args else None,
                        esm_embeddings_path=args.esm_embeddings_path,
                        require_ligand=True,
                        num_workers=args.num_workers,
                        protein_file=args.protein_file,
                        ligand_file=args.ligand_file,
                        knn_only_graph=True if not hasattr(args, 'not_knn_only_graph') else not args.not_knn_only_graph,
                        include_miscellaneous_atoms=False if not hasattr(args,
                                                                         'include_miscellaneous_atoms') else args.include_miscellaneous_atoms,
                        num_conformers=args.samples_per_complex if args.resample_rdkit and not confidence else 1)

    else:
        dataset = MOAD(transform=None, root=args.data_dir, limit_complexes=args.limit_complexes,
                       chain_cutoff=args.chain_cutoff,
                       receptor_radius=model_args.receptor_radius,
                       cache_path=args.cache_path, split=args.split,
                       remove_hs=model_args.remove_hs, max_lig_size=None,
                       c_alpha_max_neighbors=model_args.c_alpha_max_neighbors,
                       matching=not model_args.no_torsion, keep_original=True,
                       popsize=args.matching_popsize,
                       maxiter=args.matching_maxiter,
                       all_atoms=model_args.all_atoms if 'all_atoms' in model_args else False,
                       atom_radius=model_args.atom_radius if 'all_atoms' in model_args else None,
                       atom_max_neighbors=model_args.atom_max_neighbors if 'all_atoms' in model_args else None,
                       esm_embeddings_path=args.esm_embeddings_path,
                       esm_embeddings_sequences_path=args.moad_esm_embeddings_sequences_path,
                       require_ligand=True,
                       num_workers=args.num_workers,
                       knn_only_graph=True if not hasattr(args, 'not_knn_only_graph') else not args.not_knn_only_graph,
                       include_miscellaneous_atoms=False if not hasattr(args,
                                                                        'include_miscellaneous_atoms') else args.include_miscellaneous_atoms,
                       num_conformers=args.samples_per_complex if args.resample_rdkit and not confidence else 1,
                       unroll_clusters=args.unroll_clusters, remove_pdbbind=args.remove_pdbbind,
                       min_ligand_size=args.min_ligand_size,
                       max_receptor_size=args.max_receptor_size,
                       remove_promiscuous_targets=args.remove_promiscuous_targets,
                       no_randomness=True,
                       skip_matching=args.skip_matching)
    return dataset



if __name__ == '__main__':
    cache_name = datetime.now().strftime('date%d-%m_time%H-%M-%S.%f') # 创建一个基于当前时间的缓存文件名
    parser = ArgumentParser()# 创建 ArgumentParser 实例，用于处理命令行参数
    parser.add_argument('--config', type=FileType(mode='r'), default=None)#接受一个文件类型的参数，用于传入配置文件。
    parser.add_argument('--model_dir', type=str, default='workdir/test_score', help='Path to folder with trained score model and hyperparameters')#指定已训练的模型和超参数的目录路径，默认路径是 workdir/test_score。
    parser.add_argument('--ckpt', type=str, default='best_ema_inference_epoch_model.pt', help='Checkpoint to use inside the folder')#指定模型检查点文件的名称，默认是 best_ema_inference_epoch_model.pt。
    parser.add_argument('--confidence_model_dir', type=str, default=None, help='Path to folder with trained confidence model and hyperparameters')#指定置信度模型的目录路径。
    parser.add_argument('--confidence_ckpt', type=str, default='best_model_epoch75.pt', help='Checkpoint to use inside the folder')#指定置信度模型的检查点文件，默认是 best_model_epoch75.pt。
    parser.add_argument('--num_cpu', type=int, default=None, help='if this is a number instead of none, the max number of cpus used by torch will be set to this.')#指定要使用的最大 CPU 数量。如果为 None，则没有限制。
    parser.add_argument('--run_name', type=str, default='test', help='')#运行名称，默认值是 test。
    parser.add_argument('--project', type=str, default='ligbind_inf', help='')#项目名称，默认是 ligbind_inf。
    parser.add_argument('--out_dir', type=str, default=None, help='Where to save results to')#结果保存的目录路径。
    parser.add_argument('--batch_size', type=int, default=40, help='Number of poses to sample in parallel')#批处理大小，表示每次并行采样的姿势数量，默认值是 40。

    parser.add_argument('--old_score_model', action='store_true', default=False, help='')#布尔参数，如果传入则设置为 True，否则为 False。通常用于标识是否使用旧版本的评分模型。

    parser.add_argument('--old_confidence_model', action='store_true', default=True, help='')#类似 --old_score_model，用于标识是否使用旧版本的置信度模型，默认为 True。
    parser.add_argument('--matching_popsize', type=int, default=40, help='Differential evolution popsize parameter in matching')#差分进化算法的种群大小，默认为 40。
    parser.add_argument('--matching_maxiter', type=int, default=40, help='Differential evolution maxiter parameter in matching')#差分进化算法的最大迭代次数，默认为 40。

    parser.add_argument('--esm_embeddings_path', type=str, default=None, help='If this is set then the LM embeddings at that path will be used for the receptor features')#指定语言模型 (LM) 嵌入的路径，用于受体特征。如果设置了此参数，模型会使用该路径的嵌入。
    parser.add_argument('--moad_esm_embeddings_sequences_path', type=str, default=None, help='')#MOAD 特有的 ESM 嵌入序列路径。
    parser.add_argument('--chain_cutoff', type=float, default=None, help='Cutoff of the chains from the ligand') # TODO remove一个浮动数值，表示从配体中切除的链的阈值。注释中提到 TODO remove，意味着这个参数可能会被删除。
    parser.add_argument('--save_complexes', action='store_true', default=False, help='Save generated complex graphs')#布尔值参数，如果设置，则保存生成的复合物图。
    parser.add_argument('--complexes_save_path', type=str, default=None, help='')#保存复合物图的路径

    parser.add_argument('--dataset', type=str, default='moad', help='')#指定使用的数据集，默认为 'moad'
    parser.add_argument('--cache_path', type=str, default='data/cache', help='Folder from where to load/restore cached dataset')#指定缓存文件夹路径，用于加载或恢复缓存的数据集。
    parser.add_argument('--data_dir', type=str, default='../../ligbind/data/BindingMOAD_2020_ab_processed_biounit/', help='Folder containing original structures')#指定包含原始结构的数据文件夹路径，默认为 '../../ligbind/data/BindingMOAD_2020_ab_processed_biounit/'。
    parser.add_argument('--split_path', type=str, default='data/BindingMOAD_2020_ab_processed/splits/val.txt', help='Path of file defining the split')#指定定义数据集拆分（如训练集、验证集等）的文件路径，默认为 'data/BindingMOAD_2020_ab_processed/splits/val.txt'。

    parser.add_argument('--no_model', action='store_true', default=False, help='Whether to return seed conformer without running model')#布尔类型的参数。如果指定此参数（即 --no_model），则模型不会被运行，返回的将是种子构象，而不是经过模型推理生成的结果。
    parser.add_argument('--no_random', action='store_true', default=False, help='Whether to add randomness in diffusion steps')#布尔类型的参数，指定是否在扩散步骤中加入随机性。如果设置此参数，扩散过程中将不会引入随机性。
    parser.add_argument('--no_final_step_noise', action='store_true', default=False, help='Whether to add noise after the final step')#布尔类型的参数，指定是否在最后一步之后加入噪声。如果指定此参数，最后一步不会加噪声。
    parser.add_argument('--ode', action='store_true', default=False, help='Whether to run the probability flow ODE')#布尔类型的参数，指定是否运行概率流 ODE（常用于优化过程中的概率流模拟）。如果设置了此参数，将运行 ODE 模拟。
    parser.add_argument('--wandb', action='store_true', default=False, help='') # TODO remove布尔类型的参数，指定是否使用 wandb（Weights & Biases）进行实验跟踪和可视化。当前代码注释指出这个参数是“待删除”的。
    parser.add_argument('--inference_steps', type=int, default=40, help='Number of denoising steps')#指定去噪步骤的数量，默认为 40。这通常与模型推理的迭代次数有关。
    parser.add_argument('--limit_complexes', type=int, default=0, help='Limit to the number of complexes')#指定最大复合物数量的限制。如果设置为 0，则没有限制。
    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers for dataset creation')#指定用于数据集创建的工作线程数，默认为 1。增加该参数的值可以加速数据集的加载过程，尤其是在多核处理器上。
    parser.add_argument('--tqdm', action='store_true', default=False, help='Whether to show progress bar')#布尔类型的参数，指定是否使用 tqdm 显示进度条。如果指定此参数，训练或数据处理过程将显示进度条。
    parser.add_argument('--save_visualisation', action='store_true', default=True, help='Whether to save visualizations')#布尔类型的参数，指定是否保存可视化结果，默认为 True，即保存可视化。
    parser.add_argument('--samples_per_complex', type=int, default=4, help='Number of poses to sample for each complex')#每个复合物（例如蛋白质-配体复合物）采样的姿势数量，默认为 4。
    parser.add_argument('--resample_rdkit', action='store_true', default=False, help='')#布尔类型的参数，指定是否使用 RDKit 重新采样分子结构。RDKit 是一个化学信息学工具包。
    parser.add_argument('--skip_matching', action='store_true', default=False, help='')#布尔类型的参数，指定是否跳过配体与受体的匹配步骤。如果指定该参数，匹配过程将被跳过。
    parser.add_argument('--sigma_schedule', type=str, default='expbeta', help='Schedule type, no other options')#指定噪声调度的类型，默认为 'expbeta'，目前没有其他选项。
    parser.add_argument('--inf_sched_alpha', type=float, default=1, help='Alpha parameter of beta distribution for t sched')#t 调度的 alpha 参数，通常用于贝塔分布的控制，默认为 1。
    parser.add_argument('--inf_sched_beta', type=float, default=1, help='Beta parameter of beta distribution for t sched')#t 调度的 beta 参数，通常用于贝塔分布的控制，默认为 1。
    parser.add_argument('--pocket_knowledge', action='store_true', default=False, help='')#布尔类型的参数，指定是否使用口袋（pocket）知识。口袋通常指的是蛋白质的配体结合位点。
    parser.add_argument('--no_random_pocket', action='store_true', default=False, help='')#布尔类型的参数，指定是否不引入随机的口袋信息。
    parser.add_argument('--pocket_tr_max', type=float, default=3, help='')#指定口袋的最大转换半径，默认为 3。
    parser.add_argument('--pocket_cutoff', type=float, default=5, help='')#指定口袋信息的截断距离，默认为 5。
    parser.add_argument('--actual_steps', type=int, default=None, help='')#实际步骤的数量，默认为 None，可以用于指定推理过程中要执行的实际步骤数。
    parser.add_argument('--restrict_cpu', action='store_true', default=False, help='')#布尔类型的参数，指定是否限制 CPU 使用。可以用于控制模型的计算资源限制。
    parser.add_argument('--force_fixed_center_conv', action='store_true', default=False, help='')#布尔类型的参数，强制固定中心卷积。如果指定了此参数，卷积操作将在固定的中心位置进行。
    parser.add_argument('--protein_file', type=str, default='protein_processed', help='')#指定蛋白质文件的名称，默认为 'protein_processed'。该文件通常包含蛋白质的结构数据。
    parser.add_argument('--unroll_clusters', action='store_true', default=True, help='')#布尔类型的参数，指定是否展开簇群，默认为 True。
    parser.add_argument('--ligand_file', type=str, default='ligand', help='')#指定配体文件的名称，默认为 'ligand'。该文件通常包含配体的结构数据。
    parser.add_argument('--remove_pdbbind', action='store_true', default=False, help='')#布尔类型的参数，指定是否删除 PDBBind 数据集的相关数据。
    parser.add_argument('--split', type=str, default='val', help='')#指定数据集拆分的类型，默认为 'val'（验证集）。
    parser.add_argument('--limit_failures', type=float, default=5, help='')#指定训练过程中最大允许的失败次数，默认为 5。
    parser.add_argument('--min_ligand_size', type=float, default=0, help='')#指定最小配体大小，默认为 0。通常用于筛选符合大小要求的配体。
    parser.add_argument('--max_receptor_size', type=float, default=None, help='')#指定最大受体大小，默认为 None。
    parser.add_argument('--remove_promiscuous_targets', type=float, default=None, help='')#指定是否删除容易与多个分子结合的受体目标，默认为 None。
    parser.add_argument('--initial_noise_std_proportion', type=float, default=-1.0, help='Initial noise std proportion')#指定初始噪声的标准差比例，默认为 -1.0。
    parser.add_argument('--choose_residue', action='store_true', default=False, help='')#布尔类型的参数，指定是否选择特定的残基。

    parser.add_argument('--temp_sampling_tr', type=float, default=1.0)
    parser.add_argument('--temp_psi_tr', type=float, default=0.0)
    parser.add_argument('--temp_sigma_data_tr', type=float, default=0.5)
    parser.add_argument('--temp_sampling_rot', type=float, default=1.0)
    parser.add_argument('--temp_psi_rot', type=float, default=0.0)
    parser.add_argument('--temp_sigma_data_rot', type=float, default=0.5)
    parser.add_argument('--temp_sampling_tor', type=float, default=1.0)
    parser.add_argument('--temp_psi_tor', type=float, default=0.0)
    parser.add_argument('--temp_sigma_data_tor', type=float, default=0.5)

    parser.add_argument('--gnina_minimize', action='store_true', default=False, help='')#布尔类型的参数，指定是否使用 gnina 进行能量最小化。
    parser.add_argument('--gnina_path', type=str, default='gnina', help='')#指定 gnina 可执行文件的路径，默认为 'gnina'。
    parser.add_argument('--gnina_log_file', type=str, default='gnina_log.txt', help='')#指定 gnina 日志文件的路径，用于将 gnina 子进程的标准输出重定向到文件。
    parser.add_argument('--gnina_full_dock', action='store_true', default=False, help='')#布尔类型的参数，指定是否执行完整的对接过程。
    parser.add_argument('--save_gnina_metrics', action='store_true', default=False, help='')#布尔类型的参数，指定是否保存 gnina 的评估指标。
    parser.add_argument('--gnina_autobox_add', type=float, default=4.0)#指定自动框架扩展的大小，默认为 4.0。
    parser.add_argument('--gnina_poses_to_optimize', type=int, default=1)#指定要优化的姿势数，默认为 1。

    args = parser.parse_args()#解析命令行传入的参数并存储在 args 中，用于后续处理。
    # 如果提供了 config 文件路径
    if args.config:
        # 载入配置文件并转换为字典
        config_dict = yaml.load(args.config, Loader=yaml.FullLoader)
         # 将命令行参数转换为字典形式
        arg_dict = args.__dict__
        # 遍历配置字典中的每一项，将其值应用到命令行参数字典中
        for key, value in config_dict.items():
            # 如果值是列表类型，追加到现有的命令行参数中
            if isinstance(value, list):
                for v in value:
                    arg_dict[key].append(v)
            else:
                # 否则直接覆盖现有的命令行参数
                arg_dict[key] = value

    # 如果限制 CPU 使用
    if args.restrict_cpu:
        # 设置 CPU 使用的线程数为 16
        threads = 16
        # 设置环境变量，限制各个库的最大线程数为 16
        os.environ["OMP_NUM_THREADS"] = str(threads)  # export OMP_NUM_THREADS=4设置 OpenMP 线程数
        os.environ["OPENBLAS_NUM_THREADS"] = str(threads)  # export OPENBLAS_NUM_THREADS=4设置 OpenBLAS 线程数
        os.environ["MKL_NUM_THREADS"] = str(threads)  # export MKL_NUM_THREADS=6设置 MKL 线程数
        os.environ["VECLIB_MAXIMUM_THREADS"] = str(threads)  # export VECLIB_MAXIMUM_THREADS=4设置 VECLIB 线程数
        os.environ["NUMEXPR_NUM_THREADS"] = str(threads)  # export NUMEXPR_NUM_THREADS=6设置 NUMEXPR 线程数
        # 使 GPU 不可用（不使用 CUDA 设备）
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

        # 设置 PyTorch 使用的线程数
        torch.set_num_threads(threads)

    # 如果没有指定输出目录，则使用默认目录
    if args.out_dir is None: args.out_dir = f'inference_out_dir_not_specified/{args.run_name}'
    # 创建输出目录（如果不存在的话）
    os.makedirs(args.out_dir, exist_ok=True)
    # 读取模型的配置文件
    with open(f'{args.model_dir}/model_parameters.yml') as f:
        score_model_args = Namespace(**yaml.full_load(f))
        # 如果配置文件中没有某些必要的属性，设置它们为默认值
        if not hasattr(score_model_args, 'separate_noise_schedule'):  # exists for compatibility with old runs that did not have the attribute
            score_model_args.separate_noise_schedule = False
        if not hasattr(score_model_args, 'lm_embeddings_path'):
            score_model_args.lm_embeddings_path = None
        if not hasattr(score_model_args, 'tr_only_confidence'):
            score_model_args.tr_only_confidence = True
        if not hasattr(score_model_args, 'high_confidence_threshold'):
            score_model_args.high_confidence_threshold = 0.0
        if not hasattr(score_model_args, 'include_confidence_prediction'):
            score_model_args.include_confidence_prediction = False
        if not hasattr(score_model_args, 'confidence_weight'):
            score_model_args.confidence_weight = 1
        if not hasattr(score_model_args, 'asyncronous_noise_schedule'):
            score_model_args.asyncronous_noise_schedule = False
        if not hasattr(score_model_args, 'correct_torsion_sigmas'):
            score_model_args.correct_torsion_sigmas = False
        if not hasattr(score_model_args, 'esm_embeddings_path'):
            score_model_args.esm_embeddings_path = None
         # 如果启用了 `force_fixed_center_conv`，则将相关参数设置为 False
        if args.force_fixed_center_conv:
            score_model_args.not_fixed_center_conv = False
    # 如果提供了信心模型目录
    if args.confidence_model_dir is not None:
        # 读取信心模型的配置文件
        with open(f'{args.confidence_model_dir}/model_parameters.yml') as f:
            confidence_args = Namespace(**yaml.full_load(f))
        # 如果原始模型目录不存在，调整路径
        if not os.path.exists(confidence_args.original_model_dir):
            print("Path does not exist: ", confidence_args.original_model_dir)
            # 尝试使用父目录路径
            confidence_args.original_model_dir = os.path.join(*confidence_args.original_model_dir.split('/')[-2:])
            print('instead trying path: ', confidence_args.original_model_dir)
        # 设置缺失的默认参数
        if not hasattr(confidence_args, 'use_original_model_cache'):
            confidence_args.use_original_model_cache = True
        if not hasattr(confidence_args, 'esm_embeddings_path'):
            confidence_args.esm_embeddings_path = None
        if not hasattr(confidence_args, 'num_classification_bins'):
            confidence_args.num_classification_bins = 2

    # 如果指定了 num_cpu 参数，则设置 PyTorch 使用的线程数
    if args.num_cpu is not None:
        torch.set_num_threads(args.num_cpu)
    # 根据硬件环境选择设备（CUDA 或 CPU）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载测试数据集
    test_dataset = get_dataset(args, score_model_args)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    # 如果指定了 confidence_model_dir（置信度模型目录）
    if args.confidence_model_dir is not None:
        if not (confidence_args.use_original_model_cache or confidence_args.transfer_weights):
            # if the confidence model uses the same type of data as the original model then we do not need this dataset and can just use the complexes
            print('HAPPENING | confidence model uses different type of graphs than the score model. Loading (or creating if not existing) the data for the confidence model now.')
            # 加载或创建置信度模型的数据集
            confidence_test_dataset = get_dataset(args, confidence_args, confidence=True)
            # 将数据集按名称存储为字典，便于快速查找
            confidence_complex_dict = {d.name: d for d in confidence_test_dataset}

    # 定义从时间步 t 到噪声标准差 sigma 的映射
    t_to_sigma = partial(t_to_sigma_compl, args=score_model_args)

    # 如果未设置 no_model，则加载评分模型
    if not args.no_model:
        # 加载评分模型
        model = get_model(score_model_args, device, t_to_sigma=t_to_sigma, no_parallel=True, old=args.old_score_model)
        # 加载模型的状态字典（参数）
        state_dict = torch.load(f'{args.model_dir}/{args.ckpt}', map_location=torch.device('cpu'))
         # 如果加载的是 'last_model.pt' 检查点
        if args.ckpt == 'last_model.pt':
            model_state_dict = state_dict['model']
            ema_weights_state = state_dict['ema_weights']
            # 加载模型参数
            model.load_state_dict(model_state_dict, strict=True)
            # 加载 EMA 权重并应用到模型中
            ema_weights = ExponentialMovingAverage(model.parameters(), decay=score_model_args.ema_rate)
            ema_weights.load_state_dict(ema_weights_state, device=device)
            ema_weights.copy_to(model.parameters())
        else:
            # 加载普通检查点
            model.load_state_dict(state_dict, strict=True)
            model = model.to(device)
            model.eval()
        # 如果指定了置信度模型目录
        if args.confidence_model_dir is not None:
            # 如果需要权重迁移
            if confidence_args.transfer_weights:
                with open(f'{confidence_args.original_model_dir}/model_parameters.yml') as f:
                    confidence_model_args = Namespace(**yaml.full_load(f))
                # 为兼容旧版本，设置缺失的默认参数
                if not hasattr(confidence_model_args, 'separate_noise_schedule'):  # exists for compatibility with old runs that did not have the
                    # attribute
                    confidence_model_args.separate_noise_schedule = False
                if not hasattr(confidence_model_args, 'lm_embeddings_path'):
                    confidence_model_args.lm_embeddings_path = None
                if not hasattr(confidence_model_args, 'tr_only_confidence'):
                    confidence_model_args.tr_only_confidence = True
                if not hasattr(confidence_model_args, 'high_confidence_threshold'):
                    confidence_model_args.high_confidence_threshold = 0.0
                if not hasattr(confidence_model_args, 'include_confidence_prediction'):
                    confidence_model_args.include_confidence_prediction = False
                if not hasattr(confidence_model_args, 'confidence_dropout'):
                    confidence_model_args.confidence_dropout = confidence_model_args.dropout
                if not hasattr(confidence_model_args, 'confidence_no_batchnorm'):
                    confidence_model_args.confidence_no_batchnorm = False
                if not hasattr(confidence_model_args, 'confidence_weight'):
                    confidence_model_args.confidence_weight = 1
                if not hasattr(confidence_model_args, 'asyncronous_noise_schedule'):
                    confidence_model_args.asyncronous_noise_schedule = False
                if not hasattr(confidence_model_args, 'correct_torsion_sigmas'):
                    confidence_model_args.correct_torsion_sigmas = False
                if not hasattr(confidence_model_args, 'esm_embeddings_path'):
                    confidence_model_args.esm_embeddings_path = None
                if not hasattr(confidence_model_args, 'not_fixed_knn_radius_graph'):
                    confidence_model_args.not_fixed_knn_radius_graph = True
                if not hasattr(confidence_model_args, 'not_knn_only_graph'):
                    confidence_model_args.not_knn_only_graph = True
            else:
                confidence_model_args = confidence_args

            # 加载置信度模型
            confidence_model = get_model(confidence_model_args, device, t_to_sigma=t_to_sigma, no_parallel=True,
                                        confidence_mode=True, old=args.old_confidence_model)
            # 加载置信度模型检查点
            state_dict = torch.load(f'{args.confidence_model_dir}/{args.confidence_ckpt}', map_location=torch.device('cpu'))
            confidence_model.load_state_dict(state_dict, strict=True)
            confidence_model = confidence_model.to(device)
            confidence_model.eval()
        else:
            # 如果没有指定置信度模型目录，则将其置为 None
            confidence_model = None
            confidence_args = None
            confidence_model_args = None

    #检查是否启用了 wandb，用于实验追踪和日志记录。
    if args.wandb:
        run = wandb.init(
            entity='',
            settings=wandb.Settings(start_method="fork"),
            project=args.project,
            name=args.run_name,
            config=args
        )

    #计算时间步 t_max,如果启用了 pocket_knowledge 并使用了不同的调度（different_schedules），计算 t_max。表示扩散过程中最大时间步的归一化值，基于 pocket_tr_max 和评分模型的扩散噪声范围计算。将时间步 t 映射到噪声标准差 sigma 的对数范围内，并归一化到 [0, 1]。
    if args.pocket_knowledge and args.different_schedules:
        t_max = (np.log(args.pocket_tr_max) - np.log(score_model_args.tr_sigma_min)) / (
                    np.log(score_model_args.tr_sigma_max) - np.log(score_model_args.tr_sigma_min))
    else:
        t_max = 1

    #生成时间调度
    tr_schedule = get_t_schedule(sigma_schedule=args.sigma_schedule, inference_steps=args.inference_steps,
                                 inf_sched_alpha=args.inf_sched_alpha, inf_sched_beta=args.inf_sched_beta,
                                 t_max=t_max)
    t_schedule = None
    rot_schedule = tr_schedule
    tor_schedule = tr_schedule
    print('common t schedule', tr_schedule)

    #初始化结果存储变量
    rmsds_list, obrmsds, centroid_distances_list, failures, skipped, min_cross_distances_list, base_min_cross_distances_list, confidences_list, names_list = [], [], [], 0, 0, [], [], [], []
    run_times, min_self_distances_list, without_rec_overlap_list = [], [], []
    gnina_rmsds_list, gnina_score_list = [], []
    N = args.samples_per_complex
    #names_no_rec_overlap = read_strings_from_txt(f'data/splits/timesplit_test_no_rec_overlap')
    #names_no_rec_overlap = np.load("data/BindingMOAD_2020_processed/test_names_bootstrapping.npy")
    
    #处理没有受体重叠的样本
    names_no_rec_overlap = []
    print('Size of test dataset: ', len(test_dataset))

    #保存采样的复合物（如果需要）
    if args.save_complexes:
        sampled_complexes = {}

    #保存 gnina 评估指标（如果需要）
    if args.save_gnina_metrics:
        # key is complex_name, value is the gnina metrics for all samples
        gnina_metrics = {}

    #遍历测试数据集
    for idx, orig_complex_graph in tqdm(enumerate(test_loader)):
        torch.cuda.empty_cache()

        #检查置信度模型的复合物存在性
        if confidence_model is not None and not (confidence_args.use_original_model_cache or confidence_args.transfer_weights) \
                and orig_complex_graph.name[0] not in confidence_complex_dict.keys():
            skipped += 1
            print(f"HAPPENING | The confidence dataset did not contain {orig_complex_graph.name[0]}. We are skipping this complex.")
            continue
        #初始化采样流程
        success = 0
        bs = args.batch_size
        while 0 >= success > -args.limit_failures:
            try:
                #准备数据列表
                data_list = [copy.deepcopy(orig_complex_graph) for _ in range(N)]
                if args.resample_rdkit:
                    for i, g in enumerate(data_list):
                        g['ligand'].pos = g['ligand'].pos[i]

                # 随机化初始位置,以增加数据的多样性
                randomize_position(data_list, score_model_args.no_torsion, args.no_random or args.no_random_pocket,
                                   score_model_args.tr_sigma_max if not args.pocket_knowledge else args.pocket_tr_max,
                                   args.pocket_knowledge, args.pocket_cutoff,
                                   initial_noise_std_proportion=args.initial_noise_std_proportion,
                                   choose_residue=args.choose_residue)


                #准备可视化数据（如果需要）
                pdb = None
                if args.save_visualisation:
                    visualization_list = []
                    for idx, graph in enumerate(data_list):
                        lig = orig_complex_graph.mol[0]
                        pdb = PDBFile(lig)
                        pdb.add(lig, 0, 0)
                        pdb.add(((orig_complex_graph['ligand'].pos if not args.resample_rdkit else orig_complex_graph['ligand'].pos[idx]) + orig_complex_graph.original_center).detach().cpu(), 1, 0)
                        pdb.add((graph['ligand'].pos + graph.original_center).detach().cpu(), part=1, order=1)
                        visualization_list.append(pdb)
                else:
                    visualization_list = None

                #开始采样
                start_time = time.time()#使用 time.time() 记录采样开始时间。
                if not args.no_model:
                    if confidence_model is not None and not (
                            confidence_args.use_original_model_cache or confidence_args.transfer_weights):
                        confidence_data_list = [copy.deepcopy(confidence_complex_dict[orig_complex_graph.name[0]]) for _ in
                                               range(N)]
                    else:
                        confidence_data_list = None

                    #调用模型进行采样（生成新构象）。
                    data_list, confidence = sampling(data_list=data_list, model=model,
                                                     inference_steps=args.actual_steps if args.actual_steps is not None else args.inference_steps,
                                                     tr_schedule=tr_schedule, rot_schedule=rot_schedule,
                                                     tor_schedule=tor_schedule,
                                                     device=device, t_to_sigma=t_to_sigma, model_args=score_model_args,
                                                     no_random=args.no_random,
                                                     ode=args.ode, visualization_list=visualization_list,
                                                     confidence_model=confidence_model,
                                                     confidence_data_list=confidence_data_list,
                                                     confidence_model_args=confidence_model_args,
                                                     t_schedule=t_schedule,
                                                     batch_size=bs,
                                                     no_final_step_noise=args.no_final_step_noise, pivot=None,
                                                     temp_sampling=[args.temp_sampling_tr, args.temp_sampling_rot, args.temp_sampling_tor],
                                                     temp_psi=[args.temp_psi_tr, args.temp_psi_rot, args.temp_psi_tor],
                                                     temp_sigma_data=[args.temp_sigma_data_tr, args.temp_sigma_data_rot, args.temp_sigma_data_tor])

                #记录运行时间
                run_times.append(time.time() - start_time)
                #检查是否禁用扭转角并更新配体原始位置
                if score_model_args.no_torsion:
                    orig_complex_graph['ligand'].orig_pos = (orig_complex_graph['ligand'].pos.cpu().numpy() + orig_complex_graph.original_center.cpu().numpy())

                #过滤掉氢原子
                filterHs = torch.not_equal(data_list[0]['ligand'].x[:, 0], 0).cpu().numpy()

                #处理多种配体原始位置的情况
                if isinstance(orig_complex_graph['ligand'].orig_pos, list):
                    # Same pair with multiple binding positions
                    # print(f'Number of ground truth poses: {len(orig_complex_graph['ligand'].orig_pos)}')
                    if args.dataset == 'moad' or args.dataset == 'posebusters':
                        orig_ligand_pos = np.array([pos[filterHs] - orig_complex_graph.original_center.cpu().numpy() for pos in orig_complex_graph['ligand'].orig_pos[0]])
                    else:
                        orig_ligand_pos = np.array([pos[filterHs] - orig_complex_graph.original_center.cpu().numpy() for pos in [orig_complex_graph['ligand'].orig_pos[0]]])
                    print('Found ', len(orig_ligand_pos), ' ground truth poses')
                else:
                    print('default path')
                    orig_ligand_pos = np.expand_dims(
                        orig_complex_graph['ligand'].orig_pos[filterHs] - orig_complex_graph.original_center.cpu().numpy(),
                        axis=0)

                #提取预测的配体位置
                ligand_pos = np.asarray(
                        [complex_graph['ligand'].pos.cpu().numpy()[filterHs] for complex_graph in data_list])

                #使用 Gnina 优化预测的配体位置（如果启用）
                # Use gnina to minimize energy for predicted ligands.
                if args.gnina_minimize:
                    print('Running gnina on all predicted ligand positions for energy minimization.')
                    gnina_rmsds, gnina_scores = [], []
                    lig = copy.deepcopy(orig_complex_graph.mol[0])
                    positions = np.asarray([complex_graph['ligand'].pos.cpu().numpy() for complex_graph in data_list])

                    #按置信度重新排序预测的位置（如果存在置信度）
                    conf = confidence
                    if conf is not None and isinstance(confidence_args.rmsd_classification_cutoff, list):
                        conf = conf[:, 0]
                    if conf is not None:
                        conf = conf.cpu().numpy()
                        conf = np.nan_to_num(conf, nan=-1e-6)
                        re_order = np.argsort(conf)[::-1]
                        positions = positions[re_order]

                    #使用 Gnina 对预测的位置进行优化
                    for pos in positions[:args.gnina_poses_to_optimize]:
                        center = orig_complex_graph.original_center.cpu().numpy()
                        gnina_ligand_pos, gnina_mol, gnina_score = get_gnina_poses(args, lig, pos, center, name=orig_complex_graph.name[0],
                                                                                   folder=args.folder, gnina_path=args.gnina_path) # TODO set the right folder

                        #计算 RMSD 并记录 Gnina 结果
                        mol = RemoveAllHs(orig_complex_graph.mol[0])
                        rmsds = []
                        for i in range(len(orig_ligand_pos)):
                            try:
                                rmsd = get_symmetry_rmsd(mol, orig_ligand_pos[i], gnina_ligand_pos, gnina_mol)
                            except Exception as e:
                                print("Using non corrected RMSD because of the error:", e)
                                rmsd = np.sqrt(((gnina_ligand_pos - orig_ligand_pos[i]) ** 2).sum(axis=1).mean(axis=0))
                            rmsds.append(rmsd)
                        rmsds = np.asarray(rmsds)
                        rmsd = np.min(rmsds, axis=0)
                        gnina_rmsds.append(rmsd)
                        gnina_scores.append(gnina_score)

                    #验证结果形状并存储
                    gnina_rmsds = np.asarray(gnina_rmsds)
                    assert gnina_rmsds.shape == (args.gnina_poses_to_optimize,), str(gnina_rmsds.shape) + " " + str(args.gnina_poses_to_optimize)
                    gnina_rmsds_list.append(gnina_rmsds)
                    gnina_scores = np.asarray(gnina_scores)
                    gnina_score_list.append(gnina_scores)

                #移除氢原子并初始化 RMSD 计算
                mol = RemoveAllHs(orig_complex_graph.mol[0])#移除复合物分子中的所有氢原子，简化分子结构，用于 RMSD（均方根偏差）计算。
                rmsds = []
                #计算 RMSD（对称或非对称）
                for i in range(len(orig_ligand_pos)):#遍历所有真实结合位点（orig_ligand_pos）
                    try:
                        rmsd = get_symmetry_rmsd(mol, orig_ligand_pos[i], [l for l in ligand_pos])#计算预测配体位置和真实配体位置之间的对称 RMSD。
                    except Exception as e:
                        print("Using non corrected RMSD because of the error:", e)#如果对称 RMSD 计算失败（捕获异常），使用简单的非对称 RMSD 计算方式
                        rmsd = np.sqrt(((ligand_pos - orig_ligand_pos[i]) ** 2).sum(axis=2).mean(axis=1))#均方根偏差计算公式：$\sqrt{\text{mean}((\text{预测位置} - \text{真实位置})^2)}$。
                    rmsds.append(rmsd)
                rmsds = np.asarray(rmsds)#将每个结合位点的 RMSD 添加到 rmsds 列表
                rmsd = np.min(rmsds, axis=0)#对所有结合位点计算的 RMSD 取最小值（np.min），表示当前预测配体的最佳匹配
                
                #计算质心距离,首先计算每个配体位置的质心坐标（mean(axis=1)）,使用欧几里得距离公式计算质心之间的距离
                centroid_distance = np.min(np.linalg.norm(ligand_pos.mean(axis=1)[None, :] - orig_ligand_pos.mean(axis=1)[:, None], axis=2), axis=0)

                #按置信度重新排序
                if confidence is not None and isinstance(confidence_args.rmsd_classification_cutoff, list):
                    confidence = confidence[:, 0]
                if confidence is not None:
                    confidence = confidence.cpu().numpy()
                    confidence = np.nan_to_num(confidence, nan=-1e-6)
                    re_order = np.argsort(confidence)[::-1]
                    #打印 RMSD、质心距离和置信度信息
                    print(orig_complex_graph['name'], ' rmsd', np.around(rmsd, 1)[re_order], ' centroid distance',
                          np.around(centroid_distance, 1)[re_order], ' confidences ', np.around(confidence, 4)[re_order],
                          (' gnina rmsd ' + str(np.around(gnina_rmsds, 1))) if args.gnina_minimize else '')
                    confidences_list.append(confidence)
                else:
                    print(orig_complex_graph['name'], ' rmsd', np.around(rmsd, 1), ' centroid distance',
                          np.around(centroid_distance, 1))
                centroid_distances_list.append(centroid_distance)

                #计算配体之间的最小自距离
                self_distances = np.linalg.norm(ligand_pos[:, :, None, :] - ligand_pos[:, None, :, :], axis=-1)
                self_distances = np.where(np.eye(self_distances.shape[2]), np.inf, self_distances)
                min_self_distances_list.append(np.min(self_distances, axis=(1, 2)))

                # 保存采样的复合物（如果启用）
                if args.save_complexes:
                    sampled_complexes[orig_complex_graph.name[0]] = data_list

                #保存可视化结果（如果启用）
                if args.save_visualisation:
                    if confidence is not None:
                        for rank, batch_idx in enumerate(re_order):
                            visualization_list[batch_idx].write(
                                f'{args.out_dir}/{data_list[batch_idx]["name"][0]}_{rank + 1}_{rmsd[batch_idx]:.1f}_{(confidence)[batch_idx]:.1f}.pdb')
                    else:
                        for rank, batch_idx in enumerate(np.argsort(rmsd)):
                            visualization_list[batch_idx].write(
                                f'{args.out_dir}/{data_list[batch_idx]["name"][0]}_{rank + 1}_{rmsd[batch_idx]:.1f}.pdb')
                #更新其他统计信息
                without_rec_overlap_list.append(1 if orig_complex_graph.name[0] in names_no_rec_overlap else 0)#记录当前复合物是否属于没有受体重叠的样本。
                names_list.append(orig_complex_graph.name[0])#保存当前复合物名称。
                rmsds_list.append(rmsd)#保存当前复合物的 RMSD。
                success = 1
            #捕获异常并调整批量大小
            except Exception as e:
                print("Failed on", orig_complex_graph["name"], e)
                success -= 1
                if bs > 1:
                    bs = bs // 2

        #检查采样是否成功
        if success != 1:
            #为失败样本填充默认值
            rmsds_list.append(np.zeros(args.samples_per_complex) + 10000)#对于失败的复合物，填充一个全为 10000 的数组，表示无法计算的 RMSD。
            if confidence_model_args is not None:
                confidences_list.append(np.zeros(args.samples_per_complex) - 10000)#如果存在置信度模型，填充一个全为 -10000 的数组，表示无效的置信度分数。
            centroid_distances_list.append(np.zeros(args.samples_per_complex) + 10000)#填充一个全为 10000 的数组，表示质心距离不可用。
            min_self_distances_list.append(np.zeros(args.samples_per_complex) + 10000)#填充一个全为 10000 的数组，表示最小自距离不可用。
            without_rec_overlap_list.append(1 if orig_complex_graph.name[0] in names_no_rec_overlap else 0)#检查复合物是否属于无受体重叠样本，填充对应值。
            names_list.append(orig_complex_graph.name[0])#添加当前复合物的名称到样本列表中。
            failures += 1#增加失败计数器，记录当前复合物采样失败的次数。

    #打印性能统计信息
    print('Performance without hydrogens included in the loss')#标注当前性能统计不包括氢原子损失的影响。
    print(failures, "failures due to exceptions")#打印因异常导致的采样失败数量。
    print(skipped, ' skipped because complex was not in confidence dataset')#打印因置信度数据集中不存在的复合物导致跳过的样本数量。

    #保存采样结果（如果启用）
    if args.save_complexes:
        print("Saving complexes.")
        if args.complexes_save_path is not None:
            with open(os.path.join(args.complexes_save_path, "ligands.pkl"), 'wb') as f:
                pickle.dump(sampled_complexes, f)
    
    #保存 Gnina 评估指标（如果启用）
    if args.save_gnina_metrics:
        with open(f'{args.out_dir}/gnina_metrics.pkl', 'wb') as f:
            pickle.dump(gnina_metrics, f)
        print("Saved gnina metrics")

    #创建字典存储性能指标
    performance_metrics = {}
    #迭代不同的重叠情况
    for overlap in ['', 'no_overlap_']:
        #处理无受体重叠情况
        if 'no_overlap_' == overlap:
            without_rec_overlap = np.array(without_rec_overlap_list, dtype=bool)#转换为布尔数组，表示哪些复合物没有受体重叠。
            if without_rec_overlap.sum() == 0: continue
            rmsds = np.array(rmsds_list)[without_rec_overlap]
            min_self_distances = np.array(min_self_distances_list)[without_rec_overlap]
            centroid_distances = np.array(centroid_distances_list)[without_rec_overlap]
            if args.confidence_model_dir is not None:
                confidences = np.array(confidences_list)[without_rec_overlap]
            else:
                confidences = np.array(confidences_list)
            names = np.array(names_list)[without_rec_overlap]
            gnina_rmsds = np.array(gnina_rmsds_list)[without_rec_overlap] if args.gnina_minimize else None
            gnina_score = np.array(gnina_score_list)[without_rec_overlap] if args.gnina_minimize else None

        else:#处理所有数据（包括重叠和无重叠样本）e
            rmsds = np.array(rmsds_list)
            gnina_rmsds = np.array(gnina_rmsds_list) if args.gnina_minimize else None
            gnina_score = np.array(gnina_score_list) if args.gnina_minimize else None
            min_self_distances = np.array(min_self_distances_list)
            centroid_distances = np.array(centroid_distances_list)
            confidences = np.array(confidences_list)
            names = np.array(names_list)

        #保存性能指标数据
        run_times = np.array(run_times)
        np.save(f'{args.out_dir}/{overlap}min_self_distances.npy', min_self_distances)
        np.save(f'{args.out_dir}/{overlap}rmsds.npy', rmsds)
        np.save(f'{args.out_dir}/{overlap}centroid_distances.npy', centroid_distances)
        np.save(f'{args.out_dir}/{overlap}confidences.npy', confidences)
        np.save(f'{args.out_dir}/{overlap}run_times.npy', run_times)
        np.save(f'{args.out_dir}/{overlap}complex_names.npy', np.array(names))
        np.save(f'{args.out_dir}/{overlap}gnina_rmsds.npy', gnina_rmsds)
        np.save(f'{args.out_dir}/{overlap}gnina_score.npy', gnina_score)

        #更新性能指标字典
        performance_metrics.update({
            #更新性能指标（包括 RMSD、质心距离等）
            f'{overlap}run_times_std': run_times.std().__round__(2),
            f'{overlap}run_times_mean': run_times.mean().__round__(2),
            f'{overlap}mean_rmsd': rmsds.mean(),#计算所有样本的平均 RMSD。
            f'{overlap}rmsds_below_2': (100 * (rmsds < 2).sum() / len(rmsds) / N),
            f'{overlap}rmsds_below_5': (100 * (rmsds < 5).sum() / len(rmsds) / N),#计算 RMSD 小于 2 和 5 的样本的百分比，并根据样本总数（N）进行归一化。
            f'{overlap}rmsds_percentile_25': np.percentile(rmsds, 25).round(2),
            f'{overlap}rmsds_percentile_50': np.percentile(rmsds, 50).round(2),
            f'{overlap}rmsds_percentile_75': np.percentile(rmsds, 75).round(2),#计算 RMSD 的 25、50 和 75 百分位数，并保留两位小数。
            f'{overlap}min_rmsds_below_2': (100 * (np.min(rmsds, axis=1) < 2).sum() / len(rmsds)),
            f'{overlap}min_rmsds_below_5': (100 * (np.min(rmsds, axis=1) < 5).sum() / len(rmsds)),#计算每个样本的最小 RMSD 小于 2 和 5 的百分比。

            f'{overlap}mean_centroid': centroid_distances.mean().__round__(2),#计算质心距离的平均值，保留两位小数。
            f'{overlap}centroid_below_2': (100 * (centroid_distances < 2).sum() / len(centroid_distances) / N).__round__(2),
            f'{overlap}centroid_below_5': (100 * (centroid_distances < 5).sum() / len(centroid_distances) / N).__round__(2),#计算质心距离小于 2 和 5 的样本的百分比，并根据样本总数进行归一化。
            f'{overlap}centroid_percentile_25': np.percentile(centroid_distances, 25).round(2),
            f'{overlap}centroid_percentile_50': np.percentile(centroid_distances, 50).round(2),
            f'{overlap}centroid_percentile_75': np.percentile(centroid_distances, 75).round(2),#计算质心距离的 25、50 和 75 百分位数，并保留两位小数。
        })

        #更新 Gnina 相关性能指标（如果启用 Gnina 最小化）
        if args.gnina_minimize:
            score_ordering = np.argsort(gnina_score, axis=1)[:, ::-1]#根据 gnina_score 对所有样本进行排序。
            filtered_rmsds_gnina = gnina_rmsds[np.arange(gnina_rmsds.shape[0])[:, None], score_ordering][:, 0]#根据排序后的 Gnina 得分，重新排列并获取相应的 RMSD 值。

            performance_metrics.update({
                f'{overlap}gnina_rmsds_below_2': (100 * (gnina_rmsds < 2).sum() / len(gnina_rmsds) / args.gnina_poses_to_optimize) if args.gnina_minimize else None,#计算 Gnina 最小化后的 RMSD 小于 2 和 5 的百分比。
                f'{overlap}gnina_rmsds_below_5': (100 * (gnina_rmsds < 5).sum() / len(gnina_rmsds) / args.gnina_poses_to_optimize) if args.gnina_minimize else None,
                f'{overlap}gnina_min_rmsds_below_2': (100 * (np.min(gnina_rmsds, axis=1) < 2).sum() / len(gnina_rmsds)) if args.gnina_minimize else None,#计算每个样本的最小 RMSD 小于 2 和 5 的百分比。
                f'{overlap}gnina_min_rmsds_below_5': (100 * (np.min(gnina_rmsds, axis=1) < 5).sum() / len(gnina_rmsds)) if args.gnina_minimize else None,
                f'{overlap}gnina_filtered_rmsds_below_2': (100 * (filtered_rmsds_gnina < 2).sum() / len(filtered_rmsds_gnina)).__round__(2),#计算根据 Gnina 得分排序后的 RMSD 小于 2 和 5 的百分比。
                f'{overlap}gnina_filtered_rmsds_below_5': (100 * (filtered_rmsds_gnina < 5).sum() / len(filtered_rmsds_gnina)).__round__(2),
                f'{overlap}gnina_rmsds_percentile_25': np.percentile(gnina_rmsds, 25).round(2),#计算 Gnina RMSD 的 25、50 和 75 百分位数。
                f'{overlap}gnina_rmsds_percentile_50': np.percentile(gnina_rmsds, 50).round(2),
                f'{overlap}gnina_rmsds_percentile_75': np.percentile(gnina_rmsds, 75).round(2),

            })

        #处理 Top 5 样本的性能指标
        if N >= 5:#只有当样本数 N 大于或等于 5 时才会进行以下的计算。
            top5_rmsds = np.min(rmsds[:, :5], axis=1)#对于每个样本，选取与其相关的前 5 个 RMSD 值，并返回最小的 RMSD 值。np.min(rmsds[:, :5], axis=1)：从 RMSD 数组的前 5 列中获取每行的最小值。
            top5_centroid_distances = centroid_distances[
                                          np.arange(rmsds.shape[0])[:, None], np.argsort(rmsds[:, :5], axis=1)][:, 0]#根据 RMSD 排序，选取前 5 个最小的 RMSD 对应的质心距离。np.argsort(rmsds[:, :5], axis=1) 对每个样本的前 5 个 RMSD 进行排序，并通过 np.arange(rmsds.shape[0])[:, None] 创建一个索引矩阵来提取相应的质心距离。
            top5_min_self_distances = min_self_distances[
                                          np.arange(rmsds.shape[0])[:, None], np.argsort(rmsds[:, :5], axis=1)][:, 0]#同样的方法，选取前 5 个最小 RMSD 对应的最小自交距离。
            performance_metrics.update({
                f'{overlap}top5_self_intersect_fraction': (
                            100 * (top5_min_self_distances < 0.4).sum() / len(top5_min_self_distances)).__round__(2),#计算前 5 个最小 RMSD 对应的最小自交距离小于 0.4 的样本占比（按百分比计算）。
                f'{overlap}top5_rmsds_below_2': (100 * (top5_rmsds < 2).sum() / len(top5_rmsds)).__round__(2),#分别计算前 5 个最小 RMSD 小于 2 和小于 5 的样本的百分比。
                f'{overlap}top5_rmsds_below_5': (100 * (top5_rmsds < 5).sum() / len(top5_rmsds)).__round__(2),
                f'{overlap}top5_rmsds_percentile_25': np.percentile(top5_rmsds, 25).round(2),#计算前 5 个最小 RMSD 的 25、50 和 75 百分位数。
                f'{overlap}top5_rmsds_percentile_50': np.percentile(top5_rmsds, 50).round(2),
                f'{overlap}top5_rmsds_percentile_75': np.percentile(top5_rmsds, 75).round(2),

                f'{overlap}top5_centroid_below_2': (
                            100 * (top5_centroid_distances < 2).sum() / len(top5_centroid_distances)).__round__(2),#计算前 5 个最小 RMSD 对应的质心距离小于 2 和小于 5 的样本占比。
                f'{overlap}top5_centroid_below_5': (
                            100 * (top5_centroid_distances < 5).sum() / len(top5_centroid_distances)).__round__(2),
                f'{overlap}top5_centroid_percentile_25': np.percentile(top5_centroid_distances, 25).round(2),#计算前 5 个最小 RMSD 对应的质心距离的 25、50 和 75 百分位数。
                f'{overlap}top5_centroid_percentile_50': np.percentile(top5_centroid_distances, 50).round(2),
                f'{overlap}top5_centroid_percentile_75': np.percentile(top5_centroid_distances, 75).round(2),
            })

        #处理 Top 10 样本的性能指标
        if N >= 10:#只有当样本数 N 大于或等于 10 时才会进行以下的计算。
            top10_rmsds = np.min(rmsds[:, :10], axis=1)#对于每个样本，选取与其相关的前 10 个 RMSD 值，并返回最小的 RMSD 值。np.min(rmsds[:, :10], axis=1)：从 RMSD 数组的前 10 列中获取每行的最小值。
            top10_centroid_distances = centroid_distances[
                                           np.arange(rmsds.shape[0])[:, None], np.argsort(rmsds[:, :10], axis=1)][:, 0]#根据 RMSD 排序，选取前 10 个最小的 RMSD 对应的质心距离。np.argsort(rmsds[:, :10], axis=1) 对每个样本的前 10 个 RMSD 进行排序，并通过 np.arange(rmsds.shape[0])[:, None] 创建一个索引矩阵来提取相应的质心距离。
            top10_min_self_distances = min_self_distances[
                                           np.arange(rmsds.shape[0])[:, None], np.argsort(rmsds[:, :10], axis=1)][:, 0]#同样的方法，选取前 10 个最小 RMSD 对应的最小自交距离。
            performance_metrics.update({
                f'{overlap}top10_self_intersect_fraction': (
                            100 * (top10_min_self_distances < 0.4).sum() / len(top10_min_self_distances)).__round__(2),#计算前 10 个最小 RMSD 对应的最小自交距离小于 0.4 的样本占比（按百分比计算）。
                f'{overlap}top10_rmsds_below_2': (100 * (top10_rmsds < 2).sum() / len(top10_rmsds)).__round__(2),#分别计算前 10 个最小 RMSD 小于 2 和小于 5 的样本的百分比。
                f'{overlap}top10_rmsds_below_5': (100 * (top10_rmsds < 5).sum() / len(top10_rmsds)).__round__(2),
                f'{overlap}top10_rmsds_percentile_25': np.percentile(top10_rmsds, 25).round(2),#计算前 10 个最小 RMSD 的 25、50 和 75 百分位数。
                f'{overlap}top10_rmsds_percentile_50': np.percentile(top10_rmsds, 50).round(2),
                f'{overlap}top10_rmsds_percentile_75': np.percentile(top10_rmsds, 75).round(2),

                f'{overlap}top10_centroid_below_2': (
                            100 * (top10_centroid_distances < 2).sum() / len(top10_centroid_distances)).__round__(2),#计算前 10 个最小 RMSD 对应的质心距离小于 2 和小于 5 的样本占比。
                f'{overlap}top10_centroid_below_5': (
                            100 * (top10_centroid_distances < 5).sum() / len(top10_centroid_distances)).__round__(2),
                f'{overlap}top10_centroid_percentile_25': np.percentile(top10_centroid_distances, 25).round(2),#计算前 10 个最小 RMSD 对应的质心距离的 25、50 和 75 百分位数。
                f'{overlap}top10_centroid_percentile_50': np.percentile(top10_centroid_distances, 50).round(2),
                f'{overlap}top10_centroid_percentile_75': np.percentile(top10_centroid_distances, 75).round(2),
            })

        #Confidence 模型检查和排序
        if confidence_model is not None:
            confidence_ordering = np.argsort(confidences, axis=1)[:, ::-1]#通过 np.argsort(confidences, axis=1) 对置信度 confidences 按行进行排序，得到每个样本的排序索引。[:, ::-1] 通过反转排序顺序，使得排序结果从高到低，表示置信度从高到低的索引顺序。

            #根据排序索引过滤 RMSD、质心距离和最小自交距离
            filtered_rmsds = rmsds[np.arange(rmsds.shape[0])[:, None], confidence_ordering][:, 0]#根据排序后的 confidence_ordering 对 rmsds 数组进行排序，得到根据置信度排序后的 RMSD 值。
            filtered_centroid_distances = centroid_distances[np.arange(rmsds.shape[0])[:, None], confidence_ordering][:, 0]#根据排序后的 confidence_ordering 对 centroid_distances 数组进行排序，得到根据置信度排序后的质心距离。
            filtered_min_self_distances = min_self_distances[np.arange(rmsds.shape[0])[:, None], confidence_ordering][:, 0]#根据排序后的 confidence_ordering 对 min_self_distances 数组进行排序，得到根据置信度排序后的最小自交距离。
            #更新性能指标（过滤后的数据）
            performance_metrics.update({
                f'{overlap}filtered_self_intersect_fraction': (
                            100 * (filtered_min_self_distances < 0.4).sum() / len(filtered_min_self_distances)).__round__(
                    2),#计算过滤后样本中，最小自交距离小于 0.4 的样本占比（按百分比计算）。
                f'{overlap}filtered_rmsds_below_2': (100 * (filtered_rmsds < 2).sum() / len(filtered_rmsds)).__round__(2),#分别计算过滤后 RMSD 小于 2 和小于 5 的样本占比（按百分比计算）。
                f'{overlap}filtered_rmsds_below_5': (100 * (filtered_rmsds < 5).sum() / len(filtered_rmsds)).__round__(2),
                f'{overlap}filtered_rmsds_percentile_25': np.percentile(filtered_rmsds, 25).round(2),#分别计算过滤后的 RMSD 的 25%、50% 和 75% 百分位数。
                f'{overlap}filtered_rmsds_percentile_50': np.percentile(filtered_rmsds, 50).round(2),
                f'{overlap}filtered_rmsds_percentile_75': np.percentile(filtered_rmsds, 75).round(2),

                f'{overlap}filtered_centroid_below_2': (
                            100 * (filtered_centroid_distances < 2).sum() / len(filtered_centroid_distances)).__round__(2),#分别计算过滤后质心距离小于 2 和小于 5 的样本占比（按百分比计算）。
                f'{overlap}filtered_centroid_below_5': (
                            100 * (filtered_centroid_distances < 5).sum() / len(filtered_centroid_distances)).__round__(2),
                f'{overlap}filtered_centroid_percentile_25': np.percentile(filtered_centroid_distances, 25).round(2),#分别计算过滤后的质心距离的 25%、50% 和 75% 百分位数。
                f'{overlap}filtered_centroid_percentile_50': np.percentile(filtered_centroid_distances, 50).round(2),
                f'{overlap}filtered_centroid_percentile_75': np.percentile(filtered_centroid_distances, 75).round(2),
            })

            # 对 Top 5 样本进行过滤后性能评估
            if N >= 5:
                top5_filtered_rmsds = np.min(rmsds[np.arange(rmsds.shape[0])[:, None], confidence_ordering][:, :5], axis=1)#从前 5 个过滤后的 RMSD 中选取最小值。
                top5_filtered_centroid_distances = \
                centroid_distances[np.arange(rmsds.shape[0])[:, None], confidence_ordering][:, :5][
                    np.arange(rmsds.shape[0])[:, None], np.argsort(
                        rmsds[np.arange(rmsds.shape[0])[:, None], confidence_ordering][:, :5], axis=1)][:, 0]#从前 5 个过滤后的 RMSD 对应的质心距离中选取最小值。
                top5_filtered_min_self_distances = \
                min_self_distances[np.arange(rmsds.shape[0])[:, None], confidence_ordering][:, :5][
                    np.arange(rmsds.shape[0])[:, None], np.argsort(
                        rmsds[np.arange(rmsds.shape[0])[:, None], confidence_ordering][:, :5], axis=1)][:, 0]#从前 5 个过滤后的 RMSD 对应的最小自交距离中选取最小值。
                #更新 Top 5 样本的过滤后性能指标
                performance_metrics.update({
                    f'{overlap}top5_filtered_rmsds_below_2': (
                                100 * (top5_filtered_rmsds < 2).sum() / len(top5_filtered_rmsds)).__round__(2),#计算过滤后前 5 个最小 RMSD 小于 2 和小于 5 的样本占比。
                    f'{overlap}top5_filtered_rmsds_below_5': (
                                100 * (top5_filtered_rmsds < 5).sum() / len(top5_filtered_rmsds)).__round__(2),
                    f'{overlap}top5_filtered_rmsds_percentile_25': np.percentile(top5_filtered_rmsds, 25).round(2),#计算过滤后前 5 个最小 RMSD 的 25%、50% 和 75% 百分位数。
                    f'{overlap}top5_filtered_rmsds_percentile_50': np.percentile(top5_filtered_rmsds, 50).round(2),
                    f'{overlap}top5_filtered_rmsds_percentile_75': np.percentile(top5_filtered_rmsds, 75).round(2),

                    f'{overlap}top5_filtered_centroid_below_2': (100 * (top5_filtered_centroid_distances < 2).sum() / len(
                        top5_filtered_centroid_distances)).__round__(2),#计算过滤后前 5 个最小 RMSD 对应的质心距离小于 2 和小于 5 的样本占比。
                    f'{overlap}top5_filtered_centroid_below_5': (100 * (top5_filtered_centroid_distances < 5).sum() / len(
                        top5_filtered_centroid_distances)).__round__(2),
                    f'{overlap}top5_filtered_centroid_percentile_25': np.percentile(top5_filtered_centroid_distances,
                                                                                    25).round(2),#计算过滤后前 5 个最小 RMSD 对应的质心距离的 25%、50% 和 75% 百分位数。
                    f'{overlap}top5_filtered_centroid_percentile_50': np.percentile(top5_filtered_centroid_distances,
                                                                                    50).round(2),
                    f'{overlap}top5_filtered_centroid_percentile_75': np.percentile(top5_filtered_centroid_distances,
                                                                                    75).round(2),
                })
            #对 Top 10 样本进行过滤后性能评估
            if N >= 10:
                top10_filtered_rmsds = np.min(rmsds[np.arange(rmsds.shape[0])[:, None], confidence_ordering][:, :10],
                                              axis=1)#从前 10 个过滤后的 RMSD 中选取最小值。
                top10_filtered_centroid_distances = \
                centroid_distances[np.arange(rmsds.shape[0])[:, None], confidence_ordering][:, :10][
                    np.arange(rmsds.shape[0])[:, None], np.argsort(
                        rmsds[np.arange(rmsds.shape[0])[:, None], confidence_ordering][:, :10], axis=1)][:, 0]#从前 10 个过滤后的 RMSD 对应的质心距离中选取最小值。
                top10_filtered_min_self_distances = \
                min_self_distances[np.arange(rmsds.shape[0])[:, None], confidence_ordering][:, :10][
                    np.arange(rmsds.shape[0])[:, None], np.argsort(
                        rmsds[np.arange(rmsds.shape[0])[:, None], confidence_ordering][:, :10], axis=1)][:, 0]#从前 10 个过滤后的 RMSD 对应的最小自交距离中选取最小值。
                #更新 Top 10 样本的过滤后性能指标
                performance_metrics.update({
                    f'{overlap}top10_filtered_rmsds_below_2': (
                                100 * (top10_filtered_rmsds < 2).sum() / len(top10_filtered_rmsds)).__round__(2),#计算过滤后前 10 个最小 RMSD 小于 2 和小于 5 的样本占比。
                    f'{overlap}top10_filtered_rmsds_below_5': (
                                100 * (top10_filtered_rmsds < 5).sum() / len(top10_filtered_rmsds)).__round__(2),
                    f'{overlap}top10_filtered_rmsds_percentile_25': np.percentile(top10_filtered_rmsds, 25).round(2),#计算过滤后前 10 个最小 RMSD 的 25%、50% 和 75% 百分位数。
                    f'{overlap}top10_filtered_rmsds_percentile_50': np.percentile(top10_filtered_rmsds, 50).round(2),
                    f'{overlap}top10_filtered_rmsds_percentile_75': np.percentile(top10_filtered_rmsds, 75).round(2),

                    f'{overlap}top10_filtered_centroid_below_2': (100 * (top10_filtered_centroid_distances < 2).sum() / len(
                        top10_filtered_centroid_distances)).__round__(2),#计算过滤后前 10 个最小 RMSD 对应的质心距离小于 2 和小于 5 的样本占比。
                    f'{overlap}top10_filtered_centroid_below_5': (100 * (top10_filtered_centroid_distances < 5).sum() / len(
                        top10_filtered_centroid_distances)).__round__(2),
                    f'{overlap}top10_filtered_centroid_percentile_25': np.percentile(top10_filtered_centroid_distances,
                                                                                     25).round(2),#计算过滤后前 10 个最小 RMSD 对应的质心距离的 25%、50% 和 75% 百分位数。
                    f'{overlap}top10_filtered_centroid_percentile_50': np.percentile(top10_filtered_centroid_distances,
                                                                                     50).round(2),
                    f'{overlap}top10_filtered_centroid_percentile_75': np.percentile(top10_filtered_centroid_distances,
                                                                                     75).round(2),
                })

    #打印性能指标
    for k in performance_metrics:
        print(k, performance_metrics[k])

    #判断是否启用 WandB（Weights & Biases）
    if args.wandb:#检查 args 参数中是否启用了 WandB（即 wandb 是否为 True）。
        wandb.log(performance_metrics)#如果启用了 WandB，则将 performance_metrics 字典的内容记录到 WandB 中，便于后续的性能监控和可视化。
        #创建直方图度量列表
        histogram_metrics_list = [('rmsd', rmsds[:, 0]),#RMSD（均方根偏差）的第一个维度（即所有样本的 RMSD 值）。
                                  ('centroid_distance', centroid_distances[:, 0]),#质心距离的第一个维度（即所有样本的质心距离）。
                                  ('mean_rmsd', rmsds.mean(axis=1)),#每个样本的 RMSD 的均值。
                                  ('mean_centroid_distance', centroid_distances.mean(axis=1))]#每个样本的质心距离的均值。
        #条件添加更多度量（基于样本数量 N）
        if N >= 5:
            histogram_metrics_list.append(('top5_rmsds', top5_rmsds))#Top 5 样本的 RMSD 值。
            histogram_metrics_list.append(('top5_centroid_distances', top5_centroid_distances))#Top 5 样本的质心距离。
        if N >= 10:#如果样本数量 N 大于等于 10，则将 Top 10 样本的 RMSD 和质心距离添加到 histogram_metrics_list 列表中。
            histogram_metrics_list.append(('top10_rmsds', top10_rmsds))#Top 10 样本的 RMSD 值。
            histogram_metrics_list.append(('top10_centroid_distances', top10_centroid_distances))#Top 10 样本的质心距离。
        #检查并添加反向过滤后的度量（如果置信度模型存在）
        if confidence_model is not None:#检查是否存在置信度模型（confidence_model 是否为 None）。
            histogram_metrics_list.append(('reverse_filtered_rmsds', reverse_filtered_rmsds))#如果有置信度模型，则将经过反向过滤后的 RMSD 和质心距离添加到度量列表中。
            histogram_metrics_list.append(('reverse_filtered_centroid_distances', reverse_filtered_centroid_distances))
            histogram_metrics_list.append(('filtered_rmsd', filtered_rmsds))#还将经过过滤后的 RMSD 和质心距离添加到度量列表中。
            histogram_metrics_list.append(('filtered_centroid_distance', filtered_centroid_distances))
            #条件添加更多反向过滤后的度量（基于样本数量 N）
            if N >= 5:
                histogram_metrics_list.append(('top5_filtered_rmsds', top5_filtered_rmsds))
                histogram_metrics_list.append(('top5_filtered_centroid_distances', top5_filtered_centroid_distances))
                histogram_metrics_list.append(('top5_reverse_filtered_rmsds', top5_reverse_filtered_rmsds))
                histogram_metrics_list.append(
                    ('top5_reverse_filtered_centroid_distances', top5_reverse_filtered_centroid_distances))
            if N >= 10:
                histogram_metrics_list.append(('top10_filtered_rmsds', top10_filtered_rmsds))
                histogram_metrics_list.append(('top10_filtered_centroid_distances', top10_filtered_centroid_distances))
                histogram_metrics_list.append(('top10_reverse_filtered_rmsds', top10_reverse_filtered_rmsds))
                histogram_metrics_list.append(
                    ('top10_reverse_filtered_centroid_distances', top10_reverse_filtered_centroid_distances))

import copy
import os
import torch
from argparse import ArgumentParser, Namespace, FileType
from functools import partial
import numpy as np
import pandas as pd
from rdkit import RDLogger
from torch_geometric.loader import DataLoader
from rdkit.Chem import RemoveAllHs

from datasets.process_mols import write_mol_with_coords
from utils.download import download_and_extract
from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl, get_t_schedule
from utils.inference_utils import InferenceDataset, set_nones
from utils.sampling import randomize_position, sampling
from utils.utils import get_model
from utils.visualise import PDBFile
from tqdm import tqdm
# 禁用指定模式的日志记录
RDLogger.DisableLog('rdApp.*')

import yaml
# 创建ArgumentParser对象，用于解析命令行参数
parser = ArgumentParser()
# 定义命令行参数及其类型、默认值和描述信息

# 配置文件路径，默认是 'default_inference_args.yaml'
parser.add_argument('--config', type=FileType(mode='r'), default='default_inference_args.yaml')
# 输入文件的路径（CSV文件），用于指定输入数据
parser.add_argument('--protein_ligand_csv', type=str, default=None, help='Path to a .csv file specifying the input as described in the README. If this is not None, it will be used instead of the --protein_path, --protein_sequence and --ligand parameters')
# 复合物的名称，用于保存复合物结果
parser.add_argument('--complex_name', type=str, default=None, help='Name that the complex will be saved with')
# 蛋白质文件路径
parser.add_argument('--protein_path', type=str, default=None, help='Path to the protein file')
# 蛋白质序列，如果没有提供蛋白质文件，则使用此序列
parser.add_argument('--protein_sequence', type=str, default=None, help='Sequence of the protein for ESMFold, this is ignored if --protein_path is not None')
# 配体描述，通常是SMILES字符串，或者是能够被rdkit读取的分子文件的路径
parser.add_argument('--ligand_description', type=str, default='CCCCC(NC(=O)CCC(=O)O)P(=O)(O)OC1=CC=CC=C1', help='Either a SMILES string or the path to a molecule file that rdkit can read')

# 输出目录，结果将保存到该目录
parser.add_argument('--out_dir', type=str, default='results/user_inference', help='Directory where the outputs will be written to')
# 是否保存反向扩散的可视化PDB文件
parser.add_argument('--save_visualisation', action='store_true', default=False, help='Save a pdb file with all of the steps of the reverse diffusion')
# 每个复合物生成的样本数量
parser.add_argument('--samples_per_complex', type=int, default=10, help='Number of samples to generate')

# 训练好的评分模型和超参数文件夹路径
parser.add_argument('--model_dir', type=str, default=None, help='Path to folder with trained score model and hyperparameters')
# 用于评分模型的检查点文件
parser.add_argument('--ckpt', type=str, default='best_ema_inference_epoch_model.pt', help='Checkpoint to use for the score model')
# 训练好的置信度模型和超参数文件夹路径
parser.add_argument('--confidence_model_dir', type=str, default=None, help='Path to folder with trained confidence model and hyperparameters')
# 用于置信度模型的检查点文件
parser.add_argument('--confidence_ckpt', type=str, default='best_model.pt', help='Checkpoint to use for the confidence model')

# 批量大小
parser.add_argument('--batch_size', type=int, default=10, help='')
# 是否在反向扩散的最终步骤使用噪声
parser.add_argument('--no_final_step_noise', action='store_true', default=True, help='Use no noise in the final step of the reverse diffusion')
# 去噪步骤的数量
parser.add_argument('--inference_steps', type=int, default=20, help='Number of denoising steps')
# 实际执行的去噪步骤数
parser.add_argument('--actual_steps', type=int, default=None, help='Number of denoising steps that are actually performed')

# 使用旧的评分模型
parser.add_argument('--old_score_model', action='store_true', default=False, help='')
# 使用旧的置信度模型
parser.add_argument('--old_confidence_model', action='store_true', default=True, help='')
# 初始噪声标准差的比例
parser.add_argument('--initial_noise_std_proportion', type=float, default=-1.0, help='Initial noise std proportion')
# 是否选择特定的残基
parser.add_argument('--choose_residue', action='store_true', default=False, help='')

# 各种温度采样和数据处理参数
parser.add_argument('--temp_sampling_tr', type=float, default=1.0)
parser.add_argument('--temp_psi_tr', type=float, default=0.0)
parser.add_argument('--temp_sigma_data_tr', type=float, default=0.5)
parser.add_argument('--temp_sampling_rot', type=float, default=1.0)
parser.add_argument('--temp_psi_rot', type=float, default=0.0)
parser.add_argument('--temp_sigma_data_rot', type=float, default=0.5)
parser.add_argument('--temp_sampling_tor', type=float, default=1.0)
parser.add_argument('--temp_psi_tor', type=float, default=0.0)
parser.add_argument('--temp_sigma_data_tor', type=float, default=0.5)

# 是否启用gnina最小化
parser.add_argument('--gnina_minimize', action='store_true', default=False, help='')
# gnina程序的路径
parser.add_argument('--gnina_path', type=str, default='gnina', help='')
# gnina日志文件的路径，用于重定向gnina子进程的标准输出
parser.add_argument('--gnina_log_file', type=str, default='gnina_log.txt', help='')  # To redirect gnina subprocesses stdouts from the terminal window
# 是否启用gnina的完整对接
parser.add_argument('--gnina_full_dock', action='store_true', default=False, help='')
# gnina添加的自动框大小
parser.add_argument('--gnina_autobox_add', type=float, default=4.0)
# gnina优化的pose数量
parser.add_argument('--gnina_poses_to_optimize', type=int, default=1)

# 解析命令行参数
args = parser.parse_args()

# 获取存储库的URL
REPOSITORY_URL = os.environ.get("REPOSITORY_URL", "https://github.com/gcorso/DiffDock")

# 如果提供了配置文件（args.config），则加载该配置文件并更新命令行参数
if args.config:
    # 使用yaml库加载配置文件，Loader=yaml.FullLoader指定解析方式
    config_dict = yaml.load(args.config, Loader=yaml.FullLoader)
    # 获取命令行参数的字典形式（args的属性）
    arg_dict = args.__dict__
    # 遍历配置文件中的每一个键值对
    for key, value in config_dict.items():
         # 如果值是列表类型，则将配置中的所有元素追加到相应的命令行参数中
        if isinstance(value, list):
            for v in value:
                arg_dict[key].append(v)
        else:
            # 如果值是单个元素，则直接更新命令行参数
            arg_dict[key] = value
# 如果指定的模型文件夹（args.model_dir）不存在，则尝试从远程下载模型
# Download models if they don't exist locally
if not os.path.exists(args.model_dir):
    print(f"Models not found. Downloading")
    # TODO: 移除Dropbox URL，等模型上传到GitHub发布版本后使用GitHub链接
    # TODO Remove the dropbox URL once the models are uploaded to GitHub release
    remote_urls = [f"{REPOSITORY_URL}/releases/latest/download/diffdock_models.zip",
                   "https://www.dropbox.com/scl/fi/drg90rst8uhd2633tyou0/diffdock_models.zip?rlkey=afzq4kuqor2jb8adah41ro2lz&dl=1"]
     # 标志位，标识模型是否成功下载
    downloaded_successfully = False
     # 遍历多个远程下载链接，尝试下载模型文件
    for remote_url in remote_urls:
        try:
            print(f"Attempting download from {remote_url}")
            # 调用自定义的下载并解压函数（download_and_extract），将文件下载并解压到指定目录
            files_downloaded = download_and_extract(remote_url, os.path.dirname(args.model_dir))
            # 如果没有成功下载任何文件，输出失败信息并尝试下一个链接
            if not files_downloaded:
                print(f"Download from {remote_url} failed.")
                continue
            # 下载成功，打印成功信息
            print(f"Downloaded and extracted {len(files_downloaded)} files from {remote_url}")
            # 设置下载成功标志
            downloaded_successfully = True
            # Once we have downloaded the models, we can break the loop
            break
        except Exception as e:
            pass

    # 如果所有下载链接都尝试失败，抛出异常
    if not downloaded_successfully:
        raise Exception(f"Models not found locally and failed to download them from {remote_urls}")
# 创建输出目录（args.out_dir），如果该目录不存在
os.makedirs(args.out_dir, exist_ok=True)
# 加载评分模型的超参数（model_parameters.yml）并创建一个Namespace对象
with open(f'{args.model_dir}/model_parameters.yml') as f:
    score_model_args = Namespace(**yaml.full_load(f))
# 如果指定了置信度模型路径，则加载该模型的超参数
if args.confidence_model_dir is not None:
    with open(f'{args.confidence_model_dir}/model_parameters.yml') as f:
        confidence_args = Namespace(**yaml.full_load(f))

# 确定使用的设备（GPU或CPU），如果有可用的CUDA设备，则使用GPU，否则使用CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Docking will run on {device}")

# 如果指定了蛋白质-配体CSV文件（args.protein_ligand_csv），则从CSV中加载数据
if args.protein_ligand_csv is not None:
    # 使用pandas读取CSV文件
    df = pd.read_csv(args.protein_ligand_csv)
    # 获取复合物名称、蛋白质路径、蛋白质序列和配体描述的列表，处理缺失值（使用set_nones函数填充None）
    complex_name_list = set_nones(df['complex_name'].tolist())
    protein_path_list = set_nones(df['protein_path'].tolist())
    protein_sequence_list = set_nones(df['protein_sequence'].tolist())
    ligand_description_list = set_nones(df['ligand_description'].tolist())
else:
    # 如果没有提供CSV文件，则使用命令行参数中的单一输入数据（complex_name, protein_path, protein_sequence, ligand_description）
    complex_name_list = [args.complex_name if args.complex_name else f"complex_0"]
    protein_path_list = [args.protein_path]
    protein_sequence_list = [args.protein_sequence]
    ligand_description_list = [args.ligand_description]
# 如果复合物名称为空（None），则为每个复合物生成一个唯一的名称
complex_name_list = [name if name is not None else f"complex_{i}" for i, name in enumerate(complex_name_list)]
# 为每个复合物创建一个目录
for name in complex_name_list:
    write_dir = f'{args.out_dir}/{name}'
    # 如果目录不存在，创建该目录
    os.makedirs(write_dir, exist_ok=True)

# 将复合物数据预处理成几何图
# preprocessing of complexes into geometric graphs
test_dataset = InferenceDataset(out_dir=args.out_dir,# 输出目录
                                complex_names=complex_name_list,# 复合物名称列表
                                protein_files=protein_path_list,# 蛋白质文件路径列表
                                ligand_descriptions=ligand_description_list, # 配体描述列表
                                protein_sequences=protein_sequence_list,# 蛋白质序列列表
                                lm_embeddings=True,# 是否使用语言模型的嵌入
                                receptor_radius=score_model_args.receptor_radius, # 配体受体半径（从评分模型的超参数中获取）
                                remove_hs=score_model_args.remove_hs,# 是否移除氢原子
                                c_alpha_max_neighbors=score_model_args.c_alpha_max_neighbors,# 最大邻居数（Cα原子）
                                all_atoms=score_model_args.all_atoms, # 是否考虑所有原子
                                atom_radius=score_model_args.atom_radius,# 原子半径
                                atom_max_neighbors=score_model_args.atom_max_neighbors,# 原子最大邻居数
                                knn_only_graph=False if not hasattr(score_model_args, 'not_knn_only_graph') else not score_model_args.not_knn_only_graph)# 是否使用KNN图
# 使用DataLoader来加载数据集，批量大小为1，顺序加载数据
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

# 如果指定了置信度模型路径且该模型不使用原始模型缓存，加载置信度模型的数据
if args.confidence_model_dir is not None and not confidence_args.use_original_model_cache:
    print('HAPPENING | confidence model uses different type of graphs than the score model. '
          'Loading (or creating if not existing) the data for the confidence model now.')
    # 为置信度模型创建一个新的数据集（与评分模型的数据集结构可能不同）
    confidence_test_dataset = \
        InferenceDataset(out_dir=args.out_dir, 
                         complex_names=complex_name_list, 
                         protein_files=protein_path_list,
                         ligand_descriptions=ligand_description_list, 
                         protein_sequences=protein_sequence_list,
                         lm_embeddings=True,
                         receptor_radius=confidence_args.receptor_radius, 
                         remove_hs=confidence_args.remove_hs,
                         c_alpha_max_neighbors=confidence_args.c_alpha_max_neighbors,
                         all_atoms=confidence_args.all_atoms, 
                         atom_radius=confidence_args.atom_radius,
                         atom_max_neighbors=confidence_args.atom_max_neighbors,
                         precomputed_lm_embeddings=test_dataset.lm_embeddings,
                         knn_only_graph=False if not hasattr(score_model_args, 'not_knn_only_graph') else not score_model_args.not_knn_only_graph)
else:
     # 如果没有指定置信度模型或模型使用原始缓存，则置信度数据集为空
    confidence_test_dataset = None

# 定义一个部分函数，将score_model_args传递给t_to_sigma_compl函数
t_to_sigma = partial(t_to_sigma_compl, args=score_model_args)

# 获取评分模型，使用指定的设备（GPU或CPU）
model = get_model(score_model_args, device, t_to_sigma=t_to_sigma, no_parallel=True, old=args.old_score_model)
# 加载评分模型的检查点
state_dict = torch.load(f'{args.model_dir}/{args.ckpt}', map_location=torch.device('cpu'))
model.load_state_dict(state_dict, strict=True) # 加载模型参数
model = model.to(device)# 将模型移到指定设备（GPU或CPU）
model.eval()# 设置模型为评估模式

# 如果指定了置信度模型路径，则加载置信度模型
if args.confidence_model_dir is not None:
    confidence_model = get_model(confidence_args, device, t_to_sigma=t_to_sigma, no_parallel=True,
                                 confidence_mode=True, old=args.old_confidence_model)
    # 加载置信度模型的检查点
    state_dict = torch.load(f'{args.confidence_model_dir}/{args.confidence_ckpt}', map_location=torch.device('cpu'))
    confidence_model.load_state_dict(state_dict, strict=True)# 加载模型参数
    confidence_model = confidence_model.to(device)# 将模型移到指定设备
    confidence_model.eval()# 设置模型为评估模式
else:
    # 如果没有提供置信度模型路径，则将置信度模型和参数设置为None
    confidence_model = None
    confidence_args = None

# 获取去噪调度计划（用于控制去噪过程中的噪声调度）
tr_schedule = get_t_schedule(inference_steps=args.inference_steps, sigma_schedule='expbeta')

# 初始化失败和跳过计数器
failures, skipped = 0, 0
 # 每个复合物生成的样本数
N = args.samples_per_complex
print('Size of test dataset: ', len(test_dataset))# 打印测试数据集的大小
# 遍历测试数据集
for idx, orig_complex_graph in tqdm(enumerate(test_loader)):
    # 如果原始复合物图成功加载标记为 False，则跳过此复合物
    if not orig_complex_graph.success[0]:
        skipped += 1
        print(f"HAPPENING | The test dataset did not contain {test_dataset.complex_names[idx]} for {test_dataset.ligand_descriptions[idx]} and {test_dataset.protein_files[idx]}. We are skipping this complex.")
        continue# 跳过当前复合物
    try:
         # 如果提供了置信度模型数据集，加载置信度数据
        if confidence_test_dataset is not None:
            confidence_complex_graph = confidence_test_dataset[idx]
            if not confidence_complex_graph.success:
                skipped += 1
                print(f"HAPPENING | The confidence dataset did not contain {orig_complex_graph.name}. We are skipping this complex.")
                continue# 跳过当前复合物
             # 复制confidence_complex_graph，生成 N 个相同的数据
            confidence_data_list = [copy.deepcopy(confidence_complex_graph) for _ in range(N)]
        else:
            confidence_data_list = None# 如果没有置信度模型，设为 None
        # 复制原始复合物图，生成 N 个相同的数据
        data_list = [copy.deepcopy(orig_complex_graph) for _ in range(N)]
        # 随机化复合物位置，准备生成样本
        randomize_position(data_list, score_model_args.no_torsion, False, score_model_args.tr_sigma_max,
                           initial_noise_std_proportion=args.initial_noise_std_proportion,
                           choose_residue=args.choose_residue)

        lig = orig_complex_graph.mol[0] # 获取原始复合物的配体

        # 初始化可视化保存列表
        # initialize visualisation
        pdb = None
        if args.save_visualisation:# 如果需要保存可视化
            visualization_list = []
            # 为每个图添加可视化数据
            for graph in data_list:
                pdb = PDBFile(lig)
                pdb.add(lig, 0, 0)
                pdb.add((orig_complex_graph['ligand'].pos + orig_complex_graph.original_center).detach().cpu(), 1, 0)
                pdb.add((graph['ligand'].pos + graph.original_center).detach().cpu(), part=1, order=1)
                visualization_list.append(pdb)
        else:
            visualization_list = None# 如果不需要保存可视化，设为 None

        # 执行逆向扩散过程
        # run reverse diffusion
        data_list, confidence = sampling(data_list=data_list, model=model,
                                         inference_steps=args.actual_steps if args.actual_steps is not None else args.inference_steps,
                                         tr_schedule=tr_schedule, rot_schedule=tr_schedule, tor_schedule=tr_schedule,
                                         device=device, t_to_sigma=t_to_sigma, model_args=score_model_args,
                                         visualization_list=visualization_list, confidence_model=confidence_model,
                                         confidence_data_list=confidence_data_list, confidence_model_args=confidence_args,
                                         batch_size=args.batch_size, no_final_step_noise=args.no_final_step_noise,
                                         temp_sampling=[args.temp_sampling_tr, args.temp_sampling_rot,
                                                        args.temp_sampling_tor],
                                         temp_psi=[args.temp_psi_tr, args.temp_psi_rot, args.temp_psi_tor],
                                         temp_sigma_data=[args.temp_sigma_data_tr, args.temp_sigma_data_rot,
                                                          args.temp_sigma_data_tor])

         # 获取预测的配体位置，并将其调整为原始复合物的中心
        ligand_pos = np.asarray([complex_graph['ligand'].pos.cpu().numpy() + orig_complex_graph.original_center.cpu().numpy() for complex_graph in data_list])

        # 如果有置信度值，按置信度重新排序预测结果
        # reorder predictions based on confidence output
        if confidence is not None and isinstance(confidence_args.rmsd_classification_cutoff, list):
            confidence = confidence[:, 0] # 选择第一个置信度值（假设是与 RMSD 相关的分类值）
        if confidence is not None:
            confidence = confidence.cpu().numpy()# 将置信度数据转移到CPU并转换为numpy数组
            re_order = np.argsort(confidence)[::-1]# 按照置信度排序
            confidence = confidence[re_order]# 按照排序后的索引重新排列置信度
            ligand_pos = ligand_pos[re_order] # 按照排序后的索引重新排列配体位置

        # save predictions
        write_dir = f'{args.out_dir}/{complex_name_list[idx]}' # 为每个复合物创建一个目录
        for rank, pos in enumerate(ligand_pos):
            mol_pred = copy.deepcopy(lig)# 复制配体分子
            if score_model_args.remove_hs: mol_pred = RemoveAllHs(mol_pred)# 如果需要，去除氢原子
            # 保存最优预测的分子
            if rank == 0: write_mol_with_coords(mol_pred, pos, os.path.join(write_dir, f'rank{rank+1}.sdf'))
            # 保存带有置信度的预测
            write_mol_with_coords(mol_pred, pos, os.path.join(write_dir, f'rank{rank+1}_confidence{confidence[rank]:.2f}.sdf'))

        # 如果需要保存可视化，保存每一帧
        # save visualisation frames
        if args.save_visualisation:
            if confidence is not None:
                for rank, batch_idx in enumerate(re_order):
                    visualization_list[batch_idx].write(os.path.join(write_dir, f'rank{rank+1}_reverseprocess.pdb'))
            else:
                for rank, batch_idx in enumerate(ligand_pos):
                    visualization_list[batch_idx].write(os.path.join(write_dir, f'rank{rank+1}_reverseprocess.pdb'))

    except Exception as e:
        print("Failed on", orig_complex_graph["name"], e) # 如果处理某个复合物出错，记录失败信息
        failures += 1# 失败计数加1

# 输出最终的统计结果
print(f'Failed for {failures} complexes')# 输出失败的复合物数量
print(f'Skipped {skipped} complexes')# 输出跳过的复合物数量
print(f'Results are in {args.out_dir}')# 输出结果的保存目录
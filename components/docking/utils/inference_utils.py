import copy
import os
import pickle

import torch
from Bio.PDB import PDBParser
from esm import FastaBatchedDataset, pretrained
from rdkit.Chem import AddHs, MolFromSmiles
from torch_geometric.data import Dataset, HeteroData
import esm

from datasets.constants import three_to_one
from datasets.process_mols import generate_conformer, read_molecule, get_lig_graph_with_matching, moad_extract_receptor_structure

#从 PDB 文件中提取蛋白质的氨基酸序列，忽略水分子（HOH）和非标准氨基酸。每条链的序列会用 : 分隔。
def get_sequences_from_pdbfile(file_path):
    """
    从 PDB 文件中提取蛋白质序列，忽略水分子（HOH）和非标准氨基酸。

    参数:
        file_path: PDB 文件的路径。

    返回:
        sequence: 提取的蛋白质序列，包含各链的序列。链与链之间用冒号分隔。
    """
    # 使用 Biopython 的 PDBParser 解析 PDB 文件
    biopython_parser = PDBParser()
    structure = biopython_parser.get_structure('random_id', file_path)
    structure = structure[0]# 获取第一个模型（通常为唯一模型）
    sequence = None# 初始化序列
     # 遍历每个链
    for i, chain in enumerate(structure):
        seq = ''# 当前链的序列
        # 遍历链中的每个残基
        for res_idx, residue in enumerate(chain):
            # 忽略水分子（HOH）
            if residue.get_resname() == 'HOH':
                continue
            residue_coords = []# 存储残基坐标（此处未使用）
            # 提取残基中的氨基酸原子坐标
            c_alpha, n, c = None, None, None
            for atom in residue:
                if atom.name == 'CA':# α-碳原子
                    c_alpha = list(atom.get_vector())
                if atom.name == 'N':# 胺基氮原子
                    n = list(atom.get_vector())
                if atom.name == 'C':# 羧基碳原子
                    c = list(atom.get_vector())
            # 只有在三个关键原子都存在时才添加该氨基酸到序列
            if c_alpha != None and n != None and c != None:  # only append residue if it is an amino acid
                try:
                    seq += three_to_one[residue.get_resname()]# 使用三字母转一字母的映射
                except Exception as e:
                    seq += '-'# 如果遇到不认识的氨基酸，则用 '-' 代替
                    print("encountered unknown AA: ", residue.get_resname(), ' in the complex. Replacing it with a dash - .')

        # 将链的序列连接到整体序列中，链与链之间用 ':' 分隔
        if sequence is None:
            sequence = seq
        else:
            sequence += (":" + seq)

    return sequence


def set_nones(l):
    """
    将列表中的 'nan' 字符串值替换为 None。

    参数:
        l: 输入列表。

    返回:
        返回一个新的列表，其中 'nan' 被替换为 None。
    """
    return [s if str(s) != 'nan' else None for s in l]

#根据给定的蛋白质文件路径列表 protein_files 和已有序列 protein_sequences 提取或获取蛋白质序列。
def get_sequences(protein_files, protein_sequences):
    """
    从多个蛋白质文件或序列中获取蛋白质序列。如果文件路径不为空，则从 PDB 文件中提取序列，
    否则使用给定的已有序列。

    参数:
        protein_files: 蛋白质文件路径的列表，若元素为 None 则使用对应的 protein_sequences 中的序列。
        protein_sequences: 给定的蛋白质序列列表，当 protein_files 中的元素为 None 时使用。

    返回:
        new_sequences: 提取或使用的蛋白质序列列表。
    """
    new_sequences = []
    for i in range(len(protein_files)):
        # 如果文件路径不为空，则从 PDB 文件中提取序列
        if protein_files[i] is not None:
            new_sequences.append(get_sequences_from_pdbfile(protein_files[i]))
        else:
            # 否则直接使用已有序列
            new_sequences.append(protein_sequences[i])
    return new_sequences

#使用 ESM（例如 ESM-1b）模型计算蛋白质序列的嵌入向量。
def compute_ESM_embeddings(model, alphabet, labels, sequences):
    # settings used
    """
    使用 ESM 模型计算蛋白质序列的嵌入向量。

    参数:
        model: ESM 模型，用于计算嵌入。
        alphabet: ESM 字母表对象。
        labels: 蛋白质序列的标签。
        sequences: 蛋白质序列列表。

    返回:
        embeddings: 蛋白质序列的嵌入字典，键为标签，值为对应的嵌入向量。
    """
    # 设置批量处理的相关参数
    toks_per_batch = 4096
    repr_layers = [33]
    include = "per_tok"
    truncation_seq_length = 1022# 截断序列的最大长度

    # 创建一个 FastaBatchedDataset 对象，将标签和序列传入
    dataset = FastaBatchedDataset(labels, sequences)
    # print(sequences)
    # print("/")
    #print(dataset.sequence_labels,dataset.sequence_strs)
    batches = dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq=1)
    # 创建数据加载器，用于批量加载数据
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=alphabet.get_batch_converter(truncation_seq_length), batch_sampler=batches
    )
    #print(data_loader)


    # 确保所有的层级都在模型的层数范围内
    assert all(-(model.num_layers + 1) <= i <= model.num_layers for i in repr_layers)
    repr_layers = [(i + model.num_layers + 1) % (model.num_layers + 1) for i in repr_layers]
    embeddings = {} # 存储嵌入向量的字典

    # 使用 no_grad() 禁用梯度计算，提高推理效率
    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            print(f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)")
            if torch.cuda.is_available():
                toks = toks.to(device="cuda", non_blocking=True)

            # 获取模型的输出
            out = model(toks, repr_layers=repr_layers, return_contacts=False)
            representations = {layer: t.to(device="cpu") for layer, t in out["representations"].items()}

            # 提取每个序列的嵌入向量
            for i, label in enumerate(labels):
                truncate_len = min(truncation_seq_length, len(strs[i]))
                embeddings[label] = representations[33][i, 1: truncate_len + 1].clone()
    #print(embeddings["1a0q_chain_0"])
    return embeddings

#使用 ESM 模型（例如 ESM-1b 或类似的蛋白质结构预测模型）根据输入的氨基酸序列生成蛋白质的三维结构，并将结构保存到指定的 PDB 文件中。如果遇到内存不足的情况，自动调整批量大小（chunk_size）以减少内存占用。
def generate_ESM_structure(model, filename, sequence):
    """
    使用 ESM 模型生成蛋白质的三维结构，并将其保存为 PDB 文件。

    参数:
        model: ESM 模型对象，执行三维结构预测。
        filename: 保存生成的三维结构的文件路径。
        sequence: 输入的蛋白质氨基酸序列，作为结构预测的依据。

    返回:
        bool: 如果生成并成功保存了结构，则返回 True，否则返回 False。
    """
    model.set_chunk_size(256)# 设置模型的 chunk_size，控制内存占用
    chunk_size = 256# 初始的 chunk_size
    output = None# 初始化输出

    # 循环直到成功生成结构
    while output is None:
        try:
            with torch.no_grad():# 禁用梯度计算，提高推理效率
                output = model.infer_pdb(sequence)# 使用模型预测三维结构

             # 将生成的结构保存到指定的文件中
            with open(filename, "w") as f:
                f.write(output)
                print("saved", filename)# 打印保存成功的消息
        except RuntimeError as e:
            if 'out of memory' in str(e):# 如果遇到内存不足的错误（例如“out of memory”）
                print('| WARNING: ran out of memory on chunk_size', chunk_size)
                # 如果有梯度计算，清空梯度来释放内存
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory# 清空梯度
                torch.cuda.empty_cache()# 清理 CUDA 内存缓存
                # 将 chunk_size 减半，减少内存占用
                chunk_size = chunk_size // 2
                if chunk_size > 2:
                    model.set_chunk_size(chunk_size)# 更新模型的 chunk_size
                else:
                    print("Not enough memory for ESMFold")
                    break# 如果内存不足，退出循环
            else:
                raise e# 如果是其他错误，抛出异常
    return output is not None# 如果成功生成并保存了结构，返回 True，否则返回 False


"""
InferenceDataset 类是一个 PyTorch 数据集类，专门用于在药物发现和蛋白质-配体相互作用预测的任务中处理复合物数据。它支持蛋白质结构和配体的处理，并生成异质图来表示蛋白质-配体复合物。

主要功能：
初始化：

处理蛋白质文件、配体描述、氨基酸序列等输入数据，并使用 ESM 模型生成蛋白质的语言模型嵌入。
如果有缺失的蛋白质结构，使用 ESMFold 模型生成。
数据获取：

每个索引返回一个 HeteroData 对象，包含配体和受体的图结构。图数据是用于后续图神经网络训练或推理的输入。
配体通过 SMILES 字符串或文件路径读取，并生成 3D 构象。受体通过 PDB 文件读取并处理。
错误处理：

如果解析配体或受体时遇到问题，函数会捕获异常并跳过该复合物。
图处理：

对配体和受体进行中心化处理，使它们的坐标相对于质心平移。
主要方法：
__init__(): 初始化并处理输入数据，生成嵌入，缺失结构通过 ESMFold 生成。
len(): 返回数据集的大小（复合物数量）。
get(idx): 获取指定索引的复合物图数据，包含蛋白质、配体的结构信息，以及语言模型嵌入。
"""
class InferenceDataset(Dataset):
    """
    用于推理的 PyTorch 数据集类。该类主要用于处理和预处理配体-受体复合物的数据，生成相应的图结构，并生成必要的语言模型嵌入和蛋白质结构。

    参数:
        out_dir: 输出目录路径，用于存储生成的结构文件。
        complex_names: 复合物的名称列表。
        protein_files: 蛋白质结构文件列表（PDB 格式）。
        ligand_descriptions: 配体的描述，可以是 SMILES 字符串或者文件路径。
        protein_sequences: 蛋白质的氨基酸序列列表。
        lm_embeddings: 语言模型嵌入（例如 ESM 模型生成的嵌入）。
        receptor_radius: 受体的半径，用于邻居计算（默认 30）。
        c_alpha_max_neighbors: 最大邻居数（默认为 None）。
        precomputed_lm_embeddings: 预先计算的语言模型嵌入（默认为 None）。
        remove_hs: 是否移除氢原子（默认 False）。
        all_atoms: 是否使用所有原子（默认 False）。
        atom_radius: 原子半径，用于邻居计算（默认 5）。
        atom_max_neighbors: 最大原子邻居数（默认 None）。
        knn_only_graph: 是否仅使用 KNN 图（默认 False）。
    """
    def __init__(self, out_dir, complex_names, protein_files, ligand_descriptions, protein_sequences, lm_embeddings,
                 receptor_radius=30, c_alpha_max_neighbors=None, precomputed_lm_embeddings=None,
                 remove_hs=False, all_atoms=False, atom_radius=5, atom_max_neighbors=None, knn_only_graph=False):

        super(InferenceDataset, self).__init__()
        # 初始化参数
        self.receptor_radius = receptor_radius
        self.c_alpha_max_neighbors = c_alpha_max_neighbors
        self.remove_hs = remove_hs
        self.all_atoms = all_atoms
        self.atom_radius, self.atom_max_neighbors = atom_radius, atom_max_neighbors
        self.knn_only_graph = knn_only_graph

        self.complex_names = complex_names# 复合物名称
        self.protein_files = protein_files# 蛋白质文件路径列表
        self.ligand_descriptions = ligand_descriptions # 配体描述（SMILES 或文件路径）
        self.protein_sequences = protein_sequences# 蛋白质氨基酸序列列表

        # generate LM embeddings
        # 生成语言模型（LM）嵌入
        if lm_embeddings and (precomputed_lm_embeddings is None or precomputed_lm_embeddings[0] is None):
            print("Generating ESM language model embeddings")
            model_location = "esm2_t33_650M_UR50D"
            model, alphabet = pretrained.load_model_and_alphabet(model_location)
            model.eval()
            if torch.cuda.is_available():
                model = model.cuda()

            # 获取蛋白质序列
            protein_sequences = get_sequences(protein_files, protein_sequences)
            #print(protein_sequences)
            labels, sequences = [], []
            for i in range(len(protein_sequences)):
                s = protein_sequences[i].split(':')# 分割多个链
                sequences.extend(s)
                labels.extend([complex_names[i] + '_chain_' + str(j) for j in range(len(s))])

            # 计算 ESM 嵌入
            lm_embeddings = compute_ESM_embeddings(model, alphabet, labels, sequences)
            #print(lm_embeddings)

            # 保存嵌入
            self.lm_embeddings = []
            for i in range(len(protein_sequences)):
                s = protein_sequences[i].split(':')
                self.lm_embeddings.append([lm_embeddings[f'{complex_names[i]}_chain_{j}'] for j in range(len(s))])

        elif not lm_embeddings:
            # 如果没有提供语言模型嵌入，则初始化为空
            self.lm_embeddings = [None] * len(self.complex_names)

        else:
            # 如果提供了预计算的嵌入，则直接使用
            self.lm_embeddings = precomputed_lm_embeddings

        # generate structures with ESMFold
        # 使用 ESMFold 生成缺失的蛋白质结构
        if None in protein_files:
            print("generating missing structures with ESMFold")
            # 加载 ESMFold 模型
            model = esm.pretrained.esmfold_v1()
            model = model.eval().cuda()

            for i in range(len(protein_files)):
                if protein_files[i] is None:
                    self.protein_files[i] = f"{out_dir}/{complex_names[i]}/{complex_names[i]}_esmfold.pdb"
                    if not os.path.exists(self.protein_files[i]):
                        print("generating", self.protein_files[i])
                        generate_ESM_structure(model, self.protein_files[i], protein_sequences[i])

    def len(self):
        """
        返回数据集中的样本数量，即复合物的数量。
        """
        return len(self.complex_names)

    def get(self, idx):

        """
        获取给定索引的复合物图数据，包括配体、受体、语言模型嵌入等信息。

        参数:
            idx: 数据集中的索引。

        返回:
            complex_graph: 一个包含复合物图数据的 `HeteroData` 对象。
        """
        # 获取复合物名称、蛋白质文件、配体描述和语言模型嵌入
        name, protein_file, ligand_description, lm_embedding = \
            self.complex_names[idx], self.protein_files[idx], self.ligand_descriptions[idx], self.lm_embeddings[idx]

        # build the pytorch geometric heterogeneous graph
        # 创建 PyTorch Geometric 异质图（HeteroData）
        complex_graph = HeteroData()
        complex_graph['name'] = name

        # parse the ligand, either from file or smile
        # 解析配体：如果是 SMILES 字符串则生成分子对象
        try:
            mol = MolFromSmiles(ligand_description)  # check if it is a smiles or a path# 检查是否是 SMILES 或文件路径

            if mol is not None:
                mol = AddHs(mol)# 添加氢原子
                generate_conformer(mol) # 生成构象
            else:
                mol = read_molecule(ligand_description, remove_hs=False, sanitize=True)
                if mol is None:
                    raise Exception('RDKit could not read the molecule ', ligand_description)
                mol.RemoveAllConformers()
                mol = AddHs(mol)
                generate_conformer(mol)
        except Exception as e:
            print('Failed to read molecule ', ligand_description, ' We are skipping it. The reason is the exception: ', e)
            complex_graph['success'] = False
            return complex_graph

        # 解析受体并构建异质图
        try:
            # parse the receptor from the pdb file
             # 获取配体-受体匹配图
            get_lig_graph_with_matching(mol, complex_graph, popsize=None, maxiter=None, matching=False, keep_original=False,
                                        num_conformers=1, remove_hs=self.remove_hs)

            # 提取受体结构
            moad_extract_receptor_structure(
                path=os.path.join(protein_file),
                complex_graph=complex_graph,
                neighbor_cutoff=self.receptor_radius,
                max_neighbors=self.c_alpha_max_neighbors,
                lm_embeddings=lm_embedding,
                knn_only_graph=self.knn_only_graph,
                all_atoms=self.all_atoms,
                atom_cutoff=self.atom_radius,
                atom_max_neighbors=self.atom_max_neighbors)

        except Exception as e:
            print(f'Skipping {name} because of the error:')
            print(e)
            complex_graph['success'] = False
            return complex_graph

        # 计算蛋白质和配体的中心，并对坐标进行平移，使其原点为中心
        protein_center = torch.mean(complex_graph['receptor'].pos, dim=0, keepdim=True)
        complex_graph['receptor'].pos -= protein_center
        if self.all_atoms:
            complex_graph['atom'].pos -= protein_center

        ligand_center = torch.mean(complex_graph['ligand'].pos, dim=0, keepdim=True)
        complex_graph['ligand'].pos -= ligand_center

        complex_graph.original_center = protein_center
        complex_graph.mol = mol
        complex_graph['success'] = True
        #print("yes")
        #print(complex_graph['ligand'].pos)
        return complex_graph
"""
1. 处理配体描述（ligand_descriptions）
配体描述可以是 SMILES 字符串 或 文件路径。类的 get 方法尝试解析配体并生成 3D 构象。
从 SMILES 字符串解析：通过 MolFromSmiles(ligand_description) 尝试从 SMILES 字符串中生成分子对象。如果解析成功，继续添加氢原子并生成 3D 构象。
从文件路径解析：如果 ligand_description 不是 SMILES 字符串，而是一个文件路径，则通过 read_molecule 函数读取该分子文件（假设是标准的分子文件格式，如 SDF 或 MOL2）。然后同样添加氢原子，并生成 3D 构象。
错误处理：如果上述任何一步失败，捕获异常并标记该复合物（complex_graph['success'] = False）。
2. 处理受体描述（protein_files）
受体描述是一个 蛋白质结构文件路径（通常为 PDB 文件）。
配体-受体匹配：首先调用 get_lig_graph_with_matching 函数生成配体与受体的匹配图。这一步将配体的分子结构与受体进行匹配，建立它们之间的关系。
提取受体结构：调用 moad_extract_receptor_structure 函数处理蛋白质（受体）。此函数执行以下任务：
读取蛋白质的 PDB 文件（路径为 protein_file）。
计算受体的邻居信息，基于参数 neighbor_cutoff 和 max_neighbors 限制邻居数。
计算与语言模型嵌入（lm_embedding）相关的信息（用于表示蛋白质的结构特征）。
根据 all_atoms 参数决定是否处理所有原子，或者仅处理 Cα 原子。
生成基于原子半径（atom_radius）的邻居信息。
3. 处理后的配体和受体数据
在上述两个步骤完成后，配体和受体的结构信息都已经被解析和处理，并且在 complex_graph 中存储了相关的数据（如配体和受体的图结构信息）。

配体的图数据：
配体通过 get_lig_graph_with_matching 生成了与受体匹配的图数据，这些数据包含配体的 3D 坐标信息以及其他与配体相关的图结构。
受体的图数据：
受体的信息通过 moad_extract_receptor_structure 进行处理，其中包括蛋白质的 3D 坐标、邻居信息等。
受体和配体的坐标将被中心化，使其相对于质心平移，从而保证它们在空间中的位置一致。
4. 图结构的返回
最终，get 方法返回一个 HeteroData 对象，其中包含以下图数据：

complex_graph['receptor']：受体的图数据（包括原子坐标和邻居信息等）。
complex_graph['ligand']：配体的图数据（包括分子坐标和配体与受体的匹配信息）。
complex_graph['success']：表示该复合物是否成功处理。
在所有的计算完成后，复合物的坐标信息被平移，使其原点在蛋白质的质心，并且配体和受体的图数据都已准备好，可以用于后续的机器学习或图神经网络任务。

总结
配体处理：根据提供的 SMILES 字符串或文件路径，解析并生成配体的 3D 构象。
受体处理：从 PDB 文件中读取受体的结构，计算其邻居信息并生成图结构。
图结构：配体和受体的图结构在 complex_graph 中存储，并且在后续的图神经网络任务中使用。
"""
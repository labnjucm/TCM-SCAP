import os
import pickle
from argparse import ArgumentParser
from Bio.PDB import PDBParser
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from tqdm import tqdm
from Bio import SeqIO

from datasets.constants import three_to_one

#该代码的功能是解析 PDB 文件，从中提取蛋白质的氨基酸序列并将其保存为指定格式（FASTA 或二进制文件）。代码支持两种数据集：pdbbind 和 moad，分别使用不同的处理方式
parser = ArgumentParser()
parser.add_argument('--out_file', type=str, default="data/prepared_for_esm.fasta")
parser.add_argument('--dataset', type=str, default="pdbbind")
parser.add_argument('--data_dir', type=str, default='../data/BindingMOAD_2020_ab_processed_biounit/pdb_protein/', help='')
args = parser.parse_args()

#初始化 Biopython PDB 解析器
biopython_parser = PDBParser()

#从 PDB 文件中提取结构和序列
def get_structure_from_file(file_path):
    structure = biopython_parser.get_structure('random_id', file_path)
    structure = structure[0]# 获取第一个模型（通常只有一个模型）
    l = []# 用于存储每条链的序列
    #遍历链和残基，跳过水分子
    for i, chain in enumerate(structure):
        seq = ''
        for res_idx, residue in enumerate(chain):
            if residue.get_resname() == 'HOH':
                continue
            residue_coords = []
            c_alpha, n, c = None, None, None
            for atom in residue:
                if atom.name == 'CA':
                    c_alpha = list(atom.get_vector())
                if atom.name == 'N':
                    n = list(atom.get_vector())
                if atom.name == 'C':
                    c = list(atom.get_vector())
            if c_alpha != None and n != None and c != None:  # only append residue if it is an amino acid
                try:
                    seq += three_to_one[residue.get_resname()]
                except Exception as e:
                    seq += '-'
                    print("encountered unknown AA: ", residue.get_resname(), ' in the complex ', file_path, '. Replacing it with a dash - .')
        l.append(seq)
    return l

data_dir = args.data_dir
names = os.listdir(data_dir)
#处理 pdbbind 数据集
if args.dataset == 'pdbbind':
    sequences = []
    ids = []

    #初始化存储序列和 ID 的列表
    for name in tqdm(names):
        if name == '.DS_Store': continue
        if os.path.exists(os.path.join(data_dir, name, f'{name}_protein_processed.pdb')):
            rec_path = os.path.join(data_dir, name, f'{name}_protein_processed.pdb')
        else:
            rec_path = os.path.join(data_dir, name, f'{name}_protein.pdb')
        l = get_structure_from_file(rec_path)
        for i, seq in enumerate(l):
            sequences.append(seq)
            ids.append(f'{name}_chain_{i}')
    records = []
    for (index, seq) in zip(ids, sequences):
        record = SeqRecord(Seq(seq), str(index))
        record.description = ''
        records.append(record)
    SeqIO.write(records, args.out_file, "fasta")
#处理 moad 数据集
elif args.dataset == 'moad':
    names = [n[:6] for n in names]
    name_to_sequence = {}

    for name in tqdm(names):
        if name == '.DS_Store': continue
        if not os.path.exists(os.path.join(data_dir, f'{name}_protein.pdb')):
            print(f"We are skipping {name} because there was no {name}_protein.pdb")
            continue
        rec_path = os.path.join(data_dir, f'{name}_protein.pdb')
        l = get_structure_from_file(rec_path)
        for i, seq in enumerate(l):
            name_to_sequence[name + '_chain_' + str(i)] = seq

    # save to file
    with open(args.out_file, 'wb') as f:
        pickle.dump(name_to_sequence, f)


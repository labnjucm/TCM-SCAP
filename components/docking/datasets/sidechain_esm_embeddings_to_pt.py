
import os
import pickle
from argparse import ArgumentParser

import torch
from tqdm import tqdm

"""
此代码用于处理 ESM（进化关系语言模型）嵌入，将特定的蛋白质序列与对应的嵌入表示匹配并保存到输出文件中。
"""
"""
创建命令行参数解析器。
参数说明：
--esm_embeddings_path：ESM 嵌入的目录路径，默认值为指定的数据文件夹。
--output_path：保存最终嵌入文件的路径，默认为指定的 .pt 文件。
使用 parser.parse_args() 获取传递的命令行参数。
"""
parser = ArgumentParser()
parser.add_argument('--esm_embeddings_path', type=str, default='data/BindingMOAD_2020_ab_processed_biounit/moad_sequences_new', help='')
parser.add_argument('--output_path', type=str, default='data/BindingMOAD_2020_ab_processed_biounit/moad_sequences_new.pt', help='')
args = parser.parse_args()

dic = {}#初始化一个空字典 dic，用于存储蛋白质序列 ID 和其对应的 ESM 嵌入

# read text file with all sequences
#使用 pickle 加载有用的蛋白质序列集合 useful_sequences.pkl，这些序列需要被匹配到对应的嵌入。
with open('data/pdb_2021aug02/sequences_to_id.fasta') as f:
    lines = f.readlines()

# read sequences
with open('data/pdb_2021aug02/useful_sequences.pkl', 'rb') as f:
    sequences = pickle.load(f)

ids = set()
#构建序列到 ID 的映射
dict_seq_id = {seq[:-1]: str(id) for id, seq in enumerate(lines)}
#筛选有用的序列 ID
for i, seq in tqdm(enumerate(sequences)):
    ids.add(dict_seq_id[seq])
    if i == 20000: break
#统计和筛选嵌入文件
print("total", len(ids), "out of", len(os.listdir(args.esm_embeddings_path)))

available = set([filename.split('.')[0] for filename in os.listdir(args.esm_embeddings_path)])
final = available.intersection(ids)
#加载并保存最终嵌入
for idp in tqdm(final):
    dic[idp] = torch.load(os.path.join(args.esm_embeddings_path, idp+'.pt'))['representations'][33]
torch.save(dic,args.output_path)
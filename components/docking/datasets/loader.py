import torch
from torch_geometric.data import Dataset

from datasets.dataloader import DataLoader, DataListLoader
from datasets.moad import MOAD
from datasets.pdb import PDBSidechain
from datasets.pdbbind import NoiseTransform, PDBBind
from utils.utils import read_strings_from_txt

#该代码定义了 CombineDatasets 类，用于合并两个数据集，并提供了 construct_loader 函数，用于根据给定的参数构建训练和验证数据加载器。支持的数据集包括 pdbsidechain、pdbbind、moad 等，能够灵活地处理多种加载配置。

#功能：该类用于合并两个数据集，使其可以作为一个整体进行操作
class CombineDatasets(Dataset):
    def __init__(self, dataset1, dataset2):
        super(CombineDatasets, self).__init__()
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    #返回合并数据集的总长度
    def len(self):
        return len(self.dataset1) + len(self.dataset2)

    #根据索引 idx 返回对应的数据样本
    def get(self, idx):
        if idx < len(self.dataset1):
            return self.dataset1[idx]
        else:
            return self.dataset2[idx - len(self.dataset1)]

    #向 dataset1 中添加新样本
    def add_complexes(self, new_complex_list):
        self.dataset1.add_complexes(new_complex_list)


#根据参数配置构建数据加载器
def construct_loader(args, t_to_sigma, device):
    val_dataset2 = None
    #定义噪声变换，用于数据增强和预处理
    transform = NoiseTransform(t_to_sigma=t_to_sigma, no_torsion=args.no_torsion,
                               all_atom=args.all_atoms, alpha=args.sampling_alpha, beta=args.sampling_beta,
                               include_miscellaneous_atoms=False if not hasattr(args, 'include_miscellaneous_atoms') else args.include_miscellaneous_atoms,
                               crop_beyond_cutoff=args.crop_beyond)
    if args.triple_training: assert args.combined_training

    sequences_to_embeddings = None
    # ESM 嵌入加载
    if args.dataset == 'pdbsidechain' or args.triple_training:
        if args.pdbsidechain_esm_embeddings_path is not None:
            print('Loading ESM embeddings')
            id_to_embeddings = torch.load(args.pdbsidechain_esm_embeddings_path)
            sequences_list = read_strings_from_txt(args.pdbsidechain_esm_embeddings_sequences_path)
            sequences_to_embeddings = {}
            for i, seq in enumerate(sequences_list):
                if str(i) in id_to_embeddings:
                    sequences_to_embeddings[seq] = id_to_embeddings[str(i)]

    if args.dataset == 'pdbsidechain' or args.triple_training:

        common_args = {'root': args.pdbsidechain_dir, 'transform': transform, 'limit_complexes': args.limit_complexes,
                       'receptor_radius': args.receptor_radius,
                       'c_alpha_max_neighbors': args.c_alpha_max_neighbors,
                       'remove_hs': args.remove_hs, 'num_workers': args.num_workers, 'all_atoms': args.all_atoms,
                       'atom_radius': args.atom_radius, 'atom_max_neighbors': args.atom_max_neighbors,
                       'knn_only_graph': not args.not_knn_only_graph, 'sequences_to_embeddings': sequences_to_embeddings,
                       'vandermers_max_dist': args.vandermers_max_dist,
                       'vandermers_buffer_residue_num': args.vandermers_buffer_residue_num,
                       'vandermers_min_contacts': args.vandermers_min_contacts,
                       'remove_second_segment': args.remove_second_segment,
                       'merge_clusters': args.merge_clusters}
        #构建 pdbsidechain 数据集
        train_dataset3 = PDBSidechain(cache_path=args.cache_path, split='train', multiplicity=args.train_multiplicity, **common_args)

        if args.dataset == 'pdbsidechain':
            train_dataset = train_dataset3
            val_dataset = PDBSidechain(cache_path=args.cache_path, split='val', multiplicity=args.val_multiplicity, **common_args)
        loader_class = DataListLoader if torch.cuda.is_available() else DataLoader

    #构建其他数据集
    if args.dataset in ['pdbbind', 'moad', 'generalisation', 'distillation']:
        common_args = {'transform': transform, 'limit_complexes': args.limit_complexes,
                       'chain_cutoff': args.chain_cutoff, 'receptor_radius': args.receptor_radius,
                       'c_alpha_max_neighbors': args.c_alpha_max_neighbors,
                       'remove_hs': args.remove_hs, 'max_lig_size': args.max_lig_size,
                       'matching': not args.no_torsion, 'popsize': args.matching_popsize, 'maxiter': args.matching_maxiter,
                       'num_workers': args.num_workers, 'all_atoms': args.all_atoms,
                       'atom_radius': args.atom_radius, 'atom_max_neighbors': args.atom_max_neighbors,
                       'knn_only_graph': False if not hasattr(args, 'not_knn_only_graph') else not args.not_knn_only_graph,
                       'include_miscellaneous_atoms': False if not hasattr(args, 'include_miscellaneous_atoms') else args.include_miscellaneous_atoms,
                       'matching_tries': args.matching_tries}

        if args.dataset == 'pdbbind' or args.dataset == 'generalisation' or args.combined_training:
            train_dataset = PDBBind(cache_path=args.cache_path, split_path=args.split_train, keep_original=True,
                                    num_conformers=args.num_conformers, root=args.pdbbind_dir,
                                    esm_embeddings_path=args.pdbbind_esm_embeddings_path,
                                    protein_file=args.protein_file, **common_args)

        if args.dataset == 'moad' or args.combined_training:
            train_dataset2 = MOAD(cache_path=args.cache_path, split='train', keep_original=True,
                                  num_conformers=args.num_conformers, max_receptor_size=args.max_receptor_size,
                                  remove_promiscuous_targets=args.remove_promiscuous_targets, min_ligand_size=args.min_ligand_size,
                                  multiplicity= args.train_multiplicity, unroll_clusters=args.unroll_clusters,
                                  esm_embeddings_sequences_path=args.moad_esm_embeddings_sequences_path,
                                  root=args.moad_dir, esm_embeddings_path=args.moad_esm_embeddings_path,
                                  enforce_timesplit=args.enforce_timesplit, **common_args)

            if args.combined_training:
                train_dataset = CombineDatasets(train_dataset2, train_dataset)
                if args.triple_training:
                    train_dataset = CombineDatasets(train_dataset, train_dataset3)
            else:
                train_dataset = train_dataset2

        if args.dataset == 'pdbbind' or args.double_val:
            val_dataset = PDBBind(cache_path=args.cache_path, split_path=args.split_val, keep_original=True,
                                  esm_embeddings_path=args.pdbbind_esm_embeddings_path, root=args.pdbbind_dir,
                                  protein_file=args.protein_file, require_ligand=True, **common_args)
            if args.double_val:
                val_dataset2 = val_dataset

        if args.dataset == 'moad' or args.dataset == 'generalisation':
            val_dataset = MOAD(cache_path=args.cache_path, split='val', keep_original=True,
                               multiplicity= args.val_multiplicity, max_receptor_size=args.max_receptor_size,
                               remove_promiscuous_targets=args.remove_promiscuous_targets, min_ligand_size=args.min_ligand_size,
                               esm_embeddings_sequences_path=args.moad_esm_embeddings_sequences_path,
                               unroll_clusters=args.unroll_clusters, root=args.moad_dir,
                               esm_embeddings_path=args.moad_esm_embeddings_path, require_ligand=True, **common_args)

        #数据加载器类选择
        loader_class = DataListLoader if torch.cuda.is_available() else DataLoader

    #构建数据加载器
    train_loader = loader_class(dataset=train_dataset, batch_size=args.batch_size, num_workers=args.num_dataloader_workers, shuffle=True, pin_memory=args.pin_memory, drop_last=args.dataloader_drop_last)
    val_loader = loader_class(dataset=val_dataset, batch_size=args.batch_size, num_workers=args.num_dataloader_workers, shuffle=False, pin_memory=args.pin_memory, drop_last=args.dataloader_drop_last)
    return train_loader, val_loader, val_dataset2


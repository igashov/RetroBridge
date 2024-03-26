import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import subprocess

from rdkit import Chem
from src.data import utils
from src.data.abstract_dataset import MolecularDataModule
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from typing import Any, Sequence

from pdb import set_trace

DOWNLOAD_URL_TEMPLATE = 'https://zenodo.org/record/8114657/files/{fname}?download=1'
USPTO_MIT_DOWNLOAD_URL = 'https://github.com/wengong-jin/nips17-rexgen/raw/master/USPTO/data.zip'


def to_list(value: Any) -> Sequence:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    else:
        return [value]


class RetroBridgeDataset(InMemoryDataset):
    types = {
        'N': 0, 'C': 1, 'O': 2, 'S': 3, 'Cl': 4, 'F': 5, 'B': 6, 'Br': 7, 'P': 8,
        'Si': 9, 'I': 10, 'Sn': 11, 'Mg': 12, 'Cu': 13, 'Zn': 14, 'Se': 15, '*': 16,
    }

    bonds = {
        Chem.BondType.SINGLE: 0,
        Chem.BondType.DOUBLE: 1,
        Chem.BondType.TRIPLE: 2,
        Chem.BondType.AROMATIC: 3
    }

    def __init__(self, stage, root, extra_nodes=False, swap=False):
        self.stage = stage
        self.extra_nodes = extra_nodes

        if self.stage == 'train':
            self.file_idx = 0
        elif self.stage == 'val':
            self.file_idx = 1
        elif self.stage == 'test':
            self.file_idx = 2
        else:
            raise NotImplementedError

        super().__init__(root=root)
        self.data, self.slices = torch.load(self.processed_paths[self.file_idx])

        if swap:
            self.data = Data(
                x=self.data.p_x, edge_index=self.data.p_edge_index, edge_attr=self.data.p_edge_attr,
                p_x=self.data.x, p_edge_index=self.data.edge_index, p_edge_attr=self.data.edge_attr,
                y=self.data.y, idx=self.data.idx, r_smiles=self.data.p_smiles, p_smiles=self.data.r_smiles,
            )
            self.slices = {
                'x': self.slices['p_x'],
                'edge_index': self.slices['p_edge_index'],
                'edge_attr': self.slices['p_edge_attr'],
                'y': self.slices['y'],
                'idx': self.slices['idx'],
                'p_x': self.slices['x'],
                'p_edge_index': self.slices['edge_index'],
                'p_edge_attr': self.slices['edge_attr'],
                'r_smiles': self.slices['p_smiles'],
                'p_smiles': self.slices['r_smiles'],
            }

    @property
    def processed_dir(self) -> str:
        if self.extra_nodes:
            return os.path.join(self.root, f'processed_retrobridge_extra_nodes')
        else:
            return os.path.join(self.root, f'processed_retrobridge')

    @property
    def raw_file_names(self):
        return ['uspto50k_train.csv', 'uspto50k_val.csv', 'uspto50k_test.csv']

    @property
    def split_file_name(self):
        return ['uspto50k_train.csv', 'uspto50k_val.csv', 'uspto50k_test.csv']

    @property
    def split_paths(self):
        files = to_list(self.split_file_name)
        return [os.path.join(self.raw_dir, f) for f in files]

    @property
    def processed_file_names(self):
        return [f'train.pt', f'val.pt', f'test.pt']

    def download(self):
        os.makedirs(self.raw_dir, exist_ok=True)
        for fname in self.raw_file_names:
            print(f'Downloading {fname}')
            url = DOWNLOAD_URL_TEMPLATE.format(fname=fname)
            path = os.path.join(self.raw_dir, fname)
            subprocess.run(f'wget {url} -O {path}', shell=True)

    def process(self):
        table = pd.read_csv(self.split_paths[self.file_idx])
        data_list = []
        for i, reaction_smiles in enumerate(tqdm(table['reactants>reagents>production'].values)):
            reactants_smi, _, product_smi = reaction_smiles.split('>')
            rmol = Chem.MolFromSmiles(reactants_smi)
            pmol = Chem.MolFromSmiles(product_smi)
            r_num_nodes = rmol.GetNumAtoms()
            p_num_nodes = pmol.GetNumAtoms()
            assert p_num_nodes <= r_num_nodes

            if self.extra_nodes:
                new_r_num_nodes = p_num_nodes + RetroBridgeDatasetInfos.max_n_dummy_nodes
                if r_num_nodes > new_r_num_nodes:
                    print(f'Molecule with |r|-|p| > max_n_dummy_nodes: r={r_num_nodes}, p={p_num_nodes}')
                    if self.stage in ['train', 'val']:
                        continue
                    else:
                        reactants_smi, product_smi = 'C', 'C'
                        rmol = Chem.MolFromSmiles(reactants_smi)
                        pmol = Chem.MolFromSmiles(product_smi)
                        p_num_nodes = pmol.GetNumAtoms()
                        new_r_num_nodes = p_num_nodes + RetroBridgeDatasetInfos.max_n_dummy_nodes

                r_num_nodes = new_r_num_nodes

            try:
                mapping = self.compute_nodes_order_mapping(rmol)
                r_x, r_edge_index, r_edge_attr = self.compute_graph(
                    rmol, mapping, r_num_nodes, types=self.types, bonds=self.bonds
                )
                p_x, p_edge_index, p_edge_attr = self.compute_graph(
                    pmol, mapping, r_num_nodes, types=self.types, bonds=self.bonds
                )
            except Exception as e:
                print(f'Error processing molecule {i}: {e}')
                continue

            if self.stage in ['train', 'val']:
                assert len(p_x) == len(r_x)

            product_mask = ~(p_x[:, -1].bool()).squeeze()
            if len(r_x) == len(p_x) and not torch.allclose(r_x[product_mask], p_x[product_mask]):
                print(f'Incorrect atom mapping {i}')
                continue

            if self.stage == 'train' and len(p_edge_attr) == 0:
                continue

            # Shuffle nodes to avoid leaking
            if len(p_x) == len(r_x):
                new2old_idx = torch.randperm(r_num_nodes).long()
                old2new_idx = torch.empty_like(new2old_idx)
                old2new_idx[new2old_idx] = torch.arange(r_num_nodes)

                r_x = r_x[new2old_idx]
                r_edge_index = torch.stack([old2new_idx[r_edge_index[0]], old2new_idx[r_edge_index[1]]], dim=0)
                r_edge_index, r_edge_attr = self.sort_edges(r_edge_index, r_edge_attr, r_num_nodes)

                p_x = p_x[new2old_idx]
                p_edge_index = torch.stack([old2new_idx[p_edge_index[0]], old2new_idx[p_edge_index[1]]], dim=0)
                p_edge_index, p_edge_attr = self.sort_edges(p_edge_index, p_edge_attr, r_num_nodes)

                product_mask = ~(p_x[:, -1].bool()).squeeze()
                assert torch.allclose(r_x[product_mask], p_x[product_mask])

            y = torch.zeros(size=(1, 0), dtype=torch.float)
            data = Data(
                x=r_x, edge_index=r_edge_index, edge_attr=r_edge_attr, y=y, idx=i,
                p_x=p_x, p_edge_index=p_edge_index, p_edge_attr=p_edge_attr,
                r_smiles=reactants_smi, p_smiles=product_smi,
            )

            data_list.append(data)

        print(f'Dataset contains {len(data_list)} reactions')
        torch.save(self.collate(data_list), self.processed_paths[self.file_idx])

    @staticmethod
    def compute_graph(molecule, mapping, max_num_nodes, types, bonds):
        max_num_nodes = max(molecule.GetNumAtoms(), max_num_nodes)  # in case |reactants|-|product| > max_n_dummy_nodes
        type_idx = [len(types) - 1] * max_num_nodes
        for i, atom in enumerate(molecule.GetAtoms()):
            type_idx[mapping[atom.GetAtomMapNum()]] = types[atom.GetSymbol()]

        num_classes = len(types)
        x = F.one_hot(torch.tensor(type_idx), num_classes=num_classes).float()

        row, col, edge_type = [], [], []
        for bond in molecule.GetBonds():
            start_atom_map_num = molecule.GetAtomWithIdx(bond.GetBeginAtomIdx()).GetAtomMapNum()
            end_atom_map_num = molecule.GetAtomWithIdx(bond.GetEndAtomIdx()).GetAtomMapNum()
            start, end = mapping[start_atom_map_num], mapping[end_atom_map_num]
            row += [start, end]
            col += [end, start]
            edge_type += 2 * [bonds[bond.GetBondType()] + 1]

        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_type = torch.tensor(edge_type, dtype=torch.long)
        edge_attr = F.one_hot(edge_type, num_classes=len(bonds) + 1).to(torch.float)

        return x, edge_index, edge_attr

    @staticmethod
    def compute_nodes_order_mapping(molecule):
        # In case if atomic map numbers do not start from 1
        order = []
        for atom in molecule.GetAtoms():
            order.append(atom.GetAtomMapNum())
        order = {
            atom_map_num: idx
            for idx, atom_map_num in enumerate(sorted(order))
        }
        return order

    @staticmethod
    def sort_edges(edge_index, edge_attr, max_num_nodes):
        if len(edge_attr) != 0:
            perm = (edge_index[0] * max_num_nodes + edge_index[1]).argsort()
            edge_index = edge_index[:, perm]
            edge_attr = edge_attr[perm]

        return edge_index, edge_attr


class RetroBridgeMITDataset(RetroBridgeDataset):
    types = {
        'C': 0, 'O': 1, 'N': 2, 'Cl': 3, 'F': 4, 'S': 5, 'Na': 6, 'Br': 7, 'K': 8, 'P': 9, 'H': 10,
        'I': 11, 'B': 12, 'Li': 13, 'Si': 14, 'Pd': 15, 'Cs': 16, 'Al': 17, 'Cu': 18, 'Mg': 19, 'Sn': 20,
        'Zn': 21, 'Fe': 22, 'Cr': 23, 'Mn': 24, 'Ti': 25, 'Pt': 26, 'Ca': 27, 'Ag': 28, 'Se': 29, 'Ni': 30,
        'Ru': 31, 'Rh': 32, 'Co': 33, 'Os': 34, 'Ce': 35, 'Pb': 36, 'Ba': 37, 'Hg': 38, 'Zr': 39, 'As': 40,
        'Yb': 41, 'W': 42, 'Bi': 43, 'Ge': 44, 'In': 45, 'Sb': 46, 'Sc': 47, 'Tl': 48, 'Mo': 49, 'Sm': 50,
        'Re': 51, 'Ir': 52, 'Au': 53, 'Cd': 54, 'Ga': 55, 'Xe': 56, 'Nd': 57, 'Ta': 58, 'V': 59, 'La': 60,
        'Rb': 61, 'Dy': 62, 'Hf': 63, 'Y': 64, 'Te': 65, 'Ar': 66, 'Pr': 67, 'He': 68, 'Be': 69, 'Eu': 70,
        'Sr': 71, '*': 72,
    }

    @property
    def raw_file_names(self):
        return ['train.csv', 'valid.csv', 'test.csv']

    @property
    def split_file_name(self):
        return ['train.csv', 'valid.csv', 'test.csv']

    @staticmethod
    def convert_txt_to_df(path):
        reactions = []
        with open(path, 'r') as f:
            for line in tqdm(list(f.readlines())):
                rxn = line.strip().split()[0]
                reactions.append(rxn)
        return pd.DataFrame({
            'reactants>reagents>production': reactions
        })

    def download(self):
        os.makedirs(self.raw_dir, exist_ok=True)
        path = os.path.join(self.raw_dir, 'data.zip')
        subprocess.run(f'wget {USPTO_MIT_DOWNLOAD_URL} -O {path}', shell=True)
        subprocess.run(f'unzip {path} -d {self.raw_dir}', shell=True)

        for name in ['train', 'test', 'valid']:
            src_path = os.path.join(self.raw_dir, 'data', f'{name}.txt')
            dst_path = os.path.join(self.raw_dir, f'{name}.csv')
            table = self.convert_txt_to_df(src_path)
            table.to_csv(dst_path, index=False)


class RetroBridgeDataModule(MolecularDataModule):
    DATASET_CLASS = RetroBridgeDataset

    def __init__(self, data_root, batch_size, num_workers, shuffle, extra_nodes=False, evaluation=False, swap=False):
        super().__init__(batch_size, num_workers, shuffle)
        self.extra_nodes = extra_nodes
        self.evaluation = evaluation
        self.swap = swap
        self.data_root = data_root
        self.train_smiles = []
        self.prepare_data()

    def prepare_data(self) -> None:
        stage = 'val' if self.evaluation else 'train'
        datasets = {
            'train': self.DATASET_CLASS(stage=stage, root=self.data_root, extra_nodes=self.extra_nodes, swap=self.swap),
            'val': self.DATASET_CLASS(stage='val', root=self.data_root, extra_nodes=self.extra_nodes, swap=self.swap),
            'test': self.DATASET_CLASS(stage='test', root=self.data_root, extra_nodes=self.extra_nodes, swap=self.swap),
        }

        self.dataloaders = {}
        for split, dataset in datasets.items():
            self.dataloaders[split] = DataLoader(
                dataset=dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=(self.shuffle and split == 'train'),
            )

        self.train_smiles = datasets['train'].r_smiles


class RetroBridgeMITDataModule(RetroBridgeDataModule):
    DATASET_CLASS = RetroBridgeMITDataset


class RetroBridgeDatasetInfos:
    atom_encoder = {
        'N': 0, 'C': 1, 'O': 2, 'S': 3, 'Cl': 4, 'F': 5, 'B': 6, 'Br': 7, 'P': 8,
        'Si': 9, 'I': 10, 'Sn': 11, 'Mg': 12, 'Cu': 13, 'Zn': 14, 'Se': 15, '*': 16,
    }
    atom_decoder = ['N', 'C', 'O', 'S', 'Cl', 'F', 'B', 'Br', 'P', 'Si', 'I', 'Sn', 'Mg', 'Cu', 'Zn', 'Se', '*']
    max_n_dummy_nodes = 10

    def __init__(self, datamodule):
        self.name = 'USPTO50K-RetroBridge'
        self.input_dims = None
        self.output_dims = None
        self.remove_h = True
        self.max_weight = 1000
        self.possible_num_dummy_nodes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        self.valencies = None
        self.atom_weights = None

        self.dummy_nodes_dist = None
        self.n_nodes = None
        self.max_n_nodes = None
        self.node_types = None
        self.edge_types = None
        self.valency_distribution = None
        self.nodes_dist = None

        self.init_attributes(datamodule)

    def init_attributes(self, datamodule):
        self.valencies = [5, 4, 6, 6, 7, 1, 3, 7, 5, 4, 7, 4, 2, 4, 2, 6, 0]
        self.atom_weights = {
            1: 14.01, 2: 12.01, 3: 16., 4: 32.06, 5: 35.45, 6: 19., 7: 10.81, 8: 79.91, 9: 30.98,
            10: 28.01, 11: 126.9, 12: 118.71, 13: 24.31, 14: 63.55, 15: 65.38, 16: 78.97, 17: 0.0
        }

        if datamodule.extra_nodes:
            info_dir = f'{datamodule.data_root}/info_retrobridge_extra_nodes'
        else:
            info_dir = f'{datamodule.data_root}/info_retrobridge'

        os.makedirs(info_dir, exist_ok=True)

        if datamodule.evaluation and os.path.exists(f'{info_dir}/dummy_nodes_dist.txt'):
            self.dummy_nodes_dist = torch.tensor(np.loadtxt(f'{info_dir}/dummy_nodes_dist.txt'))
            self.n_nodes = torch.tensor(np.loadtxt(f'{info_dir}/n_counts.txt'))
            self.max_n_nodes = len(self.n_nodes) - 1
            self.node_types = torch.tensor(np.loadtxt(f'{info_dir}/atom_types.txt'))
            self.edge_types = torch.tensor(np.loadtxt(f'{info_dir}/edge_types.txt'))
            self.valency_distribution = torch.tensor(np.loadtxt(f'{info_dir}/valencies.txt'))
            self.nodes_dist = utils.DistributionNodes(self.n_nodes)
        else:
            self.dummy_nodes_dist = datamodule.dummy_atoms_counts(self.max_n_dummy_nodes)
            print("Distribution of number of dummy nodes", self.dummy_nodes_dist)
            np.savetxt(f'{info_dir}/dummy_nodes_dist.txt', self.dummy_nodes_dist.numpy())

            self.n_nodes = datamodule.node_counts()
            self.max_n_nodes = len(self.n_nodes) - 1
            print("Distribution of number of nodes", self.n_nodes)
            np.savetxt(f'{info_dir}/n_counts.txt', self.n_nodes.numpy())

            self.node_types = datamodule.node_types()
            print("Distribution of node types", self.node_types)
            np.savetxt(f'{info_dir}/atom_types.txt', self.node_types.numpy())

            self.edge_types = datamodule.edge_counts()
            print("Distribution of edge types", self.edge_types)
            np.savetxt(f'{info_dir}/edge_types.txt', self.edge_types.numpy())

            valencies = datamodule.valency_count(self.max_n_nodes)
            print("Distribution of the valencies", valencies)
            np.savetxt(f'{info_dir}/valencies.txt', valencies.numpy())
            self.valency_distribution = valencies
            self.nodes_dist = utils.DistributionNodes(self.n_nodes)

    def compute_input_output_dims(self, datamodule, extra_features, domain_features, use_context):
        example_batch = next(iter(datamodule.train_dataloader()))
        r_ex_dense, r_node_mask = utils.to_dense(
            example_batch.x,
            example_batch.edge_index,
            example_batch.edge_attr,
            example_batch.batch
        )
        p_ex_dense, p_node_mask = utils.to_dense(
            example_batch.p_x,
            example_batch.p_edge_index,
            example_batch.p_edge_attr,
            example_batch.batch
        )
        assert torch.all(r_node_mask == p_node_mask)

        p_example_data = {
            'X_t': p_ex_dense.X,
            'E_t': p_ex_dense.E,
            'y_t': example_batch['y'],
            'node_mask': p_node_mask
        }

        self.input_dims = {
            'X': example_batch['x'].size(1),
            'E': example_batch['edge_attr'].size(1),
            'y': example_batch['y'].size(1) + 1  # + 1 due to time conditioning
        }

        ex_extra_feat = extra_features(p_example_data)
        self.input_dims['X'] += ex_extra_feat.X.size(-1)
        self.input_dims['E'] += ex_extra_feat.E.size(-1)
        self.input_dims['y'] += ex_extra_feat.y.size(-1)

        ex_extra_molecular_feat = domain_features(p_example_data)
        self.input_dims['X'] += ex_extra_molecular_feat.X.size(-1)
        self.input_dims['E'] += ex_extra_molecular_feat.E.size(-1)
        self.input_dims['y'] += ex_extra_molecular_feat.y.size(-1)

        if use_context:
            self.input_dims['X'] += example_batch['x'].size(1)
            self.input_dims['E'] += example_batch['p_edge_attr'].size(1)

        self.output_dims = {
            'X': example_batch['x'].size(1),
            'E': example_batch['edge_attr'].size(1),
            'y': 0
        }

        print('Input dims:')
        for k, v in self.input_dims.items():
            print(f'\t{k} -> {v}')

        print('Output dims:')
        for k, v in self.output_dims.items():
            print(f'\t{k} -> {v}')


class RetroBridgeMITDatasetInfos(RetroBridgeDatasetInfos):
    atom_encoder = {
        'C': 0, 'O': 1, 'N': 2, 'Cl': 3, 'F': 4, 'S': 5, 'Na': 6, 'Br': 7, 'K': 8, 'P': 9, 'H': 10,
        'I': 11, 'B': 12, 'Li': 13, 'Si': 14, 'Pd': 15, 'Cs': 16, 'Al': 17, 'Cu': 18, 'Mg': 19, 'Sn': 20,
        'Zn': 21, 'Fe': 22, 'Cr': 23, 'Mn': 24, 'Ti': 25, 'Pt': 26, 'Ca': 27, 'Ag': 28, 'Se': 29, 'Ni': 30,
        'Ru': 31, 'Rh': 32, 'Co': 33, 'Os': 34, 'Ce': 35, 'Pb': 36, 'Ba': 37, 'Hg': 38, 'Zr': 39, 'As': 40,
        'Yb': 41, 'W': 42, 'Bi': 43, 'Ge': 44, 'In': 45, 'Sb': 46, 'Sc': 47, 'Tl': 48, 'Mo': 49, 'Sm': 50,
        'Re': 51, 'Ir': 52, 'Au': 53, 'Cd': 54, 'Ga': 55, 'Xe': 56, 'Nd': 57, 'Ta': 58, 'V': 59, 'La': 60,
        'Rb': 61, 'Dy': 62, 'Hf': 63, 'Y': 64, 'Te': 65, 'Ar': 66, 'Pr': 67, 'He': 68, 'Be': 69, 'Eu': 70,
        'Sr': 71, '*': 72,
    }
    atom_decoder = [
        'C', 'O', 'N', 'Cl', 'F', 'S', 'Na', 'Br', 'K', 'P', 'H', 'I', 'B', 'Li', 'Si', 'Pd', 'Cs', 'Al',
        'Cu', 'Mg', 'Sn', 'Zn', 'Fe', 'Cr', 'Mn', 'Ti', 'Pt', 'Ca', 'Ag', 'Se', 'Ni', 'Ru', 'Rh', 'Co',
        'Os', 'Ce', 'Pb', 'Ba', 'Hg', 'Zr', 'As', 'Yb', 'W', 'Bi', 'Ge', 'In', 'Sb', 'Sc', 'Tl', 'Mo',
        'Sm', 'Re', 'Ir', 'Au', 'Cd', 'Ga', 'Xe', 'Nd', 'Ta', 'V', 'La', 'Rb', 'Dy', 'Hf', 'Y', 'Te',
        'Ar', 'Pr', 'He', 'Be', 'Eu', 'Sr', '*'
    ]
    max_n_dummy_nodes = RetroBridgeDatasetInfos.max_n_dummy_nodes

    def init_attributes(self, datamodule):
        self.name = 'USPTO-MIT-RetroBridge'
        self.valencies = None
        self.atom_weights = None

        if datamodule.extra_nodes:
            info_dir = f'{datamodule.data_root}/info_retrobridge_extra_nodes'
        else:
            info_dir = f'{datamodule.data_root}/info_retrobridge'

        os.makedirs(info_dir, exist_ok=True)

        if True or datamodule.evaluation:
            self.dummy_nodes_dist = torch.tensor(np.loadtxt(f'{info_dir}/dummy_nodes_dist.txt'))
            self.n_nodes = torch.tensor(np.loadtxt(f'{info_dir}/n_counts.txt'))
            self.max_n_nodes = len(self.n_nodes) - 1
            self.node_types = torch.tensor(np.loadtxt(f'{info_dir}/atom_types.txt'))
            self.edge_types = torch.tensor(np.loadtxt(f'{info_dir}/edge_types.txt'))
            self.nodes_dist = utils.DistributionNodes(self.n_nodes)
        else:
            self.dummy_nodes_dist = datamodule.dummy_atoms_counts(self.max_n_dummy_nodes)
            print("Distribution of number of dummy nodes", self.dummy_nodes_dist)
            np.savetxt(f'{info_dir}/dummy_nodes_dist.txt', self.dummy_nodes_dist.numpy())

            self.n_nodes = datamodule.node_counts()
            self.max_n_nodes = len(self.n_nodes) - 1
            print("Distribution of number of nodes", self.n_nodes)
            np.savetxt(f'{info_dir}/n_counts.txt', self.n_nodes.numpy())

            self.node_types = datamodule.node_types()
            print("Distribution of node types", self.node_types)
            np.savetxt(f'{info_dir}/atom_types.txt', self.node_types.numpy())

            self.edge_types = datamodule.edge_counts()
            print("Distribution of edge types", self.edge_types)
            np.savetxt(f'{info_dir}/edge_types.txt', self.edge_types.numpy())

            self.nodes_dist = utils.DistributionNodes(self.n_nodes)

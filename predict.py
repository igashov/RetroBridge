import argparse
import pandas as pd
import torch

from src.utils import disable_rdkit_logging, set_deterministic
from src.analysis.rdkit_functions import build_molecule
from src.frameworks.markov_bridge import MarkovBridge
from src.data.retrobridge_dataset import RetroBridgeDatasetInfos, RetroBridgeDataset

from torch_geometric.data import Data
from rdkit import Chem

from pdb import set_trace


def assign_trivial_atom_mapping_numbers(molecule):
    order = {}
    for atom in molecule.GetAtoms():
        idx = atom.GetIdx()
        atom.SetAtomMapNum(idx)
        order[idx] = idx
    return molecule, order


def main(smiles, checkpoint, n_samples, n_steps, seed, device):
    set_deterministic(seed)

    # Loading the model
    model = MarkovBridge.load_from_checkpoint(checkpoint, map_location=device).to(device)
    model.T = n_steps

    # Preparing input
    pmol, mapping = assign_trivial_atom_mapping_numbers(Chem.MolFromSmiles(smiles))
    r_num_nodes = pmol.GetNumAtoms() + RetroBridgeDatasetInfos.max_n_dummy_nodes
    p_x, p_edge_index, p_edge_attr = RetroBridgeDataset.compute_graph(
        pmol, mapping, r_num_nodes, RetroBridgeDataset.types, RetroBridgeDataset.bonds
    )
    p_x = p_x.to(device)
    p_edge_index = p_edge_index.to(device)
    p_edge_attr = p_edge_attr.to(device)
    dataset, batch = [], []
    idx_offset = 0
    for i in range(n_samples):
        data = Data(idx=i, p_x=p_x, p_edge_index=p_edge_index.clone(), p_edge_attr=p_edge_attr, p_smiles=smiles)
        data.p_edge_index += idx_offset
        dataset.append(data)
        batch.append(torch.ones_like(data.p_x[:, 0]).to(torch.long) * i)
        idx_offset += len(data.p_x)

    data, _ = RetroBridgeDataset.collate(dataset)
    data.batch = torch.concat(batch)

    # Sampling
    _, _, _, _, molecule_list, _, _, _ = model.sample_chain(
        data, batch_size=n_samples, keep_chain=0, number_chain_steps_to_save=1, save_true_reactants=False
    )
    rdmols = []
    for mol in molecule_list:
        rdmol, _ = build_molecule(mol[0], mol[1], model.dataset_info.atom_decoder, return_n_dummy_atoms=True)
        smi = Chem.MolToSmiles(rdmol)
        rdmols.append(rdmol)
        print(smi)

    return rdmols


if __name__ == '__main__':
    disable_rdkit_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument('--smiles', action='store', type=str, required=True)
    parser.add_argument('--checkpoint', action='store', type=str, required=True)
    parser.add_argument('--n_samples', action='store', type=int, required=False, default=1)
    parser.add_argument('--n_steps', action='store', type=int, required=False, default=500)
    parser.add_argument('--seed', action='store', type=int, required=False, default=42)
    parser.add_argument('--device', action='store', type=str, required=False, default='cuda:0')
    args = parser.parse_args()
    main(
        smiles=args.smiles,
        checkpoint=args.checkpoint,
        n_samples=args.n_samples,
        n_steps=args.n_steps,
        seed=args.seed,
        device=args.device,
    )

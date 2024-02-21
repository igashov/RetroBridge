import sys; sys.path.append('src')

import argparse
import os
import pandas as pd

from src.utils import disable_rdkit_logging, parse_yaml_config, set_deterministic
from src.analysis.rdkit_functions import build_molecule
from src.frameworks.markov_bridge import MarkovBridge
from src.data.retrobridge_dataset import RetroBridgeMITDataModule, RetroBridgeMITDatasetInfos

from rdkit import Chem
from tqdm import tqdm

from pdb import set_trace


def main(args):
    torch_device = 'cuda:0' if args.device == 'gpu' else 'cpu'
    data_root = os.path.join(args.data, args.dataset)
    experiment_name = args.checkpoint.split('/')[-3]
    checkpoint_name = args.checkpoint.split('/')[-1].replace('.ckpt', '')

    output_dir = os.path.join(args.samples, experiment_name, f'{args.dataset}_{args.mode}')
    table_name = f'{checkpoint_name}_T={args.n_steps}_n={args.n_samples}_seed={args.sampling_seed}.csv'
    table_path = os.path.join(output_dir, table_name)

    skip_first_n = 0
    prev_table = pd.DataFrame()
    if os.path.exists(table_path):
        prev_table = pd.read_csv(table_path)
        skip_first_n = len(prev_table) // args.n_samples
        assert len(prev_table) % args.batch_size == 0

    print(f'Skipping first {skip_first_n} data points')

    os.makedirs(output_dir, exist_ok=True)
    print(f'Samples will be saved to {table_path}')

    # Loading model form checkpoint (all hparams will be automatically set)
    if args.model == 'RetroBridge':
        model_class = MarkovBridge
    else:
        raise NotImplementedError(args.model)

    print('Model class:', model_class)

    model = model_class.load_from_checkpoint(args.checkpoint, map_location=torch_device)
    datamodule = RetroBridgeMITDataModule(
        data_root=data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        extra_nodes=args.extra_nodes,
        evaluation=True,
        swap=args.swap,
    )
    dataset_infos = RetroBridgeMITDatasetInfos(datamodule)

    set_deterministic(args.sampling_seed)
    model.eval().to(torch_device)

    model.visualization_tools = None
    model.T = args.n_steps
    group_size = args.n_samples

    ident = 0
    true_molecules_smiles = []
    pred_molecules_smiles = []
    product_molecules_smiles = []
    computed_scores = []
    true_atom_nums = []
    sampled_atom_nums = []
    computed_nlls = []
    computed_ells = []

    dataloader = datamodule.test_dataloader() if args.mode == 'test' else datamodule.val_dataloader()

    for i, data in enumerate(tqdm(dataloader)):
        if i * args.batch_size < skip_first_n:
            continue

        bs = len(data.batch.unique())
        batch_groups = []
        batch_scores = []
        batch_nll = []
        batch_ell = []

        ground_truth = []
        input_products = []
        for sample_idx in range(group_size):
            data = data.to(torch_device)
            if args.model == 'OneShot':
                pred_molecule_list, true_molecule_list, products_list, scores, nlls, ells = model.sample_batch(
                    data=data,
                    batch_id=ident,
                    batch_size=bs,
                    save_final=0,
                    sample_idx=sample_idx,
                )
            elif args.model == 'DiGress':
                pred_molecule_list, true_molecule_list, products_list, scores, nlls, ells = model.sample_batch(
                    data=data,
                    batch_id=ident,
                    batch_size=bs,
                    save_final=0,
                    keep_chain=0,
                    number_chain_steps_to_save=1,
                    sample_idx=sample_idx,
                )
            else:
                pred_molecule_list, true_molecule_list, products_list, scores, nlls, ells = model.sample_batch(
                    data=data,
                    batch_id=ident,
                    batch_size=bs,
                    save_final=0,
                    keep_chain=0,
                    number_chain_steps_to_save=1,
                    sample_idx=sample_idx,
                    save_true_reactants=True,
                    use_one_hot=args.use_one_hot,
                )

            batch_groups.append(pred_molecule_list)
            batch_scores.append(scores)
            batch_nll.append(nlls)
            batch_ell.append(ells)

            if sample_idx == 0:
                ground_truth.extend(true_molecule_list)
                input_products.extend(products_list)

        # Regrouping sampled reactants for computing top-N accuracy
        grouped_samples = []
        grouped_scores = []
        grouped_nlls = []
        grouped_ells = []
        for mol_idx_in_batch in range(bs):
            mol_samples_group = []
            mol_scores_group = []
            nlls_group = []
            ells_group = []

            for batch_group, scores_group, nll_gr, ell_gr in zip(batch_groups, batch_scores, batch_nll, batch_ell):
                mol_samples_group.append(batch_group[mol_idx_in_batch])
                mol_scores_group.append(scores_group[mol_idx_in_batch])
                nlls_group.append(nll_gr[mol_idx_in_batch])
                ells_group.append(ell_gr[mol_idx_in_batch])

            assert len(mol_samples_group) == group_size
            grouped_samples.append(mol_samples_group)
            grouped_scores.append(mol_scores_group)
            grouped_nlls.append(nlls_group)
            grouped_ells.append(ells_group)

        # Writing smiles
        for true_mol, product_mol, pred_mols, pred_scores, nlls, ells in zip(
                ground_truth, input_products, grouped_samples, grouped_scores, grouped_nlls, grouped_ells,
        ):
            true_mol, true_n_dummy_atoms = build_molecule(
                true_mol[0], true_mol[1], dataset_infos.atom_decoder, return_n_dummy_atoms=True
            )
            true_smi = Chem.MolToSmiles(true_mol)

            product_mol = build_molecule(product_mol[0], product_mol[1], dataset_infos.atom_decoder)
            product_smi = Chem.MolToSmiles(product_mol)

            for pred_mol, pred_score, nll, ell in zip(pred_mols, pred_scores, nlls, ells):
                pred_mol, n_dummy_atoms = build_molecule(
                    pred_mol[0], pred_mol[1], dataset_infos.atom_decoder, return_n_dummy_atoms=True
                )
                pred_smi = Chem.MolToSmiles(pred_mol)
                true_molecules_smiles.append(true_smi)
                product_molecules_smiles.append(product_smi)
                pred_molecules_smiles.append(pred_smi)
                computed_scores.append(pred_score)
                true_atom_nums.append(RetroBridgeMITDatasetInfos.max_n_dummy_nodes - true_n_dummy_atoms)
                sampled_atom_nums.append(RetroBridgeMITDatasetInfos.max_n_dummy_nodes - n_dummy_atoms)
                computed_nlls.append(nll)
                computed_ells.append(ell)

        table = pd.DataFrame({
            'product': product_molecules_smiles,
            'pred': pred_molecules_smiles,
            'true': true_molecules_smiles,
            'score': computed_scores,
            'true_n_dummy_nodes': true_atom_nums,
            'sampled_n_dummy_nodes': sampled_atom_nums,
            'nll': computed_nlls,
            'ell': computed_ells,
        })
        full_table = pd.concat([prev_table, table])
        full_table.to_csv(table_path, index=False)


if __name__ == '__main__':
    disable_rdkit_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=argparse.FileType(mode='r'), required=True)
    parser.add_argument('--checkpoint', action='store', type=str, required=True)
    parser.add_argument('--samples', action='store', type=str, required=True)
    parser.add_argument('--model', action='store', type=str, required=True)
    parser.add_argument('--mode', action='store', type=str, required=True)
    parser.add_argument('--n_samples', action='store', type=int, required=True)
    parser.add_argument('--n_steps', action='store', type=int, required=False, default=None)
    parser.add_argument('--sampling_seed', action='store', type=int, required=False, default=None)
    parser.add_argument('--use_one_hot', action='store_true', required=False, default=False)
    main(args=parse_yaml_config(parser.parse_args()))

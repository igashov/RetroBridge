import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import os

from src.data import utils
from src.metrics.train_metrics import TrainLossDiscrete
from src.metrics.sampling_metrics import compute_retrosynthesis_metrics
from src.models.transformer_model import GraphTransformer

from torch_geometric.utils import scatter
from tqdm import tqdm

from pdb import set_trace


class OneShotModel(pl.LightningModule):
    def __init__(
            self,
            experiment_name,
            chains_dir,
            graphs_dir,
            checkpoints_dir,
            lr,
            weight_decay,
            n_layers,
            hidden_mlp_dims,
            hidden_dims,
            lambda_train,
            dataset_infos,
            train_metrics,
            sampling_metrics,
            visualization_tools,
            extra_features,
            domain_features,
            log_every_steps,
            sample_every_val,
            samples_to_generate,
            samples_to_save,
            samples_per_input,
    ):

        super().__init__()

        input_dims = dataset_infos.input_dims
        output_dims = dataset_infos.output_dims
        nodes_dist = dataset_infos.nodes_dist

        self.name = experiment_name
        self.chains_dir = chains_dir
        self.graphs_dir = graphs_dir
        self.checkpoints_dir = checkpoints_dir

        self.model_dtype = torch.float32

        self.lr = lr
        self.weight_decay = weight_decay

        self.Xdim = input_dims['X']
        self.Edim = input_dims['E']
        self.ydim = input_dims['y']
        self.Xdim_output = output_dims['X']
        self.Edim_output = output_dims['E']
        self.ydim_output = output_dims['y']
        self.node_dist = nodes_dist

        self.dataset_info = dataset_infos
        self.train_metrics = train_metrics
        self.train_loss = TrainLossDiscrete(lambda_train)
        self.val_loss = TrainLossDiscrete(lambda_train)
        self.sampling_metrics = sampling_metrics

        self.visualization_tools = visualization_tools
        self.extra_features = extra_features
        self.domain_features = domain_features

        self.model = GraphTransformer(
            n_layers=n_layers,
            input_dims=input_dims,
            hidden_mlp_dims=hidden_mlp_dims,
            hidden_dims=hidden_dims,
            output_dims=output_dims,
            act_fn_in=nn.ReLU(),
            act_fn_out=nn.ReLU()
        )

        self.save_hyperparameters(ignore=[train_metrics, sampling_metrics, visualization_tools, dataset_infos])

        self.start_epoch_time = None
        self.train_iterations = None
        self.val_iterations = None

        self.log_every_steps = log_every_steps
        self.sample_every_val = sample_every_val
        self.samples_to_generate = samples_to_generate
        self.samples_to_save = samples_to_save
        self.samples_per_input = samples_per_input
        self.val_counter = 0

    def configure_optimizers(self):
        return torch.optim.AdamW(
            params=self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            amsgrad=True,
        )

    def on_train_epoch_start(self):
        self.train_loss.reset()
        self.train_metrics.reset()

    def process_and_forward(self, data):
        # Getting graphs of reactants (target) and product (context)
        reactants, r_node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        reactants = reactants.mask(r_node_mask)

        product, p_node_mask = utils.to_dense(data.p_x, data.p_edge_index, data.p_edge_attr, data.batch)
        product = product.mask(p_node_mask)
        reactive_X_mask = (product.X[..., -1] == 1).unsqueeze(-1).bool()

        assert torch.allclose(r_node_mask, p_node_mask)
        node_mask = r_node_mask

        # Computing extra features + context and making predictions
        batch_size = product.X.shape[0]
        input_data = {
            't_int': torch.zeros((batch_size, 1), device=self.device),
            't': torch.zeros((batch_size, 1), device=self.device),
            'beta_t': torch.zeros((batch_size, 1), device=self.device),
            'alpha_s_bar': torch.zeros((batch_size, 1), device=self.device),
            'alpha_t_bar': torch.zeros((batch_size, 1), device=self.device),
            'X_t': product.X,
            'E_t': product.E,
            'y_t': product.y.to(product.X.device),
            'node_mask': node_mask
        }
        extra_data = self.compute_extra_data(input_data)
        pred = self.forward(input_data, extra_data, node_mask)
        pred.X = pred.X * reactive_X_mask + reactants.X * ~reactive_X_mask
        pred = pred.mask(node_mask)

        return reactants, product, pred, node_mask

    def training_step(self, data, i):
        reactants, product, pred, node_mask = self.process_and_forward(data)
        loss = self.train_loss(
            masked_pred_X=pred.X,
            masked_pred_E=pred.E,
            pred_y=pred.y,
            true_X=reactants.X,
            true_E=reactants.E,
            true_y=reactants.y,
        )
        self.train_metrics(
            masked_pred_X=pred.X,
            masked_pred_E=pred.E,
            true_X=reactants.X,
            true_E=reactants.E,
        )

        if i % self.log_every_steps == 0:
            self.log(f'train_loss/batch_CE', loss.detach())
            for metric_name, metric in self.train_loss.compute_metrics().items():
                self.log(f'train_loss/{metric_name}', metric)
            for metric_name, metric in self.train_metrics.compute_metrics().items():
                self.log(f'train_detailed/{metric_name}/train', metric)

            self.train_loss.reset()
            self.train_metrics.reset()

        return {'loss': loss}

    def on_validation_epoch_start(self) -> None:
        self.val_loss.reset()
        self.sampling_metrics.reset()

    def validation_step(self, data, i):
        reactants, product, pred, node_mask = self.process_and_forward(data)
        loss = self.val_loss(
            masked_pred_X=pred.X,
            masked_pred_E=pred.E,
            pred_y=pred.y,
            true_X=reactants.X,
            true_E=reactants.E,
            true_y=reactants.y,
        )

        if i % self.log_every_steps == 0:
            self.log(f'val_loss/batch_CE', loss.detach())
            for metric_name, metric in self.val_loss.compute_metrics().items():
                self.log(f'val_loss/{metric_name}', metric)

            self.train_loss.reset()
            self.train_metrics.reset()

        return {'loss': loss}

    def on_validation_epoch_end(self, outs):
        self.val_counter += 1
        if self.val_counter % self.sample_every_val == 0:
            self.sample()
            self.trainer.save_checkpoint(os.path.join(self.checkpoints_dir, 'last.ckpt'))

    def sample(self):
        samples_left_to_generate = self.samples_to_generate
        samples_left_to_save = self.samples_to_save

        samples = []
        grouped_samples = []
        ground_truth = []

        ident = 0
        print(f'Sampling epoch={self.current_epoch}')

        dataloader = self.trainer.datamodule.val_dataloader()
        for data in tqdm(dataloader, total=samples_left_to_generate // dataloader.batch_size):
            if samples_left_to_generate <= 0:
                break

            data = data.to(self.device)
            bs = len(data.batch.unique())
            to_generate = bs
            to_save = min(samples_left_to_save, bs)
            batch_groups = []
            for sample_idx in range(self.samples_per_input):
                molecule_list, true_molecule_list, products_list = self.sample_batch(
                    data=data,
                    batch_id=ident,
                    batch_size=to_generate,
                    save_final=to_save,
                    sample_idx=sample_idx,
                )
                samples.extend(molecule_list)
                batch_groups.append(molecule_list)
                if sample_idx == 0:
                    ground_truth.extend(true_molecule_list)

            ident += to_generate
            samples_left_to_save -= to_save
            samples_left_to_generate -= to_generate

            # Regrouping sampled reactants for computing top-N accuracy
            for mol_idx_in_batch in range(bs):
                mol_samples_group = []
                for batch_group in batch_groups:
                    mol_samples_group.append(batch_group[mol_idx_in_batch])
                assert len(mol_samples_group) == self.samples_per_input
                grouped_samples.append(mol_samples_group)

        to_log = compute_retrosynthesis_metrics(
            grouped_samples=grouped_samples,
            ground_truth=ground_truth,
            atom_decoder=self.dataset_info.atom_decoder
        )
        for metric_name, metric in to_log.items():
            self.log(metric_name, metric)

        to_log = self.sampling_metrics(samples)
        for metric_name, metric in to_log.items():
            self.log(metric_name, metric)

        self.sampling_metrics.reset()

    def forward(self, input_data, extra_data, node_mask):
        X = torch.cat((input_data['X_t'], extra_data.X), dim=2).float()
        E = torch.cat((input_data['E_t'], extra_data.E), dim=3).float()
        y = torch.hstack((input_data['y_t'], extra_data.y)).float()
        return self.model(X, E, y, node_mask)

    @torch.no_grad()
    def sample_batch(self, data, batch_id, batch_size, save_final, sample_idx):
        """
        :param data
        :param batch_id: int
        :param batch_size: int
        :param save_final: int: number of predictions to save to file
        :param sample_idx: int
        :return: molecule_list. Each element of this list is a tuple (atom_types, charges, positions)
        """

        # Context product
        product, node_mask = utils.to_dense(data.p_x, data.p_edge_index, data.p_edge_attr, data.batch)
        product = product.mask(node_mask)

        # Discrete context product
        product_discrete, _ = utils.to_dense(data.p_x, data.p_edge_index, data.p_edge_attr, data.batch)
        product_discrete = product_discrete.mask(node_mask, collapse=True)

        input_data = {
            't_int': torch.zeros((batch_size, 1), device=self.device),
            't': torch.zeros((batch_size, 1), device=self.device),
            'beta_t': torch.zeros((batch_size, 1), device=self.device),
            'alpha_s_bar': torch.zeros((batch_size, 1), device=self.device),
            'alpha_t_bar': torch.zeros((batch_size, 1), device=self.device),
            'X_t': product.X,
            'E_t': product.E,
            'y_t': product.y.to(product.X.device),
            'node_mask': node_mask
        }
        extra_data = self.compute_extra_data(input_data)
        pred = self.forward(input_data, extra_data, node_mask)

        pred = pred.mask(node_mask)
        pred.X = F.softmax(pred.X, dim=-1)
        pred.E = F.softmax(pred.E, dim=-1)
        pred = pred.mask(node_mask, collapse=True)

        X = pred.X
        E = pred.E

        # Saving true and predicted reactants
        true_molecule_list = utils.create_true_reactant_molecules(data, batch_size)
        products_list = utils.create_input_product_molecules(data, batch_size)
        molecule_list = utils.create_pred_reactant_molecules(X, E, data.batch, batch_size)

        current_samples_path = os.path.join(self.graphs_dir, f'epoch{self.current_epoch}_b{batch_id}')

        if self.visualization_tools is not None:
            if sample_idx == 0:
                # Visualize true reactants
                self.visualization_tools.visualize(
                    path=current_samples_path,
                    molecules=true_molecule_list,
                    num_molecules_to_visualize=save_final,
                    prefix='true_',
                )

                # Visualize input products
                self.visualization_tools.visualize(
                    path=current_samples_path,
                    molecules=products_list,
                    num_molecules_to_visualize=save_final,
                    prefix='input_product_',
                )

            # Visualize predicted reactants
            self.visualization_tools.visualize(
                path=current_samples_path,
                molecules=molecule_list,
                num_molecules_to_visualize=save_final,
                prefix=f'pred_',
                suffix=f'_{sample_idx}'
            )

        return (
            molecule_list, true_molecule_list, products_list,
            [0] * len(molecule_list), [0] * len(molecule_list), [0] * len(molecule_list)
        )

    def compute_extra_data(self, input_data):
        """ At every training step (after adding noise) and step in sampling, compute extra information and append to
            the network input. """

        extra_features = self.extra_features(input_data)
        extra_molecular_features = self.domain_features(input_data)

        extra_X = torch.cat((extra_features.X, extra_molecular_features.X), dim=-1)
        extra_E = torch.cat((extra_features.E, extra_molecular_features.E), dim=-1)
        extra_y = torch.cat((extra_features.y, extra_molecular_features.y), dim=-1)

        t = input_data['t']
        extra_y = torch.cat((extra_y, t), dim=1)
        return utils.PlaceHolder(X=extra_X, E=extra_E, y=extra_y)

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import os

from src.models.transformer_model import GraphTransformer
from src.frameworks.noise_schedule import (
    DiscreteUniformTransition,
    PredefinedNoiseScheduleDiscrete,
    MarginalUniformTransition,
)
from src.frameworks import diffusion_utils
from src.metrics.train_metrics import TrainLossDiscrete
from src.metrics.abstract_metrics import SumExceptBatchMetric, SumExceptBatchKL, NLL
from src.metrics.sampling_metrics import compute_retrosynthesis_metrics
from src.data import utils

from tqdm import tqdm

from pdb import set_trace


class DiscreteDiffusion(pl.LightningModule):
    def __init__(
            self,
            experiment_name,
            chains_dir,
            graphs_dir,
            checkpoints_dir,
            diffusion_steps,
            diffusion_noise_schedule,
            transition,
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
            chains_to_save,
            number_chain_steps_to_save,
            fix_product_nodes,
            use_context=True,
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
        self.T = diffusion_steps
        self.transition = transition

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
        self.train_loss = TrainLossDiscrete(lambda_train)

        self.val_nll = NLL()
        self.val_X_kl = SumExceptBatchKL()
        self.val_E_kl = SumExceptBatchKL()
        self.val_X_logp = SumExceptBatchMetric()
        self.val_E_logp = SumExceptBatchMetric()

        self.test_nll = NLL()
        self.test_X_kl = SumExceptBatchKL()
        self.test_E_kl = SumExceptBatchKL()
        self.test_X_logp = SumExceptBatchMetric()
        self.test_E_logp = SumExceptBatchMetric()

        self.train_metrics = train_metrics
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
        self.noise_schedule = PredefinedNoiseScheduleDiscrete(
            noise_schedule=diffusion_noise_schedule,
            timesteps=diffusion_steps,
        )

        if self.transition == 'uniform':
            self.transition_model = DiscreteUniformTransition(
                x_classes=self.Xdim_output,
                e_classes=self.Edim_output,
                y_classes=self.ydim_output
            )
            x_limit = torch.ones(self.Xdim_output) / self.Xdim_output
            e_limit = torch.ones(self.Edim_output) / self.Edim_output
            y_limit = torch.ones(self.ydim_output) / self.ydim_output
            self.limit_dist = utils.PlaceHolder(X=x_limit, E=e_limit, y=y_limit)

        elif self.transition == 'marginal':
            node_types = self.dataset_info.node_types.float()
            x_marginals = node_types / torch.sum(node_types)
            edge_types = self.dataset_info.edge_types.float()
            e_marginals = edge_types / torch.sum(edge_types)

            self.transition_model = MarginalUniformTransition(
                x_marginals=x_marginals,
                e_marginals=e_marginals,
                y_classes=self.ydim_output
            )
            y_limit = torch.ones(self.ydim_output) / self.ydim_output
            self.limit_dist = utils.PlaceHolder(X=x_marginals, E=e_marginals, y=y_limit)

        else:
            raise NotImplementedError

        self.save_hyperparameters(ignore=[train_metrics, sampling_metrics, visualization_tools, dataset_infos])

        self.start_epoch_time = None
        self.train_iterations = None
        self.val_iterations = None

        self.number_chain_steps_to_save = number_chain_steps_to_save
        self.log_every_steps = log_every_steps
        self.sample_every_val = sample_every_val
        self.samples_to_generate = samples_to_generate
        self.samples_to_save = samples_to_save
        self.samples_per_input = samples_per_input
        self.chains_to_save = chains_to_save
        self.val_counter = 0
        self.fix_product_nodes = fix_product_nodes
        self.use_context = use_context

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

        assert torch.allclose(r_node_mask, p_node_mask)
        node_mask = r_node_mask

        # Fixed and unchanged node masks
        fixed_nodes = (product.X[..., -1] == 0).unsqueeze(-1)
        modifiable_nodes = (product.X[..., -1] == 1).unsqueeze(-1)
        assert torch.all(fixed_nodes | modifiable_nodes)

        # Getting noisy data
        noisy_data = self.apply_noise(reactants.X, reactants.E, data.y, node_mask)

        # Masking unchanged part
        if self.fix_product_nodes:
            noisy_data['X_t'] = noisy_data['X_t'] * modifiable_nodes + product.X * fixed_nodes
            noisy_data['X_t'] = noisy_data['X_t'] * node_mask.unsqueeze(-1)

        # Computing extra features + context and making predictions
        context = product.clone() if self.use_context else None
        extra_data = self.compute_extra_data(noisy_data, context=context)
        pred = self.forward(noisy_data, extra_data, node_mask)

        # Masking unchanged part
        if self.fix_product_nodes:
            pred.X = pred.X * modifiable_nodes + product.X * fixed_nodes
            pred.X = pred.X * node_mask.unsqueeze(-1)

        return reactants, product, pred, node_mask, noisy_data, context

    def training_step(self, data, i):
        reactants, product, pred, node_mask, _, _ = self.process_and_forward(data)
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
        self.val_nll.reset()
        self.val_X_kl.reset()
        self.val_E_kl.reset()
        self.val_X_logp.reset()
        self.val_E_logp.reset()
        self.sampling_metrics.reset()

    def validation_step(self, data, i):
        reactants, product, pred, node_mask, noisy_data, context = self.process_and_forward(data)
        nll = self.compute_val_loss(
            pred=pred,
            noisy_data=noisy_data,
            X=reactants.X,
            E=reactants.E,
            y=reactants.y,
            node_mask=node_mask,
            context=context,
            test=False,
        )
        return {'loss': nll}

    def on_validation_epoch_end(self, outs):
        self.log(f'val/epoch_NLL', self.val_nll.compute())
        self.log(f'val/X_kl', self.val_X_kl.compute() * self.T)
        self.log(f'val/E_kl', self.val_E_kl.compute() * self.T)
        self.log(f'val/X_logp', self.val_X_logp.compute())
        self.log(f'val/E_logp', self.val_E_logp.compute())

        self.val_counter += 1
        if self.val_counter % self.sample_every_val == 0:
            self.sample()
            self.trainer.save_checkpoint(os.path.join(self.checkpoints_dir, 'last.ckpt'))

    def sample(self):
        samples_left_to_generate = self.samples_to_generate
        samples_left_to_save = self.samples_to_save
        chains_left_to_save = self.chains_to_save

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
            chains_save = min(chains_left_to_save, bs)
            batch_groups = []
            for sample_idx in range(self.samples_per_input):
                molecule_list, true_molecule_list, products_list = self.sample_batch(
                    data=data,
                    batch_id=ident,
                    batch_size=to_generate,
                    save_final=to_save,
                    keep_chain=chains_save,
                    number_chain_steps_to_save=self.number_chain_steps_to_save,
                    sample_idx=sample_idx,
                )
                samples.extend(molecule_list)
                batch_groups.append(molecule_list)
                if sample_idx == 0:
                    ground_truth.extend(true_molecule_list)

            ident += to_generate
            samples_left_to_save -= to_save
            samples_left_to_generate -= to_generate
            chains_left_to_save -= chains_save

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

    def kl_prior(self, X, E, node_mask):
        """Computes the KL between q(z1 | x) and the prior p(z1) = Normal(0, 1).

        This is essentially a lot of work for something that is in practice negligible in the loss. However, you
        compute it so that you see it when you've made a mistake in your noise schedule.
        """
        # Compute the last alpha value, alpha_T.
        ones = torch.ones((X.size(0), 1), device=X.device)
        Ts = self.T * ones
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_int=Ts)  # (bs, 1)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)

        # Compute transition probabilities
        probX = X @ Qtb.X  # (bs, n, dx_out)
        probE = E @ Qtb.E.unsqueeze(1)  # (bs, n, n, de_out)
        assert probX.shape == X.shape

        bs, n, _ = probX.shape
        limit_X = self.limit_dist.X[None, None, :].expand(bs, n, -1).type_as(probX)
        limit_E = self.limit_dist.E[None, None, None, :].expand(bs, n, n, -1).type_as(probE)

        # Make sure that masked rows do not contribute to the loss
        limit_dist_X, limit_dist_E, probX, probE = diffusion_utils.mask_distributions(
            true_X=limit_X.clone(),
            true_E=limit_E.clone(),
            pred_X=probX,
            pred_E=probE,
            node_mask=node_mask
        )

        kl_distance_X = F.kl_div(input=probX.log(), target=limit_dist_X, reduction='none')
        kl_distance_E = F.kl_div(input=probE.log(), target=limit_dist_E, reduction='none')

        return diffusion_utils.sum_except_batch(kl_distance_X) + diffusion_utils.sum_except_batch(kl_distance_E)

    def compute_Lt(self, X, E, y, pred, noisy_data, node_mask, test):
        pred_probs_X = F.softmax(pred.X, dim=-1)
        pred_probs_E = F.softmax(pred.E, dim=-1)
        pred_probs_y = F.softmax(pred.y, dim=-1)

        Qtb = self.transition_model.get_Qt_bar(noisy_data['alpha_t_bar'], self.device)
        Qsb = self.transition_model.get_Qt_bar(noisy_data['alpha_s_bar'], self.device)
        Qt = self.transition_model.get_Qt(noisy_data['beta_t'], self.device)

        # Compute distributions to compare with KL
        bs, n, d = X.shape
        prob_true = diffusion_utils.posterior_distributions(X=X, E=E, y=y, X_t=noisy_data['X_t'], E_t=noisy_data['E_t'],
                                                            y_t=noisy_data['y_t'], Qt=Qt, Qsb=Qsb, Qtb=Qtb)
        prob_true.E = prob_true.E.reshape((bs, n, n, -1))
        prob_pred = diffusion_utils.posterior_distributions(X=pred_probs_X, E=pred_probs_E, y=pred_probs_y,
                                                            X_t=noisy_data['X_t'], E_t=noisy_data['E_t'],
                                                            y_t=noisy_data['y_t'], Qt=Qt, Qsb=Qsb, Qtb=Qtb)
        prob_pred.E = prob_pred.E.reshape((bs, n, n, -1))

        # Reshape and filter masked rows
        prob_true_X, prob_true_E, prob_pred.X, prob_pred.E = diffusion_utils.mask_distributions(true_X=prob_true.X,
                                                                                                true_E=prob_true.E,
                                                                                                pred_X=prob_pred.X,
                                                                                                pred_E=prob_pred.E,
                                                                                                node_mask=node_mask)
        kl_x = (self.test_X_kl if test else self.val_X_kl)(prob_true.X, torch.log(prob_pred.X))
        kl_e = (self.test_E_kl if test else self.val_E_kl)(prob_true.E, torch.log(prob_pred.E))
        return self.T * (kl_x + kl_e)

    def reconstruction_logp(self, t, X, E, node_mask, context=None):
        # Compute noise values for t = 0.
        t_zeros = torch.zeros_like(t)
        beta_0 = self.noise_schedule(t_zeros)
        Q0 = self.transition_model.get_Qt(beta_t=beta_0, device=self.device)

        probX0 = X @ Q0.X  # (bs, n, dx_out)
        probE0 = E @ Q0.E.unsqueeze(1)  # (bs, n, n, de_out)

        sampled0 = diffusion_utils.sample_discrete_features(probX=probX0, probE=probE0, node_mask=node_mask)

        X0 = F.one_hot(sampled0.X, num_classes=self.Xdim_output).float()
        E0 = F.one_hot(sampled0.E, num_classes=self.Edim_output).float()
        y0 = sampled0.y
        assert (X.shape == X0.shape) and (E.shape == E0.shape)

        sampled_0 = utils.PlaceHolder(X=X0, E=E0, y=y0).mask(node_mask)

        # Predictions
        noisy_data = {
            'X_t': sampled_0.X,
            'E_t': sampled_0.E,
            'y_t': sampled_0.y,
            'node_mask': node_mask,
            't': torch.zeros(X0.shape[0], 1).type_as(y0)
        }
        extra_data = self.compute_extra_data(noisy_data, context=context)
        pred0 = self.forward(noisy_data, extra_data, node_mask)

        # Normalize predictions
        probX0 = F.softmax(pred0.X, dim=-1)
        probE0 = F.softmax(pred0.E, dim=-1)
        proby0 = F.softmax(pred0.y, dim=-1)

        # Set masked rows to arbitrary values that don't contribute to loss
        probX0[~node_mask] = torch.ones(self.Xdim_output).type_as(probX0)
        probE0[~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2))] = torch.ones(self.Edim_output).type_as(probE0)

        diag_mask = torch.eye(probE0.size(1)).type_as(probE0).bool()
        diag_mask = diag_mask.unsqueeze(0).expand(probE0.size(0), -1, -1)
        probE0[diag_mask] = torch.ones(self.Edim_output).type_as(probE0)

        return utils.PlaceHolder(X=probX0, E=probE0, y=proby0)

    def apply_noise(self, X, E, y, node_mask):
        """ Sample noise and apply it to the data. """

        lowest_t = 0 if self.training else 1
        t_int = torch.randint(lowest_t, self.T + 1, size=(X.size(0), 1), device=X.device).float()  # (bs, 1)
        s_int = t_int - 1

        t_float = t_int / self.T
        s_float = s_int / self.T

        # beta_t and alpha_s_bar are used for denoising/loss computation
        beta_t = self.noise_schedule(t_normalized=t_float)                         # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_float)      # (bs, 1)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)      # (bs, 1)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device=self.device)  # (bs, dx, dx), (bs, de, de)
        assert (abs(Qtb.X.sum(dim=2) - 1.) < 1e-4).all(), Qtb.X.sum(dim=2) - 1
        assert (abs(Qtb.E.sum(dim=2) - 1.) < 1e-4).all()

        # Compute transition probabilities
        probX = X @ Qtb.X  # (bs, n, dx_out)
        probE = E @ Qtb.E.unsqueeze(1)  # (bs, n, n, de_out)

        sampled_t = diffusion_utils.sample_discrete_features(probX=probX, probE=probE, node_mask=node_mask)

        X_t = F.one_hot(sampled_t.X, num_classes=self.Xdim_output)
        E_t = F.one_hot(sampled_t.E, num_classes=self.Edim_output)
        assert (X.shape == X_t.shape) and (E.shape == E_t.shape)

        z_t = utils.PlaceHolder(X=X_t, E=E_t, y=y).type_as(X_t).mask(node_mask)
        noisy_data = {
            't_int': t_int,
            't': t_float,
            'beta_t': beta_t,
            'alpha_s_bar': alpha_s_bar,
            'alpha_t_bar': alpha_t_bar,
            'X_t': z_t.X,
            'E_t': z_t.E,
            'y_t': z_t.y,
            'node_mask': node_mask
        }
        return noisy_data

    def compute_val_loss(self, pred, noisy_data, X, E, y, node_mask, context=None, test=False):
        """Computes an estimator for the variational lower bound.
           pred: (batch_size, n, total_features)
           noisy_data: dict
           X, E, y : (bs, n, dx),  (bs, n, n, de), (bs, dy)
           node_mask : (bs, n)
           Output: nll (size 1)
       """
        t = noisy_data['t']

        # 1.
        N = node_mask.sum(1).long()
        log_pN = self.node_dist.log_prob(N)

        # 2. The KL between q(z_T | x) and p(z_T) = Uniform(1/num_classes). Should be close to zero.
        kl_prior = self.kl_prior(X, E, node_mask)

        # 3. Diffusion loss
        loss_all_t = self.compute_Lt(X, E, y, pred, noisy_data, node_mask, test)

        # 4. Reconstruction loss
        # Compute L0 term : -log p (X, E, y | z_0) = reconstruction loss
        prob0 = self.reconstruction_logp(t, X, E, node_mask, context=context)

        loss_term_0 = self.val_X_logp(X * prob0.X.log()) + self.val_E_logp(E * prob0.E.log())

        # Combine terms
        nlls = - log_pN + kl_prior + loss_all_t - loss_term_0
        assert len(nlls.shape) == 1, f'{nlls.shape} has more than only batch dim.'

        # Update NLL metric object and return batch nll
        nll = (self.test_nll if test else self.val_nll)(nlls)        # Average over the batch
        return nll

    def forward(self, noisy_data, extra_data, node_mask):
        X = torch.cat((noisy_data['X_t'], extra_data.X), dim=2).float()
        E = torch.cat((noisy_data['E_t'], extra_data.E), dim=3).float()
        y = torch.hstack((noisy_data['y_t'], extra_data.y)).float()
        return self.model(X, E, y, node_mask)

    @torch.no_grad()
    def sample_batch(self, data, batch_id, batch_size, keep_chain, number_chain_steps_to_save, save_final, sample_idx):
        """
        :param data
        :param batch_id: int
        :param batch_size: int
        :param number_chain_steps_to_save: int
        :param save_final: int: number of predictions to save to file
        :param keep_chain: int: number of chains to save to file
        :param sample_idx: int
        :return: molecule_list. Each element of this list is a tuple (atom_types, charges, positions)
        """

        # Context product
        product, node_mask = utils.to_dense(data.p_x, data.p_edge_index, data.p_edge_attr, data.batch)
        product = product.mask(node_mask)

        # Creating context
        context = product.clone() if self.use_context else None

        # Masks for fixed and modifiable nodes
        fixed_nodes = (product.X[..., -1] == 0).unsqueeze(-1)
        modifiable_nodes = (product.X[..., -1] == 1).unsqueeze(-1)
        assert torch.all(fixed_nodes | modifiable_nodes)

        # Initial noise
        z_T = diffusion_utils.sample_discrete_feature_noise(limit_dist=self.limit_dist, node_mask=node_mask)
        X = z_T.X
        E = z_T.E
        y = z_T.y

        if self.fix_product_nodes:
            X = X * modifiable_nodes + product.X * fixed_nodes
            X = X * node_mask.unsqueeze(-1)

        assert (E == torch.transpose(E, 1, 2)).all()
        assert number_chain_steps_to_save < self.T

        chain_X_size = torch.Size((number_chain_steps_to_save, keep_chain, X.size(1)))
        chain_E_size = torch.Size((number_chain_steps_to_save, keep_chain, E.size(1), E.size(2)))

        chain_X = torch.zeros(chain_X_size)
        chain_E = torch.zeros(chain_E_size)

        sampled_s = None

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s_int in tqdm(reversed(range(0, self.T)), total=self.T):
            s_array = s_int * torch.ones((batch_size, 1)).type_as(y)
            t_array = s_array + 1
            s_norm = s_array / self.T
            t_norm = t_array / self.T

            # Sample z_s
            sampled_s, discrete_sampled_s = self.sample_p_zs_given_zt(
                s=s_norm,
                t=t_norm,
                X_t=X,
                E_t=E,
                y_t=y,
                node_mask=node_mask,
                context=context,
            )

            if self.fix_product_nodes:
                sampled_s.X = sampled_s.X * modifiable_nodes + product.X * fixed_nodes
                sampled_s = sampled_s.mask(node_mask)
                discrete_sampled_s = sampled_s.clone()
                discrete_sampled_s = discrete_sampled_s.mask(node_mask, collapse=True)

            X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

            # Save the first keep_chain graphs
            write_index = (s_int * number_chain_steps_to_save) // self.T
            chain_X[write_index] = discrete_sampled_s.X[:keep_chain]
            chain_E[write_index] = discrete_sampled_s.E[:keep_chain]

        # Sample
        sampled_s = sampled_s.mask(node_mask, collapse=True)
        X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

        # Prepare the chain for saving
        if keep_chain > 0:
            final_X_chain = X[:keep_chain]
            final_E_chain = E[:keep_chain]

            chain_X[0] = final_X_chain  # Overwrite last frame with the resulting X, E
            chain_E[0] = final_E_chain

            chain_X = diffusion_utils.reverse_tensor(chain_X)
            chain_E = diffusion_utils.reverse_tensor(chain_E)

            # Repeat last frame to see final sample better
            chain_X = torch.cat([chain_X, chain_X[-1:].repeat(10, 1, 1)], dim=0)
            chain_E = torch.cat([chain_E, chain_E[-1:].repeat(10, 1, 1, 1)], dim=0)
            assert chain_X.shape[0] == (number_chain_steps_to_save + 10)

        # Saving true and predicted reactants
        true_molecule_list = utils.create_true_reactant_molecules(data, batch_size)
        products_list = utils.create_input_product_molecules(data, batch_size)
        molecule_list = utils.create_pred_reactant_molecules(X, E, data.batch, batch_size)

        current_samples_path = os.path.join(self.graphs_dir, f'epoch{self.current_epoch}_b{batch_id}')
        current_chains_dir = os.path.join(self.chains_dir, f'epoch_{self.current_epoch}')

        if self.visualization_tools is not None:
            if sample_idx == 0:

                # 1. Visualize chains
                num_molecules = chain_X.shape[1]
                for i in range(num_molecules):
                    results_path = os.path.join(current_chains_dir, f'molecule_{batch_id + i}')
                    os.makedirs(results_path, exist_ok=True)
                    self.visualization_tools.visualize_chain(
                        path=results_path,
                        nodes_list=chain_X[:, i, :].numpy(),
                        adjacency_matrix=chain_E[:, i, :].numpy(),
                    )

                # 2. Visualize true reactants
                self.visualization_tools.visualize(
                    path=current_samples_path,
                    molecules=true_molecule_list,
                    num_molecules_to_visualize=save_final,
                    prefix='true_',
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

    def sample_p_zs_given_zt(self, s, t, X_t, E_t, y_t, node_mask, context=None):
        """Samples from zs ~ p(zs | zt). Only used during sampling.
           if last_step, return the graph prediction as well"""
        bs, n, dxs = X_t.shape
        beta_t = self.noise_schedule(t_normalized=t)  # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t)

        # Retrieve transitions matrix
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)  # (bs, dx, dx), (bs, de, de)
        Qsb = self.transition_model.get_Qt_bar(alpha_s_bar, self.device)  # (bs, dx, dx), (bs, de, de)
        Qt = self.transition_model.get_Qt(beta_t, self.device)  # (bs, dx, dx), (bs, de, de)

        # Neural net predictions
        noisy_data = {'X_t': X_t, 'E_t': E_t, 'y_t': y_t, 't': t, 'node_mask': node_mask}
        extra_data = self.compute_extra_data(noisy_data, context=context)
        pred = self.forward(noisy_data, extra_data, node_mask)

        # Normalize predictions
        pred_X = F.softmax(pred.X, dim=-1)               # bs, n, d0
        pred_E = F.softmax(pred.E, dim=-1)               # bs, n, n, d0

        p_s_and_t_given_0_X = diffusion_utils.compute_batched_over0_posterior_distribution(
            X_t=X_t,
            Qt=Qt.X,
            Qsb=Qsb.X,
            Qtb=Qtb.X
        )
        p_s_and_t_given_0_E = diffusion_utils.compute_batched_over0_posterior_distribution(
            X_t=E_t,
            Qt=Qt.E,
            Qsb=Qsb.E,
            Qtb=Qtb.E
        )

        # Dim of these two tensors: bs, N, d0, d_t-1
        weighted_X = pred_X.unsqueeze(-1) * p_s_and_t_given_0_X         # bs, n, d0, d_t-1
        unnormalized_prob_X = weighted_X.sum(dim=2)                     # bs, n, d_t-1
        unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-5
        prob_X = unnormalized_prob_X / torch.sum(unnormalized_prob_X, dim=-1, keepdim=True)  # bs, n, d_t-1

        pred_E = pred_E.reshape((bs, -1, pred_E.shape[-1]))
        weighted_E = pred_E.unsqueeze(-1) * p_s_and_t_given_0_E        # bs, N, d0, d_t-1
        unnormalized_prob_E = weighted_E.sum(dim=-2)
        unnormalized_prob_E[torch.sum(unnormalized_prob_E, dim=-1) == 0] = 1e-5
        prob_E = unnormalized_prob_E / torch.sum(unnormalized_prob_E, dim=-1, keepdim=True)
        prob_E = prob_E.reshape(bs, n, n, pred_E.shape[-1])

        assert ((prob_X.sum(dim=-1) - 1).abs() < 1e-4).all()
        assert ((prob_E.sum(dim=-1) - 1).abs() < 1e-4).all()

        sampled_s = diffusion_utils.sample_discrete_features(prob_X, prob_E, node_mask=node_mask)

        X_s = F.one_hot(sampled_s.X, num_classes=self.Xdim_output).float()
        E_s = F.one_hot(sampled_s.E, num_classes=self.Edim_output).float()

        assert (E_s == torch.transpose(E_s, 1, 2)).all()
        assert (X_t.shape == X_s.shape) and (E_t.shape == E_s.shape)

        out_one_hot = utils.PlaceHolder(X=X_s, E=E_s, y=torch.zeros(y_t.shape[0], 0))
        out_discrete = utils.PlaceHolder(X=X_s, E=E_s, y=torch.zeros(y_t.shape[0], 0))

        return out_one_hot.mask(node_mask).type_as(y_t), out_discrete.mask(node_mask, collapse=True).type_as(y_t)

    def compute_extra_data(self, noisy_data, context=None, condition_on_t=True):
        """ At every training step (after adding noise) and step in sampling, compute extra information and append to
            the network input. """

        extra_features = self.extra_features(noisy_data)
        extra_molecular_features = self.domain_features(noisy_data)

        extra_X = torch.cat((extra_features.X, extra_molecular_features.X), dim=-1)
        extra_E = torch.cat((extra_features.E, extra_molecular_features.E), dim=-1)
        extra_y = torch.cat((extra_features.y, extra_molecular_features.y), dim=-1)

        if context is not None:
            extra_X = torch.cat((extra_X, context.X), dim=-1)
            extra_E = torch.cat((extra_E, context.E), dim=-1)

        if condition_on_t:
            t = noisy_data['t']
            extra_y = torch.cat((extra_y, t), dim=1)

        return utils.PlaceHolder(X=extra_X, E=extra_E, y=extra_y)

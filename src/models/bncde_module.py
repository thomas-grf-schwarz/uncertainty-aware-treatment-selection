import torch
from lightning import LightningModule
import torch.nn as nn
import torchsde
import torchcde
import torch.nn.functional as F
from torch import distributions as dist
import torch.optim as optim

from torchmetrics import Metric
from torch.distributions import Normal

from src.utils.figures import plot_trajectories_with_uncertainty
import wandb

from src.utils.transforms import (
    LayerNorm1D, 
    FixedAllBackNorm,
    FixedTransferableNorm,
    FixedLayerNorm1D,
    )

class GaussianMixture:
    
    def __init__(self, means, stds): # input shape: n_obs x n_components
        self.n_obs = means.shape[0]
        self.num_components = means.shape[-1]
        self.components = [Normal(means[:, i], stds[:, i]) for i in range(self.num_components)]
    
    def sample(self, mc_samples): # n_samples = number of MC samples drawn from the mixture model
        component_idx = torch.randint(0, self.num_components,
                                      (mc_samples,))  # draw mc_samples from randomly chosen component for each obs

        return torch.stack([self.components[idx].sample() for idx in component_idx]).T # output shape: n_obs x mc_samples

    def log_prob(self, value): # input shape: either grid_shape or grid_shape x n_obs
        if len(value.shape) < self.n_obs:
            value = torch.stack([value for _ in range(self.n_obs)]).T

        component_log_probs = torch.stack([component.log_prob(value) for component in self.components])
        log_prob = torch.logsumexp(component_log_probs, dim=0) - torch.log(torch.tensor(self.num_components)).T
        return log_prob # grid_shape x n_obs

class CoverageMetric(Metric):
    def __init__(self, confidence_level=0.95, gaussian_mixture_components=False):
        super().__init__()
        self.confidence_level = confidence_level
        self.mixture = gaussian_mixture_components
        self.add_state("counter", default=torch.tensor(0))
        self.add_state("total_samples", default=torch.tensor(0))
        self.add_state("sizes", default=torch.tensor([0]))

    def update(self, y_pred, y_true):
        
        if self.mixture:
            gm = GaussianMixture(means=y_pred[:, :, 0].T, stds=y_pred[:, :, 1].T)
            y_pred = gm.sample(int(1e4))
        else:
            y_pred = y_pred.T
        lower = (1 - self.confidence_level) / 2
        upper = 1 - lower
        lower_quantile = torch.quantile(y_pred, lower, dim=1)
        upper_quantile = torch.quantile(y_pred, upper, dim=1)
        in_quantiles = torch.logical_and(y_true >= lower_quantile, y_true <= upper_quantile)

        self.counter += torch.sum(in_quantiles).item()
        self.total_samples += (y_true.shape[0]-torch.isnan(y_true).sum())
        self.sizes = torch.concatenate((self.sizes, (upper_quantile[~torch.isnan(y_true)]-lower_quantile[~torch.isnan(y_true)])))


    def compute(self):
        print('coverage:'+str(self.counter / self.total_samples))
        return self.counter / self.total_samples


    def size(self):
        return torch.median(self.sizes[1:]) # skip zero value from initialization


class SequentialModel(nn.Module): # customize weight drifts
    def __init__(self, input_size, output_size, hidden_layers):
        super(SequentialModel, self).__init__()

        # Define layers
        layers = [nn.Linear(input_size, hidden_layers[0]), nn.ReLU()]
        for i in range(1, len(hidden_layers)):
            layers.extend([nn.Linear(hidden_layers[i - 1], hidden_layers[i]), nn.ReLU()])
        layers.append(nn.Linear(hidden_layers[-1], output_size))

        # Sequential model
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)


class MC_trainable_net(LightningModule): # prediction head
    def __init__(self, input_size, output_size, hidden_sizes=None):
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = []
        elif isinstance(hidden_sizes, int):
            hidden_sizes = (hidden_sizes, )

        layer_sizes = [input_size] + list(hidden_sizes) + [output_size]
        self.layers = nn.ModuleList()

        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            nn.init.constant_(self.layers[i].bias, 0.0)

    def forward(self, x):
        # Pass the input through each layer
        for layer in self.layers[:-1]:
            x = layer(x)
            x = F.relu(x)

        x = self.layers[-1](x)
        return x

########################################################################################################################
# MC non-trainable network (neural CDE vector field)
class MC_nontrainable_net(LightningModule):
    def __init__(self, n_nets, input_size, output_size, hidden_sizes=None):
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = []
        elif isinstance(hidden_sizes, int):
            hidden_sizes = (hidden_sizes, )

        # Input size:       n_nets x batch_size x input_size
        # Shape of weights: n_nets x input_size x output_size
        # shape of biases:  n_nets x     1      x output_size
        hidden_sizes = [input_size] + list(hidden_sizes)  # Include input size as the first hidden size
        self.hidden_weights = []
        self.hidden_biases = []

        # Initialize weights and biases for each hidden layer
        for i in range(len(hidden_sizes) - 1):
            weight = torch.randn(n_nets, hidden_sizes[i], hidden_sizes[i + 1])
            bias = torch.randn(n_nets, 1, hidden_sizes[i + 1])
            self.hidden_weights.append(weight)
            self.hidden_biases.append(bias)

        self.output_weight = torch.randn(n_nets, hidden_sizes[-1], output_size)
        self.output_bias = torch.randn(n_nets, 1, output_size)

        self.num_parameters = self.count_parameters()
        self.n_nets = n_nets # number of neural networks

    def count_parameters(self):
        # counts number of parameters per network (not number of parameters of all networks)
        parameters = []
        for weight in self.hidden_weights:
            parameters.append(weight[0].numel())
        for bias in self.hidden_biases:
            parameters.append(bias[0].numel())
        parameters.append(self.output_weight[0].numel())
        parameters.append(self.output_bias[0].numel())

        return sum(parameters)

    def forward(self, x):
        for weight, bias in zip(self.hidden_weights, self.hidden_biases):
            x = x @ weight + bias
            x = F.relu(x)
            #x = F.tanh(x)
        x = x @ self.output_weight + self.output_bias
        return F.tanh(x)

    def update_weights(self, vec):
        # Unpack the flattened vector and update the weights and biases
        start = 0
        hidden_weights, hidden_biases = [], []
        for weight in self.hidden_weights:
            end = start + weight[0].numel()
            hidden_weights.append(vec[:, start:end].reshape(weight.shape))
            start = end

        for bias in self.hidden_biases:
            end = start + bias[0].numel()
            hidden_biases.append(vec[:, start:end].reshape(bias.shape))
            start = end

        self.hidden_weights = hidden_weights
        self.hidden_biases = hidden_biases
        self.output_weight = vec[:, start:start + self.output_weight[0].numel()].reshape(self.output_weight.shape)
        self.output_bias = vec[:, start + self.output_weight[0].numel():].reshape(self.output_bias.shape)


########################################################################################################################
# Adjusted CDE module: takes MC nontrainable net, returns MC controlled states

# Data-dependent CDE module
class CDE(LightningModule):
    def __init__(self, hidden_size, control_size, integrand, control_path, batch_size): #control_batch, interpolation_method, ts):
        super().__init__()

        # Integrand
        self.hidden_size = hidden_size
        self.integrand = integrand

        # Control path
        self.control_size = control_size
        self.control_path = control_path
        self.batch_size = batch_size

    def forward(self, t, z):
        # Pass through integrand
        z = self.integrand(z).reshape(self.integrand.n_nets, self.batch_size, self.hidden_size, self.control_size)
        # Matrix mult. with control path derivative and return
        return (z @ self.control_path.derivative(t).unsqueeze(-1)).squeeze(-1)

########################################################################################################################


# BNCDE-main module
class BNCDEModule(LightningModule):
    def __init__(
            self,
            covariate_size,
            treatment_size,
            outcome_size,
            mc_samples,
            hidden_size,
            sd_diffusion,
            drift_layers,
            intensity_weighting=False,
            prediction_window=1,
            interpolation_method='linear',
            learning_rate=1e-3,
            method='euler',
            coverage_metric_confidence=0.95,
            visualize=True,
            ):
        super().__init__()
        self.noise_type = "diagonal"  # required
        self.sde_type = "ito"  # required
        self.method = method
        self.mc_samples = mc_samples
        self.tau = prediction_window
        self.intensity_weighting = intensity_weighting

        self.outcome_loss = nn.MSELoss()
        self.visualize = visualize

        # Embedding network
        self.e_net = nn.Sequential(
            nn.Linear(covariate_size + outcome_size, hidden_size))
        nn.init.constant_(self.e_net[0].bias, 0.)

        # Prediction head
        self.pred_head = MC_trainable_net(input_size=hidden_size, output_size=2)

        # Only used if intensity_weighting = True:
        if intensity_weighting:
            self.obs_net = MC_trainable_net(input_size=hidden_size, output_size=1)

        self.control_size = covariate_size + outcome_size
        self.treatment_size = treatment_size

        if interpolation_method == 'linear':
            self.inter_coeffs = torchcde.linear_interpolation_coeffs
            self.interpolate = torchcde.LinearInterpolation
        elif interpolation_method == 'spline':
            self.inter_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences
            self.interpolate = torchcde.CubicSpline

        # Neural CDE vector fields
        # Encoder
        self.hidden_size = hidden_size
        self.CDE_integrand_enc = MC_nontrainable_net(n_nets=mc_samples, input_size=hidden_size + 1, hidden_sizes=(128,128),#hidden_sizes=(32, 32),
                                                     output_size=hidden_size * self.control_size)
        self.z_field_nparams_enc = self.CDE_integrand_enc.num_parameters
        # Decoder
        self.hidden_size = hidden_size
        self.CDE_integrand_dec = MC_nontrainable_net(n_nets=mc_samples, input_size=hidden_size + 1,
                                                     hidden_sizes=(128,128),#hidden_sizes=(32, 32),
                                                     output_size=hidden_size * treatment_size)
        self.z_field_nparams_dec = self.CDE_integrand_dec.num_parameters


        # Weight process
        self.w_diffusion = sd_diffusion  # diffusion coefficient
        # MC samples are represented as batch dimension in hyper network
        # Encoder
        self.w_size_enc = self.z_field_nparams_enc
        self.w_drift_enc = SequentialModel(input_size=self.w_size_enc+1,
                                           output_size=self.w_size_enc, hidden_layers=drift_layers)
        for layer in self.w_drift_enc.model:
            if isinstance(layer, nn.Linear):
                nn.init.constant_(layer.bias, 0.)
        # Decoder
        self.w_size_dec = self.z_field_nparams_dec
        self.w_drift_dec = SequentialModel(input_size=self.w_size_dec + 1,
                                           output_size=self.w_size_dec, hidden_layers=drift_layers)
        for layer in self.w_drift_dec.model:
            if isinstance(layer, nn.Linear):
                nn.init.constant_(layer.bias, 0.)

        # Variational posterior and prior at t=0
        # Encoder
        self.mu0_enc = nn.Parameter(nn.init.uniform_(torch.empty(self.w_size_enc), -0.003, 0.003),
                                    requires_grad=True)
        self.p0_enc = dist.Normal(loc=self.mu0_enc.clone().detach(), scale=self.w_diffusion)
        self.q0_enc = dist.Normal(loc=self.mu0_enc, scale=self.w_diffusion)
        # Decoder
        self.mu0_dec = nn.Parameter(nn.init.uniform_(torch.empty(self.w_size_dec), -0.003, 0.003),
                                    requires_grad=True)
        self.p0_dec = dist.Normal(loc=self.mu0_dec.clone().detach(), scale=self.w_diffusion)
        self.q0_dec = dist.Normal(loc=self.mu0_dec, scale=self.w_diffusion)


        # learning setup
        self.learning_rate = learning_rate
        self.coverage_metric = CoverageMetric(gaussian_mixture_components=True,
                                              confidence_level=coverage_metric_confidence)

        # Normalization
        self.norm_treatments = FixedTransferableNorm(
            normalized_shape=(1, treatment_size,), 
            dim_to_normalize=-2,
        )

        self.norm_outcomes = FixedAllBackNorm(
            normalized_shape=(1, outcome_size,), 
            dim_to_normalize=-2,
        )

        self.norm_covariates = FixedLayerNorm1D(
            normalized_shape=(1, covariate_size,), 
            dim_to_normalize=-2,
        )

    def forward_enc(self, x):
        # Required for control path interpolation and SDE integration
        ts = torch.linspace(0, 1, x.shape[1])
        control_path = self.interpolate(self.inter_coeffs(x, ts), ts)
        batch_size = x.shape[0]

        # Sample initial weights and compute KL-divergence at t=0
        w0 = self.mu0_enc.repeat(self.mc_samples).reshape(self.mc_samples, -1) + dist.Normal(loc=0, scale=self.w_diffusion).sample((self.mc_samples, self.w_size_enc))
        log_qp0 = 0*dist.kl_divergence(self.q0_enc, self.p0_enc).sum()

        # Transform x0 to z0
        z0 = self.e_net(control_path.evaluate(0)).reshape(-1).repeat(self.mc_samples).reshape(self.mc_samples, -1, self.hidden_size)

        # Create CDE instance based on current batch
        self.z_field = CDE(hidden_size=self.hidden_size, control_size=self.control_size,
                           integrand=self.CDE_integrand_enc, control_path=control_path, batch_size=batch_size)

        # Initialize weights
        self.z_field.integrand.update_weights(w0)

        z0, w0 = z0.reshape(self.mc_samples, -1), w0.reshape(self.mc_samples, -1)
        full_state0 = torch.cat((w0, z0, torch.full((self.mc_samples, 1), log_qp0.item())), dim=-1)

        # Integrate forward
        full_state1 = torchsde.sdeint(self, full_state0, ts, names={'drift': 'f_enc', 'diffusion': 'g_enc'},
                                      adaptive=True, method=self.method, options=dict(jump_t=ts))
        # Extract KLD
        kld = log_qp0 + 0.5 * full_state1[:, :, -1]
        # Extract hidden state
        z1 = full_state1[:, :, self.w_size_enc:-1].reshape(-1, self.mc_samples, batch_size, self.hidden_size)

        # # only last time step
        z1 = z1[-1:]
        kld = kld[-1:]
        z1 = z1.permute(1, 2, 0, 3)  # mc, batch, time, hidden_size
        kld = kld.permute((1, 0))  # mc, time

        return kld, z1

    def f_enc(self, t, full_state):
        # Note: full state has entries [[w1, w2, ..., wD,
        #                                batch1_z1, batch1_z2, ..., batchN_zK,
        #                                batch1_y1, batch2_y1, ..., batchN_y1,
        #                                batch1_a1, batch2_a1, ..., batchN_a1,
        #                                KL_term]]

        # Split full state into hidden state and weights
        w_state, z_state = full_state[:, :self.w_size_enc], full_state[:, self.w_size_enc:-1]

        # Recreate batch dimension
        z_state = z_state.reshape((self.mc_samples, -1, self.hidden_size))


        # Pass weights through hypernet and compute KL-divergence term
        w_out = self.w_drift_enc(torch.cat((torch.full((self.mc_samples, 1), t), w_state), dim=-1))
        w_state = w_out - w_state
        log_qp = (w_out ** 2).sum(dim=-1, keepdim=True) / (self.w_diffusion ** 2)

        # Apply neural CDE vector field
        self.z_field.integrand.update_weights(w_state)
        z_state = self.z_field(t, torch.concatenate((torch.full((self.mc_samples, self.z_field.batch_size, 1), t), z_state), dim=-1))

        return torch.cat((w_state, z_state.reshape(self.mc_samples, -1), log_qp), dim=-1)

    def g_enc(self, t, full_state):
        # keep diffusion constant for now
        return torch.cat((
            torch.full(size=(self.mc_samples, self.w_size_enc), fill_value=self.w_diffusion),  # weights
            torch.zeros(size=(self.mc_samples, full_state[0].numel() - self.w_size_enc)) # hidden states, kld
        ), dim=-1)

    def forward_dec(self, x, z0):
        # Required for control path interpolation and SDE integration
        ts = torch.linspace(0, 1, x.shape[1] + 1)
        x = torch.cat([x, torch.zeros_like(x)[..., -1:, :]], dim=-2)
        control_path = self.interpolate(self.inter_coeffs(x, ts), ts)
        batch_size = x.shape[0]

        # Sample initial weights and compute KL-divergence at t=0
        w0 = self.mu0_dec.repeat(self.mc_samples).reshape(self.mc_samples, -1) + dist.Normal(loc=0, scale=self.w_diffusion).sample((self.mc_samples, self.w_size_dec))
        log_qp0 = 0*dist.kl_divergence(self.q0_dec, self.p0_dec).sum()

        # Stack expected value
        z0 = z0.squeeze(dim=2).mean(dim=0).repeat(self.mc_samples, 1, 1)
        # Create CDE instance based on current batch
        self.z_field = CDE(hidden_size=self.hidden_size, control_size=self.treatment_size,
                           integrand=self.CDE_integrand_dec, control_path=control_path, batch_size=batch_size)

        # Initialize weights
        self.z_field.integrand.update_weights(w0)

        z0, w0 = z0.reshape(self.mc_samples, -1), w0.reshape(self.mc_samples, -1)
        full_state0 = torch.cat((w0, z0, torch.full((self.mc_samples, 1), log_qp0.item())), dim=-1)

        # Integrate forward
        full_state1 = torchsde.sdeint(self, full_state0, ts, names={'drift': 'f_dec', 'diffusion': 'g_dec'},
                                      adaptive=True, method=self.method, options=dict(jump_t=ts))
        # Extract KLD
        kld = log_qp0 + 0.5 * full_state1[:, :, -1]
        # Extract hidden state
        z1 = full_state1[:, :, self.w_size_dec:-1].reshape(-1, self.mc_samples, batch_size, self.hidden_size)

        # # only last time step
        # z1 = z1[-1:]
        # kld = kld[-1:]
        # only from the first step onwards
        kld = kld[:-1]
        z1 = z1[:-1]
        z1 = z1.permute(1, 2, 0, 3)  # mc, batch, time, hidden_size
        kld = kld.permute((1, 0))  # mc, time

        return kld, z1

    def f_dec(self, t, full_state):
        # Note: full state has entries [[w1, w2, ..., wD,
        #                                batch1_z1, batch1_z2, ..., batchN_zK,
        #                                batch1_y1, batch2_y1, ..., batchN_y1,
        #                                batch1_a1, batch2_a1, ..., batchN_a1,
        #                                KL_term]]

        # Split full state into hidden state and weights
        w_state, z_state = full_state[:, :self.w_size_dec], full_state[:, self.w_size_dec:-1]

        # Recreate batch dimension
        z_state = z_state.reshape((self.mc_samples, -1, self.hidden_size))

        # Pass weights through hypernet and compute KL-divergence term
        w_out = self.w_drift_dec(torch.cat((torch.full((self.mc_samples, 1), t), w_state), dim=-1))
        w_state = w_out - w_state
        log_qp = (w_out ** 2).sum(dim=-1, keepdim=True) / (self.w_diffusion ** 2)

        # Apply neural CDE vector field
        self.z_field.integrand.update_weights(w_state)
        z_state = self.z_field(t, torch.concatenate((torch.full((self.mc_samples, self.z_field.batch_size, 1), t), z_state), dim=-1))

        return torch.cat((w_state, z_state.reshape(self.mc_samples, -1), log_qp), dim=-1)

    def g_dec(self, t, full_state):
        # keep diffusion constant for now
        return torch.cat((
            torch.full(size=(self.mc_samples, self.w_size_dec), fill_value=self.w_diffusion),  # weights
            torch.zeros(size=(self.mc_samples, full_state[0].numel() - self.w_size_dec)) # hidden states, kld
        ), dim=-1)
    
    def prepare_input(
            self, 
            covariate_history, 
            treatment_history, 
            outcome_history, 
            treatments, 
            outcomes=None
            ):

        # data_X:        N_samples x (T - 5) x [cancer_volume, patient_type]
        # data_tr:       N_samples x (T - 5) x 4
        # data_y:        N_samples x (T - 5)
        # data_tr_tau:   N_samples x T x 4

        x = torch.cat(
            [covariate_history,
             outcome_history],
            dim=-1
            )
        
        tr = treatment_history

        y = torch.cat(
            [outcome_history,
             outcomes[..., :self.tau]],
            dim=-2
             )
        
        tr_tau = torch.cat(
            [treatment_history,
             treatments[..., :self.tau]],
            dim=-2
             )
                
        active_entries = torch.ones_like(y, dtype=float)

        return x, y, tr_tau, active_entries

    def compute_uncertainty(
            self, 
            covariate_history, 
            treatment_history, 
            outcome_history, 
            outcomes, 
            treatments
            ):

        mu_outcomes, var_outcomes = self.multistep_forward(
            covariate_history=covariate_history, 
            treatment_history=treatment_history, 
            outcome_history=outcome_history, 
            outcomes=outcomes, 
            treatments=treatments
        )

        return mu_outcomes, var_outcomes
    
    def forward_net(
            self,
            covariate_history, 
            treatment_history, 
            outcome_history, 
            outcomes, 
            treatments
            ):

        # Apply normalization
        outcomes = self.norm_outcomes(outcomes, outcome_history)
        outcome_history = self.norm_outcomes.transfer(outcome_history)

        treatments = self.norm_treatments(
            target_seq=treatments,
            source_seq=treatment_history
            )
        treatment_history = self.norm_treatments.transfer(treatment_history)

        covariate_history = self.norm_covariates(covariate_history)

        # Prepare input
        x, y, tr_tau, active_entries = self.prepare_input(
            covariate_history=covariate_history, 
            treatment_history=treatment_history, 
            outcome_history=outcome_history, 
            outcomes=outcomes, 
            treatments=treatments
        )

        # Multistep treatment assignments
        in_time = x.shape[1]
        # For tau-step ahead, need grid of tau+1 treatment assignments
        tr_tau = tr_tau[:, (in_time):(in_time+self.tau), :]

        # Forward
        kld, z1 = self.forward_enc(x)
        # only the last time step
        kld = kld[..., -1, :]
        z1 = z1[..., -1, :]

        if self.tau >= 1:
            kld_dec, z1 = self.forward_dec(tr_tau, z1)            
            kld_dec = kld_dec[..., -1]
            kld += kld_dec
            # z1 = z1[..., -1, None, :]  # Only the last time-step for training

        # Prediction
        pred = self.pred_head(z1)
        y_pred, sigma_pred = pred[:, :, :, 0], F.softplus(pred[:, :, :, 1])

        # Undo normalization
        y_pred = self.norm_outcomes.inverse(y_pred.permute(1, 2, 0)).permute(2, 0, 1)
        
        # Only the last step
        y_pred = y_pred[..., -1:]
        sigma_pred = sigma_pred[..., -1:]

        return y_pred, sigma_pred, active_entries, kld, x
    
    def forward(self, batch):

        return self.forward_net(
            covariate_history=batch['covariate_history'], 
            treatment_history=batch['treatment_history'], 
            outcome_history=batch['outcome_history'], 
            treatments=batch['treatments'], 
            outcomes=batch['outcomes']
        )

    def multistep_forward(
            self,
            covariate_history, 
            treatment_history, 
            outcome_history, 
            outcomes, 
            treatments
            ):

        # Apply normalization
        outcomes = self.norm_outcomes(outcomes, outcome_history)
        outcome_history = self.norm_outcomes.transfer(outcome_history)

        treatments = self.norm_treatments(
            target_seq=treatments,
            source_seq=treatment_history
            )
        treatment_history = self.norm_treatments.transfer(treatment_history)

        covariate_history = self.norm_covariates(covariate_history)

        # Prepare input
        x, y, tr_tau, active_entries = self.prepare_input(
            covariate_history=covariate_history[None, ...], 
            treatment_history=treatment_history[None, ...], 
            outcome_history=outcome_history[None, ...], 
            outcomes=outcomes[None, ...], 
            treatments=treatments[None, ...]
        )

        # Multistep treatment assignments
        in_time = x.shape[1]
        # For tau-step ahead, need grid of tau+1 treatment assignments
        tr_tau = tr_tau[:, (in_time):(in_time+self.tau), :]

        # Forward
        kld, z1 = self.forward_enc(x)
        # only the last time step
        kld = kld[..., -1, :]
        z1 = z1[..., -1, :]

        if self.tau >= 1:
            kld_dec, z1 = self.forward_dec(tr_tau, z1)

        # Prediction
        pred = self.pred_head(z1)
        y_pred, sigma_pred = pred[:, :, :, 0], F.softplus(pred[:, :, :, 1])

        y_pred = self.norm_outcomes.inverse(y_pred.permute(1, 2, 0)).permute(2, 0, 1)

        return y_pred.permute(0, 2, 1), sigma_pred.permute(0, 2, 1)

    def model_step(self, batch):

        y_pred, sigma_pred, active_entries, kld, x = self.forward(batch)

        y = batch['outcomes'][:, self.tau-1, :]

        y_pred_unmasked, y_unmasked = y_pred.clone(), y.clone()

        log_like = dist.Normal(loc=y_pred, scale=sigma_pred).log_prob(y)
        outcome_loss = self.outcome_loss(y_pred.squeeze(), y.squeeze())

        log_like = log_like.sum(dim=1)  # sum over batch
        elbo = (log_like - kld.sum(dim=-1))

        loss = -elbo.mean()  # mean over MC samples

        return loss, outcome_loss, y_pred_unmasked, sigma_pred, y_unmasked, x

    def training_step(self, batch):
        loss, outcome_loss, mu_pred, sigma_pred, y, _ = self.model_step(batch)
        self.log_dict(
            {"train/loss": loss, 
             'train/outcome loss': outcome_loss},
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            logger=True,
        )
        return {"loss": loss, "y_pred": mu_pred, "sigma_pred" : sigma_pred, "y": y}

    def validation_step(self, batch, batch_idx):
        loss, outcome_loss, mu_pred, sigma_pred, y, _ = self.model_step(batch)
        
        if self.visualize and batch_idx == 0:
            fig, ax = plot_trajectories_with_uncertainty(
                batch=batch, 
                model=self, 
                label='BNCDE',
                )
            self.logger.experiment.log({"plot_trajectories_with_uncertainty": wandb.Image(fig)})

        self.coverage_metric(torch.concatenate((mu_pred, sigma_pred), dim=-1), y.squeeze(dim=-1)) # abuse former time dimension for concatenation
        size = self.coverage_metric.size()
        coverage = self.coverage_metric.compute()
        self.log_dict(
            {"val/loss": loss, 
             "val/coverage": coverage, 
             "val/size": size, 
             "val/outcome loss": outcome_loss},
            prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, outcome_loss, mu_pred, sigma_pred, y, _ = self.model_step(batch)
        self.coverage_metric(torch.concatenate((mu_pred, sigma_pred), dim=-1),
                             y.squeeze(dim=-1))  # abuse former time dimension for concatenation
        size = self.coverage_metric.size()
        coverage = self.coverage_metric.compute()

        self.log_dict(
            {"test/loss": loss, 
             "test/coverage": coverage, 
             "test/size": size, 
             "test/outcome_loss": outcome_loss},
            prog_bar=True, 
            logger=True)

        return {"loss": loss, "y_pred": mu_pred, "sigma_pred": sigma_pred, "y": y,
                "coverage": coverage, "size": size}

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        loss, mu_pred, sigma_pred, y, x = self.model_step(batch)
        self.coverage_metric(torch.concatenate((mu_pred, sigma_pred), dim=-1),
                             y.squeeze(dim=-1))  # abuse former time dimension for concatenation
        size = self.coverage_metric.size()
        coverage = self.coverage_metric.compute()

        return {"loss": loss, "y_pred": mu_pred, "sigma_pred": sigma_pred, "y": y, "x": x,
                "coverage": coverage, "size":size}

    def configure_optimizers(self):

        params = [[self.mu0_enc],
                  self.w_drift_enc.parameters(),
                  self.e_net.parameters(),
                  self.pred_head.layers.parameters(),
                  ]
        learning_rates = [self.learning_rate, self.learning_rate,
                          self.learning_rate * 1e1, self.learning_rate * 1e1  # , self.learning_rate*1e1
                          ]
        if self.tau > 1:
            params += [[self.mu0_dec],
                        self.w_drift_dec.parameters()]
            learning_rates += [self.learning_rate, self.learning_rate]

        if self.intensity_weighting:
            params += [self.obs_net.layers.parameters()]
            learning_rates += [self.learning_rate * 1e1]

        param_groups = [{'params': param_group, 'lr': lr} for param_group, lr in zip(params, learning_rates)]

        return optim.AdamW(param_groups, lr=self.learning_rate)

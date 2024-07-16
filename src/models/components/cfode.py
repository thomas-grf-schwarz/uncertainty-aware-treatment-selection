import torch
import torch.nn as nn
import torchsde

from src.utils.rnn import permute_rnn_style
from src.utils.transforms import (
    LayerNorm1D, 
    FixedAllBackNorm,
    FixedTransferableNorm,
    FixedLayerNorm1D,
    )


class LazyMLP(nn.Module):
    def __init__(
            self,
            out_units=[32, 32],
            act_cls=nn.GELU,
            norm_cls=nn.LayerNorm
    ) -> None:
        super().__init__()
        self.act_cls = act_cls
        self.norm_cls = norm_cls

        layers = []
        for n in out_units:
            layers.append(self.make_layer(n))
        self.net = nn.Sequential(*layers)

    def make_layer(self, out_units):
        layer = nn.Sequential(nn.LazyLinear(out_units), self.act_cls())
        if self.norm_cls:
            layer.append(self.norm_cls(out_units))
        return layer

    def forward(self, x):
        return self.net(x.squeeze(0))


class MLP(nn.Module):
    def __init__(
            self,
            units=[32, 32],
            act_cls=nn.GELU,
            norm_cls=nn.LayerNorm
    ) -> None:
        super().__init__()
        self.act_cls = act_cls
        self.norm_cls = norm_cls
        self.units = units
        layers = []
        for i in range(len(units) - 1):
            layers.append(self.make_layer(
                in_units=units[i], 
                out_units=units[i+1])
                )
        self.net = nn.Sequential(*layers)

    def make_layer(self, in_units, out_units):
        layer = nn.Sequential(nn.Linear(in_units, out_units), self.act_cls())
        if self.norm_cls:
            layer.append(self.norm_cls(out_units))
        return layer

    def forward(self, x):
        return self.net(x.squeeze(0))


class LatentSDE(torchsde.SDEIto):

    def __init__(self, theta, mu, sigma, hidden_size):
        super().__init__(noise_type="diagonal")

        self.theta = theta
        self.mu = mu
        self.register_buffer("sigma", torch.tensor([[sigma]]))

        self.hidden_size = hidden_size    
        self.sde_drift = MLP(
            units=[self.hidden_size,
                   2*self.hidden_size,
                   2*self.hidden_size,
                   self.hidden_size], 
            act_cls=nn.Tanh, 
            norm_cls=nn.LayerNorm)

    def g(self, t, y):
        return self.sigma.repeat(y.shape)

    def h(self, t, y):
        return self.theta * (self.mu-y)

    def f(self, t, y):
        return self.sde_drift(y) - torch.zeros_like(y)  # output to y shape

    def f_aug(self, t, y):
        y = y[:, :self.hidden_size]
        f, g, h = self.f(t, y), self.g(t, y), self.h(t, y)
        u = torch.div(f - h, g)
        f_logqp = .5 * (u ** 2).sum(dim=1, keepdim=True)
        return torch.cat([f, f_logqp], dim=1)

    def h_aug(self, t, y):
        y = y[:, :self.hidden_size]
        f, g, h = self.f(t, y), self.g(t, y), self.h(t, y)
        u = torch.div(f - h, g)
        f_logqp = .5 * (u ** 2).sum(dim=1, keepdim=True)
        return torch.cat([h, f_logqp], dim=1)

    def g_aug(self, t, y):
        y = y[:, :self.hidden_size]
        g = self.g(t, y)
        g_logqp = torch.zeros(y.shape[0], 1).to(y)
        return torch.cat([g, g_logqp], dim=1)


class CFODE(nn.Module):
    def __init__(
            self,
            covariate_size,
            treatment_size,
            outcome_size,
            hidden_size,
            num_layers,
            theta,
            mu,
            sigma,
            bidirectional=True,
            alpha=0.0,
            ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.covariate_size = covariate_size
        self.treatment_size = treatment_size
        self.outcome_size = outcome_size
        self.bidirectional = bidirectional

        # Encoder components
        input_size = covariate_size + treatment_size + outcome_size
        self.encoder_rnn = nn.GRU(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            bidirectional=bidirectional
            )

        # Decoder components
        self.decoder_sde = LatentSDE(
            theta=theta,
            mu=mu,
            sigma=sigma,
            hidden_size=2*hidden_size,
            )
        self.decoder_rnn = nn.GRU(
            input_size=treatment_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers,
            batch_first=True, 
            bidirectional=bidirectional
            )
        self.out = nn.Linear(hidden_size, 2 * outcome_size)
        self.sigmoid = nn.Sigmoid()

        # Normalization
        self.norm_var = LayerNorm1D(
            normalized_shape=(1, outcome_size,), 
            dim_to_normalize=-2
            )

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

        self.outcome_loss = torch.nn.MSELoss()
        self.outcome_gnllloss = torch.nn.GaussianNLLLoss()
        self.alpha = alpha

    def encoder_forward(self, input_seq):        
        B, L, C = input_seq.shape
        hidden = self.init_hidden(self.num_layers, B)
        output_seq, hidden = self.encoder_rnn(input_seq, hidden)
        return output_seq, hidden.mean(0)

    @permute_rnn_style
    def decoder_rnn_forward(self, input_seq):
        B, L, C = input_seq.shape
        hidden = self.init_hidden(1, B)
        output_seq, hidden = self.decoder_rnn(input_seq, hidden)
        return hidden.mean(0)

    @permute_rnn_style
    def decoder_forward(self, input_seq, encoder_hidden):
        B, C, L = input_seq.shape

        t = torch.arange(L, dtype=float)
        
        decoder_hidden = self.decoder_rnn_forward(input_seq)
        x0 = torch.cat([encoder_hidden, decoder_hidden], dim=-1)  # B, 2*C_h
        aug_y0 = torch.cat([x0, torch.zeros(B, 1).to(x0)], dim=-1)  # B, 2*C_h + 1

        aug_y = torchsde.sdeint(
            sde=self.decoder_sde,
            y0=aug_y0,
            ts=t,
            method="euler",
            dt=0.05,
            adaptive=False,
            rtol=1e-3,
            atol=1e-3,
            names={
                'drift': 'f_aug',
                'diffusion': 'g_aug',
                }
            )

        y = aug_y[..., :self.hidden_size].permute(1, 2, 0)  # B, C, L
        logqp = aug_y[..., self.hidden_size:].permute(1, 2, 0)

        return y, logqp

    def prepare_output(self, output_seq):
        mu, var = output_seq.chunk(2, dim=-1)
        return mu, var

    def prepare_input(
            self,
            covariate_history,
            treatment_history,
            outcome_history,
            treatments,
            outcomes=None
            ):

        past_seq = torch.cat([
            covariate_history, 
            treatment_history, 
            outcome_history], 
            dim=-1
        )
        encoder_input_seq = past_seq if past_seq.ndim == 3 else past_seq[None, ...]
        decoder_input_seq = treatments if treatments.ndim == 3 else treatments[None, ...]

        return encoder_input_seq, decoder_input_seq

    def init_hidden(self, num_layers, batch_size):
        if self.bidirectional:
            return torch.zeros(2 * num_layers, batch_size, self.hidden_size)
        return torch.zeros(num_layers, batch_size, self.hidden_size)

    def forward(
            self,
            covariate_history,
            treatment_history,
            outcome_history,
            treatments,
            outcomes
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

        # Prepare inputs for the encoder and decoder
        encoder_input_seq, decoder_input_seq = self.prepare_input(
            covariate_history=covariate_history,
            treatment_history=treatment_history,
            outcome_history=outcome_history,
            treatments=treatments,
            outcomes=outcomes,
            )

        # Encoder and decoder forward
        _, encoder_hidden = self.encoder_forward(encoder_input_seq)
        decoder_output_seq, logqp = self.decoder_forward(
            input_seq=decoder_input_seq, 
            encoder_hidden=encoder_hidden
            )

        # Project to the outcome size
        decoder_output_seq = self.out(decoder_output_seq)

        # Parse output into mean and variance; bound variance between 0 and 1.
        mu_outcomes, var_outcomes = self.prepare_output(decoder_output_seq)
        # var_outcomes = torch.nn.functional.softplus(self.norm_var(var_outcomes))
        var_outcomes = self.sigmoid(self.norm_var(var_outcomes))
        mu_outcomes = self.norm_outcomes.inverse(mu_outcomes)

        return mu_outcomes, var_outcomes, logqp
    
    def infer(
            self,
            covariate_history,
            treatment_history,
            outcome_history,
            treatments,
            outcomes
            ):
        
        out = self.forward(
            covariate_history=covariate_history,
            treatment_history=treatment_history,
            outcome_history=outcome_history,
            treatments=treatments,
            outcomes=outcomes
        )

        return out

    def model_step(self, batch):
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a dict)

        :return: A dict containing losses to be logged. Must contain:
                - 'loss': tensor (overall loss)
                - 'outcome_loss': tensor (MSE of outcomes)
        """
        mu_outcomes, var_outcomes, logqp = self.forward(
            covariate_history=batch['covariate_history'],
            treatment_history=batch['treatment_history'],
            outcome_history=batch['outcome_history'],
            outcomes=batch['outcomes'],
            treatments=batch['treatments']
        )

        logqp_loss = logqp.mean()
        outcome_loss = self.outcome_loss(mu_outcomes, batch['outcomes'])
        outcome_gnllloss = self.outcome_gnllloss(
            input=mu_outcomes, 
            target=batch['outcomes'], 
            var=var_outcomes,
            )
        loss = outcome_gnllloss + self.alpha*logqp_loss

        losses = {
            'loss': loss,
            'logqp_loss': logqp_loss,
            'outcome loss': outcome_loss,
        }
        return losses
    
def test_lazy_mlp_out_shape(inp_shape):
    B, C = inp_shape
    out_shape = torch.Size([B, 12])
    inp = torch.randn(B, C)
    out = MLP(n_units=[C, 12])(inp)
    assert out.shape == out_shape, (
        'Error in test_lazy_mlp_out_shape: {out.shape} vs {out_shape}'
        )


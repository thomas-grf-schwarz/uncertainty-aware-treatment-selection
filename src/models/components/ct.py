import torch
from torch import nn
import torch.nn.functional as F
from src.utils.transforms import (
    ReversibleNorm,
    LayerNorm1D,
    TransferableNorm
    )

from src.utils.loss import HSICLoss


class MaskedAttentionHead(nn.Module):

    def __init__(self, head_dim):
        super().__init__()

        self.head_dim = head_dim
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.norm = nn.LayerNorm(self.head_dim)

    def forward(self, values, keys, queries, mask, values_pos_enc, keys_pos_enc):

        values = self.values(values)
        keys = self.keys(values)
        queries = self.queries(values)

        unreduced_energy = queries[..., None, :] * keys[..., None, :, :] + keys_pos_enc
        energy = unreduced_energy.sum(-1)
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -torch.inf)
        attention = F.softmax(energy / (self.head_dim ** 0.5), dim=-1)

        unreduced_out = attention[..., None] * values[..., None, :] + values_pos_enc
        out = unreduced_out.sum(dim=-2)

        out = self.norm(out)

        return out + values


class MulitheadAttention(nn.Module):

    def __init__(
            self, 
            embed_dim, 
            num_heads
            ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0, \
            "embed_dim needs to be divisible by num_heads"
        self.head_dim = embed_dim // num_heads

        self.heads = nn.ModuleList(
            [MaskedAttentionHead(self.head_dim) for _ in range(num_heads)]
            )

    def forward(
            self, 
            values, 
            keys, 
            queries, 
            mask, 
            values_pos_enc, 
            keys_pos_enc
            ):
        values = values.chunk(self.num_heads, dim=-1)
        keys = keys.chunk(self.num_heads, dim=-1)
        queries = queries.chunk(self.num_heads, dim=-1)
        out = []
        for i in range(self.num_heads):
            out.append(self.heads[i](
                values[i], 
                keys[i], 
                queries[i], 
                mask, 
                values_pos_enc, 
                keys_pos_enc
            ))
        return torch.cat(out, dim=-1)


class SingleInputBlock(nn.Module):

    def __init__(
            self, 
            embed_size, 
            num_heads,
            p_dropout,
            ):
        super().__init__()

        self.embed_size = embed_size
        self.num_heads = num_heads

        self.self_attn = MulitheadAttention(
            embed_size, 
            num_heads,
            )
        
        self.cross_attn1 = MulitheadAttention(embed_size, num_heads)
        self.cross_attn2 = MulitheadAttention(embed_size, num_heads)

        self.feedforward = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.ReLU(inplace=True),
            nn.Linear(embed_size, embed_size),
            nn.Dropout(p_dropout)
        )
        self.norm = nn.LayerNorm(
            embed_size
        )

    def forward(
            self,
            x,
            y,
            z,
            x_mask,
            y_mask,
            z_mask,
            keys_pos_enc,
            values_pos_enc
            ):

        out1 = self.self_attn(
            values=x,
            keys=x,
            queries=x,
            mask=x_mask,
            keys_pos_enc=keys_pos_enc,
            values_pos_enc=values_pos_enc,
            )

        out2 = self.cross_attn1(
            values=y,
            keys=y,
            queries=out1,
            mask=y_mask,
            keys_pos_enc=keys_pos_enc,
            values_pos_enc=values_pos_enc,
            )

        out2 = self.cross_attn2(
            values=z,
            keys=z,
            queries=out1,
            mask=z_mask,
            keys_pos_enc=keys_pos_enc,
            values_pos_enc=values_pos_enc,
            )

        out = self.feedforward(out1 + out2)
        return self.norm(out)


class MultiInputBlock(nn.Module):

    def __init__(
            self,
            hidden_size,
            num_heads,
            p_dropout
            ):
        super().__init__()

        self.treatment_block = SingleInputBlock(
            hidden_size, 
            num_heads,
            p_dropout,
            )
        self.outcome_block = SingleInputBlock(
            hidden_size,
            num_heads,
            p_dropout,
            )
        self.covariate_block = SingleInputBlock(
            hidden_size,
            num_heads,
            p_dropout,
            )

    def create_causal_mask(self, seq_len_1, seq_len_2):
        return torch.tril(torch.ones(seq_len_1, seq_len_2))

    def create_horizon_mask(self, active_entries):
        B, L, _ = active_entries.shape
        return active_entries.repeat(1, 1, L).permute(0, 2, 1)

    def combine_masks(self, mask1, mask2):
        return mask1 * mask2

    def forward(
            self,
            treatments,
            outcomes,
            covariates,
            active_entries,
            keys_pos_enc,
            values_pos_enc
            ):

        B, L, C = treatments.shape

        causal_mask = self.create_causal_mask(L, L)
        horizon_mask = self.create_horizon_mask(active_entries)
        horizon_mask = self.combine_masks(causal_mask, horizon_mask)

        treatments_out = self.treatment_block(
            x=treatments,
            y=outcomes,
            z=covariates,
            x_mask=causal_mask,
            y_mask=causal_mask,
            z_mask=horizon_mask,
            keys_pos_enc=keys_pos_enc,
            values_pos_enc=values_pos_enc,
            )

        outcomes_out = self.outcome_block(
            x=outcomes,
            y=covariates,
            z=treatments,
            x_mask=causal_mask,
            y_mask=causal_mask,
            z_mask=horizon_mask,
            keys_pos_enc=keys_pos_enc,
            values_pos_enc=values_pos_enc,
            )

        covariates_out = self.covariate_block(
            x=covariates,
            y=treatments,
            z=outcomes,
            x_mask=causal_mask,
            y_mask=causal_mask,
            z_mask=horizon_mask,
            keys_pos_enc=keys_pos_enc,
            values_pos_enc=values_pos_enc,
            )
        return treatments_out, outcomes_out, covariates_out


class CausalTransformer(nn.Module):

    def __init__(
            self, 
            covariate_size, 
            treatment_size, 
            outcome_size, 
            hidden_size, 
            num_layers=1, 
            num_heads=1,
            p_dropout=0.2,
            l_max=15,
            alpha=0.0,
            ):

        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.l_max = l_max

        self.covariate_size = covariate_size
        self.treatment_size = treatment_size
        self.outcome_size = outcome_size

        self.values_pos_enc_w = nn.Parameter(torch.randn(self.head_dim))
        self.keys_pos_enc_w = nn.Parameter(torch.randn(self.head_dim))

        # Normalization
        self.norm_treatments = ReversibleNorm(
            normalized_shape=(1, treatment_size,),
            dim_to_normalize=-2
            )
        self.norm_outcomes = ReversibleNorm(
            normalized_shape=(1, outcome_size,),
            dim_to_normalize=-2
            )
        self.norm_covariates = LayerNorm1D(
            normalized_shape=(1, covariate_size,),
            dim_to_normalize=-2
        )

        self.init_linear_treatments = nn.Linear(
            treatment_size, hidden_size)
        self.init_linear_outcomes = nn.Linear(
            outcome_size, hidden_size)
        self.init_linear_covariates = nn.Linear(
            covariate_size, hidden_size)

        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            self.blocks.append(
                MultiInputBlock(
                    hidden_size,
                    num_heads,
                    p_dropout,
                    )
                )
        self.elu = nn.ELU(inplace=True)
        self.outcome_out = nn.Linear(hidden_size, outcome_size)

        # Losses
        self.balancing_criterion = HSICLoss()
        self.outcome_loss = torch.nn.MSELoss()
        self.alpha = alpha

    def infer(
            self,
            covariate_history,
            treatment_history,
            outcome_history,
            outcomes,
            treatments,
            ):
        active_entries = torch.ones(*outcome_history.shape[:-1], 1)

        pred_outcomes = []
        represenations = []
        B, L_horizon, C = treatments.shape

        pred_outcomes_out, representations_out = self.forward_future(
            covariate_history=covariate_history,
            treatment_history=treatment_history,
            outcome_history=outcome_history,
            active_entries=active_entries
            )

        for l in range(L_horizon):
           
            pred_outcome = pred_outcomes_out[..., -1:, :]
            representation = representations_out[..., -1:, :]

            active_entries = torch.cat(
                [active_entries[:, 1:, :],
                 torch.zeros_like(active_entries[:, -1:, :])],
                dim=-2)

            covariate_history = torch.cat(
                [covariate_history[:, 1:, :],
                 torch.zeros_like(covariate_history[:, -1:, :])],
                dim=-2)

            treatment_history = torch.cat(
                [treatment_history[:, 1:, :],
                 treatments[:, l, None, :]],
                dim=-2)

            outcome_history = torch.cat(
                [outcome_history[:, 1:, :],
                 pred_outcome[:, -1:, :]],
                dim=-2)

            pred_outcomes_out, representations_out = self.forward_future(
                covariate_history=covariate_history,
                treatment_history=treatment_history,
                outcome_history=outcome_history,
                active_entries=active_entries
                )
            
            pred_outcomes.append(pred_outcome[:, -1:, :])
            represenations.append(representation[:, -1:, :])

        return torch.cat(pred_outcomes, dim=-2), \
            torch.cat(represenations, dim=-2)

    def get_pos_enc(self, inds, w_enc):
        toeplitz = inds[:, None] - inds[None, :]
        pos_enc = w_enc * torch.clip(toeplitz, min=-self.l_max, max=self.l_max)[..., None]
        pos_enc = (pos_enc - pos_enc.mean()) / (pos_enc.std() + 1e-6)
        return pos_enc[None, ...]

    def prepare_input(
            self,
            covariate_history,
            treatment_history,
            outcome_history,
            treatments=None,
            outcomes=None,
            active_entries=None,
            ):
        
        if active_entries is None:
            active_entries = self.sample_active_entries(
                *covariate_history.shape[:-1]
                )

        return (treatment_history,
                outcome_history,
                covariate_history,
                active_entries)

    def sample_active_entries(self, B, L):
        active_entries = torch.ones(B, L)
        l_idx = torch.randint(low=0, high=L, size=(B, 1))
        l_inds = torch.arange(L)[None, :]
        active_entries[l_idx < l_inds] = 0.0
        return active_entries[..., None]

    def forward_future(
            self,
            covariate_history,
            treatment_history,
            outcome_history,
            treatments=None,
            outcomes=None,
            active_entries=None,
            ):

        treatments, outcomes, covariates, active_entries = self.prepare_input(
            covariate_history=covariate_history,
            treatment_history=treatment_history,
            outcome_history=outcome_history,
            treatments=treatments,
            outcomes=outcomes,
            active_entries=active_entries
            )

        B, L, C = treatments.shape

        # Normalize
        treatments = self.norm_treatments(treatments)
        outcomes = self.norm_outcomes(outcomes)
        covariates = self.norm_covariates(covariates)

        # Initial linear layer
        treatments = self.init_linear_treatments(treatments)
        outcomes = self.init_linear_outcomes(outcomes)
        covariates = self.init_linear_covariates(covariates)

        # Add positional encoding
        inds = torch.arange(L)
        keys_pos_enc = self.get_pos_enc(inds, self.keys_pos_enc_w)
        values_pos_enc = self.get_pos_enc(inds, self.values_pos_enc_w)

        # Apply MultiInput Blocks
        for block in self.blocks:
            treatments, outcomes, covariates = block(
                treatments,
                outcomes,
                covariates,
                active_entries=active_entries,
                keys_pos_enc=keys_pos_enc,
                values_pos_enc=values_pos_enc,
                )

        # Combine hiddens of treatments, outcomes and covariates by averaging
        covariates[active_entries.squeeze().bool()] = 0.0
        out = torch.stack([treatments, outcomes, covariates], dim=0).sum(dim=0)
        out = out / (2 + active_entries)
        out = self.elu(out)

        pred_outcomes = self.outcome_out(out)
        pred_outcomes = self.norm_outcomes.inverse(pred_outcomes)

        return pred_outcomes, out

    def forward(
            self,
            covariate_history,
            treatment_history,
            outcome_history,
            treatments=None,
            outcomes=None,
            active_entries=None,
            ):

        pred_outcomes, representations = self.forward_future(
            covariate_history=covariate_history,
            treatment_history=treatment_history,
            outcome_history=outcome_history,
            active_entries=active_entries
            )

        # return only the outcomes for which a label exists
        representations = representations[:, :-1, :]
        pred_outcomes = pred_outcomes[:, :-1, :]

        return pred_outcomes, representations
    
    def model_step(self, batch):
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a dict)

        :return: A dict containing losses to be logged. Must contain:
                - 'loss': tensor (overall loss)
                - 'outcome_loss': tensor (MSE of outcomes)
        """
        pred_outcomes, representations = self.forward(
            covariate_history=batch['covariate_history'],
            treatment_history=batch['treatment_history'],
            outcome_history=batch['outcome_history'],
            outcomes=batch['outcomes'],
            treatments=batch['treatments']
        )

        outcome_loss = self.outcome_loss(
            pred_outcomes,
            batch['outcome_history'][:, 1:, :]
            )
        balancing_loss = self.balancing_criterion(
            representations,
            batch['treatment_history'][:, 1:, :]
            )

        loss = self.alpha * balancing_loss + outcome_loss

        losses = {
            'loss': loss,
            'balancing_loss': balancing_loss,
            'outcome loss': outcome_loss,
        }

        return losses
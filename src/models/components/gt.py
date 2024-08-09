import torch
from torch import nn
from src.utils.transforms import (
    LayerNorm1D, 
    FixedAllBackNorm,
    FixedTransferableNorm,
    FixedLayerNorm1D,
    )

from src.utils.loss import HSICLoss


class VariationalGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, p_dropout=0.2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        assert num_layers == 1

        self.gru = nn.GRUCell(input_size=input_size, hidden_size=hidden_size)
        self.p_dropout = p_dropout
        self.recurrent_dropout = nn.Dropout(p=p_dropout)
        self.input_dropout = nn.Dropout(p=p_dropout)
        self.output_dropout = nn.Dropout(p=p_dropout)

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)

    def forward_step(self, x, hidden):
        x = self.input_dropout(x)
        hidden = self.recurrent_dropout(hidden)
        hidden = self.gru(x, hidden)
        return x, self.output_dropout(hidden)

    def forward(self, x, hidden):
        B, L, C = x.shape
        out = []
        hidden = hidden[-1, ...]
        for l in range(L):
            _, hidden = self.forward_step(x[:, l, :], hidden)
            out.append(hidden)
        return torch.stack(out, dim=-2), hidden

    def multilayer_forward(self, x, hidden):
        B, L, C = x.shape
        B, C, NL = hidden.shape
        out = []
        hidden = hidden[-1, ...]
        for l in range(L):
            _, hidden = self.forward_step(x[:, l, :], hidden)
            out.append(hidden)
        return torch.stack(out, dim=-2), hidden


class GNet(nn.Module):
    def __init__(
            self,
            covariate_size,
            treatment_size,
            outcome_size,
            hidden_size,
            num_layers=1,
            p_dropout=0.2,
            ):

        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.covariate_size = covariate_size
        self.treatment_size = treatment_size
        self.outcome_size = outcome_size

        # Decoder components
        input_size = covariate_size + treatment_size + outcome_size
        self.decoder_rnn = VariationalGRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            p_dropout=p_dropout
            )
        # self.out = nn.Linear(
        #     hidden_size,
        #     outcome_size
        #     )

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

        self.norm_out = LayerNorm1D(
            normalized_shape=(1, outcome_size,), 
            dim_to_normalize=-2
            )
        
        # Losses
        self.outcome_loss = torch.nn.MSELoss()

    def decoder_forward(self, input_seq, encoder_hidden):
        output_seq, decoder_hidden = self.decoder_rnn(
            x=input_seq,
            hidden=encoder_hidden
            )
        return output_seq, decoder_hidden

    def prepare_input(
            self,
            covariate_history,
            treatment_history,
            outcome_history,
            treatments,
            outcomes=None
            ):

        return torch.cat([
            covariate_history,
            treatment_history,
            outcome_history],
            dim=-1)

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)
    
    def infer(
            self,
            covariate_history,
            treatment_history,
            outcome_history,
            treatments,
            outcomes
            ):
        
        B, L, _ = treatments.shape
       
        pred_outcomes = []

        for l in range(L):

            residual_outcomes, residual_covariates = self.sample_residual_batch(l, B)

            pred_outcomes_out, pred_covariates_out = self.forward(
                covariate_history=covariate_history,
                treatment_history=treatment_history,
                outcome_history=outcome_history,
                )
    
            covariate_history = torch.cat(
                [covariate_history[:, 1:, :],
                 pred_covariates_out[:, -1:, :] + residual_covariates],
                dim=-2)

            treatment_history = torch.cat(
                [treatment_history[:, 1:, :],
                 treatments[:, l, None, :]],
                dim=-2)

            outcome_history = torch.cat(
                [outcome_history[:, 1:, :],
                 pred_outcomes_out[:, -1:, :] + residual_outcomes],
                dim=-2)

            pred_outcomes.append(pred_outcomes_out)

        pred_outcomes = torch.stack(pred_outcomes, dim=-2)
        
        return pred_outcomes.mean(0)

    def sample_residual(self, l):
        assert hasattr(self, 'holdout_residuals')
        idx = torch.randint(high=len(self.holdout_residuals), size=(1,))
        residual_outcomes, residual_covariates = self.holdout_residuals[idx]
        return residual_outcomes[l], residual_covariates[l]
    
    def sample_residual_batch(self, l, n):
        residual_outcomes, residual_covariates = list(
            zip(*[self.sample_residual(l) for _ in range(n)])
            )
        return torch.stack(residual_outcomes), torch.stack(residual_covariates)

    def forward(
            self,
            covariate_history,
            treatment_history,
            outcome_history,
            treatments=None,
            outcomes=None,
            active_entries=None,
            ):

        treatments, outcomes, covariates = self.prepare_input(
            covariate_history=covariate_history,
            treatment_history=treatment_history,
            outcome_history=outcome_history,
            treatments=treatments,
            outcomes=outcomes,
            )

        B, L, C = treatments.shape

        # Normalize
        treatments = self.norm_treatments(treatments)
        outcomes = self.norm_outcomes(outcomes)
        covariates = self.norm_covariates(covariates)

        # Decoder forward
        out = self.decoder_forward(
            input_seq=torch.cat([treatments, outcomes, covariates], dim=-1),
            encoder_hidden=self.init_hidden(B)
        )

        # Predict both outcomes and covariates
        pred_outcomes = out[..., :self.outcome_size]
        pred_covariates = out[..., self.outcome_size:]

        return pred_outcomes, pred_covariates

    def model_step(self, batch):
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a dict)

        :return: A dict containing losses to be logged. Must contain:
                - 'loss': tensor (overall loss)
                - 'outcome_loss': tensor (MSE of outcomes)
        """
        pred_outcomes, pred_covariates = self.forward(
            covariate_history=batch['covariate_history'],
            treatment_history=batch['treatment_history'],
            outcome_history=batch['outcome_history'],
            outcomes=batch['outcomes'],
            treatments=batch['treatments']
        )

        outcome_loss = self.outcome_loss(
            pred_outcomes[:, :-1, :],
            batch['outcome_history'][:, 1:, :]
            )
        
        covariate_loss = self.outcome_loss(
            pred_covariates[:, :-1, :],
            batch['covariate_history'][:, 1:, :]
            )

        losses = {
            'loss': covariate_loss + outcome_loss,
            'outcome loss': outcome_loss,
        }

        return losses

    def on_fit_end(self) -> None:

        # adapted from https://github.com/konstantinhess/G_transformer/blob/main/src/models/gnet.py       

        self.eval()
        self.holdout_residuals = []
        for batch in self.trainer.datamodule.val_dataloader:

            pred_outcomes, pred_covariates = self.forward(
                covariate_history=batch['covariate_history'],
                treatment_history=batch['treatment_history'],
                outcome_history=batch['outcome_history'],
                outcomes=batch['outcomes'],
                treatments=batch['treatments']
            )

            # [B L C] -> N [L C]
            self.holdout_residuals.extend(
                zip(
                    (batch['outcome_history'] - pred_outcomes).tolist(),
                    (batch['covariate_history'] - pred_covariates).tolist()
                )
            )

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


class CRN(nn.Module):
    def __init__(
            self,
            covariate_size,
            treatment_size,
            outcome_size,
            hidden_size,
            num_layers=1,
            p_dropout=0.2,
            alpha=0.0,
            ):

        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.covariate_size = covariate_size
        self.treatment_size = treatment_size
        self.outcome_size = outcome_size

        # Encoder components
        input_size = covariate_size + treatment_size + outcome_size
        self.encoder_rnn = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True
            )

        # Decoder components
        self.decoder_rnn = VariationalGRU(
            input_size=treatment_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            p_dropout=p_dropout
            )
        self.out = nn.Linear(
            hidden_size,
            outcome_size
            )

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
        self.balancing_criterion = HSICLoss()
        self.outcome_loss = torch.nn.MSELoss()
        self.alpha = alpha

    def encoder_forward(self, input_seq):
        B, L, C = input_seq.shape
        hidden = self.init_hidden(B)
        output_seq, hidden = self.encoder_rnn(input_seq, hidden)
        return output_seq, hidden

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

        past_seq = torch.cat([
            covariate_history,
            treatment_history,
            outcome_history],
            dim=-1)

        encoder_input_seq = past_seq
        decoder_input_seq = treatments

        return encoder_input_seq, decoder_input_seq

    def prepare_output(self, output_seq):
        outcomes, treatments = output_seq.split(
            [self.outcome_size, self.treatment_size], 
            dim=-1
            )
        return outcomes, treatments

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
        
        out = self.forward(
            covariate_history=covariate_history,
            treatment_history=treatment_history,
            outcome_history=outcome_history,
            treatments=treatments,
            outcomes=outcomes
        )

        return out

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
            outcomes=outcomes
            )

        # Encoder and decoder forward
        _, encoder_hidden = self.encoder_forward(encoder_input_seq)
        representations, _ = self.decoder_forward(
            input_seq=decoder_input_seq,
            encoder_hidden=encoder_hidden
            )

        # Project to the outcome size
        pred_outcomes = self.out(representations)
        pred_outcomes = self.norm_outcomes.inverse(pred_outcomes)

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
            batch['outcomes']
            )
        balancing_loss = self.balancing_criterion(
            representations,
            batch['treatments']
            )

        loss = self.alpha * balancing_loss + outcome_loss

        losses = {
            'loss': loss,
            'balancing_loss': balancing_loss,
            'outcome loss': outcome_loss,
        }

        return losses
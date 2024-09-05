import torch
from torch import nn
from src.utils.transforms import (
    ReversibleNorm,
    LayerNorm1D
    )

from src.utils.rnn import permute_rnn_style

from src.utils.loss import HSICLoss


class FeaturewiseRNN(nn.Module):

    def __init__(
            self,
            input_size=10,
            num_layers=2
            ):
        super().__init__()
        
        self.input_size = input_size
        self.num_layers = num_layers

        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=input_size,
            num_layers=num_layers
            )

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.input_size)

    @permute_rnn_style
    def forward(self, x):
        return self.rnn(x, self.init_hidden(x.shape[0]))


class MultilayerVariationalGRU(nn.Module):
    def __init__(
            self,
            input_size,
            hidden_size,
            num_layers=2,
            p_dropout=0.2
            ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Create a list of GRUCells for each layer
        self.grus = nn.ModuleList([
            nn.GRUCell(input_size if i == 0 else hidden_size, hidden_size) 
            for i in range(num_layers)])
        
        self.p_dropout = p_dropout
        self.recurrent_dropout = nn.ModuleList([
            nn.Dropout(p=p_dropout) 
            for _ in range(num_layers)])
        self.input_dropout = nn.Dropout(p=p_dropout)
        self.output_dropout = nn.Dropout(p=p_dropout)

    def init_hidden(self, batch_size):
        # Return a tensor for each layer's hidden state
        return [torch.zeros(batch_size, self.hidden_size) 
                for _ in range(self.num_layers)]

    def forward_step(self, x, hidden, layer_idx):
        # No in-place operations here
        x = self.input_dropout(x) if layer_idx == 0 else x  # Only apply input dropout on the first layer's input
        hidden_next = self.grus[layer_idx](x, hidden[layer_idx])
        hidden_next = self.recurrent_dropout[layer_idx](hidden_next)
        return hidden_next

    def multilayer_forward(self, x, hidden):
        B, L, C = x.shape
        out = []
        # Iterate over each time step
        for l in range(L):
            h_input = x[:, l, :]
            new_hidden = []
            # Iterate over each layer
            for layer_idx in range(self.num_layers):
                h_input = self.forward_step(h_input, hidden, layer_idx)
                new_hidden.append(h_input)
            out.append(self.output_dropout(h_input))  # Apply output dropout after the last layer
            hidden = new_hidden  # Update hidden states for the next time step
        return torch.stack(out, dim=-2), hidden

    def forward(self, x, hidden):
        # Call the multilayer_forward method
        return self.multilayer_forward(x, hidden)


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
            alpha=0.0,
            ):

        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.covariate_size = covariate_size
        self.treatment_size = treatment_size
        self.outcome_size = outcome_size

        # Decoder components
        input_size = covariate_size + treatment_size + outcome_size
        self.decoder_rnn = MultilayerVariationalGRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            p_dropout=p_dropout
            )
        
        self.out = nn.Linear(
            hidden_size,
            outcome_size + covariate_size
            )
        self.head = FeaturewiseRNN()

        # Normalization
        self.norm_treatments = ReversibleNorm(
            normalized_shape=(1, treatment_size,),
            dim_to_normalize=-2
            )
        self.norm_outcomes = ReversibleNorm(
            normalized_shape=(1, outcome_size,),
            dim_to_normalize=-2
            )
        self.norm_covariates = ReversibleNorm(
            normalized_shape=(1, covariate_size,),
            dim_to_normalize=-2
        )

        # Losses
        self.outcome_loss = torch.nn.MSELoss()
        self.alpha = alpha

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
        
        return treatment_history, outcome_history, covariate_history

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
                 pred_covariates_out[:, -1:, :] + residual_covariates[:, None, :]],
                dim=-2)

            treatment_history = torch.cat(
                [treatment_history[:, 1:, :],
                 treatments[:, l, None, :]],
                dim=-2)

            outcome_history = torch.cat(
                [outcome_history[:, 1:, :],
                 pred_outcomes_out[:, -1:, :] + residual_outcomes[:, None, :]],
                dim=-2)

            pred_outcomes.append(pred_outcomes_out[:, -1:, :])
        pred_outcomes = torch.cat(pred_outcomes, dim=-2)
        return pred_outcomes,

    def sample_residual(self, l):
        assert len(self.holdout_residuals) > 0
        idx = torch.randint(high=len(self.holdout_residuals), size=(1,))
        residual_outcomes, residual_covariates = self.holdout_residuals[idx]
        return residual_outcomes[l], residual_covariates[l]
    
    def sample_residual_batch(self, l, n):
        residual_outcomes, residual_covariates = list(
            zip(*[self.sample_residual(l) for _ in range(n)])
            )
        return torch.tensor(residual_outcomes), torch.tensor(residual_covariates)

    def forward(
            self,
            covariate_history,
            treatment_history,
            outcome_history,
            treatments=None,
            outcomes=None,
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
        out, _ = self.decoder_forward(
            input_seq=torch.cat([treatments, outcomes, covariates], dim=-1),
            encoder_hidden=self.init_hidden(B)
        )
        out = self.out(out)
        # out = self.head(out)

        # Predict both outcomes and covariates
        pred_outcomes = out[..., :self.outcome_size]
        pred_covariates = out[..., self.outcome_size:]

        # Invert the normalization of the trajectories
        pred_outcomes = self.norm_outcomes.inverse(pred_outcomes)
        pred_covariates = self.norm_covariates.inverse(pred_covariates)

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
            'loss': outcome_loss + covariate_loss,
            'outcome loss': outcome_loss,
        }

        return losses

    @torch.inference_mode()
    def on_fit_end(self, datamodule):

        # adapted from https://github.com/konstantinhess/G_transformer/blob/main/src/models/gnet.py

        self.eval()
        self.holdout_residuals = []
        for batch in datamodule.val_dataloader():

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

if __name__ == 'main':
    gnet = GNet(1, 1, 1, 10)
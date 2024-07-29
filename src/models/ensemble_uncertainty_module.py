from typing import Any, Dict, Tuple
from src.models.uncertainty_module import UncertaintyModule
import torch
from torch import nn
import copy


class EnsembleUncertaintyModule(UncertaintyModule):
    
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        ensemble_size: int = 8,
        visualize: bool = False,
    ) -> None:

        super().__init__(net, optimizer, scheduler, compile, visualize)

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.visualize = visualize

        self.ensemble_size = ensemble_size
        self.ensemble = nn.ModuleList()
        for i in range(ensemble_size):
            self.ensemble.append(copy.deepcopy(net))
        self.ensemble.apply(self.weight_reset)
        self.test_parameters(self.ensemble)
    
    def weight_reset(self, m):
        # https://stackoverflow.com/questions/63627997/reset-parameters-of-a-neural-network-in-pytorch
        reset_parameters = getattr(m, "reset_parameters", None)
        if callable(reset_parameters):
            m.reset_parameters()

    def test_parameters(self, ensemble):
        for i in range(1, len(ensemble)):
            for p0, pi in zip(ensemble[0].parameters(), ensemble[i].parameters()):
                if pi.numel() > 1:  # exclude biases
                    assert not (p0 == pi).all(), """The ensemble shares 
                                                    parameters"""

    def on_train_batch_start(
            self,
            batch, 
            batch_idx, 
            dataloader_idx=0
            ):
        self.on_batch_start(batch_idx)

    def on_validation_batch_start(
            self,
            batch,
            batch_idx,
            dataloader_idx=0,
            ):
        self.on_batch_start(batch_idx)

    def on_test_batch_start(
            self,
            batch,
            batch_idx,
            dataloader_idx=0,
            ):
        self.on_batch_start(batch_idx)

    def on_batch_start(
            self,
            batch_idx,
            ):
        self.net = self.ensemble[batch_idx % self.ensemble_size]

    def compute_uncertainty(
            self, 
            covariate_history, 
            treatment_history, 
            outcome_history, 
            outcomes, 
            treatments
            ):
        
        pred_outcomes_batch = []
        for net in self.ensemble:
            
            pred_outcomes, _ = net.infer(
                covariate_history=covariate_history[None, ...],
                treatment_history=treatment_history[None, ...],
                outcome_history=outcome_history[None, ...],
                treatments=treatments[None, ...],
                outcomes=outcomes[None, ...],
            )

            pred_outcomes_batch.append(pred_outcomes)
        pred_outcomes_batch = torch.cat(pred_outcomes_batch, dim=0)

        return pred_outcomes_batch.mean(0), pred_outcomes_batch.var(0)


if __name__ == "__main__":
    _ = EnsembleUncertaintyModule(None, None, None, None)

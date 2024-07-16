from typing import Any, Dict, Tuple
from src.utils.treat import create_identical_batch
from src.models.uncertainty_module import UncertaintyModule
import torch
from torch import nn
import copy


class GeometricEnsembleUncertaintyModule(UncertaintyModule):
    
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        visualize: bool = False,
        max_epoch: int = 60,
    ) -> None:

        super().__init__(net, optimizer, scheduler, compile, visualize)

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.visualize = visualize

        self.max_epoch = max_epoch
        self.ensemble_size = 2
        self.ensemble = nn.ModuleList()
        for i in range(self.ensemble_size):
            self.ensemble.append(copy.deepcopy(net))
        self.ensemble.apply(self.weight_randperm)
        self.test_parameters(self.ensemble)

    def weight_randperm(self, m):
        for p in m.parameters():
            flattened = p.data.flatten()
            permuted = flattened[torch.randperm(flattened.size(0))]
            setattr(p, 'data', permuted)

    def test_parameters(self, ensemble):
        for i in range(1, len(ensemble)):
            for p0, pi in zip(ensemble[0].parameters(), ensemble[i].parameters()):
                if pi.numel() > 1:  # exclude biases
                    assert not (p0 == pi).all(), 'The ensemble shares parameters'

    def on_train_batch_start(
            self,
            pl_module,
            batch,
            batch_idx,
            ):
        self.on_batch_start(batch_idx)

    def on_validation_batch_start(
            self,
            pl_module,
            batch,
            batch_idx,
            ):
        self.on_batch_start(batch_idx)

    def on_test_batch_start(
            self,
            pl_module,
            batch,
            batch_idx,
            ):
        self.on_batch_start(batch_idx)

    def on_batch_start(
            self,
            batch_idx,
            ):
        self.net = self.nets[self.ensemble_size % batch_idx]

    def on_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch == self.max_epoch:

            for net in self.nets:
                for param in net.parameters():
                    param.requires_grad = False

            self.center_net = copy.deepcopy(net)
            for param in self.center_net.parameters():
                param.requires_grad = True

            # Reset optimizer and scheduler
            optimizer = self.hparams.optimizer(params=self.center_net.parameters())
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            trainer.optimizers = [optimizer]
            trainer.schedulers = [scheduler]
            
    def compute_uncertainty(
            self, 
            covariate_history, 
            treatment_history, 
            outcome_history, 
            outcomes, 
            treatments
            ):
        
        pred_outcomes_batch = []
        for net in self.nets:
            
            pred_outcomes, _ = net.infer(
                covariate_history=covariate_history,
                treatment_history=treatment_history,
                outcome_history=outcome_history,
                treatments=treatments,
                outcomes=outcomes,
            )

            pred_outcomes_batch.append(pred_outcomes)
        pred_outcomes_batch = torch.cat(pred_outcomes_batch, dim=0)

        return pred_outcomes_batch.mean(0), pred_outcomes_batch.var(0)


if __name__ == "__main__":
    _ = GeometricEnsembleUncertaintyModule(None, None, None, None)

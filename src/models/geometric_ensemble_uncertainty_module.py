from src.models.uncertainty_module import UncertaintyModule
import torch
from torch import nn
import copy
from functools import partial


class GeometricEnsembleUncertaintyModule(UncertaintyModule):
    
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        max_epochs: int = 40,
        n_forwards: int = 8,
        visualize: bool = False,
    ) -> None:

        super().__init__(net, optimizer, scheduler, compile, visualize)

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.visualize = visualize

        self.max_epochs = max_epochs
        self.n_forwards = n_forwards

        self.ensemble_size = 2
        self.ensemble = nn.ModuleList()
        for i in range(self.ensemble_size):
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
                    assert not (p0 == pi).all(), 'The ensemble shares parameters'

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

        if hasattr(self, 'center_net'):
            self.net = self.sample_net()
        else:
            self.net = self.ensemble[batch_idx % self.ensemble_size]

    def mix_apply(self, p_star, p, fn):
        for c_star, c in zip(p_star.children(), p.children()):
            self.mix_apply(c_star, c, fn)
        fn(p_star, p)
        return p_star

    def interpolate(self, m1, m2, weight):
        m12 = copy.deepcopy(m1)
        for p12, p2 in zip(m12.parameters(), m2.parameters()):
            interp = (1 - weight) * p12.data + weight * p2.data
            setattr(p12, 'data', interp)

    def sample_net(self):
        weight = 2 * torch.rand((1,))
        interpolate = partial(self.interpolate, weight=torch.abs(1-weight))

        if weight > 1.0:
            return self.mix_apply(
                self.center_net,
                self.ensemble[0],
                interpolate
                )
        else:
            return self.mix_apply(
                self.center_net,
                self.ensemble[1],
                interpolate
                )

    def on_train_epoch_end(self):

        if self.trainer.current_epoch == self.max_epochs:

            for net in self.ensemble:
                for param in net.parameters():
                    param.requires_grad = False

            self.center_net = copy.deepcopy(net)
            for param in self.center_net.parameters():
                param.requires_grad = True

            # Reset optimizer and scheduler
            optimizer = self.hparams.optimizer(params=self.center_net.parameters())
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            self.trainer.optimizers = [optimizer]
            self.trainer.schedulers = [scheduler]
            
    def compute_uncertainty(
            self, 
            covariate_history, 
            treatment_history, 
            outcome_history, 
            outcomes, 
            treatments
            ):

        pred_outcomes_batch = []
        for i in range(self.n_forwards):
            
            if hasattr(self, 'center_net'):
                net = self.sample_net()
            else:
                if i >= self.ensemble_size:
                    break
                net = self.ensemble[i]

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
    _ = GeometricEnsembleUncertaintyModule(None, None, None, None)

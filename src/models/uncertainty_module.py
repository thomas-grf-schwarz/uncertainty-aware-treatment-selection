from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from src.utils.figures import plot_trajectories_with_uncertainty
import matplotlib.pyplot as plt
import wandb


class UncertaintyModule(LightningModule):
    
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        visualize: bool = False,
    ) -> None:

        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net
        self.visualize = visualize

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param batch: A tensor of instances.
        :return: A tensor of logits.
        """
        return self.net(
            covariate_history=batch['covariate_history'],
            treatment_history=batch['treatment_history'],
            outcome_history=batch['outcome_history'],
            outcomes=batch['outcomes'],
            treatments=batch['treatments']
            )

    def on_train_start(self) -> None:
        pass

    def compute_uncertainty(
            self, 
            covariate_history, 
            treatment_history, 
            outcome_history, 
            outcomes, 
            treatments
            ):
        raise NotImplementedError
   
    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a dict)
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        losses = self.net.model_step(batch)

        for name, loss in losses.items():
            self.log(f"train/{name}", loss, on_step=True, on_epoch=True, prog_bar=True)

        return losses['loss']

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a dict)
        :param batch_idx: The index of the current batch.
        """
        if batch_idx == 0 and self.visualize:
            fig, ax = plot_trajectories_with_uncertainty(
                batch=batch, 
                model=self, 
                label=self.net.__class__.__name__
                )
            self.logger.experiment.log({"plot_trajectories_with_uncertainty": wandb.Image(fig)})
            plt.close(fig)

        losses = self.net.model_step(batch)

        for name, loss in losses.items():
            self.log(f"val/{name}", loss, on_step=True, on_epoch=True, prog_bar=True)

        return losses['loss']

    def on_validation_epoch_end(self) -> None:
        pass

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a dict)
        :param batch_idx: The index of the current batch.
        """
        losses = self.net.model_step(batch)
        self.log("test/outcome loss", losses['outcome loss'], on_step=True, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
    
    def on_fit_end(self):
        if hasattr(self.net, 'on_fit_end'):
            self.net.on_fit_end(self.trainer.datamodule)


if __name__ == "__main__":
    _ = UncertaintyModule(None, None, None, None)

from typing import Any, Dict, Tuple

from src.models.uncertainty_module import UncertaintyModule
import torch


class BayesianUncertaintyModule(UncertaintyModule):
    
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

    def compute_uncertainty(
            self,
            covariate_history,
            treatment_history,
            outcome_history,
            outcomes,
            treatments,
            ):

        mu_outcomes, var_outcomes, _ = self.net(
            covariate_history=covariate_history,
            treatment_history=treatment_history,
            outcome_history=outcome_history,
            treatments=treatments,
            outcomes=outcomes,
        )

        return mu_outcomes, var_outcomes


if __name__ == "__main__":
    _ = BayesianUncertaintyModule(None, None, None, None)

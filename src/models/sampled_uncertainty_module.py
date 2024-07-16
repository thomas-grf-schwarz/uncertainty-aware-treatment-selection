from typing import Any, Dict, Tuple
from src.utils.treat import create_identical_batch
from src.models.uncertainty_module import UncertaintyModule
import torch


class SampledUncertaintyModule(UncertaintyModule):
    
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        n_forwards: int = 8,
        visualize: bool = False,
    ) -> None:
        
        super().__init__(net, optimizer, scheduler, compile, visualize)

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
            treatments
            ):

        (covariate_history, 
         treatment_history, 
         outcome_history, 
         outcomes, 
         treatments) = create_identical_batch(
            (
                covariate_history, 
                treatment_history, 
                outcome_history, 
                outcomes, 
                treatments))

        self.net.train()  # enable dropout

        pred_outcomes, *_ = self.net.infer(
            covariate_history=covariate_history,
            treatment_history=treatment_history,
            outcome_history=outcome_history,
            outcomes=outcomes,
            treatments=treatments,
            )
        
        self.net.eval()
        return pred_outcomes.mean(0), pred_outcomes.var(0)


if __name__ == "__main__":
    _ = SampledUncertaintyModule(None, None, None, None)

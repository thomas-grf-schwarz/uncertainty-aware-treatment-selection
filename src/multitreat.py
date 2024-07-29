from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import rootutils
import torch
from lightning import LightningDataModule, LightningModule
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from functools import partial

import copy
import wandb
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

from src.utils.objective import (
    uncertainty_objective,
    composed_objective,
    mse_objective,
)

from src.utils.constraints import (
    leaky_clamp
)

from src.utils.treat import (
    select_treatment,
    sgdlike_loop,
    evaluate_treatment_selection,
    simulate_treatments,
)

from src.utils.figures import (
    plot_trajectories_with_uncertainty,
    plot_learning_curve,
    plot_uncertainty_penalty,
    plot_rmse_vs_least_uncertain_samples
)

from src.utils.loss import root_mean_square_error


log = RankedLogger(__name__, rank_zero_only=True)


def treat(cfg: DictConfig, model_name) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Selects treatments

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info(f"loading model weights <{cfg.ckpt_path}>")
    model.load_state_dict(torch.load(cfg.ckpt_path)['state_dict'], strict=False)

    log.info(f"Instantiating select_treatment <{cfg.select_treatment._target_}>")
    select_treatment = hydra.utils.instantiate(cfg.select_treatment)

    if cfg.get("logger"):
        log.info("Instantiating loggers...")
        logger: Logger = instantiate_loggers(cfg.get("logger"))
        if logger and isinstance(logger, list):
            logger = logger[0]

    # Simulate counterfactual data
    instance = datamodule.data_val[-1]
    counterfactual_treatments = datamodule.data_val[-2]['treatments']

    # Plot example trajectories to check on the quality of 
    # predictions and uncertainty estimates
    batch = next(iter(datamodule.val_dataloader()))
    fig1, ax1 = plot_trajectories_with_uncertainty(
        batch=batch,
        model=model,
        label=model_name,
        )
    logger.experiment.log({"plot_trajectories_with_uncertainty": wandb.Image(fig1)})

    # Set up metrics to collect
    rmses = {}
    rmses['selection'] = torch.empty(
        cfg.treat.n_replicates, 
        len(cfg.treat.uncertainty_weights)
        )
    rmses['counterfactual'] = torch.empty(
        cfg.treat.n_replicates,
        len(cfg.treat.uncertainty_weights)
        )
    rmses['all'] = torch.empty(
        cfg.treat.n_replicates, 
        len(cfg.treat.uncertainty_weights)
        )

    log.info("Started treatment selection...")
    for j, weight in enumerate(tqdm(cfg.treat.uncertainty_weights, desc="treatment selection")):

        for i in range(cfg.treat.n_replicates):

            if cfg.treat.replicate_type == 'instance':
                instance = datamodule.data_val[i]
                torch.manual_seed(cfg.seed)

            if cfg.treat.replicate_type == 'rerun':
                torch.manual_seed(i)

            # Simulate counterfactual data
            present_state = datamodule.data_val.to_state(
                outcome=instance['outcome_history'][..., -1:, :],
                covariate=instance['covariate_history'][..., -1:, :]
            )

            target_outcome = simulate_treatments(
                treatments=counterfactual_treatments,
                initial_state=present_state,
                t_horizon=datamodule.data_val.t_horizon,
                simulate_outcome=datamodule.data_val.simulate_outcome,
                )[None, :]
            target_outcome = torch.tensor(target_outcome, dtype=torch.float32)

            # Specify the objective for treatment selection to minimize
            if weight == 0.0:
                uncertainty_weight_ratio = 0.0
            else:
                uncertainty_weight_ratio = weight / (cfg.treat.mse_weight + weight)

            if cfg.treat.mse_weight == 0.0:
                mse_weight_ratio = 0.0
            else:
                mse_weight_ratio = cfg.treat.mse_weight / (cfg.treat.mse_weight + weight)
            
            uncertainty_mse_objective = partial(
                composed_objective,
                objectives={
                    uncertainty_objective: uncertainty_weight_ratio,
                    mse_objective: mse_weight_ratio
                    }
                )
                            
            # Perform treatment selection
            mu, var, treatments, losses = select_treatment(
                model=model,
                instance=copy.deepcopy(instance),
                target=target_outcome,
                objective=uncertainty_mse_objective,
                )

            # Compute the error due to the selection / optimization process
            rmses['selection'][i, j] = root_mean_square_error(
                input=mu,  # predicted outcome given selected treatment
                target=target_outcome,
            )

            # Compute the error due to the counterfactual prediction
            rmses['counterfactual'][i, j] = evaluate_treatment_selection(
                treatments=treatments,
                target=mu,  # predicted outcome given selected treatment
                initial_state=present_state,
                t_horizon=datamodule.data_val.t_horizon,
                simulate_outcome=datamodule.data_val.simulate_outcome,
                )

            # Compute the total error
            rmses['all'][i, j] = evaluate_treatment_selection(
                treatments=treatments,
                target=target_outcome,
                initial_state=present_state,
                t_horizon=datamodule.data_val.t_horizon,
                simulate_outcome=datamodule.data_val.simulate_outcome,
                )

            # Plot learning curves to check the treatment selection process
            fig2, ax2 = plot_learning_curve(
                losses,
                label=model_name,
                )
            logger.experiment.log({"treatment_selection_plt": wandb.Image(fig2)})

    object_dict = {
        'datamodule': datamodule,
        'model': model,
        'logger': logger,
    }

    metric_dict = {
        'rmses': rmses
    }

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="multitreat.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for treatment selection.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """

    fig3, ax3 = None, None
    fig3, ax3 = None, None
    fig4, ax4 = None, None
    fig5, ax5 = None, None
    fig6, ax6 = None, None

    for model_name in cfg.keys():

        # apply extra utilities
        extras(cfg[model_name])

        # perform treatment selection
        metric_dict, object_dict = treat(cfg[model_name], model_name)

        # log results
        rmse_results = wandb.Table(
            columns=cfg[model_name].uncertainty_weights,
            data=metric_dict['rmses'],
            )
        object_dict['logger'].experiment.log(
            {"RMSE results": rmse_results}
            )


        fig3, ax3 = plot_uncertainty_penalty(
            cfg[model_name].treat.uncertainty_weights, 
            rmses['selection'], 
            mean_rmses['selection'], 
            std_rmses['selection'],
            label=model_name,
            title='selection',
            fig=fig3, 
            ax=ax3,
        )
        fig4, ax4 = plot_uncertainty_penalty(
            cfg[model_name].treat.uncertainty_weights, 
            rmses['counterfactual'], 
            mean_rmses['counterfactual'], 
            std_rmses['counterfactual'],
            label=model_name,
            title='counterfactual',
            fig=fig4, 
            ax=ax4,
            )

        fig5, ax5 = plot_uncertainty_penalty(
            cfg[model_name].treat.uncertainty_weights, 
            rmses['all'], 
            mean_rmses['all'], 
            std_rmses['all'],
            label=model_name,
            title='all',
            fig=fig5, 
            ax=ax5
            )

        fig6, ax6 = plot_rmse_vs_least_uncertain_samples(
            dataset=object_dict['datamodule'].data_val,
            compute_uncertainty=object_dict['model'].compute_uncertainty,
            label=model_name,
            fig=fig6, 
            ax=ax6
            )

        object_dict['logger'].experiment.log(
            {"uncertainty_effect_plot_selection": wandb.Image(fig3)})
        object_dict['logger'].experiment.log(
            {"uncertainty_effect_plot_counterfactual": wandb.Image(fig4)})
        object_dict['logger'].experiment.log(
            {"uncertainty_effect_plot_all": wandb.Image(fig5)})
        object_dict['logger'].experiment.log(
            {"lowest_uncertainty_vs_rmse_plot": wandb.Image(fig6)})

    return None


if __name__ == "__main__":
    main()

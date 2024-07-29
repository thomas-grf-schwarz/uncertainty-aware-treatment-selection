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


def treat(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Selects treatments

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    rmses = {}
    for model_name, ckpt_path in zip(cfg.models, cfg.ckpt_paths):

        log.info(f"Instantiating model <{cfg.models[model_name]._target_}>")
        model: LightningModule = hydra.utils.instantiate(cfg.models[model_name])

        log.info(f"loading model weights <{ckpt_path}>")
        model.load_state_dict(torch.load(ckpt_path)['state_dict'], strict=False)

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

        try:
            fig6, ax6 = plot_rmse_vs_least_uncertain_samples(
                dataset=datamodule.data_val,
                compute_uncertainty=model.compute_uncertainty,
                label=model_name,
                fig=fig6, 
                ax=ax6
            )
        except UnboundLocalError:
            fig6, ax6 = plot_rmse_vs_least_uncertain_samples(
                dataset=datamodule.data_val,
                compute_uncertainty=model.compute_uncertainty,
                label=model_name,
            )

        # Impose constraint on the treatment selection
        constraints = partial(
            leaky_clamp, 
            max=cfg.treat.clamp_max, 
            slope=cfg.treat.clamp_slope
            )

        # Set up metrics to collect
        rmses[model_name] = {}
        rmses[model_name]['selection'] = torch.empty(
            cfg.treat.n_replicates, len(cfg.treat.uncertainty_weights)
            )
        rmses[model_name]['counterfactual'] = torch.empty(
            cfg.treat.n_replicates, len(cfg.treat.uncertainty_weights)
            )
        rmses[model_name]['all'] = torch.empty(
            cfg.treat.n_replicates, len(cfg.treat.uncertainty_weights)
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
                    constraints=constraints,
                    optimization_loop=sgdlike_loop,
                    optimizer_cls=getattr(torch.optim, cfg.treat.optimizer),
                    n_iter=cfg.treat.n_iter,
                    lr=cfg.treat.learning_rate
                )

                # Compute the error due to the selection / optimization process
                rmses[model_name]['selection'][i, j] = root_mean_square_error(
                    input=mu,  # predicted outcome given selected treatment
                    target=target_outcome,
                )

                # Compute the error due to the counterfactual prediction
                rmses[model_name]['counterfactual'][i, j] = evaluate_treatment_selection(
                    treatments=treatments,
                    target=mu,  # predicted outcome given selected treatment
                    initial_state=present_state,
                    t_horizon=datamodule.data_val.t_horizon,
                    simulate_outcome=datamodule.data_val.simulate_outcome,
                    )

                # Compute the total error
                rmses[model_name]['all'][i, j] = evaluate_treatment_selection(
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

        mean_rmses = {}
        std_rmses = {}
        for k, v in rmses[model_name].items():
            mean_rmses[k] = v.mean(dim=0)
            std_rmses[k] = v.std(dim=0)

        try:
            fig3, ax3 = plot_uncertainty_penalty(
                cfg.treat.uncertainty_weights, 
                rmses[model_name]['selection'], 
                mean_rmses['selection'], 
                std_rmses['selection'],
                label=model_name,
                title='selection',
                fig=fig3, 
                ax=ax3,
            )
        except UnboundLocalError:
            fig3, ax3 = plot_uncertainty_penalty(
                cfg.treat.uncertainty_weights, 
                rmses[model_name]['selection'], 
                mean_rmses['selection'], 
                std_rmses['selection'],
                label=model_name,
                title='selection',
            )

        try:
            fig4, ax4 = plot_uncertainty_penalty(
                cfg.treat.uncertainty_weights, 
                rmses[model_name]['counterfactual'], 
                mean_rmses['counterfactual'], 
                std_rmses['counterfactual'],
                label=model_name,
                title='counterfactual',
                fig=fig4, 
                ax=ax4
            )
        except UnboundLocalError:
            fig4, ax4 = plot_uncertainty_penalty(
                cfg.treat.uncertainty_weights, 
                rmses[model_name]['counterfactual'], 
                mean_rmses['counterfactual'], 
                std_rmses['counterfactual'],
                label=model_name,
                title='counterfactual',
            )


        try:
            fig5, ax5 = plot_uncertainty_penalty(
                cfg.treat.uncertainty_weights, 
                rmses[model_name]['all'], 
                mean_rmses['all'], 
                std_rmses['all'],
                label=model_name,
                title='all',
                fig=fig5, 
                ax=ax5
            )
        except UnboundLocalError:
            fig5, ax5 = plot_uncertainty_penalty(
                cfg.treat.uncertainty_weights, 
                rmses[model_name]['all'], 
                mean_rmses['all'], 
                std_rmses['all'],
                label=model_name,
                title='all',
            )

    logger.experiment.log({"uncertainty_effect_plot_selection": wandb.Image(fig3)})
    logger.experiment.log({"uncertainty_effect_plot_counterfactual": wandb.Image(fig4)})
    logger.experiment.log({"uncertainty_effect_plot_all": wandb.Image(fig5)})
    logger.experiment.log({"lowest_uncertainty_vs_rmse_plot": wandb.Image(fig6)})

    return None, None


@hydra.main(version_base="1.3", config_path="../configs", config_name="multitreat.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for treatment selection.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # perform treatment selection
    metric_dict = treat(cfg)

    return None

if __name__ == "__main__":
    main()

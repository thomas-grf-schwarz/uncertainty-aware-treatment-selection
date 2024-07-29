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
import pandas as pd
import pathlib as plb

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import (
    RankedLogger,
    extras,
    instantiate_loggers,
)

from src.utils.objective import (
    uncertainty_objective,
    composed_objective,
    mse_objective,
)

from src.utils.treat import (
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
from src.train import train


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

    log.info(f"Instantiating select_treatment <{cfg.treat.select_treatment._target_}>")
    select_treatment = hydra.utils.instantiate(cfg.treat.select_treatment)

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
                )
            target_outcome = torch.tensor(
                data=target_outcome[None, :],
                dtype=torch.float32
                )

            # Specify the objective for treatment selection to minimize
            if weight == 0.0:
                weight_ratio = 0.0
            else:
                weight_ratio = weight / (cfg.treat.mse_weight + weight)

            if cfg.treat.mse_weight == 0.0:
                mse_weight_ratio = 0.0
            else:
                mse_weight_ratio = cfg.treat.mse_weight / (cfg.treat.mse_weight + weight)
            
            uncertainty_mse_objective = partial(
                composed_objective,
                objectives={
                    uncertainty_objective: weight_ratio,
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

    # Compute uncertainty and error on the validation data
    uncertainty = {}
    uncertainty['rmses'] = []
    uncertainty['vars'] = []
    for instance in datamodule.data_val:
        instance.pop('initial_state')
        mu, var = model.compute_uncertainty(**instance)
        rmse = root_mean_square_error(mu, instance['outcomes'])
        uncertainty['rmses'].append(rmse)
        uncertainty['vars'].append(var.squeeze().mean())

    object_dict = {
        'datamodule': datamodule,
        'model': model,
        'logger': logger,
    }

    metric_dict = {
        'rmses': rmses,
        'uncertainty': uncertainty
    }

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="treat.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for treatment selection.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """

    # apply extra utilities
    extras(cfg)

    # train a model if no pretrained weights are provided
    if cfg.get('ckpt_path') is None:
        _, object_dict = train(cfg)
        trainer = object_dict['trainer']
        cfg.ckpt_path = trainer.checkpoint_callback.best_model_path

    # perform treatment selection
    metric_dict, object_dict = treat(cfg, '')

    # log uncertainty results
    results_uncertainty = pd.DataFrame(
        data=metric_dict['uncertainty'],
    )

    object_dict['logger'].experiment.log(
        {
            'uncertainty': wandb.Table(
                dataframe=results_uncertainty
            )
        }
    )

    results_uncertainty.to_csv(
        plb.Path(cfg.paths.output_dir) / 'uncertainty.csv',
        index=False
        )

    # log treatment selection metrics
    results_selection = pd.DataFrame(
        columns=[str(x) for x in cfg.treat.uncertainty_weights],
        data=[x.tolist() for x in metric_dict['rmses']['selection']]
    )

    results_counterfactual = pd.DataFrame(
        columns=[str(x) for x in cfg.treat.uncertainty_weights],
        data=[x.tolist() for x in metric_dict['rmses']['counterfactual']]
    )

    results_all = pd.DataFrame(
        columns=[str(x) for x in cfg.treat.uncertainty_weights],
        data=[x.tolist() for x in metric_dict['rmses']['all']]
    )

    results_selection.to_csv(
        plb.Path(cfg.paths.output_dir) / 'rmse_selection.csv',
        index=False
        )
    results_counterfactual.to_csv(
        plb.Path(cfg.paths.output_dir) / 'rmse_counterfactual.csv',
        index=False
        )
    results_all.to_csv(
        plb.Path(cfg.paths.output_dir) / 'rmse_all.csv',
        index=False
        )

    object_dict['logger'].experiment.log(
        {
            'RMSE selection': wandb.Table(
                dataframe=results_selection
            )
        }
    )

    object_dict['logger'].experiment.log(
        {
            'RMSE counterfactual': wandb.Table(
                dataframe=results_counterfactual
            )
        }
    )

    object_dict['logger'].experiment.log(
        {
            'RMSE all': wandb.Table(
                dataframe=results_all
            )
        }
    )

    fig, ax = plot_uncertainty_penalty(
        cfg.treat.uncertainty_weights,
        metric_dict['rmses']['all'],
        metric_dict['rmses']['all'].mean(dim=0),
        metric_dict['rmses']['all'].std(dim=0),
        label='',
        title='all'
        )

    object_dict['logger'].experiment.log(
        {"uncertainty_effect_plot_selection": wandb.Image(fig)})

    return None


if __name__ == '__main__':
    main()

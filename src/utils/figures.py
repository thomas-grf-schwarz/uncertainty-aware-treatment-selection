import torch
import matplotlib.pyplot as plt
from typing import List
import pandas as pd
import itertools

from src.utils.loss import root_mean_square_error

def plot_uncertainty_penalty_all(data: List[pd.DataFrame]):

    uncertainty_weights, rmses, mean_rmses, std_rmses = ...

    for uncertainty_weights, rmses in zip():
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title('The effect of uncertainty penalty on reliable treatment selection: counterfactual target')
        ax.plot(uncertainty_weights, rmses.T, alpha=0.3)
        ax.plot(uncertainty_weights, mean_rmses, '-o', label='Mean RMSE')
        ax.fill_between(uncertainty_weights, mean_rmses - std_rmses, mean_rmses + std_rmses, alpha=0.3)
        ax.set_xscale('log')
        ax.set_xlabel('Uncertainty Weight')
        ax.set_ylabel('RMSE')
        ax.grid(True)
        ax.legend()


def compute_uncertainty_batch(batch, model, nxn=4):

    mu, var = [], []

    for covariate_history, treatment_history, outcome_history, outcomes, treatments in zip(
        batch['covariate_history'],
        batch['treatment_history'], 
        batch['outcome_history'], 
        batch['outcomes'], 
        batch['treatments']
        ):

        mu_instance, var_instance = model.compute_uncertainty(
            covariate_history=covariate_history,
            treatment_history=treatment_history, 
            outcome_history=outcome_history, 
            outcomes=outcomes, 
            treatments=treatments
        )
        mu.append(mu_instance)
        var.append(var_instance)    

    mu = torch.stack(mu, dim=0)
    var = torch.stack(var, dim=0)
    return mu, var


def plot_performance_vs_uncertainty(batch, model, nxn=4, fig=None, axs=None):
    
    mu, var = compute_uncertainty_batch(batch, model)
    target = batch['outcomes']

    t_inds = torch.arange(target.shape[-2])

    if fig is None:
        fig, axs = plt.subplots(nxn, nxn, figsize=(12, 12), sharex='col', sharey='row')

    for idx, ax in enumerate(axs.flat):
        mu_pred = mu.squeeze()[idx].detach()
        var_pred = var.squeeze()[idx].detach().squeeze()

        target = target[idx].detach().squeeze()

        rmse = torch.sqrt((mu_pred - target)**2)

        ax.scatter(rmse, idx)

        if idx % nxn == 0:
            ax.set_ylabel('Variance')
        if idx > nxn**2 - nxn - 1:
            ax.set_xlabel('RMSE')
        ax.grid(True)
        ax.legend()

    fig.suptitle('Performance vs Uncertainty', fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    return fig, ax


def plot_learning_curve(losses, label='', fig=None, ax=None):
    if fig is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(losses, marker='o', linestyle='-')
    ax.set_title('Treatment selection: Learning Curve')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid(True)
    ax.legend([label.upper()])
    return fig, ax


def plot_uncertainty_penalty(
        uncertainty_weights,
        rmses,
        mean_rmses,
        std_rmses,
        label='',
        title='',
        fig=None,
        ax=None
        ):
    
    if fig is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # ax.set_title(f'Effect of uncertainty penalty on reliable treatment selection: {title}')
    ax.plot(uncertainty_weights, mean_rmses, '-o', label=label.upper())
    color = ax.get_lines()[-1].get_color()
    ax.plot(uncertainty_weights, rmses.T, alpha=0.3, color=color)
    ax.fill_between(uncertainty_weights, mean_rmses - std_rmses, mean_rmses + std_rmses, 
                    alpha=0.05, color=color)

    ax.set_xscale('log')
    ax.set_xlabel('Uncertainty Weight')
    ax.set_ylabel('RMSE')
    ax.grid(True)
    ax.legend()
    
    return fig, ax


@torch.no_grad()
def plot_trajectories_with_uncertainty(
    batch,
    model,
    nxn=4,
    label='',
    fig=None,
    axs=None,
    ylim_std=True,
    ):

    mus, vars = [], []
    outcomes_targets = batch['outcomes']

    for covariate_history, treatment_history, outcome_history, outcomes, treatments in zip(
        batch['covariate_history'],
        batch['treatment_history'], 
        batch['outcome_history'], 
        batch['outcomes'], 
        batch['treatments']
        ):

        mu, var = model.compute_uncertainty(
            covariate_history=covariate_history,
            treatment_history=treatment_history, 
            outcome_history=outcome_history, 
            outcomes=outcomes, 
            treatments=treatments,
        )
        mus.append(mu)
        vars.append(var)       

    mus = torch.stack(mus, dim=0)
    vars = torch.stack(vars, dim=0)

    t_inds = torch.arange(mus.shape[-2])
    if fig is None:
        fig, axs = plt.subplots(
            ncols=nxn,
            nrows=nxn,
            figsize=(12, 12),
            sharex='col',
            sharey=None if ylim_std else 'row',
        )

    for idx, ax in enumerate(axs.flatten()):
        mu_pred = mus.squeeze()[idx]
        var_pred = vars.squeeze()[idx].squeeze()
        outcome_target = outcomes_targets[idx].squeeze()

        ax.plot(t_inds, outcome_target, label='target', linestyle='--', color='b')
        ax.plot(t_inds, mu_pred, label=f'{label.upper()}: predicted', color='b')
        ax.fill_between(t_inds, mu_pred - var_pred, mu_pred + var_pred, 
                        alpha=0.3, color='b')
        if ylim_std:
            ax.set_ylim(
                bottom=mu_pred.min() - 2*mu_pred.std(),
                top=mu_pred.max() + 2*mu_pred.std()
                )

        if idx % nxn == 0:
            ax.set_ylabel('Outcome')
        if idx > nxn**2 - nxn - 1:
            ax.set_xlabel('Time indices')
        ax.grid(True)

    fig.legend(['Target', f'Predicted: {label.upper()}'], loc='lower right')
    fig.suptitle(f'{label.upper()}: Trajectories with uncertainty', fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    return fig, axs


# def plot_treatment_trajectories(): ...

@torch.no_grad()
def plot_rmse_vs_least_uncertain_samples(
    dataset, 
    compute_uncertainty,
    label='',
    fig=None,
    ax=None,
    log_scale=True
    ):

    rmses = []
    vars = []
    for i in range(len(dataset)):
        instance = dataset[i]
        instance.pop('initial_state')
        mu, var = compute_uncertainty(**instance)
        rmse = root_mean_square_error(mu, instance['outcomes'])
        var = var.mean()
        rmses.append(rmse)
        vars.append(var.squeeze().mean())

    rmses = torch.stack(rmses)
    vars = torch.stack(vars)

    sort_inds = torch.argsort(vars, descending=False)
    rmses_sorted = rmses[sort_inds]

    N = len(vars)
    rmses_n_unsorted = []
    rmses_n_sorted = []
    percent_n = []
    for n in torch.arange(0, N, 4):
        percent_n.append(100 * n / N)
        rmses_n_sorted.append(
           rmses_sorted[:n].mean().detach()
           )
        rmses_n_unsorted.append(
           rmses[:n].mean().detach()
           )

    if fig is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    if log_scale:
        ax.set_yscale('log')
        ax.set_ylabel('log RMSE', fontsize=14)
    else:
        ax.set_ylabel('RMSE', fontsize=14)

    ax.set_title('Effect of holding back uncertain samples on outcome RMSE', fontsize=16)
    ax.plot(percent_n, rmses_n_sorted, marker='o', linestyle='-', label=label.upper())
    color = ax.get_lines()[-1].get_color()
    ax.plot(percent_n, rmses_n_unsorted, linestyle='--', color=color, alpha=0.3)
    ax.set_xlabel('Bottom percent uncertain samples', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
 
    return fig, ax

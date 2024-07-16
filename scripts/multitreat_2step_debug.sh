#!/bin/bash
# Schedule execution of many runs
log_dir=/home/tschw/Documents/Neuroengineering/causal_dynamics/optimal_treatments/logs
# ckpt_paths="$log_dir/train/runs/2024-06-27_11-11-49/checkpoints/epoch_015.ckpt, $log_dir/train/runs/2024-06-26_22-23-42/checkpoints/epoch_030.ckpt, $log_dir/train/runs/2024-06-26_22-46-15/checkpoints/epoch_037.ckpt, $log_dir/train/runs/2024-06-27_12-25-22/checkpoints/epoch_063.ckpt"
ckpt_paths="$log_dir/train/runs/2024-07-01_15-11-38/checkpoints/epoch_030.ckpt, $log_dir/train/runs/2024-06-26_22-23-42/checkpoints/epoch_030.ckpt, $log_dir/train/runs/2024-06-26_22-46-15/checkpoints/epoch_037.ckpt, $log_dir/train/runs/2024-07-01_12-50-26/checkpoints/epoch_093.ckpt"

# Effect of the mse weight
# HYDRA_FULL_ERROR=1 python src/multitreat.py experiment=multitreat_2step logger=wandb +logger.wandb.name=na_mseweight_low ++ckpt_paths="[$ckpt_paths]" ++treat.n_iter=4 ++treat.n_replicates=2 ++treat.mse_weight=0.00001

# # Effect of the clamp
# HYDRA_FULL_ERROR=1 python src/multitreat.py experiment=multitreat_2step logger=wandb +logger.wandb.name=na_clampslope_low ++ckpt_paths="[$ckpt_paths]" ++treat.n_iter=4 ++treat.n_replicates=2 ++treat.clamp_slope=0.001
HYDRA_FULL_ERROR=1 python src/multitreat.py experiment=multitreat_2step logger=wandb +logger.wandb.name=na_clampmax_low ++ckpt_paths="[$ckpt_paths]" ++treat.n_iter=4 ++treat.n_replicates=2 ++treat.clamp_max=4.0 ++treat.n_iter=2
# HYDRA_FULL_ERROR=1 python src/multitreat.py experiment=multitreat_2step logger=wandb +logger.wandb.name=na_clampmax_high ++ckpt_paths="[$ckpt_paths]" ++treat.n_iter=4 ++treat.n_replicates=2 ++treat.clamp_max=16.0

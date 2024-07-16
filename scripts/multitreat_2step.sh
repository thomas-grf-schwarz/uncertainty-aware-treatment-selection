#!/bin/bash
# Schedule execution of many runs
log_dir=/home/tschw/Documents/Neuroengineering/causal_dynamics/optimal_treatments/logs
# ckpt_paths="$log_dir/train/runs/2024-06-27_11-11-49/checkpoints/epoch_015.ckpt, $log_dir/train/runs/2024-06-26_22-23-42/checkpoints/epoch_030.ckpt, $log_dir/train/runs/2024-06-26_22-46-15/checkpoints/epoch_037.ckpt, $log_dir/train/runs/2024-06-27_12-25-22/checkpoints/epoch_063.ckpt"
ckpt_paths="$log_dir/train/runs/2024-07-01_15-11-38/checkpoints/epoch_030.ckpt, $log_dir/train/runs/2024-06-26_22-23-42/checkpoints/epoch_030.ckpt, $log_dir/train/runs/2024-06-26_22-46-15/checkpoints/epoch_037.ckpt, $log_dir/train/runs/2024-07-01_12-50-26/checkpoints/epoch_093.ckpt"

# Effect of the learning rate
HYDRA_FULL_ERROR=1 python src/multitreat.py experiment=multitreat_2step logger=wandb +logger.wandb.name=multitreat_2step_lr_longer ++ckpt_paths="[$ckpt_paths]" ++treat.learning_rate=0.01 ++treat.n_iter=50

# Effect of the clamp
HYDRA_FULL_ERROR=1 python src/multitreat.py experiment=multitreat_2step logger=wandb +logger.wandb.name=multitreat_2step_clampslope_low ++ckpt_paths="[$ckpt_paths]" ++treat.clamp_slope=0.001
HYDRA_FULL_ERROR=1 python src/multitreat.py experiment=multitreat_2step logger=wandb +logger.wandb.name=multitreat_2step_clampmax_low ++ckpt_paths="[$ckpt_paths]" ++treat.clamp_max=4.0 ++treat.learning_rate=0.025
HYDRA_FULL_ERROR=1 python src/multitreat.py experiment=multitreat_2step logger=wandb +logger.wandb.name=multitreat_2step_clampmax_high ++ckpt_paths="[$ckpt_paths]" ++treat.clamp_max=16.0

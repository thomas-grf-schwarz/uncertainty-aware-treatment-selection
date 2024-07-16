#!/bin/bash
# Schedule execution of many runs
log_dir=/home/tschw/Documents/Neuroengineering/causal_dynamics/optimal_treatments/logs
# ckpt_paths="$log_dir/train/runs/2024-06-27_11-11-49/checkpoints/epoch_015.ckpt, $log_dir/train/runs/2024-06-26_22-23-42/checkpoints/epoch_030.ckpt, $log_dir/train/runs/2024-06-26_22-46-15/checkpoints/epoch_037.ckpt, $log_dir/train/runs/2024-06-27_12-25-22/checkpoints/epoch_063.ckpt"
# ckpt_paths="$log_dir/train/runs/2024-07-01_15-11-38/checkpoints/epoch_030.ckpt, $log_dir/train/runs/2024-06-26_22-23-42/checkpoints/epoch_030.ckpt, $log_dir/train/runs/2024-07-02_13-29-14/checkpoints/epoch_040.ckpt, $log_dir/train/runs/2024-07-01_12-50-26/checkpoints/epoch_093.ckpt"
# ckpt_paths="$log_dir/train/runs/2024-07-03_12-49-46/checkpoints/epoch_059.ckpt, $log_dir/train/runs/2024-06-26_22-23-42/checkpoints/epoch_030.ckpt, $log_dir/train/runs/2024-07-03_16-50-20/checkpoints/epoch_015.ckpt, $log_dir/train/runs/2024-07-03_11-16-33/checkpoints/epoch_042.ckpt"
# ckpt_paths="$log_dir/train/runs/2024-07-03_12-49-46/checkpoints/epoch_059.ckpt, $log_dir/train/runs/2024-06-26_22-23-42/checkpoints/epoch_030.ckpt, $log_dir/train/runs/2024-07-03_16-50-20/checkpoints/epoch_015.ckpt, $log_dir/train/runs/2024-07-07_21-30-59/checkpoints/epoch_097.ckpt"
# ckpt_paths="$log_dir/train/runs/2024-07-08_18-49-28/checkpoints/epoch_039.ckpt, $log_dir/train/runs/2024-07-08_19-07-50/checkpoints/epoch_096.ckpt, $log_dir/train/runs/2024-07-08_20-05-50/checkpoints/epoch_040.ckpt, $log_dir/train/runs/2024-07-08_17-39-37/checkpoints/epoch_093.ckpt"
ckpt_paths="$log_dir/train/runs/2024-07-12_15-26-25/checkpoints/epoch_099.ckpt, $log_dir/train/runs/2024-07-16_13-13-03/checkpoints/epoch_099.ckpt, $log_dir/train/runs/2024-07-16_12-51-53/checkpoints/epoch_005.ckpt, $log_dir/train/runs/2024-07-12_15-11-51/checkpoints/epoch_012.ckpt"

# Effect of the learning rate
HYDRA_FULL_ERROR=1 python src/multitreat.py experiment=multitreat_10step logger=wandb +logger.wandb.name=multitreat_10step_lr_longer ++ckpt_paths="[$ckpt_paths]" ++treat.learning_rate=0.01 ++treat.n_iter=50 data=covid_dynamics

# Effect of the clamp
HYDRA_FULL_ERROR=1 python src/multitreat.py experiment=multitreat_10step logger=wandb +logger.wandb.name=multitreat_10step_clampslope_low ++ckpt_paths="[$ckpt_paths]" ++treat.clamp_slope=0.001 data=covid_dynamics
HYDRA_FULL_ERROR=1 python src/multitreat.py experiment=multitreat_10step logger=wandb +logger.wandb.name=multitreat_10step_clampmax_low ++ckpt_paths="[$ckpt_paths]" ++treat.clamp_max=4.0 ++treat.learning_rate=0.025 data=covid_dynamics

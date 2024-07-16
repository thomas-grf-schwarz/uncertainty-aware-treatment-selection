#!/bin/bash
# Schedule execution of many runs
log_dir=/home/tschw/Documents/Neuroengineering/causal_dynamics/optimal_treatments/logs
ckpt_paths="$log_dir/train/runs/2024-06-21_11-28-58/checkpoints/epoch_090.ckpt, $log_dir/train/runs/2024-06-21_11-16-49/checkpoints/epoch_076.ckpt"

# Effect of the mse weight
HYDRA_FULL_ERROR=1 python src/multitreat.py experiment=multitreat_bncde_cfode_1step logger=wandb +logger.wandb.name=multitreat_1step_mseweight_low ++ckpt_paths="[$ckpt_paths]" ++treat.mse_weight=0.00001

# Effect of the learning rate
HYDRA_FULL_ERROR=1 python src/multitreat.py experiment=multitreat_bncde_cfode_1step logger=wandb +logger.wandb.name=multitreat_1step_lr_longer ++ckpt_paths="[$ckpt_paths]" ++treat.learning_rate=0.1 ++treat.n_iter=50

# Effect of the clamp
HYDRA_FULL_ERROR=1 python src/multitreat.py experiment=multitreat_bncde_cfode_1step logger=wandb +logger.wandb.name=multitreat_1step_clampslope_low ++ckpt_paths="[$ckpt_paths]" ++treat.clamp_slope=0.001
HYDRA_FULL_ERROR=1 python src/multitreat.py experiment=multitreat_bncde_cfode_1step logger=wandb +logger.wandb.name=multitreat_1step_clampmax_low ++ckpt_paths="[$ckpt_paths]" ++treat.clamp_max=4.0
HYDRA_FULL_ERROR=1 python src/multitreat.py experiment=multitreat_bncde_cfode_1step logger=wandb +logger.wandb.name=multitreat_1step_clampmax_high ++ckpt_paths="[$ckpt_paths]" ++treat.clamp_max=16.0

#!/bin/bash
# Schedule execution of many runs
log_dir=/home/tschw/Documents/Neuroengineering/causal_dynamics/optimal_treatments/logs
ckpt_paths="$log_dir/train/runs/2024-06-18_18-46-17/checkpoints/last.ckpt"

# Effect of the mse weight
HYDRA_FULL_ERROR=1 python src/multitreat.py experiment=multitreat logger=wandb +logger.wandb.name=crn_ratioweightswith0_mseweight_low ++model.net.theta=0.4 ++ckpt_paths="[$ckpt_paths]" ++treat.mse_weight=0.00001
HYDRA_FULL_ERROR=1 python src/multitreat.py experiment=multitreat logger=wandb +logger.wandb.name=crn_ratioweightswith0_mseweight_high ++model.net.theta=0.4 ++ckpt_paths="[$ckpt_paths]" ++treat.mse_weight=0.1
HYDRA_FULL_ERROR=1 python src/multitreat.py experiment=multitreat logger=wandb +logger.wandb.name=crn_ratioweightswith0_mseweight_higher ++model.net.theta=0.4 ++ckpt_paths="[$ckpt_paths]" ++treat.mse_weight=1.0

# Effect of the learning rate
HYDRA_FULL_ERROR=1 python src/multitreat.py experiment=multitreat logger=wandb +logger.wandb.name=crn_ratioweightswith0_lr_high ++model.net.theta=0.4 ++ckpt_paths="[$ckpt_paths]" ++treat.learning_rate=1.0
HYDRA_FULL_ERROR=1 python src/multitreat.py experiment=multitreat logger=wandb +logger.wandb.name=crn_ratioweightswith0_lr_longer ++model.net.theta=0.4 ++ckpt_paths="[$ckpt_paths]" ++treat.learning_rate=0.1 ++treat.n_iter=50
HYDRA_FULL_ERROR=1 python src/multitreat.py experiment=multitreat logger=wandb +logger.wandb.name=crn_ratioweightswith0_lr_low ++model.net.theta=0.4 ++ckpt_paths="[$ckpt_paths]" ++treat.learning_rate=0.01 ++treat.n_iter=100

# Effect of the clamp
HYDRA_FULL_ERROR=1 python src/multitreat.py experiment=multitreat logger=wandb +logger.wandb.name=crn_ratioweightswith0_clampslope_high ++model.net.theta=0.4 ++ckpt_paths="[$ckpt_paths]" ++treat.clamp_slope=0.1
HYDRA_FULL_ERROR=1 python src/multitreat.py experiment=multitreat logger=wandb +logger.wandb.name=crn_ratioweightswith0_clampslope_low ++model.net.theta=0.4 ++ckpt_paths="[$ckpt_paths]" ++treat.clamp_slope=0.001
HYDRA_FULL_ERROR=1 python src/multitreat.py experiment=multitreat logger=wandb +logger.wandb.name=crn_ratioweightswith0_clampmax_low ++model.net.theta=0.4 ++ckpt_paths="[$ckpt_paths]" ++treat.clamp_max=4.0
HYDRA_FULL_ERROR=1 python src/multitreat.py experiment=multitreat logger=wandb +logger.wandb.name=crn_ratioweightswith0_clampmax_high ++model.net.theta=0.4 ++ckpt_paths="[$ckpt_paths]" ++treat.clamp_max=16.0

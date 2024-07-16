#!/bin/bash
# Schedule execution of many runs

# Effect of mu and sigma
HYDRA_FULL_ERROR=1 python src/train.py experiment=cfode_train logger=wandb +logger.wandb.name=cfode_theta ++model.net.theta=0.4 ++model.optimizer.lr=0.008

HYDRA_FULL_ERROR=1 python src/train.py experiment=cfode_train logger=wandb +logger.wandb.name=cfode_theta ++model.net.theta=0.025 ++model.optimizer.lr=0.008

HYDRA_FULL_ERROR=1 python src/train.py experiment=cfode_train logger=wandb +logger.wandb.name=cfode_sigma ++model.net.sigma=0.4 ++model.optimizer.lr=0.008

HYDRA_FULL_ERROR=1 python src/train.py experiment=cfode_train logger=wandb +logger.wandb.name=cfode_sigma ++model.net.sigma=0.025 ++model.optimizer.lr=0.008

# Effect of capacity
HYDRA_FULL_ERROR=1 python src/train.py experiment=cfode_train logger=wandb +logger.wandb.name=cfode_2xlower_hidden ++model.net.hidden_size=64 ++model.optimizer.lr=0.008

HYDRA_FULL_ERROR=1 python src/train.py experiment=cfode_train logger=wandb +logger.wandb.name=cfode_2xhigher_hidden ++model.net.hidden_size=256 ++model.optimizer.lr=0.008

HYDRA_FULL_ERROR=1 python src/train.py experiment=cfode_train logger=wandb +logger.wandb.name=cfode_2xlayers ++model.net.num_layers=2 ++model.optimizer.lr=0.008


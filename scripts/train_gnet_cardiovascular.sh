#!/bin/bash

# HYDRA_FULL_ERROR=1 python src/train.py experiment=gnet_train logger=wandb +logger.wandb.name=gnet_run_cardiovascular_doublehidden data=cardiovascular_dynamics ++model.visualize=false ++model.net.hidden_size=128

# HYDRA_FULL_ERROR=1 python src/train.py experiment=gnet_train logger=wandb +logger.wandb.name=gnet_run_cardiovascular_quarterlr data=cardiovascular_dynamics ++model.visualize=false ++model.optimizer.lr=0.0005

# HYDRA_FULL_ERROR=1 python src/train.py experiment=gnet_train logger=wandb +logger.wandb.name=gnet_run_cardiovascular_quarterdropout data=cardiovascular_dynamics ++model.visualize=false ++model.net.p_dropout=0.05


HYDRA_FULL_ERROR=1 python src/train.py experiment=gnet_train logger=wandb +logger.wandb.name=gnet_run_cardiovascular_doublehidden_doublelayers data=cardiovascular_dynamics ++model.visualize=false ++model.net.hidden_size=128 ++model.net.num_layers=2 ++model.net.p_dropout=0.0 ++model.optimizer.lr=0.001

HYDRA_FULL_ERROR=1 python src/train.py experiment=gnet_train logger=wandb +logger.wandb.name=gnet_run_cardiovascular_doublehidden_triplelayers data=cardiovascular_dynamics ++model.visualize=false ++model.net.hidden_size=128 ++model.net.num_layers=3 ++model.net.p_dropout=0.0 ++model.optimizer.lr=0.001

HYDRA_FULL_ERROR=1 python src/train.py experiment=gnet_train logger=wandb +logger.wandb.name=gnet_run_cardiovascular_triplelayers data=cardiovascular_dynamics ++model.visualize=false ++model.net.hidden_size=64 ++model.net.num_layers=3 ++model.net.p_dropout=0.0 ++model.optimizer.lr=0.001


# HYDRA_FULL_ERROR=1 python src/train.py experiment=gnet_train logger=wandb +logger.wandb.name=gnet_run_cardiovascular_doublehidden data=cardiovascular_dynamics ++model.visualize=false ++model.net.hidden_size=64 ++model.net.num_layers=3 ++model.net.p_dropout=0.0 ++model.optimizer.lr=0.01

# HYDRA_FULL_ERROR=1 python src/train.py experiment=gnet_train logger=wandb +logger.wandb.name=gnet_run_cardiovascular_doublehidden data=cardiovascular_dynamics ++model.visualize=false ++model.net.hidden_size=64 ++model.net.num_layers=3 ++model.net.p_dropout=0.0 ++model.optimizer.lr=0.001

# HYDRA_FULL_ERROR=1 python src/train.py experiment=gnet_train logger=wandb +logger.wandb.name=gnet_run_cardiovascular_doublehidden data=cardiovascular_dynamics ++model.visualize=false ++model.net.hidden_size=64 ++model.net.num_layers=3 ++model.net.p_dropout=0.0 ++model.optimizer.lr=0.0001

# HYDRA_FULL_ERROR=1 python src/train.py experiment=gnet_train logger=wandb +logger.wandb.name=gnet_run_cardiovascular_doublehidden data=cardiovascular_dynamics ++model.visualize=false ++model.net.hidden_size=64 ++model.net.num_layers=3 ++model.net.p_dropout=0.0 ++model.optimizer.lr=0.00001


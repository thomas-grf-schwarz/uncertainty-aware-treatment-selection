#!/bin/bash

HYDRA_FULL_ERROR=1 python src/train.py experiment=cfode_train logger=wandb +logger.wandb.name=cfode_run_cardiovascular data=cardiovascular_dynamics ++model.visualize=true

HYDRA_FULL_ERROR=1 python src/train.py experiment=crn_train logger=wandb +logger.wandb.name=crn_run_cardiovascular data=cardiovascular_dynamics ++model.visualize=true

HYDRA_FULL_ERROR=1 python src/train.py experiment=ct_train logger=wandb +logger.wandb.name=ct_run_cardiovascular data=cardiovascular_dynamics ++model.visualize=true

HYDRA_FULL_ERROR=1 python src/train.py experiment=bncde_train logger=wandb +logger.wandb.name=bncde_run_cardiovascular data=cardiovascular_dynamics ++model.visualize=true
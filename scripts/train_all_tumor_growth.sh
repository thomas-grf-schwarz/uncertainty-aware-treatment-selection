#!/bin/bash

HYDRA_FULL_ERROR=1 python src/train.py experiment=cfode_train logger=wandb +logger.wandb.name=cfode_run_tumor_growth data=tumor_growth_dynamics ++model.visualize=true

HYDRA_FULL_ERROR=1 python src/train.py experiment=crn_train logger=wandb +logger.wandb.name=crn_run_tumor_growth data=tumor_growth_dynamics ++model.visualize=true

HYDRA_FULL_ERROR=1 python src/train.py experiment=ct_train logger=wandb +logger.wandb.name=ct_run_tumor_growth data=tumor_growth_dynamics ++model.visualize=true

HYDRA_FULL_ERROR=1 python src/train.py experiment=gnet_train logger=wandb +logger.wandb.name=gnet_run_tumor_growth data=tumor_growth_dynamics ++model.visualize=false
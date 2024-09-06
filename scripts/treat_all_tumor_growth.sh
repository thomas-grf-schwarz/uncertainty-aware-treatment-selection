#!/bin/bash

HYDRA_FULL_ERROR=1 python src/treat.py experiment=crn_treat logger=wandb +logger.wandb.name=crn_run_tumor_growth data=tumor_growth_dynamics ++model.visualize=false ++trainer.max_epochs=75 

HYDRA_FULL_ERROR=1 python src/treat.py experiment=cfode_treat logger=wandb +logger.wandb.name=cfode_run_tumor_growth data=tumor_growth_dynamics ++model.visualize=false ++trainer.max_epochs=75

HYDRA_FULL_ERROR=1 python src/treat.py experiment=ct_treat logger=wandb +logger.wandb.name=ct_run_tumor_growth data=tumor_growth_dynamics ++model.visualize=false ++trainer.max_epochs=75

HYDRA_FULL_ERROR=1 python src/treat.py experiment=gnet_treat logger=wandb +logger.wandb.name=gnet_run_tumor_growth data=tumor_growth_dynamics ++model.visualize=false ++trainer.max_epochs=75
#!/bin/bash

HYDRA_FULL_ERROR=1 python src/treat.py experiment=crn_treat logger=wandb +logger.wandb.name=crn_run_covid_alpha_1 data=covid_dynamics ++model.visualize=false ++trainer.max_epochs=50 

HYDRA_FULL_ERROR=1 python src/treat.py experiment=cfode_treat logger=wandb +logger.wandb.name=cfode_run_covid_alpha_1 data=covid_dynamics ++model.visualize=false ++trainer.max_epochs=50

HYDRA_FULL_ERROR=1 python src/treat.py experiment=ct_treat logger=wandb +logger.wandb.name=ct_run_covid_alpha_1 data=covid_dynamics ++model.visualize=false ++trainer.max_epochs=50
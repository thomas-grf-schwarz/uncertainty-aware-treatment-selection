#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

HYDRA_FULL_ERROR=1 python src/train.py experiment=cfode_train logger=wandb +logger.wandb.name=cfode_run_slimmer

HYDRA_FULL_ERROR=1 python src/train.py experiment=crn_train logger=wandb +logger.wandb.name=crn_run_slimmer

HYDRA_FULL_ERROR=1 python src/train.py experiment=ct_train logger=wandb +logger.wandb.name=ct_run_slimmer


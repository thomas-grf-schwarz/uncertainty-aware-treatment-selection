#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

HYDRA_FULL_ERROR=1 python src/train.py experiment=cfode_train logger=wandb +logger.wandb.name=cfode_na ++trainer.max_epochs=10 data=cardiovascular_dynamics ++data.train_dataset.n_instances=128

HYDRA_FULL_ERROR=1 python src/train.py experiment=crn_train logger=wandb +logger.wandb.name=crn_na ++trainer.max_epochs=10 data=cardiovascular_dynamics ++data.train_dataset.n_instances=128

HYDRA_FULL_ERROR=1 python src/train.py experiment=ct_train logger=wandb +logger.wandb.name=ct_na ++trainer.max_epochs=10 data=cardiovascular_dynamics ++data.train_dataset.n_instances=128

HYDRA_FULL_ERROR=1 python src/train.py experiment=bncde_train logger=wandb +logger.wandb.name=bncde_na ++trainer.max_epochs=10 data=cardiovascular_dynamics ++data.train_dataset.n_instances=128
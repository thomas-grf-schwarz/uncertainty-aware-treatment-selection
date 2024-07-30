#!/bin/bash

HYDRA_FULL_ERROR=1 python src/train.py experiment=cfode_train logger=wandb +logger.wandb.name=cfode_na ++trainer.max_epochs=10 data=covid_dynamics ++data.train_dataset.n_instances=128 ++model.visualize=true # ++model.optimizer.lr=0.0005

HYDRA_FULL_ERROR=1 python src/train.py experiment=crn_train logger=wandb +logger.wandb.name=crn_na ++trainer.max_epochs=10 data=covid_dynamics ++data.train_dataset.n_instances=128 ++model.visualize=true # ++model.optimizer.lr=0.0005

HYDRA_FULL_ERROR=1 python src/train.py experiment=ct_train logger=wandb +logger.wandb.name=ct_na ++trainer.max_epochs=10 data=covid_dynamics ++data.train_dataset.n_instances=128 ++model.visualize=true # ++model.optimizer.lr=0.00005

HYDRA_FULL_ERROR=1 python src/train.py experiment=bncde_train logger=wandb +logger.wandb.name=bncde_na ++trainer.max_epochs=10 data=covid_dynamics ++data.train_dataset.n_instances=128 # ++model.learning_rate=0.00025
#!/bin/bash

HYDRA_FULL_ERROR=1 python src/train.py experiment=cfode_train logger=wandb +logger.wandb.name=cfode_na data=tumor_growth_dynamics ++trainer.max_epochs=10 ++data.train_dataset.n_instances=128 ++model.visualize=true

HYDRA_FULL_ERROR=1 python src/train.py experiment=crn_train logger=wandb +logger.wandb.name=crn_na data=tumor_growth_dynamics ++trainer.max_epochs=10 ++data.train_dataset.n_instances=128 ++model.visualize=true

HYDRA_FULL_ERROR=1 python src/train.py experiment=ct_train logger=wandb +logger.wandb.name=ct_na data=tumor_growth_dynamics ++trainer.max_epochs=10 ++data.train_dataset.n_instances=128 ++model.visualize=true

HYDRA_FULL_ERROR=1 python src/train.py experiment=gnet_train logger=wandb +logger.wandb.name=gnet_na data=tumor_growth_dynamics ++trainer.max_epochs=10 ++data.train_dataset.n_instances=128 ++model.visualize=false
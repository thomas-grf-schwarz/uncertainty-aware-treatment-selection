#!/bin/bash

HYDRA_FULL_ERROR=1 python src/treat.py experiment=cfode_treat logger=wandb +logger.wandb.name=cfode_na data=tumor_growth_dynamics ++trainer.max_epochs=3 ++data.train_dataset.n_instances=128 ++model.visualize=true ++treat.n_replicates=2 ++treat.optimizer.n_iter=2

HYDRA_FULL_ERROR=1 python src/treat.py experiment=crn_treat logger=wandb +logger.wandb.name=crn_na data=tumor_growth_dynamics ++trainer.max_epochs=3 ++data.train_dataset.n_instances=128 ++model.visualize=true ++treat.n_replicates=2 ++treat.optimizer.n_iter=2

HYDRA_FULL_ERROR=1 python src/treat.py experiment=ct_treat logger=wandb +logger.wandb.name=ct_na data=tumor_growth_dynamics ++trainer.max_epochs=3 ++data.train_dataset.n_instances=128 ++model.visualize=true ++treat.n_replicates=2 ++treat.optimizer.n_iter=2

HYDRA_FULL_ERROR=1 python src/treat.py experiment=gnet_treat logger=wandb +logger.wandb.name=gnet_na data=tumor_growth_dynamics ++trainer.max_epochs=3 ++data.train_dataset.n_instances=128 ++model.visualize=false ++treat.n_replicates=2 ++treat.optimizer.n_iter=2


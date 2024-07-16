#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

HYDRA_FULL_ERROR=1 python src/train.py experiment=cfode_train logger=wandb +logger.wandb.name=cfode_softplus ++trainer.gradient_clip_val=0.1 ++model.visualize=true

HYDRA_FULL_ERROR=1 python src/train.py experiment=crn_train logger=wandb +logger.wandb.name=crn_instancenorm ++model.visualize=true

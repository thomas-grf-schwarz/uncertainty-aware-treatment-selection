# uncertainty-aware-treatment-selection
A framework for selecting reliable treatment through time

## Setup
```
pip install -r requirements.txt
```
## Training:

Train a model using an /experiment config:
```
python src/train.py experiment=cfode_train logger=wandb +logger.wandb.name=cfode
python src/train.py experiment=crn_train logger=wandb +logger.wandb.name=crn
python src/train.py experiment=ct_train logger=wandb +logger.wandb.name=cfode 
python src/train.py experiment=bncde_train logger=wandb +logger.wandb.name=bncde
```
Change the data or underlying dynamical system by selecting a member of the /data config group:
```
python src/train.py experiment=cfode_train logger=wandb +logger.wandb.name=cfode_cardiovascular data=cardiovascular_dynamics 
```

## Treatment selection:
To select treatments using trained models pass their paths of the corresponding checkpoints (ckpt_paths; either from the logs directory or wandb).
By default, treatment selection runs all models and combinestheir results into single figures.

Select treatments (e.g. for a CRN) using an /experiment config:
```
HYDRA_FULL_ERROR=1 python src/treat.py experiment=crn_treat logger=wandb +logger.wandb.name=crn_run_cardiovascular_alpha_1 data=cardiovascular_dynamics
```

## Configurations:
To run additional experiments, create an experiment config file.

Configs split into:
* **/data**: Here you can select the dataset and tweak parameters of tbe dataset and dynamical system. It's a config group in /data to instantiate the pytorch lightning datamodule. It contains configs for train, val and test dataset, with reference to members of dataset specific config group in data/dataset (e.g. the cardiovascular dataset). 
* **/model**: Here you can tweak / view default configs for networks, optimizers and schedulers. It's a config group model to instantiate the pytorch lightning module, including the network, optimizer and scheduler for training. One member of the config group "multi" bundles all the model specific configs (ct, crn, bncde and cfode) as one for the multitreat experiment.
* **/experiment** A top level config group to run experiments, configuring the runnable scripts in /src (**train.py**, **multitreat.py**). This config group allows to override / fill in configs and provides experiment specific configs.

Src code splits into:
* **data** contains the datamodule that bundles dataset splits (train, val, test) as one - /components contains the base classes Dynamics and DynamicsDataset and dataset specific children and treatment functions. There is still some clutter in some dataset classes
* **models** contain the pytorch lightning modules each corresponding to a different way of computing uncertainty - inheriting from a base class uncertainty module. Exception: BNCDE, the only module that was largely copied and adapted from Konstantins code as a pytorch lightning module. /components contains the networks that can be matched flexibly to uncertainty models, as init args
* **utils**: mostly functional, not object oriented. Below some configs of interest:
  * Contains treatment selection functions in **treat.py** such as the optimization loops, optimization steps, a wrapper for constraints. Supports both sgd-like optimizers and closure-style optimzers (LBFGS-like). 
  * **constraints.py**: contains functions applied to the treatments. I mostly use leaky clamp which is essentially a 2-sided ReLU to dampen growth of treatments above and below some threshold. 
  * **objective.py** and **loss.py** contain uncertainty aware treatment selection objective and HSIC

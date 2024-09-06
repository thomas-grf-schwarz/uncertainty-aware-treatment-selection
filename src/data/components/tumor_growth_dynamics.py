import numpy as np
import torch

import sys

sys.path.append('./')
from src.data.components.dynamics import Dynamics, DynamicsDataset


def damped_sin(t, initial_dose):
    
    lambda_base = 0.1  
    omega_base = 1    
    
    lambda_d = lambda_base + 0.05 * initial_dose
    omega_d = omega_base + 0.1 * initial_dose
    
    raw_output = 0.5 + 0.5 * np.exp(-lambda_d * t) * np.sin(omega_d * t)

    return 1 / (1 + np.exp(-10 * (raw_output - 0.5))) # sigmoid transformation for [0,1] values


class TumorGrowthDynamics(Dynamics):
    
    """
        Tumor growth chemotherapy dataset from https://pubmed.ncbi.nlm.nih.gov/22761472/
        The tumor is composed of proliferative (P: outcome) and nonproliferative quiescent tissue (Q) 
        Q_P: damaged quiescent cells
        C: PCV concentration in Plasma. Treatment representing the concentration of a virtual drug encompassing the 3 chemotherapeutic components of the PCV regimen.

        k_PQ: governing transition of proliferative tissue into quiescence
        lambda_P, lambda_Q: damages in proliferative and quiescent tissues respectively
        K: maximal tumor size
        delta_Q:  rate constant for elimination of the damaged quiescent tissue
        k_QP: rate constant for transfer from damaged quiescent tissue to proliferative tissue
    """
     
    def __init__(self, params):
        self.params = params

    def dxdt(self, x, t, intervention, dose):

        KDE = self.params['KDE']
        k_QP = self.params['k_QP']
        k_PQ = self.params['k_PQ']
        lambda_P = self.params['lambda_P']
        lambda_Q = self.params['lambda_Q']
        delta_Q = self.params['delta_Q']
        K = self.params['K']

        # State variables: C, P, Q, Q_P
        C, P, Q, Q_P = x
        
        dose = np.random.normal(1.0, 0.3)
        C = intervention(t, dose) # intervention on C - continuous value [0,1] over time 
        
        # Sum of all cellular compartments
        P_star = P + Q + Q_P
        
        # Drug concentration kinetics
        dCdt = -KDE * C
        
        # Dynamics of proliferative cells
        dPdt = lambda_P * P * (1 - (P_star / K)) + k_QP * Q_P - k_PQ * P - lambda_P * C * KDE * P
        
        # Dynamics of quiescent cells
        dQdt = k_PQ * P - lambda_Q * C * KDE * Q
        
        # Dynamics of damaged quiescent cells
        dQ_Pdt = lambda_Q * C * KDE * Q - k_QP * Q_P - delta_Q * Q_P
        
        return np.array([dCdt, dPdt, dQdt, dQ_Pdt])

    def get_initial_condition(self):
        # Initial conditions based on the estimates provided in the paper https://pubmed.ncbi.nlm.nih.gov/22761472/
        initial_C = 1.0  # Initial concentration of PCV - change to [0,1] sample
        initial_P = np.random.normal(1.45, 1.65)  # Mean and std dev for P
        initial_Q = np.random.normal(41.7, 22.70)  # Mean and std dev for Q
        initial_Q_P = 0.0  # Initial damaged quiescent cells

        initial_state = np.array([initial_C, initial_P, initial_Q, initial_Q_P])

        initial_condition = {
            'initial_state': initial_state, # includes C
            'initial_dose': initial_C, 
            'sampled_params': {}
        }

        return initial_condition


class TumorGrowthDataset(DynamicsDataset):
    def simulate_outcome(self, initial_state, treatment_dose, t, intervention): 
        outcomes = self.simulate_step(
            initial_state=initial_state, 
            treatment_dose=treatment_dose, 
            t=t, 
            intervention=intervention
            ).T
        return outcomes[1]  

    def to_state(self, outcome, covariate):

        C = covariate[:, 0].squeeze()
        P = outcome.squeeze()
        Q = covariate[:, 1].squeeze()
        Q_P = covariate[:, 2].squeeze()

        state = torch.stack([C, P, Q, Q_P])  
        return self.denormalize(state)

    def __getitem__(self, idx):
        
        if self.simulate_online:
            initial_condition = self.initial_conditions[idx]
            self.dynamics.params.update(initial_condition['sampled_params'])

            initial_state = initial_condition['initial_state']
            initial_dose = initial_condition['initial_dose']
            print('initial dose:', initial_dose)
            print('initial state:', initial_state)

            state = self.simulate_with_confounding(initial_state, initial_dose, self.t, self.intervention)
        else:
            *state, initial_condition = self.data[idx]
            state = np.stack(state, axis=0)

        instance = {
            'outcome_history': state[1, :-self.t_horizon, None],  # P history
            'treatment_history': state[0, :-self.t_horizon, None],  # treatments
            'covariate_history': np.concatenate(
                    [
                        state[None, 0, :-self.t_horizon],
                        state[None, 2, :-self.t_horizon],
                        state[None, 3, :-self.t_horizon],
                    ]
                ).T,  # C, Q and Q_P history
            'outcomes': state[1, -self.t_horizon:, None],  # P outcomes
            'treatments': state[0, -self.t_horizon:, None]  # treatments
        }

        for k, data in instance.items():
            instance[k] = torch.tensor(data, dtype=torch.float32)

        instance['initial_state'] = initial_condition['initial_state']

        return instance
    
    
if __name__ == "__main__":

    params = { # sample parameters for different patients
        'KDE': 0.1,       # Drug elimination constant
        'k_QP': 0.05,     # Rate from damaged quiescent to proliferative
        'k_PQ': 0.03,     # Rate from proliferative to quiescent
        'lambda_P': 0.1,  # Damage to proliferative cells
        'lambda_Q': 0.05, # Damage to quiescent cells
        'delta_Q': 0.01,  # Elimination rate of damaged quiescent cells
        'K': 50           # Maximal tumor size
    }

    dynamics = TumorGrowthDynamics(params=params)

    n_instances = 32  
    t_horizon = 10   
    t_end = 20

    dataset = TumorGrowthDataset(
        dynamics=dynamics,
        n_instances=n_instances,
        intervention=damped_sin,
        t_horizon=t_horizon,
        t_end=t_end,
        covariate_size=3,
        treatment_size=1,
        outcome_size=1,
        simulate_online=False
    )

    dataset.visualize_stats()
    import pdb; pdb.set_trace()
    dataset.visualize_trajectories()
    import pdb; pdb.set_trace()
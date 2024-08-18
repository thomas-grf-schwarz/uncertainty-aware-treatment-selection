import numpy as np
import torch
from src.data.components.dynamics import Dynamics, DynamicsDataset

import numpy as np
import torch

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

    def dxdt(self, x, t, intervention):

        KDE = self.params['KDE']
        k_QP = self.params['k_QP']
        k_PQ = self.params['k_PQ']
        lambda_P = self.params['lambda_P']
        lambda_Q = self.params['lambda_Q']
        delta_Q = self.params['delta_Q']
        K = self.params['K']

        # State variables: C, P, Q, Q_P
        C, P, Q, Q_P = x
        
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
        initial_C = 1.0  # Initial concentration of PCV
        initial_P = np.random.normal(1.45, 1.65)  # Mean and std dev for P
        initial_Q = np.random.normal(41.7, 22.70)  # Mean and std dev for Q
        initial_Q_P = 0.0  # Initial damaged quiescent cells

        initial_state = np.array([initial_C, initial_P, initial_Q, initial_Q_P])

        initial_condition = {
            'initial_state': initial_state,
            'initial_dose': initial_C,
            'sampled_params': {}
        }

        return initial_condition

class TumorGrowthDataset(DynamicsDataset):
    def simulate_outcome(self, initial_state, t, intervention):
        outcomes = self.simulate_step(initial_state, t, intervention).T
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
            outcomes = self.simulate_with_confounding(initial_state, initial_dose, self.t, self.intervention)
        else:
            outcomes, initial_condition = self.data[idx]
            outcomes = np.stack(outcomes, axis=0)

        instance = {
            'outcome_history': outcomes[1, :-self.t_horizon, None],  # P history
            'treatment_history': outcomes[0, :-self.t_horizon, None],  # C history
            'covariate_history': outcomes[2:-1, :-self.t_horizon].T,  # Q and Q_P history
            'outcomes': outcomes[1, -self.t_horizon:, None],  # P outcomes
            'treatments': outcomes[0, -self.t_horizon:, None]  # C treatments
        }

        for k, data in instance.items():
            instance[k] = torch.tensor(data, dtype=torch.float32)

        instance['initial_state'] = initial_condition['initial_state']

        return instance
    
    
if __name__ == "__main__":

    params = {
        'KDE': 0.1,       # Drug elimination constant
        'k_QP': 0.05,     # Rate from damaged quiescent to proliferative
        'k_PQ': 0.03,     # Rate from proliferative to quiescent
        'lambda_P': 0.1,  # Damage to proliferative cells
        'lambda_Q': 0.05, # Damage to quiescent cells
        'delta_Q': 0.01,  # Elimination rate of damaged quiescent cells
        'K': 50           # Maximal tumor size
    }

    dynamics = TumorGrowthDynamics(params=params)

    n_instances = 100  
    t_horizon = 10     
    t_end = 50     

    def exponential_decay(t, initial_dose=1.0):
        half_life = 10  
        return initial_dose * np.exp(-np.log(2) * t / half_life)

    dataset = TumorGrowthDataset(
        dynamics=dynamics,
        n_instances=n_instances,
        intervention=exponential_decay,
        t_horizon=t_horizon,
        t_end=t_end,
        simulate_online=False
    )
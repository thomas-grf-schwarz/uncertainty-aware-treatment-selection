import numpy as np
import scipy.integrate
import torch
import scipy
from src.data.components.dynamics import Dynamics, DynamicsDataset


def fluids_input(t):
    return 5*np.exp(-((t-5)/2)**2)


def dosed_fluids_input(t, dose):
    return dose*5*np.exp(-((t-5)/2)**2)


def delayed_fluids_input(t, delay):
    return 5*np.exp(-((t-5-delay)/2)**2)


def sample_with_confounding(d_w, alpha=1.0):
    beta = (alpha - 1) / d_w + 2 - alpha
    return scipy.stats.beta.rvs(alpha, beta, size=1)[0]


def v_fun(x):
    return 0.02*(np.cos(5*x-0.2) * (5-x)**2)**2


def sigmoid(x):
    return 1/(1 + np.exp(-x))


class CovidDynamics(Dynamics):

    def __init__(self, params):
        self.params = params
   
    def dxdt(self, x, t, intervention, dose):

        # adapted from https://github.com/ZhaozhiQIAN/Hybrid-ODE-NeurIPS-2021/blob/main/dataloader.py

        hill_cure = self.params["hill_cure"]
        hill_patho = self.params["hill_patho"]
        ec50_patho = self.params["ec50_patho"]
        emax_patho = self.params["emax_patho"]
        k_dexa = self.params["k_dexa"]
        k_discure_immunereact = self.params["k_discure_immunereact"]
        k_discure_immunity = self.params["k_discure_immunity"]
        k_disprog = self.params["k_disprog"]
        k_immune_disease = self.params["k_immune_disease"]
        k_immune_feedback = self.params["k_immune_feedback"]
        k_immune_off = self.params["k_immune_off"]
        k_immunity = self.params["k_immunity"]
        kel = self.params["kel"]

        # Unpack x
        disease = x[0]
        immune_react = x[1]
        immunity = x[2]
        dose2 = x[3]

        # Define the dose at time t
        additive_term = intervention(t, dose)

        # Define the derivatives
        dxdt0 = (
            disease * k_disprog
            - disease * immunity ** hill_cure * k_discure_immunity
            - disease * immune_react * k_discure_immunereact
        )

        dxdt1 = (
            disease * k_immune_disease
            - immune_react * k_immune_off
            + disease * immune_react * k_immune_feedback
            + (immune_react ** hill_patho * emax_patho) / (ec50_patho ** hill_patho + immune_react ** hill_patho)
            - dose2 * immune_react * k_dexa
        )

        dxdt2 = immune_react * k_immunity

        dxdt3 = kel * additive_term - kel * dose2

        return np.array([dxdt0, dxdt1, dxdt2, dxdt3])

    def get_initial_condition(self):

        x0 = np.random.exponential(scale=0.01)
        x1 = np.random.exponential(scale=0.01)
        x2 = np.random.exponential(scale=0.01)
        x3 = np.random.exponential(scale=0.01)

        init_state = np.stack((x0, x1, x2, x3))
        initial_dose = 2 + 0.5 * np.random.randn()

        initial_condition = {
            'initial_state': init_state,
            'initial_dose': initial_dose,
            'sampled_params': {}
        }

        return initial_condition


class CovidDataset(DynamicsDataset):

    def simulate_outcome(self, initial_state, treatment_dose, t, intervention):
        out = self.simulate_step(initial_state, treatment_dose, t, intervention).T
        return out[0]

    def to_state(self, outcome, covariate):

        x1 = outcome.squeeze()
        x2 = covariate[:, 0].squeeze()
        x3 = covariate[:, 1].squeeze()
        x4 = covariate[:, 2].squeeze()

        state = torch.stack([x1, x2, x3, x4]) # order matters

        return self.denormalize(state)

    def __getitem__(self, idx):

        if self.simulate_online:

            initial_condition = self.initial_conditions[idx]
            self.dynamics.params.update(initial_condition['sampled_params'])

            initial_state = initial_condition['initial_state']
            initial_dose = initial_condition['initial_dose']
            out = self.simulate_with_confounding(
                initial_state=initial_state,
                initial_dose=initial_dose,
                t=self.t,
                intervention=self.intervention,
                )
        else:
            *out, initial_condition = self.data[idx]
            out = np.stack(out, axis=0)

        instance = {}

        instance['outcome_history'] = out[0, :-self.t_horizon, None]
        instance['treatment_history'] = out[-1, :-self.t_horizon, None]
        instance['covariate_history'] = out[1:-1, :-self.t_horizon].T
        instance['outcomes'] = out[0, -self.t_horizon:, None]
        instance['treatments'] = out[-1, -self.t_horizon:, None]

        for k, data in instance.items():
            instance[k] = torch.tensor(data, dtype=torch.float32)

        instance['initial_state'] = initial_condition['initial_state']

        # L, C or C, L

        return instance


if __name__ == "__main__":
    params = {
        "k_IR": 0.2,
        "k_PF": 0.2,
        "k_O": 1.0,
        "E_max": 1.0,
        "E_C": 1.0,
        "k_DP": 4.0,
        "k_IIR": 0.1,
        "k_DC": 0.1,
        "h_P": 2,
        "k_1": 1,
        "k_2": 1,
        "k_3": 1,
        "h_C": 8
    }

    dynamics = CovidDynamics(
        params=params
    )

    dataset = CovidDataset(
        dynamics=dynamics,
        n_instances=32,
        intervention=dosed_fluids_input,
        t_horizon=20,
        t_end=40,
        simulate_online=False,
        noise_st=0.01
        )
    
    import pdb; pdb.set_trace()

    dataset.visualize_stats()
    dataset.visualize_trajectories()
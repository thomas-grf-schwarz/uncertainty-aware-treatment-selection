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

        # Parameters:
        k_IR = self.params["k_IR"]
        k_PF = self.params["k_PF"]
        k_O = self.params["k_O"]
        E_max = self.params["E_max"]
        E_C = self.params["E_C"]
        k_Dex = self.params["k_Dex"]
        k_DP = self.params["k_DP"]
        k_IIR = self.params["k_IIR"]
        k_DC = self.params["k_DC"]
        h_P = self.params["h_P"]
        k_1 = self.params["k_1"]
        k_2 = self.params["k_2"]
        k_3 = self.params["k_3"]
        h_C = self.params["h_C"]

        additive_term = intervention(t, dose)
        dx = np.empty_like(x)
        dx[0] = k_IR * x[3] + k_PF * x[3] * x[0] - k_O * x[0] \
            + (E_max * (x[0]**h_P)) / (E_C + (x[0]**h_P)) - k_Dex * x[0] * x[1]
        dx[1] = -k_2*x[1] + k_3*x[2]
        dx[2] = -k_3 * x[2] + additive_term
        dx[3] = k_DP * x[3] - k_IIR * x[3] * x[0] - k_DC * x[3] * (x[4]**h_C)
        dx[4] = k_1 * x[0]
        # if (dx > 100).any():
        #     print(dx)
        #     print("========================")
        #     import pdb; pdb.set_trace()
        return dx

    def get_initial_condition(self):

        x0 = np.random.exponential(10 / 2)
        x1 = np.random.exponential(1/100)
        x2 = np.random.exponential(1/100)
        x3 = np.random.exponential(10 / 2)
        x4 = np.random.exponential(1)

        init_state = np.stack((x0, x1, x2, x3, x4))
        initial_dose = 2 + 0.5 * np.random.randn()
        k_Dex = 1 + 15*np.random.rand()

        initial_condition = {
            'initial_state': init_state,
            'initial_dose': initial_dose,
            'sampled_params': {
                'k_Dex': k_Dex,
            }
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
        x5 = covariate[:, 3].squeeze()

        state = torch.stack([x1, x2, x3, x4, x4, x5]) # order matters

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
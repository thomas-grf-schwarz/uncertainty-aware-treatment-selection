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


def ut(t, dose, phi, delta):
    return dose * np.sin(phi * t) * np.exp(delta * t)


def sigmoid(x):
    return 1/(1 + np.exp(-x))


class PendulumDynamics(Dynamics):

    def __init__(self, params):
        self.params = params

    def dxdt(self, x, t, intervention, dose):

        # Parameters:
        g = self.params["g"]
        l = self.params["l"]

        dx = np.empty_like(x)

        u = intervention(t, dose)
        dx[0] = x[1]
        dx[1] = (g / l) * (1 + u) * np.sin(x[0])
        
        return dx

    def get_initial_condition(self):

        x0 = np.random.rand() + 0.5  # theta
        x1 = 0  # velocity

        init_state = np.stack((x0, x1))
        initial_dose = 10 + np.random.randn()
        l = np.random.rand() * 4 + 0.5

        initial_condition = {
            'initial_state': init_state,
            'initial_dose': initial_dose,
            'sampled_params': {
                'l': l,
            }
        }

        return initial_condition


class PendulumDataset(DynamicsDataset):

    def simulate_outcome(self, initial_state, treatment_dose, t, intervention):
        out = self.simulate_step(initial_state, treatment_dose, t, intervention).T
        return out[0]

    def to_state(self, outcome, covariate):

        x0 = outcome.squeeze()
        x1 = covariate[:, 0].squeeze()

        state = torch.stack([x0, x1]) # order matters

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


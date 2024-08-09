import numpy as np
import torch
from src.data.components.dynamics import Dynamics, DynamicsDataset


def fluids_input(t):
    return 5*np.exp(-((t-5)/2)**2)


def dosed_fluids_input(t, dose):
    return dose*5*np.exp(-((t-5)/2)**2)


def delayed_fluids_input(t, delay):
    return 5*np.exp(-((t-5-delay)/2)**2)


def v_fun(x):
    return 0.02*(np.cos(5*x-0.2) * (5-x)**2)**2


def sigmoid(x):
    return 1/(1 + np.exp(-x))


class CovidDynamics(Dynamics):

    """
    A concrete implementation of the Dynamics class for simulating the 
    dynamics of the immune system challenged by a covid infection
    Adapted from https://github.com/ZhaozhiQIAN/Hybrid-ODE-NeurIPS-2021/blob/main/dataloader.py

    Attributes
    ----------
    params : dict
        A dictionary containing model parameters for the immune system.

    Methods
    -------
    dxdt(self, x, t, intervention, dose):
        Calculate the time derivative of the immune system state vector at 
        time `t`.

    get_initial_condition(self):
        Generate and return the initial condition for the immune 
        system's state.
    """
    def __init__(self, params):
        self.params = params
   
    def dxdt(self, x, t, intervention, dose):

        """
        Calculate the time derivative of the immune system state vector `x` 
        at time `t`.

        Parameters
        ----------
        x : array-like
            The current state of the immune system, including disease state,
            innate immune reaction, immunity and dexamethasone
        t : float
            The current time.
        intervention : callable
            A function representing an external intervention applied to the 
            system, i.e. dexamethasone
        dose : float
            The dose associated with the intervention.

        Returns
        -------
        np.ndarray
            The time derivative of the immune system state vector.
        """

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

        """
        Generate and return the initial condition for the immune 
        system's state.

        The initial condition includes randomly sampled initial values for the
        disease state, innate immune reaction, immunity and dexamethasone       
        as well as an initial dose for the intervention.

        Returns
        -------
        dict
            A dictionary containing:
            - 'initial_state': np.ndarray of initial state variables (disease 
            state, innate immune reaction, immunity and dexamethasone).
            - 'initial_dose': float representing the initial dose.
            - 'sampled_params': dict of any additional sampled parameters 
              (currently empty).
        """

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

        """
        Simulate the disease state outcome for the 
        immune system.

        This method simulates the immune system dynamics using the given 
        initial state, treatment dose, and intervention over the time 
        steps `t`. The disease state is extracted and 
        returned as the outcome of interest.

        Parameters
        ----------
        initial_state : array-like
            The initial state of the immune system
        treatment_dose : float
            The treatment dose applied to the system during the simulation.
        t : array-like
            The time steps over which to simulate the system.
        intervention : callable
            A function representing the intervention applied to the system 
            during the simulation.

        Returns
        -------
        np.ndarray
            The simulated disease state over the time steps `t`.
        """

        out = self.simulate_step(initial_state, treatment_dose, t, intervention).T
        return out[0]

    def to_state(self, outcome, covariate):

        """
        Convert the outcome and covariate arrays into a state tensor for the 
        immune system.

        This method combines the outcome (disease state) with the 
        covariates (innate immune reaction, immunity and  dexamethasone) to 
        form a state tensor representing the immune system. The resulting 
        state tensor is denormalized before returning.

        Parameters
        ----------
        outcome : array-like
            The outcome array
        covariate : array-like
            The covariate array

        Returns
        -------
        torch.Tensor
            The state tensor representing the immune system, ordered in the 
            same way as the state in the corresponding dynamics class
        """

        x1 = outcome.squeeze()
        x2 = covariate[:, 0].squeeze()
        x3 = covariate[:, 1].squeeze()
        x4 = covariate[:, 2].squeeze()

        state = torch.stack([x1, x2, x3, x4]) # order matters

        return self.denormalize(state)

    def __getitem__(self, idx):

        """
        Retrieve the covid dataset instance at the given index.

        This method returns a dictionary containing the history of outcomes, 
        treatments, covariates, and the initial state for the immune 
        system at the specified index. If `simulate_online` is True, the data 
        is simulated on-the-fly using the stored initial conditions; 
        otherwise, it is retrieved from precomputed data.

        The returned dictionary contains the following keys:
        
        - 'outcome_history': torch.Tensor
            The history of the disease state up until the 
            prediction horizon.
        - 'treatment_history': torch.Tensor
            The history of the treatment doses applied before the prediction 
            horizon.
        - 'covariate_history': torch.Tensor
            The history of covariates (innate immune reaction, immunity and 
            dexamethasone) up until the prediction horizon.
        - 'outcomes': torch.Tensor
            The disease state over the prediction horizon.
        - 'treatments': torch.Tensor
            The treatment doses applied over the prediction horizon.
        - 'initial_state': torch.Tensor
            The initial state of the immune system at the start of the 
            simulation.

        Parameters
        ----------
        idx : int
            The index of the desired dataset instance.

        Returns
        -------
        dict
            A dictionary containing the specified keys with corresponding data.
        """

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
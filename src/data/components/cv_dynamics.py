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


class CardioVascularDynamics(Dynamics):

    """
    A concrete implementation of the Dynamics class for simulating dynamics of
    arterial and venous blood pressure.
    Adapted from https://github.com/microsoft/cf-ode/blob/main/causalode/cv_data_utils.py   

    Attributes
    ----------
    params : dict
        A dictionary containing model parameters for the cardiovascular system.

    Methods
    -------
    dxdt(self, x, t, intervention, dose):
        Calculate the time derivative of the cardiovascular state vector at 
        time `t`.

    get_initial_condition(self):
        Generate and return the initial condition for the cardiovascular 
        system's state.
    """

    def __init__(self, params):
        self.params = params

    def dxdt(self, x, t, intervention, dose):

        """
        Calculate the time derivative of the cardiovascular state vector `x` 
        at time `t`.

        Parameters
        ----------
        x : array-like
            The current state of the cardiovascular system, including arterial 
            pressure, venous pressure, autonomic baroreflex tone and cardiac 
            stroke volume
        t : float
            The current time.
        intervention : callable
            A function representing an external intervention applied to the 
            system, i.e. fluid
        dose : float
            The dose associated with the intervention.

        Returns
        -------
        np.ndarray
            The time derivative of the cardiovascular state vector.
        """

        # Parameters:
        f_hr_max = self.params["f_hr_max"]
        f_hr_min = self.params["f_hr_min"]
        r_tpr_max = self.params["r_tpr_max"]
        r_tpr_min = self.params["r_tpr_min"]
        ca = self.params["ca"]
        cv = self.params["cv"]
        k_width = self.params["k_width"]
        p_aset = self.params["p_aset"]
        tau = self.params["tau"]
        
        # Treatment:
        i_ext = intervention(t, dose)
        
        # Unknown parameters
        r_tpr_mod = self.params["r_tpr_mod"]
        sv_mod = self.params["sv_mod"]

        # State variables
        p_a = 100. * x[0]
        p_v = 10. * x[1]
        s = x[2]
        sv = 100. * x[3]

        # Building f_hr and r_tpr:
        f_hr = s * (f_hr_max - f_hr_min) + f_hr_min
        r_tpr = s * (r_tpr_max - r_tpr_min) + r_tpr_min - r_tpr_mod

        # Building dp_a/dt and dp_v/dt:
        dva_dt = -1. * (p_a - p_v) / r_tpr + sv * f_hr
        dvv_dt = -1. * dva_dt + i_ext
        dpa_dt = dva_dt / (ca * 100.)
        dpv_dt = dvv_dt / (cv * 10.)

        # Building dS/dt:
        ds_dt = (1. / tau) * (1. - 1. / (1 + np.exp(-1 * k_width * (p_a - p_aset))) - s)
        dsv_dt = i_ext * sv_mod

        # State derivative
        return np.array([dpa_dt, dpv_dt, ds_dt, dsv_dt])

    def get_initial_condition(self):

        """
        Generate and return the initial condition for the cardiovascular 
        system's state.

        The initial condition includes randomly sampled initial values for 
        arterial pressure, venous pressure, autonomic baroreflex tone and 
        cardiac stroke volume, as well as an initial dose for the intervention.

        Returns
        -------
        dict
            A dictionary containing:
            - 'initial_state': np.ndarray of initial state variables (arterial 
              pressure, venous pressure, sympathetic activity, stroke volume).
            - 'initial_dose': float representing the initial dose.
            - 'sampled_params': dict of any additional sampled parameters 
              (currently empty).
        """

        max_sv = 1.0
        min_sv = 0.9

        max_pa = 85.0
        min_pa = 75.0

        max_pv = 7.0
        min_pv = 3.0

        max_s = 0.25
        min_s = 0.15

        initial_sv = (np.random.rand() * (max_sv - min_sv) + min_sv)
        initial_pa = (np.random.rand() * (max_pa - min_pa) + min_pa) / 100.0
        initial_pv = (np.random.rand() * (max_pv - min_pv) + min_pv) / 10.0
        initial_s = (np.random.rand() * (max_s - min_s) + min_s)

        initial_dose = np.random.rand()

        initial_state = np.array([
            initial_pa, 
            initial_pv, 
            initial_s, 
            initial_sv]
            )

        initial_condition = {
            'initial_state': initial_state,
            'initial_dose': initial_dose,
            'sampled_params': {}
        }

        return initial_condition


class CardioVascularDataset(DynamicsDataset):
    
    def simulate_outcome(self, initial_state, treatment_dose, t, intervention):

        """
        Simulate the pulmonary venous pressure (pv) outcome for the 
        cardiovascular system.

        This method simulates the cardiovascular dynamics using the given 
        initial state, treatment dose, and intervention over the time 
        steps `t`. The pulmonary venous pressure (pv) is extracted and 
        returned as the outcome of interest.

        Parameters
        ----------
        initial_state : array-like
            The initial state of the cardiovascular system
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
            The simulated pulmonary venous pressure (pv) over the time steps `t`.
        """

        pa, pv, s, sv = self.simulate_step(initial_state, treatment_dose, t, intervention).T
        
        return pv
   
    def to_state(self, outcome, covariate):

        """
        Convert the outcome and covariate arrays into a state tensor for the 
        cardiovascular system.

        This method combines the outcome (pulmonary venous pressure) with the 
        covariates (arterial pressure, autonomic baroreflex tone, and stroke 
        volume) to form a state tensor representing the cardiovascular system.
        The resulting state tensor is denormalized before returning.

        Parameters
        ----------
        outcome : array-like
            The outcome array, typically representing the pulmonary venous 
            pressure (pv).
        covariate : array-like
            The covariate array, typically containing arterial pressure (pa), 
            autonomic baroreflex tone (s), and stroke volume (sv).

        Returns
        -------
        torch.Tensor
            The state tensor representing the cardiovascular system, 
            with the order [pa, pv, s, sv].
        """

        pa = covariate[:, 0].squeeze()
        pv = outcome.squeeze()
        s = covariate[:, 1].squeeze()
        sv = covariate[:, 2].squeeze()

        state = torch.stack([pa, pv, s, sv]) # order matters

        return self.denormalize(state)

    def __getitem__(self, idx):

        """
        Retrieve the cardiovascular dataset instance at the given index.

        This method returns a dictionary containing the history of outcomes, 
        treatments, covariates, and the initial state for the cardiovascular 
        system at the specified index. If `simulate_online` is True, the data 
        is simulated on-the-fly using the stored initial conditions; 
        otherwise, it is retrieved from precomputed data.

        The returned dictionary contains the following keys:
        
        - 'outcome_history': torch.Tensor
            The history of the pulmonary venous pressure (pv) up until the 
            prediction horizon.
        - 'treatment_history': torch.Tensor
            The history of the treatment doses applied before the prediction 
            horizon.
        - 'covariate_history': torch.Tensor
            The history of covariates (arterial pressure, autonomic baroreflex 
            tone, stroke volume) up until the prediction horizon.
        - 'outcomes': torch.Tensor
            The pulmonary venous pressure (pv) over the prediction horizon.
        - 'treatments': torch.Tensor
            The treatment doses applied over the prediction horizon.
        - 'initial_state': torch.Tensor
            The initial state of the cardiovascular system at the start of the 
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
            pa, pv, s, sv, treatments = self.simulate_with_confounding(
                initial_state=initial_state,
                initial_dose=initial_dose,
                t=self.t,
                intervention=self.intervention,
                )
        else:
            pa, pv, s, sv, treatments, initial_condition = self.data[idx]

        instance = {}

        instance['outcome_history'] = pv[:-self.t_horizon, None]
        instance['treatment_history'] = treatments[:-self.t_horizon, None]
        instance['covariate_history'] = np.stack(
            [pa[:-self.t_horizon], s[:-self.t_horizon], sv[:-self.t_horizon]],
             axis=-1
             )
        instance['outcomes'] = pv[-self.t_horizon:, None]
        instance['treatments'] = treatments[-self.t_horizon:, None]
        instance['initial_state'] = initial_condition['initial_state']

        for k, data in instance.items():
            instance[k] = torch.tensor(data, dtype=torch.float32)

        return instance


if __name__ == "__main__":
    params = {
        "r_tpr_mod": 0.,
        "f_hr_max": 3.0,
        "f_hr_min": 2.0 / 3.0,
        "r_tpr_max": 2.134,
        "r_tpr_min": 0.5335,
        "sv_mod": 0.001,
        "ca": 4.0,
        "cv": 111.0,
        "k_width": 0.1838,
        "p_aset": 70,
        "tau": 20,
        "p_0lv": 2.03,
        "r_valve": 0.0025,
        "k_elv": 0.066,
        "v_ed0": 7.14,
        "T_sys": 4. / 15.,
        "cprsw_max": 103.8,
        "cprsw_min": 25.9,
        }

    dynamics = CardioVascularDynamics(
        params=params
    )

    dataset = CardioVascularDataset(
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
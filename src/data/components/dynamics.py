import numpy as np
import scipy.integrate
import scipy
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import torch


class Dynamics:

    """
    A base class for modeling dynamical systems. This class defines the interface 
    for calculating the time derivative of the state (`dxdt`) and obtaining 
    the initial condition of the system.

    Methods
    -------
    dxdt(self, x, t, params, intervention, dose):
        Calculate the time derivative of the state vector at a given time `t`.
        This method should be implemented by subclasses.

    get_initial_condition(self):
        Generate and return the initial condition for the system's state.
        This method should be implemented by subclasses.
    """

    def dxdt(self, x, t, params, intervention, dose):
    
        """
        Calculate the time derivative of the state vector `x` at time `t`.

        Parameters
        ----------
        x : array-like
            The current state of the system.
        t : float
            The current time.
        params : dict
            A dictionary of parameters for the system.
        intervention : callable
            A function of time representing an intervention applied to 
            the system.
        dose : float
            The dose associated with the intervention.

        Returns
        -------
        array-like
            The time derivative of the state vector.
        """

        raise NotImplementedError
    
    def get_initial_condition(self):

        """
        Generate and return the initial condition of the system.

        Returns
        -------
        dict
            A dictionary containing the following keys: initial_state, 
            initial_dose, and sampled_params
        """

        raise NotImplementedError


class DynamicsDataset(Dataset):

    def __init__(
            self,
            dynamics,
            n_instances,
            intervention,
            t_horizon,
            t_end,
            covariate_size,
            treatment_size,
            outcome_size,
            simulate_online=True,
            noise_st=0.1,
            do_normalize=False,
            alpha=1.0,
            ):

        self.dynamics = dynamics
        self.intervention = intervention
        self.initial_conditions = [
            self.dynamics.get_initial_condition()
            for _ in range(n_instances)
            ]

        self.t_horizon = t_horizon
        self.t_end = t_end
        self.t = np.arange(t_end + t_horizon)

        self.covariate_size = covariate_size
        self.treatment_size = treatment_size
        self.outcome_size = outcome_size

        self.simulate_online = simulate_online
        self.noise_std = noise_st
        self.alpha = alpha

        if self.simulate_online:
            assert not do_normalize, """normalize not supported 
                                        for simulate_online"""

        else:
            self.data = []

            for initial_condition in self.initial_conditions:
                
                self.dynamics.params.update(initial_condition['sampled_params'])
                history = self.simulate_with_confounding(
                    initial_state=initial_condition['initial_state'],
                    initial_dose=initial_condition['initial_dose'],
                    t=self.t,
                    intervention=self.intervention)
                
                self.data.append((*history, initial_condition))
            if do_normalize:
                self.normalize()

        # import pdb; pdb.set_trace()

    def normalize(self):

        means = []
        stds = []
        for instance in self.data:
            
            initial_condition = instance[-1]
            history = instance[:-1]

            trajectory_means = []
            trajectory_stds = []
            
            for i, trajectory in enumerate(history):
                
                trajectory_means.append(trajectory.mean())
                trajectory_stds.append(trajectory.std())
            
            means.append(trajectory_means)
            stds.append(trajectory_stds)

        self.state_means = np.array(means).mean(0)
        self.state_stds = np.array(stds).mean(0)

        for i in range(len(self.data)):
            normalized_state = []

            initial_condition = self.data[i][-1]
            history = self.data[i][:-1]  # exclude the initial condition
            for j, trajectory in enumerate(history):
                normalized_state.append(
                    (trajectory - self.state_means[j]) / self.state_stds[j]
                )
            normalized_state.append(initial_condition)
            self.data[i] = normalized_state
    
    def denormalize(self, state):

        if hasattr(self, 'state_means'):
            denormalized_state = []
            for i, trajectory in enumerate(state):
                denormalized_state.append(
                    trajectory * self.state_stds[i] + self.state_means[i]
                )
            return denormalized_state
        else:
            return state

    def sample_with_confounding(self, d_w):
        beta = (self.alpha - 1) / d_w + 2 - self.alpha
        return scipy.stats.beta.rvs(self.alpha, beta, size=1)[0]

    def simulate_with_confounding(
            self,
            initial_state,
            initial_dose,
            t,
            intervention,
            n_treatments=5
            ):

        """
        Args:
            initial_state (array-like): The initial state of the system.
            initial_dose (float): The initial treatment dose.
            t (array-like): A sequence of time points over which the 
                simulation runs.
            intervention (callable): A function that takes time and dose as 
                inputs 
                and returns the effect of the intervention at each time point.
            n_treatments (int, optional): The number of treatment cycles into 
                which the time sequence `t` is split. Defaults to 5.

        Returns:
            np.ndarray: A 2D array where each row corresponds to a time step 
            and each column corresponds to either the system state or the 
            applied treatment over time. The final history also includes added 
            noise.
        """

        history = []
        treatment_dose = initial_dose
        for t_split in np.array_split(t, n_treatments):

            treatment_dose = self.sample_with_confounding(treatment_dose)

            t_reset = t_split - t_split.min()

            state = self.simulate_step(
                initial_state=initial_state,
                treatment_dose=treatment_dose,
                t=t_reset,
                intervention=intervention,
                )

            treatments = np.array([
                [intervention(t_step, treatment_dose)] for t_step in t_reset]
                )

            history.append(
                np.concatenate([state, treatments], axis=-1)
            )

        history = np.concatenate(history, axis=0).T
        return history + self.noise_std * np.random.randn(*history.shape)

    def simulate_outcome(
            self,
            initial_state,
            treatment_dose,
            t,
            intervention
            ):

        """
        Simulate the outcome of a dynamical system given an initial state, 
        treatment dose, and intervention.

        This method should be implemented by subclasses to return the outcome 
        variable among the state variables of the system.

        Parameters
        ----------
        initial_state : array-like
            The initial state of the system, typically including state variables relevant to the dynamics.
        treatment_dose : float
            The treatment dose applied to the system during the simulation.
        t : array-like
            The time steps over which to simulate the system.
        intervention : callable
            A function representing the intervention applied to the system during the simulation.

        Returns
        -------
        np.ndarray
            The simulated outcome over the time steps `t`, specific to the system being modeled.
        """    
    
        raise NotImplementedError

    def simulate_step(
            self,
            initial_state,
            treatment_dose,
            t,
            intervention
            ):
        args = (intervention, treatment_dose)
        out = scipy.integrate.odeint(
            func=self.dynamics.dxdt, 
            y0=initial_state, 
            t=t, 
            args=args,
            rtol=1e-6, 
            atol=1e-9
            )
        return out

    def __len__(self):
        return len(self.initial_conditions)

    def __getitem__(self, idx):

        """
        Retrieve the dataset instance at the given index.

        This method should be implemented by subclasses to return a specific 
        instance of the dataset, including any relevant history of outcomes, 
        treatments, covariates, and the initial state.
        
        The returned dictionary should contain the following keys:
        
        - 'outcome_history': torch.Tensor
            The history of the outcome variable up until the prediction 
            horizon.
        - 'treatment_history': torch.Tensor
            The history of the treatment doses applied before the prediction 
            horizon.
        - 'covariate_history': torch.Tensor
            The history of covariates up until the prediction horizon.
        - 'outcomes': torch.Tensor
            The outcome variable over the prediction horizon.
        - 'treatments': torch.Tensor
            The treatment doses applied over the prediction horizon.
        - 'initial_state': torch.Tensor
            The initial state of the system at the start of the simulation.

        Parameters
        ----------
        idx : int
            The index of the desired dataset instance.

        Returns
        -------
        dict
            A dictionary containing the specified keys with corresponding data.
        """

        raise NotImplementedError
    
    def visualize_stats(self):

        assert hasattr(self, 'data')
        
        fig, axs = plt.subplots(ncols=3, nrows=2)
        axs = axs.flatten()

        treatment_values = torch.cat(
            [self[idx]['treatments'] for idx in range(20)]
            )
        axs[0].set_title('treatments')
        axs[0].hist(treatment_values.flatten(), bins=40)
    
        outcome_values = torch.cat(
            [self[idx]['outcomes'] for idx in range(20)]
            )
        axs[1].set_title('outcomes')
        axs[1].hist(outcome_values.flatten(), bins=40)

        treatment_values = torch.cat(
            [self[idx]['treatment_history'] for idx in range(20)]
            )
        axs[2].set_title('treatment_history')
        axs[2].hist(treatment_values.flatten(), bins=40)

        outcome_values = torch.cat(
            [self[idx]['outcome_history'] for idx in range(20)]
            )
        axs[3].set_title('outcome_history')
        axs[3].hist(outcome_values.flatten(), bins=40)
    
        covariate_values = torch.cat(
            [self[idx]['covariate_history'] for idx in range(20)]
            )
        axs[4].set_title('covariate_history')
        axs[4].hist(covariate_values.flatten(), bins=40)

        fig.tight_layout()    
        fig.show()
    
    def visualize_trajectories(self):

        fig, axs = plt.subplots(ncols=3, nrows=3)
        fig.suptitle('treatments')
        for idx, ax in enumerate(axs.flatten()):
            instance = self[idx]
            ax.plot(instance['treatments'])
            ax.set_xlabel('time')
            ax.set_ylabel('value')
            ax.grid(True)
        fig.tight_layout()
        fig.show()

        fig, axs = plt.subplots(ncols=3, nrows=3)
        fig.suptitle('outcomes')
        for idx, ax in enumerate(axs.flatten()):
            instance = self[idx]
            ax.plot(instance['outcomes'])
            ax.set_xlabel('time')
            ax.set_ylabel('value')
            ax.grid(True)
        fig.tight_layout()
        fig.show()

        fig, axs = plt.subplots(ncols=3, nrows=3)
        fig.suptitle('treatment_history')
        for idx, ax in enumerate(axs.flatten()):
            instance = self[idx]
            ax.plot(instance['treatment_history'])
            ax.set_xlabel('time')
            ax.set_ylabel('value')
            ax.grid(True)
        fig.tight_layout()
        fig.show()
 
        fig, axs = plt.subplots(ncols=3, nrows=3)
        fig.suptitle('outcome_history')
        for idx, ax in enumerate(axs.flatten()):
            instance = self[idx]
            ax.plot(instance['outcome_history'])
            ax.set_xlabel('time')
            ax.set_ylabel('value')
            ax.grid(True)
        fig.tight_layout()
        fig.show()

        fig, axs = plt.subplots(ncols=3, nrows=3)
        fig.suptitle('covariate_history')
        for idx, ax in enumerate(axs.flatten()):
            instance = self[idx]
            ax.plot(instance['covariate_history'])
            ax.set_xlabel('time')
            ax.set_ylabel('value')
            ax.grid(True)
        fig.tight_layout()
        fig.show()

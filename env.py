import gymnasium as gym
from gymnasium import spaces
import numpy as np
import copy
from scipy.stats import poisson

import utils

MIN_LAMBDA = 0.1

class DynamicCrimeEnv(gym.Env):
    """Environment for simulating the observation of crimes in N different areas
       modeled using N Poisson distributions, as described in the paper "Fair Algorithms for Learning in Allocation Problems",
       with the addition of a dynamic factor that updates the lambda of each area based on the action taken by the agent.
    """
    def __init__(self, initial_lambdas, dynamic_factor, resources, max_crimes, max_t):
        """
        Args:
            initial_lambdas (list[int]): list of initial lambdas for each area
            dynamic_factor (float): factor governing the dynamic component of the environment; static env if 0.
            resources (int): number of patrols available at each iteration
            max_crimes (int): maximum number of crimes that can be committed in any area (Poisson samples's are clipped to this upper bound)
            max_t (int): the number of iterations before the environment is terminated
        """

        self.initial_lambdas = initial_lambdas
        self.lambdas = copy.deepcopy(self.initial_lambdas)
        self.dynamic_factor = dynamic_factor
        self.n_areas = len(self.lambdas)
        self.resources = resources
        self.max_crimes = max_crimes
        self.max_t = max_t

        self.prevented_crimes = None

        super().__init__() 
        self.action_space = spaces.MultiDiscrete(np.ones(self.n_areas, dtype=int) * resources)
        self.observation_space = spaces.MultiDiscrete(np.array([max_crimes] * self.n_areas, dtype=int))


    def step(self, action):
        """Simulates one step of the environment, updating the lambdas if dynamic and sampling crimes."""
        self.t += 1

        # sample crimes
        self.tot_crimes = self._sample()

        # compute prevented and committed crimes
        self.prevented_crimes = np.minimum(action, self.tot_crimes)
        committed_crimes = self.tot_crimes - self.prevented_crimes

        # compute metrics
        accuracy = np.sum(self.prevented_crimes) / np.sum(self.tot_crimes)
        eq_discovery = self._equality_of_discovery_probability(action)
        eq_wellness = self._equality_of_wellness(committed_crimes)

        # update state
        self.lambdas = self._update(action)

        # obtain output
        observation = self._get_obs()
        reward = None
        terminated = False
        truncated = self.t >= self.max_t
        info = {
            "prevented_crimes": self.prevented_crimes,
            "committed_crimes": committed_crimes,
            "accuracy": accuracy,
            "equality_discovery_prob": eq_discovery,
            "equality_wellness": eq_wellness,
            "true_lambdas": self.lambdas,
        }

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """Resets the environment to the initial state"""
        super().reset(seed=seed, options=options)
        self.t = 0
        self.lambdas = copy.deepcopy(self.initial_lambdas)
        self.tot_crimes = self._sample()

        observation = self._get_obs()
        info = {"true_lambdas": self.lambdas}
        return observation, info

    def render(self):
        """For compatibility with gym interface."""
        print("")

    def close(self):
        """For compatibility with gym interface."""
        pass

    def _update(self, action):
        """Update the lambdas of the environment based on the action taken by the agent.

        Args:
            action (list[int]): allocation of patrols in the previous iteration

        Returns:
            np.array: lambdas of the environment after the update
        """
        diff = np.clip(self.lambdas - action, a_min=-self.max_crimes, a_max=self.max_crimes)

        # rescale positive entries of diff such that its sum is 0
        diff = diff - (diff > 0) * np.sum(diff) / np.sum(diff > 0)

        assert np.sum(diff) <= 1e-3, f"Sum of diff is {np.sum(diff)}, should be approximately 0"
        return np.clip(self.lambdas + self.dynamic_factor * diff, MIN_LAMBDA, self.max_crimes)

    def _sample(self):
        """Draws samples from the Poisson distribution for each area as observation of crimes"""
        return np.clip(np.random.poisson(self.lambdas), 0, self.max_crimes)
    
    def _get_obs(self):
        """Returns the current observation of crimes in each area"""
        return self.prevented_crimes if self.prevented_crimes is not None else self.tot_crimes
    
    def _equality_of_discovery_probability(self, action):
        """Computes the fairness metric "Equality of discovery probability", as defined in the paper.

        Args:
            action (list[int]): the allocation of patrols in the previous iteration

        Returns:
            float: the maximum difference between the discovery probabilities of the areas
        """
        discovery_probability = np.zeros(self.n_areas)

        domain = np.arange(self.max_crimes+1)
        for area in range(self.n_areas):
            pmf = poisson.pmf(domain, self.lambdas[area])
            discovery_probability[area] = utils.discovery_prob(action[area], pmf, domain)

        return np.max(discovery_probability) - np.min(discovery_probability)
    
    def _equality_of_wellness(self, committed_crimes):
        """Computes the fairness metric "Equality of wellness, a proposal of this work.

        Args:
            committed_crimes (list[int]): the number of crimes committed in each area

        Returns:
            int: the maximum difference between the wellness of the areas, measured as crimes committed in that area
        """
        return np.max(committed_crimes) - np.min(committed_crimes)
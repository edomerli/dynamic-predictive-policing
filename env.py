import gymnasium as gym
import numpy as np
from gymnasium import spaces
import copy
from scipy.stats import poisson

import utils


class DynamicCrimeEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, initial_lambdas, dynamic_factor, resources, max_crimes, max_t):

        self.initial_lambdas = initial_lambdas
        self.lambdas = copy.deepcopy(self.initial_lambdas)
        self.dynamic_factor = dynamic_factor
        self.n_areas = len(self.lambdas)
        self.resources = resources
        self.max_crimes = max_crimes
        self.max_t = max_t

        self.discovered_crimes = None

        super().__init__() 
        self.action_space = spaces.MultiDiscrete(np.ones(self.n_areas, dtype=int) * resources)
        self.observation_space = spaces.MultiDiscrete(np.array([max_crimes] * self.n_areas, dtype=int))


    def step(self, action):
        self.t += 1

        # compute prevented and committed crimes
        # TODO: reformulate these, and double check code!
        # Devo fare: action -> sample -> minimum
        prevented_crimes = np.minimum(action, self.tot_crimes)
        committed_crimes = self.tot_crimes - prevented_crimes

        # compute metrics
        accuracy = np.sum(prevented_crimes) / np.sum(self.tot_crimes)   # TODO: considera il caso in cui sum di tot_crimes = 0
        eq_discovery = self._equality_of_discovery_probability(action)
        eq_wellness = self._equality_of_wellness(committed_crimes)

        # update state and sample
        self.lambdas = self._update(action)
        self.tot_crimes = self._sample()
        self.discovered_crimes = np.minimum(action, self.tot_crimes)


        observation = self._get_obs()
        reward = None
        terminated = False
        truncated = self.t >= self.max_t
        info = {
            "prevented_crimes": prevented_crimes,
            "committed_crimes": committed_crimes,
            "accuracy": accuracy,
            "equality_discovery_prob": eq_discovery,
            "equality_wellness": eq_wellness,
            "true_lambdas": self.lambdas,
        }

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.t = 0
        self.lambdas = copy.deepcopy(self.initial_lambdas)
        self.tot_crimes = self._sample()

        observation = self._get_obs()
        info = {"true_lambdas": self.lambdas}
        return observation, info

    def render(self):
        print(f"True lambdas: {self.lambdas} --- Crimes: {self.tot_crimes}")

    def close(self):
        pass

    def _update(self, action):
        return np.clip(self.lambdas - self.dynamic_factor * action + self.dynamic_factor * (action == 0), 0, self.max_crimes)

    def _sample(self):
        return np.clip(np.random.poisson(self.lambdas), 0, self.max_crimes)
    
    def _get_obs(self):
        return self.discovered_crimes if self.discovered_crimes is not None else self.tot_crimes
    
    def _equality_of_discovery_probability(self, action):
        discovery_probability = np.zeros(self.n_areas)

        domain = np.arange(self.max_crimes+1)
        for area in range(self.n_areas):
            pmf = poisson.pmf(domain, self.lambdas[area])
            discovery_probability[area] = utils.discovery_prob(action[area], pmf, domain)

        return np.max(discovery_probability) - np.min(discovery_probability)
    
    def _equality_of_wellness(self, committed_crimes):
        # TODO: anche qui considera che non stai sommando over t
        return np.max(committed_crimes) - np.min(committed_crimes)
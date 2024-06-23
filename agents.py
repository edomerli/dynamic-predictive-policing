
import numpy as np
from scipy.stats import poisson
import utils
import copy

from models import StaticCrimeMLEModel


class FixedUniformAgent():
    """Baseline agent that uniformly distributes resources among the areas, with a fixed action for the entire simulation."""
    # Decided to remove it from the paper, left it here in case it came useful in the future
    def __init__(self, resources, areas):
        self.resources = resources
        self.areas = areas
        self.action = np.array([resources//areas for _ in range(areas)], dtype=int)

    def seed(self, seed):
        np.random.seed(seed)    # note: no sources of randomness here

    def act(self, state, timestep):
        return self.action, {}

class SamplingUniformAgent():
    """Baseline agent that uniformly distributes resources among the areas, sampling a new action at each timestep."""
    def __init__(self, resources, areas):
        self.resources = resources
        self.areas = areas

    def seed(self, seed):
        np.random.seed(seed)

    def act(self, state, timestep):
        return np.random.multinomial(self.resources, np.ones(self.areas) / self.areas), {}


class StaticMLEAgent():
    """Agent that computes the MLE of the lambda parameters of the Poisson distributions for each area,
       and samples actions according to the MLE lambdas at each timestep after a burn-in period.
    """
    def __init__(self, resources, areas, burnin_steps, max_crimes, mle_interval):
        """Constructor

        Args:
            resources (int): number of patrols available
            areas (int): number of areas
            burnin_steps (int): number of initial steps used to observe and allocate uniformly at random
            max_crimes (int): upper bound on number of crimes
            mle_interval (int): frequency of steps at which to compute estimated lambdas using MLE
        """
        self.resources = resources
        self.areas = areas
        self.burnin_steps = burnin_steps
        self.max_crimes = max_crimes
        self.mle_interval = mle_interval

        self.dataset = {
            "crimes": [[] for _ in range(areas)],
            "actions": [[] for _ in range(areas)]
        }

    def seed(self, seed):
        np.random.seed(seed)

    def act(self, state, timestep):
        # update dataset
        if timestep != 0:
            for area in range(self.areas):
                self.dataset['crimes'][area].append(state[area])

        if timestep < self.burnin_steps:
            # sample uniformly at random during burnin period
            action = np.random.multinomial(self.resources, np.ones(self.areas) / self.areas)
            for area in range(self.areas):
                self.dataset['actions'][area].append(action[area])

            info_agent = {}

        else:
            if timestep % self.mle_interval == 0 or timestep == self.burnin_steps:
                self.lambdas = self.compute_mle_lambdas()

            # sample according to MLE lambdas
            action = np.random.multinomial(self.resources, self.lambdas / np.sum(self.lambdas))
            for area in range(self.areas):
                self.dataset['actions'][area].append(action[area])

            info_agent = {
                "pred_lambdas": self.lambdas
            }

        return action, info_agent
            
    
    def compute_mle_lambdas(self):
        lambdas = np.zeros(self.areas)
        for area in range(self.areas):
            initial_lambda = np.mean(self.dataset['crimes'][area])  # prior lambda is just the mean of the crimes observed in the area
            model = StaticCrimeMLEModel(
                np.array([(self.dataset['crimes'][area][i], self.dataset['actions'][area][i]) for i in range(len(self.dataset['crimes'][area]))]),
            )

            res = model.fit(start_params = np.array([initial_lambda]), disp=0)
            lambdas[area] = res.params[0]
            
        return lambdas



class FairAgent(StaticMLEAgent):
    """Agent proposed in the paper "Fair Algorithms for Learning in Allocation Problems" to obtain an
       alpha-fair allocation while maximizing utility
    """
    def __init__(self, resources, areas, burnin_steps, max_crimes, mle_interval, alpha):
        """Constructor

        Args:
            alpha (float): upper bound on the difference in discovery probability between any two areas
        """
        super().__init__(resources, areas, burnin_steps, max_crimes, mle_interval)
        self.alpha = alpha
        self.lambdas = [0.0 for _ in range(areas)]
        self.allocation = None

    def act(self, state, timestep):
        if timestep != 0:
            for area in range(self.areas):
                self.dataset['crimes'][area].append(state[area])

        if timestep < self.burnin_steps:
            action = np.random.multinomial(self.resources, np.ones(self.areas) / self.areas)
            for area in range(self.areas):
                self.dataset['actions'][area].append(action[area])

            info_agent = {}

        else:
            if timestep % self.mle_interval == 0 or timestep == self.burnin_steps:
                self.lambdas = self.compute_mle_lambdas()
                self.allocation = self._optimal_fair_allocation()
                
            action = self.allocation
            for area in range(self.areas):
                self.dataset['actions'][area].append(action[area])

            info_agent = {"pred_lambdas": self.lambdas}

        return action, info_agent

    def _optimal_fair_allocation(self):
        """Implements algorithm 1 from the paper to obtain an alpha-fair allocation while maximizing utility

        Returns:
            np.array: the allocation
        """
        # initialization
        opt_fair_allocation = np.zeros(self.areas)
        max_utility = -1
        upper_bounds = np.ones(self.areas) * self.resources
        lower_bounds = np.zeros(self.areas)

        # build tables to optimize runtime
        # discovery probability table for every combination of area (with a given lambda) and resources allocated to it
        disc_probability_table = np.zeros((self.areas, self.resources+1))
        domain = np.arange(self.max_crimes+1)
        for area in range(self.areas):
            pmf = poisson.pmf(domain, self.lambdas[area])
            for res in range(self.resources+1):
                disc_probability_table[area, res] = utils.discovery_prob(res, pmf, domain)

        # table of tail probabilities for every area and resource allocated to that area
        tail_prob_table = np.zeros((self.areas, self.resources+1))
        for area in range(self.areas):
            tail_prob_table[area] = utils.tail_prob(np.array(range(1, self.resources+2)), self.lambdas[area])


        # try all possible areas as the one with the highest discovery probability
        for area in range(self.areas):
            allocation = np.zeros(self.areas, dtype=np.int32)

            # try all possible allocations of resources to the area
            for res in range(self.resources+1):
                allocation[area] = res
                disc_probability = disc_probability_table[area, res]
                upper_bounds[area] = res
                lower_bounds[area] = res

                feasible = True
                for other_area in range(self.areas):
                    if other_area != area:
                        # compute range of resources that satisfy the fairness constraint
                        res_in_bounds = np.where(np.logical_and(disc_probability_table[other_area] >= disc_probability - self.alpha, disc_probability_table[other_area] <= disc_probability))
                        if res_in_bounds[0].shape[0] == 0:
                            feasible = False
                            break
                        
                        # update bounds and assign lower bound to allocation
                        upper_bounds[other_area] = res_in_bounds[0][-1]
                        lower_bounds[other_area] = res_in_bounds[0][0]
                        allocation[other_area] = lower_bounds[other_area]

                # skip if allocation is not feasible or if it exceeds the total resources
                if allocation.sum() > self.resources or not feasible:
                    continue    
                
                # greedily allocate remaining resources to maximize increase in utility through increase in tail probability
                res_left = self.resources - allocation.sum()
                for _ in range(res_left):
                    delta_tails = np.ones(self.areas) * (-np.inf)
                    for area_j in range(self.areas):
                        if allocation[area_j] < upper_bounds[area_j]:
                            delta_tails[area_j] = tail_prob_table[area, allocation[area_j]+1] - tail_prob_table[area, allocation[area_j]]
                    
                    best_greedy_area = np.argmax(delta_tails)
                    allocation[best_greedy_area] += 1

                # compute utility of allocation
                allocation_utility = utils.utility(allocation, self.lambdas)

                # update optimal allocation
                if allocation_utility > max_utility:
                    max_utility = allocation_utility
                    opt_fair_allocation = copy.deepcopy(allocation)
                
        return opt_fair_allocation
    



def round_series_retain_integer_sum(xs, new_sum):
    """Normalizes and scale the input serie to sum to new_sum, 
       then rounds the values to integers while retaining the sum, 
       minimizing the error in the integer approximation (L1 distance between the two series).

    Args:
        xs (iterable[float], list or np.array): the serie of floats
        new_sum (int): new desired sum of the series

    Returns:
        np.array[int]: the rounded serie
    """
    xs = (xs / np.sum(xs)) * new_sum
    N = round(np.sum(xs))
    Rs = [int(x) for x in xs]
    K = N - sum(Rs)
    assert K == int(K), f"K is not an integer: {K}"
    fs = [x - int(x) for x in xs]
    indices = [i for order, (e, i) in enumerate(reversed(sorted((e,i) for i,e in enumerate(fs)))) if order < K]
    ys = np.array([R + 1 if i in indices else R for i,R in enumerate(Rs)])

    assert np.sum(ys) == new_sum
    return ys

class SteeringAgent(FairAgent):
    """Proposed agent that steers the distribution of crimes by reallocating patrols to areas with higher lambda
       in order to deal with the dynamic environment
    """
    def __init__(self, resources, areas, burnin_steps, max_crimes, mle_interval, alpha, steering_factor, exploiting_range):
        """Constructor

        Args:
            steering_factor (float): factor governing the steering by means of reallocating resources to areas with higher lambda
            exploiting_range (float): range of relative distance from the mean lambda to exploit using the fair algorithm of FairAgent
        """
        super().__init__(resources, areas, burnin_steps, max_crimes, mle_interval, alpha)
        self.steering_factor = steering_factor
        self.exploiting_range = exploiting_range

    def act(self, state, timestep):
        # update dataset
        if timestep != 0:
            for area in range(self.areas):
                self.dataset['crimes'][area].append(state[area])

        if timestep < self.burnin_steps:
            action = np.random.multinomial(self.resources, np.ones(self.areas) / self.areas)
            for area in range(self.areas):
                self.dataset['actions'][area].append(action[area])

            info_agent = {}

        else:
            if timestep % self.mle_interval == 0 or timestep == self.burnin_steps:
                self.lambdas = self.compute_mle_lambdas_dynamic(window=self.burnin_steps)

                # if the lambdas are close to the mean, exploit the fair algorithm, otherwise steer the allocation
                average_rel_dist_from_mean_lambda = np.mean(np.abs(self.lambdas - np.mean(self.lambdas)) / np.mean(self.lambdas))
                if average_rel_dist_from_mean_lambda < self.exploiting_range:
                    self.allocation = self._optimal_fair_allocation()
                else:
                    self.allocation = round_series_retain_integer_sum(self.lambdas, self.resources)
                    self.allocation = self._steering_allocation(self.allocation)
                
            action = self.allocation
            for area in range(self.areas):
                self.dataset['actions'][area].append(action[area])

            info_agent = {"pred_lambdas": self.lambdas}

        return action, info_agent
    

    def compute_mle_lambdas_dynamic(self, window):
        """Runs MLE on the dataset, but only on the last "window"-observations

        Args:
            window (int): number of most recent datapoints to consider

        Returns:
            np.array[float]: the array of estimated lambdas
        """
        for area in range(self.areas):
            self.dataset['actions'][area] = self.dataset['actions'][area][-window:]
            self.dataset['crimes'][area] = self.dataset['crimes'][area][-window:]

        return self.compute_mle_lambdas()

    def _steering_allocation(self, allocation):
        """Computes the steering allocation by reallocating resources to areas with higher lambda

        Args:
            allocation (np.array[int]): the allocation before steering

        Returns:
            np.array[int]: the steering allocation
        """
        mean_lambdas = np.mean(self.lambdas)
        directions = np.where(self.lambdas > mean_lambdas, 1, -1)

        decay = np.tanh((np.abs(self.lambdas - mean_lambdas) / mean_lambdas) * 2)

        uniques, counts = np.unique(directions, return_counts=True)
        counts_dict = dict(zip(uniques, counts))

        margin = np.array([round(self.steering_factor * decay[i]) if allocation[i] - round(self.steering_factor * decay[i]) > 1 else np.clip(int(allocation[i] - 1), a_min=0, a_max=None) for i in range(self.areas)])
        margin = np.where(directions == -1, margin, 0)

        reserved_resources = np.sum(margin)

        base_redistribution = reserved_resources // counts_dict[1]
        redistributed_resources = np.where(directions == 1, base_redistribution, -margin)

        remainder = reserved_resources % counts_dict[1]

        if remainder > 0:
            remainder_distribution = np.random.default_rng(42).multivariate_hypergeometric([1] * counts_dict[1], remainder)
            assert np.sum(remainder_distribution) == remainder, "Remainder distribution is not correct"

            j = 0
            for i in range(self.areas):
                if directions[i] == 1:
                    redistributed_resources[i] += remainder_distribution[j]
                    j += 1

        steering_allocation = allocation + redistributed_resources

        assert np.sum(steering_allocation) == self.resources, f"Resources are not all used, sum: {np.sum(steering_allocation)}"

        return steering_allocation
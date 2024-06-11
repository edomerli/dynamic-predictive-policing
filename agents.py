
import numpy as np
# import pymc as pm
import pytensor
import pytensor.tensor as pt
# from pymc.pytensorf import collect_default_updates
from scipy.stats import poisson
import time

import utils
import copy

from models import StaticCrimeMLEModel

# TODO: remove, with imports above
# LOG_PYMC = False

class FixedUniformAgent():
    def __init__(self, resources, areas):

        self.resources = resources
        self.areas = areas
        self.action = np.array([resources//areas for _ in range(areas)], dtype=int)

    def seed(self, seed):
        np.random.seed(seed)    # no sources of randomness here

    def act(self, state, timestep):
        return self.action, {}

class SamplingUniformAgent():
    def __init__(self, resources, areas):
        self.resources = resources
        self.areas = areas

    def seed(self, seed):
        np.random.seed(seed)

    def act(self, state, timestep):
        return np.random.multinomial(self.resources, np.ones(self.areas) / self.areas, ), {}
    

# TODO: o modificare questo agent, o farne uno nuovo, che faccia learning/MLE e poi usi sempre la stessa azione ottenuta dalla distribuzione
#       delle lambda imparate
class StaticMLEAgent():
    def __init__(self, resources, areas, burnin_steps, max_crimes, mle_interval):
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
            action = np.random.multinomial(self.resources, np.ones(self.areas) / self.areas)
            for area in range(self.areas):
                self.dataset['actions'][area].append(action[area])

            info_agent = {}

        else:
        
            if timestep % self.mle_interval == 0 or timestep == self.burnin_steps:
                self.lambdas = self.compute_mle_lambdas()

            action = np.random.multinomial(self.resources, self.lambdas / np.sum(self.lambdas))
            for area in range(self.areas):
                self.dataset['actions'][area].append(action[area])

            info_agent = {
                "pred_lambdas": self.lambdas
            }

        return action, info_agent
            
    
    def compute_mle_lambdas(self):
        # compute MLE
        lambdas = np.zeros(self.areas)
        for area in range(self.areas):
            initial_lambda = np.mean(self.dataset['crimes'][area])
            model = StaticCrimeMLEModel(
                np.array([(self.dataset['crimes'][area][i], self.dataset['actions'][area][i]) for i in range(len(self.dataset['crimes'][area]))]),
            )

            res = model.fit(start_params = np.array([initial_lambda]), disp=0)
            lambdas[area] = res.params[0]
            
        return lambdas

    ### OLD VERSION USING PYMC ###  
    # def compute_mle_lambdas(self):
    #     # compute MLE
    #     with pm.Model(coords={'idx': np.arange(self.areas)}) as model:
    #         lambdas = pm.Uniform('lambdas', 0, self.max_crimes, dims=('idx',))
    #         crimes = pm.Poisson('crimes', lambdas)
    #         obs_crimes = pm.Deterministic("obs_crimes", pm.math.minimum(crimes, self.dataset['actions']), observed=self.dataset['crimes'])

    #         trace = pm.sample(draws=1000, tune=1000, cores=1, chains=4, progressbar=LOG_PYMC)
    #         pred_lambdas = trace.posterior['lambdas'].mean(axis=(0, 1)).to_numpy()

    #     return pred_lambdas
    ### ###


### OLD VERSION USING PYMC ###
# def crime_dist(lambdas_init, actions, dynamic_factor, max_crimes, size):
#     # TODO: actions to 
#     def crime_step(action, lambdas, dynamic_factor, max_crimes):
#         # TODO: se il problema di questo modellamento è che non posso clippare lambda a zero (perchè poisson prende lambda > 0) devo fixarlo anche negli altri
#         #       posti con un min_lambda_value = e.g. 0.01
#         lambdas = pm.math.clip(lambdas - dynamic_factor * action + dynamic_factor * pm.math.eq(action, 0), 0.01, max_crimes)
#         crimes = pm.Poisson.dist(lambdas)

#         return (lambdas, crimes), collect_default_updates([crimes])
    
#     [result_lambdas, result_crimes], updates = pytensor.scan(
#         fn=crime_step, 
#         sequences=[actions],
#         outputs_info=[lambdas_init, None],
#         non_sequences=[dynamic_factor, max_crimes],
#         n_steps=actions.shape[0],
#         strict=True)
    
#     return result_crimes
### ###


class DynamicMLEAgent():
    def __init__(self, resources, areas, burnin_steps, max_crimes, dynamic_factor, mle_interval):
        self.resources = resources
        self.areas = areas
        self.burnin_steps = burnin_steps
        self.max_crimes = max_crimes
        self.dynamic_factor = dynamic_factor
        self.mle_interval = mle_interval    # TODO: usa mle_interval in act per fare il sampling solo ogni mle_interval timesteps

        self.dataset = {'crimes': [], 'actions': []}

    ### OLD VERSION USING PYMC ###
    # def act(self, state, timestep):
    #     # update dataset
    #     self.dataset['crimes'].append(state)

    #     if timestep < self.burnin_steps:
    #         burnin_action = np.random.multinomial(self.resources, np.ones(self.areas) / self.areas)
    #         self.dataset['actions'].append(burnin_action)
    #         return burnin_action, {}
        
    #     coords = {'areas': np.arange(self.areas), 
    #               'steps': np.arange(len(self.dataset['crimes'])-1), 
    #               'observations': np.arange(len(self.dataset['crimes']))
    #               }
    #     # compute MLE
    #     # TODO: idea -> potrei fare il modello cosi in init, tenere la lunghezza fissa a tipo 100 cosi da non far esplodere la 
    #     #       computation complexity (costante nella size dell'input) e poi, in act, solo settare le ultime 100 variabili + fare sampling
    #     #       (per velocizzare, potrei provare di accorciare la lunghezza da 100 a tipo 50 e vedere come cambiano i risultati!)
    #     with pm.Model(coords=coords) as model:
    #         actions_dataset = pt.as_tensor_variable(self.dataset['actions'])

    #         crimes_init_obs = pm.Data("crimes_init_obs", np.zeros(self.areas), dims=('areas',))
    #         lambdas_init = pm.Uniform('lambdas_init', 0, self.max_crimes, dims=('areas',))
    #         crimes_init = pm.Poisson('crimes_init', lambdas_init, observed=crimes_init_obs)

    #         crimes_obs = pm.Data('crimes_obs', np.zeros((len(self.dataset['crimes'])-1, self.areas)), dims=('steps', 'areas',))
    #         crimes = pm.CustomDist(
    #             'crimes_dist',
    #             lambdas_init,
    #             actions_dataset,
    #             self.dynamic_factor,
    #             self.max_crimes,
    #             dist=crime_dist,
    #             observed=crimes_obs
    #         )

    #         crimes_init = crimes_init.reshape((1, self.areas))
    #         crimes = pm.Deterministic("crimes", pt.concatenate([crimes_init, crimes], axis=0), dims=('observations', 'areas',))

    #     crimes_init_obs.set_value(np.array(self.dataset['crimes'][0]))
    #     crimes_obs.set_value(np.array(self.dataset['crimes'][1:]))

    #     with model:

    #         print("Start sampling")
    #         start = time.time()
    #         trace = pm.sample(draws=1000, tune=1000, cores=1, chains=4, progressbar=LOG_PYMC)
    #         end = time.time()
    #         print("End sampling")
    #         print(end - start, "time in seconds")
    #         pred_lambdas = trace.posterior['lambdas_init'].mean(axis=(0, 1)).to_numpy()
    #         print(f"Predicted initial lambdas: {pred_lambdas}")
    #         exit()

    #     # TODO: vedere come usare il lambda backtracker che funzia sopra
    #     action = np.random.multinomial(self.resources, pred_lambdas / np.sum(pred_lambdas))
    #     info_agent = {
    #         "pred_lambdas": pred_lambdas
    #     }
    #     # print(pred_lambdas)
    #     # print(np.random.multinomial(self.resources, pred_lambdas / np.sum(pred_lambdas)))

    #     # update dataset with the action taken 
    #     self.dataset['actions'].append(action)
    #     return action, info_agent
    ### ###



class FairAgent(StaticMLEAgent):
    def __init__(self, resources, areas, burnin_steps, max_crimes, mle_interval, alpha):
        super().__init__(resources, areas, burnin_steps, max_crimes, mle_interval)
        self.alpha = alpha
        self.lambdas = [0.0 for _ in range(areas)]

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
        opt_fair_allocation = np.zeros(self.areas)
        max_utility = -1
        upper_bounds = np.ones(self.areas) * self.resources
        lower_bounds = np.zeros(self.areas)

        print(f"Beliefs {self.lambdas}")

        # build tables to optimize runtime
        disc_probability_table = np.zeros((self.areas, self.resources+1))
        domain = np.arange(self.max_crimes+1)
        for area in range(self.areas):
            pmf = poisson.pmf(domain, self.lambdas[area])
            for res in range(self.resources+1):
                disc_probability_table[area, res] = utils.discovery_prob(res, pmf, domain)



        tail_prob_table = np.zeros((self.areas, self.resources+1))
        for area in range(self.areas):
            tail_prob_table[area] = utils.tail_prob(np.array(range(1, self.resources+2)), self.lambdas[area])


        for area in range(self.areas):
            allocation = np.zeros(self.areas, dtype=np.int32)

            for res in range(self.resources+1):
                allocation[area] = res
                disc_probability = disc_probability_table[area, res]
                upper_bounds[area] = res
                lower_bounds[area] = res

                # print(f"Area {area}, resource {res}, disc_probability {disc_probability}")

                feasible = True
                for other_area in range(self.areas):
                    if other_area != area:
                        res_in_bounds = np.where(np.logical_and(disc_probability_table[other_area] >= disc_probability - self.alpha, disc_probability_table[other_area] <= disc_probability))
                        if res_in_bounds[0].shape[0] == 0:
                            feasible = False
                            break

                        upper_bounds[other_area] = res_in_bounds[0][-1]
                        lower_bounds[other_area] = res_in_bounds[0][0]
                        allocation[other_area] = lower_bounds[other_area]

                if allocation.sum() > self.resources or not feasible:
                    continue    
                
                res_left = self.resources - allocation.sum()
                for _ in range(res_left):
                    delta_tails = np.ones(self.areas) * (-np.inf)
                    for area_j in range(self.areas):
                        if allocation[area_j] < upper_bounds[area_j]:
                            delta_tails[area_j] = tail_prob_table[area, allocation[area_j]+1] - tail_prob_table[area, allocation[area_j]]
                    
                    best_greedy_area = np.argmax(delta_tails)
                    allocation[best_greedy_area] += 1


                allocation_utility = utils.utility(allocation, self.lambdas)

                if allocation_utility > max_utility:
                    max_utility = allocation_utility
                    opt_fair_allocation = copy.deepcopy(allocation)
                
        print(f"Chosen fair allocation {opt_fair_allocation}, with utility: {max_utility}\n")
        return opt_fair_allocation
    
from scipy.stats import poisson
import numpy as np


def discovery_prob(resource, pmf, domain):
    # using the definition of expected value
    domain_wout_zero = domain[1:]
    exp_argument = np.minimum([resource] * domain_wout_zero.shape[0], domain_wout_zero) / domain_wout_zero
    return  pmf[0] + np.sum(pmf[1:] * exp_argument)




def utility(allocation, lambdas):
    allocation_utility = 0
    for area_i in range(len(allocation)):
        allocation_utility += tail_prob(np.array(range(1, allocation[area_i]+1)), lambdas[area_i]).sum()
    
    return allocation_utility

def tail_prob(c, lmbda):
    return 1 - poisson.cdf(c-1, lmbda)
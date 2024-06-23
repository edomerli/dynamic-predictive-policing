from scipy.stats import poisson
import numpy as np


def discovery_prob(resource, pmf, domain):
    """Computes discovery probability for a given resource allocation on a given 
       probability mass function defined over domain, assuming min(.) discovery model

    Args:
        resource (int): the resource allocated to the area
        pmf (np.array): the probability mass function
        domain (np.array): the domain of the pmf

    Returns:
        float: discovery probability using min(.) discovery model
    """
    domain_wout_zero = domain[1:]
    exp_argument = np.minimum([resource] * domain_wout_zero.shape[0], domain_wout_zero) / domain_wout_zero
    return  pmf[0] + np.sum(pmf[1:] * exp_argument)




def utility(allocation, lambdas):
    """The utility of an allocation of resources to areas described by poissons of parameters lambdas

    Returns:
        float: the utility of the allocation
    """
    allocation_utility = 0
    for area_i in range(len(allocation)):
        allocation_utility += tail_prob(np.array(range(1, allocation[area_i]+1)), lambdas[area_i]).sum()
    
    return allocation_utility

def tail_prob(c, lmbda):
    """Computes the tail probability (P(X) >= c) of a poisson distribution with parameter lmbda at point c

    Args:
        c (int): the point at which to compute the tail probability (inclusive)
        lmbda (int): lambda parameter of the poisson distribution

    Returns:
        float: the tail probability P(X) >= c
    """
    return 1 - poisson.cdf(c-1, lmbda)
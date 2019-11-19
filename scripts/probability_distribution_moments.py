# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 16:06:22 2019

@author: LaurencT
"""

def beta_moments(mean, sd):
    """Calculates the moments of a beta distribution based on mean and sd
       as they are more intuitive inputs
       More info on beta: https://en.wikipedia.org/wiki/Beta_distribution
       Inputs:
           mean - a float - the chosen mean of the distribution
           sd - a float - the chosen standard deviation of the distribution
       Returns:
           dict - keys: alpha and beta, values of corresponding moments
    """
    if (sd**2) >= (mean*(1 - mean)):
        raise ValueError('Variance (sd^2) must be less than (mean*(1-mean))')
    else:
        term = mean*(1 - mean)/(sd**2) - 1
        alpha = mean*term
        beta = (1 - mean)*term
        return {'alpha': alpha, 'beta': beta}
    
def gamma_moments(mean, sd):
    """Calculates the moments of a gamma distribution based on mean and sd
       as they are more intuitive inputs
       More info on gamma: https://en.wikipedia.org/wiki/Gamma_distribution
       Inputs:
           mean - a float - the chosen mean of the distribution
           sd - a float - the chosen standard deviation of the distribution
       Returns:
           dict - keys: shape and scale, values of corresponding moments
    """
    if mean < 0:
        raise ValueError('The mean must be above 0')
    else:
        scale = sd**2/mean
        shape = mean/scale
        return {'scale':scale, 'shape':shape}

def gamma_moments_burden(mean, sd):
    """Calculates the moments of a gamma distribution based on mean and sd
       as they are more intuitive inputs
       More info on gamma: https://en.wikipedia.org/wiki/Gamma_distribution
       Inputs:
           mean - a float - the chosen mean of the distribution
           sd - a float - the chosen standard deviation of the distribution
       Returns:
           dict - keys: shape and scale, values of corresponding moments
    """
    if mean < 0:
        raise ValueError('The mean must be above 0')
    # To deal with case where burden is 0 in mean, lower and upper to ensure 
    # that it stays very close to 0, which still allowing the burden simulation
    # to run
    elif mean == 0:
        return {'scale':0, 'shape':100000}
    elif sd == 0:
        sd = mean/10
        scale = sd**2/mean
        shape = mean/scale
        return {'scale':scale, 'shape':shape}
    else:
        scale = sd**2/mean
        shape = mean/scale
        return {'scale':scale, 'shape':shape}
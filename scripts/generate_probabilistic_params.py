# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 15:03:53 2019

@author: LaurencT
"""

import re
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.stats import beta
from scipy.stats import gamma
from probability_distribution_moments import beta_moments
from probability_distribution_moments import gamma_moments

def get_probabilistic_column_names(param_user):
    """Based on user inputted parameters, takes the parameters and subsets the 
    relevant parameters for the probabilistic sensitivity analysis
    Inputs:
        param_user - dict - keys: id_codes, values: series of parameters
    Returns:
        columns_names: a list of relevant column names for probabilistic 
            sensitivity analysis
    """
    # Gets all parameter names, indexes a dictionary which is fine as all values 
    # have the same indexes
    param_names = list(param_user.values())[0].index.tolist()
    column_names = [val for val in param_names if not re.search('upper|lower', val)]
    return column_names

def columns_to_add(column_names):
    """Subsets the list for elements that contain mean and then rename them
       so they aren't called mean, these will be the new names for columns 
       to be simluated probabilistically
       Inputs:
           column_names - a list of parameter names / df indexes 
       Returns:
           new_columns - a list of names of columns to be created
    """
    new_columns = [val for val in column_names if re.search('mean', val)]
    new_columns = [re.sub('_mean', '', val) for val in new_columns]
    return new_columns

def simulate_column(mean, sd, column, num_trials):
    """Simulates num_trials values of a parameter based on the mean, sd and column type
       Inputs:
           mean - float - the base case value of that parameter
           sd - float - the standard deviation of that parameter
           column - str - the name of column to be  simulated
           num_trials - int - the num of trials to be simulated
       Returns:
           a numpy array of simulated values
    """
    if sd == 0:
        data = np.array([mean for i in range(1,num_trials+1)])
    # Use normal for things that vary around 1, inflation factor will need
    # changing probaby #~
    elif column in ['population', 'inflation_factor', 'share',
                    'coverage', 'prob_cover']:    
        data = norm.rvs(size = num_trials, loc = mean, scale = sd)
    # Use beta distribution for all paramters that are a proportion
    elif column in ['disease_1_prop', 'disease_2_prop', 'disease_3_prop',
                    'intervention_cut', 'efficacy']:
        # Disease prop may not be a proportion - may inflate the naive population
        if (re.search('disease_[1-3]_prop', column) and mean>1):
            data = norm.rvs(size = num_trials, loc = mean, scale = sd)
        else:
            data = beta.rvs(a = beta_moments(mean, sd)['alpha'], 
                        b = beta_moments(mean, sd)['beta'], 
                        size = num_trials)
    # Use gamma for parameters that are non-negative and have a right skew
    elif column in ['endem_thresh']:
        data = gamma.rvs(a = gamma_moments(mean, sd)['shape'], 
                         scale = gamma_moments(mean, sd)['scale'], 
                         size = num_trials)
    # If a new parameter has been added will have to add it to one of the lists
    # above or this ValueError will be thrown every time
    else:
        raise ValueError(column, ' is an invalid column name')
    return data

def create_prob_df(param_prob, id_code, new_columns, num_trials):
    """Simulate new columns of paramters probabilistically based on assumed 
       means and sds, these columns are then added the data frames
       Inputs:
           param_prob - dict - keys: id_codes, values: the relevant parameters
               for probabilistic simulation
           id_code - the id_code for this project
           new_columns - list - names of all columns to be added
           num_trials - the number of trials to be simulated
       Returns:
           df - with the probabilistic and non-probabilistic columns
    """
    # Select the relevant parameters
    param_example = param_prob[id_code]
    # Create df and add repeated columns of non-prob param values
    prob_df = pd.DataFrame()
    for param in param_example.index.tolist():
        series = pd.Series(param_example.loc[param], index = range(1, num_trials + 1), name = param)
        prob_df = prob_df.append(series)
    # Transpose the df
    prob_df = prob_df.T
    # Generate new columns based on probability distributions
    for column in new_columns:
        mean = float(param_example[column+'_mean'])
        sd = float(param_example[column+'_SD'])
        data = simulate_column(mean, sd, column, num_trials)
        # Turn the relevant new data into a series (which becomes a column)
        new_column = pd.Series(data, index = range(1, num_trials + 1), name = column)
        prob_df = pd.concat([prob_df, new_column.T], axis = 1, sort = False)
        # Drop unnecessary columns
        columns_to_drop = [name for name in list(prob_df) if re.search("mean|SD", name)]
        prob_df = prob_df.drop(columns_to_drop, axis =1)
    return prob_df
    
def get_probabilistic_params(analysis_type, param_user):
    """Turns param_user dict into a set of parameters for each of the probabilistic
       sensitivity analyses
       Inputs:
           analysis_type - the dictionary summarising what type of analysis is 
               being undertaken, must contain the keys 'run_probabilistic' and 
               'num_trials'
           param_user - dict - keys: id_codes, values: series of parameters
       Returns:
           param_dict - a dictionary with id_codes as keys and dfs of
               parameters for each of the probabilistic trials
    """
    # Remove parameters related to upper and lower bounds, focus on means and sds
    num_trials = analysis_type['num_trials']
    column_names = get_probabilistic_column_names(param_user)
    new_columns = columns_to_add(column_names)
    param_prob = {k:v[column_names] for k, v in param_user.items()}
    # Generate the relevant id_codes
    id_codes = list(param_user.keys())
    param_dict = {code: create_prob_df(param_prob, code, new_columns, num_trials) 
                  for code in id_codes}
    return param_dict

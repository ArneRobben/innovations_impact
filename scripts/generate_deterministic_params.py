# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 15:01:30 2019

@author: LaurencT
"""

import re
import pandas as pd

def lists_to_dict(list_keys, list_values):
    """two ordered lists to a dict where list_keys are the keys and list_values 
       are the values
       Inputs:
           list_keys - a list of n unique elements, where  n = 0, 1 or many
           list_values - a list of n elements
       Returns:
           a dict of length n
    """
    if len(list_keys) > len(set(list_keys)):
        raise ValueError('The list of keys is not unique')
    elif len(list_keys) != len(list_values):
        raise ValueError('The lists are not the same length')
    else:
        return {k:v for k,v in zip(list_keys, list_values)}

def deterministic_lists(analysis_type, param_user):
    """Defines the different LTLI scenarios and which list of parameters should
       be used for each of them
       Inputs:
           analysis_type - the dictionary summarising what type of analysis is 
               being undertaken, must contain the key 'run_deterministic'
           param_user - dict - keys: id_codes, values: series of parameters
       Returns:
           A dict - keys: scenario names, values - the list of parameters to
               be called for the corresponding scenario. Example:
               {'base': [burden_mean,
                         ...,
                         coverage_mean],
                '....': [...],
                'burden_lower': [burden_lower,
                                 ...,
                                 coverage_mean]}
    """
    # Create scenario names based on parameter names
    # This takes a random series from the dictionary's values to get the indexes
    # this is fine as all the values have the same indexes
    param_names = list(param_user.values())[0].index.tolist()
    scenario_names = [name for name in param_names if re.search('upper|lower', name)]
                      
    # Exclude disease proportions because they vary with disease burden
    scenario_names = [name for name in scenario_names if not re.search('disease.*prop', name)]
                      
    # Add in the base case and placeholders for burden scenarios
    scenario_names = ['base'] + scenario_names + ['burden_upper', 'burden_lower']
    
    # Create a list of parameter series indexes to select the right paramters 
    # for each of the scenarios
    columns_to_select = [column for column in param_names 
                         if not re.search('lower|upper|SD', column)]

    # Create a dictionary keys: scenario names, values: lists of relevant column names
    dict_of_columns = {}
    for scenario in scenario_names:
        # Don't change anything in the base case
        if scenario == 'base':
            new_columns = columns_to_select.copy()
        # Burden scenarios influence strain proportion, will use GBD ranges - 
        # not assumed ones so no adjustment factor for burden estimates yet
        elif re.search('burden', scenario):
            scenario_type = scenario[scenario.rindex('_')+1:]
            new_columns = [re.sub('mean', scenario_type, column) 
                           if re.search('disease.*prop', column) else column 
                           for column in columns_to_select]
        # All others replace the column index for mean with upper or lower
        else:
            scenario_focus = scenario[:scenario.rindex('_')] 
            column_name_to_replace = scenario_focus + '_mean'
            new_columns = [scenario 
                           if column == column_name_to_replace 
                           else column 
                           for column in columns_to_select]
        dict_of_columns[scenario] = new_columns.copy()
    return dict_of_columns

def scenario_param(param_user, param_list, id_code):
    """Subsets a series of parameters based on which are required for this 
       scenario, and renames the parameters to standard names (not upper, lower
       or mean anymore)
       Inputs:
           param_list: a list of parameter indexes required for this scenario
           param_user - dict - keys: id_codes, values: series of parameters
           id_code: a string used to index to get data on the right project
       Returns:
           param: a series that has been indexed and renamed
    """
    param = param_user[id_code]
    # Series to get the relevant parameters for the scenario
    param = param.reindex(param_list)
    # Get the names of those parameters and replace them with standard ones so 
    # all the parameters have standard names (no upper, lower, mean)
    param_indexes = param.index.values.tolist()
    param_indexes_new = [re.sub('_mean|_lower|_upper', '', x) 
                         if re.search('mean|lower|upper', x) else x 
                         for x in param_indexes]
    # Generate mapping of old indexes to new indexes and use that to rename
    indexes_dict = lists_to_dict(param_indexes, param_indexes_new)
    param = param.rename(indexes_dict)
    # Return the relevant parameters with standardised names
    return param 

def get_deterministic_params(analysis_type, param_user):
    """Turns param_user dict into a set of parameters for each of the 
       deterministic analyses
       Inputs:
           analysis_type - the dictionary summarising what type of analysis is 
               being undertaken, must contain the keys 'run_deterministic' and
               'num_trials'
           param_user - dict - keys: id_codes, values: series of parameters
       Returns:
           param_dict - a dict- keys: id_codes, values dfs of parameters for 
               each of the deterministic scenarios
    """
    id_codes = list(param_user.keys())
    # Generate a dictionary of which parameters are relevant for each scenario
    scenarios_dict = deterministic_lists(analysis_type, param_user)
    # Create a dictionary where id_codes are the keys, and the values are dfs
    # of the parameter values for each scenario
    param_dict = {}
    for code in id_codes:
        # Create a list to be filled with series, where each series is a set
        # of parameters for a scenario
        scenario_params = []
        for scenario in scenarios_dict.keys():
            param_list = scenarios_dict[scenario]
            param = scenario_param(param_user, param_list, code)
            param = param.rename(scenario)
            scenario_params.append(param)
        # Concatenate the series and transpose
        param_df = pd.concat(scenario_params, axis = 1, 
                             join_axes=[scenario_params[0].index])
        param_df = param_df.transpose()
        # populate the dictionary
        param_dict[code] = param_df
    return param_dict
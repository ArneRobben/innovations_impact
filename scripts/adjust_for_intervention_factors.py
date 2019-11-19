# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 15:10:33 2019

@author: LaurencT
"""

import re
import numpy as np

def create_target_population(cov_pop_burden_df, param_df, index):
    """Adds a new column to cov_pop_burden_df which is the target population
       Inputs:
           cov_pop_burden_df - a df which must contain 'incidence_number' and 
               'pop_0-0' columns
           param_df - a df of parameters which must contain the column 'intervention_type'
               where all of the values are 'Therapeutic', 'Diagnostic' or 'Vaccine'
           index - a string that is one of the indexes of param_df
       Returns:
           a cov_pop_burden_df with target_pop column added
    """
    intervention_type = param_df.loc[index, 'intervention_type'] 
    # Select incidence as the target population if it is a therapeutic or diagnostic   
    if re.search('Therapeutic', intervention_type):
        cov_pop_burden_df['target_pop'] = cov_pop_burden_df['incidence_number']
    elif intervention_type in ['Device', 'Rapid diagnostic test']:
        cov_pop_burden_df['target_pop'] = cov_pop_burden_df['incidence_number']
    # Select population column if it is a vaccine 
    #~ This assumes it is an infant vaccination
    elif param_df.loc[index, 'intervention_type'] == 'Vaccine':
        cov_pop_burden_df['target_pop'] = cov_pop_burden_df['pop_0-0']
    else:
        raise ValueError('The value of intervention_type for is not valid')
    return cov_pop_burden_df
   
def apply_endemicity_threshold(cov_pop_burden_df, param_df, index):
    """Reduces coverage rates in df if burden is below the threshold
       Inputs:
           cov_pop_burden_df - a df with the column 'coverage'
           param_df - a df of parameters, must contain the columns coverage_below_threshold,
               endem_thresh and endem_thresh_metric
           index - a string that is one of the indexes of param_df
       Returns:
           a df with coverage reduced in geographies with burden below the threshold
    """
    new_cov_pop_burden_df = cov_pop_burden_df.copy()
    # Defines which burden column is relevant for the endemicity threshold
    endem_thresh_column = param_df.loc[index, 'endem_thresh_metric']+'_rate'
    # Defines the threshold and coverage below the threshold
    endem_thresh = param_df.loc[index, 'endem_thresh']
    coverage_below_threshold = param_df.loc[index, 'coverage_below_threshold']
    # Applies the endemicity threshold
    new_cov_pop_burden_df['coverage'] = (
            np.where(new_cov_pop_burden_df[endem_thresh_column] < endem_thresh,
            coverage_below_threshold,
            new_cov_pop_burden_df['coverage'])
                                        )
    return new_cov_pop_burden_df

def apply_diagnostic_inflation(cov_pop_burden_df, param_df, index):
    """Inflates the target_pop in df 
       Inputs:
           cov_pop_burden_df - a df with the column 'target_pop'
           param_df - a df of parameters, must contain the columns 'inflation_factor'
           index - a string that is one of the indexes of param_df
       Returns:
           a df with inflated target_pop column
    """
    new_cov_pop_burden_df = cov_pop_burden_df.copy()
    new_cov_pop_burden_df['target_pop'] = (new_cov_pop_burden_df['target_pop'] * 
                                           param_df.loc[index, 'inflation_factor'])
    return new_cov_pop_burden_df

def apply_intervention_cut(cov_pop_burden_df, param_df, index):
    """Reduces prob_cover by the intervention_cut
       Inputs:
           cov_pop_burden_df - a df with the column 'prob_cover'
           param_df - a df of parameters, must contain the columns 'intervention_cut'
           index - a string that is one of the indexes of param_df
       Returns:
           a df with inflated target_pop column
    """
    cov_pop_burden_df['prob_cover'] = (cov_pop_burden_df['prob_cover'] * 
                                       param_df.loc[index, 'intervention_cut'])
    return cov_pop_burden_df

def adjust_for_intervention_factors(cov_pop_burden_dict, param_dict):
    """Applies several adjustments to the dfs that are the values of the nested
       dict cov_pop_burden_dicts
       Inputs:
           cov_pop_burden_dict - a nested dict where keys are id_codes and then
               scenarios and the values are dfs 
           param_dict - a dict where the keys are id_codes and the values are dfs
               with a row of parameters for each of the scenarios
       Returns:
           a nested dict of updated dfs
    """
    for code in param_dict.keys():
        param_df = param_dict[code]
        for index in param_df.index.tolist():
            cov_pop_burden_df = cov_pop_burden_dict[code][index].copy()
            intervention_type = param_df.loc[index, 'intervention_type'] 
            cov_pop_burden_df = create_target_population(cov_pop_burden_df, 
                                                         param_df, 
                                                         index)
            if intervention_type in ['Device', 'Rapid diagnostic test']:
                cov_pop_burden_df = apply_diagnostic_inflation(cov_pop_burden_df, 
                                                               param_df, 
                                                               index)
            elif intervention_type  == 'Vaccine':
                cov_pop_burden_df = apply_endemicity_threshold(cov_pop_burden_df, 
                                                               param_df, 
                                                               index)
            cov_pop_burden_df = apply_intervention_cut(cov_pop_burden_df, 
                                                       param_df, 
                                                       index)
            cov_pop_burden_dict[code][index] = cov_pop_burden_df
    return cov_pop_burden_dict




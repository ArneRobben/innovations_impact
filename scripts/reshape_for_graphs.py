# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 11:38:34 2019

@author: LaurencT
"""

import pandas as pd
import numpy as np
import re
from restructure_cov_pop_burden import aggregate_burden_df

def isolate_deterministic_rows(param_df):
    """Filter dfs by whether the index is text (signifying a deterministic scenario)
       Inputs:
           param_df - a df of paramters where the indexes are strings for 
               deterministic scenarios and numbers for probabilistic trials
       Returns:
           a filtered df where all the data is for deterministic scenarios
    """
    param_determ_df = param_df.loc[param_df.index.str.contains('[a-z]', na=False)].copy()
    return param_determ_df

def isolate_probabilistic_rows(param_df, analysis_type):
    """Filter dfs by whether the index is a number in the range 1:num_trials
       (signifying a probabilistic trial)
       Inputs:
           param_df - a df of paramters where the indexes are strings for 
               deterministic scenarios and numbers for probabilistic trials
       Returns:
           a filtered df where all the data is for probabilistic trials      
    """
    num_trials = analysis_type['num_trials']
    param_prob_df = param_df.loc[param_df.index.isin(range(1,num_trials+1))].copy()
    return param_prob_df

def separate_param_dict(param_dict, analysis_type):
    """Creates a dictionary of two dictionaries, one for deterministic scenarios
       and another for probabilistic trials
       Inputs:
           param_dict - a dict where keys are id_codes and values are dfs
       Returns:
           a dict of two dicts 
    """
    deterministic_dict = {k: isolate_deterministic_rows(v) 
                          for k, v in param_dict.items()}
    probabilistic_dict = {k: isolate_probabilistic_rows(v, analysis_type) 
                          for k, v in param_dict.items()}
    return {'det': deterministic_dict, 'prob': probabilistic_dict}

# Section 12:

def restructure_graph_data_deterministic(param_df, variable):
    """Restructures data to the form needed by the tornado diagram
       Inputs:
           param_df - a df that must have indexes representing the different
               deterministic scenarios being modelled
           variable - str - the name of the variable to be plotted on the x axis
               normally either lives touched or lives improved
       Returns:
           base - float -  the base case estimate for the variable
           graph_data - a df in the format needed for tornado_matplotlib()
    """
    base = param_df.loc['base', variable]
    lower_series = param_df[param_df.index.str.contains('lower')][variable]
    lower_series.name = 'lower'
    lower_series.index = [re.sub('_lower', '', index) 
                          for index in lower_series.index]
    upper_series = param_df[param_df.index.str.contains('upper')][variable]
    upper_series.name = 'upper'
    upper_series.index = [re.sub('_upper', '', index) 
                          for index in upper_series.index]
    graph_data = pd.concat([upper_series, lower_series], axis = 1)
    graph_data['ranges'] = abs(graph_data['upper'] - graph_data['lower'])
    graph_data['variables'] = graph_data.index
    graph_data = graph_data[graph_data['ranges']>0]
    graph_data.index = range(len(graph_data.index))
    return base, graph_data


def vaccine_bridge_adjustments(cov_pop_burden_df, intervention_cut):
    """Calculates the remainder of people included at each model stage and returns
       the stage names and that remainder
    """
    cov_pop_burden_df['birth_cohort_endem'] = np.where(cov_pop_burden_df['coverage']>0.01,
                                                 cov_pop_burden_df['target_pop'],
                                                 0)
    cov_pop_burden_df['lives_touched_base'] = (
            cov_pop_burden_df['birth_cohort_endem']
            *cov_pop_burden_df['prob_cover'] * 
            cov_pop_burden_df['coverage'] / 
            intervention_cut)
    # Aggregate the columns to make a relevant for the graph
    stage = ['Total birth cohort', 'Birth cohort in\nendemic countries', 
              'Expected lives touched\nfor a base vaccine\nin endemic countries',
              'Lives touched', 'Lives improved']
    remainder = [cov_pop_burden_df['target_pop'].sum(), 
                 cov_pop_burden_df['birth_cohort_endem'].sum(),
                 cov_pop_burden_df['lives_touched_base'].sum(),
                 cov_pop_burden_df['lives_touched'].sum(),
                 cov_pop_burden_df['lives_improved'].sum()
                ]
    return stage, remainder

def diagnostic_bridge_adjustments(cov_pop_burden_df, intervention_cut):
    """Calculates the remainder of people included at each model stage and returns
       the stage names and that remainder
    """
    stage = ['Total patient pool', 'Relevant patient pool', 
             'Target diagnostic pool',
             'Expected lives touched\nfor a base diagnostic',
             'Lives touched', 'Lives improved']
    remainder = [cov_pop_burden_df['naive_incidence_number'].sum(), 
                 cov_pop_burden_df['incidence_number'].sum(),
                 cov_pop_burden_df['target_pop'].sum(),
                 cov_pop_burden_df['lives_touched_base'].sum(),
                 cov_pop_burden_df['lives_touched'].sum(),
                 cov_pop_burden_df['lives_improved'].sum()]  
    return stage, remainder

def therapeutic_bridge_adjustments(cov_pop_burden_df, intervention_cut):
    """Calculates the remainder of people included at each model stage and returns
       the stage names and that remainder
    """
    stage = ['Total patient pool', 'Relevant patient pool', 
         'Expected lives touched\nfor a base therapeutic',
         'Lives touched', 'Lives improved']
    remainder = [cov_pop_burden_df['naive_incidence_number'].sum(), 
                 cov_pop_burden_df['target_pop'].sum(),
                 cov_pop_burden_df['lives_touched_base'].sum(),
                 cov_pop_burden_df['lives_touched'].sum(),
                 cov_pop_burden_df['lives_improved'].sum()]
    return stage, remainder


def get_bridging_data(base_cov_pop_burden_dict, burden_dict_unadjusted, param_user_dict):
    """Restructures data from various model stages into a form suitable for the bridging chart
       Inputs:
           base_cov_pop_burden_dict - 
           burden_dict_unadjusted - 
           param_user_dict - 
       Returns:
           a dict - keys are strings (id_code), values are dfs with the 
               columns stage, adjustment and remainder
    """
    bridge_data_dict = {}
    for code in base_cov_pop_burden_dict.keys():
        intervention_type = param_user_dict[code]['intervention_type']
        intervention_cut = param_user_dict[code]['intervention_cut_mean']
        burden_df_unadjusted = burden_dict_unadjusted[code]        
        cov_pop_burden_df = base_cov_pop_burden_dict[code]
        # Create these columns because they are the same across modalities
        cov_pop_burden_df['naive_incidence_number'] = (
                aggregate_burden_df(burden_df_unadjusted)['incidence_number_mean']
                                                      )
        cov_pop_burden_df['lives_touched_base'] = (
                cov_pop_burden_df['target_pop'] * 
                cov_pop_burden_df['prob_cover'] * 
                cov_pop_burden_df['coverage'] / 
                intervention_cut)
        
        cov_pop_burden_df['lives_touched'] = (
                cov_pop_burden_df['lives_touched_base'] * 
                intervention_cut)
        
        cov_pop_burden_df['lives_improved'] = (
                cov_pop_burden_df['lives_touched'] * 
                param_user_dict[code]['efficacy_mean'])
        
        if intervention_type == 'Vaccine':
            # Create vaccine specific columns
            stage, remainder = vaccine_bridge_adjustments(cov_pop_burden_df, 
                                                          intervention_cut)
        elif intervention_type in ['Device', 'Rapid diagnostic test']:
            # Aggregate the columns to make a relevant for the graph
            stage, remainder = diagnostic_bridge_adjustments(cov_pop_burden_df, 
                                                             intervention_cut)
        elif re.search('Therapeutic', intervention_type):
            # Aggregate the columns to make a relevant for the graph
            stage, remainder = therapeutic_bridge_adjustments(cov_pop_burden_df, 
                                                              intervention_cut)
        else:
            raise ValueError(intervention_type+' is not a recognised intervention_type')
        
        # Calculate the adjustment based on the remainder at each stage
        adjustment = [0]+[remainder[i]-remainder[i+1] for i in range(len(remainder)-1)]
        bridge_graph_df = pd.DataFrame({'stage':stage, 
                                        'adjustment':adjustment, 
                                        'remainder':remainder})
        bridge_data_dict[code] = bridge_graph_df
    return(bridge_data_dict)

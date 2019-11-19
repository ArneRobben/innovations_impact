# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 15:15:17 2019

@author: LaurencT
"""

def calculate_lives_touched(cov_pop_burden_df):
    """Creates a column for lives touched by multiplying relevant columns
       Inputs:
           cov_pop_burden_df - a df that must contain columns 'target_pop', 
               'coverage' and 'prob_cover'
       Returns:
           a df with a lives_touched column
    """
    cov_pop_burden_df['lives_touched'] = (cov_pop_burden_df['target_pop']*
                                          cov_pop_burden_df['coverage']*
                                          cov_pop_burden_df['prob_cover'])
    return cov_pop_burden_df

def update_lives_touched(cov_pop_burden_dict, param_dict):
    """Updates lives_touched figures for each of the scenarios in the param_dfs
       Inputs:
           cov_pop_burden_dict - a nested dict where keys are id_codes and then
               scenarios and the values are dfs
           param_dict - a dict where the keys are id_codes and the values are dfs
               with a row of parameters for each of the scenarios
       Returns:
           param_dict with updated lives_touched figures for each of the rows
               of the dfs (the values of the dict)
    """
    for code in param_dict.keys():
        param_df = param_dict[code]
        for index in param_df.index.tolist():
            cov_pop_burden_df = cov_pop_burden_dict[code][index].copy()
            cov_pop_burden_df = calculate_lives_touched(cov_pop_burden_df)
            lives_touched = cov_pop_burden_df['lives_touched'].sum()
            param_df.loc[index, 'lives_touched'] = lives_touched
        param_dict[code] = param_df
    return param_dict
    
def calculate_lives_improved(param_df):
    """Create a new column for lives improved in a df
       Inputs:
           param_df - a df of parameters must contain columns: 'lives_touched' and 'efficacy'
       Returns:
           a df with a lives_improved column added
    """
    param_df['lives_improved'] = (param_df['lives_touched']*param_df['efficacy'])
    return param_df

def update_lives_improved(param_dict):
    """Update dfs in param_dict with lives improved columns
       Inputs:
           param_dict - a dict where keys are id_codes and the values are 
               dfs of parameters
       Returns:
           a dict of dfs with a lives improved column added for each of them
    """
    param_dict = {k: calculate_lives_improved(v) for k, v in param_dict.items()}
    return param_dict
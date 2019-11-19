# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 08:53:02 2019

@author: LaurencT
"""

import numpy as np
import pandas as pd

def clear_exceptions(estimates_output, param_dict):
    """Clears exceptions column of estimates_output df for all models that are 
       going to be run (this is needed because the str defining exceptions get 
       concatenated)
    """
    estimates_output = estimates_output.copy()
    for code in param_dict.keys():
        estimates_output.loc[code, 'exception_count'] = 0
        estimates_output.loc[code, 'exception_comment'] = '.'
    return estimates_output

def update_exceptions(estimates_output, code, new_comment):
    """Updates exception_count and exception comment columns of param_user for
       appropriate codes
       Inputs:
           estimates_output - a df of parameters and estimates
           code - a str - id_code in the form "dddddd000d[A-Z]"
           new_comment - a str - explaining the nature of the exception
       Returns:
           param_user_all - a df of parameters and estimates (with updated 
               exception details)
    """
    estimates_output.loc[code, 'exception_count'] += 1
    estimates_output.loc[code, 'exception_comment'] += (new_comment+' ')
    return(estimates_output)

def apply_geography_exceptions(cov_pop_burden_dict, estimates_output):
    """Applying exceptions where target_pop is non-zero only in certain geographies
       Inputs:
           cov_pop_burden_dict - a nested dict where keys are id_codes and then
               scenarios and the values are dfs
           estimates_output - a df of parameters and estimates
       Returns: 
           cov_pop_burden_dict - updated with geographical expections applied
    """
    for code in cov_pop_burden_dict.keys():
        cov_pop_burden_trials = cov_pop_burden_dict[code]
        if code == '2123460002B':
            # Regional exception based on assumed product profile
            super_regions = ['South Asia']
            for index in cov_pop_burden_trials.keys():
                cov_pop_burden_df = cov_pop_burden_trials[index]
                cov_pop_burden_df['target_pop'] = (
                        np.where(np.isin(cov_pop_burden_df['super_region'], super_regions),
                                 cov_pop_burden_df['target_pop'],
                                 0)
                                                  )
            new_comment = 'Only allowed for coverage in South Asia as in the TPP'
            estimates_output= update_exceptions(estimates_output, code, new_comment)
        elif code == '2123460002C':
            # Regional exception based on assumed product profile
            super_regions = ["Sub-Saharan Africa"]
            for index in cov_pop_burden_trials.keys():
                cov_pop_burden_df = cov_pop_burden_trials[index]
                cov_pop_burden_df['target_pop'] = (
                        np.where(np.isin(cov_pop_burden_df['super_region'], super_regions),
                                 cov_pop_burden_df['target_pop'],
                                 0)
                                                  )
            new_comment = 'Only allowed for coverage in Africa as in the TPP.'
            estimates_output = update_exceptions(estimates_output, code, new_comment)
        else:
            # As there is no expceptions relevant here
            pass
    return(estimates_output, cov_pop_burden_dict)
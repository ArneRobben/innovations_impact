# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 09:28:07 2019

@author: LaurencT
"""
# select_columns_burden test

import pytest
import pandas as pd
from scipy.stats import norm
import numpy as np
import re

np.random.seed(1)

def select_columns_burden(burden_df, index):
    """This probabilistically varies columns by GBD SDs, selects the correct deterministic
       columns and subsets and renames the columns
    """
    new_burden_df = burden_df.copy()
    # Create column roots e.g. DALY_rate
    column_roots = [re.sub('_mean', '', column) for column in list(new_burden_df) if re.search('mean', column)]
    # Vary relevant column deterministically or probabilistically based on its root
    for root in column_roots:    
        try:
            int(index)
            new_burden_df[root + '_mean'] = norm.rvs(loc = new_burden_df[root + '_mean'],
                     scale = (new_burden_df[root + '_mean']-new_burden_df[root + '_lower'])/3)
            new_burden_df[root + '_mean'] = np.where(new_burden_df[root + '_mean'] <0, 
                                            0, new_burden_df[root + '_mean'])
        except ValueError:
            if index == 'burden_lower':
                new_burden_df[root + '_mean'] = new_burden_df[root + '_lower']
            elif index == 'burden_upper':
                new_burden_df[root + '_mean'] = new_burden_df[root + '_upper']
        
    # Remove upper and lower columns as the correct column is the the mean
    relevant_columns = [column for column in list(new_burden_df) if not re.search('upper|lower', column)]
    new_burden_df = new_burden_df[relevant_columns]
    # Create new column name mapping
    new_column_names_dict = {column: re.sub('_mean', '', column) for column in relevant_columns}  
    # Rename columns
    new_burden_df = new_burden_df.rename(columns = new_column_names_dict)
    return new_burden_df

burden_df = pd.DataFrame({'DALYs_Number_mean':[1000,2000,3000],
                          'DALYs_Number_lower':[500,1500,2500],
                          'DALYs_Number_upper':[1500,2500,3500],
                          'DALYs_Rate_mean':[10,20,30],
                          'DALYs_Rate_lower':[5,15,25],
                          'DALYs_Rate_upper':[15,25,35]},
                        index = ['UK', 'US', 'China'] )

index_test_1 = 'base'

expected_burden_df_test_1 = pd.DataFrame({'DALYs_Number':[1000,2000,3000],
                                   'DALYs_Rate':[10,20,30]},
                                  index = ['UK', 'US', 'China'])

def test_select_columns_burden_1():
    assert(select_columns_burden(burden_df, index_test_1).equals(expected_burden_df_test_1))

index_test_2 = 'burden_lower'

expected_burden_df_test_2 = pd.DataFrame({'DALYs_Number':[500,1500,2500],
                                   'DALYs_Rate':[5,15,25]},
                                  index = ['UK', 'US', 'China'])

def test_select_columns_burden_2():
    assert(select_columns_burden(burden_df, index_test_2).equals(expected_burden_df_test_2))

index_test_3 = 'burden_upper'

expected_burden_df_test_3 = pd.DataFrame({'DALYs_Number':[1500,2500,3500],
                                   'DALYs_Rate':[15,25,35]},
                                  index = ['UK', 'US', 'China'])

def test_select_columns_burden_3():
    assert(select_columns_burden(burden_df, index_test_3).equals(expected_burden_df_test_3))

index_test_4 = '1'

expected_burden_df_test_4 = pd.DataFrame({'DALYs_Number':[1270.724227, 1898.040598, 2911.971375],
                                   'DALYs_Rate':[8.211719, 21.442346, 26.164102]},
                                  index = ['UK', 'US', 'China'])

def test_select_columns_burden_4():
    assert((select_columns_burden(burden_df, index_test_4) - expected_burden_df_test_4).values.sum() < 1)

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 09:28:07 2019

@author: LaurencT
"""
# select_columns_burden test

import os
os.chdir('C:/Users/laurenct/OneDrive - Wellcome Cloud/My Documents/python/lives_touched_lives_improved/scripts')

#

import pytest
import pandas as pd
from scipy.stats import norm
import numpy as np
import re
from lives_touched_lives_improved import select_columns_burden

np.random.seed(1)

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

test_select_columns_burden_1()

index_test_2 = 'burden_lower'

expected_burden_df_test_2 = pd.DataFrame({'DALYs_Number':[500,1500,2500],
                                   'DALYs_Rate':[5,15,25]},
                                  index = ['UK', 'US', 'China'])

def test_select_columns_burden_2():
    assert(select_columns_burden(burden_df, index_test_2).equals(expected_burden_df_test_2))

test_select_columns_burden_2()

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

test_select_columns_burden_3()

def test_select_columns_burden_4():
    assert((select_columns_burden(burden_df, index_test_4) - expected_burden_df_test_4).values.sum() < 1)

test_select_columns_burden_4()
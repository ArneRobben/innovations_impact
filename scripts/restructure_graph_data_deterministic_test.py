# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 14:59:06 2019

@author: LaurencT
"""

import os
os.chdir('C:/Users/laurenct/OneDrive - Wellcome Cloud/My Documents/python/lives_touched_lives_improved/scripts')

#

import pytest
import pandas as pd
from scipy.stats import norm
import numpy as np
import re
from lives_touched_lives_improved import restructure_graph_data_deterministic
import matplotlib.pyplot as plt

# Test 1: lower is lower than upper 
param_df_test_1 = pd.DataFrame({'lives_improved':[4000,6000,2000, 8000, 1000],
                         'lives_touched':[5000,7500,2500, 10000, 1250]},
                        index = ['base', 'population_upper', 'population_lower', 'coverage_upper', 'coverage_lower'] )

variable_test_1 = 'lives_touched'

test_1_results = restructure_graph_data_deterministic(param_df_test_1 , variable_test_1)

test_1_assumed_results = (5000, pd.DataFrame({'upper': [7500, 10000],
                                              'lower': [2500, 1250],
                                              'ranges': [5000, 8750],
                                              'variables':['population', 'coverage']}))

def test_restructure_graph_data_deterministic_1():
    assert(test_1_results[1].equals(test_1_assumed_results[1]))
    assert(test_1_results[0] == test_1_assumed_results[0])

test_restructure_graph_data_deterministic_1()

# Test 2: lower is higher than upper for population
param_df_test_2 = pd.DataFrame({'lives_improved':[4000, 2000, 6000, 8000, 1000],
                         'lives_touched':[5000, 2500, 7500, 10000, 1250]},
                        index = ['base', 'population_upper', 'population_lower', 'coverage_upper', 'coverage_lower'] )

variable_test_2 = 'lives_touched'

test_2_results = restructure_graph_data_deterministic(param_df_test_2 , variable_test_2)

test_2_assumed_results = (5000, pd.DataFrame({'upper': [2500, 10000],
                                              'lower': [7500, 1250],
                                              'ranges': [5000, 8750],
                                              'variables':['population', 'coverage']}))

def test_restructure_graph_data_deterministic_2():
    assert(test_2_results[1].equals(test_2_assumed_results[1]))
    assert(test_2_results[0] == test_2_assumed_results[0])

test_restructure_graph_data_deterministic_2()

# Test 3: variable takes a different value
param_df_test_3 = pd.DataFrame({'lives_improved':[4000,6000,2000, 8000, 1000],
                                'lives_touched':[5000,7500,2500, 10000, 1250]},
                        index = ['base', 'population_upper', 'population_lower', 'coverage_upper', 'coverage_lower'] )

variable_test_3 = 'lives_improved'

test_3_results = restructure_graph_data_deterministic(param_df_test_3 , variable_test_3)

test_3_assumed_results = (4000, pd.DataFrame({'upper': [6000, 8000],
                                              'lower': [2000, 1000],
                                              'ranges': [4000, 7000],
                                              'variables':['population', 'coverage']}))

def test_restructure_graph_data_deterministic_3():
    assert(test_3_results[1].equals(test_3_assumed_results[1]))
    assert(test_3_results[0] == test_3_assumed_results[0])

test_restructure_graph_data_deterministic_3()

# Check the way the graph looks to confirm

graph_data = test_2_assumed_results[1]
base = test_2_assumed_results[0]

def tornado_matplotlib(graph_data, base):
    """Creates a tornado diagram and saves it to a prespecified directory
       Inputs:
           graphs_data - a df which must contain the columns 'variables' for 
               the names of the variables being graphed, 'lower' for the lower
               bounds of the deterministic sensitivity and 'ranges' for the total
               range between the lower and upper
           base - a float with the base case value
           directory - str - a path to a directory where the plot should be saved
           file_name - str- a useful name by which the plot with be saved
       Exports:
           A chart to the prespecified directory
    """
    # Sort the graph data so the widest range is at the top and reindex
    graph_data.copy()
    graph_data = graph_data.sort_values('ranges', ascending = False)
    graph_data.index = list(range(len(graph_data.index)))[::-1]

    # The actual drawing part
    fig = plt.figure()
    
    # Plot the bars, one by one
    for index in graph_data.index:
        # The width of the 'low' and 'high' pieces
        
        # If to ensure visualisation is resilient to lower value of parameter
        # leading to higher estimate of variable
        if graph_data.loc[index, 'upper']>graph_data.loc[index, 'lower']:
            low = graph_data.loc[index, 'lower']
            face_colours = ['red', 'green']
        else:
            low = graph_data.loc[index, 'upper']
            face_colours = ['green', 'red']
        value = graph_data.loc[index, 'ranges']
        low_width = base - low
        high_width = low + value - base
    
        # Each bar is a "broken" horizontal bar chart
        plt.broken_barh(
            [(low, low_width), (base, high_width)],
            (index - 0.4, 0.8),
            facecolors= face_colours,  # Try different colors if you like
            edgecolors=['black', 'black'],
            linewidth=1,
        )
    
    # Draw a vertical line down the middle
    plt.axvline(base, color='black', linestyle='dashed')
    
    # Position the x-axis and hide unnecessary axes
    ax = plt.gca()  # (gca = get current axes)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    
    # Make the y-axis display the variables
    plt.yticks(graph_data.index.tolist(), graph_data['variables'])
    
    # Set the portion of the x- and y-axes to show
    plt.xlim(left = 0)
    plt.ylim(-1, len(graph_data.index))
    
    # Stop scientific formats
    ax.get_xaxis().get_major_formatter().set_scientific(False)
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    fig.autofmt_xdate()
    
    # Change axis text format
    plt.rcParams.update({'font.size': 14})
    plt.rcParams.update({'font.family': 'calibri'})
    
    # Make it tight format 
    plt.tight_layout()
    plt.show()
    plt.close(fig=None)

tornado_matplotlib(graph_data, base)

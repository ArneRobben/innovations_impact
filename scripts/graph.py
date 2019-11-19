# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 11:12:46 2019

@author: LaurencT
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re
import os
import numpy as np

def tornado_matplotlib(graph_data, base, directory, file_name, variable):
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
    
    # Set axis labels
    label = re.sub('_', ' ', variable).title()
    plt.xlabel(label)
    plt.ylabel('Model Parameters')
    
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
    plt.rcParams.update({'font.size': 12})
    plt.rcParams.update({'font.family': 'calibri'})
    
    # Make it tight format 
    plt.tight_layout()
    
    # Set up export path and export the chart
    path = os.path.join(directory, file_name+'.png')
    fig.savefig(path)
    print('Please find the chart at', path)
    plt.close(fig=None)

def probability_histogram(graph_data, variable, directory, file_name):
    """Creates a probability histogram and exports it to a directory
       Inputs:
           graph_data - df 
           variable - str - the variable in graph_data to be histogrammed
           directory - str - a path to a directory where the plot should be saved
           file_name - str- a useful name by which the plot with be saved
       Exports:
           A chart to the prespecified directory
    """
    # Get the right series and make it np
    graph_data = graph_data[variable]
    
    # Calculate confidence intervals
    upper = graph_data.quantile(0.975)
    lower = graph_data.quantile(0.025)
    # Histogram:
    # Bin it
    n, bin_edges = np.histogram(graph_data, 30)
    # Normalize it, so that every bins value gives the probability of that bin
    bin_probability = n/float(n.sum()) 
    # Get the mid points of every bin
    bin_middles = (bin_edges[1:]+bin_edges[:-1])/2.
    # Compute the bin-width
    bin_width = bin_edges[1]-bin_edges[0]
    # Plot the histogram as a bar plot
    fig, ax = plt.subplots()
    plt.bar(bin_middles, bin_probability, width=bin_width, color = '#003170')
    
    # Add 95% confidence interval lines
    plt.axvline(upper, color='black', linestyle='dashed')
    plt.axvline(lower, color='black', linestyle='dashed')
    
    # Add 95% confidencei interval labels - REMOVED BECAUSE IT OVERLAPPED WITH BARS / LINES
    # max_prob = max(bin_probability)
    # plt.text((upper+lower/8), max_prob*2/3, 'Upper (97.5%)', rotation = 90)
    # plt.text((lower-lower/5), max_prob*2/3, 'Lower (2.5%)', rotation = 90)
    
    # Axis labels
    label = re.sub('_', ' ', variable).title()
    plt.xlabel(label)
    plt.ylabel('Probability')
    
    # Set the portion of the x- and y-axes to show
    plt.xlim(left = 0)
    
    # Stop axis scientific formats
    ax.get_xaxis().get_major_formatter().set_scientific(False)
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    ax.get_yaxis().get_major_formatter().set_scientific(False)
    fig.autofmt_xdate()
    
    # Change axis text format
    plt.rcParams.update({'font.size': 12})
    plt.rcParams.update({'font.family': 'calibri'})
    
    # Make it tight format 
    plt.tight_layout()
    
    # Export
    path = os.path.join(directory, file_name+'.png')   
    plt.savefig(path)
    print('Please find the chart at', path)
    plt.close(fig=None)
    
def bridge_plot(graph_data, directory, file_name):
    """Plots a bridge bar chart and exports it to a directory
       Inputs:
           graph_data - df with the columns stage, adjustment and remainder
           directory - str - a path to a directory where the plot should be saved
           file_name - str- a useful name by which the plot with be saved
       Exports:
           A chart to the prespecified directory
    """
    ind = np.arange(len(graph_data.index))    # the x locations for the groups
    width_remainder = 0.4       # the width of the bars: can also be len(x) sequence
    width_adjustment = 0.2
    
    fig, ax = plt.subplots()
    
    p1 = plt.bar(ind, graph_data['remainder'], width_remainder, color = '#003170')
    p2 = plt.bar(ind, graph_data['adjustment'], width_adjustment,
                 bottom=graph_data['remainder'])
    connector = plt.plot(graph_data['remainder'], marker='o', color='k')
       
    # plt.ylabel('Number of people')
    # plt.title('Number of people at different model stages')
    plt.xticks(ind, graph_data['stage'])
    plt.legend((p1[0], p2[0]), ('Remainder', 'Adjustment'))
    
    # Stop axis scientific formats
    ax.get_yaxis().get_major_formatter().set_scientific(False)
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    fig.autofmt_xdate()
    
    # Change axis text format
    plt.rcParams.update({'font.size': 12})
    plt.rcParams.update({'font.family': 'calibri'})
    
    # Make it tight format 
    plt.tight_layout()
    
    # Export
    path = os.path.join(directory, file_name+'.png')
    plt.savefig(path)
    print('Please find the chart at', path)
    plt.close(fig=None)

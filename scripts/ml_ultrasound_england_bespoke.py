# -*- coding: utf-8 -*-
"""
Created on Tue May 28 08:30:34 2019

@author: LaurencT
"""

# Machine learning ultrasound bespoke LTLI

import pandas as pd
import numpy as np
import re
import os
from scipy.stats import norm
from scipy.stats import gamma
from scipy.stats import beta
os.chdir('C:/Users/laurenct/OneDrive - Wellcome Cloud/My Documents/python/lives_touched_lives_improved/scripts')
directory = 'C:/Users/laurenct/OneDrive - Wellcome Cloud/My Documents/python/lives_touched_lives_improved/graphs/'

"""CAUTION
"""
##### ALL OF THESE NEED UPDATING TO WHERE THESE FUNCTIONS ARE NOW - THE STRUCTURE OF THE REPOSITORY CHANGED#####
from lives_touched_lives_improved import bridge_plot
from lives_touched_lives_improved import tornado_matplotlib
from lives_touched_lives_improved import restructure_graph_data_deterministic
from lives_touched_lives_improved import beta_moments
from lives_touched_lives_improved import gamma_moments
from lives_touched_lives_improved import probability_histogram
from lives_touched_lives_improved import graphs_to_slides

"""CAUTION
"""

# https://www.gov.uk/government/publications/nhs-screening-programmes-kpi-reports-2017-to-2018
ultrasound_data = pd.read_csv('C:/Users/laurenct/OneDrive - Wellcome Cloud/My Documents/python/lives_touched_lives_improved/data/ultrasound_screening_programme_england.csv')

# http://www.eurocat-network.eu/accessprevalencedata/prevalencetables
abnormalities_data = pd.read_csv('C:/Users/laurenct/OneDrive - Wellcome Cloud/My Documents/python/lives_touched_lives_improved/data/prevalence_foetal_abnormalities.csv',
                                 encoding = "ISO-8859-1")
# Ratio of babies in UK to just England: https://www.ons.gov.uk/peoplepopulationandcommunity/birthsdeathsandmarriages/livebirths/datasets/birthsbyareaofusualresidenceofmotheruk
adjustment_for_uk = 1.168

abnormalities_data = abnormalities_data[abnormalities_data['Relevant']=='Yes']
abnormalities_data['Total Prevalence'] = pd.to_numeric(abnormalities_data['Total Prevalence'])
cardiac_abnormalities = ['Transposition of great vessels', 'Atrioventricular septal defect', 'Tetralogy of Fallot', 'Hypoplastic left heart']
cardiac_abnormalities_data = abnormalities_data[abnormalities_data['Anomaly'].isin(cardiac_abnormalities)]

incidence = abnormalities_data['Total Prevalence'].sum()/10000
cardiac_incidence = cardiac_abnormalities_data['Total Prevalence'].sum()/10000

intervention_cut = 0.3
current_expected_sensitivity = (abnormalities_data['Current Detection Rate'] * abnormalities_data['Total Prevalence']).sum() / abnormalities_data['Total Prevalence'].sum()
cardiac_expected_sensitivity = (cardiac_abnormalities_data['Current Detection Rate'] * cardiac_abnormalities_data['Total Prevalence']).sum() / cardiac_abnormalities_data['Total Prevalence'].sum()
current_sensitivity = current_expected_sensitivity - 0.1
current_cardiac_sensitivity = cardiac_expected_sensitivity - 0.1
sensitivity = current_sensitivity + 0.2
cardiac_sensitivity = current_cardiac_sensitivity + 0.2
sensitivity_gain = sensitivity - current_sensitivity
cardiac_sensitivity_gain = cardiac_sensitivity - current_cardiac_sensitivity
specificity = 0.99
accuracy_rate = (1 - incidence)*specificity + incidence*sensitivity

# Remove irrelevant columns pandas added in
ultrasound_data = ultrasound_data.iloc[:,0:7]

# Trim the column names 
ultrasound_data.columns = [column.strip() for column in ultrasound_data.columns]

# Clean the numerical series to numbers
ultrasound_data = ultrasound_data.replace('One or more return missing', 'NaN')
ultrasound_data = ultrasound_data.replace('No returns', 'NaN')
ultrasound_data = ultrasound_data.apply(lambda x: x.str.replace(',', ''))
ultrasound_data.loc[:,['Numerator', 'Denominator', 'Performance (%)']] = ultrasound_data.loc[:,['Numerator', 'Denominator', 'Performance (%)']].apply(lambda x: pd.to_numeric(x, errors = 'coerce'))

# Replace nans with the average
average_numerator = ultrasound_data['Numerator'].mean()
average_denominator = ultrasound_data['Denominator'].mean()

ultrasound_data['Numerator'] = ultrasound_data['Numerator'].fillna(value = average_numerator)
ultrasound_data['Denominator'] = ultrasound_data['Denominator'].fillna(value = average_denominator)

# Calculate target pop and coverage of ultrasound in general
ultrasound_data['Performance (%)'] = ultrasound_data['Numerator']/ultrasound_data['Denominator']

numerator_sum = ultrasound_data['Numerator'].sum()
denominator_sum = ultrasound_data['Denominator'].sum()
coverage = numerator_sum / denominator_sum

uk_denominator_sum = denominator_sum*adjustment_for_uk
uk_numerator_sum = numerator_sum*adjustment_for_uk

# Bridging plot
lives_touched = uk_numerator_sum*intervention_cut
lives_improved = lives_touched*incidence*sensitivity_gain
cardiac_lives_improved = lives_touched*cardiac_incidence*cardiac_sensitivity_gain

stage = ['Pregnancies', 'Target pregnancies', 'Pregnancies receiving ultrasound', 'Lives touched', 'Lives improved']
remainder = [uk_denominator_sum, uk_denominator_sum, uk_numerator_sum , lives_touched, lives_improved]
adjustment = [0, 0, uk_denominator_sum-uk_numerator_sum , uk_numerator_sum - lives_touched, lives_touched - lives_improved]

cardiac_remainder = [uk_denominator_sum, uk_denominator_sum, uk_denominator_sum*coverage, lives_touched, cardiac_lives_improved]
cardiac_adjustment = [0, 0, uk_denominator_sum*(1-coverage), uk_denominator_sum*coverage - lives_touched, lives_touched - cardiac_lives_improved]

# Set up deterministic scenarios for analysis

scenarios = ['base',
             'pregnancies_lower',
             'pregnancies_upper',
             'coverage_lower',
             'coverage_upper',
             'intervention_cut_lower',
             'intervention_cut_upper',
             'sensitivity_lower',
             'sensitivity_upper',
             'specificity_lower',
             'specificity_upper',
             'incidence_lower',
             'incidence_upper']

scenarios_changes = {'base' : 1,
                     'pregnancies_lower' : uk_denominator_sum*0.85,
                     'pregnancies_upper' : uk_denominator_sum*1.15,
                     'coverage_lower' : coverage-0.04,
                     'coverage_upper' : coverage+0.005,
                     'intervention_cut_lower' : intervention_cut - 0.2,
                     'intervention_cut_upper' : intervention_cut + 0.2,
                     'sensitivity_lower' : sensitivity-0.1,
                     'sensitivity_upper' : sensitivity+0.1,
                     'specificity_lower' : specificity-0.01,
                     'specificity_upper' : specificity+0.005,
                     'incidence_lower' : incidence-0.001,
                     'incidence_upper' : incidence+0.001}

scenario_params_base = {'pregnancies': uk_denominator_sum,
                        'coverage': coverage,
                        'intervention_cut': intervention_cut,
                        'sensitivity': sensitivity,
                        'specificity': specificity,
                        'incidence': incidence}

def calculate_scenarios(scenarios, scenarios_changes, scenario_params_base, current_sensitivity):    
    # Turn the parameters into scenarios
    scenario_params_base = pd.Series(scenario_params_base)
    
    scenario_params_unadjusted = [scenario_params_base for scen in scenarios]
    
    scenario_params_df = pd.concat(scenario_params_unadjusted, axis = 1).transpose()
    scenario_params_df.index = scenarios
    
    # Ignore base, but replaces the parameters where relevant for the scenarios
    for scen in scenarios[1:]:
        root = re.sub('_lower|_upper', '', scen)
        scenario_params_df.loc[scen, root] = scenarios_changes[scen]
    
    # Calculate LT and LI
    scenario_params_df['accuracy_rate'] = (1 - scenario_params_df['incidence'])*scenario_params_df['specificity'] + scenario_params_df['incidence']*scenario_params_df['sensitivity']
    scenario_params_df['lives_touched'] = scenario_params_df['pregnancies']*scenario_params_df['coverage']*scenario_params_df['intervention_cut']
    scenario_params_df['sensitivity_gain'] = scenario_params_df['sensitivity']-current_sensitivity
    scenario_params_df['lives_improved'] = scenario_params_df['lives_touched']*scenario_params_df['incidence']*scenario_params_df['sensitivity_gain']
    
    return scenario_params_df


scenario_params_df = calculate_scenarios(scenarios, scenarios_changes, scenario_params_base, current_sensitivity)

cardiac_scenarios_changes = {'base' : 1,
                             'pregnancies_lower' : uk_denominator_sum*0.85,
                             'pregnancies_upper' : uk_denominator_sum*1.15,
                             'coverage_lower' : coverage-0.04,
                             'coverage_upper' : coverage+0.005,
                             'intervention_cut_lower' : intervention_cut - 0.2,
                             'intervention_cut_upper' : intervention_cut + 0.2,
                             'sensitivity_lower' : cardiac_sensitivity-0.1,
                             'sensitivity_upper' : cardiac_sensitivity+0.1,
                             'specificity_lower' : specificity-0.01,
                             'specificity_upper' : specificity+0.005,
                             'incidence_lower' : cardiac_incidence-0.0005,
                             'incidence_upper' : cardiac_incidence+0.0005}

cardiac_scenario_params_base = {'pregnancies': uk_denominator_sum,
                                'coverage': coverage,
                                'intervention_cut': intervention_cut,
                                'sensitivity': cardiac_sensitivity,
                                'specificity': specificity,
                                'incidence': cardiac_incidence}

cardiac_scenario_params_df = calculate_scenarios(scenarios, 
                                                 cardiac_scenarios_changes, 
                                                 cardiac_scenario_params_base,
                                                 current_cardiac_sensitivity)

# Probabilistic scenarios
def calculate_param_range(scenarios_changes):
    param_ranges = {}
    for scen in scenarios_changes.keys():
        if scen == 'base':
            pass
        else:
            root = re.sub('_lower|_upper', '', scen)
            root_range = scenarios_changes[root+'_upper'] - scenarios_changes[root+'_lower']
            param_ranges[root] = root_range
    return param_ranges


def create_prob_df(param_prob, new_columns, num_trials):
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
    param_example = param_prob
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
        sd = float(param_example[column+'_sd'])
        if sd == 0:
            data = np.array([mean for i in range(1,num_trials+1)])
        # Use normal for things that vary around 1, inflation factor will need
        # changing probaby #~
        elif column in ['pregnancies']:    
            data = norm.rvs(size = num_trials, loc = mean, scale = sd)
        # Use beta distribution for all paramters that are a proportion
        elif column in ['coverage', 'cardiac_incidence', 'incidence', 'intervention_cut',
                        'cardiac_sensitivity', 'sensitivity', 'specificity']:
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
        # Turn the relevant new data into a series (which becomes a column)
        new_column = pd.Series(data, index = range(1, num_trials + 1), name = column)
        prob_df = pd.concat([prob_df, new_column.T], axis = 1, sort = False)
        # Drop unnecessary columns
        columns_to_drop = [name for name in list(prob_df) if re.search("mean|SD", name)]
        prob_df = prob_df.drop(columns_to_drop, axis =1)
    return prob_df

scenarios_changes_all = {'base' : 1,
                         'pregnancies_lower' : uk_denominator_sum*0.85,
                         'pregnancies_upper' : uk_denominator_sum*1.15,
                         'coverage_lower' : coverage-0.04,
                         'coverage_upper' : coverage+0.005,
                         'intervention_cut_lower' : intervention_cut - 0.2,
                         'intervention_cut_upper' : intervention_cut + 0.2,
                         'sensitivity_lower' : sensitivity-0.1,
                         'sensitivity_upper' : sensitivity+0.1,
                         'cardiac_sensitivity_lower' : cardiac_sensitivity-0.1,
                         'cardiac_sensitivity_upper' : cardiac_sensitivity+0.1,
                         'specificity_lower' : specificity-0.01,
                         'specificity_upper' : specificity+0.005,
                         'incidence_lower' : incidence-0.001,
                         'incidence_upper' : incidence+0.001,
                         'cardiac_incidence_lower' : cardiac_incidence-0.001,
                         'cardiac_incidence_upper' : cardiac_incidence+0.001}

scenario_params_base_all = {'pregnancies': uk_denominator_sum,
                            'coverage': coverage,
                            'intervention_cut': intervention_cut,
                            'cardiac_sensitivity': cardiac_sensitivity,
                            'sensitivity': sensitivity,
                            'specificity': specificity,
                            'incidence': incidence,
                            'cardiac_incidence': cardiac_incidence}


scenario_params_mean = {key+'_mean': scenario_params_base_all[key] for key in scenario_params_base_all.keys()}

param_ranges = calculate_param_range(scenarios_changes_all)
param_sds = {root+'_sd': param_ranges[root]/4 for root in param_ranges.keys()}
param_prob = pd.concat([pd.Series(scenario_params_mean), pd.Series(param_sds)])
num_trials = 1000
new_columns = list(scenario_params_base_all.keys())
probabilistic_variables = create_prob_df(param_prob, new_columns, num_trials)

probabilistic_variables['accuracy_rate'] = (1 - probabilistic_variables['incidence'])*probabilistic_variables['specificity'] + probabilistic_variables['incidence']*probabilistic_variables['sensitivity']
probabilistic_variables['lives_touched'] = probabilistic_variables['pregnancies']*probabilistic_variables['coverage']*probabilistic_variables['intervention_cut']
probabilistic_variables['sensitivity_gain'] = probabilistic_variables['sensitivity']-current_sensitivity
probabilistic_variables['cardiac_sensitivity_gain'] = probabilistic_variables['cardiac_sensitivity']-current_cardiac_sensitivity
probabilistic_variables['lives_improved'] = probabilistic_variables['lives_touched']*probabilistic_variables['incidence']*probabilistic_variables['sensitivity_gain']
probabilistic_variables['cardiac_lives_improved'] = probabilistic_variables['lives_touched']*probabilistic_variables['cardiac_incidence']*probabilistic_variables['cardiac_sensitivity_gain']


project_id = '30000020001A'

# Bridge diagram
bridge_data = pd.DataFrame({'stage': stage,
                            'remainder': remainder,
                            'adjustment': adjustment})

filename = project_id+'_bridge_graph'

bridge_plot(bridge_data, directory, filename)

# Deterministics graphs
variable = 'lives_touched'
base, graph_data = restructure_graph_data_deterministic(scenario_params_df, variable)
file_name = project_id+'_deterministic_'+variable
tornado_matplotlib(graph_data, base, directory, file_name, variable)

variable = 'lives_improved'
base, graph_data = restructure_graph_data_deterministic(scenario_params_df, variable)
file_name = project_id+'_deterministic_'+variable
tornado_matplotlib(graph_data, base, directory, file_name, variable)

# Probabilistic graphs
graph_data = probabilistic_variables
variable = 'lives_touched'
file_name = project_id+'_probabilistic_'+variable
probability_histogram(graph_data, variable, directory, file_name)

lives_touched_upper = probabilistic_variables[variable].quantile(0.975)
lives_touched_lower = probabilistic_variables[variable].quantile(0.025)

variable = 'lives_improved'
file_name = project_id+'_probabilistic_'+variable
probability_histogram(graph_data, variable, directory, file_name)

lives_improved_upper = probabilistic_variables[variable].quantile(0.975)
lives_improved_lower = probabilistic_variables[variable].quantile(0.025)

graphs_to_slides(project_id)

project_id = '30000020001B'

# Bridge diagram
cardiac_bridge_data = pd.DataFrame({'stage': stage,
                            'remainder': cardiac_remainder,
                            'adjustment': cardiac_adjustment})

filename = project_id+'_bridge_graph'

bridge_plot(cardiac_bridge_data, directory, filename)

# Deterministics graphs
variable = 'lives_touched'
base, graph_data = restructure_graph_data_deterministic(cardiac_scenario_params_df, variable)
file_name = project_id+'_deterministic_'+variable
tornado_matplotlib(graph_data, base, directory, file_name, variable)

variable = 'lives_improved'
base, graph_data = restructure_graph_data_deterministic(cardiac_scenario_params_df, variable)
file_name = project_id+'_deterministic_'+variable
tornado_matplotlib(graph_data, base, directory, file_name, variable)

# Probabilistic graphs
graph_data = probabilistic_variables
variable = 'lives_touched'
file_name = project_id+'_probabilistic_'+variable
probability_histogram(graph_data, variable, directory, file_name)

graph_data['lives_improved'] = graph_data['cardiac_lives_improved']

variable = 'lives_improved'
file_name = project_id+'_probabilistic_'+variable
probability_histogram(graph_data, variable, directory, file_name)

cardiac_lives_improved_upper = probabilistic_variables['cardiac_'+variable].quantile(0.975)
cardiac_lives_improved_lower = probabilistic_variables['cardiac_'+variable].quantile(0.025)

graphs_to_slides(project_id)


# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 14:52:22 2019

@author: LaurencT
"""

import pandas as pd
import numpy as np
import re
from probability_distribution_moments import gamma_moments_burden
from scipy.stats import gamma
from scipy.stats import norm

def get_relevant_burden(param_dict, burden_all):
    """Return a dict of burden data frames with the relevant conditions isolated
       Inputs:
           param_dict - keys: id_codes, values dfs of parameters for 
               each of the trials / scenarios
           burden_all - a df of burden data, must have the columns cause and age
       Returns:
           a dict - keys: id_codes, values dfs of burden data
    """
    # Create dictionary of lists to filter disease column by
    disease_lists = {k: [param_dict[k]['disease_1'][1],
                         param_dict[k]['disease_2'][1],
                         param_dict[k]['disease_3'][1]]
                     for k in param_dict.keys()}
    # Create dictionary of burden dfs
    burden_dict = {k: burden_all[burden_all['cause'].isin(disease_lists[k])]
                   for k in disease_lists.keys()}
    # Filter based on age
    burden_dict = {k: burden_dict[k][burden_dict[k]['age'] == param_dict[k]['age'][1]]
                   for k in burden_dict.keys()}
    return burden_dict

def select_columns_burden(burden_df, index):
    """This probabilistically varies columns by GBD ranges for probabilistic 
       trials it selects the correct deterministic columns for deterministic 
       trials and subsets and renames the columns.
       Inputs:
           burden_df - a df of burden data with upper, lower and mean estimates
               for at least one burden measure
           index - a string to indicate which deterministic scenario or which
               trial it is
       Returns:
           a df of burden data with just one column for each measure
    """
    new_burden_df = burden_df.copy()
    # Create column roots e.g. DALY_rate
    column_roots = [re.sub('_mean', '', column) 
                    for column in list(new_burden_df) 
                    if re.search('mean', column)]
    # Vary relevant column deterministically or probabilistically based on its root
    for root in column_roots:    
        try:
            int(index)
            #~ changed from normal to gamma
            mean = new_burden_df[root + '_mean']
            sd = (
                    new_burden_df[root + '_mean'] -
                    new_burden_df[root + '_lower']
                 )/2
            prop_lower_mean = (new_burden_df[root + '_mean'] / 
                               new_burden_df[root + '_mean']).mean()
            gamma_vals = pd.DataFrame([gamma_moments_burden(mean_val, sd_val) 
                                       for mean_val, sd_val in zip(mean, sd)])
            new_burden_df[root + '_mean'] = gamma.rvs(a = gamma_vals['shape'], 
                                                      scale = gamma_vals['scale'], 
                                                      size = len(gamma_vals['shape']))
            new_burden_df[root + '_mean'] = new_burden_df[root + '_mean'] * \
                                            norm.rvs(1, (1-prop_lower_mean)/4)
            new_burden_df[root + '_mean'] = np.where(new_burden_df[root + '_mean'] <0, 
                                                     0, new_burden_df[root + '_mean'])
        except ValueError:
            if index == 'burden_lower':
                new_burden_df[root + '_mean'] = new_burden_df[root + '_lower']
            elif index == 'burden_upper':
                new_burden_df[root + '_mean'] = new_burden_df[root + '_upper']
        
    # Remove upper and lower columns as the relevant column is now the mean
    relevant_columns = [column for column in list(new_burden_df) 
                        if not re.search('upper|lower', column)]
    new_burden_df = new_burden_df[relevant_columns]
    # Create new column name mapping
    new_column_names_dict = {column: re.sub('_mean', '', column) 
                             for column in relevant_columns}  
    # Rename columns
    new_burden_df = new_burden_df.rename(columns = new_column_names_dict)
    return new_burden_df

def strain_adjust_burden_df(burden_df, param_df, index):
    """Adjusts a data frame of disease burden downwards in case not all of the
       patients would benefit from the interventions (due to sub disease strains
       or other factors)
       Inputs:
           burden_df - a df with columns of disease burden where the names of 
               all of those columns contain a _ because they are 'measure_metric', 
               there is also a cause column 
           param_dict - a df of parameters with columns for 3 diseases in the form
               disease_[1-3] and with corresponding strain proportions for each of
               those diseases in the form disease_[1-3]_prop
           index - a string to indicate which deterministic scenario or which
               trial it is
       Returns:
           a burden df that has been adjusted for those substrains     
    """
    # Copy the burden_df to avoid side effects
    new_burden_df = burden_df.copy()
    # Create a list of burden columns because they are the form 'measure_metric'
    column_list = [column for column in list(new_burden_df) 
                   if re.search('number|rate', column)]
    # Create a mapping of the disease and proportion
    disease_dict = {k: k+"_prop" for k in ['disease_1', 'disease_2', 'disease_3']}
    # Adjust the df for the substrain prop for each disease
    for column in column_list:
        for disease in disease_dict.keys():
            new_burden_df[column] = np.where(
                    new_burden_df['cause'] == param_df.loc[index, disease], 
                    new_burden_df[column]*param_df.loc[index, disease_dict[disease]], 
                    new_burden_df[column])
    return new_burden_df

def aggregate_burden_df(burden_df):
    """Sums the burden from various conditions targetted by the intervention
       Inputs:
           burden_df - a df with columns for country, age, cause and disease
               burden
       Returns:
           a df where the cause is now renamed to be a list of causes and the
           burden from original causes is now aggregated
    """
    # Create a copy of burden_df to avoid side effects
    new_burden_df = burden_df.copy()    
    # Create list then string of causes
    causes = set(new_burden_df['cause'])
    new_cause_name = ', '.join(causes)
    # Aggregate by age and country
    summed_burden_df = new_burden_df.groupby(['country', 'age']).sum()
    # Turn the indexes back into columns in a df
    index_list = summed_burden_df.index.tolist()
    summed_df = pd.DataFrame(index_list, 
                             columns = ['country', 'age'], 
                             index = index_list)
    # Add the new cause name to the df
    summed_df['cause'] = new_cause_name
    # Merge the burden with the country, age group, and cause columns
    summed_df = pd.concat([summed_df, summed_burden_df], axis = 1)
    # Reindex so the country is the only index
    summed_df.index = [i[0] for i in summed_df.index.tolist()]
    # Merge in region / super region columns
    other_columns = new_burden_df[['country', 'super_region', 'region']].drop_duplicates()
    summed_df = summed_df.merge(other_columns, on = 'country')
    summed_df.index = summed_df['country']
    return summed_df
    
def adjust_burden_dict(burden_dict, param_dict):
    """Adjusts the burden numbers down for substrains
       Inputs:
           burden_dict - a dict - keys are id_codes for the projects
               values are dataframes of all relevant burden for those projects
           param_dict - a dict - keys are id_codes for the projects and values 
               are dfs of parameters for the different scenarios and trials
       Returns:
           a dictionary keys are the id_codes for projects, values are dicts
           each trial key responding to a homogenously proportioned burden df
           with the relevant figure for that project / trial
    """
    burden_dict_new = {}
    # Loop through each of the id_codes 
    for code in param_dict.keys():
        # For each code find the relevant burden data and parameters 
        burden_scenarios_dict = {}
        burden_df = burden_dict[code].copy()
        param_df = param_dict[code].copy()
        # For each trial adjust the burden data to make sure it is relevant
        # for the trial
        for index in param_df.index.tolist():
            burden_df_index = select_columns_burden(burden_df, index)
            burden_df_index = strain_adjust_burden_df(burden_df_index, 
                                                      param_df, 
                                                      index)
            burden_df_index = aggregate_burden_df(burden_df_index)
            burden_scenarios_dict[index] = burden_df_index            
        # Add the new dictionary for the trials to the outer dictionary
        burden_dict_new[code] = burden_scenarios_dict
    return burden_dict_new


def create_coverage_population_dict(coverage, population, param_dict):
    """Creates a dictionary where keys are id_codes and values are dfs
       of relevant coverage / population data for the intervention
       Inputs:
           coverage - a df of coverage data containing the columns 'country' and
               coverage and prob_cover columns for each modality
           population - a df if population data with the columns 'country' and 
               population columns
           param_dict - a df with the column 'intervention_type'
       Returns:
           dict where keys are id_codes and values are dfs of relevant coverage 
               / population data for the intervention        
    """
    cov_pop_dict = {}
    # Loop through the id_codes to be keys in the dictionary
    for code in param_dict.keys():
        new_coverage = coverage.copy()
        population = population.copy()   
        intervention_type = param_dict[code]['intervention_type'][1]
        # Select therapeutic coverage columns if it is a Therapeutic (therapeutic)
        if  intervention_type == 'Therapeutic':
            new_coverage = new_coverage[['country',
                                 'therapeutic_coverage', 
                                 'therapeutic_prob_cover']]
        # Select therapeutic mental health coverage columns if it is a therapeutic
        # for a mental health condition
        elif intervention_type == 'Therapeutic mental health':
            new_coverage = new_coverage[['country', 
                                 'therapeutic_mental_health_coverage', 
                                 'therapeutic_mental_health_prob_cover']]
        # Select vaccine coverage columns if it is a vaccine
        elif intervention_type == 'Vaccine':
            new_coverage = new_coverage[['country', 
                                 'vaccine_coverage', 
                                 'vaccine_prob_cover']]
        # Select RDT coverage columns if it is a RDT 
        elif intervention_type == 'Rapid diagnostic test':
            new_coverage = new_coverage[['country', 
                                 'rapid_diagnostic_test_coverage', 
                                 'rapid_diagnostic_test_prob_cover']]
        # Select device coverage columns if it is a device
        elif intervention_type == 'Device':
            new_coverage = new_coverage[['country', 
                                 'device_coverage', 
                                 'device_prob_cover']]
        else:
            raise ValueError('The value of intervention_type for '+code+' is not valid')
        # Create new column names and rename
        new_column_names = {column: re.sub(('vaccine_|rapid_diagnostic_test_|device_|'
                                           'therapeutic_mental_health_|therapeutic_'), 
                                           '', column) 
                            for column in list(new_coverage)}
        new_coverage = new_coverage.rename(columns = new_column_names)
        # Merge the coverage and population data
        cov_pop_df = pd.concat([new_coverage, population], axis = 1, sort = True)
        cov_pop_dict[code] = cov_pop_df
    return cov_pop_dict

def adjust_cov_pop_df(cov_pop_df, index, param_df):
    """Adjust the coverage and proportion based on the parameters for each scenario
       Inputs:
           cov_pop_df - a df with population columns and coverage columns in the
               form prob_cover
           index - a string to indicate which deterministic scenario or which
               trial it is
           param_df - a df with parameters for each of the scenarios must contain
               columns: 'coverage' and 'population'
       Returns:
           a df with the population and coverage columns adjusted
    """
    new_cov_pop_df = cov_pop_df.copy()
    # Adjust population columns by the population assumption for this scenario
    pop_columns = [column for column in list(new_cov_pop_df) 
                   if re.search('pop', column)]
    for column in pop_columns:
        new_cov_pop_df[column] = (new_cov_pop_df[column] *
                                  param_df.loc[index, 'population'])
    # Adjust coverage columns by the coverage assumption for this scenario
    cov_columns = [column for column in list(new_cov_pop_df) 
                   if re.search('coverage', column)]
    for column in cov_columns:
        new_cov_pop_df[column] = (new_cov_pop_df[column] * 
                                  param_df.loc[index, 'coverage'])
        new_cov_pop_df[column] = np.where(new_cov_pop_df[column] > 0.95, 
                                          0.95, new_cov_pop_df[column])
    # Adjust prob_cover columns by the coverage assumption for this scenario
    cov_columns = [column for column in list(new_cov_pop_df) 
                   if re.search('prob_cover', column)]
    for column in cov_columns:
        new_cov_pop_df[column] = (new_cov_pop_df[column] * 
                                  param_df.loc[index, 'prob_cover'])
        new_cov_pop_df[column] = np.where(new_cov_pop_df[column] > 0.95, 
                                          0.95, new_cov_pop_df[column]) 
    return new_cov_pop_df  

def adjust_cov_pop_for_trials(cov_pop_dict, param_dict):
    """Adjusts the cov_pop_dict so its values are now a dictionary of scenario
       and dfs to use in each of those scenarios
       Inputs:
           cov_pop_dict - a dictionary where the keys are id_codes and the 
               values are dfs of coverage and population data
           param_dict - a dictionary where the keys are id_codes and the values
               are dfs of paramters for each of hte scenarios
       Returns:
           a dict where the keys are id_codes and the values are dicts of where
           keys are scenarios and values are dfs of appropriate coverage / population
           data
    """
    new_cov_pop_dict = {}
    # Loop through the dictionary by keys
    for code in cov_pop_dict.keys():
        # Set up the dict and dfs
        cov_pop_scenarios_dict = {}
        cov_pop_df = cov_pop_dict[code]
        param_df = param_dict[code]
        # Loop through each of the scenarios 
        for index in param_df.index.tolist():
            # Adjust the cov_pop_df based on the scenario parameters
            new_cov_pop_df = adjust_cov_pop_df(cov_pop_df, index, param_df)
            cov_pop_scenarios_dict[index] = new_cov_pop_df.copy()
        new_cov_pop_dict[code] = cov_pop_scenarios_dict    
    return new_cov_pop_dict
    
def merge_cov_pop_and_burden(burden_dict, cov_pop_dict):
    """Merges the dataframes within a nested dictionary structure
       
       Inputs:
           burden_dict - keys - id_code, values are dictionary of scenarios
               and burden data dfs
           cov_pop_dict - keys - id_code, values are dictionary of scenarios
               and coverage and population data dfs
           (both sets of keys, and indexes of the dfs have to be equivalent)
       Returns:
           a merged data_dict of dictionaries
    """
    data_dict = {}
    for code in burden_dict.keys():
        scenario_dict = {}
        scenario_dict_burden = burden_dict[code]
        scenario_dict_cov_pop = cov_pop_dict[code]
        for scen in scenario_dict_burden.keys():
             # Merge population and burden data
             merged_df = pd.concat([scenario_dict_burden[scen],
                                            scenario_dict_cov_pop[scen]],
                                            axis = 1)
             # Deduplicate columns from the df
             merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]
             scenario_dict[scen] = merged_df
        data_dict[code] = scenario_dict
    return data_dict
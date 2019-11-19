# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 15:30:47 2019

@author: LaurencT
"""

import pandas as pd
import numpy as np
import re

def check_analysis_type(analysis_type):
    """Checks the values in the analysis type dictionary to make sure they are
       bools or ints as appropriate
       Inputs:
           analysis_type: a dict where values are ints or bools depending on
               what they determine
    """
    # Take out the values that are supposed to be bools
    analysis_values = [analysis_type['run_all'], 
                       analysis_type['overwrite_estimates']]
    # Test to see if they are bools
    for value in analysis_values:
        if not type(value) is bool:
            raise ValueError('all of the run_... parameters in analysis_type should be booleans')
    # Test to make sure num trials is an int
    if not type(analysis_type['num_trials']) is int:
        raise ValueError('the number of trialsin analysis_type has to be an integer')

def check_indexes(param_user_all):
    """Checks to ensure all of the id_codes for each separate analysis are unique
       Inputs:
           param_user_all - a df of user inputted parameters
    """
    indexes = param_user_all.index.tolist()
    if len(set(indexes)) < len(indexes):
        raise ValueError('id_codes should be unique, please ensure there are no duplicate id codes in the LTLI_parameters.csv')

def check_columns(param_user_all):
    """Checks to ensure all of the required column names are contained in the 
       parameters tab
       Inputs:
           param_user_all - a df of user inputted parameters
    """
    necessary_columns = ['id_name', 'intervention_type', 'disease_1', 'disease_2', 
                         'disease_3', 'age', 'population_upper', 
                         'population_mean', 'population_lower', 'population_SD', 
                         'disease_1_prop_mean', 'disease_1_prop_lower', 
                         'disease_1_prop_upper', 'disease_1_prop_SD', 
                         'disease_2_prop_mean', 'disease_2_prop_lower', 
                         'disease_2_prop_upper', 'disease_2_prop_SD', 
                         'disease_3_prop_mean', 'disease_3_prop_lower', 
                         'disease_3_prop_upper', 'disease_3_prop_SD', 
                         'endem_thresh_mean', 'endem_thresh_lower', 
                         'endem_thresh_upper', 'endem_thresh_SD', 
                         'endem_thresh_metric', 'inflation_factor_upper', 
                         'inflation_factor_mean', 'inflation_factor_lower', 
                         'inflation_factor_SD', 'intervention_cut_upper', 
                         'intervention_cut_mean', 'intervention_cut_lower', 
                         'intervention_cut_SD', 'prob_cover_upper', 
                         'prob_cover_mean', 'prob_cover_lower', 
                         'prob_cover_SD', 'coverage_upper', 'coverage_mean', 
                         'coverage_lower', 'coverage_SD', 'coverage_below_threshold', 
                         'share_upper', 'share_mean', 'share_lower', 'share_SD', 
                         'efficacy_upper', 'efficacy_mean', 'efficacy_lower', 
                         'efficacy_SD', 'lives_touched', 'lives_touched_975', 
                         'lives_touched_025', 'lives_improved', 'lives_improved_975', 
                         'lives_improved_025', 'exception_count', 'exception_comment']
    necessary_columns_missing = [column for column in necessary_columns
                                 if column not in list(param_user_all) ]
    if len(necessary_columns_missing)>0:
        raise ValueError('The following columns are missing from the LTLI_parameters.csv '+str(necessary_columns_missing))

def check_not_negative(param_user_all):
    """Checks to ensure all of the numerical parameters are non-negative
       Inputs:
           param_user_all - a df of parameters, some columns are numerical
       Raises:
           a value error if another of the parameter values are negative
    """
    column_roots = [re.sub('_mean', '', column)
                    for column in param_user_all if re.search('mean', column)]
    numerical_columns = [[column+'_mean', column+'_upper', column+'_lower', column+'_SD']
                         for column in column_roots]
    numerical_columns = [column for sublist in numerical_columns for column in sublist]
    contains_negative = [any(param_user_all[column]<0) for column in numerical_columns]
    if any(contains_negative):
        negative_columns = np.array(numerical_columns)[np.array(contains_negative)]
        raise ValueError('The following columns in LTLI_parameters.csv contain negative values and should not '+str(negative_columns))

def check_iterable_1_not_smaller(iterable_1, iterable_2):
    """Checks two iterables of the same length for whether each element in 1
       is at least as big as the corresponding element of 2
       Inputs:
           iterable_1 - an iterable of arbitary length n
           iterable_2 - an iterable of length n which is ordered to correspond 
               to iterable_1
       Returns:
           bool reflecting whether all elements are not smaller
    """
    if len(iterable_1) == len(iterable_2):
        bool_list = map(lambda x,y: x>=y, iterable_1, iterable_2)        
    else:
        raise ValueError('the iterables must be the same length')
    return all(list(bool_list))

def check_upper_lower(param_user_all):
    """Checks to ensure every value of the upper parameters is at least as great
       as the mean and the mean is at least as great as the lower
       inputs:
           param_user_all - a df of user inputted parameters
    """
    column_roots = [re.sub('_mean', '', column)
                    for column in param_user_all if re.search('mean', column)]
    column_mapping = [(column+"_upper", column+"_mean", column+"_lower") 
                       for column in column_roots]
    for mapping in column_mapping:
        # Check upper >= mean
        if not check_iterable_1_not_smaller(iterable_1 = param_user_all[mapping[0]],
                                            iterable_2 = param_user_all[mapping[1]]):
            raise ValueError(mapping[1]+' in LTLI_parameters.csv is greater than '+mapping[0])            
        # Check mean >= lower
        elif not check_iterable_1_not_smaller(iterable_1 = param_user_all[mapping[1]],
                                            iterable_2 = param_user_all[mapping[2]]):
            raise ValueError(mapping[2]+' in LTLI_parameters.csv is greater than  '+mapping[1])
        # Upper>=lower by transitivity therefore is valid
        else:
            print('Values of '+mapping[0]+', '+mapping[1]+', '+mapping[2]+' are consistent')

def column_checker(column, ther_diag_param, correct_value):
    """Checks if a column of a df is all one value, raises an error if it's not
       inputs:
           column - a string which is the name of a column in ther_diag_param
           ther_diag_param - a df of parameters that is filtered so it only
               contains entries on diagnostics and therapeutics
           correct_value - a number that is the expected column value
    """
    if not all(v == correct_value for v in ther_diag_param[column]):
        raise ValueError('Values in '+column+' in LTLI_parameters.csv should all be '+str(correct_value)+' for diagnostics and therapeutics')
    
def check_diag_ther(param_user_all):
    """Checks diagnostics and therapeutics to ensure they don't have incorrect 
       values for population and endemicity - as those do not feed into these 
       analyses
       Inputs:
           param_user_all - a df of all user inputted parameters
    """
    # Filter the df so it is relevant to therapeutics and diagnostics
    ther_diag_param = param_user_all[param_user_all.intervention_type.isin(['Therapeutic', 'Therapeutic mental health', 'Diagnostic'])]
    # Find the parameters that aren't relevant for the analysis and make sure they won't affect results
    population_columns = [column for column in param_user_all if re.search('population', column)]
    endem_columns = [column for column in param_user_all if re.search('endem.*[d-z]$', column)]
    # Check each column in turn to ensure it is the right value
    for column in population_columns:
        # Population estimates don't feed in to these analysis so shouldn't be 
        # have a distribution
        if re.search("SD", column):                    
            column_checker(column, ther_diag_param, correct_value = 0)
        # Population estimates don't feed in to these analysis so should all be 1 
        else:
            column_checker(column, ther_diag_param, correct_value = 1)
        # Endemicity threshold shouldn't be applied here, as countries with 
        # low burden may use it, on their limited number of patients
    for column in endem_columns:
        column_checker(column, ther_diag_param, correct_value = 0)

def check_disease_selection(param_user_all, burden_all):
    """Checks to confirm that all selected diseases are valid
       Inputs:
           param_user_all - a df of inputted parameters must contain disease_1
               disease_2, and disease_3 columns
           burden_all - a df of burden data, disease column must be called cause
    """
    # All diseases in the dataset - GBD+aetiology+malaria breakdown#~ - Empty is 
    # only other valid option
    valid_diseases = set(burden_all['cause'].tolist()+['Empty'])
    # Makes a list of all diseases inputted by the user
    disease_choices = (param_user_all['disease_1'].tolist()+
                       param_user_all['disease_2'].tolist()+
                       param_user_all['disease_3'].tolist())
    for disease in disease_choices:
        if not disease in valid_diseases:
            #~ could write in a fuzzy lookup for potential valid names
            raise ValueError(disease+' in LTLI_parameters.csv is not a valid disease name, please consult handover guidance for valid diseases')

def flatten(x):
    """Flattens nested list"""
    if isinstance(x, list):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]

def check_burden(burden_all):
    """Checks to make sure all the column names in the burden data are consist
       ent with what are called in the later code
       Inputs:
           burden_all - names of disease burden columns contain a _ because 
               they are 'measure_metric', there are columns for 'country', 
               'age', 'cause'
    """
    # Specify necessary categorising columns
    expected_columns = ['age', 'super_region', 'region', 'country', 'cause']
    
    # Specify the different burden data columns expected
    expected_burden_columns = ['dalys', 'deaths', 'incidence', 'prevalence']
    expected_burden_columns = [[column+'_number', column+'_rate'] 
                                for column in expected_burden_columns]
    expected_burden_columns = flatten(expected_burden_columns)
    expected_burden_columns = [[column+'_upper', column+'_mean', column+'_lower']
                                for column in expected_burden_columns]
    expected_burden_columns = flatten(expected_burden_columns)
    
    # Combine all the expected columns
    all_expected_columns = expected_columns + expected_burden_columns
    
    # Generate TRUEs where they are missing
    missing_column_bool = [column not in list(burden_all) for column in all_expected_columns]
    if any(missing_column_bool):
        missing_columns = np.array(all_expected_columns)[np.array(missing_column_bool)]
        # Raise an error about missing necessary columns
        raise ValueError(str(missing_columns)+' columns are missing from gbd_data_wide_2017.csv')
    

def check_population(population):
    """Checks to make sure all the column names and indexes in the population 
       data are consistent with what are called in the later code
       Inputs:
           population - a df of population data
    """
    # Check the column names are there
    expected_names = ['pop_0-0', 'pop_1-4', 'pop_5-14', 'pop_15-69','pop_70-100']
    expected_names_present = [name in list(population) for name in expected_names]
    if not all(expected_names_present):
        raise ValueError('The population df does not have the expected columns')
    # Check the index for country names
    elif 'France' not in population.index:
        raise ValueError('The population df does not have countries as indexes, ensure GBD_population_2016_reshaped has the column location_name')
    else:
        # do nothing because not problems
        pass 

def check_coverage(coverage):
    """Checks to make sure all the column names and indexes in the coverage data 
       are consistent with what are called in the later code
       Inputs:
           population - a df of coverage data
    """
    expected_names = ['vaccine_coverage',  'vaccine_prob_cover',
                     'device_coverage', 'device_prob_cover',
                     'rapid_diagnostic_test_coverage', 'rapid_diagnostic_test_prob_cover',
                     'therapeutic_coverage', 'therapeutic_prob_cover',
                     'therapeutic_mental_health_coverage', 
                     'therapeutic_mental_health_prob_cover']
    expected_names_missing = [name for name in expected_names
                              if name not in list(coverage)]
    if len(expected_names_missing)>0:
        raise ValueError('The following columns are missing from intervention_coverage_assumptions.csv '+str(expected_names_missing))
    # Check the index for country names
    elif 'France' not in coverage.index:
        raise ValueError('The coverage_df does not have countries as indexes, ensure intervention_coverage_assumptions.csv sheet Penetration Assumptions has a column called country')
    else:
        # do nothing because not problems
        pass 

def check_inputs(analysis_type, param_user_all, population, coverage, burden_all):
    """Checks all user inputs using a variety of functions - raises error if 
       not valid inputs
       Inputs:
           analysis_type - a dict 
           param_user_all - a df of input parameters
           population - a df of population data
           coverage - a df of coverage data
           burden_all - a df of burden data
    """
    # Make sure user inputted parameters are valid
    check_analysis_type(analysis_type)
    check_indexes(param_user_all)
    check_columns(param_user_all)
    check_not_negative(param_user_all)
    check_upper_lower(param_user_all)
    check_diag_ther(param_user_all)
    check_disease_selection(param_user_all, burden_all)
    # Make sure burden / population / coverage datasets imported have the 
    # right country names and column names
    check_burden(burden_all)
    check_population(population)
    check_coverage(coverage)
    
def check_run_all(analysis_type, param_user_dict):
    """Gets a user input of the right id_code if not running all previous 
       Inputs:
           analysis_type - the dictionary summarising what type of analysis is 
           being undertaken, must contain the key 'run_all'
           param_user_all - df of parameters, must have 'id_code' as indexes
       Returns:
           the user code input by the user
    """
    id_codes = list(param_user_dict.keys())
    if analysis_type['run_all'] == False:
        print('\nHere is a list of possible id_codes: \n', id_codes, '\n')
        id_user = input('Please input a relevant id_code from the csv: ')
        while id_user not in id_codes:
            id_user = input("That is not a valid id_code, please try again: ")
        param_user_dict = {id_user: param_user_dict[id_user]} 
    return param_user_dict

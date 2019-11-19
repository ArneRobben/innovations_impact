# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 15:08:49 2019

@author: LaurencT
"""

import pandas as pd
import os
import re

def load_params(csv_name, directory):
    """Imports the params and returns them as a df"""
    csv_path = os.path.join(directory, csv_name)
    param_user_all = pd.read_csv(csv_path)
    param_user_all = param_user_all.set_index('id_code')
    return param_user_all
    
def create_param_dict(param_user_all):
    """Turns df of parameters into a dict"""
    param_user_dict = {code : param_user_all.loc[code] 
                       for code in param_user_all.index.tolist()}
    return param_user_dict


def load_population_data(csv_name, directory):
    """Imports population data, regroups columns, renames cols, returns a df"""
    # Importing the relevant population data
    csv_path = os.path.join(directory, csv_name)
    population = pd.read_csv(csv_path)
    
    # Subset the most relevant columns - this list could be updated to capture other 
    # relevant age groups - they presently capture infants, young children, children/
    # adolescents, working age people and then retired people
    pop_columns = ['location_name', 'Both_<1 year', 'Both_1 to 4',
                   'Both_5-14 years', 'Both_15-49 years', 'Both_50-69 years',
                   'Both_70+ years']
    
    population = population.loc[:, pop_columns]
    
    # Merge two of the columns to make a more relevant one
    population.insert(loc = 5, column =  'Both_15-69 years', 
                      value = (population['Both_15-49 years'] + population['Both_50-69 years']))
    
    # Remove the columns which are reflected in new merged column
    pop_new_columns = [column for column in list(population) 
                       if not column in ['Both_15-49 years', 'Both_50-69 years']]
    
    population = population[pop_new_columns]
    
    # Rename columns to more consistent / intuitive names
    pop_new_names = {'location_name' : 'country', 'Both_<1 year' : 'pop_0-0',
                     'Both_1 to 4' : 'pop_1-4', 'Both_5-14 years' : 'pop_5-14', 
                     'Both_15-69 years' : 'pop_15-69', 'Both_70+ years' : 'pop_70-100'}
    
    population = population.rename(columns = pop_new_names)
    
    # Reindex so country is the index
    population = population.set_index(keys = 'country', drop = True)
    
    return population

# Importing the relevant disease burden data
def load_burden_data(csv_name, directory):
    """Imports burden data, cleans col names, returns df"""
    csv_path = os.path.join(directory, csv_name)
    burden_all = pd.read_csv(csv_path)
    burden_all.columns = [column.lower() for column in burden_all.columns.tolist()]
    return burden_all

# Importing the relevant coverage data
def load_coverage_assumptions(excel_name, sheet_name, directory):
    """Imports coverage assumptions, cleans and returns a df"""
    xls_path = os.path.join(directory, excel_name)
    coverage = pd.read_excel(xls_path, sheet_name)
    
    coverage.columns = coverage.iloc[11]
    coverage = coverage.iloc[12:, 1:]
    
    cov_new_columns = [column for column in list(coverage) if re.search('cover|^country', column)]
    coverage = coverage[cov_new_columns]
    
    cov_new_names = {name : str.lower(re.sub(" ", "_", name)) for name in list(coverage)}
    coverage = coverage.rename(columns = cov_new_names)
    coverage.index = coverage['country']
    return coverage
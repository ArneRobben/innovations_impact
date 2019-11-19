# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 10:36:57 2019

@author: LaurencT
"""

from argparse import ArgumentParser

def ltli_argument_parser(DATA_DIR, GRAPH_DIR, OUTPUTS_DIR, BACKUP_DIR, SLIDES_DIR,
                         PARAM_CSV_NAME, ESTIMATES_CSV_NAME, POPULATION_CSV_NAME,
                         BURDEN_CSV_NAME, COVERAGE_XLS_NAME, COVERAGE_SHEET_NAME,
                         PPT_TEMPLATE_NAME):
    """Takes strings reflecting directories and file names and returns an 
       argument that contains user overrides of those strings where relevant
    """
    parser = ArgumentParser(description = 'Input strings to override the default ' +
                        'folder locations required to get data and write outputs')

    parser.add_argument('--DATA_DIR', 
                        type = str, 
                        help = 'A string indicating the folder containing the' +
                               ' model data. The default is ' + DATA_DIR, 
                        default = DATA_DIR,
                        required = False)
    parser.add_argument('--GRAPH_DIR', 
                        type = str, 
                        help = 'A string indicating the folder where the graphs' +
                               ' should be exported. The default is ' + GRAPH_DIR, 
                        default = GRAPH_DIR,
                        required = False)
    parser.add_argument('--OUTPUTS_DIR', 
                        type = str, 
                        help = 'A string indicating the folder where the outputs' +
                               ' should be exported. The default is ' + OUTPUTS_DIR, 
                        default = OUTPUTS_DIR,
                        required = False)
    parser.add_argument('--BACKUP_DIR', 
                        type = str, 
                        help = 'A string indicating the folder where the parameters' +
                               ' should be backed up. The default is ' + BACKUP_DIR, 
                        default = BACKUP_DIR,
                        required = False)
    parser.add_argument('--SLIDES_DIR', 
                        type = str, 
                        help = 'A string indicating the folder where the ppt slides' +
                               ' should be exported. The default is ' + SLIDES_DIR, 
                        default = SLIDES_DIR,
                        required = False)
    parser.add_argument('--PARAM_CSV_NAME', 
                        type = str, 
                        help = 'A string indicating the file name for the model' +
                               ' parameters. The default is ' + PARAM_CSV_NAME, 
                        default = PARAM_CSV_NAME,
                        required = False)
    parser.add_argument('--ESTIMATES_CSV_NAME', 
                        type = str, 
                        help = 'A string indicating the file name for the output' +
                               ' estimates. The default is ' + ESTIMATES_CSV_NAME, 
                        default = ESTIMATES_CSV_NAME,
                        required = False)
    parser.add_argument('--POPULATION_CSV_NAME', 
                        type = str, 
                        help = 'A string indicating the file name for the population' +
                               ' data. The default is ' + POPULATION_CSV_NAME, 
                        default = POPULATION_CSV_NAME,
                        required = False)
    parser.add_argument('--BURDEN_CSV_NAME', 
                        type = str, 
                        help = 'A string indicating the file name for the burden' +
                               ' data. The default is ' + BURDEN_CSV_NAME, 
                        default = BURDEN_CSV_NAME,
                        required = False)
    parser.add_argument('--COVERAGE_XLS_NAME', 
                        type = str, 
                        help = 'A string indicating the file name for the coverage' +
                               ' data. The default is ' + COVERAGE_XLS_NAME, 
                        default = COVERAGE_XLS_NAME,
                        required = False)
    parser.add_argument('--COVERAGE_SHEET_NAME', 
                        type = str, 
                        help = 'A string indicating the excel sheet name for the' +
                               ' coverage data. The default is ' + COVERAGE_SHEET_NAME, 
                        default = COVERAGE_SHEET_NAME,
                        required = False)
    parser.add_argument('--PPT_TEMPLATE_NAME', 
                        type = str, 
                        help = 'A string indicating the file name for the ppt template ' +
                               ' to use for the slides. The default is ' + PPT_TEMPLATE_NAME, 
                        default = PPT_TEMPLATE_NAME,
                        required = False)
    
    args = parser.parse_args()
    
    return args

def print_selected_arguments(DATA_DIR, GRAPH_DIR, OUTPUTS_DIR, BACKUP_DIR, SLIDES_DIR,
                             PARAM_CSV_NAME, ESTIMATES_CSV_NAME, POPULATION_CSV_NAME,
                             BURDEN_CSV_NAME, COVERAGE_XLS_NAME, COVERAGE_SHEET_NAME,
                             PPT_TEMPLATE_NAME):
    """Prints strings indicating the directories / file names selected by the user
    """
    print('You selected', DATA_DIR, 'as the DATA_DIR', sep = ' ')
    print('You selected', GRAPH_DIR, 'as the GRAPH_DIR', sep = ' ')
    print('You selected', OUTPUTS_DIR, 'as the OUTPUTS_DIR', sep = ' ')
    print('You selected', BACKUP_DIR, 'as the BACKUP_DIR', sep = ' ')
    print('You selected', SLIDES_DIR, 'as the SLIDES_DIR', sep = ' ')
    print('You selected', PARAM_CSV_NAME, 'as the PARAM_CSV_NAME', sep = ' ')
    print('You selected', ESTIMATES_CSV_NAME, 'as the ESTIMATES_CSV_NAME', sep = ' ')
    print('You selected', POPULATION_CSV_NAME, 'as the POPULATION_CSV_NAME', sep = ' ')
    print('You selected', BURDEN_CSV_NAME, 'as the BURDEN_CSV_NAME', sep = ' ')
    print('You selected', COVERAGE_XLS_NAME, 'as the COVERAGE_XLS_NAME', sep = ' ')
    print('You selected', COVERAGE_SHEET_NAME, 'as the COVERAGE_SHEET_NAME', sep = ' ')
    print('You selected', PPT_TEMPLATE_NAME, 'as the PPT_TEMPLATE_NAME', sep = ' ')
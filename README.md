# Lives touched, lives improved model

This repository is for Innovations lives touched, lives improved model, which seeks to estimate the potential impact of interventions were they to be successfully developed. This model relies on assumptions abstract in order to model scenarios of potential impact - this results in highly uncertain estimates which is quantified.

## How to setup:

This repository requires Python 3.6+.

Replicate the folder structure demonstrated [here](https://wellcomecloud.sharepoint.com/sites/innovations/PAD/Forms/AllItems.aspx?id=%2Fsites%2Finnovations%2FPAD%2FPAD%20Archive%2FData%20Analyst%20Handover%2Flives%5Ftouched%5Flives%5Fimproved) in you "My Documents" folder. All of the required files are found in the data folder.

The required packages are set out in requirements.txt:

```
pip install --user -r requirements.txt 
```

You also need to sync this [folder](https://wellcomecloud.sharepoint.com/sites/innovations/PAD/Forms/AllItems.aspx?id=%2Fsites%2Finnovations%2FPAD%2FLives%20touched%2C%20lives%20improved%20model%20results) to OneDrive to be able to upload the model charts to the right place.

More detailed instructions on how to set up your computer and run the model can be found [here](https://wellcomecloud.sharepoint.com/:p:/r/sites/innovations/PAD/PAD%20Archive/Data%20Analyst%20Handover/lives_touched_lives_improved_support/Instruction%20for%20updating%20LTLI%20V2.pptm?d=we7c4a802284d48528e301f0a36865c16&csf=1&e=0H6OI9).

## How to run:

lives_touched_lives_improved.py is the main file to run the model from (it calls on the other modules in the 
repository). 

It can be called from the command line and arguments passed to it to indicate the directories and file names where the required files (mentioned in the How to setup section above) can be found. Use the following command to see the possible inputs:

```

lives_touched_lives_improved.py -h

  --DATA_DIR DATA_DIR   A string indicating the folder containing the model
                        data. The default is C:/Users/laurenct/OneDrive -
                        Wellcome Cloud/My Documents/python/lives_touched_lives_improved/data/
  --GRAPH_DIR GRAPH_DIR
                        A string indicating the folder where the graphs should
                        be exported. The default is C:/Users/laurenct/OneDrive
                        - Wellcome Cloud/My
                        Documents/python/lives_touched_lives_improved/graphs/
  --OUTPUTS_DIR OUTPUTS_DIR
                        A string indicating the folder where the outputs
                        should be exported. The default is
                        C:/Users/laurenct/OneDrive - Wellcome Cloud/My
                        Documents/python/lives_touched_lives_improved/outputs/
  --BACKUP_DIR BACKUP_DIR
                        A string indicating the folder where the parameters
                        should be backed up. The default is
                        C:/Users/laurenct/OneDrive - Wellcome Cloud/My Documen
                        ts/python/lives_touched_lives_improved/data/backup/
  --SLIDES_DIR SLIDES_DIR
                        A string indicating the folder where the ppt slides
                        should be exported. The default is
                        C:/Users/laurenct/Wellcome Cloud/Innovations - Lives
                        touched, lives improved model results/
  --PARAM_CSV_NAME PARAM_CSV_NAME
                        A string indicating the file name for the model
                        parameters. The default is LTLI_parameters.csv
  --ESTIMATES_CSV_NAME ESTIMATES_CSV_NAME
                        A string indicating the file name for the output
                        estimates. The default is LTLI_outputs_baseline.csv
  --POPULATION_CSV_NAME POPULATION_CSV_NAME
                        A string indicating the file name for the population
                        data. The default is GBD_population_2016_reshaped.csv
  --BURDEN_CSV_NAME BURDEN_CSV_NAME
                        A string indicating the file name for the burden data.
                        The default is gbd_data_wide_2017.csv
  --COVERAGE_XLS_NAME COVERAGE_XLS_NAME
                        A string indicating the file name for the coverage
                        data. The default is
                        intervention_coverage_assumptions.xlsm
  --COVERAGE_SHEET_NAME COVERAGE_SHEET_NAME
                        A string indicating the excel sheet name for the
                        coverage data. The default is Penetration assumptions
  --PPT_TEMPLATE_NAME PPT_TEMPLATE_NAME
                        A string indicating the file name for the ppt template
                        to use for the slides. The default is
                        mm_template_impact.pptx
```

It can be also run from an IDE - at a minimum the file names and directory paths will need updating to be consistent with the folder structures on your machine - see more detailed instructions on that [here](https://wellcomecloud.sharepoint.com/:p:/r/sites/innovations/PAD/PAD%20Archive/Data%20Analyst%20Handover/lives_touched_lives_improved_support/Instruction%20for%20updating%20LTLI%20V2.pptm?d=we7c4a802284d48528e301f0a36865c16&csf=1&e=0H6OI9).

To change the default behaviour of the lives_touched_lives_improved.py change the default terms in ANALYSIS_TYPE.

If ANALYSIS_TYPE is set 'run_all': False in lives_touched_lives_improved.py then you will have to input a valid id_code to run your chosen model. The id_codes come from LTLI_paramters.csv - check there to see which code you want to run. I would recommend having 'run_all': False as the default - as each model takes a couple of minutes to run on a standard Wellcome laptop.

## How to model new projects:

All of the model parameters are contained in LTLI_parameters.csv. To model a new project then simply add a row to the csv with the new parameters. 

Find instructions on how to input valid and sensible parameters [here](https://wellcomecloud.sharepoint.com/:p:/r/sites/innovations/PAD/PAD%20Archive/Data%20Analyst%20Handover/lives_touched_lives_improved_support/Instruction%20for%20updating%20LTLI%20V2.pptm?d=we7c4a802284d48528e301f0a36865c16&csf=1&e=0H6OI9). 

## Note carefully:

1. There are not tests associated with all of the functions in this package.
2. There are input_checks which will throw errors if the datasets do not have the required columns, if in doubt then follow the format of the files handed over [here](https://wellcomecloud.sharepoint.com/sites/innovations/PAD/Forms/AllItems.aspx?id=%2Fsites%2Finnovations%2FPAD%2FPAD%20Archive%2FData%20Analyst%20Handover%2Flives%5Ftouched%5Flives%5Fimproved) (column names etc.).
3. The apply_exceptions.py module uses id_codes to make adjustments for a couple of projects modelled. These exception comments are tracked in the LTLI_outputs.csv - before making changes to an id_code (which I would not recommend generally) see if there is a relevant exception you are going to negate. The only exception I have applied is geographical limits on projects, a more general module could be written to apply this based on parameters, but hasn't been completed at time of writing. 


from cluster_programme import process_surveydata, remove_missing
import subprocess
import pandas as pd 

#import data from survey
newfile=pd.read_csv('../datafiles/fakesurvey.csv')
#create list of variables to use
var= pd.read_excel('../datafiles/survey_vbl.xlsx')
var_new= var['variables'].tolist()

new_frame=process_surveydata(dataframe=newfile, variables=var_new)
new_frame.to_csv('../output/treated_survey.csv', index=False)

new_frame=remove_missing(new_frame)
new_frame.to_csv('../output/nomissing_survey.csv', index=False)
#imputing in R won't work for this few rows:
subprocess.call(['Rscript', 'impute.R'])
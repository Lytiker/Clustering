import pandas as pd
from cluster_programme import true_labels, select_numeric, diagnose_dict_y,make_agecolumn,make_year,first_diag,last_diag,diagnose_dict,last_diagnoses

new_df=pd.read_csv('../output/faketrain.csv')
print new_df.columns
new_df=make_agecolumn(new_df, 'birthdate', 'dateres', 'surveyage')
new_df=make_year(new_df, 'birthdate', 'birthyear')
new_df=first_diag(new_df)
new_df=make_year(new_df,'first_diag', 'first_diagyear')
new_df=make_agecolumn(new_df, 'birthdate', 'first_diag', 'first_diagage')


new_df=last_diag(new_df)
new_df=make_year(new_df,'last_diag', 'last_diagyear')
new_df=make_agecolumn(new_df, 'birthdate', 'last_diag', 'last_diagage')

new_df=diagnose_dict(new_df)
new_df=diagnose_dict_y(new_df)

new_df=last_diagnoses(new_df)
new_df=select_numeric(new_df)
new_df=true_labels(new_df)

new_df.to_csv('../output/new_attr.csv')
print new_df.columns.values
print new_df.shape


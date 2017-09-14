"""two distance matrices from an imputed training dataset with 54 attributes, from two
different row measures"""
import pandas as pd
import numpy as np
from cluster_programme import distance_matrix, distance_matrix2
import time

#read dataset with new atttributes
new_df=pd.read_csv('../output/new_attr.csv', index_col=0)
#drop id and ground truth labels
new_df=new_df.drop(['ID', 'Worst_diagnosis', 'true_labels', 'true_labels2'], axis=1)
#create a list of categorical/continuoustypes
var_df= pd.read_excel('../datafiles/survey_vbl.xlsx')
var_types=var_df['VariableType'].tolist()
var_types.extend(['continuous','continuous','continuous','continuous',
	'continuous','continuous','continuous','continuous','categorical'])
del var_types[0:2]
del var_types[2]
print new_df.columns.values
print var_types
print 'length of variabeltype-list and dataframe shape: %i, %s' % (len(var_types), new_df.shape)

#new_df=new_df[:100]
#create distance matrix

start= time.time()
print('start matrix1')
distance_matrix=distance_matrix(new_df, var_types)
print('making a file for the matrix1')
np.savetxt('../output/matrix1.csv', distance_matrix, delimiter=',')
end1= time.time()
print('matrix1 finished at time %f, start matrix2') % (end1-start)



distance_matrix2=distance_matrix2(new_df)
print('making a file for the matrix2')
np.savetxt('../output/matrix2.csv', distance_matrix2, delimiter=',')
end2= time.time()
print('matrix2 finished! time was %f') % (end2-end1)

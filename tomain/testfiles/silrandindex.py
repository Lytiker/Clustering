import pandas as pd 
import numpy as np 
from cluster_programme import qual_measure

#load all clusterfiles
kproto=pd.read_csv('../output/kproto.csv', index_col=0) 
kmeans=pd.read_csv('../output/kmeans.csv', index_col=0) 
DBSCAN=pd.read_csv('../output/DBSCAN.csv', index_col=0) 
daiDBSCAN=pd.read_csv('../output/daisy_DBSCAN.csv', index_col=0) 
m1DBSCAN=pd.read_csv('../output/matrix1_DBSCAN.csv', index_col=0) 
daisyclust =pd.read_csv('../output/daisycluster.csv', index_col=0) 
matrix1clust=pd.read_csv('../output/matrix1cluster.csv', index_col=0) 


print'data loaded'
daisy_matrix=pd.read_csv('../output/daisymatrix.csv', header=None)
matrix1=pd.read_csv('../output/matrix1.csv',header=None)
print 'distance matrix loaded'

print 'algorithm: (rand index , silhouette index)'
print 'kprototype: {0}'.format(qual_measure(daisy_matrix, kproto['true_labels'], kproto['kproto_cluster']))
print 'kmeans: {0}'.format(qual_measure(daisy_matrix, kmeans['true_labels'], kmeans['hypertools_Cluster']))
print 'DBSCAN with daisy distance metric: {0}'.format(qual_measure(daisy_matrix, DBSCAN['true_labels'], DBSCAN['DBSCAN_cluster']))
#print qual_measure(daisy_matrix, daiDBSCAN['true_labels'], daiDBSCAN['DBSCAN_Cluster_ownmetric'])
#print qual_measure(matrix1, m1DBSCAN['true_labels'], m1DBSCAN['DBSCAN_Cluster_ownmetric'])
print 'hierarchical clustering with gower distance metric: {0}'.format(qual_measure(daisy_matrix, daisyclust['true_labels'], daisyclust['daisycluster']))
print 'hierarchical clustering of matrix1: {0}'.format(qual_measure(matrix1, matrix1clust['true_labels'], matrix1clust['matrix1cluster']))
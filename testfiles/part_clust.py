import pandas as pd 
from cluster_programme import kmeans_cluster, kproto, DBSCAN_cluster, DBSCAN_cluster2
from seaborn import heatmap
from scipy.cluster.hierarchy import cut_tree
import time

#load training data
new_df=pd.read_csv('../output/new_attr.csv', index_col=0) 

#load selected distance matrix made from training data
data1=pd.read_csv('../output/daisymatrix.csv', header=None)	
data2=pd.read_csv('../output/matrix1.csv', header=None)	
data3=pd.read_csv('../output/matrix2.csv', header=None)	
print data2
print data3

#drop and save ids and ground truth
worst=new_df['Worst_diagnosis'].values
true1=new_df['true_labels'].values
true2=new_df['true_labels2'].values
new_df=new_df.drop(['Worst_diagnosis', 'true_labels', 'true_labels2'], axis=1)
ids= new_df.pop('ID').values

#create list of index of the categorical columns for k-prototype
cat_cols=range(0,11)
cat_cols.append(13)
cat_cols.extend(range(18,25))
cat_cols.append(36)
print cat_cols

#time and save dataset with clusterlabels from k-prototype
start=time.time()
new_dfk=kproto(new_df, cat_cols, nr_clusters=4)
print 'kproto finished'
new_dfk['ID']=ids
new_dfk['Worst_diagnosis']=worst
new_df['true_labels']=true1
new_df['true_labels2']=true2
new_dfk.to_csv('../output/kproto.csv', index=False)
end = time.time()
print(end - start)

#time and save dataset with clusterlabels from k-means
start=time.time()
print new_df.head(1)
cols= new_df.columns[cat_cols]
print len(cols), new_df.shape
new_dfh=kmeans_cluster(new_df ,cols, nr_clusters=4)
print 'hypertools finished'
new_dfh['ID']=ids
new_dfh['Worst_diagnosis']=worst
new_dfh.to_csv('../output/kmeans.csv', index=False)
end = time.time()
print(end - start)

#time and save dataset with clusterlabels from DBSCAN clustering
start=time.time()
new_dfD=DBSCAN_cluster(new_df)
print 'DBSCAN finished'
new_dfD['ID']=ids
new_dfD['Worst_diagnosis']=worst
new_dfD.to_csv('../output/DBSCAN.csv', index=False)
end = time.time()
print(end - start)

#time and save datasets with clusterlabels from DBSCAN clustering
#clustering with 3 different distance matrices
start=time.time()
new_dfD1=DBSCAN_cluster2(new_df, data1)
new_dfD2=DBSCAN_cluster2(new_df, data2)
new_dfD3=DBSCAN_cluster2(new_df, data3)
print 'DBSCAN2 finished'
new_dfD1['ID']=ids
new_dfD1['Worst_diagnosis']=worst
new_dfD1.to_csv('../output/daisy_DBSCAN.csv', index=False)
new_dfD2['ID']=ids
new_dfD2['Worst_diagnosis']=worst
new_dfD2.to_csv('../output/matrix1_DBSCAN.csv', index=False)
new_dfD3['ID']=ids
new_dfD3['Worst_diagnosis']=worst
new_dfD3.to_csv('../output/matrix2_DBSCAN.csv', index=False)

end = time.time()
print(end - start)


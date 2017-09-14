import pandas as pd 
import matplotlib.pyplot as plt
from cluster_programme import qual_measure
kproto=pd.read_csv('../output/kproto.csv', index_col=0) 
kmeans=pd.read_csv('../output/kmeans.csv', index_col=0) 
DBSCAN=pd.read_csv('../output/DBSCAN.csv', index_col=0) 
daiDBSCAN=pd.read_csv('../output/daisy_DBSCAN.csv', index_col=0) 
m1DBSCAN=pd.read_csv('../output/matrix1_DBSCAN.csv', index_col=0) 
daisyclust =pd.read_csv('../output/daisycluster.csv', index_col=0) 
matrix1clust=pd.read_csv('../output/matrix1cluster.csv', index_col=0) 

python_df=pd.read_csv('../output/new_attr.csv', index_col=0) #original preprocessed data

#gather all clusters in one file
python_df['hypertools_Cluster']=kmeans['hypertools_Cluster'].values
python_df['kproto_cluster']=kproto['kproto_cluster'].values
python_df['DBSCAN_cluster']=DBSCAN['DBSCAN_cluster'].values
python_df['DBSCAN_daisy']=daiDBSCAN['DBSCAN_Cluster_ownmetric'].values
python_df['DBSCAN_m1']=m1DBSCAN['DBSCAN_Cluster_ownmetric'].values 
python_df['daisycluster']=daisyclust['daisycluster'].values
python_df['matrix1cluster']=matrix1clust['matrix1cluster'].values


hyp_clust=python_df.groupby(['hypertools_Cluster', 'true_labels']).agg({'true_labels':'count'}).add_suffix('_count').reset_index()
kprot_clust=python_df.groupby(['kproto_cluster', 'true_labels']).agg({'true_labels':'count'}).add_suffix('_count').reset_index()
DB_clust=python_df.groupby(['DBSCAN_cluster', 'true_labels']).agg({'true_labels':'count'}).add_suffix('_count').reset_index()
DB_clust2=python_df.groupby(['DBSCAN_daisy', 'true_labels']).agg({'true_labels':'count'}).add_suffix('_count').reset_index()
DB_clust3=python_df.groupby(['kproto_cluster', 'true_labels']).agg({'true_labels':'count'}).add_suffix('_count').reset_index()
daisy=python_df.groupby(['DBSCAN_cluster', 'true_labels']).agg({'true_labels':'count'}).add_suffix('_count').reset_index()
matrix1=python_df.groupby(['DBSCAN_daisy', 'true_labels']).agg({'true_labels':'count'}).add_suffix('_count').reset_index()



print kprot_clust
print hyp_clust
print DB_clust
print DB_clust2
print DB_clust3
print daisy
print matrix1

print sum(DB_clust['true_labels_count'])
DB_clust2=DB_clust2.groupby(['DBSCAN_daisy'])
newf=DB_clust2['true_labels_count'].get_group(0)

fig = plt.figure()
newf.plot.pie(autopct='%.2f%%', labels=[0,1,2,3,4], figsize=(6,6))
plt.title('Distribution of true labels in density-based clustering from matrix\nCluster0')
plt.axes().set_ylabel('')
plt.savefig('../output/piechart.pdf')
plt.close(fig)


import pandas as pd 
import matplotlib.pyplot as plt
from sklearn import manifold, datasets, decomposition, ensemble,discriminant_analysis, random_projection
from cluster_programme import pca_tsne
from sklearn.preprocessing import StandardScaler
import numpy as np 


data=pd.read_csv('../output/new_attr.csv') #training data
X_pca = pca_tsne(data)[0]
X_tsne = pca_tsne(data)[1]

#plotting figure
fig = plt.figure(figsize=(10,5))
plt.subplot2grid((1,2), (0,0))
plt.title('PRINCIPAL COMPONENTS ANALYSIS')
plt.scatter(X_pca[:, 0], X_pca[:, 1])
#make tsne on same plot
plt.subplot2grid((1,2), (0,1), rowspan=1, colspan=2) 
plt.title('t-SNE')
plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
#save output to file
plt.savefig('../output/tnse.pdf')
plt.close(fig)
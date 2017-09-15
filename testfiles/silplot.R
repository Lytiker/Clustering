library(readr)
library(cluster)
library(raster)
library(gplots)
library(fastcluster)
library(data.table)
library(grDevices)
library(Matrix)
daisy=fread('../output/daisymatrix.csv') #distance matrix from daisy
matrix1=fread('../output/matrix1.csv') #distance matrix from cluster_programme

#print(as.dist(daisy))
data1=fread('../output/kmeans.csv')
data2=fread('../output/kproto.csv')
data3=fread('../output/daisycluster.csv')
data4=fread('../output/matrix1cluster.csv')

length(daisy)
clusters1=data1$hypertools_Cluster
sil1=silhouette(clusters1, daisy)
clusters2=data2$kproto_cluster
sil2=silhouette(clusters2, daisy)
clusters3=data3$daisycluster
sil3=silhouette(clusters3, daisy)
clusters4=data4$matrix1cluster
sil4=silhouette(clusters4, matrix1)
x11()
pdf('../output/silplot.pdf')
plot(sil1, main='silhouetteplot for k-means with daisymatrix measure')
plot(sil2, main='silhouetteplot for k-prototype with daisymatrix measure')
plot(sil3, main='silhouetteplot for hierarchial clustering from daisymatrix')
plot(sil4, main='silhouetteplot for hierarchial clustering from matrix1')
dev.off()
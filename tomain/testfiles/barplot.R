library(mlbench)
library(data.table)
data=fread('../output/new_attr.csv', drop=1:2)
par(mfrow=c(1,4))
#x11()
pdf('../output/barplots_data.pdf')
for(i in 1:40){
  counts = table(data[,i, with=FALSE])
  name = names(data)[i]
  barplot(counts, main=name)
}
dev.off()

#create barplot of variables on one of the hypertools clusters
data2=fread('../output/kmeans.csv', drop=1:2)
out=split(data2, data2$hypertools_Cluster)
cluster1=out[[1]]

pdf('../output/clust1hyp.pdf')
for(i in 1:40){
  counts = table(cluster1[,i, with=FALSE])
  name = names(cluster1)[i]
  barplot(counts, main=name)
}
dev.off()


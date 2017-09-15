library(readr)
library(cluster)
library(raster)
library(gplots)
library(fastcluster)
library(data.table)

dataframe=fread('../output/new_attr.csv')
dataframe1=fread('../output/daisymatrix.csv')
dataframe2=fread('../output/matrix1.csv')
dataframe3=fread('../output/matrix2.csv')
print('loaded')
matrix1=as.matrix(dataframe1)
matrix2=as.matrix(dataframe2)
matrix3=as.matrix(dataframe3)

print('making heatmap')

pdf('../output/heatmaps.pdf', title='daisy, matrix1 and matrix2')
#can drop clustering of rows and UseRaster to optimize calculation speed

heatmap(matrix1, useRaster=TRUE, scale='row', main = 'gower_heatmap',labCol= FALSE) 
heatmap(matrix2, useRaster=TRUE, scale='row', main = 'matrix1_heatmap',labCol= FALSE) 
dev.off()

print('heatmap finished!')
print('making dendogram')

hc1=hclust(as.dist(dataframe1))
hc2=hclust(as.dist(dataframe2))
hc3=hclust(as.dist(dataframe3))
pdf('../output/dendograms.pdf')
plot(hc1, main = 'gower_dendogram')
plot(hc2, main = 'matrix1_dendogram')
dev.off()
labels1=cutree(hc1, k=5)
labels2=cutree(hc2, k=5)


df1=dataframe 
df1$daisycluster=labels1

df2=dataframe 
df2$matrix1cluster=labels2


write.csv(df1, file="../output/daisycluster.csv", quote = FALSE, row.names = FALSE)
write.csv(df2, file="../output/matrix1cluster.csv", quote = FALSE, row.names = FALSE)

print('dendograms finished!')
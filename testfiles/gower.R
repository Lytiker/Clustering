#!/usr/bin/env format_dataframe
library(readr)
library(cluster)
library(raster)
library(gplots)
library(fastcluster)
library(data.table)
library(grDevices)
library(Matrix)

fullframe=fread('../output/new_attr.csv', drop=1)
dataframe=fullframe
#remove those not to cluster for
dataframe$ID <- NULL
dataframe$Worst_diagnosis <- NULL
dataframe$true_labels <- NULL
dataframe$true_labels2 <- NULL
head(dataframe)
#convert columns to correct datatypes
dataframe[, c(1:9,19:22,26, 37)] <- lapply(dataframe[, c(1:9,19:22,26, 37)], as.factor)
dataframe[, c(10:12,14,23:25)] <- lapply(dataframe[, c(10:12,14,23:25)], as.logical)
dataframe[, c(13,15:18,27:36)] <- lapply(dataframe[, c(13,15:18,27:36)], as.numeric)


str(dataframe)
daisymatrix = daisy(dataframe, metric='gower')
daisymatrix= as.matrix(daisymatrix)
write.table(daisymatrix, '../output/daisymatrix.csv', sep=',', col.names= FALSE, row.names = FALSE,qmethod =  "double")


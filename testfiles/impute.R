library(mice)
#load dataset without dependent variables
datasets = read.csv("../output/treated_survey.csv")
saved_data=datasets #save original dataframe
datasets$ID = NULL
datasets$birthdate=NULL
datasets$dateres=NULL

#create a list of method for each variable (however a whole lot faster if only one method is used)
method_array = c('polyreg', 'polyreg', 'polyreg', 'polyreg', 'polyreg', 
	'polyreg', 'polyreg', 'polyreg', 'polyreg', 'logreg','logreg','logreg',
	'pmm', 'logreg','pmm', 'pmm', 'pmm', 'pmm', 'polyreg','polyreg',
	'polyreg','polyreg','logreg','logreg','logreg','polyreg','pmm','pmm')
fact = c('polyreg','logreg') #categorical varables must be factors
factorvars=which(method_array %in% fact) #create list of indexes to use polyreg or logreg 

#change indexed variables to factors and store data in dataframe
#select variables
ncol(datasets[factorvars])
length(method_array)
length(datasets)
#make variables factortype
datasets[factorvars]=lapply(datasets[factorvars], factor)
sapply(datasets, class)
#impute dataset
tempData=mice(datasets, m=5, maxit=50, method=method_array)
completedData <- complete(tempData,5) #use first imputation

#insert back rows
completedData$ID<-saved_data$ID
total_set =completedData[,c(28, 1:27)]
completedData$birthdate<-saved_data$birthdate
completedData$dateres<-saved_data$dateres
length(total_set)
with_birthdate = completedData[,c(26,27,28,1:25)]

#save dataset as csv
write.csv(with_birthdate, file="../output/imputed_surveyset.csv", quote = FALSE, row.names = FALSE)


#checkups
pMiss = function(x){sum(is.na(x))/length(x)*100}
apply(completedData,2, pMiss) #percentage of NaN in dataset
warnings()
str(data)


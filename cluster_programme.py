# -*- coding: utf-8 -*-

import pandas as pd 
import numpy as np 
import hypertools as hyp 
import matplotlib.pyplot as plt
import scipy
import random
import seaborn as sns
import datetime as dt
import re
from collections import defaultdict
from kmodes import kprototypes
from dateutil.relativedelta import relativedelta
from pandas.plotting import parallel_coordinates
from pandas.tools.plotting import andrews_curves
from scipy.spatial.distance import squareform, is_valid_dm, pdist, squareform
from scipy.cluster.hierarchy import linkage
from scipy.stats import mode
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score


def merge_frames(df1, df2):
	"""Merge two dataframes by 'ID' attribute. """
	return df1.merge(df2, on='ID')

def partition_data(dataframe, k=0.2):
	"""Partition data randomly into test, training and validation set.
	optional k- for separation persentage, default 20 percent test and validationset"""
	mixed_rows = dataframe.reindex(np.random.permutation(dataframe.index))
	rows = dataframe.shape[0]
	nr_trows = 0.2*rows
	nr_valrows = nr_trows + 0.2*rows 
	t_data = mixed_rows[:int(nr_trows)]
	val_data = mixed_rows[int(nr_trows):int(nr_valrows)]
	train_data = mixed_rows[int(nr_valrows):]
	return [t_data, val_data, train_data]

def missing_values(dataframe):
	return dataframe.isnull().sum()

def remove_missing(dataframe):
	"""Create initial dataset without NaN for testing of functions
	by filling with random values from the column. """
	dataframe = dataframe.apply(lambda x: x.fillna(random.choice(x.dropna())), axis=1)
	return dataframe

def impute_methods_list(excel_file):
	"""Reads an excelfile and returns a list of 
	methods from variabletypes to use in mice imputations"""
	variable_file = pd.read_excel(excel_file)
	variabletypes = variable_file['Responsetype'].tolist()
	method = []
	for i in variabletypes:
		if i=='categorical':
			method.append('polyreg')
		if i=='binary':
			method.append('logreg')
		if i=='continuous':
			method.append('pmm')
	return method


def process_surveydata(dataframe, variables):
	"""Select variables and treat dependent variables in survey dataframe. """

	variable_list = variables
	selected_set = dataframe.ix[:, variable_list] 
	#manually replace missing values with known dependence
	#if not used condom and missing for age of first use, replace missing with value 999
	selected_set.loc[(selected_set.q9condom == 0) & (selected_set['q9agecon'\
		].isnull()), 'q9agecon'] = 250
	#if never had intercourse, and missing for age of first time, replace missing with 999
	selected_set.loc[(selected_set.q11asx == 0) & (selected_set['q11aagsx'\
		].isnull()), 'q11aagsx'] = 250
	#if never had intercsourse, and missing for partners age at first intercourse, replace missing with 999
	selected_set.loc[(selected_set.q11asx == 0) & (selected_set['q11aagpa'\
		].isnull()), 'q11aagpa'] = 250
	#if never intercourse and lifetime nr partners in missing, replace missing with 9999
	selected_set.loc[(selected_set.q11asx == 0) & (selected_set['q12totpa'\
		].isnull()), 'q12totpa'] = 250
	#if never intercourse and nr of new partners missing, replace missing with 999
	selected_set.loc[(selected_set.q11asx == 0) & (selected_set['q14newpa'\
		].isnull()), 'q14newpa'] = 250
	return selected_set

def process_screen(dataframe):
	"""take screening dataset and return a dataset of ID, birthdate, worst diagnosis, nr screenings
	and list of date and diagnosis"""

	data = dataframe
	#remove uninformative diagnoses
	data=(data[(data.diagnosis1 !=9)]) 
	data=(data[(data.diagnosis1 !=10)])
	data=(data[(data.diagnosis1 !=99)])

	#remove columns not used
	clean_data = data.drop(['type', 'diagnosis2', 'stage',\
		'lab_nr',  'reg', 'censordate'], axis=1)
	#gather diagnosis and date in a list
	clean_data['diagnosis_list'] = list(clean_data['diagnosisdate']+ 
		': ' + clean_data['diagnosis1'].map(str))
	#make new attribute from giving worst diagnosis for each ID
	clean_data['Worst_diagnosis'] = clean_data.groupby(['ID'\
		])['diagnosis1'].transform('max')
	clean_data = clean_data.drop(['diagnosisdate','diagnosis1'], axis=1)
	#make new attribute of number of screenings for each ID
	clean_data['nr_screenings'] = clean_data.groupby(by='ID'\
		)['ID'].transform('count')
	#create single row for each ID by gathering the diagnoses in a list
	clean_data=clean_data.groupby(['ID', 'Worst_diagnosis', \
		'nr_screenings']).agg(lambda col: ';'.join(col))
	clean_data['diagnosis_list'] = clean_data['diagnosis_list'].tolist()
	#avoid multiindexing of dataframe
	clean_data.reset_index(level=['ID', 'Worst_diagnosis', \
		'nr_screenings'], inplace=True)
	return clean_data

def make_agecolumn(dataframe, column1, column2, columnnew='Age'):
	"""creates a new column from difference of two columns with dates"""
	list1= dataframe[column1].values.tolist()
	list2= dataframe[column2].values.tolist()
	liste1b = [dt.datetime.strptime(date, '%d.%m.%Y').date() for date in list1]
	liste2b = [dt.datetime.strptime(date, '%d.%m.%Y').date() for date in list2]
	Ages =  [relativedelta(x,y).years for x,y in zip(liste2b, liste1b)]
	dataframe[columnnew] = np.array(Ages)
	return dataframe

def make_year(dataframe, column, columnnew='year'):
	"""creates a new column with years from a column with dates"""
	liste1 = dataframe[column].values.tolist()
	liste1b = [dt.datetime.strptime(date, '%d.%m.%Y').date() for date in liste1]
	dataframe[columnnew] =  [x.year for x in liste1b]
	return dataframe

def first_diag(dataframe, column= 'diagnosis_list', columnnew='first_diag'):
	"""return dates of first diagnosis for each list of diagnoses"""
	liste1 = dataframe[column].values.tolist()
	listt=[]
	for i in liste1:
		match = re.findall(r'\d{2}.\d{2}.\d{4}', i)
		early_date = min(match)
		listt.append(early_date)
	dataframe[columnnew] = listt
	return dataframe

def last_diag(dataframe, column='diagnosis_list', columnnew='last_diag'):
	"""return last diagnosis for each list of diagnoses"""
	liste1 = dataframe[column].values.tolist()
	listt=[]
	for i in liste1:
		match = re.findall(r'\d{2}.\d{2}.\d{4}', i)
		late_date = max(match)
		listt.append(late_date)
	dataframe[columnnew] = listt
	return dataframe

def diagnose_dict(dataframe, column='diagnosis_list', columnnew='diagnose_dict'):
	"""return a new column with dictionary of diagnose date as key with diagnosis as value"""
	liste1 = dataframe[column].values.tolist()
	listt=[]
	for i in liste1:
		dict1=dict((k,int(v)) for k, v in (e.split(':') for e in i.split(';')))
		listt.append(dict1)
	dataframe[columnnew] = listt	
	return dataframe

def diagnose_dict_y(dataframe, column='diagnosis_list', columnnew='diagnose_dict_y'):
	"""return a new column with dictionary of diagnose year as key with diagnosis as value"""
	dlist=dataframe[column]
	d2list=[]
	for i in dlist:
		listt=[]
		nlist=list((k,int(v)) for k, v in (e.split(':') for e in i.split(';')))
		#print nlist
		for j in nlist:
			y= dt.datetime.strptime(j[0],'%d.%m.%Y').year
			diag=j[1]
			listt.append([y,diag])
		#print listt
		d=defaultdict(list)
		for k,v in listt:
			d[k].append(v)
		E=dict(d)
		d2list.append(E)
	dataframe[columnnew] = d2list
	return dataframe


def last_diagnoses(dataframe, column='diagnose_dict_y', nryears=5):
	"""return list of diagnoses of the last five years"""
	nlist= dataframe[column]
	tlist=[]
	plist=[]
	mlist=[]
	for i in nlist:
		#print i 
		last_year=max(i.keys())
		years=i.keys()
		listt=[]
		for j in years:
			if j >= (last_year-nryears):
				listt.append(j)
		#print listt
		list2=[]
		for z in listt:
			for l in i[z]:
				#rather have HPV diagnosis than normal
				list2.append(l)
				# if l!= 11:
				# 	list2.append(l)
				# else:
				# 	list2.append(0)
		#print list2
		tlist.append(list2)
		plist.append(max(list2))
		mlist.append(len(list2))
	dataframe['diag_last_years']=tlist
	dataframe['nrscreen__lastyears']=mlist
	dataframe['worst_last_years']=plist
	return dataframe

def select_numeric(dataframe):
	return dataframe.select_dtypes(exclude=[object]) #select all non-object-types

def true_labels(dataframe):
	"""Need a dataframe with 'Worst_diagnosis' and return set of clusterlabels without specified
	cancer type (true_labels) and with (true_labels2) for ground truth setting"""
	dataframe['true_labels']= dataframe['Worst_diagnosis']
	dataframe['true_labels2']= dataframe['Worst_diagnosis']
	true_list=[0,0,1,1,1,1,1,1,1,2,3,3,3,4,4,4]
	true_list2=[0,0,1,1,1,1,1,1,1,2,3,3,3,4,5,6]
	worst_list=[11,20,12,13,14,15,16,17,18,31,32,33,35,41,42,43]
	dataframe['true_labels']=dataframe['true_labels'].replace(worst_list, true_list)
	dataframe['true_labels2']=dataframe['true_labels2'].replace(worst_list, true_list2)
	return dataframe
	
def dist_cat(u,v):
	"""Measure distance between two values. Tested."""

	if u == v:
		return 0
	else:
		return 1

def dist_cont(u,v):
	"""continuous variables"""
	try:
		return abs(float(u)-v)/(max(u,v)+1)
	except:
		return dist_cat(u,v)

def distance_measure_rows(list1, list2, list_types=None):
	"""Summarize the distance between two lists of equal length. Tested."""

	dist=0
	if list_types == None:
		list_types = ['categorical']*len(list1)
	try:
		len(list1)==len(list_types)
	except:
		print 'len list types not like len list'
		raise
	for i in xrange(len(list1)):
		if list_types[i]=='categorical' or list_types[i]=='binary':
			dist += dist_cat(list1[i],list2[i])
		else:
			dist += (dist_cont(list1[i],list2[i]))
	return dist/float(len(list1))


def distance_measure_row2(series1, series2):

	dictionary={'categorical':['study','typeres','q2amarit','q3school','q4health','q5asmoke', 'q6bbeer',
	'q6bvodk','q6csixdr','c6aagdrk','q15risk','q15chla','q15herp','q15tric','q15gono','q21necc','worst_last_years'], 
 	'binary':['q7apregn','q8acontr','q9condom','q11asx','q16heard','q17hadgw','q20know'], 'continuous': 	
 	['c6b2beer','c6b3soda','c6b4rwin','c6b5wwin','q11aagsx','q11aagpa','q12totpa','q14newpa','q23heigh', 
 	'q23weigh','q9agecon','nr_screenings', 'surveyage', 'birthyear', 'first_diagyear', 'first_diagage',
       'last_diagyear', 'last_diagage', 'nrscreen__lastyears']}

	list1 = series1.get(dictionary['categorical']).dropna().values
	list2 = series2.get(dictionary['categorical']).dropna().values
	dist = map(dist_cat, list1, list2) 	

	list1 = series1.get(dictionary['binary']).dropna().values
	list2 = series2.get(dictionary['binary']).dropna().values
	dist += map(dist_cat, list1, list2) 	

	list1 = series1.get(dictionary['continuous']).dropna().values
	list2 = series2.get(dictionary['continuous']).dropna().values
	dist += map(dist_cont, list1, list2) 	
	return abs(sum(dist))/float(len(list1))

def distance_matrix(dataframe, list_types = None):
	"""Generate a distance matrix from a pandas dataframe.
	For a weighted matrix append a list of weights. Tested, very timeconsuming."""
	#normalize dataframe
	nr_observations = dataframe.shape[0] # nr rows
	result = np.zeros((nr_observations, nr_observations))
	try:
		dataframe.shape[1]==len(list_types)
	except:
		print "ensure list_types for all attributes are included"
		raise
	for i in xrange(0,nr_observations-1): # save memory with xrange
		for j in xrange(i,nr_observations):
			row_i = dataframe.iloc[i].values
			row_j = dataframe.iloc[j].values
			result[i][j] = distance_measure_rows(row_i, row_j, list_types)
			result[j][i] =result[i][j]
		#print 'finish iteration %d' % i 
			#gower: sum(abs(distance_measure_rows(row_i, row_j, w))/coldiff)
	return result

def distance_matrix2(dataframe):
	"""Generate a distance matrix from a pandas dataframe.
	For a weighted matrix append a list of weights. Tested, very timeconsuming."""
	#normalize dataframe
	nr_observations = dataframe.shape[0] # nr rows
	result = np.zeros((nr_observations, nr_observations))

	for i in xrange(0,nr_observations-1): # save memory with xrange
		for j in xrange(i,nr_observations):
			row_i = dataframe.iloc[i]
			row_j = dataframe.iloc[j]
			result[i][j] = distance_measure_row2(row_i, row_j)
			result[j][i] =result[i][j]
		#print 'finish iteration %d' % i 
		#gower: sum(abs(distance_measure_rows(row_i, row_j, w))/coldiff)
	return result

def pdist_matrix(dataframe, method = 'euclidean'):
	"""Input pandas dataframe with no missing values, output distance matrix.
	Not used because the distance metric method is not optimal for mixed datatypes. """
	
	#scaling the dataframe
	df=StandardScaler().fit_transform(dataframe)
	distances = pdist(df, metric=method) #many different methods in pdist
	DistMatrix = squareform(distances)
	return DistMatrix


def linkage_cluster(distance_matrix):
	"""cluster a preprocessed dataframe. Require scipy.
	The function is not bein used because of an output
	giving restricted possibilities for further analyses. """

	try:
		is_valid_dm(distance_matrix)
	except:
		print 'not valid distance matrix'
		raise
	cond_matrix = squareform(distance_matrix)
	linkage_matrix = linkage(cond_matrix, method='average')
	return linkage_matrix

def kproto(dataframe,cat_cols=[], nr_clusters = 11):
	""" Clustering using k-prototype algorithm optimal for mixed categorical
	and numerical data. Need a list of indexes for categorical columns. """

	kp = kprototypes.KPrototypes(n_clusters=nr_clusters, init='Cao', verbose=2)
	clusters = kp.fit_predict(dataframe.values, categorical=cat_cols)
	dataframe['kproto_cluster'] = clusters
	return dataframe

def DBSCAN_cluster(dataframe,eps=8, min_samples=2):
	"""cluster a preprocessed dataframe. Require Sklearn. Tested."""
	dframe=StandardScaler().fit_transform(dataframe)
	db=DBSCAN(eps=eps).fit(dframe)
	dataframe["DBSCAN_cluster"] = db.labels_
	return dataframe

def DBSCAN_cluster2(dataframe, distance_matrix, eps=8, min_samples=2):
	"""cluster a preprocessed dataframe with its square distance matrix. Require Sklearn. 
	Tested. """
	db=DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
	dataframe["DBSCAN_Cluster_ownmetric"] = db.fit(distance_matrix).labels_
	return dataframe

def kmeans_cluster(dataframe, cols, nr_clusters=8):
	"""Cluster preprocessed dataframe, OneHot-encodes categorical columns.
	Require Hypertools."""
	dataframe=pd.get_dummies(dataframe, prefix=cols, columns=cols)
	normalized = hyp.tools.normalize(dataframe)
	labels=hyp.tools.cluster(normalized, n_clusters=nr_clusters)
	dataframe['hypertools_Cluster'] = labels
	return dataframe

def pca_tsne(dataframe, metric='euclidean'):
	"""use dataframe of imputed and merged dataset, if metric
	is 'precomputed' input must be a distance matrix """
	dataframe = StandardScaler().fit(dataframe).transform(dataframe)
	X_pca = PCA(n_components=2).fit_transform(dataframe)
	tsne = TSNE(n_components=2, metric='euclidean', random_state=0,perplexity=50,verbose=1,n_iter=1500)
	X_tsne = tsne.fit_transform(dataframe)
	return X_pca, X_tsne

def parallel_coords(dataframe):
	"""use dataframe of imputed and merged dataset. Need proper scaling. """
	normalized = StandardScaler().fit(dataframe).transform(dataframe)
	scaling =(dataframe-dataframe.min())/(dataframe.max() -dataframe.min()) #x_norm
	parallel_coordinates(scaling, 'true_labels', colormap='gist_rainbow')
	plt.xticks(rotation=90)
	plt.subplots_adjust(bottom=0.3)
	return plt.show()

def andrews_curv(dataframe):
	plt.figure()
	andrews_curves(dataframe, 'true_labels', colormap='gist_rainbow')
	return plt.show()

def pie_chart(dataframe):
	dataframe.plot.pie(xticks=dataframe.columns, subplots=True)
	return plt.show()

def qual_measure(distance_matrix, labels_true, labels_pred):
	"""input distance matrix. Output rand and silhouette index. """

	rand = adjusted_rand_score(labels_true, labels_pred)
	silhouette = silhouette_score(distance_matrix,labels_pred,metric ='precomputed')
	return rand, silhouette

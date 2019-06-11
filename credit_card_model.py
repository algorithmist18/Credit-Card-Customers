"""
	
	Data set problem: Watch data with one and analyze with another

"""
#Importing libraries

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt  
import math
import scipy.stats as ss
from sklearn.preprocessing import MinMaxScaler, Normalizer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from scipy import linalg, mat, dot 
from collections import Counter
from dython.nominal import conditional_entropy, correlation_ratio, theils_u, cramers_v
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


#Handle NA values
def clean_data(df):

	for col in df.columns:
		if df[col].dtype == object:
			for i in range(0, 1000):
				if pd.isna(df[col][i]):
					print('{} {}'.format(col, i))
		else:
			df.fillna(df.mean(), inplace = True)
	return df

#Computes similarity of two matrices
def similarity(k1, k2):

	a1 = mat(k1)
	b1 = mat(k2)
	c = dot(a1, b1.T)/linalg.norm(a1)/linalg.norm(b1)
	print('\n\n\nMatrices #1 and #2 are {}{} similar.'.format(c[0][0]*100, '%'))

#Drop redundant columns
def drop_redundancy(df):

	df = df.loc[:, (df != 0).any(axis = 0)]
	return df

#Normalize data

def normalize(df):

	min_max = MinMaxScaler()
	df_org = df
	cols = ['Age', 'Job', 'Credit amount', 'Duration']
	for col in cols:
		df[col] = ((df[col] - df[col].min())/(df[col].max() - df[col].min()))
	drop_redundancy(df)
	return df

#Encoding categorical data

def categorize(df):

	for col in df.columns:
		if col == 'Sex' or col == 'Housing' or col == 'Saving accounts' or col == 'Checking account' or col == 'Purpose':
			dummies = pd.get_dummies(df[col])
			df.drop(col, axis = 1, inplace = True)
			df = df.merge(dummies, left_index = True, right_index = True)

	df.rename(columns = {'little_x' : 'little_saving', 'moderate_x' : 'mod_saving',
						'rich_x' : 'rich_saving', 'quite_rich' : 'qrich_saving', 'little_y' : 'little_checking', 'moderate_y' : 'mod_checking',
						'rich_y' : 'rich_checking'}, inplace = True)
	return df

#Visualize data

def visualize(df):

	tsne = TSNE(n_components = 2, random_state = 0)
	df_vis = tsne.fit_transform(df)

	for idx, cl in enumerate(np.unique(df_vis)):
		if(idx < 522):
			plt.scatter(x=df_vis[idx][0], y=df_vis[idx][1])

	#Applying KMeans to 9D data
	kmeans1 = KMeans(n_clusters = 2, random_state = 0).fit(df)
	k1 = kmeans1.labels_

	#Applying KMeans to 2D data
	kmeans2 = KMeans(n_clusters = 2, random_state=0).fit(df_vis)
	k2 = kmeans2.labels_
	print('\n\n\n')
	print(k2)

	plt.scatter(kmeans2.cluster_centers_[0][0], kmeans2.cluster_centers_[0][1], c='k')
	plt.scatter(kmeans2.cluster_centers_[1][0], kmeans2.cluster_centers_[1][1], c='k')

	plt.show()

	return k2

def correlate(df):

	corr = df.corr(method = 'spearman')
	data = []
	for x in corr['risk']:
		if x != 1 and x != -1:
			if x < 0:
				data.append(-x)
			else:
				data.append(x)
	max = -1
	idx = 0

	for i in range(0, len(data)):
		if data[i] > max:
			max = data[i]
			idx = i 

	print(corr['risk'])
	print('Maximum dependency = {:.2f}, Attribute : {}'.format(max, df.columns[idx]))

def jobs_genders(data):

	male = 0
	female = 0
	n_males = 0
	n_females = 0
	pos = [0, 3]

	for i in range(0, 1000):

		male += data['male'][i] * data['Job'][i]
		n_males += data['male'][i]
		female += data['female'][i] * data['Job'][i]
		n_females += data['female'][i]

	#print('{} {}'.format(n_males, n_females))
	l = {'male' : male/n_males, 'female' : female/n_females}
	plt.bar(pos, l.values(), width = 0.34)
	plt.xticks(pos, l.keys())
	plt.title('Average number of jobs taken gender-wise')
	plt.xlabel('Gender')
	plt.ylabel('Number of jobs')
	plt.show()

	print('On an average, men took {:.2f} jobs.\n'.format(l['male']))
	print('On an average, women took {:.2f} jobs.\n'.format(l['female']))

def duration_genders(df):

	male = 0
	female = 0
	n_males = 0
	n_females = 0
	pos = [0, 3]

	for i in range(0, 1000):
		male += df['male'][i] * df['Duration'][i]
		female += df['female'][i] * df['Duration'][i]
		n_males += df['male'][i]

	#print('{} {}'.format(n_males, 1000-n_males))
	l = {'male' : male/n_males, 'female' : female/(1000-n_males)}
	plt.bar(pos, l.values(), width = 0.25)
	plt.xticks(pos, l.keys())
	plt.text(10, l['male'], l['male'])
	plt.title('Average duration of loan gender-wise')
	plt.xlabel('Gender')
	plt.ylabel('Days')
	plt.show()

	print('On an average, men took {:.0f} days for a loan.\n'.format(l['male']))
	print('On an average, women took {:.0f} days for a loan.'.format(l['female']))

#Analyzing tendencies between housing and risk
def correlate_column_risk(column):

	elements = df[column].unique()
	good = []
	bad = []
	p = df.groupby(column)
	for element in elements:
		#print(element)
		d = p.get_group(element)
		c = d['Risk'].value_counts() #good = x, bad = y
		good.append(100*c['good']/(c['good']+c['bad']))
		bad.append(100*c['bad']/(c['good']+c['bad']))
		#print('{:.2f}{}'.format(100*c['good']/(c['good']+c['bad']), '%'))

	#Visualizing two sets of histograms
	idx = np.arange(len(p))
	width = 0.35
	fig = plt.figure()
	ax = fig.add_subplot(111)
	r1 = ax.bar(idx-0.15, good, width, color = 'royalblue')
	r2 = ax.bar(idx + width-0.15, bad, width, color = 'seagreen')
	plt.title('Relationship with {} and risk'.format(column))
	plt.xticks(idx, elements)
	plt.xlabel(column)
	plt.ylabel('Risk probability (%)')
	plt.show()

	#Computing maximum probability and minimum probability
	max = good[0]
	min = good[0]
	max_idx = 0
	min_idx = 0

	for i in range(1, len(good)):

		if good[i] > max:
			max = good[i]
			max_idx = i

		if good[i] < min:
			min = good[i]
			min_idx = i 

	#print('{} {}'.format(elements[max_idx], elements[min_idx]))
	return elements[max_idx], elements[min_idx]

def numerical_correlation_risk(df):

	good = []
	bad = []

	for i in range(0, 1000):
		if df['Risk'][i] == 'good':
			good.append(df['Age'][i])
		else:
			bad.append(df['Age'][i])
	plt.hist(good)
	plt.xlabel('Age')
	plt.ylabel('Number of non-defaulters')
	plt.show()

def correlation(df, y):

	#Co-relating between categorical and categorical variables
	for col in df.columns:
		if df[col].dtype == object:
			corr = theils_u(df[col], y)
			print('{} = {:.3f}'.format(col, corr))
		else:
			corr = correlation_ratio(y, df[col])
			print('{} = {:.3f}'.format(col, corr))

#Taking the data set as input
df = pd.read_csv('german_credit_classifier.csv', error_bad_lines = False)
df.drop(['Unnamed: 0'], axis = 1, inplace = True)

max, min = correlate_column_risk('Housing')
print('\nPeople having {} housing had highest percentage of safe loans, while people having {} housing had lowest percentage of safe loans.'.format(max, min))
max, min = correlate_column_risk('Purpose')
print('\nPeople using loan for {} had highest percentage of safe loans, while people using the loan for {} had lowest percentage for safe loans.'.format(max, min))
max, min = correlate_column_risk('Job')
print('\nPeople having {} job(s) had lowest percentage of defaults while people having {} job(s) had the highest.'.format(max, min))

y = df['Risk']
df.drop(['Risk'], axis = 1, inplace = True)

print('\nCorrelation of different variables with Risk:\n')
correlation(df, y)
print()
#Predictive modeling 

#Changing text data into numbers: assigning higher values to more rise
data = categorize(df)

#Job statistics of genders

jobs_genders(data)
duration_genders(data)

data = normalize(data)
data = drop_redundancy(data)

X = data.iloc[:,:]
y = y.map({'good' : 1, 'bad' : 0})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

print('\nApplying Logistic Regression to data:\n')
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)

print('Accuracy score = {:.2f}{}'.format(accuracy_score(predictions, y_test)*100, '%'))

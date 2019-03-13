# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 20:16:06 2019

@author: ELCOT
"""
import pandas as pd
from sklearn.feature_selection import SelectKBest
import numpy as array
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

#URL for loading the dataset
#Import chi2 for performing chi square test from sklearn.feature_selection import chi2

#URL for loading the dataset

url ="Lung_Cancer.csv"

#Define the attribute names

names = ['A1','A2','A3','A4','A5','A6','A7','A8','A9','A10','A11','A12','A13','A14','A15','A16','A17','A18','A19','A20','A21','A22','A23','A24','A25','A26','A27','A28','A29','A30','A31','A32','A33','A34','A35','A36','A37','A38','A39','A40','A41','A42','A43','A44','A45','A46','A47','A48','A49','A50','A51','A52','A53','A54','A55','A56','DECISION']

#Create pandas data frame by loading the data from URL

dataset = pd.read_csv(url, names=names)

#Create array from data values
#array = dataframe.values

dataset_factorize = dataset.apply(lambda X:pd.factorize(X)[0])

array = dataset_factorize.values

X = dataset_factorize.iloc[:].values
Y = dataset_factorize.iloc[:,-1].values

#Split the data into input and target

#X = array[:,0:25]

#Y = array[:,25]

#We will select the features using chi square

# feature extraction
model = LogisticRegression()
rfe = RFE(model, 30)
fit = rfe.fit(X, Y)
print("Num Features: %d", fit.n_features_)
print("Selected Features: %s", fit.support_)
print("Feature Ranking: %s" ,fit.ranking_)
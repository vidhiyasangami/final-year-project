# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 12:02:24 2019

@author: ELCOT
"""

import pandas as pd
import numpy
from sklearn.ensemble import ExtraTreesClassifier

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

model = ExtraTreesClassifier()
model.fit(X, Y)
print(model.feature_importances_)
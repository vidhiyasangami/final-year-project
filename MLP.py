# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 19:00:00 2019

@author: ELCOT
"""

import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn import metrics
#dataset
dataset = pd.read_csv("Lung_Cancer_Updated.csv")

#data preprosessing
dataset_factorize = dataset.apply(lambda x:pd.factorize(x)[0])

#splitting of data into features and labels
x = dataset_factorize.iloc[:].values
y = dataset_factorize.iloc[:,-1].values

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(x, y)



expected = y
predicted = clf.predict(x)

# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))
print(accuracy_score(expected, predicted))
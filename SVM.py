# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 10:47:26 2019

@author: ELCOT
"""
import pandas as pd
from sklearn.metrics import accuracy_score
#dataset
dataset = pd.read_csv("Lung_Cancer_Updated.csv")

#data preprosessing
dataset_factorize = dataset.apply(lambda x:pd.factorize(x)[0])

#splitting of data into features and labels
x = dataset_factorize.iloc[:].values
y = dataset_factorize.iloc[:,-1].values

from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.20)  

from sklearn.svm import SVC  
svclassifier = SVC(kernel='linear')  
svclassifier.fit(X_train, y_train)  

y_pred = svclassifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred))  
print(accuracy_score(y_test, y_pred))
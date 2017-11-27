# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 18:20:12 2017

@author: Naresh
"""
# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import DataSet
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

#Take care of Missing Values
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

#Take Care of Categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X = LabelEncoder()
X[:,0] = labelEncoder_X.fit_transform(X[:,0])
oneHotEncoder_X = OneHotEncoder(categorical_features=[0])
X = oneHotEncoder_X.fit_transform(X).toarray()
labelEncoder_Y = LabelEncoder()
y = labelEncoder_Y.fit_transform(y)

# Split into training and test set
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size= 0.2 , random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
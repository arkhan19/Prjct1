#importing the libraries

import numpy as np #mathmaticallibrary
import matplotlib.pyplot as plt #ploting mathmatical graphs
import pandas as pd #dataset management
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder

#importing the dataset
#specify working directory folder first
dataset = pd.read_csv('/Users/f3n1Xx/Documents/PycharmProjects/Prjct1/Data.csv')
#creating matrix of features
x = dataset.iloc[:, :-1].values #all columens of dataset, except the last col
y = dataset.iloc[:, 3].values #Last colm array

#missing data can cause problem. so we use some shit. Using sklearn Imputer
#Taking care of missing data
im = Imputer(missing_values='NaN', strategy='mean', axis=0)
im = im.fit(x[:, 1:3]) #excluding 2nd col
x[:, 1:3] = im.transform(x[:, 1:3])
#x matrix will be fixed now.......

#Categorical Variable containing anything with text, we just need numbers. So we will encode the categorical variables. We will use LabelEncoder from sklearn preprocessing
le_x = LabelEncoder()
x[:, 0]= le_x.fit_transform(x[:, 0]) #takes only first colm and transforms them
ohe = OneHotEncoder(categorical_features=[0]) #colm name of category
x = ohe.fit_transform(x).toarray()
le_y = LabelEncoder() #cannot use same label encoder as it was fitted to x in line24
y = le_y.fit_transform(y)










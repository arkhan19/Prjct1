#importing the libraries

import numpy as np #mathmaticallibrary
import matplotlib.pyplot as plt #ploting mathmatical graphs
import pandas as pd #dataset management
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


#importing the dataset ##specify working directory folder first ###creating matrix of features
dataset = pd.read_csv('/Users/f3n1Xx/Documents/PycharmProjects/Prjct1/Data.csv') #read.csv is also a thing XD only in R
x = dataset.iloc[:, :-1].values #all columens of dataset, except the last col
y = dataset.iloc[:, 3].values #Last colm array

#missing data can cause problem. so we use some shit. Using sklearn Imputer ##Taking care of missing data
im = Imputer(missing_values='NaN', strategy='mean', axis=0)
im = im.fit(x[:, 1:3]) #excluding 2nd col
x[:, 1:3] = im.transform(x[:, 1:3]) #x matrix will be fixed now.......

#Categorical Variable containing anything with text, we just need numbers. So we will encode the categorical variables. We will use LabelEncoder from sklearn preprocessing
le_x = LabelEncoder()
x[:, 0]= le_x.fit_transform(x[:, 0]) #takes only first colm and transforms them
ohe = OneHotEncoder(categorical_features=[0]) #colm name of category #Needed when we need to encode a dependent variable. Very important. Only Hot Encode Dependent Variable not Independent.
x = ohe.fit_transform(x).toarray()
le_y = LabelEncoder() #cannot use same label encoder as it was fitted to x in line24
y = le_y.fit_transform(y)


#We will divide the dataset into training set and testing set.
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)

#Now how to Feature Scale because ED will be fucked up. One feature will dominate other feature. So lets feature scale using Standardisation or Normalisation #No need to take ss for y array as it is just 0 and 1
ssx = StandardScaler()
x_train = ssx.fit_transform(x_train)
x_test = ssx.transform(x_test)








#importing the libraries

import numpy as np #mathmaticallibrary
import matplotlib.pyplot as plt #ploting mathmatical graphs
import pandas as pd #dataset management
from sklearn.preprocessing import Imputer

#importing the dataset
#specify working directory folder first
dataset = pd.read_csv('Data.csv')
#creating matrix of features
x = dataset.iloc[:, :-1].values #all columens of dataset, except the last col
y = dataset.iloc[:, :3].values

#missing data can cause problem. so we use some shit. Using sklearn Imputer
#Taking care of missing data
im = Imputer(missing_values='NaN', strategy='mean', axis=0)
im = im.fit(x[:, 1:3]) #excluding 2nd col
x[:, 1:3] = im.transform(x[:, 1:3])
#x matrix will be fixed now.......





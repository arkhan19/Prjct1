import pandas as pd
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.sparse import *

# ....PreProcessing....
# Importing Data
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, 3:5].values  # 2nd and 3rd col
y = dataset.iloc[:, 5].values  # Last column array
Y = y.reshape(-1, 1)  # Needed Here

# Missing Values
dataset = dataset.replace(np.nan, ' ', regex=True)  # Only works on dataframe object not on ndarray

corpus = []  # For Question 1
corpus2 = []  # For Question 2

for i in range(0, 300000):
    # Cleaning the dataset from all unless characters
    question = re.sub('[^a-zA-Z]', ' ', dataset['question1'][i])  # removes everything other than a-z or A-Z from
    # Question [1]
    question = question.lower()  # syntax is different, causes confusion. #HL
    # nltk.download('stopwords')
    question = question.split()
    ps = PorterStemmer()
    question = [ps.stem(word) for word in question if word not in set(stopwords.words('english'))]  # will keep the root
    # words, using porterstemmer helps in creating bag of words
    question = ' '.join(question)  # join all the elements of the lists. Final step
    corpus.append(question)

for i in range(0, 300000):
    # Cleaning the dataset from all uness characters
    question = re.sub('[^a-zA-Z]', ' ', dataset['question2'][i])  # removes everything other than a-z or A-Z from
    # Question [1]
    question = question.lower()  # syntax is different, causes confusion. #HL
    # nltk.download('stopwords')
    question = question.split()
    ps = PorterStemmer()
    question = [ps.stem(word) for word in question if not word in set(stopwords.words('english'))]  # will keep the root
    # words, using porterstemmer helps in creating bag of words
    question = ' '.join(question)  # join all the elements of the lists. Final step
    corpus2.append(question)


# Tokenization
cv = CountVectorizer(max_features=1500)
X1 = cv.fit_transform(corpus).toarray()
X2 = cv.fit_transform(corpus2).toarray()
X_D = np.concatenate((X1,X2),axis = 1)

# Splitting the dataset into Training set and Test set
X_train, X_test, Y_train, Y_test = train_test_split(X_D, Y, test_size=0.20, random_state=0)



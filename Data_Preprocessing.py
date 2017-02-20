import pandas as pd
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords


#....PreProcessing....#
#Importing Data
dataset = pd.read_csv('/Users/f3n1Xx/Documents/PycharmProjects/Prjct1/Data.csv')
X = dataset.iloc[:, 3:5].values #2nd and 3rd col
y = dataset.iloc[:, 5].values #Last colm array
#X = x.reshape(-1,1) #Not Needed
Y = y.reshape(-1,1) #Needed


corpus = []
corpus2 = []
for i in range (0, 300000):
    #Cleaning the dataset from all uness characters
    question = re.sub('[^a-zA-Z0-9]',' ', dataset['question1'][i]) #removes everything other than a-z or A-Z from Question [1]
    question = question.lower() #syntax is different, causes confusion. #HL
    #nltk.download('stopwords')
    question = question.split()
    ps = PorterStemmer()
    question = [ps.stem(word) for word in question if word not in set(stopwords.words('english'))] #will keep the root words, using porterstemmer helps in creating bag of words
    question = ' '.join(question) #join all the elements of the lists. Final step
    corpus.append(question)

for i in range (0, 300000):
    #Cleaning the dataset from all uness characters
    question = re.sub('[^a-zA-Z]',' ', dataset['question2'][i]) #removes everything other than a-z or A-Z from Question [1]
    question = question.lower() #syntax is different, causes confusion. #HL
    #nltk.download('stopwords')
    question = question.split()
    ps = PorterStemmer()
    question = [ps.stem(word) for word in question if not word in set(stopwords.words('english'))] #will keep the root words, using porterstemmer helps in creating bag of words
    question = ' '.join(question) #join all the elements of the lists. Final step
    corpus2.append(question)

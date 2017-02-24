# NaiveBayes
from Data_Preprocessing import *
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


# Fitting Naive Bayes to the Training Set
clf = GaussianNB()
clf.fit(X_train, Y_train)

# Predicting the Test set results
Y_pred = clf.predict(X_test)

# Accuracy
Accuracy = accuracy_score(Y_test, Y_pred, normalize=False, sample_weight=None)


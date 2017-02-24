# NaiveBayes
from Data_Preprocessing import *
# from sklearn.naive_bayes import GaussianNB
# from sklearn.metrics import accuracy_score
#
#
# # Fitting Naive Bayes to the Training Set
# clf = GaussianNB()
# clf.fit(X_train, Y_train)
#
# # Predicting the Test set results
# Y_pred = clf.predict(X_test)
#
# # Accuracy
# Accuracy = accuracy_score(Y_test, Y_pred, normalize=False, sample_weight=None)

import keras
from keras.models import Sequential #initialise neural network
from keras.layers import Dense  #building of layers
from sklearn.metrics import accuracy_score


# Initializer for ANN
classifier = Sequential()

# Adding I/P - Hidden - O/P layers
classifier.add(Dense(output_dim=55, init='uniform', activation='relu', input_dim= 110))
classifier.add(Dense(output_dim=55, init='uniform', activation='relu'))  # No Input Dim for 2nd Hidden layer
classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))  # Output Layer; Sigmoid function will
                                                                            # produce probablites for Output layer

# ANN Compilation
classifier.compile(optimizer='adam', metrics=['accuracy'], loss ='binary_crossentropy')

#Fitting
classifier.fit(X_train, Y_train, batch_size= 19, nb_epoch=99, shuffle=True)

#Prediction
Y_pred = classifier.predict(X_test)

# Accuracy
Accuracy = accuracy_score(Y_test, Y_pred, normalize=False, sample_weight=None)
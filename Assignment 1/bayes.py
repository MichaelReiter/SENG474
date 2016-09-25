#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
import logging
import sys
from time import time


class MyBayesClassifier():
    # For graduate and undergraduate students to implement Bernoulli Bayes
    def __init__(self, smooth=1):
        self._smooth = smooth
        self._feat_prob = []
        self._class_prob = []
        self._Ncls = []
        self._Nfeat = []

    def classifier_probabilities(self, y):
        return [float(self.occurrences_of_element(element, y)) / len(y) for element in range(self._Ncls)]

    def number_of_unique_elements(self, x):
        return len(set(x))

    def occurrences_of_element(self, element, y):
        count = 0
        for i in y:
            if i == element:
                count += 1
        return count

    def p_of_x_given_y(self, x, y):
        return p_of_x_and_y(x, y) / p_of_y(y)

    def p_of_x_and_y(self, x, y):
        return 0

    def p_of_y(self, y):
        return self._class_prob[y]

    def train(self, X, y):
        self._Ncls = self.number_of_unique_elements(y)
        self._Nfeat = len(X[0])
        self._class_prob = self.classifier_probabilities(y)
        self._feat_prob = [0] * self._Nfeat

    def predict(self, X):
        # This is just a place holder so that the code still runs.
        # Your code goes here.
        return np.zeros([X.shape[0],1])

class MyMultinomialBayesClassifier():
    # For graduate students only
    def __init__(self, smooth=1):
        self._smooth = smooth # This is for add one smoothing, don't forget!
        self._feat_prob = []
        self._class_prob = []
        self._Ncls = []
        self._Nfeat = []

    # Train the classifier using features in X and class labels in Y
    def train(self, X, y):
        # Your code goes here.
        return

    # should return an array of predictions, one for each row in X
    def predict(self, X):
        # This is just a place holder so that the code still runs.
        # Your code goes here.
        return np.zeros([X.shape[0],1])
        


""" 
Here is the calling code

"""

categories = [
        'alt.atheism',
        'talk.religion.misc',
        'comp.graphics',
        'sci.space',
    ]
remove = ('headers', 'footers', 'quotes')

data_train = fetch_20newsgroups(subset='train', categories=categories,
                                shuffle=True, random_state=42,
                                remove=remove)

data_test = fetch_20newsgroups(subset='test', categories=categories,
                               shuffle=True, random_state=42,
                               remove=remove)
print('data loaded')

y_train, y_test = data_train.target, data_test.target

print("Extracting features from the training data using a count vectorizer")
t0 = time()

vectorizer = CountVectorizer(stop_words='english', binary=False)#, analyzer='char', ngram_range=(1,3))
X_train = vectorizer.fit_transform(data_train.data).toarray()
X_test = vectorizer.transform(data_test.data).toarray()
feature_names = vectorizer.get_feature_names()

alpha = 1
clf = MyBayesClassifier(alpha)
clf.train(X_train,y_train)
y_pred = clf.predict(X_test)
print 'alpha=%i accuracy = %f' %(alpha, np.mean((y_test-y_pred)==0))


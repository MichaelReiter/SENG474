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
        self._feature_probabilities = []
        self._classifier_probabilities = []
        self._number_of_classifiers = []
        self._number_of_features = []
        self._feature_counts = {}

    def classifier_probabilities(self, y):
        return [float(self.occurrences_of_classifier(classifier, y)) / len(y) for classifier in range(self._number_of_classifiers)]

    def number_of_unique_elements(self, x):
        return len(set(x))

    def occurrences_of_classifier(self, classifier, y):
        count = 0
        for i in y:
            if i == classifier:
                count += 1
        return count

    def p_of_feature_given_classifier(self, feature, classifier):
        # times it was feature x and classifier y
        occurrences_of_feature = self._feature_counts[classifier].get(feature, 0)
        # all times it was classifier y
        occurrences_of_all_features = sum(self._feature_counts[classifier].values())
        # P(feature|classifier) = P(feature and classifier) / P(classifier)
        return occurrences_of_feature / occurrences_of_all_features

    def p_of_classifier(self, classifier):
        return self._classifier_probabilities[classifier]

    def train(self, X, y):
        self._number_of_classifiers = self.number_of_unique_elements(y)
        self._number_of_features = len(X[0])
        self._classifier_probabilities = self.classifier_probabilities(y)
        self._feature_probabilities = [0] * self._number_of_features
        
        for classifier in range(self._number_of_classifiers):
            self._feature_counts[classifier] = [0] * self._number_of_features

        for row, classifier in zip(X, y):
            for feature in range(len(row)):
                self._feature_counts[classifier][feature] += float(row[feature])




    def predict(self, X):
        score = 0
        prediction = None

        # outer loop is argmax
        for classifier in range(_number_of_classifiers):
            # bayes = P(y)
            bayes = p_of_classifier(classifier)
            for row in X:    
                for feature in row:
                    bayes *= p_of_feature_given_classifier(feature, classifier)



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


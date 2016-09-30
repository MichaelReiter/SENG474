#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
import logging
import sys
from time import time
from math import exp, log

class MyBayesClassifier():
    def __init__(self, smooth=1):
        self._smooth = smooth
        self._all_labels = []
        self._feature_counts = {}
        self._label_counts = {}
        self._label_and_feature_counts = {}
        self._length_of_training_set = 0.0

    def p_of_feature(self, feature):
        return self._feature_counts[feature] / self._length_of_training_set

    def p_of_feature_given_label(self, feature, label):
        count_of_label_and_feature = self._label_and_feature_counts[label][feature]
        count_of_label = self._label_counts[label]
        return count_of_label_and_feature / count_of_label

    def bayes(self, row, label):
        total = self._label_counts[label] / self._length_of_training_set
        for feature in range(len(row)):
            p = (self.p_of_feature_given_label(feature, label) / self.p_of_feature(feature) * row[feature]) + self._smooth
            total *= p
        return total / (self.p_of_feature(feature) + 2.0)

    def train(self, X, y):
        # Initialize class variables
        self._all_labels = set(y)
        self._length_of_training_set = len(y)

        for feature in range(len(X[0])):
            self._feature_counts[feature] = 0.0

        for label in self._all_labels:
            self._label_counts[label] = 0.0
            self._label_and_feature_counts[label] = {}

        # Count feature and label information
        for row, label in zip(X, y):
            self._label_counts[label] += 1.0
            for feature in range(len(row)):
                self._feature_counts[feature] += float(row[feature])
                if feature not in self._label_and_feature_counts[label]:
                    self._label_and_feature_counts[label][feature] = 0.0
                self._label_and_feature_counts[label][feature] += row[feature]

    def predict(self, X):
        result = []

        for row in X:
            best_probability = 0.0
            classification = None

            for label in self._all_labels:
                p_of_label_given_row = self.bayes(row, label)
                print p_of_label_given_row
                if p_of_label_given_row > best_probability:
                    best_probability = p_of_label_given_row
                    classification = label

            result.append(classification)

        print result
        return result

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

# , analyzer='char', ngram_range=(1,3))
vectorizer = CountVectorizer(stop_words='english', binary=True)

X_train = vectorizer.fit_transform(data_train.data).toarray()
X_test = vectorizer.transform(data_test.data).toarray()
feature_names = vectorizer.get_feature_names()

alpha = 1
clf = MyBayesClassifier(alpha)
clf.train(X_train, y_train)
print X_test
y_pred = clf.predict(X_test)
print 'alpha=%i accuracy = %f' % (alpha, np.mean((y_test-y_pred) == 0))

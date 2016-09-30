#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Michael Reiter
# V00831568

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
import logging
import sys
from time import time
from math import log

class MyBayesClassifier():
    def __init__(self, smooth=1.0):
        self._smooth = smooth
        self._all_labels = []
        self._label_counts = {}
        self._label_and_feature_counts = {}
        self._length_of_training_set = 0.0

    # compute P(label)
    def p_of_label(self, label):
        count_of_label = self._label_counts[label] + self._smooth
        count_of_rows = self._length_of_training_set + (self._smooth * len(self._all_labels))
        return count_of_label / count_of_rows

    # compute P(feature|label)
    def p_of_feature_given_label(self, feature, label):
        count_of_label_and_feature = self._label_and_feature_counts[label][feature] + self._smooth
        count_of_label = self._label_counts[label] + (self._smooth * 2.0)
        return count_of_label_and_feature / count_of_label

    # compute P(label|row of features)
    def p_of_label_given_row(self, row, label):
        total = log(self.p_of_label(label))
        for feature in range(len(row)):
            probability = self.p_of_feature_given_label(feature, label)
            total += log(probability) * row[feature]
        return total

    # Initialize class variables then count feature and label information
    def train(self, X, y):
        self._all_labels = set(y)
        self._length_of_training_set = float(len(X))

        for label in self._all_labels:
            self._label_counts[label] = 0.0
            self._label_and_feature_counts[label] = {}

        for row, label in zip(X, y):
            self._label_counts[label] += 1.0
            for feature in range(len(row)):
                if feature not in self._label_and_feature_counts[label]:
                    self._label_and_feature_counts[label][feature] = 0.0
                self._label_and_feature_counts[label][feature] += float(row[feature])

    # Return an array of classifications corresponding to the test set
    def predict(self, X):
        result = [0] * len(X)

        # Argmax
        for i, row in enumerate(X):
            best_probability = -float('inf')
            classified_label = None

            # Apply Bayes theorem
            for label in self._all_labels:
                bayes = self.p_of_label_given_row(row, label)
                if bayes > best_probability:
                    best_probability = bayes
                    classified_label = label

            result[i] = classified_label

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

vectorizer = CountVectorizer(stop_words='english', binary=True)

X_train = vectorizer.fit_transform(data_train.data).toarray()
X_test = vectorizer.transform(data_test.data).toarray()
feature_names = vectorizer.get_feature_names()

alpha = 1
clf = MyBayesClassifier(alpha)
clf.train(X_train, y_train)
y_pred = clf.predict(X_test)
print 'alpha=%i accuracy = %f' % (alpha, np.mean((y_test-y_pred) == 0))

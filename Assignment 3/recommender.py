#!/usr/bin/env python
# -*- coding: utf-8 -*-
import scipy.stats as scistats
import pickle
import numpy as np
import math
from sklearn.feature_extraction.text import CountVectorizer

# This helpful method will print the jokes for you
def print_jokes(p, n_jokes):
    for i in range(N_clusters):
        print '\n------------------------------'
        print '     cluster ' + str(i) + '   '
        print '------------------------------'
        for j in idx[p == i][:n_jokes]:
            print jokes[j] + '\n'

data = pickle.load(open('jester_train_test.p'))
# rows are users, columns are jokes
data_test = data['data_test']
data_train = data['data_train']

# Here are the joke texts, idx is used to index in print_jokes
jokes = pickle.load(open('jokes.pck', 'rb'))
idx = np.arange(len(jokes))

# Here we compute the similarity matrix. Refer to slides
# to see how to use the calculated numbers
d_user_user = np.zeros([data_test.shape[0],data_train.shape[0]])
d_item_item = np.zeros([data_train.shape[1],data_train.shape[1]])

# These calculations take a while, so you might want to save the matrices
# and load them each time instead of computing from scratch.
for i in range(data_test.shape[0]):
    ri = data_test[i]
    for j in range(data_train.shape[0]):
        rj = data_train[j]
        # use elements for which both users have given ratings
        inds = np.logical_and(ri != 0, rj != 0)
        # some users gave the same rating to all jokes :(
        if np.std(ri[inds])==0 or np.std(rj[inds])==0:
            continue
        d_user_user[i,j] = scistats.pearsonr(ri[inds],rj[inds])[0]

for i in range(data_train.shape[1]):
    ri = data_train[:,i]
    d_item_item[i,i] = 1
    for j in range(i+1, data_train.shape[1]):
        rj = data_train[:,j]
        # consider only those users who have given ratings
        inds = np.logical_and(ri != 0, rj != 0)
        d_item_item[i,j] = scistats.pearsonr(ri[inds],rj[inds])[0]
        d_item_item[j,i] = d_item_item[i,j]

# If the rating is 0, then the user has not rated that item
# You can use this mask to select for rated or unrated jokes
d_mask = (data_test == 0)

def predict_user_user_rating(u, i, ratings, similarity):
    numerator = 0
    denominator = 0
    # for each user v
    for v in range(ratings.shape[0]):
        # if user v rated item i and u is not v
        if ratings[v][i] > 0 and u != v:
            numerator += (ratings[v][i] * similarity[u][v])
            denominator += similarity[u][v]

    # return 0 if no user has rated joke i
    if denominator == 0:
        return 0
    else:
        return float(numerator) / float(denominator)

def get_user_user_root_mean_squared_error(ratings, similarity):
    numerator = 0
    denominator = 0
    # for each user u
    for u in range(ratings.shape[0]):
        for joke in range(ratings.shape[1]):
            rating_actual = ratings[u][joke]
            # if it was rated
            if rating_actual > 0:
                rating_predicted = predict_user_user_rating(u, joke, ratings, similarity)
                numerator += ((rating_actual - rating_predicted) ** 2)
                denominator += 1

    return math.sqrt(float(numerator) / float(denominator))

def predict_item_item_rating(u, i, ratings, similarity):
    numerator = 0
    denominator = 0
    # for each joke j
    for j in range(ratings.shape[1]):
        # if joke j was rated and i is not j
        if ratings[u][j] > 0 and i != j:
            numerator += (ratings[u][j] * similarity[i][j])
            denominator += similarity[i][j]

    return float(numerator) / float(denominator)

def get_item_item_root_mean_squared_error(ratings, similarity):
    numerator = 0
    denominator = 0
    # for each user u
    for u in range(ratings.shape[0]):
        for joke in range(ratings.shape[1]):
            rating_actual = ratings[u][joke]
            # if it was rated
            if rating_actual > 0:
                rating_predicted = predict_item_item_rating(u, joke, ratings, similarity)
                numerator += ((rating_actual - rating_predicted) ** 2)
                denominator += 1

    return math.sqrt(float(numerator) / float(denominator))

# ------------------- user user --------------------- #
print "\n*******User User similarity*******"
rmse = get_user_user_root_mean_squared_error(data_test, d_user_user)
for u in range(data_test.shape[0]):
    joke_rec = 0
    best_rating = 0
    for joke in range(data_test.shape[1]):
        # if the joke is unrated
        if data_test[u][joke] == 0:
            prediction = predict_user_user_rating(u, joke, data_test, d_user_user)
            if prediction > best_rating:
                best_rating = prediction
                joke_rec = joke

    print "Test instance", str(u), "Recommend joke:", str(joke_rec)

print "RMSE for all predictions:", str(rmse)

# ------------------ item item ----------------------- #
print "\n*******Item Item similarity*******"
rmse = get_item_item_root_mean_squared_error(data_test, d_item_item)
for u in range(data_test.shape[0]):
    joke_rec = 0
    best_rating = 0
    for joke in range(data_test.shape[1]):
        # if the joke is unrated
        if data_test[u][joke] == 0:
            prediction = predict_item_item_rating(u, joke, data_test, d_item_item)
            if prediction > best_rating:
                best_rating = prediction
                joke_rec = joke

    print "Test instance", str(u), "Recommend joke:", str(joke_rec)

print "RMSE for all predictions:", str(rmse)

'''
# ------- Clustering question  -------- #
N_clusters = 10
# ------- jokes clustering based on user votes  -------- #
print "\n*******Clustering based on user votes*******"

####################################
# put your code here
####################################
# Here you should apply KMeans clustering to the ratings in the
# *TRAINING DATA* <--- very important!  
# You should *not* use the test data for this question at all.
# Replace jokes_cluster with the output of your clustering.
# Below is a random clustering that ignores the joke text
# so the code still works.
####################################

jokes_cluster = np.random.randint(low=N_clusters,size=data_train.shape[1])
print_jokes(jokes_cluster, 3)
print jokes_cluster

# ------ jokes clustering based on text similariy ------#
print "\n*******Clustering based on text similarity*******"
# Use the following vectorizer to turn the text of the jokes
# into vectors that can be clustered
vectorizer = CountVectorizer(stop_words='english',
                            max_df=0.95,
                            min_df=0.05,
                            analyzer='char',
                            ngram_range = [2,5], binary=True)
####################################
# put your code here
####################################
# Here you should apply the vectorizer to the text of the jokes
# and then use the vector representation as input to KMeans clustering.
# Replace jokes_cluster with the output of your clustering.
# Below is a random clustering that ignores the joke text
# so the code still works.
jokes_cluster = np.random.randint(low=N_clusters,size=data_train.shape[1])
print_jokes(jokes_cluster, 3)
print jokes_cluster
'''
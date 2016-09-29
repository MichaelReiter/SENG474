# Cross validation

from sklearn import cross_validation
from sklearn import tree
from sklearn import datasets

iris = datasets.load_iris()

data_train, data_test, labels_train, labels_test = cross_validation.train_test_split(
  iris.data,
  iris.target,
  test_size=0.4,
  random_state=0
)

classifier = tree.DecisionTreeClassifier().fit(data_train, labels_train)

print classifier.score(data_test, labels_test)








# Gaussian Naive Bayes

from sklearn.naive_bayes import GaussianNB

# height (cm), weight (lbs), shoe size
data = [
  [175, 180, 30],
  [175, 175, 22],
  [155, 150, 24],
  [125, 155, 22],
  [150, 145, 18],
  [177, 150, 20],
  [200, 150, 22]
]

# 1 = man, 2 = woman
labels = [
  1,
  1,
  1,
  2,
  2,
  2,
  2
]

classifier = GaussianNB().fit(data, labels)

test = [
  [180, 160, 20],
  [125, 150, 18]
]

# classifies the first as a man, and the second as a woman
print classifier.predict(test)

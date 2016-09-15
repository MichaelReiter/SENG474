from sklearn import tree

data = [
  [1,1,1],
  [2,2,2],
  [3,3,3],
]

training = [1,2,3]

classifier = tree.DecisionTreeClassifier()

classifier = classifier.fit(data, training)

test = [
  [1,1,2],
  [2,2,3],
  [4,4,4],
]

print classifier.predict(test)

from sklearn import svm

data = [
  (1,1), (1,3), (1,5), (1,7),
  (2,2), (2,4), (2,6), (2,8),
  (5,1), (5,3), (5,5), (5,7),
  (6,2), (6,4), (6,6), (6,8)
]

categories = [
  0, 0, 0, 0, 0, 0, 0, 0,
  1, 1, 1, 1, 1, 1, 1, 1
]

classifier = svm.SVC(kernel='linear').fit(data, categories)

test = [
  [2,3],
  [3,3],
  [4,3],
]

print classifier.predict(test)

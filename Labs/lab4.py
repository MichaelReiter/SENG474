from sklearn import linear_model

# Linear Regression

data = [
  [0, 0],
  [1, 1],
  [2, 2]
]

labels = [
  0,
  1,
  2
]

model = linear_model.LinearRegression().fit(data, labels)

# print model.coef_

test = [
  [1, 1],
  [4, 4],
  [0, 3]
]

print model.predict(test)


# -----------------------------------------------------------

# Perceptrons

data = [
  [0,0,0],
  [0,0,1],
  [0,1,0],
  [0,1,1],
  # [1,0,0],
  [1,0,1],
  [1,1,0],
  [1,1,1],
]

labels = [
  -1,
  -1,
  -1,
   1,
  # -1,
   1,
   1,
   1,
]

model = linear_model.Perceptron().fit(data, labels)

test = [
  [1,0,0]
]

print model.predict(test)

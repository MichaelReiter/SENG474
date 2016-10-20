import theano.tensor as T
from theano import function

vec = T.dvector()
scal = T.dscalar()
sv_add = vec + scal

f_add = function([vec, scal], sv_add)

print f_add([1,2,3,4], 2)

x = T.dmatrix()
y = T.dmatrix()
z = x + y

f = function([x, y], z)
print f([[1,1], [2,2]], [[3,3], [4,4]])

x, y = T.dmatrices('x', 'y')
s = 1/(1 + T.exp(-x))

logistic = function([x], s)
print logistic([[0,1], [-1, -2]])

a, b = T.dmatrices('a', 'b')
diff = a - b
abs_diff = abs(diff)
diff_squared = diff**2

f = function([a, b], [diff, abs_diff, diff_squared])
print f([[1,1], [1,1]], [[0,1], [2,3]])

from theano import shared

state = shared(0)
inc = T.iscalar('inc')
f = function([inc], state, updates=[(state, state+inc)])

f(1)
print state.get_value()

import theano
import numpy

rng = numpy.random
N = 400
features = 700

D = (rng.randn(N, features), rng.randint(size=N, low=0, high=2))
training_steps = 10000

x = T.dmatrix('x')
y = T.dmatrix('y')

weights = shared(rng.randn(features), name='weights')
b = shared(0., name='b')

p_1 = 1/(1 + T.exp(-T.dot(x, weights) - b))
prediction = p_1 > 0.5
xent = -y * T.log(p_1) - (1 - y)*T.log(1 - p_1)
cost = xent.mean() + 0.01 * (weights**2).sum()
gw, gb = T.grad(cost, [weights, b])

train = function(inputs=[x,y], outputs=[prediction, xent], updates=([weights, weights - 0.1*gw], (b, b-0.1*gb)))
predict = function(inputs=[x], outputs=[prediction])

for i in range(training_steps):
  pred, err = train(D[0], D[1])

# print predict(D[0])

import myad
import numpy as np
from myad import ops
from myad import F

# x = np.ones((1, 3))
# y = np.ones((3, 1))

x = np.array([[1, 2, 3]])
y = np.array([[1, 2, 3]]).T
# x = np.array([[1, 2, 3], [1, 2, 3]]).T
# y = np.array([[1, 2, 3], [1, 2, 3]])

# out = F.dot(x, y)

# print('in')
# print(x, x.shape)
# print(y, y.shape)
# print('out')
# print(out)
# print(out.shape)

f = myad.grad(F.dot)
l = f(x,y)
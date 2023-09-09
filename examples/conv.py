import slope
from slope import environment as sev

x = sev.ones((1, 3, 16, 16))
y = sev.ones((8, 3, 3, 3)) * 2
print(x.shape, y.shape)
print(sev.conv2d(x, y).shape)

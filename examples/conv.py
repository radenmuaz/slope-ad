import slope


x = slope.ones((1, 3, 16, 16))
y = slope.ones((8, 3, 3, 3)) * 2
print(x.shape, y.shape)
print(slope.conv(x, y).shape)

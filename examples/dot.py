import slope
import numpy as np
from slope import ad, ops

BSZ = 4
x1 = np.random.randn(BSZ, 1, 3)
y1 = np.random.randn(BSZ, 3, 2)
out1 = ad.vmap(ops.mm, (0, 0))(x1, y1)
print(f" x:\t{x1.shape}")
print(f" y:\t{y1.shape}")
print(f" ou1t:\t{out1.shape}\n")

x2 = x1.transpose(1, 2, 0)
y2 = y1.transpose(1, 2, 0)
out2 = ad.vmap(ops.mm, (2, 2))(x2, y2)
print(f" x:\t{x2.shape}")
print(f" y:\t{y2.shape}")
print(f" out2:\t{out2.shape}\n")

assert np.allclose(out2, out1)
x3 = x1.transpose(1, 0, 2)
y3 = y1.transpose(1, 0, 2)
out3 = ad.vmap(ops.mm, (1, 1))(x3, y3)
print(f" x:\t{x3.shape}")
print(f" y:\t{y3.shape}")
print(f" out3:\t{out3.shape}\n")

assert np.allclose(out3, out1)
assert np.allclose(out3, out2)

out4 = ad.vmap(ops.mm, (0, 2))(x1, y2)
print(f" x:\t{x1.shape}")
print(f" y:\t{y2.shape}")
print(f" out4:\t{out4.shape}\n")

assert np.allclose(out4, out1)

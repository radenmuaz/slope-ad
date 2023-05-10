import slope
import numpy as np
from slope import ad, ops

# BSZ = 4
# x = np.random.randn(BSZ, 1, 3)
# y = np.random.randn(BSZ, 3, 2)
# out = ad.vmap(ops.mm_old, (0, 0))(x, y)
# print(f' x:\t{x.shape}')
# print(f' y:\t{y.shape}')
# print(f' out:\t{out.shape}')


# BSZ = 4
# x = np.random.randn(1, 3, BSZ)
# y = np.random.randn(3, 2, BSZ)
# out = ad.vmap(ops.mm_old, (2, 2))(x, y)
# print(f' x:\t{x.shape}')
# print(f' y:\t{y.shape}')
# print(f' out:\t{out.shape}')


# BSZ = 4
# x = np.random.randn(1, BSZ, 3)
# y = np.random.randn(3, BSZ, 2)
# out = ad.vmap(ops.mm_old, (1, 1))(x, y)
# print(f' x:\t{x.shape}')
# print(f' y:\t{y.shape}')
# print(f' out:\t{out.shape}')

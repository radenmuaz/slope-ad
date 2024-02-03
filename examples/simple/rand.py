import slope

x = slope.tensor([1, 1], dtype=slope.uint64)
y = slope.rng_bits(x)
print(y)
# print(slope.rand(5))
# print(slope.rand(5))

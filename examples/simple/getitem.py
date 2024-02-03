import slope

x = slope.tensor([[0.0, 1.0], [2.0, 3.0]], dtype=slope.float32)
w = slope.tensor([1], dtype=slope.int32)

# y = x.gather_nd(w)
y = x[w]

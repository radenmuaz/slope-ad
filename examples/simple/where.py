import slope

x = slope.ones(5)
w = slope.arange(5, dtype=x.dtype)
y = (x > w).where(x,w)


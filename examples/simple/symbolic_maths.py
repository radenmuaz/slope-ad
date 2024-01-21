import slope

x1 = slope.tensor((1,))
x2 = slope.tensor((2,))
y1 = x1+x2

with slope.symbolic_run():
    sym_y1 = x1+x2
y2 = x1+x2

breakpoint()
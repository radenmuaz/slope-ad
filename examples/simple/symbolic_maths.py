import slope

x1 = slope.tensor((1,))
x2 = slope.tensor((2,))
y1 = x1+x2
with slope.symbolic_run():
    sym_y = x1+x2
y2 = x1+x2
print(f"{y1=}")
print(f"{sym_y=}")
print(f"{y2=}")
# x1 = slope.symbolic_tensor((1,))
# x2 = slope.symbolic_tensor((1,))
# with slope.symbolic_run():
#     y = x1+x2
#     breakpoint()
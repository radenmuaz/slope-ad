import slope

print("#1")
x = slope.tensor([[0.0, 1.0], [2.0, 3.0]], dtype=slope.float32)
# w = slope.tensor([[1,0],[0,1]], dtype=slope.int64)
w = slope.tensor([[1, 0], [0, 1]], dtype=slope.int32)
print(f"{x=}")
print(f"{w=}")
y = x.gather_nd(w, 0)
print(f"{y=}")


@slope.jit
def f(x, w):
    y = x.gather_nd(w, 0)
    y = y.sum()
    return y


gL_x = slope.grad(f)(x, w)
print(gL_x)


print("\n#2")
x = slope.tensor([[0.0, 1.0], [2.0, 3.0]], dtype=slope.float32)
w = slope.tensor([[1], [0]], dtype=slope.int32)
print(f"{x=}")
print(f"{w=}")
y = x.gather_nd(w)
print(f"{y=}")

print("\n#3")
x = slope.tensor([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=slope.float32)
w = slope.tensor([[0, 1], [1, 0]], dtype=slope.int32)
print(f"{x=}")
print(f"{w=}")
y = x.gather_nd(w)
print(f"{y=}")


print("\n#4")
x = slope.tensor([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=slope.float32)
w = slope.tensor([[[0, 1]], [[1, 0]]], dtype=slope.int32)
print(f"{x=}")
print(f"{w=}")
y = x.gather_nd(w)
print(f"{y=}")


print("\n#5")
x = slope.tensor([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=slope.float32)
w = slope.tensor([[1], [0]], dtype=slope.int32)
print(f"{x=}")
print(f"{w=}")
y = x.gather_nd(w, 1)
print(f"{y=}")

# Other values
# x = slope.arange(10, dtype=slope.float32).reshape(2,5)
# w = slope.tensor([1,0], dtype=slope.int32)[..., None]

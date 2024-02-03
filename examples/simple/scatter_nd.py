import slope

x = slope.zeros(8, dtype=slope.float32)
print(f"before: {x=}")
w = slope.tensor([[4], [3], [1], [7]], dtype=slope.int32)  # iree
u = slope.tensor([9.0, 10.0, 11.0, 12.0], dtype=slope.float32)
y = slope.scatter_nd(x, w, u)
print(f"{x=}")
print(f"{w=}")
print(f"{u=}")
print(f"{y=}")

x = slope.zeros((2, 2), dtype=slope.float32)
print(f"before: {x=}")
w = slope.tensor([[1, 0], [0, 1]], dtype=slope.int32)
u = slope.tensor([1, 2], dtype=slope.float32)
y = slope.scatter_nd(x, w, u)
print(f"{x=}")
print(f"{w=}")
print(f"{u=}")
print(f"{y=}")

x = slope.tensor(
    [
        [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
        [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
        [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]],
        [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]],
    ],
    dtype=slope.float32,
)
w = slope.tensor([[0], [2]], dtype=slope.int32)
u = slope.tensor(
    [
        [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
        [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]],
    ],
    dtype=slope.float32,
)
y = slope.scatter_nd(x, w, u)
print(f"{x=}")
print(f"{w=}")
print(f"{u=}")
print(f"{y=}")


@slope.jit
def f(x, w, u):
    y = slope.scatter_nd(x, w, u)
    y = y.sum()
    return y


gL_x = slope.grad(f)(x, w, u)
print(f"{gL_x=}")

# NOTE: onnxruntime:
# updates rank: q + r - indices_shape[-1] - 1.
# updates tensor should have shape equal to indices.shape[:-1] + data.shape[indices.shape[-1]:]

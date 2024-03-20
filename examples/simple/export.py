import slope


x = slope.tensor([[1, 2], [3, 4], [5, 6]], dtype=slope.float32)
# c = x.ones_like()


@slope.jit
def f(x):
    y = x.sum(1)
    return y


# print(f(x,))
# f_jitobj = f.lower(x)
# f.export("/tmp/f", (x,), input_names=['x'], output_names=['y'], dynamic_axes=dict(x=[0], y=[0]))
f.export("/tmp/f", (x,), input_names=["x"], output_names=["y"], dynamic_axes=dict(x={0: "batch"}), y={0: "batch"})

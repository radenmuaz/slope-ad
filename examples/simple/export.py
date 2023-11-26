import slope

@slope.jit
def f(x1, x2):
    y = x.sum() + x2
    return y

x = slope.tensor([[1,2],[3,4]], dtype=slope.float32)
# print(f(x,))
f_jitobj = f.get_jit_object(x, slope.tensor([[5,6],[7,8]], dtype=slope.float32))
f_jitobj.export('/tmp/f')
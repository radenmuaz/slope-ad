import slope


x = slope.tensor([[1,2],[3,4]], dtype=slope.float32)
c = x

@slope.jit
def f(x):
    y = (x+c).sum()
    return y
# print(f(x,))
f_jitobj = f.get_jit_object(x)
f_jitobj.export('/tmp/f', x)
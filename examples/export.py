import slope

@slope.jit
def f(x):
    # y = x + x
    y = x.sum()
    return y

x = slope.randn((3,))
print(f(x))

f_jitobj = f.get_jit_object(x)
print(f_jitobj)
f_jitobj.export('/tmp/f')
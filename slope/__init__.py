# from slope.ad import Runtime, vmap, jvp, make_jaxpr, linearize, vjp, grad
from slope.ad import Runtime
from slope.numpy_backend import numpy_backend

RT = Runtime()
backend = numpy_backend
# RT.add_op(base_ops.add)
# RT.set_backend(numpy_backend)
# RT_ = None
# def RT():
#     global RT_
#     if RT_ is None:
#         from slope import ad
#         RT_ = ad.Runtime()
#     return RT_

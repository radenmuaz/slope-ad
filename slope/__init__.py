# from slope.ad import Runtime, vmap, jvp, make_jaxpr, linearize, vjp, grad
from slope import ad

RT = ad.Runtime()

# patch numpy
import numpy as np

# from slope import tracer

# np.ndarray.broadcast = tracer.TracerArray.broadcast
# np.ndarray.exp = tracer.TracerArray.exp

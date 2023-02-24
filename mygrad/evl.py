from mygrad import pm, reg, trc
import numpy as np
class EvalTrace(trc.Trace):
  pure = lift = lambda self, x: x  # no boxing in Tracers needed

  def process_primitive(self, primitive, tracers, params):
    return reg.impl_rules[primitive](*tracers, **params)

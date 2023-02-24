import numpy as np
from mygrad import trc
from typing import List, Optional, Dict, Callable


from mygrad import pm
from mygrad import pp
jax_types = {bool, int, float,
             np.bool_, np.int32, np.int64, np.float32, np.float64, np.ndarray}

trace_stack: List[trc.MainTrace] = []
dynamic_trace: Optional[trc.MainTrace] = None

impl_rules = {}
jvp_rules = {}
pp_rules: Dict[pm.Primitive, Callable[..., pp.PPrint]] = {}

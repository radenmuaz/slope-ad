import numpy as np
from mygrad import tracing as trc
from typing import List, Optional, Dict, Callable, Type


from mygrad import primitives as pm
from mygrad import pretty_print as pp
from mygrad import pytrees as pt

jax_types = {
    bool,
    int,
    float,
    np.bool_,
    np.int32,
    np.int64,
    np.float32,
    np.float64,
    np.ndarray,
}

trace_stack: List[trc.MainTrace] = []
dynamic_trace: Optional[trc.MainTrace] = None

impl_rules = {}
jvp_rules = {}
pp_rules: Dict[pm.Primitive, Callable[..., pp.PPrint]] = {}
node_types: Dict[Type, pt.NodeType] = {}

import slope as sp
from dataclasses import dataclass

class V1Opset(sp.Opset):
def load_os(rt):
    import ops_defs
    import procs_defs
    import backend_defs
from operators import operator_set
from procedures import procedure_set
from compiler import compiler
from slope.core import Backend
example_backend = Backend(operator_set, procedure_set, compiler)

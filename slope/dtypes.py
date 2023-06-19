from typing import (
    Any,
    Dict,
    List,
    Union,
)
import numpy as np

bool_: type = np.bool_
int_: type = np.int32
uint: type = np.uint32
float_: type = np.float32
_default_types: Dict[str, type] = {
    "b": bool_,
    "i": int_,
    "u": uint,
    "f": float_,
}

# Trivial vectorspace datatype needed for tangent values of int/bool primals
float0: np.dtype = np.dtype([("float0", np.void, 0)])

_dtype_to_32bit_dtype: Dict[Any, Any] = {
    np.dtype("int64"): np.dtype("int32"),
    np.dtype("uint64"): np.dtype("uint32"),
    np.dtype("float64"): np.dtype("float32"),
}

# Note: we promote narrow types to float32 here for backward compatibility
# with earlier approaches. We might consider revisiting this, or perhaps
# tying the logic more closely to the type promotion lattice.
_dtype_to_inexact: Dict[Any, Any] = {
    np.dtype(k): np.dtype(v)
    for k, v in [
        ("bool", "float32"),
        ("uint8", "float32"),
        ("int8", "float32"),
        ("uint16", "float32"),
        ("int16", "float32"),
        ("uint32", "float32"),
        ("int32", "float32"),
    ]
}


def to_numeric_dtype(dtype: Any) -> Any:
    """Promotes a dtype into an numeric dtype, if it is not already one."""
    dtype_ = np.dtype(dtype)
    return np.dtype("int32") if dtype_ == np.dtype("bool") else dtype_


def to_inexact_dtype(dtype: Any) -> Any:
    """Promotes a dtype into an inexact dtype, if it is not already one."""
    dtype_ = np.dtype(dtype)
    return _dtype_to_inexact.get(dtype_, dtype_)


def to_complex_dtype(dtype: Any) -> Any:
    ftype = to_inexact_dtype(dtype)
    if ftype in [np.dtype("float64"), np.dtype("complex128")]:
        return np.dtype("complex128")
    return np.dtype("complex64")


# Default dtypes corresponding to Python scalars.
python_scalar_dtypes: Dict[type, Any] = {
    bool: np.dtype("bool"),
    int: np.dtype("int64"),
    float: np.dtype("float64"),
    complex: np.dtype("complex128"),
}


def scalar_type_of(x: Any) -> type:
    """Return the scalar type associated with a JAX value."""
    typ = x.dtype
    if np.issubdtype(typ, np.bool_):
        return bool
    elif np.issubdtype(typ, np.integer):
        return int
    elif np.issubdtype(typ, np.floating):
        return float
    elif np.issubdtype(typ, np.complexfloating):
        return complex
    else:
        raise TypeError(f"Invalid scalar value {x}")


def _issubclass(a: Any, b: Any) -> bool:
    """Determines if ``a`` is a subclass of ``b``.

    Similar to issubclass, but returns False instead of an exception if `a` is not
    a class.
    """
    try:
        return issubclass(a, b)
    except TypeError:
        return False


_type_classes = {
    np.generic,
    np.number,
    np.flexible,
    np.character,
    np.integer,
    np.signedinteger,
    np.unsignedinteger,
    np.inexact,
    np.floating,
}


def _is_typeclass(a: Any) -> bool:
    try:
        return a in _type_classes
    except TypeError:
        return False


def issubdtype(a: Any, b: Any) -> bool:
    # Canonicalizes all concrete types to np.dtype instances
    a = a if _is_typeclass(a) else np.dtype(a)
    b = b if _is_typeclass(b) else np.dtype(b)
    return np.issubdtype(a, b)


can_cast = np.can_cast
issubsctype = np.issubsctype

Dtype = Union[type, Any]

_bool_types: List[Dtype] = [np.dtype(bool)]

_int_types = [
    np.dtype("uint8"),
    np.dtype("uint16"),
    np.dtype("uint32"),
    np.dtype("uint64"),
    np.dtype("int8"),
    np.dtype("int16"),
    np.dtype("int32"),
    np.dtype("int64"),
]

_float_types: List[Dtype] = [
    np.dtype("float16"),
    np.dtype("float32"),
    np.dtype("float64"),
]
_slope_types = _bool_types + _int_types + _float_types
_slope_dtype_set = {float0, *_bool_types, *_int_types, *_float_types}


def _slope_type(dtype: Any, weak_type: bool) -> Dtype:
    """Return the jax type for a dtype and weak type."""
    if weak_type:
        if dtype == bool:
            return dtype
        return type(dtype.type(0).item())
    return dtype


def is_python_scalar(x: Any) -> bool:
    return type(x) in python_scalar_dtypes


def check_valid_dtype(dtype: Any) -> None:
    if dtype not in _slope_dtype_set:
        raise TypeError(
            f"Dtype {dtype} is not a valid Slope array "
            "type. Only arrays of numeric types are supported by Slope."
        )

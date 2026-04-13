"""contracts/numpy_types.py — Pydantic v2 custom types for numpy arrays.

Provides NdArrayFloat32 and NdArrayBool as Annotated-compatible Pydantic
validators + serializers.

Serialization format (JSON):
    {
        "dtype":  "float32" | "bool",
        "shape":  [N],
        "data":   "<base64-encoded raw bytes>",
    }

Validation accepts:
    - np.ndarray (passthrough, cast to target dtype)
    - dict with "dtype"/"shape"/"data" keys (JSON round-trip deserialization)
    - list / sequence (fixture/test compatibility coercion path)
"""
from __future__ import annotations

import base64
from typing import Any

import numpy as np
from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema


def _make_ndarray_annotation(dtype_str: str, np_dtype: type) -> type:
    """Factory: returns a class usable as Pydantic Annotated metadata."""

    class _NdArrayAnnotation:
        @classmethod
        def __get_pydantic_core_schema__(
            cls,
            source_type: Any,  # noqa: ARG003
            handler: GetCoreSchemaHandler,  # noqa: ARG003
        ) -> core_schema.CoreSchema:
            return core_schema.no_info_plain_validator_function(
                cls._validate,
                serialization=core_schema.plain_serializer_function_ser_schema(
                    cls._serialize,
                    info_arg=False,
                ),
            )

        @classmethod
        def _validate(cls, v: Any) -> np.ndarray:
            if isinstance(v, np.ndarray):
                return v.astype(np_dtype)
            if isinstance(v, dict) and "data" in v:
                # JSON deserialization path
                raw_bytes = base64.b64decode(v["data"])
                stored_dtype = np.dtype(v.get("dtype", dtype_str))
                arr = np.frombuffer(raw_bytes, dtype=stored_dtype)
                shape = v.get("shape")
                if shape:
                    arr = arr.reshape(shape)
                return arr.astype(np_dtype)
            # list / sequence coercion — supports test fixtures and legacy paths
            return np.array(v, dtype=np_dtype)

        @classmethod
        def _serialize(cls, v: np.ndarray) -> dict[str, Any]:
            contiguous: np.ndarray = np.ascontiguousarray(v, dtype=np_dtype)
            return {
                "dtype": dtype_str,
                "shape": list(contiguous.shape),
                "data": base64.b64encode(contiguous.tobytes()).decode("ascii"),
            }

    _NdArrayAnnotation.__name__ = f"NdArray_{dtype_str}"
    _NdArrayAnnotation.__qualname__ = f"NdArray_{dtype_str}"
    return _NdArrayAnnotation


# Public annotation markers — use with Annotated[np.ndarray, NdArrayFloat32]
NdArrayFloat32 = _make_ndarray_annotation("float32", np.float32)
NdArrayBool = _make_ndarray_annotation("bool", np.bool_)

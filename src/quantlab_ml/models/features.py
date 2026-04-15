from __future__ import annotations

from typing import Any

import numpy as np

from quantlab_ml.contracts import ObservationContext


def observation_feature_vector(observation: ObservationContext) -> list[float]:
    return observation_feature_array(observation, dtype=np.float32).tolist()


def observation_feature_array(
    observation: ObservationContext,
    *,
    dtype: np.dtype[Any] | type[np.generic] = np.float32,
) -> np.ndarray:
    raw_feature_count = 0
    for scale in observation.observation_schema.scale_axis:
        tensor = observation.raw_surface[scale.label]
        raw_feature_count += (
            len(tensor.values)
            + len(tensor.age)
            + len(tensor.padding)
            + len(tensor.unavailable_by_contract)
            + len(tensor.missing)
            + len(tensor.stale)
        )

    derived_feature_count = 0
    derived_channels = []
    if observation.derived_surface is not None:
        derived_channels = sorted(observation.derived_surface.channels, key=lambda item: item.key)
        derived_feature_count = sum(len(channel.values) for channel in derived_channels)

    resolved_dtype = np.dtype(dtype)
    features = np.empty(raw_feature_count + derived_feature_count + 1, dtype=resolved_dtype)
    cursor = 0
    for scale in observation.observation_schema.scale_axis:
        tensor = observation.raw_surface[scale.label]
        cursor = _copy_array(features, cursor, tensor.values, dtype=resolved_dtype)
        cursor = _copy_array(features, cursor, tensor.age, dtype=resolved_dtype)
        cursor = _copy_bool_array(features, cursor, tensor.padding, dtype=resolved_dtype)
        cursor = _copy_bool_array(features, cursor, tensor.unavailable_by_contract, dtype=resolved_dtype)
        cursor = _copy_bool_array(features, cursor, tensor.missing, dtype=resolved_dtype)
        cursor = _copy_bool_array(features, cursor, tensor.stale, dtype=resolved_dtype)

    for channel in derived_channels:
        cursor = _copy_array(features, cursor, channel.values, dtype=resolved_dtype)

    asset_count = max(len(observation.observation_schema.asset_axis), 1)
    features[cursor] = observation.target_asset_index / float(asset_count)
    return features


def _to_float_list(values: np.ndarray | list[float]) -> list[float]:
    """Convert a float array (ndarray or list) to a plain Python list[float]."""
    if isinstance(values, np.ndarray):
        return values.astype(np.float32).tolist()
    return list(values)


def _bool_to_float_list(values: np.ndarray | list[bool]) -> list[float]:
    """Convert a bool array (ndarray or list) to a plain Python list[float] (0.0/1.0)."""
    if isinstance(values, np.ndarray):
        return values.astype(np.float32).tolist()
    return [1.0 if v else 0.0 for v in values]


def _copy_array(
    destination: np.ndarray,
    start: int,
    values: np.ndarray | list[float],
    *,
    dtype: np.dtype[Any],
) -> int:
    array = np.asarray(values, dtype=dtype)
    end = start + int(array.shape[0])
    destination[start:end] = array
    return end


def _copy_bool_array(
    destination: np.ndarray,
    start: int,
    values: np.ndarray | list[bool],
    *,
    dtype: np.dtype[Any],
) -> int:
    array = np.asarray(values, dtype=np.bool_)
    end = start + int(array.shape[0])
    destination[start:end] = array.astype(dtype, copy=False)
    return end


# Backward-compat alias — some tests may import this directly
_bools_to_floats = _bool_to_float_list

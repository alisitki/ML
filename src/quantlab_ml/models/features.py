from __future__ import annotations

import numpy as np

from quantlab_ml.contracts import ObservationContext


def observation_feature_vector(observation: ObservationContext) -> list[float]:
    features: list[float] = []
    for scale in observation.observation_schema.scale_axis:
        tensor = observation.raw_surface[scale.label]
        features.extend(_to_float_list(tensor.values))
        features.extend(_to_float_list(tensor.age))
        features.extend(_bool_to_float_list(tensor.padding))
        features.extend(_bool_to_float_list(tensor.unavailable_by_contract))
        features.extend(_bool_to_float_list(tensor.missing))
        features.extend(_bool_to_float_list(tensor.stale))

    if observation.derived_surface is not None:
        for channel in sorted(observation.derived_surface.channels, key=lambda item: item.key):
            features.extend(_to_float_list(channel.values))

    asset_count = max(len(observation.observation_schema.asset_axis), 1)
    features.append(observation.target_asset_index / float(asset_count))
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


# Backward-compat alias — some tests may import this directly
_bools_to_floats = _bool_to_float_list

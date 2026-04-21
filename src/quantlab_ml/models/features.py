from __future__ import annotations

from typing import Any

import numpy as np

from quantlab_ml.contracts import ObservationContext, PolicyState

PHASE1A_POLICY_STATE_FEATURE_DIM = 9


def observation_feature_vector(observation: ObservationContext) -> list[float]:
    return observation_feature_array(observation, dtype=np.float32).tolist()


def phase1a_feature_vector(
    observation: ObservationContext,
    policy_state: PolicyState | None,
    *,
    venue_choices: list[str],
) -> list[float]:
    return phase1a_feature_array(
        observation,
        policy_state,
        venue_choices=venue_choices,
        dtype=np.float32,
    ).tolist()


def observation_feature_segment_manifest(observation: ObservationContext) -> list[dict[str, int | str]]:
    segments: list[dict[str, int | str]] = []
    cursor = 0

    for scale in observation.observation_schema.scale_axis:
        tensor = observation.raw_surface[scale.label]
        for block_name, values in (
            ("values", tensor.values),
            ("age", tensor.age),
            ("padding", tensor.padding),
            ("unavailable_by_contract", tensor.unavailable_by_contract),
            ("missing", tensor.missing),
            ("stale", tensor.stale),
        ):
            length = int(len(values))
            segments.append(
                {
                    "name": f"raw/{scale.label}/{block_name}",
                    "start": cursor,
                    "length": length,
                }
            )
            cursor += length

    derived_length = 0
    if observation.derived_surface is not None:
        derived_length = sum(len(channel.values) for channel in observation.derived_surface.channels)
    segments.append(
        {
            "name": "derived",
            "start": cursor,
            "length": int(derived_length),
        }
    )
    cursor += int(derived_length)
    segments.append(
        {
            "name": "target_asset_index",
            "start": cursor,
            "length": 1,
        }
    )
    return segments


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


def policy_state_feature_array(
    policy_state: PolicyState | None,
    *,
    venue_choices: list[str],
    dtype: np.dtype[Any] | type[np.generic] = np.float32,
) -> np.ndarray:
    resolved_dtype = np.dtype(dtype)
    features = np.zeros(3 + len(venue_choices) + 1 + 2, dtype=resolved_dtype)
    state = policy_state or PolicyState()

    side_to_index = {"flat": 0, "long": 1, "short": 2}
    features[side_to_index[state.previous_position_side]] = 1.0

    venue_offset = 3
    none_index = venue_offset + len(venue_choices)
    if state.previous_venue is None:
        features[none_index] = 1.0
    else:
        for venue_index, venue in enumerate(venue_choices):
            if venue == state.previous_venue:
                features[venue_offset + venue_index] = 1.0
                break
        else:
            features[none_index] = 1.0

    tail_offset = none_index + 1
    features[tail_offset] = min(float(state.hold_age_steps), 32.0) / 32.0
    features[tail_offset + 1] = min(float(state.turnover_accumulator), 32.0) / 32.0
    return features


def phase1a_feature_array(
    observation: ObservationContext,
    policy_state: PolicyState | None,
    *,
    venue_choices: list[str],
    dtype: np.dtype[Any] | type[np.generic] = np.float32,
) -> np.ndarray:
    observation_features = observation_feature_array(observation, dtype=dtype)
    policy_features = policy_state_feature_array(
        policy_state,
        venue_choices=venue_choices,
        dtype=dtype,
    )
    return np.concatenate((observation_features, policy_features), axis=0)


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

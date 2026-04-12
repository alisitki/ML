from __future__ import annotations

from quantlab_ml.contracts import ObservationContext


def observation_feature_vector(observation: ObservationContext) -> list[float]:
    features: list[float] = []
    for scale in observation.observation_schema.scale_axis:
        tensor = observation.raw_surface[scale.label]
        features.extend(tensor.values)
        features.extend(tensor.age)
        features.extend(_bools_to_floats(tensor.padding))
        features.extend(_bools_to_floats(tensor.unavailable_by_contract))
        features.extend(_bools_to_floats(tensor.missing))
        features.extend(_bools_to_floats(tensor.stale))

    asset_count = max(len(observation.observation_schema.asset_axis), 1)
    features.append(observation.target_asset_index / float(asset_count))
    return features


def _bools_to_floats(values: list[bool]) -> list[float]:
    return [1.0 if value else 0.0 for value in values]

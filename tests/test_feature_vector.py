from __future__ import annotations

import pytest

from quantlab_ml.models.features import observation_feature_vector
from quantlab_ml.models.linear_policy import LinearPolicyParameters


def test_observation_feature_vector_includes_derived_surface_values(trajectory_bundle) -> None:
    observation = trajectory_bundle.splits["train"][0].steps[0].observation
    feature_vector = observation_feature_vector(observation)
    assert observation.derived_surface is not None
    assert observation.derived_surface.channels

    raw_feature_count = 0
    for tensor in observation.raw_surface.values():
        raw_feature_count += (
            len(tensor.values)
            + len(tensor.age)
            + len(tensor.padding)
            + len(tensor.unavailable_by_contract)
            + len(tensor.missing)
            + len(tensor.stale)
        )

    derived_values: list[float] = []
    for channel in sorted(observation.derived_surface.channels, key=lambda item: item.key):
        derived_values.extend(channel.values)

    assert feature_vector[raw_feature_count : raw_feature_count + len(derived_values)] == pytest.approx(
        derived_values
    )
    assert len(feature_vector) == raw_feature_count + len(derived_values) + 1


def test_linear_policy_parameters_match_current_feature_extractor(
    trajectory_bundle,
    policy_artifact,
) -> None:
    observation = trajectory_bundle.splits["validation"][0].steps[0].observation
    feature_vector = observation_feature_vector(observation)
    parameters = LinearPolicyParameters.model_validate_json(policy_artifact.policy_payload.blob)

    assert len(parameters.feature_mean) == len(feature_vector)
    assert len(parameters.feature_std) == len(feature_vector)

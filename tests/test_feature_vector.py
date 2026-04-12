from __future__ import annotations

import pytest

from quantlab_ml.common import load_yaml
from quantlab_ml.contracts import ActionSpaceSpec, TrajectorySpec
from quantlab_ml.data import LocalFixtureSource
from quantlab_ml.models.features import observation_feature_vector
from quantlab_ml.models.linear_policy import LinearPolicyParameters
from quantlab_ml.runtime_contract import build_strict_runtime_contract
from quantlab_ml.trajectories import TrajectoryBuilder


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


def test_production_profile_matches_canonical_scale_preset_and_runtime_contract(
    repo_root,
    fixture_path,
    dataset_spec,
    reward_spec,
) -> None:
    raw = load_yaml(repo_root / "configs" / "training" / "production.yaml")
    trajectory_spec = TrajectorySpec.model_validate(raw["trajectory"])
    action_space = ActionSpaceSpec.model_validate(raw["action_space"])
    events = LocalFixtureSource(fixture_path).load_events(dataset_spec)
    bundle = TrajectoryBuilder(dataset_spec, trajectory_spec, action_space, reward_spec).build(events)
    observation = bundle.splits["train"][0].steps[0].observation
    strict_contract = build_strict_runtime_contract(bundle.observation_schema, policy_kind="linear-policy-v1")

    assert [scale.label for scale in trajectory_spec.scale_preset] == ["1m", "5m", "15m", "60m"]
    assert [scale.num_buckets for scale in trajectory_spec.scale_preset] == [8, 8, 8, 12]
    assert [scale.label for scale in strict_contract.required_scale_specs] == ["1m", "5m", "15m", "60m"]
    assert len(observation_feature_vector(observation)) == strict_contract.expected_feature_dim

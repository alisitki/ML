from __future__ import annotations

import logging

import pytest

from quantlab_ml.contracts import (
    ExecutionIntent,
    LEGACY_POLICY_ARTIFACT_SCHEMA_VERSION,
    POLICY_ARTIFACT_SCHEMA_VERSION,
)
from quantlab_ml.policies import PolicyRuntimeBridge


def test_policy_artifact_runtime_metadata_matches_canonical_contract(policy_artifact) -> None:
    metadata = policy_artifact.runtime_metadata
    strict_contract = metadata.strict_runtime_contract

    assert policy_artifact.artifact_id
    assert policy_artifact.artifact_version == POLICY_ARTIFACT_SCHEMA_VERSION
    assert policy_artifact.training_snapshot_id
    assert policy_artifact.training_config_hash
    assert policy_artifact.code_commit_hash
    assert policy_artifact.evaluation_surface_id
    assert policy_artifact.allowed_venues == metadata.allowed_venues
    assert policy_artifact.reward_version == metadata.reward_version
    assert metadata.required_streams
    assert metadata.required_field_families
    assert metadata.required_scale_preset
    assert metadata.artifact_compatibility_tags
    assert strict_contract is not None
    assert strict_contract.policy_kind == policy_artifact.policy_payload.runtime_adapter
    assert strict_contract.expected_feature_dim > 0


def test_runtime_bridge_builds_execution_intent_from_artifact_and_observation(
    trajectory_bundle,
    policy_artifact,
) -> None:
    bridge = PolicyRuntimeBridge()
    observation = _validation_observation(trajectory_bundle)

    intent = bridge.build_execution_intent(policy_artifact, observation, ttl_seconds=45)

    assert isinstance(intent, ExecutionIntent)
    assert intent.policy_id == policy_artifact.policy_id
    assert intent.artifact_id == policy_artifact.artifact_id
    assert intent.target_asset == observation.target_symbol
    assert intent.venue in policy_artifact.allowed_venues
    assert intent.action in policy_artifact.allowed_action_family
    assert intent.selector_trace_id.startswith("selector-")
    assert intent.intent_id.startswith("intent-")
    if intent.action == "abstain":
        assert intent.notional_or_size == pytest.approx(0.0)
        assert intent.leverage == pytest.approx(0.0)
    else:
        assert intent.notional_or_size > 0.0
        assert intent.leverage > 0.0


def test_runtime_bridge_rejects_new_artifact_missing_strict_runtime_contract(
    trajectory_bundle,
    policy_artifact,
) -> None:
    bridge = PolicyRuntimeBridge()
    observation = _validation_observation(trajectory_bundle)
    broken = policy_artifact.model_copy(
        update={
            "runtime_metadata": policy_artifact.runtime_metadata.model_copy(
                update={"strict_runtime_contract": None},
                deep=True,
            )
        },
        deep=True,
    )

    with pytest.raises(ValueError, match="requires runtime_metadata.strict_runtime_contract"):
        bridge.decide(broken, observation)


def test_runtime_bridge_rejects_scale_spec_mismatch(policy_artifact, trajectory_bundle) -> None:
    bridge = PolicyRuntimeBridge()
    observation = _validation_observation(trajectory_bundle)
    first_scale = observation.observation_schema.scale_axis[0]
    updated_scale = first_scale.model_copy(update={"num_buckets": first_scale.num_buckets + 1})
    observation.observation_schema = observation.observation_schema.model_copy(
        update={"scale_axis": [updated_scale, *observation.observation_schema.scale_axis[1:]]},
        deep=True,
    )

    with pytest.raises(ValueError, match="scale spec does not match"):
        bridge.decide(policy_artifact, observation)


def test_runtime_bridge_rejects_raw_shape_mismatch(policy_artifact, trajectory_bundle) -> None:
    bridge = PolicyRuntimeBridge()
    observation = _validation_observation(trajectory_bundle)
    tensor = observation.raw_surface["1m"]
    broken_shape = [*tensor.shape]
    broken_shape[0] += 1
    observation.raw_surface["1m"] = tensor.model_copy(update={"shape": broken_shape}, deep=True)

    with pytest.raises(ValueError, match="raw surface shapes do not match"):
        bridge.decide(policy_artifact, observation)


def test_runtime_bridge_rejects_derived_contract_version_mismatch(policy_artifact, trajectory_bundle) -> None:
    bridge = PolicyRuntimeBridge()
    observation = _validation_observation(trajectory_bundle)
    assert observation.derived_surface is not None
    observation.derived_surface = observation.derived_surface.model_copy(
        update={"contract_version": "derived_surface_v999"},
        deep=True,
    )

    with pytest.raises(ValueError, match="derived contract version mismatch"):
        bridge.decide(policy_artifact, observation)


@pytest.mark.parametrize("mode", ["reordered", "missing", "extra"])
def test_runtime_bridge_rejects_derived_channel_identity_mismatches(
    policy_artifact,
    trajectory_bundle,
    mode: str,
) -> None:
    bridge = PolicyRuntimeBridge()
    observation = _validation_observation(trajectory_bundle)
    assert observation.derived_surface is not None
    channels = list(observation.derived_surface.channels)
    assert channels

    if mode == "reordered":
        if len(channels) < 2:
            pytest.skip("fixture observation must contain at least two derived channels")
        mutated_channels = channels[1:] + channels[:1]
    elif mode == "missing":
        mutated_channels = channels[1:]
    else:
        mutated_channels = [
            *channels,
            channels[0].model_copy(update={"key": "extra_runtime_contract_channel"}, deep=True),
        ]

    observation.derived_surface = observation.derived_surface.model_copy(
        update={"channels": mutated_channels},
        deep=True,
    )

    with pytest.raises(ValueError, match="derived channel identity/order mismatch"):
        bridge.decide(policy_artifact, observation)


def test_runtime_bridge_rejects_feature_dimension_mismatch(policy_artifact, trajectory_bundle) -> None:
    bridge = PolicyRuntimeBridge()
    observation = _validation_observation(trajectory_bundle)
    strict_contract = policy_artifact.runtime_metadata.strict_runtime_contract
    assert strict_contract is not None
    broken = policy_artifact.model_copy(
        update={
            "runtime_metadata": policy_artifact.runtime_metadata.model_copy(
                update={
                    "strict_runtime_contract": strict_contract.model_copy(
                        update={"expected_feature_dim": strict_contract.expected_feature_dim + 1},
                        deep=True,
                    )
                },
                deep=True,
            )
        },
        deep=True,
    )

    with pytest.raises(ValueError, match="payload feature dimension does not match runtime contract"):
        bridge.decide(broken, observation)


def test_runtime_bridge_accepts_known_safe_legacy_linear_artifact_via_compat_window(
    policy_artifact,
    trajectory_bundle,
    caplog: pytest.LogCaptureFixture,
) -> None:
    bridge = PolicyRuntimeBridge()
    observation = _validation_observation(trajectory_bundle)
    legacy_artifact = _legacy_linear_artifact(policy_artifact)

    caplog.set_level(logging.WARNING)
    decision = bridge.decide(legacy_artifact, observation)

    assert decision.action_key in policy_artifact.allowed_action_family
    assert "legacy artifact accepted via compat window" in caplog.text


def test_runtime_bridge_rejects_legacy_artifact_with_ambiguous_contract(
    policy_artifact,
    trajectory_bundle,
) -> None:
    bridge = PolicyRuntimeBridge()
    observation = _validation_observation(trajectory_bundle)
    legacy_artifact = _legacy_linear_artifact(policy_artifact)
    legacy_artifact.runtime_metadata = legacy_artifact.runtime_metadata.model_copy(
        update={"required_scale_preset": ["5m"]},
        deep=True,
    )

    with pytest.raises(ValueError, match="required_scale_preset is inconsistent"):
        bridge.decide(legacy_artifact, observation)


def test_runtime_bridge_rejects_deprecated_momentum_artifacts(
    policy_artifact,
    trajectory_bundle,
) -> None:
    bridge = PolicyRuntimeBridge()
    observation = _validation_observation(trajectory_bundle)
    momentum_artifact = _legacy_linear_artifact(policy_artifact)
    momentum_artifact.policy_payload = momentum_artifact.policy_payload.model_copy(
        update={"runtime_adapter": "momentum-baseline-v1"},
        deep=True,
    )
    momentum_artifact.runtime_metadata = momentum_artifact.runtime_metadata.model_copy(
        update={"runtime_adapter": "momentum-baseline-v1"},
        deep=True,
    )

    with pytest.raises(ValueError, match="deprecated and unsupported"):
        bridge.decide(momentum_artifact, observation)


def test_runtime_bridge_rejects_unknown_decision_venue(policy_artifact, trajectory_bundle) -> None:
    observation = _validation_observation(trajectory_bundle)

    class _BadVenueBridge(PolicyRuntimeBridge):
        def decide(self, artifact, observation):
            return super().decide(artifact, observation).model_copy(update={"venue": "not-a-venue"})

    bad_bridge = _BadVenueBridge()
    with pytest.raises(ValueError, match="not allowed by the artifact"):
        bad_bridge.build_execution_intent(policy_artifact, observation)


def _legacy_linear_artifact(policy_artifact):
    legacy_tags = [
        tag
        for tag in policy_artifact.runtime_metadata.artifact_compatibility_tags
        if not tag.startswith(
            (
                "runtime_contract:",
                "policy_kind:",
                "derived_contract:",
                "derived_signature:",
                "feature_dim:",
                "compat_mode:",
            )
        )
    ]
    legacy_metadata = policy_artifact.runtime_metadata.model_copy(
        update={
            "strict_runtime_contract": None,
            "artifact_compatibility_tags": legacy_tags,
        },
        deep=True,
    )
    return policy_artifact.model_copy(
        update={
            "schema_version": LEGACY_POLICY_ARTIFACT_SCHEMA_VERSION,
            "artifact_version": LEGACY_POLICY_ARTIFACT_SCHEMA_VERSION,
            "runtime_metadata": legacy_metadata,
        },
        deep=True,
    )


def _validation_observation(trajectory_bundle):
    return trajectory_bundle.splits["validation"][0].steps[0].observation.model_copy(deep=True)

from __future__ import annotations

import pytest

from quantlab_ml.contracts import ExecutionIntent
from quantlab_ml.policies import PolicyRuntimeBridge


def test_policy_artifact_runtime_metadata_matches_canonical_contract(policy_artifact) -> None:
    metadata = policy_artifact.runtime_metadata

    assert policy_artifact.artifact_id
    assert policy_artifact.artifact_version == "policy_artifact_v1"
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


def test_runtime_bridge_builds_execution_intent_from_artifact_and_observation(
    trajectory_bundle,
    policy_artifact,
) -> None:
    bridge = PolicyRuntimeBridge()
    observation = trajectory_bundle.splits["validation"][0].steps[0].observation

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


def test_runtime_bridge_rejects_incompatible_observation_schema(policy_artifact, trajectory_bundle) -> None:
    bridge = PolicyRuntimeBridge()
    observation = trajectory_bundle.splits["validation"][0].steps[0].observation.model_copy(deep=True)
    incompatible_streams = [stream for stream in observation.observation_schema.stream_axis if stream != "funding"]
    incompatible_field_axis = {
        stream: fields
        for stream, fields in observation.observation_schema.field_axis.items()
        if stream != "funding"
    }
    observation.observation_schema = observation.observation_schema.model_copy(
        update={
            "stream_axis": incompatible_streams,
            "field_axis": incompatible_field_axis,
        }
    )

    with pytest.raises(ValueError, match="missing required streams"):
        bridge.decide(policy_artifact, observation)


def test_runtime_bridge_rejects_unknown_decision_venue(policy_artifact, trajectory_bundle) -> None:
    observation = trajectory_bundle.splits["validation"][0].steps[0].observation

    class _BadVenueBridge(PolicyRuntimeBridge):
        def decide(self, artifact, observation):
            return super().decide(artifact, observation).model_copy(update={"venue": "not-a-venue"})

    bad_bridge = _BadVenueBridge()
    with pytest.raises(ValueError, match="not allowed by the artifact"):
        bad_bridge.build_execution_intent(policy_artifact, observation)

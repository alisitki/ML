from __future__ import annotations

from pathlib import Path
import shutil

import pytest

from quantlab_ml.contracts import (
    ActionFeasibilitySurface,
    EvaluationBoundary,
    FeasibilityCell,
)
from quantlab_ml.data import LocalFixtureSource
from quantlab_ml.evaluation import EvaluationEngine
from quantlab_ml.models import RuntimeDecision
from quantlab_ml.policies import PolicyRuntimeBridge
from quantlab_ml.rewards import RewardEngine
from quantlab_ml.trajectories import TrajectoryBuilder, TrajectoryDirectoryStore
from quantlab_ml.trajectories.tensor_cache import tensor_cache_directory


def test_evaluation_respects_boundary(trajectory_bundle, policy_artifact, evaluation_boundary: EvaluationBoundary):
    report = EvaluationEngine(evaluation_boundary).evaluate(trajectory_bundle, policy_artifact)

    assert report.boundary.fill_assumption_mode == "next_mark_price"
    assert report.boundary.infeasible_action_treatment == "force_abstain"
    assert report.boundary.fill_assumption_mode == trajectory_bundle.reward_spec.timestamping
    assert report.total_steps == sum(len(item.steps) for item in trajectory_bundle.splits["validation"])
    assert set(report.action_counts) >= {"abstain", "enter_long", "enter_short"}
    assert report.realized_trade_count >= 0


def test_evaluation_records_matches_bundle_path(
    trajectory_bundle,
    policy_artifact,
    evaluation_boundary: EvaluationBoundary,
):
    bundle_engine = EvaluationEngine(evaluation_boundary)
    stream_engine = EvaluationEngine(evaluation_boundary)

    bundle_report = bundle_engine.evaluate(trajectory_bundle, policy_artifact)
    stream_report = stream_engine.evaluate_records(
        trajectory_bundle.dataset_spec,
        trajectory_bundle.reward_spec,
        (record.model_copy(deep=True) for record in trajectory_bundle.splits["validation"]),
        policy_artifact,
    )

    assert stream_report.total_steps == bundle_report.total_steps
    assert stream_report.total_net_return == bundle_report.total_net_return
    assert stream_report.action_counts == bundle_report.action_counts
    assert stream_report.coverage_symbols == bundle_report.coverage_symbols
    assert stream_report.coverage_venues == bundle_report.coverage_venues
    assert stream_report.coverage_streams == bundle_report.coverage_streams


def test_evaluation_directory_matches_record_stream_path(
    fixture_path: Path,
    tmp_path: Path,
    dataset_spec,
    training_bundle,
    reward_spec,
    policy_artifact,
    evaluation_boundary: EvaluationBoundary,
) -> None:
    trajectory_spec, action_space, _ = training_bundle
    events = LocalFixtureSource(fixture_path).load_events(dataset_spec)
    builder = TrajectoryBuilder(dataset_spec, trajectory_spec, action_space, reward_spec)
    builder.build_to_directory(events, tmp_path)

    manifest = TrajectoryDirectoryStore.read_manifest(tmp_path)
    cache_engine = EvaluationEngine(evaluation_boundary)
    record_engine = EvaluationEngine(evaluation_boundary)

    cache_report = cache_engine.evaluate_directory(
        manifest=manifest,
        directory=tmp_path,
        artifact=policy_artifact,
        split_name="validation",
    )
    record_report = record_engine.evaluate_records(
        manifest.dataset_spec,
        manifest.reward_spec,
        TrajectoryDirectoryStore.iter_records(tmp_path, "validation"),
        policy_artifact,
    )

    assert cache_report.total_steps == record_report.total_steps
    assert cache_report.total_net_return == record_report.total_net_return
    assert cache_report.action_counts == record_report.action_counts
    assert cache_report.coverage_symbols == record_report.coverage_symbols
    assert cache_report.coverage_venues == record_report.coverage_venues
    assert cache_report.coverage_streams == record_report.coverage_streams


def test_evaluation_directory_uses_tensor_cache_batch_path(
    fixture_path: Path,
    tmp_path: Path,
    dataset_spec,
    training_bundle,
    reward_spec,
    policy_artifact,
    evaluation_boundary: EvaluationBoundary,
    monkeypatch,
) -> None:
    trajectory_spec, action_space, _ = training_bundle
    events = LocalFixtureSource(fixture_path).load_events(dataset_spec)
    builder = TrajectoryBuilder(dataset_spec, trajectory_spec, action_space, reward_spec)
    builder.build_to_directory(events, tmp_path)

    def _boom(*args, **kwargs):
        raise AssertionError("tensor-cache evaluation must not fall back to JSONL iteration")

    def _bridge_boom(self, artifact, observation):
        raise AssertionError("tensor-cache evaluation must not use PolicyRuntimeBridge.decide()")

    monkeypatch.setattr(TrajectoryDirectoryStore, "iter_records", _boom)
    monkeypatch.setattr(PolicyRuntimeBridge, "decide", _bridge_boom)

    manifest = TrajectoryDirectoryStore.read_manifest(tmp_path)
    engine = EvaluationEngine(evaluation_boundary)
    report = engine.evaluate_directory(
        manifest=manifest,
        directory=tmp_path,
        artifact=policy_artifact,
        split_name="validation",
    )

    assert report.total_steps > 0


def test_evaluation_directory_requires_explicit_jsonl_fallback_when_cache_missing(
    fixture_path: Path,
    tmp_path: Path,
    dataset_spec,
    training_bundle,
    reward_spec,
    policy_artifact,
    evaluation_boundary: EvaluationBoundary,
) -> None:
    trajectory_spec, action_space, _ = training_bundle
    events = LocalFixtureSource(fixture_path).load_events(dataset_spec)
    builder = TrajectoryBuilder(dataset_spec, trajectory_spec, action_space, reward_spec)
    builder.build_to_directory(events, tmp_path)
    shutil.rmtree(tensor_cache_directory(tmp_path))

    manifest = TrajectoryDirectoryStore.read_manifest(tmp_path)
    engine = EvaluationEngine(evaluation_boundary)

    with pytest.raises(ValueError, match="tensor cache manifest missing"):
        engine.evaluate_directory(
            manifest=manifest,
            directory=tmp_path,
            artifact=policy_artifact,
            split_name="validation",
        )

    report = engine.evaluate_directory(
        manifest=manifest,
        directory=tmp_path,
        artifact=policy_artifact,
        split_name="validation",
        allow_jsonl_fallback=True,
    )
    assert report.total_steps > 0


def test_evaluation_uses_explicit_requested_venue(
    trajectory_bundle,
    policy_artifact,
    evaluation_boundary: EvaluationBoundary,
):
    engine = EvaluationEngine(evaluation_boundary)
    engine.runtime_bridge = _ExplicitVenueBridge(venue="bybit")
    bybit_report = engine.evaluate(trajectory_bundle, policy_artifact)

    engine.runtime_bridge = _ExplicitVenueBridge(venue="binance")
    binance_report = engine.evaluate(trajectory_bundle, policy_artifact)

    assert bybit_report.total_steps == binance_report.total_steps
    assert bybit_report.total_net_return != binance_report.total_net_return


def test_evaluation_propagates_selected_venue_into_reward_context(
    trajectory_bundle,
    policy_artifact,
    evaluation_boundary: EvaluationBoundary,
):
    engine = EvaluationEngine(evaluation_boundary)
    engine.runtime_bridge = _ExplicitVenueBridge(venue="bybit")

    engine.evaluate(trajectory_bundle, policy_artifact)

    first_step = trajectory_bundle.splits["validation"][0].steps[0]
    assert first_step.reward_context.selected_venue == "bybit"
    assert first_step.reward_snapshot.context is not None
    assert first_step.reward_snapshot.context.selected_venue == "bybit"


def _make_infeasible_feasibility(action_space) -> ActionFeasibilitySurface:
    """abstain=True, directional=False olan feasibility surface."""
    surface = {}
    for action in action_space.actions:
        surface[action.key] = {}
        feasible = action.key == "abstain"
        for venue in ["binance", "bybit", "okx"]:
            surface[action.key][venue] = {
                "micro": {
                    "low": FeasibilityCell(feasible=feasible, reason="" if feasible else "forced_infeasible"),
                    "medium": FeasibilityCell(feasible=feasible, reason="" if feasible else "forced_infeasible"),
                },
                "small": {
                    "low": FeasibilityCell(feasible=feasible, reason="" if feasible else "forced_infeasible"),
                    "medium": FeasibilityCell(feasible=feasible, reason="" if feasible else "forced_infeasible"),
                },
            }
    return ActionFeasibilitySurface(surface=surface)


def test_evaluation_force_abstain_keeps_infeasible_penalty(
    trajectory_bundle,
    policy_artifact,
    evaluation_boundary: EvaluationBoundary,
):
    reward_engine = RewardEngine(trajectory_bundle.reward_spec, trajectory_bundle.action_space)
    validation_trajectory = trajectory_bundle.splits["validation"][0].model_copy(deep=True)

    infeasible_feasibility = _make_infeasible_feasibility(trajectory_bundle.action_space)

    for step in validation_trajectory.steps:
        # Replace feasibility with all-infeasible surface (except abstain)
        object.__setattr__(step, "action_feasibility", infeasible_feasibility)
        # Rebuild reward_snapshot using V2 API
        new_snapshot = reward_engine.build_snapshot(
            event_time=step.event_time,
            reward_context=step.reward_context,
            reward_timeline=step.reward_timeline,
            action_feasibility=infeasible_feasibility,
        )
        object.__setattr__(step, "reward_snapshot", new_snapshot)

    trajectory_bundle.splits["validation"] = [validation_trajectory]

    engine = EvaluationEngine(evaluation_boundary)
    engine.runtime_bridge = _AlwaysLongBridge()
    report = engine.evaluate(trajectory_bundle, policy_artifact)

    assert report.infeasible_action_count == len(validation_trajectory.steps)
    assert report.infeasible_penalty_total == (
        trajectory_bundle.reward_spec.infeasible_action_penalty * len(validation_trajectory.steps)
    )
    assert report.total_net_return == report.infeasible_penalty_total
    assert report.action_counts["abstain"] == len(validation_trajectory.steps)
    assert report.realized_trade_count == 0


class _AlwaysLongBridge:
    def decide(self, artifact, observation) -> RuntimeDecision:
        return RuntimeDecision(
            action_key="enter_long",
            venue="binance",
            size_band_key="micro",
            leverage_band_key="low",
        )


class _ExplicitVenueBridge:
    def __init__(self, venue: str):
        self.venue = venue

    def decide(self, artifact, observation) -> RuntimeDecision:
        return RuntimeDecision(
            action_key="enter_long",
            venue=self.venue,
            size_band_key="micro",
            leverage_band_key="low",
        )

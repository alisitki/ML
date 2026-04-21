from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quantlab_ml.common import load_yaml  # noqa: E402
from quantlab_ml.contracts import (  # noqa: E402
    ActionSpaceSpec,
    DatasetSpec,
    EvaluationBoundary,
    NormalizedMarketEvent,
    RewardEventSpec,
    TrajectorySpec,
)
from quantlab_ml.data import LocalFixtureSource  # noqa: E402
from quantlab_ml.evaluation import EvaluationEngine  # noqa: E402
from quantlab_ml.scoring import PolicyScorer  # noqa: E402
from quantlab_ml.training import LinearPolicyTrainer, TrainingConfig  # noqa: E402
from quantlab_ml.trajectories import TrajectoryBuilder  # noqa: E402


@pytest.fixture
def repo_root() -> Path:
    return ROOT


@pytest.fixture
def fixture_path(repo_root: Path) -> Path:
    return repo_root / "tests" / "fixtures" / "market_events.ndjson"


@pytest.fixture
def dataset_spec(repo_root: Path) -> DatasetSpec:
    return DatasetSpec.model_validate(load_yaml(repo_root / "configs" / "data" / "fixture.yaml")["dataset"])


@pytest.fixture
def training_bundle(repo_root: Path) -> tuple[TrajectorySpec, ActionSpaceSpec, TrainingConfig]:
    raw = load_yaml(repo_root / "configs" / "training" / "default.yaml")
    return (
        TrajectorySpec.model_validate(raw["trajectory"]),
        ActionSpaceSpec.model_validate(raw["action_space"]),
        TrainingConfig.model_validate(raw["trainer"]),
    )


@pytest.fixture
def search_training_bundle(repo_root: Path) -> tuple[TrajectorySpec, ActionSpaceSpec, TrainingConfig]:
    raw = load_yaml(repo_root / "configs" / "training" / "search-small.yaml")
    return (
        TrajectorySpec.model_validate(raw["trajectory"]),
        ActionSpaceSpec.model_validate(raw["action_space"]),
        TrainingConfig.model_validate(raw["trainer"]),
    )


@pytest.fixture
def phase1a_training_bundle(repo_root: Path) -> tuple[TrajectorySpec, ActionSpaceSpec, TrainingConfig]:
    raw = load_yaml(repo_root / "configs" / "training" / "phase1a-flat-v2.yaml")
    return (
        TrajectorySpec.model_validate(raw["trajectory"]),
        ActionSpaceSpec.model_validate(raw["action_space"]),
        TrainingConfig.model_validate(raw["trainer"]),
    )


@pytest.fixture
def phase1a_dataset_spec() -> DatasetSpec:
    return DatasetSpec.model_validate(
        {
            "dataset_hash": "phase1a-fixture-dataset",
            "slice_id": "fixture-phase1a-v2",
            "exchanges": ["binance", "bybit", "okx"],
            "symbols": ["BTCUSDT"],
            "stream_universe": ["mark_price"],
            "available_streams_by_exchange": {
                "binance": ["mark_price"],
                "bybit": ["mark_price"],
                "okx": ["mark_price"],
            },
            "train_range": {
                "start": "2024-01-01T00:00:00Z",
                "end": "2024-01-01T00:05:00Z",
            },
            "validation_range": {
                "start": "2024-01-01T00:06:00Z",
                "end": "2024-01-01T00:09:00Z",
            },
            "final_untouched_test_range": {
                "start": "2024-01-01T00:10:00Z",
                "end": "2024-01-01T00:11:00Z",
            },
            "walkforward": {
                "train_window_steps": 4,
                "validation_window_steps": 2,
                "step_size_steps": 1,
            },
            "sampling_interval_seconds": 60,
        }
    )


@pytest.fixture
def reward_spec(repo_root: Path) -> RewardEventSpec:
    return RewardEventSpec.model_validate(load_yaml(repo_root / "configs" / "reward" / "default.yaml")["reward"])


@pytest.fixture
def evaluation_boundary(repo_root: Path) -> EvaluationBoundary:
    return EvaluationBoundary.model_validate(
        load_yaml(repo_root / "configs" / "evaluation" / "default.yaml")["evaluation"]
    )


@pytest.fixture
def trajectory_bundle(
    fixture_path: Path,
    dataset_spec: DatasetSpec,
    training_bundle: tuple[TrajectorySpec, ActionSpaceSpec, TrainingConfig],
    reward_spec: RewardEventSpec,
):
    trajectory_spec, action_space, _ = training_bundle
    source = LocalFixtureSource(fixture_path)
    events = source.load_events(dataset_spec)
    builder = TrajectoryBuilder(dataset_spec, trajectory_spec, action_space, reward_spec)
    return builder.build(events)


@pytest.fixture
def policy_artifact(
    trajectory_bundle,
    training_bundle: tuple[TrajectorySpec, ActionSpaceSpec, TrainingConfig],
):
    _, _, training_config = training_bundle
    trainer = LinearPolicyTrainer(training_config)
    return trainer.train(trajectory_bundle)


@pytest.fixture
def phase1a_events(phase1a_dataset_spec: DatasetSpec) -> list[NormalizedMarketEvent]:
    base_prices = {
        "binance": 100.0,
        "bybit": 99.0,
        "okx": 98.5,
    }
    events: list[NormalizedMarketEvent] = []
    for minute in range(12):
        event_time = datetime(2024, 1, 1, 0, minute, tzinfo=UTC)
        for exchange, base in base_prices.items():
            events.append(
                NormalizedMarketEvent.model_validate(
                    {
                        "event_time": event_time,
                        "exchange": exchange,
                        "symbol": phase1a_dataset_spec.symbols[0],
                        "stream_type": "mark_price",
                        "fields": {
                            "mark_price": base + minute,
                            "event_delta": 1.0,
                            "index_price_if_available": base + minute,
                        },
                    }
                )
            )
    return events


@pytest.fixture
def phase1a_trajectory_bundle(
    phase1a_dataset_spec: DatasetSpec,
    phase1a_events: list[NormalizedMarketEvent],
    phase1a_training_bundle: tuple[TrajectorySpec, ActionSpaceSpec, TrainingConfig],
    reward_spec: RewardEventSpec,
):
    trajectory_spec, action_space, _ = phase1a_training_bundle
    builder = TrajectoryBuilder(phase1a_dataset_spec, trajectory_spec, action_space, reward_spec)
    return builder.build(phase1a_events)


@pytest.fixture
def phase1a_policy_artifact(
    phase1a_trajectory_bundle,
    phase1a_training_bundle: tuple[TrajectorySpec, ActionSpaceSpec, TrainingConfig],
):
    _, _, training_config = phase1a_training_bundle
    trainer = LinearPolicyTrainer(training_config)
    return trainer.train(phase1a_trajectory_bundle)


@pytest.fixture
def evaluation_report(trajectory_bundle, policy_artifact, evaluation_boundary: EvaluationBoundary):
    engine = EvaluationEngine(evaluation_boundary)
    return engine.evaluate(trajectory_bundle, policy_artifact)


@pytest.fixture
def policy_score(evaluation_report):
    return PolicyScorer().score(evaluation_report)

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quantlab_ml.common import load_yaml  # noqa: E402
from quantlab_ml.contracts import ActionSpaceSpec, DatasetSpec, EvaluationBoundary, RewardEventSpec, TrajectorySpec  # noqa: E402
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
def evaluation_report(trajectory_bundle, policy_artifact, evaluation_boundary: EvaluationBoundary):
    engine = EvaluationEngine(evaluation_boundary)
    return engine.evaluate(trajectory_bundle, policy_artifact)


@pytest.fixture
def policy_score(evaluation_report):
    return PolicyScorer().score(evaluation_report)

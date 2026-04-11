from __future__ import annotations

import json
from pathlib import Path

import typer

from quantlab_ml.common import dump_json_data, dump_model, hash_payload, load_model, load_yaml
from quantlab_ml.contracts import (
    ActionSpaceSpec,
    DatasetSpec,
    EvaluationBoundary,
    EvaluationReport,
    PolicyArtifact,
    PolicyScore,
    RewardEventSpec,
    TrajectorySpec,
)
from quantlab_ml.data import LocalFixtureSource, LocalParquetSource, S3CompactedSource
from quantlab_ml.evaluation import EvaluationEngine
from quantlab_ml.policies import PolicyRuntimeBridge
from quantlab_ml.registry import LocalRegistryStore
from quantlab_ml.scoring import PolicyScorer
from quantlab_ml.training import MomentumBaselineTrainer, TrainingConfig
from quantlab_ml.trajectories import TrajectoryBuilder, TrajectoryStore

app = typer.Typer(help="QuantLab ML scaffold CLI.")


@app.command("build-trajectories")
def build_trajectories(
    input: Path | None = typer.Option(None, help="Raw fixture or parquet input."),
    output: Path = typer.Option(..., help="Trajectory bundle output path."),
    data_config: Path = typer.Option(Path("configs/data/default.yaml"), exists=True),
    training_config: Path = typer.Option(Path("configs/training/default.yaml"), exists=True),
    reward_config: Path = typer.Option(Path("configs/reward/default.yaml"), exists=True),
    source: str = typer.Option("local", help="Data source: local or s3-compact."),
    s3_env_file: Path | None = typer.Option(None, help="S3 compact env file for source=s3-compact."),
) -> None:
    dataset_spec = _load_dataset_spec(data_config)
    trajectory_spec, action_space, _ = _load_training_bundle(training_config)
    reward_spec = _load_reward_spec(reward_config)
    events = _resolve_source(input, source, s3_env_file).load_events(dataset_spec)
    builder = TrajectoryBuilder(dataset_spec, trajectory_spec, action_space, reward_spec)
    bundle = builder.build(events)
    TrajectoryStore.write(output, bundle)
    typer.echo(f"wrote trajectories to {output}")


@app.command("inspect-s3-compact")
def inspect_s3_compact(
    env_file: Path = typer.Option(Path(".env"), exists=True, help="S3 compact env file."),
    data_config: Path | None = typer.Option(None, help="Optional dataset config to filter state summary."),
    output: Path | None = typer.Option(None, help="Optional JSON output path."),
) -> None:
    source = S3CompactedSource.from_env_file(env_file)
    dataset_spec = _load_dataset_spec(data_config) if data_config is not None else None
    summary = source.summarize_state(dataset_spec)
    if output is not None:
        dump_json_data(output, summary)
        typer.echo(f"wrote S3 compact summary to {output}")
        return
    typer.echo(json.dumps(summary, indent=2))


@app.command("train")
def train(
    trajectories: Path = typer.Option(..., exists=True, readable=True),
    output: Path = typer.Option(..., help="Policy artifact output path."),
    training_config: Path = typer.Option(Path("configs/training/default.yaml"), exists=True),
    registry_root: Path | None = typer.Option(None, help="Optional registry root."),
    parent_policy_id: str | None = typer.Option(None, help="Optional parent policy id."),
) -> None:
    bundle = TrajectoryStore.read(trajectories)
    _, _, training_config_model = _load_training_bundle(training_config)
    trainer = MomentumBaselineTrainer(training_config_model)
    artifact = trainer.train(bundle, parent_policy_id=parent_policy_id)
    dump_model(output, artifact)
    if registry_root is not None:
        registry = LocalRegistryStore(registry_root)
        registry.register_candidate(
            artifact,
            bundle,
            reward_config_hash=hash_payload(bundle.reward_spec),
            training_config_hash=hash_payload(training_config_model),
        )
    typer.echo(f"wrote policy artifact to {output}")


@app.command("evaluate")
def evaluate(
    trajectories: Path = typer.Option(..., exists=True, readable=True),
    policy: Path = typer.Option(..., exists=True, readable=True),
    output: Path = typer.Option(..., help="Evaluation report output path."),
    evaluation_config: Path = typer.Option(Path("configs/evaluation/default.yaml"), exists=True),
) -> None:
    bundle = TrajectoryStore.read(trajectories)
    artifact = load_model(policy, PolicyArtifact)
    boundary = _load_evaluation_boundary(evaluation_config)
    engine = EvaluationEngine(boundary)
    report = engine.evaluate(bundle, artifact)
    dump_model(output, report)
    typer.echo(f"wrote evaluation report to {output}")


@app.command("score")
def score(
    policy: Path = typer.Option(..., exists=True, readable=True),
    evaluation: Path = typer.Option(..., exists=True, readable=True),
    output: Path = typer.Option(..., help="Score output path."),
    registry_root: Path | None = typer.Option(None, help="Optional registry root."),
) -> None:
    artifact = load_model(policy, PolicyArtifact)
    report = load_model(evaluation, EvaluationReport)
    scorer = PolicyScorer()
    score_card = scorer.score(report)
    dump_model(output, score_card)
    if registry_root is not None:
        registry = LocalRegistryStore(registry_root)
        registry.append_score(artifact.policy_id, score_card, report)
    typer.echo(f"wrote score to {output}")


@app.command("export-policy")
def export_policy(
    policy: Path = typer.Option(..., exists=True, readable=True),
    score: Path = typer.Option(..., exists=True, readable=True),
    output: Path = typer.Option(..., help="Inference artifact export output path."),
) -> None:
    artifact = load_model(policy, PolicyArtifact)
    score_model = load_model(score, PolicyScore)
    bridge = PolicyRuntimeBridge()
    export = bridge.export(artifact, score_model)
    dump_model(output, export)
    typer.echo(f"wrote inference artifact export to {output}")


def main() -> None:
    app()


def _resolve_source(
    path: Path | None,
    source_kind: str,
    s3_env_file: Path | None,
) -> LocalFixtureSource | LocalParquetSource | S3CompactedSource:
    normalized_source = source_kind.lower()
    if normalized_source == "s3-compact":
        if s3_env_file is None:
            raise typer.BadParameter("--s3-env-file is required when --source s3-compact is used")
        return S3CompactedSource.from_env_file(s3_env_file)
    if normalized_source != "local":
        raise typer.BadParameter(f"unsupported source: {source_kind}")
    if path is None:
        raise typer.BadParameter("--input is required when --source local is used")
    if not path.exists():
        raise typer.BadParameter(f"input path does not exist: {path}")
    if path.suffix == ".parquet" or path.is_dir():
        return LocalParquetSource(path)
    return LocalFixtureSource(path)


def _load_dataset_spec(path: Path) -> DatasetSpec:
    return DatasetSpec.model_validate(load_yaml(path)["dataset"])


def _load_reward_spec(path: Path) -> RewardEventSpec:
    return RewardEventSpec.model_validate(load_yaml(path)["reward"])


def _load_evaluation_boundary(path: Path) -> EvaluationBoundary:
    return EvaluationBoundary.model_validate(load_yaml(path)["evaluation"])


def _load_training_bundle(path: Path) -> tuple[TrajectorySpec, ActionSpaceSpec, TrainingConfig]:
    config = load_yaml(path)
    trajectory = TrajectorySpec.model_validate(config["trajectory"])
    action_space = ActionSpaceSpec.model_validate(config["action_space"])
    training_config = TrainingConfig.model_validate(config["trainer"])
    return trajectory, action_space, training_config

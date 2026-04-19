from __future__ import annotations

import json
from pathlib import Path

import typer

from quantlab_ml.common import configure_logging, dump_json_data, dump_model, hash_payload, load_model, load_yaml
from quantlab_ml.contracts import (
    ActionSpaceSpec,
    DatasetSpec,
    EvaluationBoundary,
    EvaluationReport,
    PolicyArtifact,
    PolicyScore,
    RewardEventSpec,
    TrajectoryBundle,
    TrajectoryManifest,
    TrajectorySpec,
)
from quantlab_ml.data import LocalFixtureSource, LocalParquetSource, S3CompactedSource
from quantlab_ml.evaluation import EvaluationEngine
from quantlab_ml.policies import PolicyRuntimeBridge
from quantlab_ml.registry import (
    LocalRegistryStore,
    audit_registry_continuity,
    build_offline_evidence_pack,
    render_offline_evidence_pack_markdown,
)
from quantlab_ml.scoring import PolicyScorer
from quantlab_ml.training import LinearPolicyTrainer, TrainingConfig, TrainingSearchResult
from quantlab_ml.trajectories import TrajectoryBuilder, TrajectoryDirectoryStore, TrajectoryStore

app = typer.Typer(
    help="QuantLab ML CLI for canonical data, offline training/evaluation, and runtime artifact workflows."
)


@app.callback()
def app_callback() -> None:
    configure_logging()


@app.command("build-trajectories")
def build_trajectories(
    input: Path | None = typer.Option(None, help="Raw fixture or parquet input."),
    output: Path = typer.Option(..., help="Trajectory directory output path (streaming format)."),
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
    # PRODUCTION PATH: streaming build to JSONL directory (no giant bundle in memory)
    builder.build_to_directory(events, output)
    typer.echo(f"wrote trajectory directory to {output}")


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
    allow_jsonl_fallback: bool = typer.Option(
        False,
        help="Allow temporary JSONL-only compatibility fallback when tensor cache is missing.",
    ),
) -> None:
    _, _, training_config_model = _load_training_bundle(training_config)
    trainer = LinearPolicyTrainer(training_config_model)

    if TrajectoryDirectoryStore.is_trajectory_directory(trajectories):
        # PRODUCTION PATH: tensor-cache directory train; JSONL fallback is explicit compat only.
        manifest = TrajectoryDirectoryStore.read_manifest(trajectories)
        search_result = trainer.train_search_from_directory(
            manifest,
            trajectories,
            parent_policy_id=parent_policy_id,
            allow_jsonl_fallback=allow_jsonl_fallback,
        )
        dump_model(output, search_result.selected_artifact)
        if len(search_result.candidate_results) > 1:
            _write_search_artifacts(output, search_result)
        if registry_root is not None:
            registry = LocalRegistryStore(registry_root)
            _register_training_candidates_from_manifest(
                registry,
                manifest,
                trajectories,
                search_result,
            )
    else:
        # FIXTURE / TEST COMPAT PATH: in-memory bundle (legacy JSON file)
        bundle = TrajectoryStore.read(trajectories)
        search_result = trainer.train_search(bundle, parent_policy_id=parent_policy_id)
        dump_model(output, search_result.selected_artifact)
        if len(search_result.candidate_results) > 1:
            _write_search_artifacts(output, search_result)
        if registry_root is not None:
            registry = LocalRegistryStore(registry_root)
            _register_training_candidates(registry, bundle, search_result)

    typer.echo(f"wrote policy artifact to {output}")


@app.command("evaluate")
def evaluate(
    trajectories: Path = typer.Option(..., exists=True, readable=True),
    policy: Path = typer.Option(..., exists=True, readable=True),
    output: Path = typer.Option(..., help="Evaluation report output path."),
    evaluation_config: Path = typer.Option(Path("configs/evaluation/default.yaml"), exists=True),
    allow_jsonl_fallback: bool = typer.Option(
        False,
        help="Allow temporary JSONL-only compatibility fallback when tensor cache is missing.",
    ),
) -> None:
    artifact = load_model(policy, PolicyArtifact)
    boundary = _load_evaluation_boundary(evaluation_config)
    engine = EvaluationEngine(boundary)

    if TrajectoryDirectoryStore.is_trajectory_directory(trajectories):
        # PRODUCTION PATH: tensor-cache evaluation from the trajectory directory.
        manifest = TrajectoryDirectoryStore.read_manifest(trajectories)
        report = engine.evaluate_directory(
            manifest=manifest,
            directory=trajectories,
            artifact=artifact,
            split_name="final_untouched_test",
            allow_jsonl_fallback=allow_jsonl_fallback,
        )
    else:
        # FIXTURE / TEST COMPAT PATH
        bundle = TrajectoryStore.read(trajectories)
        report = engine.evaluate(bundle, artifact, split="final_untouched_test")

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


@app.command("record-paper-sim")
def record_paper_sim(
    registry_root: Path = typer.Option(..., help="Registry root."),
    policy_id: str = typer.Option(..., help="Registered policy id."),
    report: Path = typer.Option(..., exists=True, readable=True, help="Paper/sim report path."),
    comparison_report_id: str | None = typer.Option(
        None,
        help="Comparison report id. Required when recording paper/sim for a challenger against a current champion.",
    ),
) -> None:
    registry = LocalRegistryStore(registry_root)
    evidence = registry.record_paper_sim_evidence(
        policy_id,
        report,
        comparison_report_id=comparison_report_id,
    )
    typer.echo(f"recorded paper/sim evidence {evidence.evidence_id}")


@app.command("compare-policies")
def compare_policies(
    registry_root: Path = typer.Option(..., help="Registry root."),
    challenger_policy_id: str = typer.Option(..., help="Registered challenger policy id."),
    champion_policy_id: str | None = typer.Option(None, help="Optional explicit champion policy id."),
    output: Path | None = typer.Option(None, help="Optional comparison report output path."),
) -> None:
    registry = LocalRegistryStore(registry_root)
    report = registry.record_comparison_report(
        challenger_policy_id,
        champion_policy_id=champion_policy_id,
    )
    if output is not None:
        dump_model(output, report)
        typer.echo(f"wrote comparison report to {output}")
        return
    typer.echo(f"recorded comparison report {report.comparison_report_id}")


@app.command("build-offline-evidence-pack")
def build_offline_evidence_pack_command(
    registry_root: list[Path] = typer.Option(..., help="Registry root. Repeat for multiple retained surfaces."),
    inspected_evidence_kind: list[str] = typer.Option(
        ["external-retained-evidence"],
        help=(
            "Inspected evidence class. Repeat to align with registry roots, or pass once to apply to all roots. "
            "Allowed values mirror audit-continuity."
        ),
    ),
    authority_status: list[str] = typer.Option(
        [],
        help="Authority status per retained surface. Repeat to align with registry roots, or omit to use defaults.",
    ),
    output: Path = typer.Option(..., help="Offline evidence pack output path (.md or .json)."),
) -> None:
    inspected_evidence_kinds = _expand_option_values(
        registry_root,
        inspected_evidence_kind,
        option_name="inspected_evidence_kind",
    )
    authority_statuses = _expand_option_values(
        registry_root,
        authority_status,
        option_name="authority_status",
        allow_empty=True,
    )
    pack = build_offline_evidence_pack(
        registry_roots=registry_root,
        inspected_evidence_kinds=inspected_evidence_kinds,
        authority_statuses=authority_statuses,
    )
    if output.suffix.lower() == ".json":
        dump_json_data(output, pack)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(render_offline_evidence_pack_markdown(pack), encoding="utf-8")
    typer.echo(f"wrote offline evidence pack to {output}")


@app.command("audit-continuity")
def audit_continuity(
    registry_root: Path = typer.Option(..., help="Registry root."),
    output: Path | None = typer.Option(None, help="Optional JSON output path."),
    inspected_evidence_kind: str = typer.Option(
        "external-retained-evidence",
        help=(
            "How to classify the inspected scope: "
            "repo-tracked-artifact, external-retained-evidence, or authoritative-evidence."
        ),
    ),
    authority_status: str | None = typer.Option(
        None,
        help="Optional authority status: confirmed, unconfirmed, or unknown.",
    ),
) -> None:
    registry = LocalRegistryStore(registry_root)
    summary = audit_registry_continuity(
        registry,
        inspected_evidence_kind=inspected_evidence_kind,
        authority_status=authority_status,
    )
    if output is not None:
        dump_json_data(output, summary)
        typer.echo(f"wrote continuity audit to {output}")
        return
    typer.echo(json.dumps(summary, indent=2))


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


def _expand_option_values(
    registry_roots: list[Path],
    raw_values: list[str],
    *,
    option_name: str,
    allow_empty: bool = False,
) -> list[str | None]:
    if not raw_values:
        if allow_empty:
            return [None] * len(registry_roots)
        raise typer.BadParameter(f"--{option_name.replace('_', '-')} is required")
    if len(raw_values) == 1:
        return raw_values * len(registry_roots)
    if len(raw_values) != len(registry_roots):
        raise typer.BadParameter(
            f"--{option_name.replace('_', '-')} must be provided once or once per --registry-root"
        )
    return raw_values


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


def _write_search_artifacts(output: Path, search_result: TrainingSearchResult) -> None:
    candidate_dir = output.with_name(f"{output.stem}_candidates")
    manifest_path = output.with_name(f"{output.stem}_search.json")
    candidates: list[dict[str, object]] = []

    for candidate in search_result.candidate_results:
        artifact_path = output
        if not candidate.selected_candidate:
            artifact_path = candidate_dir / f"{candidate.artifact.policy_id}.json"
            dump_model(artifact_path, candidate.artifact)
        candidates.append(
            {
                "policy_id": candidate.artifact.policy_id,
                "artifact_id": candidate.artifact.artifact_id,
                "artifact_path": str(artifact_path),
                "candidate_index": candidate.candidate_index,
                "candidate_rank": candidate.candidate_rank,
                "selected_candidate": candidate.selected_candidate,
                "candidate_spec": candidate.candidate_spec.as_dict(),
                "best_validation_total_net_return": candidate.best_validation_total_net_return,
                "best_validation_composite_rank": candidate.best_validation_composite_rank,
            }
        )

    dump_json_data(
        manifest_path,
        {
            "training_run_id": search_result.training_run_id,
            "selected_policy_id": search_result.selected_artifact.policy_id,
            "selected_artifact_path": str(output),
            "search_budget_summary": search_result.search_budget_summary.model_dump(mode="json"),
            "candidates": candidates,
        },
    )


def _register_training_candidates(
    registry: LocalRegistryStore,
    bundle: TrajectoryBundle,
    search_result: TrainingSearchResult,
) -> None:
    """FIXTURE / TEST COMPAT PATH — register from in-memory bundle."""
    reward_config_hash = hash_payload(bundle.reward_spec)
    candidates = (
        search_result.candidate_results
        if len(search_result.candidate_results) > 1
        else [search_result.candidate_results[0]]
    )
    for candidate in candidates:
        registry.register_candidate(
            candidate.artifact,
            bundle,
            reward_config_hash=reward_config_hash,
            training_config_hash=candidate.artifact.training_config_hash,
        )


def _register_training_candidates_from_manifest(
    registry: LocalRegistryStore,
    manifest: "TrajectoryManifest",
    trajectory_directory: Path,
    search_result: TrainingSearchResult,
) -> None:
    """PRODUCTION PATH — register from streaming manifest (no bundle needed)."""
    reward_config_hash = hash_payload(manifest.reward_spec)
    candidates = (
        search_result.candidate_results
        if len(search_result.candidate_results) > 1
        else [search_result.candidate_results[0]]
    )
    for candidate in candidates:
        registry.register_candidate_from_manifest(
            candidate.artifact,
            manifest,
            reward_config_hash=reward_config_hash,
            training_config_hash=candidate.artifact.training_config_hash,
            trajectory_directory=trajectory_directory,
        )

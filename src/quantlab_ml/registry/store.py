from __future__ import annotations

from pathlib import Path

from quantlab_ml.common import dump_model, load_model, utcnow
from quantlab_ml.contracts import (
    CoverageStats,
    EvaluationReport,
    PolicyArtifact,
    PolicyScore,
    RegistryIndex,
    RegistryRecord,
    ScoreSnapshot,
    TimeRange,
    TrajectoryBundle,
)
from quantlab_ml.selection import CandidateSelector


class LocalRegistryStore:
    def __init__(self, root: Path):
        self.root = root
        self.artifacts_dir = root / "artifacts"
        self.records_dir = root / "records"
        self.scores_dir = root / "scores"
        self.evaluations_dir = root / "evaluations"
        for directory in (
            self.root,
            self.artifacts_dir,
            self.records_dir,
            self.scores_dir,
            self.evaluations_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def register_candidate(
        self,
        artifact: PolicyArtifact,
        bundle: TrajectoryBundle,
        reward_config_hash: str,
        training_config_hash: str,
    ) -> RegistryRecord:
        artifact_path = self.artifacts_dir / f"{artifact.policy_id}.json"
        dump_model(artifact_path, artifact)
        coverage = _coverage_from_bundle(bundle)
        now = utcnow()
        record = RegistryRecord(
            policy_id=artifact.policy_id,
            artifact_path=str(artifact_path),
            dataset_hash=bundle.dataset_spec.dataset_hash,
            slice_id=bundle.dataset_spec.slice_id,
            reward_config_hash=reward_config_hash,
            training_config_hash=training_config_hash,
            parent_policy_id=artifact.executor_metadata.lineage_pointer.parent_policy_id,
            lineage_chain=_lineage_chain(artifact),
            status="candidate",
            train_window=bundle.dataset_spec.train_range,
            coverage=coverage,
            created_at=now,
            updated_at=now,
        )
        dump_model(self.records_dir / f"{record.policy_id}.json", record)
        self._recompute_index()
        return self.get_record(artifact.policy_id)

    def append_score(
        self,
        policy_id: str,
        score: PolicyScore,
        evaluation_report: EvaluationReport,
    ) -> RegistryRecord:
        record = self.get_record(policy_id)
        if record is None:
            raise FileNotFoundError(f"registry record not found for policy {policy_id}")
        record.score_history.append(ScoreSnapshot(recorded_at=utcnow(), score=score))
        record.eval_window = evaluation_report.active_date_range
        record.coverage.eval_sample_count = evaluation_report.total_steps
        record.coverage.reward_event_count = evaluation_report.total_steps
        record.coverage.realized_trade_count = evaluation_report.realized_trade_count
        record.coverage.active_date_range = _merge_ranges(
            record.train_window, evaluation_report.active_date_range
        )
        record.updated_at = utcnow()
        dump_model(self.scores_dir / f"{policy_id}.json", score)
        dump_model(self.evaluations_dir / f"{policy_id}.json", evaluation_report)
        dump_model(self.records_dir / f"{policy_id}.json", record)
        self._recompute_index()
        return self.get_record(policy_id)

    def get_record(self, policy_id: str) -> RegistryRecord | None:
        path = self.records_dir / f"{policy_id}.json"
        if not path.exists():
            return None
        return load_model(path, RegistryRecord)

    def list_records(self) -> list[RegistryRecord]:
        return [load_model(path, RegistryRecord) for path in sorted(self.records_dir.glob("*.json"))]

    def load_index(self) -> RegistryIndex:
        path = self.root / "index.json"
        if not path.exists():
            return RegistryIndex()
        return load_model(path, RegistryIndex)

    def _recompute_index(self) -> None:
        records = self.list_records()
        selector = CandidateSelector()
        champion_policy_id = selector.choose_champion(records)
        challenger_ids = []
        for record in records:
            if champion_policy_id is not None and record.policy_id == champion_policy_id:
                record.status = "champion"
            elif record.score_history:
                record.status = "challenger"
                challenger_ids.append(record.policy_id)
            else:
                record.status = "candidate"
            record.updated_at = utcnow()
            dump_model(self.records_dir / f"{record.policy_id}.json", record)
        dump_model(
            self.root / "index.json",
            RegistryIndex(champion_policy_id=champion_policy_id, challenger_policy_ids=challenger_ids),
        )


def _coverage_from_bundle(bundle: TrajectoryBundle) -> CoverageStats:
    train_steps = sum(len(trajectory.steps) for trajectory in bundle.splits["train"])
    return CoverageStats(
        train_sample_count=train_steps,
        eval_sample_count=0,
        covered_symbols=bundle.dataset_spec.symbols,
        covered_venues=bundle.dataset_spec.exchanges,
        covered_streams=bundle.dataset_spec.stream_universe,
        active_date_range=bundle.dataset_spec.train_range,
        reward_event_count=0,
        realized_trade_count=0,
    )


def _merge_ranges(left: TimeRange, right: TimeRange) -> TimeRange:
    return TimeRange(
        start=min(left.start, right.start),
        end=max(left.end, right.end),
    )


def _lineage_chain(artifact: PolicyArtifact) -> list[str]:
    if artifact.executor_metadata.lineage_pointer.parent_policy_id is None:
        return []
    return [artifact.executor_metadata.lineage_pointer.parent_policy_id]

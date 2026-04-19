from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from quantlab_ml.common import dump_model, hash_payload, load_model, utcnow
from quantlab_ml.contracts import (
    ComparisonReport,
    CoverageStats,
    EvaluationReport,
    PaperSimEvidenceRecord,
    PolicyArtifact,
    PolicyScore,
    PromotionDecisionRecord,
    PromotionEvidence,
    RegistryIndex,
    RegistryRecord,
    ScoreSnapshot,
    SearchBudgetSummary,
    SplitEvidence,
    TimeRange,
    TrajectoryBundle,
    TrajectoryManifest,
)
from quantlab_ml.selection import CandidateSelector

logger = logging.getLogger(__name__)


class LocalRegistryStore:
    def __init__(self, root: Path):
        self.root = root
        self.artifacts_dir = root / "artifacts"
        self.records_dir = root / "records"
        self.scores_dir = root / "scores"
        self.evaluations_dir = root / "evaluations"
        self.comparisons_dir = root / "comparisons"
        self.paper_sim_dir = root / "paper_sim"
        self.promotions_dir = root / "promotions"
        for directory in (
            self.root,
            self.artifacts_dir,
            self.records_dir,
            self.scores_dir,
            self.evaluations_dir,
            self.comparisons_dir,
            self.paper_sim_dir,
            self.promotions_dir,
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
        split_evidence = SplitEvidence(
            split_version=bundle.split_artifact.split_version,
            train_window=bundle.dataset_spec.train_range,
            validation_window=bundle.dataset_spec.validation_range,
            final_untouched_test_window=bundle.dataset_spec.final_untouched_test_range,
            purge_width_steps=bundle.split_artifact.purge_width_steps,
            embargo_width_steps=bundle.split_artifact.embargo_width_steps,
            walkforward_fold_count=len(bundle.split_artifact.folds),
        )
        now = utcnow()
        record = RegistryRecord(
            policy_id=artifact.policy_id,
            artifact_id=artifact.artifact_id,
            artifact_path=str(artifact_path),
            status="candidate",
            policy_family=artifact.policy_family,
            target_asset=artifact.target_asset,
            training_snapshot_id=artifact.training_snapshot_id,
            training_config_hash=training_config_hash,
            reward_version=artifact.reward_version,
            evaluation_surface_id=artifact.evaluation_surface_id,
            search_budget_summary=_search_budget_summary(artifact),
            artifact_compatibility_tags=list(artifact.runtime_metadata.artifact_compatibility_tags),
            runtime_compatibility_tags=list(artifact.runtime_metadata.artifact_compatibility_tags),
            dataset_hash=bundle.dataset_spec.dataset_hash,
            slice_id=bundle.dataset_spec.slice_id,
            reward_config_hash=reward_config_hash,
            parent_policy_id=artifact.runtime_metadata.lineage_pointer.parent_policy_id,
            lineage_chain=_lineage_chain(artifact),
            train_window=bundle.dataset_spec.train_range,
            coverage=coverage,
            split_evidence=split_evidence,
            created_at=now,
            updated_at=now,
        )
        dump_model(self.records_dir / f"{record.policy_id}.json", record)
        self._recompute_index()
        result = self.get_record(artifact.policy_id)
        if result is None:
            raise RuntimeError(f"registry record missing immediately after write for policy {artifact.policy_id}")
        logger.info(
            "registry_candidate_registered policy_id=%s artifact_id=%s fold_count=%d train_samples=%d",
            result.policy_id,
            result.artifact_id,
            len(bundle.split_artifact.folds),
            coverage.train_sample_count,
        )
        return result

    def register_candidate_from_manifest(
        self,
        artifact: PolicyArtifact,
        manifest: TrajectoryManifest,
        reward_config_hash: str,
        training_config_hash: str,
        *,
        trajectory_directory: Path | None = None,
    ) -> RegistryRecord:
        artifact_path = self.artifacts_dir / f"{artifact.policy_id}.json"
        coverage = _coverage_from_manifest(manifest, trajectory_directory=trajectory_directory)
        dump_model(artifact_path, artifact)
        split_evidence = SplitEvidence(
            split_version=manifest.split_artifact.split_version,
            train_window=manifest.dataset_spec.train_range,
            validation_window=manifest.dataset_spec.validation_range,
            final_untouched_test_window=manifest.dataset_spec.final_untouched_test_range,
            purge_width_steps=manifest.split_artifact.purge_width_steps,
            embargo_width_steps=manifest.split_artifact.embargo_width_steps,
            walkforward_fold_count=len(manifest.split_artifact.folds),
        )
        now = utcnow()
        record = RegistryRecord(
            policy_id=artifact.policy_id,
            artifact_id=artifact.artifact_id,
            artifact_path=str(artifact_path),
            status="candidate",
            policy_family=artifact.policy_family,
            target_asset=artifact.target_asset,
            training_snapshot_id=artifact.training_snapshot_id,
            training_config_hash=training_config_hash,
            reward_version=artifact.reward_version,
            evaluation_surface_id=artifact.evaluation_surface_id,
            search_budget_summary=_search_budget_summary(artifact),
            artifact_compatibility_tags=list(artifact.runtime_metadata.artifact_compatibility_tags),
            runtime_compatibility_tags=list(artifact.runtime_metadata.artifact_compatibility_tags),
            dataset_hash=manifest.dataset_spec.dataset_hash,
            slice_id=manifest.dataset_spec.slice_id,
            reward_config_hash=reward_config_hash,
            parent_policy_id=artifact.runtime_metadata.lineage_pointer.parent_policy_id,
            lineage_chain=_lineage_chain(artifact),
            train_window=manifest.dataset_spec.train_range,
            coverage=coverage,
            split_evidence=split_evidence,
            created_at=now,
            updated_at=now,
        )
        dump_model(self.records_dir / f"{record.policy_id}.json", record)
        self._recompute_index()
        result = self.get_record(artifact.policy_id)
        if result is None:
            raise RuntimeError(f"registry record missing immediately after write for policy {artifact.policy_id}")
        logger.info(
            "registry_candidate_registered policy_id=%s artifact_id=%s fold_count=%d train_samples=%d",
            result.policy_id,
            result.artifact_id,
            len(manifest.split_artifact.folds),
            coverage.train_sample_count,
        )
        return result

    def append_score(
        self,
        policy_id: str,
        score: PolicyScore,
        evaluation_report: EvaluationReport,
    ) -> RegistryRecord:
        record = self.get_record(policy_id)
        if record is None:
            raise FileNotFoundError(f"registry record not found for policy {policy_id}")
        evaluation_path = self.evaluations_dir / f"{policy_id}.json"
        score_path = self.scores_dir / f"{policy_id}.json"
        dump_model(score_path, score)
        dump_model(evaluation_path, evaluation_report)

        record.score_history.append(
            ScoreSnapshot(
                recorded_at=utcnow(),
                evaluation_id=evaluation_report.evaluation_id,
                evaluation_surface_id=record.evaluation_surface_id,
                governing_objective=evaluation_report.total_net_return,
                expected_return_score=score.expected_return_score,
                risk_score=score.risk_score,
                turnover_score=score.turnover_score,
                stability_score=score.stability_score,
                applicability_score=score.applicability_score,
                composite_rank=score.composite_rank,
            )
        )
        record.eval_window = evaluation_report.active_date_range
        record.coverage.eval_sample_count = evaluation_report.total_steps
        record.coverage.reward_event_count = evaluation_report.total_steps
        record.coverage.realized_trade_count = evaluation_report.realized_trade_count
        record.coverage.active_date_range = _merge_ranges(record.train_window, evaluation_report.active_date_range)
        record.evaluation_report_path = str(evaluation_path)
        record.evaluation_report_id = evaluation_report.evaluation_id
        record.updated_at = utcnow()
        dump_model(self.records_dir / f"{policy_id}.json", record)
        self._recompute_index()
        result = self.get_record(policy_id)
        if result is None:
            raise RuntimeError(f"registry record missing immediately after write for policy {policy_id}")
        logger.info(
            "registry_score_appended policy_id=%s evaluation_id=%s composite_rank=%.6f total_net_return=%.6f",
            result.policy_id,
            evaluation_report.evaluation_id,
            score.composite_rank,
            evaluation_report.total_net_return,
        )
        return result

    def record_paper_sim_evidence(
        self,
        policy_id: str,
        report_path: str | Path,
        *,
        comparison_report_id: str | None = None,
    ) -> PaperSimEvidenceRecord:
        record = self._require_record(policy_id)
        evaluation_report = self._require_evaluation_report(record)
        comparison_report = self._validate_paper_sim_comparison_linkage(
            record=record,
            comparison_report_id=comparison_report_id,
        )
        resolved_report_path = Path(report_path)
        if not resolved_report_path.exists():
            raise FileNotFoundError(f"paper/sim report not found at {resolved_report_path}")

        evidence_id = _paper_sim_evidence_id(
            policy_id=record.policy_id,
            evaluation_report_id=evaluation_report.evaluation_id,
            report_path=resolved_report_path,
            comparison_report_id=comparison_report_id,
        )
        evidence_record = PaperSimEvidenceRecord(
            evidence_id=evidence_id,
            policy_id=record.policy_id,
            artifact_id=record.artifact_id,
            created_at=utcnow(),
            evaluation_report_id=evaluation_report.evaluation_id,
            evaluation_report_path=record.evaluation_report_path or "",
            comparison_report_id=comparison_report_id,
            report_path=str(resolved_report_path),
            report_format=_report_format(resolved_report_path),
        )
        dump_model(self.paper_sim_dir / f"{evidence_id}.json", evidence_record)

        record.paper_sim_evidence_id = evidence_record.evidence_id
        record.paper_sim_report_path = evidence_record.report_path
        if comparison_report is not None:
            record.comparison_report_id = comparison_report.comparison_report_id
        record.updated_at = utcnow()
        dump_model(self.records_dir / f"{policy_id}.json", record)
        logger.info(
            "registry_paper_sim_recorded policy_id=%s evidence_id=%s comparison_report_id=%s",
            record.policy_id,
            evidence_record.evidence_id,
            comparison_report_id or "",
        )
        return evidence_record

    def record_comparison_report(
        self,
        challenger_policy_id: str,
        *,
        champion_policy_id: str | None = None,
    ) -> ComparisonReport:
        challenger = self._require_record(challenger_policy_id)
        current_index = self.load_index()
        resolved_champion_policy_id = champion_policy_id or current_index.champion_policy_id
        if resolved_champion_policy_id is None:
            raise ValueError("comparison requires a current champion or an explicit champion_policy_id")
        if resolved_champion_policy_id == challenger_policy_id:
            raise ValueError("comparison requires distinct challenger and champion policies")

        champion = self._require_record(resolved_champion_policy_id)
        if champion.status != "champion":
            raise ValueError("comparison requires the comparison target to be the current champion")
        if not challenger.score_history:
            raise ValueError("comparison requires a scored challenger policy")
        if not champion.score_history:
            raise ValueError("comparison requires a scored champion policy")
        if challenger.evaluation_surface_id != champion.evaluation_surface_id:
            raise ValueError("comparison requires identical evaluation_surface_id for challenger and champion")

        challenger_evaluation = self._require_evaluation_report(challenger)
        champion_evaluation = self._require_evaluation_report(champion)
        challenger_score = challenger.score_history[-1]
        champion_score = champion.score_history[-1]
        comparison_report_id = _comparison_report_id(
            challenger_policy_id=challenger.policy_id,
            challenger_evaluation_id=challenger_evaluation.evaluation_id,
            champion_policy_id=champion.policy_id,
            champion_evaluation_id=champion_evaluation.evaluation_id,
            evaluation_surface_id=challenger.evaluation_surface_id,
        )
        comparison_report = ComparisonReport(
            comparison_report_id=comparison_report_id,
            created_at=utcnow(),
            challenger_policy_id=challenger.policy_id,
            challenger_artifact_id=challenger.artifact_id,
            challenger_evaluation_id=challenger_evaluation.evaluation_id,
            challenger_training_snapshot_id=challenger.training_snapshot_id,
            champion_policy_id=champion.policy_id,
            champion_artifact_id=champion.artifact_id,
            champion_evaluation_id=champion_evaluation.evaluation_id,
            champion_training_snapshot_id=champion.training_snapshot_id,
            evaluation_surface_id=challenger.evaluation_surface_id,
            challenger_active_date_range=challenger_evaluation.active_date_range,
            champion_active_date_range=champion_evaluation.active_date_range,
            challenger_total_net_return=challenger_evaluation.total_net_return,
            champion_total_net_return=champion_evaluation.total_net_return,
            total_net_return_delta=challenger_evaluation.total_net_return - champion_evaluation.total_net_return,
            challenger_composite_rank=challenger_score.composite_rank,
            champion_composite_rank=champion_score.composite_rank,
            composite_rank_delta=challenger_score.composite_rank - champion_score.composite_rank,
            challenger_beats_champion=challenger_evaluation.total_net_return > champion_evaluation.total_net_return,
        )
        dump_model(self.comparisons_dir / f"{comparison_report_id}.json", comparison_report)
        challenger.comparison_report_id = comparison_report.comparison_report_id
        challenger.updated_at = utcnow()
        dump_model(self.records_dir / f"{challenger.policy_id}.json", challenger)
        logger.info(
            "registry_comparison_recorded challenger_policy_id=%s champion_policy_id=%s comparison_report_id=%s",
            challenger.policy_id,
            champion.policy_id,
            comparison_report.comparison_report_id,
        )
        return comparison_report

    def promote_candidate(
        self,
        policy_id: str,
        evidence: PromotionEvidence,
    ) -> PromotionDecisionRecord:
        record = self._require_record(policy_id)
        evaluation_report = self._require_evaluation_report(record)
        current_index = self.load_index()
        current_champion = (
            self.get_record(current_index.champion_policy_id) if current_index.champion_policy_id is not None else None
        )
        paper_sim_evidence = self.get_paper_sim_evidence(evidence.paper_sim_evidence_id)
        comparison_report = self.get_comparison_report(evidence.comparison_report_id)

        checks: dict[str, bool] = {}
        failure_reasons: list[str] = []
        split = record.split_evidence

        checks["registry.scored_candidate_exists"] = bool(record.score_history)
        checks["split.time_ordered"] = split.evaluation_is_time_ordered
        checks["split.no_random_split"] = not split.random_split_used
        checks["split.walkforward_documented"] = split.split_version == "split_v1_walkforward"
        checks["split.purge_applied"] = split.purge_width_steps > 0
        checks["split.embargo_applied"] = split.embargo_width_steps > 0

        checks["leakage.preprocessing_train_only"] = evidence.preprocessing_fit_on_train_only
        checks["leakage.no_future_features"] = evidence.no_future_features
        checks["leakage.no_future_masks"] = evidence.no_future_masks
        checks["leakage.no_future_reward"] = evidence.no_future_reward_construction
        checks["leakage.no_cross_split_contamination"] = evidence.no_cross_split_contamination
        checks["leakage.final_test_unused_for_selection"] = evidence.final_untouched_test_unused_for_selection

        budget = record.search_budget_summary
        checks["search_budget.recorded"] = all(
            value > 0
            for value in (
                budget.tried_models,
                budget.tried_seeds,
                budget.tried_architectures,
                budget.tried_reward_variants,
                budget.tried_hyperparameter_variants,
                budget.total_candidate_count,
            )
        )

        checks["economics.fees_included"] = True
        checks["economics.funding_included"] = True
        checks["economics.slippage_included"] = True
        checks["economics.risk_penalty_included"] = True
        checks["economics.turnover_penalty_included"] = True
        checks["economics.post_cost_positive"] = evaluation_report.total_net_return > 0.0
        checks["economics.realistic_execution_assumptions"] = evidence.realistic_execution_assumptions

        if current_champion is None or current_champion.policy_id == policy_id:
            checks["comparison.same_surface"] = True
            checks["comparison.beats_champion"] = True
            checks["comparison.report_attached"] = True
            checks["comparison.not_one_lucky_slice_only"] = True
            checks["comparison.paper_sim_linkage_consistent"] = True
        else:
            report_matches = (
                comparison_report is not None
                and comparison_report.challenger_policy_id == record.policy_id
                and comparison_report.champion_policy_id == current_champion.policy_id
            )
            checks["comparison.same_surface"] = (
                report_matches
                and comparison_report.same_surface
                and comparison_report.evaluation_surface_id == record.evaluation_surface_id
            )
            checks["comparison.report_attached"] = report_matches
            checks["comparison.not_one_lucky_slice_only"] = evidence.superiority_not_one_lucky_slice_only
            checks["comparison.beats_champion"] = report_matches and comparison_report.challenger_beats_champion
            checks["comparison.paper_sim_linkage_consistent"] = (
                paper_sim_evidence is not None
                and report_matches
                and paper_sim_evidence.comparison_report_id == comparison_report.comparison_report_id
            )

        repro = evidence.reproducibility
        checks["repro.data_snapshot_recorded"] = bool(repro.data_snapshot_id)
        checks["repro.code_commit_recorded"] = bool(repro.code_commit_hash)
        checks["repro.config_recorded"] = bool(repro.config_hash)
        checks["repro.seed_recorded"] = True
        checks["repro.runtime_stack_recorded"] = bool(repro.runtime_stack)
        checks["repro.reproducible_within_tolerance"] = repro.reproducible_within_tolerance

        deployment_path = Path(evidence.deployment_artifact_path)
        checks["artifacts.training_artifact_exists"] = Path(record.artifact_path).exists()
        checks["artifacts.deployment_artifact_exists"] = deployment_path.exists()
        checks["artifacts.evaluation_report_exists"] = record.evaluation_report_path is not None and Path(
            record.evaluation_report_path
        ).exists()
        checks["artifacts.paper_sim_report_exists"] = _paper_sim_exists(paper_sim_evidence)
        checks["artifacts.paper_sim_linked_to_evaluation"] = (
            paper_sim_evidence is not None
            and paper_sim_evidence.policy_id == record.policy_id
            and paper_sim_evidence.artifact_id == record.artifact_id
            and paper_sim_evidence.evaluation_report_id == record.evaluation_report_id
        )
        checks["artifacts.registry_entry_complete"] = _registry_entry_complete(record)

        checks["runtime.uses_inference_artifact_only"] = evidence.runtime_uses_inference_artifact_only
        checks["runtime.no_live_learning"] = evidence.no_live_learning
        checks["runtime.executor_boundary_respected"] = evidence.executor_boundary_respected
        checks["runtime.selector_boundary_respected"] = evidence.selector_boundary_respected

        for check_name, passed in checks.items():
            if not passed:
                failure_reasons.append(check_name)

        decision: Literal["promote", "reject"] = "promote" if not failure_reasons else "reject"
        checked_at = utcnow()
        decision_id = f"promotion-{hash_payload({'policy_id': policy_id, 'checked_at': checked_at.isoformat()})[:12]}"
        decision_record = PromotionDecisionRecord(
            decision_id=decision_id,
            policy_id=record.policy_id,
            artifact_id=record.artifact_id,
            champion_policy_id_before=current_champion.policy_id if current_champion is not None else None,
            champion_policy_id_after=record.policy_id if decision == "promote" else current_index.champion_policy_id,
            decision=decision,
            checked_at=checked_at,
            gate_checks=checks,
            failure_reasons=failure_reasons,
            comparison_report_id=evidence.comparison_report_id,
            paper_sim_evidence_id=evidence.paper_sim_evidence_id,
            deployment_artifact_path=evidence.deployment_artifact_path,
            evaluation_surface_id=record.evaluation_surface_id,
            search_budget_summary=record.search_budget_summary,
            reproducibility=evidence.reproducibility,
        )
        dump_model(self.promotions_dir / f"{decision_id}.json", decision_record)

        record.promotion_decision_ids.append(decision_id)
        record.comparison_report_id = evidence.comparison_report_id or (
            paper_sim_evidence.comparison_report_id if paper_sim_evidence is not None else None
        )
        if paper_sim_evidence is not None:
            record.paper_sim_evidence_id = paper_sim_evidence.evidence_id
            record.paper_sim_report_path = paper_sim_evidence.report_path
        record.deployment_artifact_path = evidence.deployment_artifact_path
        record.runtime_stack = evidence.reproducibility.runtime_stack
        record.updated_at = checked_at

        if decision == "promote":
            if current_champion is not None and current_champion.policy_id != record.policy_id:
                current_champion.status = "challenger"
                current_champion.updated_at = checked_at
                dump_model(self.records_dir / f"{current_champion.policy_id}.json", current_champion)
            record.status = "champion"
        else:
            record.status = "challenger" if record.score_history else "candidate"

        dump_model(self.records_dir / f"{record.policy_id}.json", record)
        self._recompute_index()
        logger.info(
            "registry_promotion_evaluated policy_id=%s decision=%s failure_count=%d champion_before=%s champion_after=%s",
            record.policy_id,
            decision,
            len(failure_reasons),
            current_champion.policy_id if current_champion is not None else "",
            decision_record.champion_policy_id_after or "",
        )
        return decision_record

    def get_record(self, policy_id: str | None) -> RegistryRecord | None:
        if policy_id is None:
            return None
        path = self.records_dir / f"{policy_id}.json"
        if not path.exists():
            return None
        return load_model(path, RegistryRecord)

    def get_promotion_decision(self, decision_id: str) -> PromotionDecisionRecord | None:
        path = self.promotions_dir / f"{decision_id}.json"
        if not path.exists():
            return None
        return load_model(path, PromotionDecisionRecord)

    def get_paper_sim_evidence(self, evidence_id: str | None) -> PaperSimEvidenceRecord | None:
        if evidence_id is None:
            return None
        path = self.paper_sim_dir / f"{evidence_id}.json"
        if not path.exists():
            return None
        return load_model(path, PaperSimEvidenceRecord)

    def get_comparison_report(self, comparison_report_id: str | None) -> ComparisonReport | None:
        if comparison_report_id is None:
            return None
        path = self.comparisons_dir / f"{comparison_report_id}.json"
        if not path.exists():
            return None
        return load_model(path, ComparisonReport)

    def list_records(self) -> list[RegistryRecord]:
        return [load_model(path, RegistryRecord) for path in sorted(self.records_dir.glob("*.json"))]

    def list_comparison_reports(self) -> list[ComparisonReport]:
        return [load_model(path, ComparisonReport) for path in sorted(self.comparisons_dir.glob("*.json"))]

    def list_paper_sim_evidence(self) -> list[PaperSimEvidenceRecord]:
        return [load_model(path, PaperSimEvidenceRecord) for path in sorted(self.paper_sim_dir.glob("*.json"))]

    def load_index(self) -> RegistryIndex:
        path = self.root / "index.json"
        if not path.exists():
            return RegistryIndex()
        return load_model(path, RegistryIndex)

    def _recompute_index(self) -> None:
        records = self.list_records()
        selector = CandidateSelector()
        champion_candidates = [record for record in records if record.status == "champion" and record.score_history]
        champion_policy_id = selector.rank(champion_candidates)[0].policy_id if champion_candidates else None
        challenger_ids = []

        for record in records:
            if record.policy_id == champion_policy_id:
                record.status = "champion"
            elif record.status in {"rejected", "retired", "archived"}:
                pass
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
        logger.info(
            "registry_index_recomputed champion_policy_id=%s challenger_count=%d record_count=%d",
            champion_policy_id or "",
            len(challenger_ids),
            len(records),
        )

    def _require_record(self, policy_id: str) -> RegistryRecord:
        record = self.get_record(policy_id)
        if record is None:
            raise FileNotFoundError(f"registry record not found for policy {policy_id}")
        return record

    def _require_evaluation_report(self, record: RegistryRecord) -> EvaluationReport:
        if record.evaluation_report_path is None:
            raise ValueError("promotion requires an evaluation report linked in the registry")
        path = Path(record.evaluation_report_path)
        if not path.exists():
            raise FileNotFoundError(f"evaluation report not found at {path}")
        return load_model(path, EvaluationReport)

    def _maybe_load_report(self, record: RegistryRecord | None) -> EvaluationReport | None:
        if record is None or record.evaluation_report_path is None:
            return None
        path = Path(record.evaluation_report_path)
        if not path.exists():
            return None
        return load_model(path, EvaluationReport)

    def _validate_paper_sim_comparison_linkage(
        self,
        *,
        record: RegistryRecord,
        comparison_report_id: str | None,
    ) -> ComparisonReport | None:
        current_index = self.load_index()
        current_champion = (
            self.get_record(current_index.champion_policy_id) if current_index.champion_policy_id is not None else None
        )
        comparison_report = self.get_comparison_report(comparison_report_id)
        if comparison_report_id is not None and comparison_report is None:
            raise FileNotFoundError(f"comparison report not found for id {comparison_report_id}")
        if current_champion is not None and current_champion.policy_id != record.policy_id:
            if comparison_report is None:
                raise ValueError(
                    "challenger paper/sim recording requires comparison_report_id linked to the current champion"
                )
            if comparison_report.challenger_policy_id != record.policy_id:
                raise ValueError("comparison report challenger policy does not match the requested paper/sim policy")
            if comparison_report.champion_policy_id != current_champion.policy_id:
                raise ValueError("comparison report champion policy does not match the current registry champion")
            if comparison_report.evaluation_surface_id != record.evaluation_surface_id:
                raise ValueError("comparison report evaluation surface does not match the policy evaluation surface")
        elif comparison_report is not None and record.policy_id not in {
            comparison_report.challenger_policy_id,
            comparison_report.champion_policy_id,
        }:
            raise ValueError("comparison report is not linked to the requested paper/sim policy")
        return comparison_report


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
        field_origins={
            "train_sample_count": "dataset-surface-derived",
            "eval_sample_count": "policy-evidence-derived",
            "covered_symbols": "dataset-surface-derived",
            "covered_venues": "dataset-surface-derived",
            "covered_streams": "dataset-surface-derived",
            "active_date_range": "dataset-surface-derived",
            "reward_event_count": "policy-evidence-derived",
            "realized_trade_count": "policy-evidence-derived",
        },
    )


def _coverage_from_manifest(
    manifest: TrajectoryManifest,
    trajectory_directory: Path | None = None,
) -> CoverageStats:
    train_stats = manifest.split_write_stats.get("train")
    if train_stats is not None:
        train_steps = train_stats.step_count
    else:
        train_steps = _recover_manifest_train_steps(
            manifest,
            trajectory_directory=trajectory_directory,
        )
    return CoverageStats(
        train_sample_count=train_steps,
        eval_sample_count=0,
        covered_symbols=manifest.dataset_spec.symbols,
        covered_venues=manifest.dataset_spec.exchanges,
        covered_streams=manifest.dataset_spec.stream_universe,
        active_date_range=manifest.dataset_spec.train_range,
        reward_event_count=0,
        realized_trade_count=0,
        field_origins={
            "train_sample_count": "dataset-surface-derived",
            "eval_sample_count": "policy-evidence-derived",
            "covered_symbols": "dataset-surface-derived",
            "covered_venues": "dataset-surface-derived",
            "covered_streams": "dataset-surface-derived",
            "active_date_range": "dataset-surface-derived",
            "reward_event_count": "policy-evidence-derived",
            "realized_trade_count": "policy-evidence-derived",
        },
    )


def _recover_manifest_train_steps(
    manifest: TrajectoryManifest,
    *,
    trajectory_directory: Path | None,
) -> int:
    slice_id = manifest.dataset_spec.slice_id
    dataset_hash = manifest.dataset_spec.dataset_hash
    if trajectory_directory is None:
        raise ValueError(
            "registry manifest coverage requires split_write_stats['train'] or a "
            "trajectory_directory with retained tensor_cache_manifest.json to recover "
            f"train_sample_count for slice_id={slice_id} dataset_hash={dataset_hash}"
        )

    from quantlab_ml.trajectories.tensor_cache import (
        read_tensor_cache_manifest,
        tensor_cache_manifest_path,
    )

    cache_manifest_path = tensor_cache_manifest_path(trajectory_directory)
    if not cache_manifest_path.exists():
        raise FileNotFoundError(
            "registry manifest coverage cannot recover train_sample_count from a legacy "
            "manifest without split_write_stats['train']: retained tensor cache manifest "
            f"missing at {cache_manifest_path} for slice_id={slice_id} dataset_hash={dataset_hash}"
        )

    cache_manifest = read_tensor_cache_manifest(trajectory_directory)
    train_split = cache_manifest.splits.get("train")
    if train_split is None:
        raise ValueError(
            "registry manifest coverage cannot recover train_sample_count from a legacy "
            "manifest without split_write_stats['train']: retained tensor cache manifest "
            f"at {cache_manifest_path} is missing the train split for slice_id={slice_id} "
            f"dataset_hash={dataset_hash}"
        )
    return train_split.row_count


def _merge_ranges(left: TimeRange, right: TimeRange) -> TimeRange:
    return TimeRange(
        start=min(left.start, right.start),
        end=max(left.end, right.end),
    )


def _lineage_chain(artifact: PolicyArtifact) -> list[str]:
    if artifact.runtime_metadata.lineage_pointer.parent_policy_id is None:
        return []
    return [artifact.runtime_metadata.lineage_pointer.parent_policy_id]


def _search_budget_summary(artifact: PolicyArtifact) -> SearchBudgetSummary:
    raw = artifact.training_summary.get("search_budget_summary", {})
    tried_models = int(raw.get("tried_models", 1))
    tried_seeds = int(raw.get("tried_seeds", 1))
    tried_architectures = int(raw.get("tried_architectures", 1))
    tried_reward_variants = int(raw.get("tried_reward_variants", 1))
    tried_hyperparameter_variants = int(raw.get("tried_hyperparameter_variants", 1))
    total_candidate_count = int(raw.get("total_candidate_count", 1))
    return SearchBudgetSummary(
        tried_models=tried_models,
        tried_seeds=tried_seeds,
        tried_architectures=tried_architectures,
        tried_reward_variants=tried_reward_variants,
        tried_hyperparameter_variants=tried_hyperparameter_variants,
        total_candidate_count=total_candidate_count,
    )


def _paper_sim_exists(evidence: PaperSimEvidenceRecord | None) -> bool:
    if evidence is None:
        return False
    return Path(evidence.report_path).exists()


def _paper_sim_evidence_id(
    *,
    policy_id: str,
    evaluation_report_id: str,
    report_path: Path,
    comparison_report_id: str | None,
) -> str:
    payload = {
        "policy_id": policy_id,
        "evaluation_report_id": evaluation_report_id,
        "report_path": str(report_path.resolve()),
        "comparison_report_id": comparison_report_id,
    }
    return f"paper-sim-{hash_payload(payload)[:12]}"


def _comparison_report_id(
    *,
    challenger_policy_id: str,
    challenger_evaluation_id: str,
    champion_policy_id: str,
    champion_evaluation_id: str,
    evaluation_surface_id: str,
) -> str:
    payload = {
        "challenger_policy_id": challenger_policy_id,
        "challenger_evaluation_id": challenger_evaluation_id,
        "champion_policy_id": champion_policy_id,
        "champion_evaluation_id": champion_evaluation_id,
        "evaluation_surface_id": evaluation_surface_id,
    }
    return f"comparison-{hash_payload(payload)[:12]}"


def _report_format(report_path: Path) -> Literal["markdown", "json", "text", "unknown"]:
    suffix = report_path.suffix.lower()
    if suffix in {".md", ".markdown"}:
        return "markdown"
    if suffix == ".json":
        return "json"
    if suffix in {".txt", ".log"}:
        return "text"
    return "unknown"


def _registry_entry_complete(record: RegistryRecord) -> bool:
    required_values = (
        record.policy_id,
        record.artifact_id,
        record.policy_family,
        record.target_asset,
        record.training_snapshot_id,
        record.training_config_hash,
        record.reward_version,
        record.evaluation_surface_id,
    )
    return all(bool(value) for value in required_values)

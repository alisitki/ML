from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import os
from pathlib import Path
from typing import Any, Literal
import warnings

import numpy as np

from quantlab_ml.common import current_code_commit_hash, hash_payload, utcnow
from quantlab_ml.contracts import (
    ACTION_SPACE_VERSION,
    DYNAMIC_TARGET_ASSET,
    OBSERVATION_SCHEMA_VERSION,
    EvaluationBoundary,
    EvaluationReport,
    LineagePointer,
    NumericBand,
    OpaquePolicyPayload,
    PolicyArtifact,
    POLICY_ARTIFACT_SCHEMA_VERSION,
    PolicyScore,
    RuntimeMetadata,
    SearchBudgetSummary,
    TrajectoryBundle,
    TrajectoryManifest,
    TrajectoryRecord,
    TrajectoryStep,
    WalkForwardFold,
)
from quantlab_ml.models.features import observation_feature_vector
from quantlab_ml.models.linear_policy import LinearPolicyParameters
from quantlab_ml.runtime_contract import build_strict_runtime_contract
from quantlab_ml.scoring import PolicyScorer
from quantlab_ml.training.config import TrainingConfig
from . import compat_matrix_first
from .compat_matrix_first import CompatPreparedTrainingData as _PreparedTrainingData

logger = logging.getLogger(__name__)

TrainingBackendName = Literal["numpy", "pytorch"]

_STREAMING_BATCH_TARGET_BYTES = 128 * 1024 * 1024
_STREAMING_BATCH_MAX_SIZE = 4096
_STREAMING_BATCH_LABEL_OVERHEAD_BYTES = (
    np.dtype(np.int64).itemsize * 2 + np.dtype(np.bool_).itemsize
)


@dataclass(frozen=True, slots=True)
class _DeviceResolution:
    training_device: str
    cuda_available: bool
    device_name: str
    compute_device: Any | None


@dataclass(frozen=True, slots=True)
class TrainingCandidateSpec:
    seed: int
    learning_rate: float
    l2_weight: float

    def as_dict(self) -> dict[str, int | float]:
        return {
            "seed": self.seed,
            "learning_rate": self.learning_rate,
            "l2_weight": self.l2_weight,
        }


@dataclass(slots=True)
class TrainingCandidateResult:
    artifact: PolicyArtifact
    candidate_index: int
    candidate_rank: int
    selected_candidate: bool
    candidate_spec: TrainingCandidateSpec
    best_validation_total_net_return: float
    best_validation_composite_rank: float


@dataclass(slots=True)
class TrainingSearchResult:
    training_run_id: str
    selected_artifact: PolicyArtifact
    candidate_results: list[TrainingCandidateResult]
    search_budget_summary: SearchBudgetSummary


@dataclass(frozen=True, slots=True)
class FoldValidationScore:
    fold_id: str
    validation_total_net_return: float
    validation_composite_rank: float
    validation_step_count: int

    def as_dict(self) -> dict[str, str | float | int]:
        return {
            "fold_id": self.fold_id,
            "validation_total_net_return": self.validation_total_net_return,
            "validation_composite_rank": self.validation_composite_rank,
            "validation_step_count": self.validation_step_count,
        }


@dataclass(slots=True)
class _CandidateSelectionRun:
    candidate_spec: TrainingCandidateSpec
    candidate_index: int
    fold_scores: list[FoldValidationScore]
    selection_total_net_return: float
    selection_composite_rank: float


@dataclass(slots=True)
class StreamingFeatureStats:
    count: int = 0
    mean: np.ndarray | None = None
    m2: np.ndarray | None = None

    def update(self, features: np.ndarray) -> None:
        vector = np.asarray(features, dtype=np.float64)
        if self.mean is None or self.m2 is None:
            self.mean = np.zeros_like(vector, dtype=np.float64)
            self.m2 = np.zeros_like(vector, dtype=np.float64)
        self.count += 1
        delta = vector - self.mean
        self.mean = self.mean + (delta / self.count)
        delta2 = vector - self.mean
        self.m2 = self.m2 + (delta * delta2)

    @property
    def feature_dim(self) -> int:
        if self.mean is None:
            return 0
        return int(self.mean.shape[0])

    def finalize(self) -> tuple[np.ndarray, np.ndarray]:
        if self.count <= 0 or self.mean is None or self.m2 is None:
            raise ValueError("streaming feature stats require at least one training example")
        feature_mean = self.mean.astype(np.float32)
        feature_std = np.sqrt(self.m2 / max(self.count, 1))
        feature_std = np.where(feature_std < 1e-6, 1.0, feature_std).astype(np.float32)
        return feature_mean, feature_std


@dataclass(frozen=True, slots=True)
class StreamingBatchPlan:
    batch_target_bytes: int
    bytes_per_example: int
    effective_batch_size: int
    estimated_batch_bytes: int
    batches_per_epoch: int


@dataclass(frozen=True, slots=True)
class StreamingWindow:
    split_name: str
    start: datetime | None = None
    end: datetime | None = None
    exclusive_end: datetime | None = None

    def includes(self, event_time: datetime) -> bool:
        if self.start is not None and event_time < self.start:
            return False
        if self.end is not None and event_time > self.end:
            return False
        if self.exclusive_end is not None and event_time >= self.exclusive_end:
            return False
        return True


@dataclass(slots=True)
class StreamingEpochResult:
    epoch: int
    total_loss: float
    validation_report: EvaluationReport
    validation_score: PolicyScore
    is_best: bool


@dataclass(slots=True)
class _StreamingPreparedData:
    train_step_count: int
    val_step_count: int
    action_keys: list[str]
    venue_choices: list[str]
    feature_mean: np.ndarray
    feature_std: np.ndarray
    batch_plan: StreamingBatchPlan

    @property
    def feature_dim(self) -> int:
        return int(self.feature_mean.shape[0])


class LinearPolicyTrainer:
    def __init__(self, config: TrainingConfig, *, backend_name: TrainingBackendName = "pytorch"):
        self.config = config
        self._backend = _resolve_training_backend(backend_name)

    def train(self, bundle: TrajectoryBundle, parent_policy_id: str | None = None) -> PolicyArtifact:
        """⚠️  FIXTURE / TEST COMPAT PATH — do not call from production code.

        Loads entire bundle.  Use train_search_from_directory() for production.
        """
        return self.train_search(bundle, parent_policy_id=parent_policy_id).selected_artifact

    def train_search(self, bundle: TrajectoryBundle, parent_policy_id: str | None = None) -> TrainingSearchResult:
        """⚠️  FIXTURE / TEST COMPAT PATH — do not call from production code.

        Loads entire bundle.  Use train_search_from_directory() for production.
        """
        candidate_specs = self._candidate_specs()
        code_commit_hash = current_code_commit_hash()
        training_run_id = self._training_run_id(
            bundle,
            parent_policy_id=parent_policy_id,
            code_commit_hash=code_commit_hash,
            candidate_specs=candidate_specs,
        )
        search_budget_summary = _search_budget_summary(candidate_specs)
        logger.info(
            "training_search_started training_run_id=%s candidate_count=%d split_version=%s reward_version=%s "
            "training_backend=%s training_device=%s cuda_available=%s device_name=%s",
            training_run_id,
            len(candidate_specs),
            bundle.split_artifact.split_version,
            bundle.reward_spec.reward_version,
            self._backend.backend_name,
            self._backend.device_resolution.training_device,
            self._backend.device_resolution.cuda_available,
            self._backend.device_resolution.device_name,
        )
        selection_runs = [
            self._select_candidate_via_walkforward(
                bundle=bundle,
                candidate_spec=candidate_spec,
                candidate_index=candidate_index,
                parent_policy_id=parent_policy_id,
                training_run_id=training_run_id,
                code_commit_hash=code_commit_hash,
            )
            for candidate_index, candidate_spec in enumerate(candidate_specs)
        ]
        ranked_selections = sorted(selection_runs, key=_selection_ranking_key)
        prepared = self._prepare_training_data(bundle)

        candidate_results: list[TrainingCandidateResult] = []
        for candidate_rank, selection_run in enumerate(ranked_selections, start=1):
            selected_candidate = candidate_rank == 1
            candidate_run = self._train_candidate(
                bundle=bundle,
                prepared=prepared,
                candidate_spec=selection_run.candidate_spec,
                candidate_index=selection_run.candidate_index,
                parent_policy_id=parent_policy_id,
                training_run_id=training_run_id,
                code_commit_hash=code_commit_hash,
            )
            candidate_summary = self._candidate_training_summary(
                prepared=prepared,
                candidate_run=candidate_run,
                selection_run=selection_run,
                training_run_id=training_run_id,
                candidate_rank=candidate_rank,
                selected_candidate=selected_candidate,
                search_budget_summary=search_budget_summary,
                training_data_flow="matrix_first_compat",
                validation_data_flow="bundle_evaluation",
                normalization_strategy="matrix_first_train_only",
                proxy_validation_used=False,
            )
            artifact = self._build_artifact(
                bundle=bundle,
                config=candidate_run.config,
                training_run_id=training_run_id,
                code_commit_hash=code_commit_hash,
                parameters=candidate_run.best_parameters,
                parent_policy_id=parent_policy_id,
                validation_total_net_return=candidate_run.best_validation_total_net_return,
                validation_score=candidate_run.best_validation_score,
                training_summary=candidate_summary,
                search_metadata=_ArtifactSearchMetadata(
                    candidate_index=selection_run.candidate_index,
                    candidate_rank=candidate_rank,
                    selected_candidate=selected_candidate,
                ),
            )
            candidate_results.append(
                TrainingCandidateResult(
                    artifact=artifact,
                    candidate_index=selection_run.candidate_index,
                    candidate_rank=candidate_rank,
                    selected_candidate=selected_candidate,
                    candidate_spec=selection_run.candidate_spec,
                    best_validation_total_net_return=candidate_run.best_validation_total_net_return,
                    best_validation_composite_rank=candidate_run.best_validation_score.composite_rank,
                )
            )

        result = TrainingSearchResult(
            training_run_id=training_run_id,
            selected_artifact=candidate_results[0].artifact,
            candidate_results=candidate_results,
            search_budget_summary=search_budget_summary,
        )
        logger.info(
            "training_search_completed training_run_id=%s selected_policy_id=%s candidate_count=%d "
            "selected_validation_total_net_return=%.6f selected_validation_composite_rank=%.6f",
            training_run_id,
            result.selected_artifact.policy_id,
            len(candidate_results),
            candidate_results[0].best_validation_total_net_return,
            candidate_results[0].best_validation_composite_rank,
        )
        return result

    def _candidate_specs(self) -> list[TrainingCandidateSpec]:
        search = self.config.candidate_search
        seeds = search.seeds if search is not None and search.seeds else [self.config.seed]
        learning_rates = search.learning_rates if search is not None and search.learning_rates else [self.config.learning_rate]
        l2_weights = search.l2_weights if search is not None and search.l2_weights else [self.config.l2_weight]

        seen: set[tuple[int, float, float]] = set()
        candidate_specs: list[TrainingCandidateSpec] = []
        for seed in seeds:
            for learning_rate in learning_rates:
                for l2_weight in l2_weights:
                    key = (seed, learning_rate, l2_weight)
                    if key in seen:
                        continue
                    seen.add(key)
                    candidate_specs.append(
                        TrainingCandidateSpec(
                            seed=seed,
                            learning_rate=learning_rate,
                            l2_weight=l2_weight,
                        )
                    )
        return candidate_specs

    def _training_run_id(
        self,
        bundle: TrajectoryBundle,
        *,
        parent_policy_id: str | None,
        code_commit_hash: str,
        candidate_specs: list[TrainingCandidateSpec],
    ) -> str:
        run_payload = {
            "dataset_hash": bundle.dataset_spec.dataset_hash,
            "slice_id": bundle.dataset_spec.slice_id,
            "split_version": bundle.split_artifact.split_version,
            "reward_version": bundle.reward_spec.reward_version,
            "training_backend": self._backend.backend_name,
            "trainer_config": self.config.model_dump(mode="json", exclude_none=False),
            "candidate_specs": [candidate_spec.as_dict() for candidate_spec in candidate_specs],
            "parent_policy_id": parent_policy_id,
            "code_commit_hash": code_commit_hash,
        }
        return f"trainrun-{hash_payload(run_payload)[:12]}"

    def _prepare_training_data(self, bundle: TrajectoryBundle) -> _PreparedTrainingData:
        """⚠️  FIXTURE / TEST COMPAT PATH — matrix-first helper wrapper."""

        return compat_matrix_first.prepare_training_data(bundle)

    def _select_candidate_via_walkforward(
        self,
        *,
        bundle: TrajectoryBundle,
        candidate_spec: TrainingCandidateSpec,
        candidate_index: int,
        parent_policy_id: str | None,
        training_run_id: str,
        code_commit_hash: str,
    ) -> _CandidateSelectionRun:
        fold_scores: list[FoldValidationScore] = []

        for fold in bundle.split_artifact.folds:
            fold_bundle = self._build_fold_bundle(bundle, fold)
            prepared = self._prepare_training_data(fold_bundle)
            fold_run = self._train_candidate(
                bundle=fold_bundle,
                prepared=prepared,
                candidate_spec=candidate_spec,
                candidate_index=candidate_index,
                parent_policy_id=parent_policy_id,
                training_run_id=f"{training_run_id}:{fold.fold_id}",
                code_commit_hash=code_commit_hash,
            )
            fold_step_count = prepared.val_step_count
            fold_scores.append(
                FoldValidationScore(
                    fold_id=fold.fold_id,
                    validation_total_net_return=fold_run.best_validation_total_net_return,
                    validation_composite_rank=fold_run.best_validation_score.composite_rank,
                    validation_step_count=fold_step_count,
                )
            )

        selection_total_net_return = _weighted_mean(
            [score.validation_total_net_return for score in fold_scores],
            [score.validation_step_count for score in fold_scores],
        )
        selection_composite_rank = _weighted_mean(
            [score.validation_composite_rank for score in fold_scores],
            [score.validation_step_count for score in fold_scores],
        )
        logger.info(
            "training_candidate_walkforward_completed candidate_index=%d seed=%d learning_rate=%.6f "
            "l2_weight=%.6f fold_count=%d selection_total_net_return=%.6f selection_composite_rank=%.6f",
            candidate_index,
            candidate_spec.seed,
            candidate_spec.learning_rate,
            candidate_spec.l2_weight,
            len(fold_scores),
            selection_total_net_return,
            selection_composite_rank,
        )
        return _CandidateSelectionRun(
            candidate_spec=candidate_spec,
            candidate_index=candidate_index,
            fold_scores=fold_scores,
            selection_total_net_return=selection_total_net_return,
            selection_composite_rank=selection_composite_rank,
        )

    def _build_fold_bundle(self, bundle: TrajectoryBundle, fold: WalkForwardFold) -> TrajectoryBundle:
        interval = timedelta(seconds=bundle.dataset_spec.sampling_interval_seconds)
        purge_cutoff = fold.validation_window.start - (interval * fold.purge_width_steps)
        train_records = self._slice_records(
            bundle.development_records,
            split_name="train",
            start=fold.train_window.start,
            end=fold.train_window.end,
            exclusive_end=purge_cutoff if fold.purge_width_steps > 0 else None,
        )
        validation_records = self._slice_records(
            bundle.development_records,
            split_name="validation",
            start=fold.validation_window.start,
            end=fold.validation_window.end,
        )
        return bundle.model_copy(
            deep=True,
            update={
                "splits": {
                    "train": train_records,
                    "validation": validation_records,
                    "final_untouched_test": [],
                }
            },
        )

    def _slice_records(
        self,
        records: list[TrajectoryRecord],
        *,
        split_name: Literal["train", "validation"],
        start: datetime,
        end: datetime,
        exclusive_end: datetime | None = None,
    ) -> list[TrajectoryRecord]:
        sliced: list[TrajectoryRecord] = []
        for record in records:
            selected_steps = [
                step.model_copy(deep=True)
                for step in record.steps
                if start <= step.event_time <= end and (exclusive_end is None or step.event_time < exclusive_end)
            ]
            if not selected_steps:
                continue
            sliced.append(
                TrajectoryRecord(
                    trajectory_id=f"{split_name}-{record.trajectory_id}",
                    split=split_name,
                    target_symbol=record.target_symbol,
                    start_time=selected_steps[0].event_time,
                    end_time=selected_steps[-1].event_time,
                    steps=selected_steps,
                    terminal=True,
                    terminal_reason=record.terminal_reason,
                )
            )
        return sliced

    def _train_candidate(
        self,
        *,
        bundle: TrajectoryBundle,
        prepared: _PreparedTrainingData,
        candidate_spec: TrainingCandidateSpec,
        candidate_index: int,
        parent_policy_id: str | None,
        training_run_id: str,
        code_commit_hash: str,
    ) -> _CandidateTrainingRun:
        config = self.config.model_copy(
            update={
                "seed": candidate_spec.seed,
                "learning_rate": candidate_spec.learning_rate,
                "l2_weight": candidate_spec.l2_weight,
                "candidate_search": None,
            }
        )
        state = self._backend.initialize_state(
            seed=config.seed,
            action_count=len(prepared.action_keys),
            venue_count=len(prepared.venue_choices),
            feature_dim=prepared.feature_dim,
        )

        loss_history: list[float] = []
        validation_history: list[float] = []
        best_validation_total_net_return: float | None = None
        best_epoch = 0
        best_parameters: LinearPolicyParameters | None = None
        best_validation_score = None

        for epoch in range(1, config.epochs + 1):
            total_loss = self._backend.step(
                state=state,
                prepared=prepared,
                config=config,
            )
            loss_history.append(total_loss)

            parameters = self._backend.parameters(
                state=state,
                action_keys=prepared.action_keys,
                venue_choices=prepared.venue_choices,
                feature_mean=prepared.feature_mean,
                feature_std=prepared.feature_std,
                config=config,
            )
            validation_artifact = self._build_artifact(
                bundle=bundle,
                config=config,
                training_run_id=training_run_id,
                code_commit_hash=code_commit_hash,
                parameters=parameters,
                parent_policy_id=parent_policy_id,
                validation_total_net_return=0.0,
                validation_score=None,
                training_summary={},
                search_metadata=None,
            )
            validation_report = self._validation_report(bundle, validation_artifact)
            validation_history.append(validation_report.total_net_return)
            validation_score = PolicyScorer().score(validation_report)

            if (
                best_validation_total_net_return is None
                or validation_report.total_net_return > best_validation_total_net_return
            ):
                best_validation_total_net_return = validation_report.total_net_return
                best_epoch = epoch
                best_parameters = parameters
                best_validation_score = validation_score

        assert best_parameters is not None
        assert best_validation_total_net_return is not None
        assert best_validation_score is not None
        logger.info(
            "training_candidate_completed candidate_index=%d seed=%d learning_rate=%.6f l2_weight=%.6f "
            "best_epoch=%d best_validation_total_net_return=%.6f best_validation_composite_rank=%.6f "
            "training_backend=%s",
            candidate_index,
            candidate_spec.seed,
            candidate_spec.learning_rate,
            candidate_spec.l2_weight,
            best_epoch,
            best_validation_total_net_return,
            best_validation_score.composite_rank,
            self._backend.backend_name,
        )

        return _CandidateTrainingRun(
            config=config,
            candidate_spec=candidate_spec,
            candidate_index=candidate_index,
            best_epoch=best_epoch,
            best_parameters=best_parameters,
            best_validation_total_net_return=best_validation_total_net_return,
            best_validation_score=best_validation_score,
            loss_history=loss_history,
            validation_history=validation_history,
        )

    def _candidate_training_summary(
        self,
        *,
        prepared: _PreparedTrainingData | _StreamingPreparedData,
        candidate_run: _CandidateTrainingRun,
        selection_run: _CandidateSelectionRun,
        training_run_id: str,
        candidate_rank: int,
        selected_candidate: bool,
        search_budget_summary: SearchBudgetSummary,
        training_data_flow: str,
        validation_data_flow: str,
        normalization_strategy: str,
        proxy_validation_used: bool,
        batch_plan: StreamingBatchPlan | None = None,
    ) -> dict[str, object]:
        summary: dict[str, object] = {
            "trainer_name": candidate_run.config.trainer_name,
            "surface_version": "v2",
            "training_backend": self._backend.backend_name,
            "training_device": self._backend.device_resolution.training_device,
            "cuda_available": self._backend.device_resolution.cuda_available,
            "device_name": self._backend.device_resolution.device_name,
            "training_run_id": training_run_id,
            "train_step_count": prepared.train_step_count,
            "validation_step_count": prepared.val_step_count,
            "feature_dim": prepared.feature_dim,
            "epochs": candidate_run.config.epochs,
            "seed": candidate_run.config.seed,
            "learning_rate": candidate_run.config.learning_rate,
            "l2_weight": candidate_run.config.l2_weight,
            "candidate_index": candidate_run.candidate_index,
            "candidate_rank": candidate_rank,
            "selected_candidate": selected_candidate,
            "candidate_spec": candidate_run.candidate_spec.as_dict(),
            "selection_protocol": "walkforward_cv_then_canonical_refit",
            "selection_fold_count": len(selection_run.fold_scores),
            "selection_aggregate_metric": "step_weighted_mean_validation_total_net_return",
            "selection_aggregate_total_net_return": selection_run.selection_total_net_return,
            "selection_aggregate_composite_rank": selection_run.selection_composite_rank,
            "candidate_fold_scores": [score.as_dict() for score in selection_run.fold_scores],
            "best_epoch": candidate_run.best_epoch,
            "best_validation_total_net_return": candidate_run.best_validation_total_net_return,
            "best_validation_composite_rank": candidate_run.best_validation_score.composite_rank,
            "selection_split": "validation",
            "selection_metric": "total_net_return",
            "final_untouched_test_used": False,
            "learned_normalization_fit_split": "train",
            "training_data_flow": training_data_flow,
            "validation_data_flow": validation_data_flow,
            "normalization_strategy": normalization_strategy,
            "proxy_validation_used": proxy_validation_used,
            "training_loss_history": candidate_run.loss_history,
            "validation_objective_history": candidate_run.validation_history,
            "search_budget_summary": search_budget_summary.model_dump(mode="json"),
        }
        if batch_plan is not None:
            summary.update(
                {
                    "effective_batch_size": batch_plan.effective_batch_size,
                    "estimated_batch_bytes": batch_plan.estimated_batch_bytes,
                    "batches_per_epoch": batch_plan.batches_per_epoch,
                    "batch_target_bytes": batch_plan.batch_target_bytes,
                }
            )
        else:
            summary.update(
                {
                    "effective_batch_size": None,
                    "estimated_batch_bytes": None,
                    "batches_per_epoch": None,
                    "batch_target_bytes": None,
                }
            )
        return summary

    def _validation_report(self, bundle: TrajectoryBundle, artifact: PolicyArtifact) -> EvaluationReport:
        from quantlab_ml.evaluation import EvaluationEngine

        return EvaluationEngine(self._evaluation_boundary(bundle.reward_spec.timestamping)).evaluate(
            bundle,
            artifact,
            split="validation",
        )

    def _evaluation_boundary(self, timestamping: str) -> EvaluationBoundary:
        return EvaluationBoundary(
            fee_handling="shared_reward_contract",
            funding_handling="carry_from_funding_stream",
            slippage_handling="fixed_bps",
            fill_assumption_mode=timestamping,
            timeout_semantics="force_terminal_at_data_end",
            terminal_semantics="trajectory_boundary_is_terminal",
            infeasible_action_treatment="force_abstain",
        )

    def _build_artifact(
        self,
        *,
        bundle: TrajectoryBundle,
        config: TrainingConfig,
        training_run_id: str,
        code_commit_hash: str,
        parameters: LinearPolicyParameters,
        parent_policy_id: str | None,
        validation_total_net_return: float,
        validation_score: PolicyScore | None,
        training_summary: dict[str, object],
        search_metadata: _ArtifactSearchMetadata | None,
    ) -> PolicyArtifact:
        payload_blob = parameters.model_dump_json()
        payload = OpaquePolicyPayload(
            runtime_adapter=config.runtime_adapter,
            payload_format="json",
            payload_format_version="json-v1",
            blob=payload_blob,
            digest=hash_payload(parameters),
        )
        lineages = LineagePointer(
            parent_policy_id=parent_policy_id,
            generation=0 if parent_policy_id is None else 1,
            notes=["v2 surface - real supervised linear policy trainer"],
        )
        training_config_hash = hash_payload(config)
        training_snapshot_id = f"{bundle.dataset_spec.dataset_hash}:{bundle.dataset_spec.slice_id}"
        artifact_identity = hash_payload(
            {
                "payload_digest": payload.digest,
                "training_config_hash": training_config_hash,
                "training_snapshot_id": training_snapshot_id,
                "training_run_id": training_run_id,
            }
        )
        policy_id = f"policy-{artifact_identity[:12]}"
        artifact_id = f"artifact-{artifact_identity[:12]}"
        evaluation_surface_id = (
            f"{bundle.dataset_spec.slice_id}:{bundle.split_artifact.split_version}:{bundle.reward_spec.reward_version}"
        )
        target_asset = bundle.dataset_spec.symbols[0] if len(bundle.dataset_spec.symbols) == 1 else DYNAMIC_TARGET_ASSET
        required_context: dict[str, object] = {}
        if target_asset == DYNAMIC_TARGET_ASSET:
            required_context = {"target_symbol_source": "observation.target_symbol"}

        expected_return_score = validation_total_net_return / max(
            sum(len(item.steps) for item in bundle.splits["validation"]),
            1,
        )
        risk_score = best_effort_metric(validation_score, "risk_score")
        turnover_score = best_effort_metric(validation_score, "turnover_score")
        confidence_or_quality_score = min(0.99, max(best_effort_metric(validation_score, "composite_rank"), 0.0))

        size_band = _band_by_key(bundle.action_space.size_bands, config.preferred_size_band)
        leverage_band = _band_by_key(bundle.action_space.leverage_bands, config.preferred_leverage_band)
        strict_runtime_contract = build_strict_runtime_contract(
            bundle.observation_schema,
            policy_kind=config.runtime_adapter,
        )
        artifact_tags = [
            f"runtime_adapter:{config.runtime_adapter}",
            f"reward:{bundle.reward_spec.reward_version}",
            f"split:{bundle.split_artifact.split_version}",
            f"observation:{OBSERVATION_SCHEMA_VERSION}",
            f"action_space:{ACTION_SPACE_VERSION}",
            f"runtime_contract:{strict_runtime_contract.runtime_contract_version}",
            f"policy_kind:{strict_runtime_contract.policy_kind}",
            f"derived_contract:{strict_runtime_contract.derived_contract_version}",
            f"derived_signature:{strict_runtime_contract.derived_channel_template_signature}",
            f"feature_dim:{strict_runtime_contract.expected_feature_dim}",
            "compat_mode:strict",
        ]
        if search_metadata is not None:
            artifact_tags.extend(
                [
                    f"search_run_id:{training_run_id}",
                    f"search_candidate_index:{search_metadata.candidate_index}",
                    f"search_candidate_rank:{search_metadata.candidate_rank}",
                    f"search_selected:{str(search_metadata.selected_candidate).lower()}",
                ]
            )

        return PolicyArtifact(
            artifact_id=artifact_id,
            artifact_version=POLICY_ARTIFACT_SCHEMA_VERSION,
            policy_id=policy_id,
            policy_family=config.trainer_name,
            training_snapshot_id=training_snapshot_id,
            training_config_hash=training_config_hash,
            code_commit_hash=code_commit_hash,
            reward_version=bundle.reward_spec.reward_version,
            evaluation_surface_id=evaluation_surface_id,
            target_asset=target_asset,
            allowed_venues=bundle.dataset_spec.exchanges,
            allowed_action_family=bundle.action_space.action_keys,
            required_context=required_context,
            created_at=utcnow(),
            observation_schema=bundle.observation_schema,
            action_space=bundle.action_space,
            policy_payload=payload,
            runtime_metadata=RuntimeMetadata(
                target_asset=target_asset,
                allowed_venues=bundle.dataset_spec.exchanges,
                action_space_version=ACTION_SPACE_VERSION,
                required_streams=bundle.dataset_spec.stream_universe,
                required_field_families={
                    stream: bundle.observation_schema.field_axis.get(stream, [])
                    for stream in bundle.dataset_spec.stream_universe
                },
                required_scale_preset=[scale.label for scale in bundle.trajectory_spec.scale_preset],
                observation_schema_version=OBSERVATION_SCHEMA_VERSION,
                reward_version=bundle.reward_spec.reward_version,
                policy_state_requirements=[
                    "previous_position_side",
                    "previous_venue",
                    "hold_age_steps",
                    "turnover_accumulator",
                ],
                expected_return_score=expected_return_score,
                risk_score=risk_score,
                turnover_score=turnover_score,
                confidence_or_quality_score=confidence_or_quality_score,
                min_capital_requirement=500.0,
                size_bounds=size_band,
                leverage_bounds=leverage_band,
                artifact_compatibility_tags=artifact_tags,
                runtime_adapter=config.runtime_adapter,
                strict_runtime_contract=strict_runtime_contract,
                required_context=required_context,
                lineage_pointer=lineages,
            ),
            training_run_id=training_run_id,
            parent_artifact_id=parent_policy_id,
            training_summary=training_summary,
        )

    # ------------------------------------------------------------------
    # PRODUCTION streaming train path
    # ------------------------------------------------------------------

    def train_search_from_directory(
        self,
        manifest: TrajectoryManifest,
        directory: Path,
        parent_policy_id: str | None = None,
    ) -> TrainingSearchResult:
        """PRODUCTION PATH — train from a streaming JSONL trajectory directory.

        Streams development.jsonl per fold for walk-forward CV, then streams
        train.jsonl / validation.jsonl for final model fit.  Never assembles
        a full TrajectoryBundle in memory.
        """
        from quantlab_ml.trajectories.streaming_store import TrajectoryDirectoryStore

        candidate_specs = self._candidate_specs()
        code_commit_hash = current_code_commit_hash()
        training_run_id = self._training_run_id_from_manifest(
            manifest,
            parent_policy_id=parent_policy_id,
            code_commit_hash=code_commit_hash,
            candidate_specs=candidate_specs,
        )
        search_budget_summary = _search_budget_summary(candidate_specs)
        logger.info(
            "training_search_started training_run_id=%s candidate_count=%d split_version=%s reward_version=%s "
            "training_backend=%s training_device=%s cuda_available=%s device_name=%s",
            training_run_id,
            len(candidate_specs),
            manifest.split_artifact.split_version,
            manifest.reward_spec.reward_version,
            self._backend.backend_name,
            self._backend.device_resolution.training_device,
            self._backend.device_resolution.cuda_available,
            self._backend.device_resolution.device_name,
        )

        # Walk-forward fold cv: stream development.jsonl per fold
        selection_runs = [
            self._select_candidate_via_walkforward_streaming(
                manifest=manifest,
                directory=directory,
                candidate_spec=candidate_spec,
                candidate_index=candidate_index,
                parent_policy_id=parent_policy_id,
                training_run_id=training_run_id,
                code_commit_hash=code_commit_hash,
            )
            for candidate_index, candidate_spec in enumerate(candidate_specs)
        ]
        ranked_selections = sorted(selection_runs, key=_selection_ranking_key)

        final_train_window = StreamingWindow(split_name="train")
        final_validation_window = StreamingWindow(split_name="validation")
        prepared = self._prepare_training_data_streaming(
            manifest,
            directory,
            TrajectoryDirectoryStore,
            train_window=final_train_window,
            validation_window=final_validation_window,
        )

        candidate_results: list[TrainingCandidateResult] = []
        for candidate_rank, selection_run in enumerate(ranked_selections, start=1):
            selected_candidate = candidate_rank == 1
            candidate_run = self._train_candidate_from_manifest(
                manifest=manifest,
                directory=directory,
                prepared=prepared,
                train_window=final_train_window,
                validation_window=final_validation_window,
                store_cls=TrajectoryDirectoryStore,
                candidate_spec=selection_run.candidate_spec,
                candidate_index=selection_run.candidate_index,
                parent_policy_id=parent_policy_id,
                training_run_id=training_run_id,
                code_commit_hash=code_commit_hash,
            )
            candidate_summary = self._candidate_training_summary(
                prepared=prepared,
                candidate_run=candidate_run,
                selection_run=selection_run,
                training_run_id=training_run_id,
                candidate_rank=candidate_rank,
                selected_candidate=selected_candidate,
                search_budget_summary=search_budget_summary,
                training_data_flow="streaming_batch",
                validation_data_flow="streaming_evaluation",
                normalization_strategy="train_only_two_pass_streaming",
                proxy_validation_used=False,
                batch_plan=prepared.batch_plan,
            )
            artifact = self._build_artifact_from_manifest(
                manifest=manifest,
                config=candidate_run.config,
                training_run_id=training_run_id,
                code_commit_hash=code_commit_hash,
                parameters=candidate_run.best_parameters,
                parent_policy_id=parent_policy_id,
                validation_total_net_return=candidate_run.best_validation_total_net_return,
                validation_score=candidate_run.best_validation_score,
                training_summary=candidate_summary,
                search_metadata=_ArtifactSearchMetadata(
                    candidate_index=selection_run.candidate_index,
                    candidate_rank=candidate_rank,
                    selected_candidate=selected_candidate,
                ),
                validation_step_count=prepared.val_step_count,
            )
            candidate_results.append(
                TrainingCandidateResult(
                    artifact=artifact,
                    candidate_index=selection_run.candidate_index,
                    candidate_rank=candidate_rank,
                    selected_candidate=selected_candidate,
                    candidate_spec=selection_run.candidate_spec,
                    best_validation_total_net_return=candidate_run.best_validation_total_net_return,
                    best_validation_composite_rank=candidate_run.best_validation_score.composite_rank,
                )
            )

        result = TrainingSearchResult(
            training_run_id=training_run_id,
            selected_artifact=candidate_results[0].artifact,
            candidate_results=candidate_results,
            search_budget_summary=search_budget_summary,
        )
        logger.info(
            "training_search_completed training_run_id=%s selected_policy_id=%s candidate_count=%d "
            "selected_validation_total_net_return=%.6f selected_validation_composite_rank=%.6f",
            training_run_id,
            result.selected_artifact.policy_id,
            len(candidate_results),
            candidate_results[0].best_validation_total_net_return,
            candidate_results[0].best_validation_composite_rank,
        )
        return result

    def _training_run_id_from_manifest(
        self,
        manifest: TrajectoryManifest,
        *,
        parent_policy_id: str | None,
        code_commit_hash: str,
        candidate_specs: list[TrainingCandidateSpec],
    ) -> str:
        run_payload = {
            "dataset_hash": manifest.dataset_spec.dataset_hash,
            "slice_id": manifest.dataset_spec.slice_id,
            "split_version": manifest.split_artifact.split_version,
            "reward_version": manifest.reward_spec.reward_version,
            "training_backend": self._backend.backend_name,
            "trainer_config": self.config.model_dump(mode="json", exclude_none=False),
            "candidate_specs": [c.as_dict() for c in candidate_specs],
            "parent_policy_id": parent_policy_id,
            "code_commit_hash": code_commit_hash,
        }
        return f"trainrun-{hash_payload(run_payload)[:12]}"

    def _iter_window_records(
        self,
        directory: Path,
        window: StreamingWindow,
        *,
        store_cls: Any,
    ) -> Any:
        for record in store_cls.iter_records(directory, window.split_name):
            selected_steps = [step for step in record.steps if window.includes(step.event_time)]
            if not selected_steps:
                continue
            trajectory_id = record.trajectory_id
            if record.split != window.split_name:
                trajectory_id = f"{window.split_name}-{record.trajectory_id}"
            yield TrajectoryRecord(
                trajectory_id=trajectory_id,
                split=window.split_name,  # type: ignore[arg-type]
                target_symbol=record.target_symbol,
                start_time=selected_steps[0].event_time,
                end_time=selected_steps[-1].event_time,
                steps=selected_steps,
                terminal=record.terminal,
                terminal_reason=record.terminal_reason,
            )

    def _count_window_steps(
        self,
        directory: Path,
        window: StreamingWindow,
        *,
        store_cls: Any,
    ) -> int:
        step_count = 0
        for record in store_cls.iter_records(directory, window.split_name):
            step_count += sum(1 for step in record.steps if window.includes(step.event_time))
        return step_count

    def _streaming_feature_stats(
        self,
        directory: Path,
        window: StreamingWindow,
        *,
        store_cls: Any,
    ) -> StreamingFeatureStats:
        stats = StreamingFeatureStats()
        for record in store_cls.iter_records(directory, window.split_name):
            for step in record.steps:
                if not window.includes(step.event_time):
                    continue
                stats.update(np.asarray(observation_feature_vector(step.observation), dtype=np.float64))
        if stats.count <= 0:
            raise ValueError(
                f"streaming stats window {window.split_name!r} returned 0 qualifying examples"
            )
        return stats

    def _streaming_batch_plan(self, *, feature_dim: int, train_step_count: int) -> StreamingBatchPlan:
        bytes_per_example = (feature_dim * np.dtype(np.float64).itemsize) + _STREAMING_BATCH_LABEL_OVERHEAD_BYTES
        effective_batch_size = max(
            1,
            min(_STREAMING_BATCH_MAX_SIZE, _STREAMING_BATCH_TARGET_BYTES // max(bytes_per_example, 1)),
        )
        estimated_batch_bytes = effective_batch_size * bytes_per_example
        batches_per_epoch = math.ceil(train_step_count / effective_batch_size)
        return StreamingBatchPlan(
            batch_target_bytes=_STREAMING_BATCH_TARGET_BYTES,
            bytes_per_example=bytes_per_example,
            effective_batch_size=int(effective_batch_size),
            estimated_batch_bytes=int(estimated_batch_bytes),
            batches_per_epoch=int(batches_per_epoch),
        )

    def _prepare_training_data_streaming(
        self,
        manifest: TrajectoryManifest,
        directory: Path,
        store_cls: type,
        *,
        train_window: StreamingWindow,
        validation_window: StreamingWindow,
    ) -> _StreamingPreparedData:
        stats = self._streaming_feature_stats(directory, train_window, store_cls=store_cls)
        feature_mean, feature_std = stats.finalize()
        val_step_count = self._count_window_steps(directory, validation_window, store_cls=store_cls)
        if val_step_count <= 0:
            raise ValueError("validation split is empty")
        batch_plan = self._streaming_batch_plan(
            feature_dim=stats.feature_dim,
            train_step_count=stats.count,
        )
        logger.info(
            "streaming_training_data_prepared train_examples=%d validation_examples=%d "
            "feature_dim=%d effective_batch_size=%d estimated_batch_bytes=%d "
            "batches_per_epoch=%d batch_target_bytes=%d",
            stats.count,
            val_step_count,
            stats.feature_dim,
            batch_plan.effective_batch_size,
            batch_plan.estimated_batch_bytes,
            batch_plan.batches_per_epoch,
            batch_plan.batch_target_bytes,
        )
        return _StreamingPreparedData(
            train_step_count=stats.count,
            val_step_count=val_step_count,
            action_keys=manifest.action_space.action_keys,
            venue_choices=manifest.dataset_spec.exchanges,
            feature_mean=feature_mean,
            feature_std=feature_std,
            batch_plan=batch_plan,
        )

    def _select_candidate_via_walkforward_streaming(
        self,
        *,
        manifest: TrajectoryManifest,
        directory: Path,
        candidate_spec: TrainingCandidateSpec,
        candidate_index: int,
        parent_policy_id: str | None,
        training_run_id: str,
        code_commit_hash: str,
    ) -> _CandidateSelectionRun:
        from quantlab_ml.trajectories.streaming_store import TrajectoryDirectoryStore

        fold_scores: list[FoldValidationScore] = []
        interval = timedelta(seconds=manifest.dataset_spec.sampling_interval_seconds)

        for fold in manifest.split_artifact.folds:
            purge_cutoff = fold.validation_window.start - (
                interval * fold.purge_width_steps
            )
            train_window = StreamingWindow(
                split_name="development",
                start=fold.train_window.start,
                end=fold.train_window.end,
                exclusive_end=purge_cutoff if fold.purge_width_steps > 0 else None,
            )
            validation_window = StreamingWindow(
                split_name="development",
                start=fold.validation_window.start,
                end=fold.validation_window.end,
            )
            fold_prepared = self._prepare_training_data_streaming(
                manifest,
                directory,
                TrajectoryDirectoryStore,
                train_window=train_window,
                validation_window=validation_window,
            )
            fold_run = self._train_candidate_from_manifest(
                manifest=manifest,
                directory=directory,
                prepared=fold_prepared,
                train_window=train_window,
                validation_window=validation_window,
                store_cls=TrajectoryDirectoryStore,
                candidate_spec=candidate_spec,
                candidate_index=candidate_index,
                parent_policy_id=parent_policy_id,
                training_run_id=f"{training_run_id}:{fold.fold_id}",
                code_commit_hash=code_commit_hash,
            )
            fold_scores.append(
                FoldValidationScore(
                    fold_id=fold.fold_id,
                    validation_total_net_return=fold_run.best_validation_total_net_return,
                    validation_composite_rank=fold_run.best_validation_score.composite_rank,
                    validation_step_count=fold_prepared.val_step_count,
                )
            )
        selection_total_net_return = _weighted_mean(
            [s.validation_total_net_return for s in fold_scores],
            [s.validation_step_count for s in fold_scores],
        )
        selection_composite_rank = _weighted_mean(
            [s.validation_composite_rank for s in fold_scores],
            [s.validation_step_count for s in fold_scores],
        )
        logger.info(
            "training_candidate_walkforward_completed candidate_index=%d seed=%d "
            "learning_rate=%.6f l2_weight=%.6f fold_count=%d "
            "selection_total_net_return=%.6f selection_composite_rank=%.6f",
            candidate_index, candidate_spec.seed, candidate_spec.learning_rate,
            candidate_spec.l2_weight, len(fold_scores),
            selection_total_net_return, selection_composite_rank,
        )
        return _CandidateSelectionRun(
            candidate_spec=candidate_spec,
            candidate_index=candidate_index,
            fold_scores=fold_scores,
            selection_total_net_return=selection_total_net_return,
            selection_composite_rank=selection_composite_rank,
        )

    def _train_candidate_from_manifest(
        self,
        *,
        manifest: TrajectoryManifest,
        directory: Path,
        prepared: _StreamingPreparedData,
        train_window: StreamingWindow,
        validation_window: StreamingWindow,
        store_cls: Any,
        candidate_spec: TrainingCandidateSpec,
        candidate_index: int,
        parent_policy_id: str | None,
        training_run_id: str,
        code_commit_hash: str,
    ) -> _CandidateTrainingRun:
        """PRODUCTION PATH — streaming batch training with streaming validation."""
        config = self.config.model_copy(
            update={
                "seed": candidate_spec.seed,
                "learning_rate": candidate_spec.learning_rate,
                "l2_weight": candidate_spec.l2_weight,
                "candidate_search": None,
            }
        )
        state = self._backend.initialize_state(
            seed=config.seed,
            action_count=len(prepared.action_keys),
            venue_count=len(prepared.venue_choices),
            feature_dim=prepared.feature_dim,
        )
        logger.info(
            "streaming_training_candidate_started candidate_index=%d seed=%d learning_rate=%.6f "
            "l2_weight=%.6f effective_batch_size=%d estimated_batch_bytes=%d "
            "batches_per_epoch=%d batch_target_bytes=%d",
            candidate_index,
            candidate_spec.seed,
            candidate_spec.learning_rate,
            candidate_spec.l2_weight,
            prepared.batch_plan.effective_batch_size,
            prepared.batch_plan.estimated_batch_bytes,
            prepared.batch_plan.batches_per_epoch,
            prepared.batch_plan.batch_target_bytes,
        )
        loss_history: list[float] = []
        validation_history: list[float] = []
        best_validation_total_net_return: float | None = None
        best_parameters: LinearPolicyParameters | None = None
        best_validation_score: PolicyScore | None = None
        best_epoch = 0
        for epoch in range(1, config.epochs + 1):
            import time as _time
            _t_epoch0 = _time.perf_counter()
            total_loss = self._train_streaming_epoch(
                directory=directory,
                prepared=prepared,
                train_window=train_window,
                store_cls=store_cls,
                state=state,
                config=config,
            )
            _t_epoch1 = _time.perf_counter()
            logger.info(
                "epoch_timing epoch=%d wall_sec=%.1f total_loss=%.6f",
                epoch, _t_epoch1 - _t_epoch0, total_loss,
            )
            loss_history.append(total_loss)
            parameters = self._backend.parameters(
                state=state,
                action_keys=prepared.action_keys,
                venue_choices=prepared.venue_choices,
                feature_mean=prepared.feature_mean,
                feature_std=prepared.feature_std,
                config=config,
            )
            validation_artifact = self._build_artifact_from_manifest(
                manifest=manifest,
                config=config,
                training_run_id=training_run_id,
                code_commit_hash=code_commit_hash,
                parameters=parameters,
                parent_policy_id=parent_policy_id,
                validation_total_net_return=0.0,
                validation_score=None,
                training_summary={},
                search_metadata=None,
                validation_step_count=prepared.val_step_count,
            )
            validation_report = self._validation_report_from_manifest(
                manifest=manifest,
                directory=directory,
                artifact=validation_artifact,
                validation_window=validation_window,
                store_cls=store_cls,
            )
            validation_score = PolicyScorer().score(validation_report)
            validation_history.append(validation_report.total_net_return)
            is_best = (
                best_validation_total_net_return is None
                or validation_report.total_net_return > best_validation_total_net_return
            )
            epoch_result = StreamingEpochResult(
                epoch=epoch,
                total_loss=total_loss,
                validation_report=validation_report,
                validation_score=validation_score,
                is_best=is_best,
            )
            if epoch_result.is_best:
                best_epoch = epoch
                best_parameters = parameters
                best_validation_total_net_return = validation_report.total_net_return
                best_validation_score = validation_score

        assert best_parameters is not None
        assert best_validation_total_net_return is not None
        assert best_validation_score is not None
        logger.info(
            "training_candidate_completed candidate_index=%d seed=%d "
            "best_epoch=%d best_validation_total_net_return=%.6f "
            "best_validation_composite_rank=%.6f effective_batch_size=%d "
            "estimated_batch_bytes=%d batches_per_epoch=%d training_backend=%s",
            candidate_index,
            candidate_spec.seed,
            best_epoch,
            best_validation_total_net_return,
            best_validation_score.composite_rank,
            prepared.batch_plan.effective_batch_size,
            prepared.batch_plan.estimated_batch_bytes,
            prepared.batch_plan.batches_per_epoch,
            self._backend.backend_name,
        )
        return _CandidateTrainingRun(
            config=config,
            candidate_spec=candidate_spec,
            candidate_index=candidate_index,
            best_epoch=best_epoch,
            best_parameters=best_parameters,
            best_validation_total_net_return=best_validation_total_net_return,
            best_validation_score=best_validation_score,
            loss_history=loss_history,
            validation_history=validation_history,
        )

    def _train_streaming_epoch(
        self,
        *,
        directory: Path,
        prepared: _StreamingPreparedData,
        train_window: StreamingWindow,
        store_cls: Any,
        state: object,
        config: TrainingConfig,
    ) -> float:
        import time as _time  # diagnostic timing — non-semantic, safe to remove later

        batch_size = prepared.batch_plan.effective_batch_size
        action_key_to_index = {key: idx for idx, key in enumerate(prepared.action_keys)}
        venue_to_index = {venue: idx for idx, venue in enumerate(prepared.venue_choices)}
        feature_batch = np.empty((batch_size, prepared.feature_dim), dtype=np.float64)
        action_batch = np.empty(batch_size, dtype=np.int64)
        venue_mask_batch = np.empty(batch_size, dtype=np.bool_)
        venue_batch = np.empty(batch_size, dtype=np.int64)
        weighted_loss_total = 0.0
        seen = 0
        batch_row = 0

        # --- diagnostic timing state ---
        _prof_batch_num: int = 0
        _prof_t_feature: float = 0.0  # observation_feature_vector + np.asarray
        _prof_t_norm: float = 0.0     # mean/std normalization
        _prof_t_assembly: float = 0.0 # batch row assignment
        _prof_t_gpu: float = 0.0      # backend.batch_step (host->device + fwd/bwd)
        _prof_step_count: int = 0     # steps accumulated in this partial batch
        _PROF_LOG_FIRST_N = 10
        _PROF_LOG_EVERY_N = 50

        for record in store_cls.iter_records(directory, train_window.split_name):
            for step in record.steps:
                if not train_window.includes(step.event_time):
                    continue

                _t0 = _time.perf_counter()
                features = np.asarray(observation_feature_vector(step.observation), dtype=np.float64)
                _t1 = _time.perf_counter()
                features -= prepared.feature_mean
                features /= prepared.feature_std
                _t2 = _time.perf_counter()

                action_key, venue = _best_label(step)
                feature_batch[batch_row] = features
                action_batch[batch_row] = action_key_to_index[action_key]
                venue_mask_batch[batch_row] = venue is not None
                venue_batch[batch_row] = venue_to_index[venue] if venue is not None else 0
                _t3 = _time.perf_counter()

                _prof_t_feature += _t1 - _t0
                _prof_t_norm += _t2 - _t1
                _prof_t_assembly += _t3 - _t2
                _prof_step_count += 1
                batch_row += 1

                if batch_row == batch_size:
                    _tg0 = _time.perf_counter()
                    batch_loss = self._backend.batch_step(
                        state=state,
                        batch_features=feature_batch,
                        batch_action_labels=action_batch,
                        batch_venue_mask=venue_mask_batch,
                        batch_venue_labels=venue_batch,
                        config=config,
                    )
                    _tg1 = _time.perf_counter()
                    _prof_t_gpu += _tg1 - _tg0

                    weighted_loss_total += batch_loss * batch_row
                    seen += batch_row

                    if _prof_batch_num < _PROF_LOG_FIRST_N or _prof_batch_num % _PROF_LOG_EVERY_N == 0:
                        _batch_total = _prof_t_feature + _prof_t_norm + _prof_t_assembly + _prof_t_gpu
                        logger.info(
                            "batch_timing batch=%d steps=%d "
                            "t_feature_ms=%.1f t_norm_ms=%.1f t_assembly_ms=%.1f t_gpu_ms=%.1f "
                            "t_batch_total_ms=%.1f t_per_step_feature_ms=%.2f feature_dim=%d",
                            _prof_batch_num, _prof_step_count,
                            _prof_t_feature * 1000,
                            _prof_t_norm * 1000,
                            _prof_t_assembly * 1000,
                            _prof_t_gpu * 1000,
                            _batch_total * 1000,
                            (_prof_t_feature / max(_prof_step_count, 1)) * 1000,
                            prepared.feature_dim,
                        )

                    _prof_batch_num += 1
                    _prof_t_feature = 0.0
                    _prof_t_norm = 0.0
                    _prof_t_assembly = 0.0
                    _prof_t_gpu = 0.0
                    _prof_step_count = 0
                    batch_row = 0

        if batch_row > 0:
            _tg0 = _time.perf_counter()
            batch_loss = self._backend.batch_step(
                state=state,
                batch_features=feature_batch[:batch_row],
                batch_action_labels=action_batch[:batch_row],
                batch_venue_mask=venue_mask_batch[:batch_row],
                batch_venue_labels=venue_batch[:batch_row],
                config=config,
            )
            _tg1 = _time.perf_counter()
            weighted_loss_total += batch_loss * batch_row
            seen += batch_row

        if seen <= 0:
            raise ValueError("train split is empty")
        return float(weighted_loss_total / seen)

    def _validation_report_from_manifest(
        self,
        *,
        manifest: TrajectoryManifest,
        directory: Path,
        artifact: PolicyArtifact,
        validation_window: StreamingWindow,
        store_cls: Any,
    ) -> EvaluationReport:
        from quantlab_ml.evaluation import EvaluationEngine

        engine = EvaluationEngine(self._evaluation_boundary(manifest.reward_spec.timestamping))
        return engine.evaluate_records(
            manifest.dataset_spec,
            manifest.reward_spec,
            self._iter_window_records(
                directory,
                validation_window,
                store_cls=store_cls,
            ),
            artifact,
        )

    def _build_artifact_from_manifest(
        self,
        *,
        manifest: TrajectoryManifest,
        config: TrainingConfig,
        training_run_id: str,
        code_commit_hash: str,
        parameters: LinearPolicyParameters,
        parent_policy_id: str | None,
        validation_total_net_return: float,
        validation_score: PolicyScore | None,
        training_summary: dict[str, object],
        search_metadata: _ArtifactSearchMetadata | None,
        validation_step_count: int,
    ) -> PolicyArtifact:
        payload_blob = parameters.model_dump_json()
        payload = OpaquePolicyPayload(
            runtime_adapter=config.runtime_adapter,
            payload_format="json",
            payload_format_version="json-v1",
            blob=payload_blob,
            digest=hash_payload(parameters),
        )
        lineages = LineagePointer(
            parent_policy_id=parent_policy_id,
            generation=0 if parent_policy_id is None else 1,
            notes=["v2 surface - streaming linear policy trainer"],
        )
        training_config_hash = hash_payload(config)
        training_snapshot_id = (
            f"{manifest.dataset_spec.dataset_hash}:{manifest.dataset_spec.slice_id}"
        )
        artifact_identity = hash_payload(
            {
                "payload_digest": payload.digest,
                "training_config_hash": training_config_hash,
                "training_snapshot_id": training_snapshot_id,
                "training_run_id": training_run_id,
            }
        )
        policy_id = f"policy-{artifact_identity[:12]}"
        artifact_id = f"artifact-{artifact_identity[:12]}"
        evaluation_surface_id = (
            f"{manifest.dataset_spec.slice_id}"
            f":{manifest.split_artifact.split_version}"
            f":{manifest.reward_spec.reward_version}"
        )
        target_asset = (
            manifest.dataset_spec.symbols[0]
            if len(manifest.dataset_spec.symbols) == 1
            else DYNAMIC_TARGET_ASSET
        )
        required_context: dict[str, object] = {}
        if target_asset == DYNAMIC_TARGET_ASSET:
            required_context = {"target_symbol_source": "observation.target_symbol"}

        expected_return_score = validation_total_net_return / max(validation_step_count, 1)
        risk_score = best_effort_metric(validation_score, "risk_score")
        turnover_score = best_effort_metric(validation_score, "turnover_score")
        confidence_or_quality_score = min(
            0.99, max(best_effort_metric(validation_score, "composite_rank"), 0.0)
        )
        size_band = _band_by_key(manifest.action_space.size_bands, config.preferred_size_band)
        leverage_band = _band_by_key(manifest.action_space.leverage_bands, config.preferred_leverage_band)
        strict_runtime_contract = build_strict_runtime_contract(
            manifest.observation_schema, policy_kind=config.runtime_adapter
        )
        artifact_tags = [
            f"runtime_adapter:{config.runtime_adapter}",
            f"reward:{manifest.reward_spec.reward_version}",
            f"split:{manifest.split_artifact.split_version}",
            f"observation:{OBSERVATION_SCHEMA_VERSION}",
            f"action_space:{ACTION_SPACE_VERSION}",
            f"runtime_contract:{strict_runtime_contract.runtime_contract_version}",
            f"policy_kind:{strict_runtime_contract.policy_kind}",
            f"derived_contract:{strict_runtime_contract.derived_contract_version}",
            f"derived_signature:{strict_runtime_contract.derived_channel_template_signature}",
            f"feature_dim:{strict_runtime_contract.expected_feature_dim}",
            "compat_mode:strict",
        ]
        if search_metadata is not None:
            artifact_tags.extend(
                [
                    f"search_run_id:{training_run_id}",
                    f"search_candidate_index:{search_metadata.candidate_index}",
                    f"search_candidate_rank:{search_metadata.candidate_rank}",
                    f"search_selected:{str(search_metadata.selected_candidate).lower()}",
                ]
            )
        return PolicyArtifact(
            artifact_id=artifact_id,
            artifact_version=POLICY_ARTIFACT_SCHEMA_VERSION,
            policy_id=policy_id,
            policy_family=config.trainer_name,
            training_snapshot_id=training_snapshot_id,
            training_config_hash=training_config_hash,
            code_commit_hash=code_commit_hash,
            reward_version=manifest.reward_spec.reward_version,
            evaluation_surface_id=evaluation_surface_id,
            target_asset=target_asset,
            allowed_venues=manifest.dataset_spec.exchanges,
            allowed_action_family=manifest.action_space.action_keys,
            required_context=required_context,
            created_at=utcnow(),
            observation_schema=manifest.observation_schema,
            action_space=manifest.action_space,
            policy_payload=payload,
            runtime_metadata=RuntimeMetadata(
                target_asset=target_asset,
                allowed_venues=manifest.dataset_spec.exchanges,
                action_space_version=ACTION_SPACE_VERSION,
                required_streams=manifest.dataset_spec.stream_universe,
                required_field_families={
                    stream: manifest.observation_schema.field_axis.get(stream, [])
                    for stream in manifest.dataset_spec.stream_universe
                },
                required_scale_preset=[
                    scale.label for scale in manifest.trajectory_spec.scale_preset
                ],
                observation_schema_version=OBSERVATION_SCHEMA_VERSION,
                reward_version=manifest.reward_spec.reward_version,
                policy_state_requirements=[
                    "previous_position_side",
                    "previous_venue",
                    "hold_age_steps",
                    "turnover_accumulator",
                ],
                expected_return_score=expected_return_score,
                risk_score=risk_score,
                turnover_score=turnover_score,
                confidence_or_quality_score=confidence_or_quality_score,
                min_capital_requirement=500.0,
                size_bounds=size_band,
                leverage_bounds=leverage_band,
                artifact_compatibility_tags=artifact_tags,
                runtime_adapter=config.runtime_adapter,
                strict_runtime_contract=strict_runtime_contract,
                required_context=required_context,
                lineage_pointer=lineages,
            ),
            training_run_id=training_run_id,
            parent_artifact_id=parent_policy_id,
            training_summary=training_summary,
        )

class MomentumBaselineTrainer(LinearPolicyTrainer):
    def __init__(self, config: TrainingConfig):
        warnings.warn(
            "MomentumBaselineTrainer is deprecated; use LinearPolicyTrainer.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(config, backend_name="numpy")


@dataclass(slots=True)
class _ArtifactSearchMetadata:
    candidate_index: int
    candidate_rank: int
    selected_candidate: bool


@dataclass(slots=True)
class _CandidateTrainingRun:
    config: TrainingConfig
    candidate_spec: TrainingCandidateSpec
    candidate_index: int
    best_epoch: int
    best_parameters: LinearPolicyParameters
    best_validation_total_net_return: float
    best_validation_score: PolicyScore
    loss_history: list[float]
    validation_history: list[float]


class _LinearTrainingBackend:
    backend_name: TrainingBackendName
    device_resolution: _DeviceResolution

    def initialize_state(
        self,
        *,
        seed: int,
        action_count: int,
        venue_count: int,
        feature_dim: int,
    ) -> object:
        raise NotImplementedError

    def step(
        self,
        *,
        state: object,
        prepared: _PreparedTrainingData,
        config: TrainingConfig,
    ) -> float:
        raise NotImplementedError

    def batch_step(
        self,
        *,
        state: object,
        batch_features: np.ndarray,
        batch_action_labels: np.ndarray,
        batch_venue_mask: np.ndarray,
        batch_venue_labels: np.ndarray,
        config: TrainingConfig,
    ) -> float:
        raise NotImplementedError

    def parameters(
        self,
        *,
        state: object,
        action_keys: list[str],
        venue_choices: list[str],
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
        config: TrainingConfig,
    ) -> LinearPolicyParameters:
        raise NotImplementedError


@dataclass(slots=True)
class _NumpyTrainingState:
    action_weight: np.ndarray
    action_bias: np.ndarray
    venue_weight: np.ndarray
    venue_bias: np.ndarray


class _NumpyLinearTrainingBackend(_LinearTrainingBackend):
    backend_name: TrainingBackendName = "numpy"
    device_resolution = _DeviceResolution(
        training_device="cpu",
        cuda_available=False,
        device_name="cpu",
        compute_device=None,
    )

    def initialize_state(
        self,
        *,
        seed: int,
        action_count: int,
        venue_count: int,
        feature_dim: int,
    ) -> _NumpyTrainingState:
        action_weight, action_bias, venue_weight, venue_bias = _initial_parameter_arrays(
            seed=seed,
            action_count=action_count,
            venue_count=venue_count,
            feature_dim=feature_dim,
        )
        return _NumpyTrainingState(
            action_weight=action_weight,
            action_bias=action_bias,
            venue_weight=venue_weight,
            venue_bias=venue_bias,
        )

    def step(
        self,
        *,
        state: object,
        prepared: _PreparedTrainingData,
        config: TrainingConfig,
    ) -> float:
        return self._step_arrays(
            state=state,
            batch_features=prepared.normalized_train.astype(np.float64),
            batch_action_labels=prepared.action_labels,
            batch_venue_mask=prepared.venue_mask,
            batch_venue_labels=prepared.venue_labels,
            config=config,
        )

    def batch_step(
        self,
        *,
        state: object,
        batch_features: np.ndarray,
        batch_action_labels: np.ndarray,
        batch_venue_mask: np.ndarray,
        batch_venue_labels: np.ndarray,
        config: TrainingConfig,
    ) -> float:
        return self._step_arrays(
            state=state,
            batch_features=batch_features,
            batch_action_labels=batch_action_labels,
            batch_venue_mask=batch_venue_mask,
            batch_venue_labels=batch_venue_labels,
            config=config,
        )

    def _step_arrays(
        self,
        *,
        state: object,
        batch_features: np.ndarray,
        batch_action_labels: np.ndarray,
        batch_venue_mask: np.ndarray,
        batch_venue_labels: np.ndarray,
        config: TrainingConfig,
    ) -> float:
        training_state = _expect_numpy_state(state)
        action_logits = batch_features @ training_state.action_weight.T + training_state.action_bias
        action_probabilities = _softmax_matrix(action_logits)
        action_loss = _cross_entropy_loss(action_probabilities, batch_action_labels)
        action_gradient = action_probabilities.copy()
        action_gradient[np.arange(len(batch_action_labels)), batch_action_labels] -= 1.0
        action_gradient /= len(batch_action_labels)

        action_weight_gradient = action_gradient.T @ batch_features
        action_bias_gradient = action_gradient.sum(axis=0)

        venue_loss = 0.0
        venue_weight_gradient = np.zeros_like(training_state.venue_weight)
        venue_bias_gradient = np.zeros_like(training_state.venue_bias)
        if batch_venue_mask.any():
            masked_inputs = batch_features[batch_venue_mask]
            masked_labels = batch_venue_labels[batch_venue_mask]
            venue_logits = masked_inputs @ training_state.venue_weight.T + training_state.venue_bias
            venue_probabilities = _softmax_matrix(venue_logits)
            venue_loss = _cross_entropy_loss(venue_probabilities, masked_labels)
            venue_gradient = venue_probabilities.copy()
            venue_gradient[np.arange(len(masked_labels)), masked_labels] -= 1.0
            venue_gradient /= len(masked_labels)
            venue_weight_gradient = venue_gradient.T @ masked_inputs
            venue_bias_gradient = venue_gradient.sum(axis=0)

        total_loss = action_loss + venue_loss
        if config.l2_weight > 0.0:
            total_loss += config.l2_weight * (
                float(np.sum(training_state.action_weight**2)) + float(np.sum(training_state.venue_weight**2))
            )
            action_weight_gradient += config.l2_weight * training_state.action_weight
            venue_weight_gradient += config.l2_weight * training_state.venue_weight

        training_state.action_weight -= config.learning_rate * action_weight_gradient
        training_state.action_bias -= config.learning_rate * action_bias_gradient
        training_state.venue_weight -= config.learning_rate * venue_weight_gradient
        training_state.venue_bias -= config.learning_rate * venue_bias_gradient
        return float(total_loss)

    def parameters(
        self,
        *,
        state: object,
        action_keys: list[str],
        venue_choices: list[str],
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
        config: TrainingConfig,
    ) -> LinearPolicyParameters:
        training_state = _expect_numpy_state(state)
        return _build_linear_policy_parameters(
            action_keys=action_keys,
            venue_choices=venue_choices,
            feature_mean=feature_mean,
            feature_std=feature_std,
            config=config,
            action_weight=training_state.action_weight.tolist(),
            action_bias=training_state.action_bias.tolist(),
            venue_weight=training_state.venue_weight.tolist(),
            venue_bias=training_state.venue_bias.tolist(),
        )


@dataclass(slots=True)
class _TorchTrainingState:
    action_weight: Any
    action_bias: Any
    venue_weight: Any
    venue_bias: Any


class _TorchLinearTrainingBackend(_LinearTrainingBackend):
    backend_name: TrainingBackendName = "pytorch"

    def __init__(self) -> None:
        self._torch = _require_torch()
        self.device_resolution = _resolve_torch_device(self._torch)

    def initialize_state(
        self,
        *,
        seed: int,
        action_count: int,
        venue_count: int,
        feature_dim: int,
    ) -> _TorchTrainingState:
        action_weight, action_bias, venue_weight, venue_bias = _initial_parameter_arrays(
            seed=seed,
            action_count=action_count,
            venue_count=venue_count,
            feature_dim=feature_dim,
        )
        torch_module = self._torch
        torch_module.manual_seed(seed)
        if self.device_resolution.cuda_available and hasattr(torch_module.cuda, "manual_seed_all"):
            torch_module.cuda.manual_seed_all(seed)
        if hasattr(torch_module, "use_deterministic_algorithms"):
            # cuBLAS on CUDA >= 10.2 requires this env var for deterministic
            # GEMM operations.  Set before the first cuBLAS call.
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch_module.use_deterministic_algorithms(True)
        compute_device = self.device_resolution.compute_device
        return _TorchTrainingState(
            action_weight=torch_module.tensor(action_weight, dtype=torch_module.float64, device=compute_device),
            action_bias=torch_module.tensor(action_bias, dtype=torch_module.float64, device=compute_device),
            venue_weight=torch_module.tensor(venue_weight, dtype=torch_module.float64, device=compute_device),
            venue_bias=torch_module.tensor(venue_bias, dtype=torch_module.float64, device=compute_device),
        )

    def step(
        self,
        *,
        state: object,
        prepared: _PreparedTrainingData,
        config: TrainingConfig,
    ) -> float:
        """Mini-batch SGD step on a random sample of the training data.

        Moving the full training matrix (potentially 26 GB float32) to GPU
        at once would exceed 24 GB VRAM on a 3090.  Instead, a random
        BATCH_SIZE subsample is moved to GPU per step.  Each batch tensor
        is small (~1-2 GB for 1024 examples), well within VRAM limits.

        Convergence: with num_epochs=200 and BATCH_SIZE=1024, each example
        is seen ~21 times on average (200 * 1024 / n_train), sufficient for
        cross-entropy minimization on a linear model.
        """
        n = prepared.train_step_count
        batch_size = min(1024, n)

        # Full-batch when data fits within BATCH_SIZE (e.g. fixture / test data).
        # This preserves numerical parity with the NumPy backend.
        # For production data (n > 1024), a random mini-batch is used so that
        # no single batch tensor exceeds available GPU VRAM (~1-2 GB per 1024 rows).
        if n <= 1024:
            batch_idx = np.arange(n, dtype=np.int64)  # all examples, deterministic order
        else:
            batch_idx = np.random.randint(0, n, size=batch_size)  # random subsample
        return self.batch_step(
            state=state,
            batch_features=prepared.normalized_train[batch_idx].astype(np.float64),
            batch_action_labels=prepared.action_labels[batch_idx],
            batch_venue_mask=prepared.venue_mask[batch_idx],
            batch_venue_labels=prepared.venue_labels[batch_idx],
            config=config,
        )

    def batch_step(
        self,
        *,
        state: object,
        batch_features: np.ndarray,
        batch_action_labels: np.ndarray,
        batch_venue_mask: np.ndarray,
        batch_venue_labels: np.ndarray,
        config: TrainingConfig,
    ) -> float:
        training_state = _expect_torch_state(state)
        torch_module = self._torch
        device = self.device_resolution.compute_device
        batch_size = int(batch_action_labels.shape[0])

        x_batch = torch_module.tensor(batch_features, dtype=torch_module.float64, device=device)
        labels_batch = torch_module.tensor(batch_action_labels, dtype=torch_module.int64, device=device)
        venue_mask_batch = torch_module.tensor(batch_venue_mask, dtype=torch_module.bool, device=device)
        venue_labels_batch = torch_module.tensor(batch_venue_labels, dtype=torch_module.int64, device=device)

        action_logits = x_batch @ training_state.action_weight.transpose(0, 1) + training_state.action_bias
        action_probabilities = torch_module.softmax(action_logits, dim=1)
        action_loss = _torch_cross_entropy_loss(torch_module, action_probabilities, labels_batch)
        action_gradient = action_probabilities.clone()
        action_gradient[
            torch_module.arange(batch_size, device=device),
            labels_batch,
        ] -= 1.0
        action_gradient /= batch_size

        action_weight_gradient = action_gradient.transpose(0, 1) @ x_batch
        action_bias_gradient = action_gradient.sum(dim=0)

        venue_loss = torch_module.tensor(
            0.0, dtype=torch_module.float64, device=training_state.action_weight.device,
        )
        venue_weight_gradient = torch_module.zeros_like(training_state.venue_weight)
        venue_bias_gradient = torch_module.zeros_like(training_state.venue_bias)
        if bool(venue_mask_batch.any().item()):
            masked_inputs = x_batch[venue_mask_batch]
            masked_labels = venue_labels_batch[venue_mask_batch]
            masked_size = int(masked_labels.shape[0])
            venue_logits = masked_inputs @ training_state.venue_weight.transpose(0, 1) + training_state.venue_bias
            venue_probabilities = torch_module.softmax(venue_logits, dim=1)
            venue_loss = _torch_cross_entropy_loss(torch_module, venue_probabilities, masked_labels)
            venue_gradient = venue_probabilities.clone()
            venue_gradient[
                torch_module.arange(masked_size, device=device), masked_labels,
            ] -= 1.0
            venue_gradient /= batch_size
            venue_weight_gradient = venue_gradient.transpose(0, 1) @ masked_inputs
            venue_bias_gradient = venue_gradient.sum(dim=0)

        total_loss = action_loss + venue_loss
        if config.l2_weight > 0.0:
            total_loss = total_loss + config.l2_weight * (
                torch_module.sum(training_state.action_weight**2) + torch_module.sum(training_state.venue_weight**2)
            )
            action_weight_gradient = action_weight_gradient + config.l2_weight * training_state.action_weight
            venue_weight_gradient = venue_weight_gradient + config.l2_weight * training_state.venue_weight

        training_state.action_weight = training_state.action_weight - (config.learning_rate * action_weight_gradient)
        training_state.action_bias = training_state.action_bias - (config.learning_rate * action_bias_gradient)
        training_state.venue_weight = training_state.venue_weight - (config.learning_rate * venue_weight_gradient)
        training_state.venue_bias = training_state.venue_bias - (config.learning_rate * venue_bias_gradient)
        return float(total_loss.item())

    def parameters(
        self,
        *,
        state: object,
        action_keys: list[str],
        venue_choices: list[str],
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
        config: TrainingConfig,
    ) -> LinearPolicyParameters:
        training_state = _expect_torch_state(state)
        return _build_linear_policy_parameters(
            action_keys=action_keys,
            venue_choices=venue_choices,
            feature_mean=feature_mean,
            feature_std=feature_std,
            config=config,
            action_weight=training_state.action_weight.detach().cpu().tolist(),
            action_bias=training_state.action_bias.detach().cpu().tolist(),
            venue_weight=training_state.venue_weight.detach().cpu().tolist(),
            venue_bias=training_state.venue_bias.detach().cpu().tolist(),
        )


def _search_budget_summary(candidate_specs: list[TrainingCandidateSpec]) -> SearchBudgetSummary:
    unique_hyperparameters = {(candidate.learning_rate, candidate.l2_weight) for candidate in candidate_specs}
    return SearchBudgetSummary(
        tried_models=len(candidate_specs),
        tried_seeds=len({candidate.seed for candidate in candidate_specs}),
        tried_architectures=1,
        tried_reward_variants=1,
        tried_hyperparameter_variants=len(unique_hyperparameters),
        total_candidate_count=len(candidate_specs),
    )


def _selection_ranking_key(candidate_run: _CandidateSelectionRun) -> tuple[float, float, int]:
    return (
        -candidate_run.selection_total_net_return,
        -candidate_run.selection_composite_rank,
        candidate_run.candidate_index,
    )


def _weighted_mean(values: list[float], weights: list[int]) -> float:
    total_weight = sum(weights)
    if total_weight <= 0:
        raise ValueError("weighted mean requires positive total weight")
    return float(sum(value * weight for value, weight in zip(values, weights, strict=True)) / total_weight)


def _best_label(step: TrajectoryStep) -> tuple[str, str | None]:
    abstain_reward = step.reward_snapshot.for_action("abstain").net_reward
    best_directional = None
    for reward in step.reward_snapshot.action_rewards:
        if reward.action_key == "abstain" or not reward.applicable:
            continue
        if best_directional is None or reward.net_reward > best_directional.net_reward:
            best_directional = reward
    if best_directional is None or best_directional.net_reward <= abstain_reward:
        return "abstain", None
    return best_directional.action_key, best_directional.venue


def _softmax_matrix(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis=1, keepdims=True)


def _cross_entropy_loss(probabilities: np.ndarray, labels: np.ndarray) -> float:
    chosen = probabilities[np.arange(len(labels)), labels]
    clipped = np.clip(chosen, 1e-12, 1.0)
    return float(-np.mean(np.log(clipped)))


def _torch_cross_entropy_loss(torch_module: Any, probabilities: Any, labels: Any) -> Any:
    chosen = probabilities[torch_module.arange(labels.shape[0], device=labels.device), labels]
    clipped = torch_module.clamp(chosen, min=1e-12, max=1.0)
    return -torch_module.mean(torch_module.log(clipped))


def _initial_parameter_arrays(
    *,
    seed: int,
    action_count: int,
    venue_count: int,
    feature_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    action_weight = rng.normal(0.0, 0.01, size=(action_count, feature_dim)).astype(np.float64)
    action_bias = np.zeros(action_count, dtype=np.float64)
    venue_weight = rng.normal(0.0, 0.01, size=(venue_count, feature_dim)).astype(np.float64)
    venue_bias = np.zeros(venue_count, dtype=np.float64)
    return action_weight, action_bias, venue_weight, venue_bias


def _build_linear_policy_parameters(
    *,
    action_keys: list[str],
    venue_choices: list[str],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    config: TrainingConfig,
    action_weight: list[list[float]],
    action_bias: list[float],
    venue_weight: list[list[float]],
    venue_bias: list[float],
) -> LinearPolicyParameters:
    return LinearPolicyParameters(
        action_keys=action_keys,
        venue_choices=venue_choices,
        feature_mean=feature_mean.tolist(),
        feature_std=feature_std.tolist(),
        action_weight=action_weight,
        action_bias=action_bias,
        venue_weight=venue_weight,
        venue_bias=venue_bias,
        preferred_size_band=config.preferred_size_band,
        preferred_leverage_band=config.preferred_leverage_band,
    )


def _resolve_training_backend(backend_name: TrainingBackendName) -> _LinearTrainingBackend:
    if backend_name == "numpy":
        return _NumpyLinearTrainingBackend()
    if backend_name == "pytorch":
        return _TorchLinearTrainingBackend()
    raise ValueError(f"unsupported training backend: {backend_name}")


def _expect_numpy_state(state: object) -> _NumpyTrainingState:
    if not isinstance(state, _NumpyTrainingState):
        raise TypeError("expected NumPy training state")
    return state


def _expect_torch_state(state: object) -> _TorchTrainingState:
    if not isinstance(state, _TorchTrainingState):
        raise TypeError("expected PyTorch training state")
    return state


def _require_torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised only in missing-ml environments.
        raise RuntimeError(
            "PyTorch backend requires the optional ML stack; install quantlab-ml with '.[dev,ml]' "
            "or add 'torch>=2.4,<3' to the active environment."
        ) from exc
    return torch


def _resolve_torch_device(torch_module: Any) -> _DeviceResolution:
    cuda_available = bool(hasattr(torch_module, "cuda") and torch_module.cuda.is_available())
    if cuda_available:
        compute_device = torch_module.device("cuda")
        try:
            device_name = str(torch_module.cuda.get_device_name(0))
        except Exception:  # pragma: no cover - defensive fallback
            device_name = "cuda"
        return _DeviceResolution(
            training_device="cuda",
            cuda_available=True,
            device_name=device_name,
            compute_device=compute_device,
        )

    return _DeviceResolution(
        training_device="cpu",
        cuda_available=False,
        device_name="cpu",
        compute_device=torch_module.device("cpu"),
    )


def _band_by_key(bands: list[NumericBand], key: str) -> NumericBand:
    for band in bands:
        if band.key == key:
            return band
    raise KeyError(f"unknown numeric band key: {key}")


def best_effort_metric(score: PolicyScore | None, field_name: str) -> float:
    if score is None:
        return 0.0
    return float(getattr(score, field_name))

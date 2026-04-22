from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Literal
import warnings

import numpy as np

from quantlab_ml.common import current_code_commit_hash, dump_model, hash_payload, load_model, utcnow
from quantlab_ml.contracts import (
    ACTION_SPACE_VERSION,
    ACTION_SPACE_VERSION_V2_PHASE1A,
    DYNAMIC_TARGET_ASSET,
    JOINT_ACTION_VOCABULARY_VERSION_PHASE1A,
    OBSERVATION_SCHEMA_VERSION,
    POLICY_STATE_FEATURE_VERSION_PHASE1A,
    ActionSpaceSpec,
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
    PolicyState,
    WalkForwardFold,
)
from quantlab_ml.contracts.policies import build_evaluation_surface_id
from quantlab_ml.models.features import observation_feature_vector, phase1a_feature_array
from quantlab_ml.models.linear_policy import LinearPolicyParameters, LinearPolicyV2Parameters
from quantlab_ml.runtime_contract import build_strict_runtime_contract
from quantlab_ml.scoring import PolicyScorer
from quantlab_ml.training.config import TrainingConfig
from quantlab_ml.registry.bundle_errors import DanglingTensorCacheManifestError
from quantlab_ml.registry.bundle_integrity import infer_bundle_payload_error_for_directory
from quantlab_ml.trajectories.tensor_cache import (
    TENSOR_CACHE_FORMAT_VERSION,
    TensorCacheManifest,
    best_label_from_step,
    load_tensor_cache_shard,
    read_tensor_cache_manifest,
    tensor_cache_payload_status,
    window_row_indices,
)
from quantlab_ml.rewards import RewardEngine
from . import compat_matrix_first
from .compat_matrix_first import CompatPreparedTrainingData as _PreparedTrainingData
from .phase1a_oracle import (
    apply_phase1a_joint_action,
    phase1a_joint_action_keys,
    phase1a_joint_action_mask,
    phase1a_label_available,
    solve_phase1a_oracle,
)
from .phase1a_supervision import (
    LoadedPhase1ASupervisionShard,
    Phase1ASupervisionManifest,
    Phase1ASupervisionShardManifest,
    load_phase1a_supervision_shard,
    materialize_phase1a_supervision,
    phase1a_supervision_directory,
    phase1a_supervision_payload_status,
    read_phase1a_supervision_manifest,
)

logger = logging.getLogger(__name__)

TrainingBackendName = Literal["numpy", "pytorch"]

_STREAMING_BATCH_TARGET_BYTES = 128 * 1024 * 1024
_STREAMING_BATCH_MAX_SIZE = 4096
_STREAMING_BATCH_LABEL_OVERHEAD_BYTES = (
    np.dtype(np.int64).itemsize * 2 + np.dtype(np.bool_).itemsize
)
_PHASE1A_VALUE_LABEL_OVERHEAD_BYTES = np.dtype(np.float64).itemsize
_PHASE1A_VALUE_GRAD_CLIP_NORM = 1_000.0
_PHASE1A_AUX_VALUE_HUBER_DELTA = 1.0


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
        self.update_batch(np.asarray(features, dtype=np.float64).reshape(1, -1))

    def update_batch(self, features: np.ndarray) -> None:
        matrix = np.asarray(features, dtype=np.float64)
        if matrix.ndim != 2:
            raise ValueError("streaming feature stats update_batch expects a 2D array")
        if matrix.shape[0] <= 0:
            return
        batch_count = int(matrix.shape[0])
        batch_mean = matrix.mean(axis=0)
        centered = matrix - batch_mean
        batch_m2 = np.sum(centered * centered, axis=0)
        if self.mean is None or self.m2 is None:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
            return
        total_count = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean = self.mean + (delta * (batch_count / total_count))
        self.m2 = self.m2 + batch_m2 + ((delta * delta) * (self.count * batch_count / total_count))
        self.count = total_count

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


@dataclass(slots=True)
class _Phase1APreparedData:
    train_step_count: int
    val_step_count: int
    feature_mean: np.ndarray
    feature_std: np.ndarray
    batch_plan: StreamingBatchPlan
    venue_choices: list[str]
    joint_action_keys: list[str]
    oracle_masked_row_count: int
    oracle_source_row_count: int
    train_row_selections: list["_Phase1ATrainShardSelection"] | None = None
    validation_window: StreamingWindow | None = None
    tensor_cache_used: bool = False
    phase1a_supervision_used: bool = False

    @property
    def feature_dim(self) -> int:
        return int(self.feature_mean.shape[0])

    @property
    def oracle_label_coverage_ratio(self) -> float:
        if self.oracle_source_row_count <= 0:
            return 0.0
        return self.train_step_count / self.oracle_source_row_count


TrajectoryFactory = Callable[[], Any]


@dataclass(frozen=True, slots=True)
class _Phase1ATrainShardSelection:
    directory: Path
    cache_shard: Any
    supervision_shard: Phase1ASupervisionShardManifest
    row_indices: np.ndarray


@dataclass(slots=True)
class _Phase1AEpochMetrics:
    total_loss: float
    batch_assembly_wall_sec: float
    batch_compute_wall_sec: float
    numerics: "_Phase1ANumericsTelemetry"


@dataclass(slots=True)
class _Phase1ANumericsTelemetry:
    joint_ce_loss: float = 0.0
    aux_value_loss_raw: float = 0.0
    aux_value_loss_weighted: float = 0.0
    total_loss: float = 0.0
    action_logit_abs_max: float = 0.0
    action_entropy: float = 0.0
    value_pred_abs_max: float = 0.0
    value_grad_norm_pre_clip: float = 0.0
    value_grad_norm_post_clip: float = 0.0
    clip_applied_count: int = 0
    first_nonfinite_component: str | None = None
    first_nonfinite_batch_context: dict[str, object] | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "joint_ce_loss": self.joint_ce_loss,
            "aux_value_loss_raw": self.aux_value_loss_raw,
            "aux_value_loss_weighted": self.aux_value_loss_weighted,
            "total_loss": self.total_loss,
            "action_logit_abs_max": self.action_logit_abs_max,
            "action_entropy": self.action_entropy,
            "value_pred_abs_max": self.value_pred_abs_max,
            "value_grad_norm_pre_clip": self.value_grad_norm_pre_clip,
            "value_grad_norm_post_clip": self.value_grad_norm_post_clip,
            "clip_applied_count": self.clip_applied_count,
            "first_nonfinite_component": self.first_nonfinite_component,
            "first_nonfinite_batch_context": self.first_nonfinite_batch_context,
        }

    @classmethod
    def from_mapping(cls, payload: dict[str, object] | None) -> "_Phase1ANumericsTelemetry":
        if payload is None:
            return cls()
        context = payload.get("first_nonfinite_batch_context")
        return cls(
            joint_ce_loss=float(payload.get("joint_ce_loss", 0.0) or 0.0),
            aux_value_loss_raw=float(payload.get("aux_value_loss_raw", 0.0) or 0.0),
            aux_value_loss_weighted=float(payload.get("aux_value_loss_weighted", 0.0) or 0.0),
            total_loss=float(payload.get("total_loss", 0.0) or 0.0),
            action_logit_abs_max=float(payload.get("action_logit_abs_max", 0.0) or 0.0),
            action_entropy=float(payload.get("action_entropy", 0.0) or 0.0),
            value_pred_abs_max=float(payload.get("value_pred_abs_max", 0.0) or 0.0),
            value_grad_norm_pre_clip=float(payload.get("value_grad_norm_pre_clip", 0.0) or 0.0),
            value_grad_norm_post_clip=float(payload.get("value_grad_norm_post_clip", 0.0) or 0.0),
            clip_applied_count=int(payload.get("clip_applied_count", 0) or 0),
            first_nonfinite_component=payload.get("first_nonfinite_component")
            if isinstance(payload.get("first_nonfinite_component"), str)
            else None,
            first_nonfinite_batch_context=dict(context) if isinstance(context, dict) else None,
        )

    def merge(self, other: "_Phase1ANumericsTelemetry") -> None:
        if other.joint_ce_loss > 0.0:
            self.joint_ce_loss = other.joint_ce_loss
        if other.aux_value_loss_raw > 0.0:
            self.aux_value_loss_raw = other.aux_value_loss_raw
        if other.aux_value_loss_weighted > 0.0:
            self.aux_value_loss_weighted = other.aux_value_loss_weighted
        if other.total_loss > 0.0:
            self.total_loss = other.total_loss
        self.action_logit_abs_max = max(self.action_logit_abs_max, other.action_logit_abs_max)
        if other.action_entropy > 0.0:
            self.action_entropy = other.action_entropy
        self.value_pred_abs_max = max(self.value_pred_abs_max, other.value_pred_abs_max)
        self.value_grad_norm_pre_clip = max(
            self.value_grad_norm_pre_clip,
            other.value_grad_norm_pre_clip,
        )
        self.value_grad_norm_post_clip = max(
            self.value_grad_norm_post_clip,
            other.value_grad_norm_post_clip,
        )
        self.clip_applied_count += other.clip_applied_count
        if self.first_nonfinite_component is None and other.first_nonfinite_component is not None:
            self.first_nonfinite_component = other.first_nonfinite_component
            self.first_nonfinite_batch_context = (
                dict(other.first_nonfinite_batch_context)
                if other.first_nonfinite_batch_context is not None
                else None
            )


@dataclass(slots=True, frozen=True)
class _Phase1ABatchStepResult:
    joint_ce_loss: float
    aux_value_loss_raw: float
    aux_value_loss_weighted: float
    total_loss: float
    action_logit_abs_max: float
    action_entropy: float
    numerics: _Phase1ANumericsTelemetry


@dataclass(slots=True)
class _Phase1ANumericsError(RuntimeError):
    component: str
    batch_context: dict[str, object]
    numerics: _Phase1ANumericsTelemetry | None = None

    def __str__(self) -> str:
        return (
            f"phase1a numerics failure component={self.component} "
            f"context={json.dumps(self.batch_context, sort_keys=True)}"
        )


@dataclass(frozen=True, slots=True)
class _Phase1ASearchOutputPaths:
    final_output: Path
    partial_manifest_path: Path
    partial_candidate_dir: Path
    checkpoint_root: Path
    search_state_path: Path


class LinearPolicyTrainer:
    def __init__(self, config: TrainingConfig, *, backend_name: TrainingBackendName = "pytorch"):
        self.config = config
        self._backend = _resolve_training_backend(backend_name)
        self.last_phase1a_profile_report: dict[str, object] | None = None

    def train(self, bundle: TrajectoryBundle, parent_policy_id: str | None = None) -> PolicyArtifact:
        """⚠️  FIXTURE / TEST COMPAT PATH — do not call from production code.

        Loads entire bundle.  Use train_search_from_directory() for production.
        """
        return self.train_search(bundle, parent_policy_id=parent_policy_id).selected_artifact

    def train_search(self, bundle: TrajectoryBundle, parent_policy_id: str | None = None) -> TrainingSearchResult:
        """⚠️  FIXTURE / TEST COMPAT PATH — do not call from production code.

        Loads entire bundle.  Use train_search_from_directory() for production.
        """
        if self.config.runtime_adapter == "linear-policy-v2":
            return self._train_search_phase1a_bundle(bundle, parent_policy_id=parent_policy_id)
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
                tensor_cache_used=False,
                jsonl_fallback_used=False,
                tensor_cache_format=None,
                tensor_cache_shard_count=None,
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

    def _trajectory_factory_from_bundle(
        self,
        bundle: TrajectoryBundle,
        split_name: Literal["train", "validation", "development", "final_untouched_test"],
    ) -> TrajectoryFactory:
        def _factory() -> list[TrajectoryRecord]:
            return [record.model_copy(deep=True) for record in bundle.splits[split_name]]

        return _factory

    def _trajectory_factory_from_window(
        self,
        directory: Path,
        window: StreamingWindow,
        *,
        store_cls: Any,
    ) -> TrajectoryFactory:
        def _factory() -> Any:
            return self._iter_window_records(directory, window, store_cls=store_cls)

        return _factory

    def _count_steps_from_factory(self, factory: TrajectoryFactory) -> int:
        return sum(len(record.steps) for record in factory())

    def _phase1a_streaming_batch_plan(
        self,
        *,
        feature_dim: int,
        joint_action_count: int,
        train_step_count: int,
    ) -> StreamingBatchPlan:
        bytes_per_example = (
            (feature_dim * np.dtype(np.float64).itemsize)
            + np.dtype(np.int64).itemsize
            + (joint_action_count * np.dtype(np.bool_).itemsize)
            + _PHASE1A_VALUE_LABEL_OVERHEAD_BYTES
        )
        effective_batch_size = max(
            1,
            min(_STREAMING_BATCH_MAX_SIZE, _STREAMING_BATCH_TARGET_BYTES // max(bytes_per_example, 1)),
        )
        estimated_batch_bytes = effective_batch_size * bytes_per_example
        batches_per_epoch = math.ceil(train_step_count / effective_batch_size)
        return StreamingBatchPlan(
            batch_target_bytes=_STREAMING_BATCH_TARGET_BYTES,
            bytes_per_example=int(bytes_per_example),
            effective_batch_size=int(effective_batch_size),
            estimated_batch_bytes=int(estimated_batch_bytes),
            batches_per_epoch=int(batches_per_epoch),
        )

    def _prepare_phase1a_training_data(
        self,
        *,
        train_factory: TrajectoryFactory,
        validation_factory: TrajectoryFactory,
        reward_spec: Any,
        action_space: ActionSpaceSpec,
        venue_choices: list[str],
    ) -> _Phase1APreparedData:
        stats = StreamingFeatureStats()
        reward_engine = RewardEngine(reward_spec, action_space)
        source_row_count = 0
        masked_row_count = 0
        train_step_count = 0
        horizon_steps = self.config.bootstrap_horizon_steps

        for trajectory in train_factory():
            rows = trajectory.steps
            current_policy_state = PolicyState()
            source_row_count += len(rows)
            for row_index, row in enumerate(rows):
                if not phase1a_label_available(
                    row_count=len(rows),
                    row_index=row_index,
                    horizon_steps=horizon_steps,
                ):
                    masked_row_count += len(rows) - row_index
                    break
                stats.update(
                    phase1a_feature_array(
                        row.observation,
                        current_policy_state,
                        venue_choices=venue_choices,
                        dtype=np.float64,
                    )
                )
                oracle = solve_phase1a_oracle(
                    rows=rows,
                    row_index=row_index,
                    horizon_steps=horizon_steps,
                    venue_choices=venue_choices,
                    reward_engine=reward_engine,
                    policy_state=current_policy_state,
                    preferred_size_band=self.config.preferred_size_band,
                    preferred_leverage_band=self.config.preferred_leverage_band,
                )
                applied = apply_phase1a_joint_action(
                    reward_engine=reward_engine,
                    row=row,
                    joint_action_key=oracle.joint_action_key,
                    policy_state=current_policy_state,
                    preferred_size_band=self.config.preferred_size_band,
                    preferred_leverage_band=self.config.preferred_leverage_band,
                )
                current_policy_state = reward_engine.advance_policy_state(current_policy_state, applied)
                train_step_count += 1

        if train_step_count <= 0:
            raise ValueError("phase1a training split yielded 0 supervised examples")
        feature_mean, feature_std = stats.finalize()
        val_step_count = self._count_steps_from_factory(validation_factory)
        if val_step_count <= 0:
            raise ValueError("validation split is empty")
        joint_action_keys = phase1a_joint_action_keys(venue_choices)
        batch_plan = self._phase1a_streaming_batch_plan(
            feature_dim=stats.feature_dim,
            joint_action_count=len(joint_action_keys),
            train_step_count=train_step_count,
        )
        logger.info(
            "phase1a_training_data_prepared train_examples=%d validation_examples=%d "
            "oracle_source_rows=%d oracle_masked_rows=%d label_coverage_ratio=%.4f "
            "feature_dim=%d effective_batch_size=%d estimated_batch_bytes=%d "
            "batches_per_epoch=%d batch_target_bytes=%d",
            train_step_count,
            val_step_count,
            source_row_count,
            masked_row_count,
            train_step_count / max(source_row_count, 1),
            stats.feature_dim,
            batch_plan.effective_batch_size,
            batch_plan.estimated_batch_bytes,
            batch_plan.batches_per_epoch,
            batch_plan.batch_target_bytes,
        )
        return _Phase1APreparedData(
            train_step_count=train_step_count,
            val_step_count=val_step_count,
            feature_mean=feature_mean,
            feature_std=feature_std,
            batch_plan=batch_plan,
            venue_choices=venue_choices,
            joint_action_keys=joint_action_keys,
            oracle_masked_row_count=masked_row_count,
            oracle_source_row_count=source_row_count,
        )

    def _count_window_steps_from_tensor_cache(
        self,
        *,
        directory: Path,
        cache_manifest: TensorCacheManifest,
        window: StreamingWindow,
    ) -> int:
        split_manifest = cache_manifest.splits.get(window.split_name)
        if split_manifest is None:
            raise ValueError(f"tensor cache is missing split {window.split_name!r}")
        total = 0
        for shard in split_manifest.shards:
            loaded = load_tensor_cache_shard(directory, shard)
            row_idx = window_row_indices(
                loaded.event_time_ms,
                start=window.start,
                end=window.end,
                exclusive_end=window.exclusive_end,
            )
            total += int(row_idx.shape[0])
        return total

    def _prepare_phase1a_training_data_from_tensor_cache(
        self,
        *,
        manifest: TrajectoryManifest,
        directory: Path,
        cache_manifest: TensorCacheManifest,
        supervision_manifest: Phase1ASupervisionManifest,
        train_window: StreamingWindow,
        validation_window: StreamingWindow,
    ) -> _Phase1APreparedData:
        stats = StreamingFeatureStats()
        source_row_count = 0
        masked_row_count = 0
        train_step_count = 0
        venue_choices = list(manifest.dataset_spec.exchanges)
        joint_action_keys = phase1a_joint_action_keys(venue_choices)
        split_manifest = cache_manifest.splits.get(train_window.split_name)
        supervision_split = supervision_manifest.splits.get(train_window.split_name)
        if split_manifest is None or supervision_split is None:
            raise ValueError(f"phase1a tensor cache train split {train_window.split_name!r} is unavailable")
        train_row_selections: list[_Phase1ATrainShardSelection] = []
        for cache_shard, supervision_shard in zip(split_manifest.shards, supervision_split.shards, strict=True):
            loaded_cache = load_tensor_cache_shard(directory, cache_shard)
            loaded_supervision = load_phase1a_supervision_shard(directory, supervision_shard)
            row_idx = window_row_indices(
                loaded_cache.event_time_ms,
                start=train_window.start,
                end=train_window.end,
                exclusive_end=train_window.exclusive_end,
            )
            source_row_count += int(row_idx.shape[0])
            if row_idx.size == 0:
                continue
            supervised_idx = row_idx[loaded_supervision.supervised_mask[row_idx]]
            masked_row_count += int(row_idx.shape[0] - supervised_idx.shape[0])
            if supervised_idx.size == 0:
                continue
            raw_features = np.concatenate(
                (
                    loaded_cache.features[supervised_idx].astype(np.float64, copy=False),
                    loaded_supervision.policy_state_features[supervised_idx].astype(np.float64, copy=False),
                ),
                axis=1,
            )
            stats.update_batch(raw_features)
            train_step_count += int(supervised_idx.shape[0])
            train_row_selections.append(
                _Phase1ATrainShardSelection(
                    directory=directory,
                    cache_shard=cache_shard,
                    supervision_shard=supervision_shard,
                    row_indices=np.asarray(supervised_idx, dtype=np.int64),
                )
            )
        if train_step_count <= 0:
            raise ValueError("phase1a training split yielded 0 supervised examples")
        feature_mean, feature_std = stats.finalize()
        val_step_count = self._count_window_steps_from_tensor_cache(
            directory=directory,
            cache_manifest=cache_manifest,
            window=validation_window,
        )
        if val_step_count <= 0:
            raise ValueError("validation split is empty")
        batch_plan = self._phase1a_streaming_batch_plan(
            feature_dim=stats.feature_dim,
            joint_action_count=len(joint_action_keys),
            train_step_count=train_step_count,
        )
        logger.info(
            "phase1a_tensor_cache_training_data_prepared train_examples=%d validation_examples=%d "
            "oracle_source_rows=%d oracle_masked_rows=%d label_coverage_ratio=%.4f "
            "feature_dim=%d effective_batch_size=%d estimated_batch_bytes=%d "
            "batches_per_epoch=%d batch_target_bytes=%d tensor_cache_used=true "
            "phase1a_supervision_used=true jsonl_fallback_used=false",
            train_step_count,
            val_step_count,
            source_row_count,
            masked_row_count,
            train_step_count / max(source_row_count, 1),
            stats.feature_dim,
            batch_plan.effective_batch_size,
            batch_plan.estimated_batch_bytes,
            batch_plan.batches_per_epoch,
            batch_plan.batch_target_bytes,
        )
        return _Phase1APreparedData(
            train_step_count=train_step_count,
            val_step_count=val_step_count,
            feature_mean=feature_mean,
            feature_std=feature_std,
            batch_plan=batch_plan,
            venue_choices=venue_choices,
            joint_action_keys=joint_action_keys,
            oracle_masked_row_count=masked_row_count,
            oracle_source_row_count=source_row_count,
            train_row_selections=train_row_selections,
            validation_window=validation_window,
            tensor_cache_used=True,
            phase1a_supervision_used=True,
        )

    def _train_phase1a_epoch(
        self,
        *,
        train_factory: TrajectoryFactory,
        prepared: _Phase1APreparedData,
        reward_spec: Any,
        action_space: ActionSpaceSpec,
        state: object,
        config: TrainingConfig,
        candidate_index: int,
        epoch: int,
    ) -> _Phase1AEpochMetrics:
        import time as _time

        batch_size = prepared.batch_plan.effective_batch_size
        reward_engine = RewardEngine(reward_spec, action_space)
        joint_action_to_index = {
            key: index for index, key in enumerate(prepared.joint_action_keys)
        }
        compute_dtype = np.float32 if config.phase1a_compute_dtype == "float32" else np.float64
        feature_batch = np.empty((batch_size, prepared.feature_dim), dtype=compute_dtype)
        joint_action_batch = np.empty(batch_size, dtype=np.int64)
        joint_mask_batch = np.empty((batch_size, len(prepared.joint_action_keys)), dtype=np.bool_)
        value_batch = np.empty(batch_size, dtype=compute_dtype)
        weighted_loss_total = 0.0
        joint_ce_loss_total = 0.0
        aux_value_loss_raw_total = 0.0
        aux_value_loss_weighted_total = 0.0
        action_entropy_total = 0.0
        action_logit_abs_max = 0.0
        seen = 0
        batch_row = 0
        batch_assembly_wall_sec = 0.0
        batch_compute_wall_sec = 0.0
        numerics = _Phase1ANumericsTelemetry()

        if prepared.train_row_selections is not None:
            for selection in prepared.train_row_selections:
                assembly_started_at = _time.perf_counter()
                loaded_cache = load_tensor_cache_shard(selection.directory, selection.cache_shard)
                loaded_supervision = load_phase1a_supervision_shard(selection.directory, selection.supervision_shard)
                batch_assembly_wall_sec += _time.perf_counter() - assembly_started_at
                row_indices = selection.row_indices
                if row_indices.size == 0:
                    continue
                for row_offset in range(0, int(row_indices.size), batch_size):
                    batch_indices = row_indices[row_offset : row_offset + batch_size]
                    assembly_started_at = _time.perf_counter()
                    observation_features = loaded_cache.features[batch_indices].astype(compute_dtype, copy=False)
                    policy_state_features = loaded_supervision.policy_state_features[batch_indices].astype(
                        compute_dtype,
                        copy=False,
                    )
                    full_features = np.concatenate((observation_features, policy_state_features), axis=1)
                    normalized = full_features.copy()
                    normalized -= prepared.feature_mean.astype(compute_dtype, copy=False)
                    normalized /= prepared.feature_std.astype(compute_dtype, copy=False)
                    joint_masks = loaded_supervision.joint_mask[batch_indices]
                    labels = loaded_supervision.joint_labels[batch_indices]
                    value_targets = loaded_supervision.value_targets[batch_indices].astype(compute_dtype, copy=False)
                    batch_assembly_wall_sec += _time.perf_counter() - assembly_started_at
                    compute_started_at = _time.perf_counter()
                    batch_result = _phase1a_batch_step(
                        backend=self._backend,
                        state=state,
                        batch_features=normalized,
                        batch_joint_action_labels=labels,
                        batch_joint_action_masks=joint_masks,
                        batch_value_targets=value_targets,
                        config=config,
                        batch_context={
                            "path_kind": "phase1a_supervision_tensor_cache",
                            "candidate_index": candidate_index,
                            "epoch": epoch,
                            "split_name": selection.cache_shard.split_name,
                            "shard_index": selection.cache_shard.shard_index,
                            "row_offset": int(row_offset),
                            "row_count": int(batch_indices.shape[0]),
                            "row_index_min": int(batch_indices[0]),
                            "row_index_max": int(batch_indices[-1]),
                            "batch_ordinal_in_shard": int(row_offset // batch_size),
                        },
                    )
                    batch_compute_wall_sec += _time.perf_counter() - compute_started_at
                    numerics.merge(batch_result.numerics)
                    action_logit_abs_max = max(action_logit_abs_max, batch_result.action_logit_abs_max)
                    joint_ce_loss_total += batch_result.joint_ce_loss * int(batch_indices.shape[0])
                    aux_value_loss_raw_total += batch_result.aux_value_loss_raw * int(batch_indices.shape[0])
                    aux_value_loss_weighted_total += batch_result.aux_value_loss_weighted * int(batch_indices.shape[0])
                    action_entropy_total += batch_result.action_entropy * int(batch_indices.shape[0])
                    weighted_loss_total += batch_result.total_loss * int(batch_indices.shape[0])
                    seen += int(batch_indices.shape[0])
        else:
            batch_context: dict[str, object] | None = None
            for trajectory in train_factory():
                rows = trajectory.steps
                current_policy_state = PolicyState()
                for row_index, row in enumerate(rows):
                    if not phase1a_label_available(
                        row_count=len(rows),
                        row_index=row_index,
                        horizon_steps=config.bootstrap_horizon_steps,
                    ):
                        break
                    assembly_started_at = _time.perf_counter()
                    raw_features = phase1a_feature_array(
                        row.observation,
                        current_policy_state,
                        venue_choices=prepared.venue_choices,
                        dtype=compute_dtype,
                    )
                    normalized = raw_features.copy()
                    normalized -= prepared.feature_mean.astype(compute_dtype, copy=False)
                    normalized /= prepared.feature_std.astype(compute_dtype, copy=False)
                    joint_mask = phase1a_joint_action_mask(
                        venue_choices=prepared.venue_choices,
                        action_feasibility=row.action_feasibility,
                        policy_state=current_policy_state,
                        preferred_size_band=config.preferred_size_band,
                        preferred_leverage_band=config.preferred_leverage_band,
                    )
                    oracle = solve_phase1a_oracle(
                        rows=rows,
                        row_index=row_index,
                        horizon_steps=config.bootstrap_horizon_steps,
                        venue_choices=prepared.venue_choices,
                        reward_engine=reward_engine,
                        policy_state=current_policy_state,
                        preferred_size_band=config.preferred_size_band,
                        preferred_leverage_band=config.preferred_leverage_band,
                    )
                    label_index = joint_action_to_index[oracle.joint_action_key]
                    if not bool(joint_mask[label_index]):
                        raise ValueError("phase1a oracle produced a label outside the legal joint action mask")
                    feature_batch[batch_row] = normalized
                    joint_action_batch[batch_row] = label_index
                    joint_mask_batch[batch_row] = joint_mask
                    value_batch[batch_row] = oracle.oracle_return
                    batch_row += 1
                    batch_context = {
                        "path_kind": "phase1a_streaming_jsonl",
                        "candidate_index": candidate_index,
                        "epoch": epoch,
                        "split_name": trajectory.split,
                        "trajectory_id": trajectory.trajectory_id,
                        "row_offset": int(row_index),
                        "row_count": int(batch_row),
                        "row_index_min": max(int(row_index - batch_row + 1), 0),
                        "row_index_max": int(row_index),
                    }
                    batch_assembly_wall_sec += _time.perf_counter() - assembly_started_at

                    applied = apply_phase1a_joint_action(
                        reward_engine=reward_engine,
                        row=row,
                        joint_action_key=oracle.joint_action_key,
                        policy_state=current_policy_state,
                        preferred_size_band=config.preferred_size_band,
                        preferred_leverage_band=config.preferred_leverage_band,
                    )
                    current_policy_state = reward_engine.advance_policy_state(current_policy_state, applied)

                    if batch_row == batch_size:
                        compute_started_at = _time.perf_counter()
                        batch_result = _phase1a_batch_step(
                            backend=self._backend,
                            state=state,
                            batch_features=feature_batch,
                            batch_joint_action_labels=joint_action_batch,
                            batch_joint_action_masks=joint_mask_batch,
                            batch_value_targets=value_batch,
                            config=config,
                            batch_context=batch_context,
                        )
                        batch_compute_wall_sec += _time.perf_counter() - compute_started_at
                        numerics.merge(batch_result.numerics)
                        action_logit_abs_max = max(action_logit_abs_max, batch_result.action_logit_abs_max)
                        joint_ce_loss_total += batch_result.joint_ce_loss * batch_row
                        aux_value_loss_raw_total += batch_result.aux_value_loss_raw * batch_row
                        aux_value_loss_weighted_total += batch_result.aux_value_loss_weighted * batch_row
                        action_entropy_total += batch_result.action_entropy * batch_row
                        weighted_loss_total += batch_result.total_loss * batch_row
                        seen += batch_row
                        batch_row = 0

        if batch_row > 0:
            compute_started_at = _time.perf_counter()
            batch_result = _phase1a_batch_step(
                backend=self._backend,
                state=state,
                batch_features=feature_batch[:batch_row],
                batch_joint_action_labels=joint_action_batch[:batch_row],
                batch_joint_action_masks=joint_mask_batch[:batch_row],
                batch_value_targets=value_batch[:batch_row],
                config=config,
                batch_context=batch_context,
            )
            batch_compute_wall_sec += _time.perf_counter() - compute_started_at
            numerics.merge(batch_result.numerics)
            action_logit_abs_max = max(action_logit_abs_max, batch_result.action_logit_abs_max)
            joint_ce_loss_total += batch_result.joint_ce_loss * batch_row
            aux_value_loss_raw_total += batch_result.aux_value_loss_raw * batch_row
            aux_value_loss_weighted_total += batch_result.aux_value_loss_weighted * batch_row
            action_entropy_total += batch_result.action_entropy * batch_row
            weighted_loss_total += batch_result.total_loss * batch_row
            seen += batch_row

        if seen <= 0:
            raise ValueError("phase1a train split is empty")
        numerics.joint_ce_loss = float(joint_ce_loss_total / seen)
        numerics.aux_value_loss_raw = float(aux_value_loss_raw_total / seen)
        numerics.aux_value_loss_weighted = float(aux_value_loss_weighted_total / seen)
        numerics.total_loss = float(weighted_loss_total / seen)
        numerics.action_logit_abs_max = action_logit_abs_max
        numerics.action_entropy = float(action_entropy_total / seen)
        return _Phase1AEpochMetrics(
            total_loss=numerics.total_loss,
            batch_assembly_wall_sec=batch_assembly_wall_sec,
            batch_compute_wall_sec=batch_compute_wall_sec,
            numerics=numerics,
        )

    def _train_candidate_phase1a(
        self,
        *,
        prepared: _Phase1APreparedData,
        train_factory: TrajectoryFactory,
        candidate_spec: TrainingCandidateSpec,
        candidate_index: int,
        reward_spec: Any,
        action_space: ActionSpaceSpec,
        validate_fn: Callable[[LinearPolicyV2Parameters, TrainingConfig], EvaluationReport],
    ) -> _CandidateTrainingRun:
        config = self.config.model_copy(
            update={
                "seed": candidate_spec.seed,
                "learning_rate": candidate_spec.learning_rate,
                "l2_weight": candidate_spec.l2_weight,
                "candidate_search": None,
            }
        )
        state = _phase1a_initialize_state(
            backend=self._backend,
            seed=config.seed,
            joint_action_count=len(prepared.joint_action_keys),
            feature_dim=prepared.feature_dim,
            compute_dtype=config.phase1a_compute_dtype,
        )
        fallback_parameters = _phase1a_parameters(
            backend=self._backend,
            state=state,
            joint_action_keys=prepared.joint_action_keys,
            venue_choices=prepared.venue_choices,
            feature_mean=prepared.feature_mean,
            feature_std=prepared.feature_std,
            config=config,
        )
        loss_history: list[float] = []
        validation_history: list[float] = []
        validation_wall_sec_history: list[float] = []
        best_validation_total_net_return: float | None = None
        best_parameters: LinearPolicyV2Parameters | None = None
        best_validation_score: PolicyScore | None = None
        best_epoch = 0
        batch_assembly_wall_sec = 0.0
        batch_compute_wall_sec = 0.0
        numerics = _Phase1ANumericsTelemetry()
        best_epoch_numerics: _Phase1ANumericsTelemetry | None = None

        for epoch in range(1, config.epochs + 1):
            import time as _time

            try:
                epoch_metrics = self._train_phase1a_epoch(
                    train_factory=train_factory,
                    prepared=prepared,
                    reward_spec=reward_spec,
                    action_space=action_space,
                    state=state,
                    config=config,
                    candidate_index=candidate_index,
                    epoch=epoch,
                )
            except _Phase1ANumericsError as exc:
                if exc.numerics is not None:
                    numerics.merge(exc.numerics)
                logger.warning(
                    "phase1a_training_candidate_numerics_failure candidate_index=%d seed=%d "
                    "learning_rate=%.6f l2_weight=%.6f epoch=%d component=%s batch_context=%s",
                    candidate_index,
                    candidate_spec.seed,
                    candidate_spec.learning_rate,
                    candidate_spec.l2_weight,
                    epoch,
                    exc.component,
                    json.dumps(exc.batch_context, sort_keys=True),
                )
                break
            total_loss = epoch_metrics.total_loss
            batch_assembly_wall_sec += epoch_metrics.batch_assembly_wall_sec
            batch_compute_wall_sec += epoch_metrics.batch_compute_wall_sec
            numerics.merge(epoch_metrics.numerics)
            loss_history.append(total_loss)
            parameters = _phase1a_parameters(
                backend=self._backend,
                state=state,
                joint_action_keys=prepared.joint_action_keys,
                venue_choices=prepared.venue_choices,
                feature_mean=prepared.feature_mean,
                feature_std=prepared.feature_std,
                config=config,
            )
            if not _phase1a_parameters_are_finite(parameters):
                logger.warning(
                    "phase1a_training_candidate_non_finite_parameters candidate_index=%d seed=%d "
                    "learning_rate=%.6f l2_weight=%.6f epoch=%d",
                    candidate_index,
                    candidate_spec.seed,
                    candidate_spec.learning_rate,
                    candidate_spec.l2_weight,
                    epoch,
                )
                break
            validation_started_at = _time.perf_counter()
            validation_report = validate_fn(parameters, config)
            validation_wall_sec_history.append(_time.perf_counter() - validation_started_at)
            validation_history.append(validation_report.total_net_return)
            validation_score = PolicyScorer().score(validation_report)
            if (
                best_validation_total_net_return is None
                or validation_report.total_net_return > best_validation_total_net_return
            ):
                best_validation_total_net_return = validation_report.total_net_return
                best_parameters = parameters
                best_validation_score = validation_score
                best_epoch = epoch
                best_epoch_numerics = _Phase1ANumericsTelemetry.from_mapping(epoch_metrics.numerics.as_dict())

        if best_parameters is None:
            import time as _time

            logger.warning(
                "phase1a_training_candidate_falling_back_to_initial_parameters candidate_index=%d seed=%d "
                "learning_rate=%.6f l2_weight=%.6f",
                candidate_index,
                candidate_spec.seed,
                candidate_spec.learning_rate,
                candidate_spec.l2_weight,
            )
            validation_started_at = _time.perf_counter()
            validation_report = validate_fn(fallback_parameters, config)
            validation_wall_sec_history.append(_time.perf_counter() - validation_started_at)
            validation_history.append(validation_report.total_net_return)
            best_parameters = fallback_parameters
            best_validation_total_net_return = validation_report.total_net_return
            best_validation_score = PolicyScorer().score(validation_report)
            best_epoch = 0

        assert best_validation_total_net_return is not None
        assert best_validation_score is not None
        logger.info(
            "phase1a_training_candidate_completed candidate_index=%d seed=%d learning_rate=%.6f "
            "l2_weight=%.6f best_epoch=%d best_validation_total_net_return=%.6f "
            "best_validation_composite_rank=%.6f training_backend=%s",
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
            validation_wall_sec_history=validation_wall_sec_history,
            batch_assembly_wall_sec=batch_assembly_wall_sec,
            batch_compute_wall_sec=batch_compute_wall_sec,
            numerics=best_epoch_numerics or numerics,
        )

    def _phase1a_validation_report_from_factory(
        self,
        *,
        dataset_spec: Any,
        reward_spec: Any,
        validation_factory: TrajectoryFactory,
        artifact: PolicyArtifact,
    ) -> EvaluationReport:
        from quantlab_ml.evaluation import EvaluationEngine

        engine = EvaluationEngine(self._evaluation_boundary(reward_spec.timestamping))
        return engine.evaluate_records(
            dataset_spec,
            reward_spec,
            validation_factory(),
            artifact,
        )

    def _phase1a_validation_report_for_window(
        self,
        *,
        manifest: TrajectoryManifest,
        directory: Path,
        artifact: PolicyArtifact,
        window: StreamingWindow,
    ) -> EvaluationReport:
        from quantlab_ml.evaluation import EvaluationEngine

        engine = EvaluationEngine(self._evaluation_boundary(manifest.reward_spec.timestamping))
        report = engine.evaluate_directory(
            manifest=manifest,
            directory=directory,
            artifact=artifact,
            split_name=window.split_name,
            start=window.start,
            end=window.end,
            exclusive_end=window.exclusive_end,
        )
        self.last_phase1a_profile_report = engine.last_phase1a_profile_report
        return report

    def _prepare_training_data(self, bundle: TrajectoryBundle) -> _PreparedTrainingData:
        """⚠️  FIXTURE / TEST COMPAT PATH — matrix-first helper wrapper."""

        prepared = compat_matrix_first.prepare_training_data(bundle)
        logger.info(
            "training_data_prepared train_examples=%d validation_examples=%d "
            "feature_dim=%d action_count=%d venue_count=%d path_classification=temporary_compatibility_maintenance",
            prepared.train_step_count,
            prepared.val_step_count,
            prepared.feature_dim,
            len(prepared.action_keys),
            len(prepared.venue_choices),
        )
        return prepared

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

    def _train_search_phase1a_bundle(
        self,
        bundle: TrajectoryBundle,
        *,
        parent_policy_id: str | None,
    ) -> TrainingSearchResult:
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
            "phase1a_training_search_started training_run_id=%s candidate_count=%d split_version=%s "
            "reward_version=%s training_backend=%s training_device=%s cuda_available=%s device_name=%s",
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
            self._select_candidate_via_walkforward_phase1a_bundle(
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
        train_factory = self._trajectory_factory_from_bundle(bundle, "train")
        validation_factory = self._trajectory_factory_from_bundle(bundle, "validation")
        prepared = self._prepare_phase1a_training_data(
            train_factory=train_factory,
            validation_factory=validation_factory,
            reward_spec=bundle.reward_spec,
            action_space=bundle.action_space,
            venue_choices=bundle.dataset_spec.exchanges,
        )

        candidate_results: list[TrainingCandidateResult] = []
        for candidate_rank, selection_run in enumerate(ranked_selections, start=1):
            selected_candidate = candidate_rank == 1

            def _validate(parameters: LinearPolicyV2Parameters, config: TrainingConfig) -> EvaluationReport:
                artifact = self._build_artifact(
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
                return self._phase1a_validation_report_from_factory(
                    dataset_spec=bundle.dataset_spec,
                    reward_spec=bundle.reward_spec,
                    validation_factory=validation_factory,
                    artifact=artifact,
                )

            candidate_run = self._train_candidate_phase1a(
                prepared=prepared,
                train_factory=train_factory,
                candidate_spec=selection_run.candidate_spec,
                candidate_index=selection_run.candidate_index,
                reward_spec=bundle.reward_spec,
                action_space=bundle.action_space,
                validate_fn=_validate,
            )
            candidate_summary = self._candidate_training_summary(
                prepared=prepared,
                candidate_run=candidate_run,
                selection_run=selection_run,
                training_run_id=training_run_id,
                candidate_rank=candidate_rank,
                selected_candidate=selected_candidate,
                search_budget_summary=search_budget_summary,
                training_data_flow="phase1a_oracle_streaming",
                validation_data_flow="phase1a_bundle_evaluation",
                normalization_strategy="phase1a_train_only_streaming",
                proxy_validation_used=False,
                tensor_cache_used=False,
                jsonl_fallback_used=False,
                tensor_cache_format=None,
                tensor_cache_shard_count=None,
                batch_plan=prepared.batch_plan,
            )
            candidate_summary.update(
                {
                    "bootstrap_horizon_steps": self.config.bootstrap_horizon_steps,
                    "aux_value_loss_weight": self.config.aux_value_loss_weight,
                    "policy_state_feature_version": self.config.policy_state_feature_version,
                    "joint_action_vocabulary_version": self.config.joint_action_vocabulary_version,
                    "oracle_masked_row_count": prepared.oracle_masked_row_count,
                    "oracle_source_row_count": prepared.oracle_source_row_count,
                    "oracle_label_coverage_ratio": prepared.oracle_label_coverage_ratio,
                }
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
            "phase1a_training_search_completed training_run_id=%s selected_policy_id=%s candidate_count=%d "
            "selected_validation_total_net_return=%.6f selected_validation_composite_rank=%.6f",
            training_run_id,
            result.selected_artifact.policy_id,
            len(candidate_results),
            candidate_results[0].best_validation_total_net_return,
            candidate_results[0].best_validation_composite_rank,
        )
        return result

    def _select_candidate_via_walkforward_phase1a_bundle(
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
            train_factory = self._trajectory_factory_from_bundle(fold_bundle, "train")
            validation_factory = self._trajectory_factory_from_bundle(fold_bundle, "validation")
            try:
                prepared = self._prepare_phase1a_training_data(
                    train_factory=train_factory,
                    validation_factory=validation_factory,
                    reward_spec=fold_bundle.reward_spec,
                    action_space=fold_bundle.action_space,
                    venue_choices=fold_bundle.dataset_spec.exchanges,
                )
            except ValueError as exc:
                if "0 supervised examples" not in str(exc):
                    raise
                logger.warning(
                    "phase1a_walkforward_fold_skipped fold_id=%s reason=no_supervised_examples "
                    "bootstrap_horizon_steps=%d",
                    fold.fold_id,
                    self.config.bootstrap_horizon_steps,
                )
                continue

            def _validate(parameters: LinearPolicyV2Parameters, config: TrainingConfig) -> EvaluationReport:
                artifact = self._build_artifact(
                    bundle=fold_bundle,
                    config=config,
                    training_run_id=f"{training_run_id}:{fold.fold_id}",
                    code_commit_hash=code_commit_hash,
                    parameters=parameters,
                    parent_policy_id=parent_policy_id,
                    validation_total_net_return=0.0,
                    validation_score=None,
                    training_summary={},
                    search_metadata=None,
                )
                return self._phase1a_validation_report_from_factory(
                    dataset_spec=fold_bundle.dataset_spec,
                    reward_spec=fold_bundle.reward_spec,
                    validation_factory=validation_factory,
                    artifact=artifact,
                )

            fold_run = self._train_candidate_phase1a(
                prepared=prepared,
                train_factory=train_factory,
                candidate_spec=candidate_spec,
                candidate_index=candidate_index,
                reward_spec=fold_bundle.reward_spec,
                action_space=fold_bundle.action_space,
                validate_fn=_validate,
            )
            fold_scores.append(
                FoldValidationScore(
                    fold_id=fold.fold_id,
                    validation_total_net_return=fold_run.best_validation_total_net_return,
                    validation_composite_rank=fold_run.best_validation_score.composite_rank,
                    validation_step_count=prepared.val_step_count,
                )
            )
        if not fold_scores:
            logger.warning(
                "phase1a_walkforward_no_qualifying_folds fallback=train_validation "
                "bootstrap_horizon_steps=%d",
                self.config.bootstrap_horizon_steps,
            )
            train_factory = self._trajectory_factory_from_bundle(bundle, "train")
            validation_factory = self._trajectory_factory_from_bundle(bundle, "validation")
            prepared = self._prepare_phase1a_training_data(
                train_factory=train_factory,
                validation_factory=validation_factory,
                reward_spec=bundle.reward_spec,
                action_space=bundle.action_space,
                venue_choices=bundle.dataset_spec.exchanges,
            )

            def _validate(parameters: LinearPolicyV2Parameters, config: TrainingConfig) -> EvaluationReport:
                artifact = self._build_artifact(
                    bundle=bundle,
                    config=config,
                    training_run_id=f"{training_run_id}:fallback",
                    code_commit_hash=code_commit_hash,
                    parameters=parameters,
                    parent_policy_id=parent_policy_id,
                    validation_total_net_return=0.0,
                    validation_score=None,
                    training_summary={},
                    search_metadata=None,
                )
                return self._phase1a_validation_report_from_factory(
                    dataset_spec=bundle.dataset_spec,
                    reward_spec=bundle.reward_spec,
                    validation_factory=validation_factory,
                    artifact=artifact,
                )

            fold_run = self._train_candidate_phase1a(
                prepared=prepared,
                train_factory=train_factory,
                candidate_spec=candidate_spec,
                candidate_index=candidate_index,
                reward_spec=bundle.reward_spec,
                action_space=bundle.action_space,
                validate_fn=_validate,
            )
            fold_scores.append(
                FoldValidationScore(
                    fold_id="fallback-train-validation",
                    validation_total_net_return=fold_run.best_validation_total_net_return,
                    validation_composite_rank=fold_run.best_validation_score.composite_rank,
                    validation_step_count=prepared.val_step_count,
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
            "phase1a_training_candidate_walkforward_completed candidate_index=%d seed=%d "
            "learning_rate=%.6f l2_weight=%.6f fold_count=%d selection_total_net_return=%.6f "
            "selection_composite_rank=%.6f",
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

    def _train_search_from_directory_phase1a(
        self,
        *,
        manifest: TrajectoryManifest,
        directory: Path,
        store_cls: Any,
        parent_policy_id: str | None,
    ) -> TrainingSearchResult:
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
            "phase1a_training_search_started training_run_id=%s candidate_count=%d split_version=%s "
            "reward_version=%s training_backend=%s training_device=%s cuda_available=%s device_name=%s "
            "tensor_cache_used=false jsonl_fallback_used=false",
            training_run_id,
            len(candidate_specs),
            manifest.split_artifact.split_version,
            manifest.reward_spec.reward_version,
            self._backend.backend_name,
            self._backend.device_resolution.training_device,
            self._backend.device_resolution.cuda_available,
            self._backend.device_resolution.device_name,
        )
        selection_runs = [
            self._select_candidate_via_walkforward_phase1a_streaming(
                manifest=manifest,
                directory=directory,
                store_cls=store_cls,
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
        train_factory = self._trajectory_factory_from_window(directory, final_train_window, store_cls=store_cls)
        validation_factory = self._trajectory_factory_from_window(
            directory,
            final_validation_window,
            store_cls=store_cls,
        )
        prepared = self._prepare_phase1a_training_data(
            train_factory=train_factory,
            validation_factory=validation_factory,
            reward_spec=manifest.reward_spec,
            action_space=manifest.action_space,
            venue_choices=manifest.dataset_spec.exchanges,
        )

        candidate_results: list[TrainingCandidateResult] = []
        for candidate_rank, selection_run in enumerate(ranked_selections, start=1):
            selected_candidate = candidate_rank == 1

            def _validate(parameters: LinearPolicyV2Parameters, config: TrainingConfig) -> EvaluationReport:
                artifact = self._build_artifact_from_manifest(
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
                return self._phase1a_validation_report_from_factory(
                    dataset_spec=manifest.dataset_spec,
                    reward_spec=manifest.reward_spec,
                    validation_factory=validation_factory,
                    artifact=artifact,
                )

            candidate_run = self._train_candidate_phase1a(
                prepared=prepared,
                train_factory=train_factory,
                candidate_spec=selection_run.candidate_spec,
                candidate_index=selection_run.candidate_index,
                reward_spec=manifest.reward_spec,
                action_space=manifest.action_space,
                validate_fn=_validate,
            )
            candidate_summary = self._candidate_training_summary(
                prepared=prepared,
                candidate_run=candidate_run,
                selection_run=selection_run,
                training_run_id=training_run_id,
                candidate_rank=candidate_rank,
                selected_candidate=selected_candidate,
                search_budget_summary=search_budget_summary,
                training_data_flow="phase1a_oracle_streaming",
                validation_data_flow="phase1a_streaming_evaluation",
                normalization_strategy="phase1a_train_only_streaming",
                proxy_validation_used=False,
                tensor_cache_used=False,
                jsonl_fallback_used=False,
                tensor_cache_format=None,
                tensor_cache_shard_count=None,
                batch_plan=prepared.batch_plan,
            )
            candidate_summary.update(
                {
                    "bootstrap_horizon_steps": self.config.bootstrap_horizon_steps,
                    "aux_value_loss_weight": self.config.aux_value_loss_weight,
                    "policy_state_feature_version": self.config.policy_state_feature_version,
                    "joint_action_vocabulary_version": self.config.joint_action_vocabulary_version,
                    "oracle_masked_row_count": prepared.oracle_masked_row_count,
                    "oracle_source_row_count": prepared.oracle_source_row_count,
                    "oracle_label_coverage_ratio": prepared.oracle_label_coverage_ratio,
                }
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
            "phase1a_training_search_completed training_run_id=%s selected_policy_id=%s candidate_count=%d "
            "selected_validation_total_net_return=%.6f selected_validation_composite_rank=%.6f",
            training_run_id,
            result.selected_artifact.policy_id,
            len(candidate_results),
            candidate_results[0].best_validation_total_net_return,
            candidate_results[0].best_validation_composite_rank,
        )
        return result

    def _phase1a_search_output_paths(self, output: Path) -> _Phase1ASearchOutputPaths:
        resolved_output = output.expanduser().resolve()
        return _Phase1ASearchOutputPaths(
            final_output=resolved_output,
            partial_manifest_path=resolved_output.with_name(f"{resolved_output.stem}_search.partial.json"),
            partial_candidate_dir=resolved_output.with_name(f"{resolved_output.stem}_candidates_partial"),
            checkpoint_root=resolved_output.parent / "checkpoints",
            search_state_path=(resolved_output.parent / "checkpoints" / "phase1a_search_state.json"),
        )

    def _phase1a_resume_compatibility(
        self,
        *,
        cache_manifest: TensorCacheManifest,
        supervision_manifest: Phase1ASupervisionManifest,
    ) -> dict[str, object]:
        return {
            "tensor_cache_manifest_hash": hash_payload(cache_manifest),
            "phase1a_supervision_manifest_hash": hash_payload(supervision_manifest),
            "training_config_hash": hash_payload(self.config),
            "phase1a_compute_dtype": self.config.phase1a_compute_dtype,
            "action_space_version": ACTION_SPACE_VERSION_V2_PHASE1A,
            "policy_state_feature_version": self.config.policy_state_feature_version,
            "reward_version": supervision_manifest.reward_version,
            "bootstrap_horizon_steps": self.config.bootstrap_horizon_steps,
        }

    def _load_phase1a_search_state(
        self,
        *,
        paths: _Phase1ASearchOutputPaths | None,
        resume_search: bool,
        compatibility: dict[str, object],
    ) -> dict[str, object]:
        if paths is None:
            return {
                "compatibility": compatibility,
                "selection_runs": {},
                "candidate_results": [],
            }
        if paths.search_state_path.exists():
            if not resume_search:
                raise ValueError(
                    f"phase1a partial search state already exists at {paths.search_state_path}; "
                    "pass --resume-search or choose a new --output path"
                )
            payload = json.loads(paths.search_state_path.read_text(encoding="utf-8"))
            if payload.get("compatibility") != compatibility:
                raise ValueError(
                    "phase1a resume compatibility mismatch; refusing to resume with changed tensor cache, "
                    "supervision manifest, training config, dtype, or action-space metadata"
                )
            return payload
        return {
            "compatibility": compatibility,
            "selection_runs": {},
            "candidate_results": [],
        }

    def _write_phase1a_search_state(
        self,
        *,
        paths: _Phase1ASearchOutputPaths | None,
        payload: dict[str, object],
    ) -> None:
        if paths is None:
            return
        _write_json_atomic(paths.search_state_path, payload)

    def _write_phase1a_partial_manifest(
        self,
        *,
        paths: _Phase1ASearchOutputPaths | None,
        training_run_id: str,
        search_budget_summary: SearchBudgetSummary,
        candidate_results: list[dict[str, object]],
    ) -> None:
        if paths is None:
            return
        selected_policy_id = None
        if candidate_results:
            ranked = sorted(candidate_results, key=lambda item: int(item["candidate_rank"]))
            selected_policy_id = str(ranked[0]["policy_id"])
        _write_json_atomic(
            paths.partial_manifest_path,
            {
                "status": "partial",
                "training_run_id": training_run_id,
                "selected_policy_id": selected_policy_id,
                "search_budget_summary": search_budget_summary.model_dump(mode="json"),
                "candidates": candidate_results,
            },
        )

    def _write_phase1a_fold_checkpoint(
        self,
        *,
        paths: _Phase1ASearchOutputPaths | None,
        compatibility: dict[str, object],
        candidate_index: int,
        fold_id: str,
        candidate_spec: TrainingCandidateSpec,
        fold_score: FoldValidationScore,
        fold_wall_sec: float,
    ) -> None:
        if paths is None:
            return
        checkpoint_path = (
            paths.checkpoint_root
            / "selection"
            / f"candidate_{candidate_index}"
            / f"{fold_id}.json"
        )
        _write_json_atomic(
            checkpoint_path,
            {
                "compatibility": compatibility,
                "candidate_index": candidate_index,
                "candidate_spec": candidate_spec.as_dict(),
                "fold_score": fold_score.as_dict(),
                "fold_wall_sec": fold_wall_sec,
            },
        )

    def _load_phase1a_fold_checkpoint(
        self,
        *,
        paths: _Phase1ASearchOutputPaths | None,
        compatibility: dict[str, object],
        candidate_index: int,
        fold_id: str,
    ) -> tuple[FoldValidationScore, float] | None:
        if paths is None:
            return None
        checkpoint_path = (
            paths.checkpoint_root
            / "selection"
            / f"candidate_{candidate_index}"
            / f"{fold_id}.json"
        )
        if not checkpoint_path.exists():
            return None
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if payload.get("compatibility") != compatibility:
            raise ValueError(
                f"phase1a fold checkpoint compatibility mismatch at {checkpoint_path}"
            )
        fold_payload = payload["fold_score"]
        return (
            FoldValidationScore(
                fold_id=str(fold_payload["fold_id"]),
                validation_total_net_return=float(fold_payload["validation_total_net_return"]),
                validation_composite_rank=float(fold_payload["validation_composite_rank"]),
                validation_step_count=int(fold_payload["validation_step_count"]),
            ),
            float(payload.get("fold_wall_sec", 0.0)),
        )

    def _train_search_from_directory_phase1a_tensor_cache(
        self,
        *,
        manifest: TrajectoryManifest,
        directory: Path,
        cache_manifest: TensorCacheManifest,
        supervision_manifest: Phase1ASupervisionManifest,
        parent_policy_id: str | None,
        search_output: Path | None,
        resume_search: bool,
    ) -> TrainingSearchResult:
        import time as _time

        candidate_specs = self._candidate_specs()
        code_commit_hash = current_code_commit_hash()
        training_run_id = self._training_run_id_from_manifest(
            manifest,
            parent_policy_id=parent_policy_id,
            code_commit_hash=code_commit_hash,
            candidate_specs=candidate_specs,
        )
        search_budget_summary = _search_budget_summary(candidate_specs)
        tensor_cache_shard_count = sum(split.shard_count for split in cache_manifest.splits.values())
        compatibility = self._phase1a_resume_compatibility(
            cache_manifest=cache_manifest,
            supervision_manifest=supervision_manifest,
        )
        paths = self._phase1a_search_output_paths(search_output) if search_output is not None else None
        search_state = self._load_phase1a_search_state(
            paths=paths,
            resume_search=resume_search,
            compatibility=compatibility,
        )
        logger.info(
            "phase1a_training_search_started training_run_id=%s candidate_count=%d split_version=%s "
            "reward_version=%s training_backend=%s training_device=%s cuda_available=%s device_name=%s "
            "tensor_cache_used=true phase1a_supervision_used=true jsonl_fallback_used=false "
            "tensor_cache_format=%s tensor_cache_shard_count=%d",
            training_run_id,
            len(candidate_specs),
            manifest.split_artifact.split_version,
            manifest.reward_spec.reward_version,
            self._backend.backend_name,
            self._backend.device_resolution.training_device,
            self._backend.device_resolution.cuda_available,
            self._backend.device_resolution.device_name,
            cache_manifest.format_version,
            tensor_cache_shard_count,
        )

        interval = timedelta(seconds=manifest.dataset_spec.sampling_interval_seconds)
        prepared_windows: list[tuple[str, _Phase1APreparedData]] = []
        for fold in manifest.split_artifact.folds:
            purge_cutoff = fold.validation_window.start - (interval * fold.purge_width_steps)
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
            try:
                prepared = self._prepare_phase1a_training_data_from_tensor_cache(
                    manifest=manifest,
                    directory=directory,
                    cache_manifest=cache_manifest,
                    supervision_manifest=supervision_manifest,
                    train_window=train_window,
                    validation_window=validation_window,
                )
            except ValueError as exc:
                if "0 supervised examples" not in str(exc):
                    raise
                logger.warning(
                    "phase1a_walkforward_fold_skipped fold_id=%s reason=no_supervised_examples "
                    "bootstrap_horizon_steps=%d",
                    fold.fold_id,
                    self.config.bootstrap_horizon_steps,
                )
                continue
            prepared_windows.append((fold.fold_id, prepared))
        if not prepared_windows:
            logger.warning(
                "phase1a_walkforward_no_qualifying_folds fallback=train_validation "
                "bootstrap_horizon_steps=%d",
                self.config.bootstrap_horizon_steps,
            )
            prepared_windows.append(
                (
                    "fallback-train-validation",
                    self._prepare_phase1a_training_data_from_tensor_cache(
                        manifest=manifest,
                        directory=directory,
                        cache_manifest=cache_manifest,
                        supervision_manifest=supervision_manifest,
                        train_window=StreamingWindow(split_name="train"),
                        validation_window=StreamingWindow(split_name="validation"),
                    ),
                )
            )

        fold_wall_sec_history: list[float] = []
        selection_runs: list[_CandidateSelectionRun] = []
        selection_state = dict(search_state.get("selection_runs", {}))
        for candidate_index, candidate_spec in enumerate(candidate_specs):
            cached_selection = selection_state.get(str(candidate_index))
            if cached_selection is not None:
                selection_runs.append(
                    _CandidateSelectionRun(
                        candidate_spec=candidate_spec,
                        candidate_index=candidate_index,
                        fold_scores=[
                            FoldValidationScore(
                                fold_id=str(item["fold_id"]),
                                validation_total_net_return=float(item["validation_total_net_return"]),
                                validation_composite_rank=float(item["validation_composite_rank"]),
                                validation_step_count=int(item["validation_step_count"]),
                            )
                            for item in cached_selection["fold_scores"]
                        ],
                        selection_total_net_return=float(cached_selection["selection_total_net_return"]),
                        selection_composite_rank=float(cached_selection["selection_composite_rank"]),
                    )
                )
                continue
            logger.info(
                "[PROGRESS] marker=candidate_started stage=selection candidate_index=%d seed=%d "
                "learning_rate=%.6f l2_weight=%.6f",
                candidate_index,
                candidate_spec.seed,
                candidate_spec.learning_rate,
                candidate_spec.l2_weight,
            )
            fold_scores: list[FoldValidationScore] = []
            for fold_id, prepared in prepared_windows:
                restored = self._load_phase1a_fold_checkpoint(
                    paths=paths,
                    compatibility=compatibility,
                    candidate_index=candidate_index,
                    fold_id=fold_id,
                )
                if restored is not None:
                    fold_score, fold_wall_sec = restored
                    fold_scores.append(fold_score)
                    fold_wall_sec_history.append(fold_wall_sec)
                    continue
                fold_started_at = _time.perf_counter()

                def _validate(parameters: LinearPolicyV2Parameters, config: TrainingConfig) -> EvaluationReport:
                    artifact = self._build_artifact_from_manifest(
                        manifest=manifest,
                        config=config,
                        training_run_id=f"{training_run_id}:{fold_id}",
                        code_commit_hash=code_commit_hash,
                        parameters=parameters,
                        parent_policy_id=parent_policy_id,
                        validation_total_net_return=0.0,
                        validation_score=None,
                        training_summary={},
                        search_metadata=None,
                        validation_step_count=prepared.val_step_count,
                    )
                    assert prepared.validation_window is not None
                    return self._phase1a_validation_report_for_window(
                        manifest=manifest,
                        directory=directory,
                        artifact=artifact,
                        window=prepared.validation_window,
                    )

                fold_run = self._train_candidate_phase1a(
                    prepared=prepared,
                    train_factory=lambda: (),
                    candidate_spec=candidate_spec,
                    candidate_index=candidate_index,
                    reward_spec=manifest.reward_spec,
                    action_space=manifest.action_space,
                    validate_fn=_validate,
                )
                fold_score = FoldValidationScore(
                    fold_id=fold_id,
                    validation_total_net_return=fold_run.best_validation_total_net_return,
                    validation_composite_rank=fold_run.best_validation_score.composite_rank,
                    validation_step_count=prepared.val_step_count,
                )
                fold_wall_sec = _time.perf_counter() - fold_started_at
                fold_scores.append(fold_score)
                fold_wall_sec_history.append(fold_wall_sec)
                self._write_phase1a_fold_checkpoint(
                    paths=paths,
                    compatibility=compatibility,
                    candidate_index=candidate_index,
                    fold_id=fold_id,
                    candidate_spec=candidate_spec,
                    fold_score=fold_score,
                    fold_wall_sec=fold_wall_sec,
                )
                logger.info(
                    "[PROGRESS] marker=fold_completed candidate_index=%d fold_id=%s fold_wall_sec=%.3f",
                    candidate_index,
                    fold_id,
                    fold_wall_sec,
                )
            selection_total_net_return = _weighted_mean(
                [score.validation_total_net_return for score in fold_scores],
                [score.validation_step_count for score in fold_scores],
            )
            selection_composite_rank = _weighted_mean(
                [score.validation_composite_rank for score in fold_scores],
                [score.validation_step_count for score in fold_scores],
            )
            selection_run = _CandidateSelectionRun(
                candidate_spec=candidate_spec,
                candidate_index=candidate_index,
                fold_scores=fold_scores,
                selection_total_net_return=selection_total_net_return,
                selection_composite_rank=selection_composite_rank,
            )
            selection_state[str(candidate_index)] = {
                "fold_scores": [score.as_dict() for score in fold_scores],
                "selection_total_net_return": selection_total_net_return,
                "selection_composite_rank": selection_composite_rank,
            }
            search_state["selection_runs"] = selection_state
            self._write_phase1a_search_state(paths=paths, payload=search_state)
            selection_runs.append(selection_run)

        ranked_selections = sorted(selection_runs, key=_selection_ranking_key)
        final_prepared = self._prepare_phase1a_training_data_from_tensor_cache(
            manifest=manifest,
            directory=directory,
            cache_manifest=cache_manifest,
            supervision_manifest=supervision_manifest,
            train_window=StreamingWindow(split_name="train"),
            validation_window=StreamingWindow(split_name="validation"),
        )
        partial_candidate_entries = list(search_state.get("candidate_results", []))
        partial_candidate_by_index = {int(entry["candidate_index"]): entry for entry in partial_candidate_entries}
        candidate_results: list[TrainingCandidateResult] = []
        candidate_wall_sec_history: list[float] = []
        batch_assembly_wall_sec = 0.0
        batch_compute_wall_sec = 0.0
        numerics = _Phase1ANumericsTelemetry()

        for candidate_rank, selection_run in enumerate(ranked_selections, start=1):
            cached_result = partial_candidate_by_index.get(selection_run.candidate_index)
            if cached_result is not None:
                artifact = load_model(Path(str(cached_result["artifact_path"])), PolicyArtifact)
                numerics.merge(_Phase1ANumericsTelemetry.from_mapping(artifact.training_summary))
                candidate_results.append(
                    TrainingCandidateResult(
                        artifact=artifact,
                        candidate_index=selection_run.candidate_index,
                        candidate_rank=int(cached_result["candidate_rank"]),
                        selected_candidate=bool(cached_result["selected_candidate"]),
                        candidate_spec=selection_run.candidate_spec,
                        best_validation_total_net_return=float(cached_result["best_validation_total_net_return"]),
                        best_validation_composite_rank=float(cached_result["best_validation_composite_rank"]),
                    )
                )
                if "candidate_wall_sec" in cached_result:
                    candidate_wall_sec_history.append(float(cached_result["candidate_wall_sec"]))
                continue
            selected_candidate = candidate_rank == 1
            logger.info(
                "[PROGRESS] marker=candidate_started stage=refit candidate_index=%d candidate_rank=%d seed=%d "
                "learning_rate=%.6f l2_weight=%.6f",
                selection_run.candidate_index,
                candidate_rank,
                selection_run.candidate_spec.seed,
                selection_run.candidate_spec.learning_rate,
                selection_run.candidate_spec.l2_weight,
            )
            candidate_started_at = _time.perf_counter()

            def _validate(parameters: LinearPolicyV2Parameters, config: TrainingConfig) -> EvaluationReport:
                artifact = self._build_artifact_from_manifest(
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
                    validation_step_count=final_prepared.val_step_count,
                )
                assert final_prepared.validation_window is not None
                return self._phase1a_validation_report_for_window(
                    manifest=manifest,
                    directory=directory,
                    artifact=artifact,
                    window=final_prepared.validation_window,
                )

            candidate_run = self._train_candidate_phase1a(
                prepared=final_prepared,
                train_factory=lambda: (),
                candidate_spec=selection_run.candidate_spec,
                candidate_index=selection_run.candidate_index,
                reward_spec=manifest.reward_spec,
                action_space=manifest.action_space,
                validate_fn=_validate,
            )
            batch_assembly_wall_sec += candidate_run.batch_assembly_wall_sec
            batch_compute_wall_sec += candidate_run.batch_compute_wall_sec
            if candidate_run.numerics is not None:
                numerics.merge(candidate_run.numerics)
            candidate_summary = self._candidate_training_summary(
                prepared=final_prepared,
                candidate_run=candidate_run,
                selection_run=selection_run,
                training_run_id=training_run_id,
                candidate_rank=candidate_rank,
                selected_candidate=selected_candidate,
                search_budget_summary=search_budget_summary,
                training_data_flow="phase1a_supervision_tensor_cache",
                validation_data_flow="phase1a_compiled_tensor_cache_evaluation",
                normalization_strategy="phase1a_train_only_tensor_cache",
                proxy_validation_used=False,
                tensor_cache_used=True,
                jsonl_fallback_used=False,
                tensor_cache_format=cache_manifest.format_version,
                tensor_cache_shard_count=tensor_cache_shard_count,
                batch_plan=final_prepared.batch_plan,
            )
            candidate_summary.update(
                {
                    "bootstrap_horizon_steps": self.config.bootstrap_horizon_steps,
                    "aux_value_loss_weight": self.config.aux_value_loss_weight,
                    "policy_state_feature_version": self.config.policy_state_feature_version,
                    "joint_action_vocabulary_version": self.config.joint_action_vocabulary_version,
                    "oracle_masked_row_count": final_prepared.oracle_masked_row_count,
                    "oracle_source_row_count": final_prepared.oracle_source_row_count,
                    "oracle_label_coverage_ratio": final_prepared.oracle_label_coverage_ratio,
                    "phase1a_supervision_used": True,
                    "phase1a_compute_dtype": self.config.phase1a_compute_dtype,
                }
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
                validation_step_count=final_prepared.val_step_count,
            )
            artifact_path = (
                paths.partial_candidate_dir / f"{artifact.policy_id}.json"
                if paths is not None
                else Path(f"/tmp/{artifact.policy_id}.json")
            )
            if paths is not None:
                dump_model(artifact_path, artifact)
            candidate_wall_sec = _time.perf_counter() - candidate_started_at
            candidate_wall_sec_history.append(candidate_wall_sec)
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
            partial_candidate_entries.append(
                {
                    "policy_id": artifact.policy_id,
                    "artifact_id": artifact.artifact_id,
                    "artifact_path": str(artifact_path),
                    "candidate_index": selection_run.candidate_index,
                    "candidate_rank": candidate_rank,
                    "selected_candidate": selected_candidate,
                    "candidate_spec": selection_run.candidate_spec.as_dict(),
                    "best_validation_total_net_return": candidate_run.best_validation_total_net_return,
                    "best_validation_composite_rank": candidate_run.best_validation_score.composite_rank,
                    "candidate_wall_sec": candidate_wall_sec,
                }
            )
            search_state["candidate_results"] = partial_candidate_entries
            self._write_phase1a_search_state(paths=paths, payload=search_state)
            self._write_phase1a_partial_manifest(
                paths=paths,
                training_run_id=training_run_id,
                search_budget_summary=search_budget_summary,
                candidate_results=partial_candidate_entries,
            )
            logger.info(
                "[PROGRESS] marker=candidate_completed candidate_index=%d candidate_rank=%d "
                "policy_id=%s candidate_wall_sec=%.3f",
                selection_run.candidate_index,
                candidate_rank,
                artifact.policy_id,
                candidate_wall_sec,
            )

        result = TrainingSearchResult(
            training_run_id=training_run_id,
            selected_artifact=candidate_results[0].artifact,
            candidate_results=candidate_results,
            search_budget_summary=search_budget_summary,
        )
        batch_compute_share = batch_compute_wall_sec / max(batch_assembly_wall_sec + batch_compute_wall_sec, 1e-9)
        self.last_phase1a_profile_report = {
            "candidate_wall_sec": candidate_wall_sec_history[-1] if candidate_wall_sec_history else 0.0,
            "candidate_wall_sec_history": candidate_wall_sec_history,
            "fold_wall_sec_history": fold_wall_sec_history,
            "fold_wall_sec": float(np.mean(np.asarray(fold_wall_sec_history, dtype=np.float64)))
            if fold_wall_sec_history
            else 0.0,
            "batch_assembly_wall_sec": batch_assembly_wall_sec,
            "batch_compute_wall_sec": batch_compute_wall_sec,
            "batch_compute_share": batch_compute_share,
            "tensor_cache_used": True,
            "phase1a_supervision_used": True,
            "jsonl_fallback_used": False,
            "resume_compatible": True,
            "completed_candidate_count": len(candidate_results),
            "completed_fold_count": len(fold_wall_sec_history),
        }
        self.last_phase1a_profile_report.update(numerics.as_dict())
        self._write_phase1a_partial_manifest(
            paths=paths,
            training_run_id=training_run_id,
            search_budget_summary=search_budget_summary,
            candidate_results=partial_candidate_entries,
        )
        logger.info(
            "phase1a_training_search_completed training_run_id=%s selected_policy_id=%s candidate_count=%d "
            "selected_validation_total_net_return=%.6f selected_validation_composite_rank=%.6f",
            training_run_id,
            result.selected_artifact.policy_id,
            len(candidate_results),
            candidate_results[0].best_validation_total_net_return,
            candidate_results[0].best_validation_composite_rank,
        )
        return result

    def _select_candidate_via_walkforward_phase1a_streaming(
        self,
        *,
        manifest: TrajectoryManifest,
        directory: Path,
        store_cls: Any,
        candidate_spec: TrainingCandidateSpec,
        candidate_index: int,
        parent_policy_id: str | None,
        training_run_id: str,
        code_commit_hash: str,
    ) -> _CandidateSelectionRun:
        fold_scores: list[FoldValidationScore] = []
        interval = timedelta(seconds=manifest.dataset_spec.sampling_interval_seconds)

        for fold in manifest.split_artifact.folds:
            purge_cutoff = fold.validation_window.start - (interval * fold.purge_width_steps)
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
            train_factory = self._trajectory_factory_from_window(directory, train_window, store_cls=store_cls)
            validation_factory = self._trajectory_factory_from_window(
                directory,
                validation_window,
                store_cls=store_cls,
            )
            try:
                prepared = self._prepare_phase1a_training_data(
                    train_factory=train_factory,
                    validation_factory=validation_factory,
                    reward_spec=manifest.reward_spec,
                    action_space=manifest.action_space,
                    venue_choices=manifest.dataset_spec.exchanges,
                )
            except ValueError as exc:
                if "0 supervised examples" not in str(exc):
                    raise
                logger.warning(
                    "phase1a_walkforward_fold_skipped fold_id=%s reason=no_supervised_examples "
                    "bootstrap_horizon_steps=%d",
                    fold.fold_id,
                    self.config.bootstrap_horizon_steps,
                )
                continue

            def _validate(parameters: LinearPolicyV2Parameters, config: TrainingConfig) -> EvaluationReport:
                artifact = self._build_artifact_from_manifest(
                    manifest=manifest,
                    config=config,
                    training_run_id=f"{training_run_id}:{fold.fold_id}",
                    code_commit_hash=code_commit_hash,
                    parameters=parameters,
                    parent_policy_id=parent_policy_id,
                    validation_total_net_return=0.0,
                    validation_score=None,
                    training_summary={},
                    search_metadata=None,
                    validation_step_count=prepared.val_step_count,
                )
                return self._phase1a_validation_report_from_factory(
                    dataset_spec=manifest.dataset_spec,
                    reward_spec=manifest.reward_spec,
                    validation_factory=validation_factory,
                    artifact=artifact,
                )

            fold_run = self._train_candidate_phase1a(
                prepared=prepared,
                train_factory=train_factory,
                candidate_spec=candidate_spec,
                candidate_index=candidate_index,
                reward_spec=manifest.reward_spec,
                action_space=manifest.action_space,
                validate_fn=_validate,
            )
            fold_scores.append(
                FoldValidationScore(
                    fold_id=fold.fold_id,
                    validation_total_net_return=fold_run.best_validation_total_net_return,
                    validation_composite_rank=fold_run.best_validation_score.composite_rank,
                    validation_step_count=prepared.val_step_count,
                )
            )
        if not fold_scores:
            logger.warning(
                "phase1a_walkforward_no_qualifying_folds fallback=train_validation "
                "bootstrap_horizon_steps=%d",
                self.config.bootstrap_horizon_steps,
            )
            train_window = StreamingWindow(split_name="train")
            validation_window = StreamingWindow(split_name="validation")
            train_factory = self._trajectory_factory_from_window(directory, train_window, store_cls=store_cls)
            validation_factory = self._trajectory_factory_from_window(
                directory,
                validation_window,
                store_cls=store_cls,
            )
            prepared = self._prepare_phase1a_training_data(
                train_factory=train_factory,
                validation_factory=validation_factory,
                reward_spec=manifest.reward_spec,
                action_space=manifest.action_space,
                venue_choices=manifest.dataset_spec.exchanges,
            )

            def _validate(parameters: LinearPolicyV2Parameters, config: TrainingConfig) -> EvaluationReport:
                artifact = self._build_artifact_from_manifest(
                    manifest=manifest,
                    config=config,
                    training_run_id=f"{training_run_id}:fallback",
                    code_commit_hash=code_commit_hash,
                    parameters=parameters,
                    parent_policy_id=parent_policy_id,
                    validation_total_net_return=0.0,
                    validation_score=None,
                    training_summary={},
                    search_metadata=None,
                    validation_step_count=prepared.val_step_count,
                )
                return self._phase1a_validation_report_from_factory(
                    dataset_spec=manifest.dataset_spec,
                    reward_spec=manifest.reward_spec,
                    validation_factory=validation_factory,
                    artifact=artifact,
                )

            fold_run = self._train_candidate_phase1a(
                prepared=prepared,
                train_factory=train_factory,
                candidate_spec=candidate_spec,
                candidate_index=candidate_index,
                reward_spec=manifest.reward_spec,
                action_space=manifest.action_space,
                validate_fn=_validate,
            )
            fold_scores.append(
                FoldValidationScore(
                    fold_id="fallback-train-validation",
                    validation_total_net_return=fold_run.best_validation_total_net_return,
                    validation_composite_rank=fold_run.best_validation_score.composite_rank,
                    validation_step_count=prepared.val_step_count,
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
            "phase1a_training_candidate_walkforward_completed candidate_index=%d seed=%d "
            "learning_rate=%.6f l2_weight=%.6f fold_count=%d selection_total_net_return=%.6f "
            "selection_composite_rank=%.6f",
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
        validation_wall_sec_history: list[float] = []
        batch_assembly_wall_sec = 0.0
        batch_compute_wall_sec = 0.0
        best_validation_total_net_return: float | None = None
        best_epoch = 0
        best_parameters: LinearPolicyParameters | None = None
        best_validation_score = None

        for epoch in range(1, config.epochs + 1):
            import time as _time

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
            validation_started_at = _time.perf_counter()
            validation_report = self._validation_report(bundle, validation_artifact)
            validation_wall_sec_history.append(_time.perf_counter() - validation_started_at)
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
            validation_wall_sec_history=validation_wall_sec_history,
            batch_assembly_wall_sec=batch_assembly_wall_sec,
            batch_compute_wall_sec=batch_compute_wall_sec,
        )

    def _candidate_training_summary(
        self,
        *,
        prepared: _PreparedTrainingData | _StreamingPreparedData | _Phase1APreparedData,
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
        tensor_cache_used: bool,
        jsonl_fallback_used: bool,
        tensor_cache_format: str | None,
        tensor_cache_shard_count: int | None,
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
            "tensor_cache_used": tensor_cache_used,
            "jsonl_fallback_used": jsonl_fallback_used,
            "tensor_cache_format": tensor_cache_format,
            "tensor_cache_shard_count": tensor_cache_shard_count,
            "training_loss_history": candidate_run.loss_history,
            "validation_objective_history": candidate_run.validation_history,
            "validation_wall_sec_history": candidate_run.validation_wall_sec_history,
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
        summary.update((candidate_run.numerics or _Phase1ANumericsTelemetry()).as_dict())
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
        parameters: LinearPolicyParameters | LinearPolicyV2Parameters,
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
        evaluation_surface_id = build_evaluation_surface_id(
            slice_id=bundle.dataset_spec.slice_id,
            split_version=bundle.split_artifact.split_version,
            reward_version=bundle.reward_spec.reward_version,
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
            f"action_space:{bundle.action_space.action_space_version}",
            f"runtime_contract:{strict_runtime_contract.runtime_contract_version}",
            f"policy_kind:{strict_runtime_contract.policy_kind}",
            f"derived_contract:{strict_runtime_contract.derived_contract_version}",
            f"derived_signature:{strict_runtime_contract.derived_channel_template_signature}",
            f"feature_dim:{strict_runtime_contract.expected_feature_dim}",
            "compat_mode:strict",
        ]
        if strict_runtime_contract.policy_state_feature_version is not None:
            artifact_tags.append(f"policy_state_features:{strict_runtime_contract.policy_state_feature_version}")
        if strict_runtime_contract.joint_action_vocabulary_version is not None:
            artifact_tags.append(f"joint_action_vocabulary:{strict_runtime_contract.joint_action_vocabulary_version}")
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
                action_space_version=bundle.action_space.action_space_version,
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
        *,
        allow_jsonl_fallback: bool = False,
        search_output: Path | None = None,
        resume_search: bool = False,
    ) -> TrainingSearchResult:
        """PRODUCTION PATH — train from a tensor-cache backed trajectory directory.

        Uses `tensor_cache_v1` shards when present. JSONL streaming remains only
        as an explicit temporary compatibility fallback.
        """
        from quantlab_ml.trajectories.streaming_store import TrajectoryDirectoryStore

        self.last_phase1a_profile_report = None
        cache_status = tensor_cache_payload_status(directory)
        jsonl_training_ready = (
            TrajectoryDirectoryStore.is_trajectory_directory(directory)
            and all(TrajectoryDirectoryStore.split_exists(directory, split_name) for split_name in manifest.split_names)
        )
        if self.config.runtime_adapter == "linear-policy-v2":
            if cache_status.payload_complete:
                materialization = materialize_phase1a_supervision(
                    trajectories_directory=directory,
                    output_directory=phase1a_supervision_directory(directory),
                    manifest=manifest,
                    training_config=self.config,
                )
                return self._train_search_from_directory_phase1a_tensor_cache(
                    manifest=manifest,
                    directory=directory,
                    cache_manifest=read_tensor_cache_manifest(directory),
                    supervision_manifest=materialization.manifest,
                    parent_policy_id=parent_policy_id,
                    search_output=search_output,
                    resume_search=resume_search,
                )
            if not jsonl_training_ready:
                typed_error = infer_bundle_payload_error_for_directory(directory)
                if typed_error is not None:
                    raise typed_error
                raise ValueError(
                    "linear-policy-v2 phase1a training requires payload-complete tensor_cache_v1 "
                    "or explicit JSONL fallback with readable split records"
                )
            if allow_jsonl_fallback:
                logger.warning(
                    "phase1a_training_directory_jsonl_fallback path=%s tensor_cache_used=false "
                    "phase1a_supervision_used=false jsonl_fallback_used=true "
                    "path_classification=temporary_compatibility_maintenance",
                    directory,
                )
                self.last_phase1a_profile_report = {
                    "candidate_wall_sec": 0.0,
                    "candidate_wall_sec_history": [],
                    "fold_wall_sec_history": [],
                    "fold_wall_sec": 0.0,
                    "batch_assembly_wall_sec": 0.0,
                    "batch_compute_wall_sec": 0.0,
                    "batch_compute_share": 0.0,
                    "tensor_cache_used": False,
                    "phase1a_supervision_used": False,
                    "jsonl_fallback_used": True,
                    "resume_compatible": False,
                    "completed_candidate_count": 0,
                    "completed_fold_count": 0,
                    "joint_ce_loss": 0.0,
                    "aux_value_loss_raw": 0.0,
                    "aux_value_loss_weighted": 0.0,
                    "total_loss": 0.0,
                    "action_logit_abs_max": 0.0,
                    "action_entropy": 0.0,
                    "value_pred_abs_max": 0.0,
                    "value_grad_norm_pre_clip": 0.0,
                    "value_grad_norm_post_clip": 0.0,
                    "clip_applied_count": 0,
                    "first_nonfinite_component": None,
                    "first_nonfinite_batch_context": None,
                }
                return self._train_search_from_directory_phase1a(
                    manifest=manifest,
                    directory=directory,
                    store_cls=TrajectoryDirectoryStore,
                    parent_policy_id=parent_policy_id,
                )
            raise ValueError(
                "linear-policy-v2 phase1a training requires tensor_cache_v1 + phase1a_supervision_v1; "
                "pass allow_jsonl_fallback=True only for temporary compatibility maintenance"
            )

        if cache_status.payload_complete:
            return self._train_search_from_directory_tensor_cache(
                manifest=manifest,
                directory=directory,
                cache_manifest=read_tensor_cache_manifest(directory),
                parent_policy_id=parent_policy_id,
            )
        if cache_status.manifest_present and not cache_status.payload_complete:
            if allow_jsonl_fallback and jsonl_training_ready:
                logger.warning(
                    "training_directory_tensor_cache_dangling path=%s tensor_cache_used=false "
                    "jsonl_fallback_used=true path_classification=temporary_compatibility_maintenance",
                    directory,
                )
                return self._train_search_from_directory_jsonl(
                    manifest=manifest,
                    directory=directory,
                    parent_policy_id=parent_policy_id,
                )
            raise DanglingTensorCacheManifestError(
                detail=(
                    "trajectory directory contains tensor_cache_manifest.json references "
                    "without readable shard payloads"
                ),
                bundle_root=directory.expanduser().resolve(),
            )
        if not allow_jsonl_fallback:
            typed_error = infer_bundle_payload_error_for_directory(directory)
            if typed_error is not None:
                raise typed_error
            raise ValueError(
                "tensor cache manifest missing for trajectory directory; "
                "pass allow_jsonl_fallback=True only for temporary compatibility maintenance"
            )
        if not jsonl_training_ready:
            typed_error = infer_bundle_payload_error_for_directory(directory)
            if typed_error is not None:
                raise typed_error
            raise ValueError(
                "requested JSONL fallback but one or more required split files are unavailable "
                "for trajectory directory training"
            )
        logger.warning(
            "training_directory_tensor_cache_missing path=%s tensor_cache_used=false "
            "jsonl_fallback_used=true path_classification=temporary_compatibility_maintenance",
            directory,
        )
        return self._train_search_from_directory_jsonl(
            manifest=manifest,
            directory=directory,
            parent_policy_id=parent_policy_id,
        )

    def _train_search_from_directory_tensor_cache(
        self,
        *,
        manifest: TrajectoryManifest,
        directory: Path,
        cache_manifest: TensorCacheManifest,
        parent_policy_id: str | None,
    ) -> TrainingSearchResult:
        if cache_manifest.format_version != TENSOR_CACHE_FORMAT_VERSION:
            raise ValueError(
                f"unsupported tensor cache format: {cache_manifest.format_version!r}"
            )
        candidate_specs = self._candidate_specs()
        code_commit_hash = current_code_commit_hash()
        training_run_id = self._training_run_id_from_manifest(
            manifest,
            parent_policy_id=parent_policy_id,
            code_commit_hash=code_commit_hash,
            candidate_specs=candidate_specs,
        )
        search_budget_summary = _search_budget_summary(candidate_specs)
        tensor_cache_shard_count = sum(split.shard_count for split in cache_manifest.splits.values())
        logger.info(
            "training_search_started training_run_id=%s candidate_count=%d split_version=%s reward_version=%s "
            "training_backend=%s training_device=%s cuda_available=%s device_name=%s "
            "tensor_cache_format=%s tensor_cache_used=true jsonl_fallback_used=false "
            "tensor_cache_shard_count=%d cache_feature_dtype=%s cache_feature_dim=%d",
            training_run_id,
            len(candidate_specs),
            manifest.split_artifact.split_version,
            manifest.reward_spec.reward_version,
            self._backend.backend_name,
            self._backend.device_resolution.training_device,
            self._backend.device_resolution.cuda_available,
            self._backend.device_resolution.device_name,
            cache_manifest.format_version,
            tensor_cache_shard_count,
            cache_manifest.feature_dtype,
            cache_manifest.feature_dim,
        )

        selection_runs = [
            self._select_candidate_via_walkforward_tensor_cache(
                manifest=manifest,
                directory=directory,
                cache_manifest=cache_manifest,
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
        prepared = self._prepare_training_data_tensor_cache(
            manifest=manifest,
            directory=directory,
            cache_manifest=cache_manifest,
            train_window=final_train_window,
            validation_window=final_validation_window,
        )

        candidate_results: list[TrainingCandidateResult] = []
        for candidate_rank, selection_run in enumerate(ranked_selections, start=1):
            selected_candidate = candidate_rank == 1
            candidate_run = self._train_candidate_from_tensor_cache(
                manifest=manifest,
                directory=directory,
                cache_manifest=cache_manifest,
                prepared=prepared,
                train_window=final_train_window,
                validation_window=final_validation_window,
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
                training_data_flow="tensor_shard_batch",
                validation_data_flow="tensor_shard_evaluation",
                normalization_strategy="train_only_two_pass_tensor_cache",
                proxy_validation_used=False,
                tensor_cache_used=True,
                jsonl_fallback_used=False,
                tensor_cache_format=cache_manifest.format_version,
                tensor_cache_shard_count=tensor_cache_shard_count,
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
            "selected_validation_total_net_return=%.6f selected_validation_composite_rank=%.6f "
            "tensor_cache_used=true jsonl_fallback_used=false",
            training_run_id,
            result.selected_artifact.policy_id,
            len(candidate_results),
            candidate_results[0].best_validation_total_net_return,
            candidate_results[0].best_validation_composite_rank,
        )
        return result

    def _train_search_from_directory_jsonl(
        self,
        *,
        manifest: TrajectoryManifest,
        directory: Path,
        parent_policy_id: str | None,
    ) -> TrainingSearchResult:
        """TEMPORARY COMPAT PATH — train from streaming JSONL only."""
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
            "training_backend=%s training_device=%s cuda_available=%s device_name=%s "
            "tensor_cache_used=false jsonl_fallback_used=true path_classification=temporary_compatibility_maintenance",
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
                tensor_cache_used=False,
                jsonl_fallback_used=True,
                tensor_cache_format=None,
                tensor_cache_shard_count=None,
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
            "selected_validation_total_net_return=%.6f selected_validation_composite_rank=%.6f "
            "tensor_cache_used=false jsonl_fallback_used=true",
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

    def _tensor_cache_split_manifest(
        self,
        cache_manifest: TensorCacheManifest,
        split_name: str,
    ) -> Any:
        split_manifest = cache_manifest.splits.get(split_name)
        if split_manifest is None:
            raise ValueError(f"tensor cache is missing split {split_name!r}")
        return split_manifest

    def _count_window_steps_tensor_cache(
        self,
        directory: Path,
        cache_manifest: TensorCacheManifest,
        window: StreamingWindow,
    ) -> int:
        step_count = 0
        for shard in self._tensor_cache_split_manifest(cache_manifest, window.split_name).shards:
            loaded = load_tensor_cache_shard(directory, shard)
            step_count += int(
                window_row_indices(
                    loaded.event_time_ms,
                    start=window.start,
                    end=window.end,
                    exclusive_end=window.exclusive_end,
                ).shape[0]
            )
        return step_count

    def _tensor_cache_feature_stats(
        self,
        directory: Path,
        cache_manifest: TensorCacheManifest,
        window: StreamingWindow,
    ) -> tuple[int, np.ndarray, np.ndarray]:
        total_count = 0
        feature_sum: np.ndarray | None = None
        feature_sum_sq: np.ndarray | None = None

        for shard in self._tensor_cache_split_manifest(cache_manifest, window.split_name).shards:
            loaded = load_tensor_cache_shard(directory, shard)
            row_idx = window_row_indices(
                loaded.event_time_ms,
                start=window.start,
                end=window.end,
                exclusive_end=window.exclusive_end,
            )
            if row_idx.size == 0:
                continue
            batch = loaded.features[row_idx].astype(np.float64, copy=False)
            if feature_sum is None or feature_sum_sq is None:
                feature_sum = np.zeros(batch.shape[1], dtype=np.float64)
                feature_sum_sq = np.zeros(batch.shape[1], dtype=np.float64)
            feature_sum += batch.sum(axis=0)
            feature_sum_sq += np.square(batch).sum(axis=0)
            total_count += int(row_idx.shape[0])

        if total_count <= 0 or feature_sum is None or feature_sum_sq is None:
            raise ValueError(f"tensor cache stats window {window.split_name!r} returned 0 qualifying examples")

        feature_mean = feature_sum / total_count
        feature_var = np.maximum((feature_sum_sq / total_count) - np.square(feature_mean), 0.0)
        feature_std = np.sqrt(feature_var)
        feature_std = np.where(feature_std < 1e-6, 1.0, feature_std).astype(np.float32)
        return total_count, feature_mean.astype(np.float32), feature_std

    def _prepare_training_data_tensor_cache(
        self,
        *,
        manifest: TrajectoryManifest,
        directory: Path,
        cache_manifest: TensorCacheManifest,
        train_window: StreamingWindow,
        validation_window: StreamingWindow,
    ) -> _StreamingPreparedData:
        expected_cache_feature_dim = build_strict_runtime_contract(
            manifest.observation_schema,
            policy_kind=self.config.runtime_adapter,
        ).expected_feature_dim
        if cache_manifest.feature_dim != expected_cache_feature_dim:
            raise ValueError(
                "tensor cache feature dimension does not match manifest observation schema: "
                f"cache_feature_dim={cache_manifest.feature_dim}, "
                f"expected_feature_dim={expected_cache_feature_dim}"
            )
        train_step_count, feature_mean, feature_std = self._tensor_cache_feature_stats(
            directory,
            cache_manifest,
            train_window,
        )
        val_step_count = self._count_window_steps_tensor_cache(directory, cache_manifest, validation_window)
        if val_step_count <= 0:
            raise ValueError("validation split is empty")
        batch_plan = self._streaming_batch_plan(
            feature_dim=int(feature_mean.shape[0]),
            train_step_count=train_step_count,
        )
        logger.info(
            "tensor_cache_training_data_prepared train_examples=%d validation_examples=%d "
            "feature_dim=%d effective_batch_size=%d estimated_batch_bytes=%d "
            "batches_per_epoch=%d batch_target_bytes=%d tensor_cache_format=%s "
            "tensor_cache_used=true jsonl_fallback_used=false",
            train_step_count,
            val_step_count,
            feature_mean.shape[0],
            batch_plan.effective_batch_size,
            batch_plan.estimated_batch_bytes,
            batch_plan.batches_per_epoch,
            batch_plan.batch_target_bytes,
            cache_manifest.format_version,
        )
        return _StreamingPreparedData(
            train_step_count=train_step_count,
            val_step_count=val_step_count,
            action_keys=manifest.action_space.action_keys,
            venue_choices=manifest.dataset_spec.exchanges,
            feature_mean=feature_mean,
            feature_std=feature_std,
            batch_plan=batch_plan,
        )

    def _select_candidate_via_walkforward_tensor_cache(
        self,
        *,
        manifest: TrajectoryManifest,
        directory: Path,
        cache_manifest: TensorCacheManifest,
        candidate_spec: TrainingCandidateSpec,
        candidate_index: int,
        parent_policy_id: str | None,
        training_run_id: str,
        code_commit_hash: str,
    ) -> _CandidateSelectionRun:
        fold_scores: list[FoldValidationScore] = []
        interval = timedelta(seconds=manifest.dataset_spec.sampling_interval_seconds)

        for fold in manifest.split_artifact.folds:
            purge_cutoff = fold.validation_window.start - (interval * fold.purge_width_steps)
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
            fold_prepared = self._prepare_training_data_tensor_cache(
                manifest=manifest,
                directory=directory,
                cache_manifest=cache_manifest,
                train_window=train_window,
                validation_window=validation_window,
            )
            fold_run = self._train_candidate_from_tensor_cache(
                manifest=manifest,
                directory=directory,
                cache_manifest=cache_manifest,
                prepared=fold_prepared,
                train_window=train_window,
                validation_window=validation_window,
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
            [score.validation_total_net_return for score in fold_scores],
            [score.validation_step_count for score in fold_scores],
        )
        selection_composite_rank = _weighted_mean(
            [score.validation_composite_rank for score in fold_scores],
            [score.validation_step_count for score in fold_scores],
        )
        logger.info(
            "training_candidate_walkforward_completed candidate_index=%d seed=%d "
            "learning_rate=%.6f l2_weight=%.6f fold_count=%d "
            "selection_total_net_return=%.6f selection_composite_rank=%.6f "
            "tensor_cache_used=true jsonl_fallback_used=false",
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

    def _train_tensor_cache_epoch(
        self,
        *,
        directory: Path,
        cache_manifest: TensorCacheManifest,
        prepared: _StreamingPreparedData,
        train_window: StreamingWindow,
        state: object,
        config: TrainingConfig,
    ) -> float:
        batch_size = prepared.batch_plan.effective_batch_size
        weighted_loss_total = 0.0
        seen = 0

        for shard in self._tensor_cache_split_manifest(cache_manifest, train_window.split_name).shards:
            loaded = load_tensor_cache_shard(directory, shard)
            row_idx = window_row_indices(
                loaded.event_time_ms,
                start=train_window.start,
                end=train_window.end,
                exclusive_end=train_window.exclusive_end,
            )
            if row_idx.size == 0:
                continue

            normalized = loaded.features[row_idx].astype(np.float64, copy=False)
            normalized -= prepared.feature_mean
            normalized /= prepared.feature_std
            action_labels = loaded.action_labels[row_idx]
            venue_mask = loaded.venue_mask[row_idx]
            venue_labels = loaded.venue_labels[row_idx]

            for start_idx in range(0, int(row_idx.shape[0]), batch_size):
                end_idx = min(start_idx + batch_size, int(row_idx.shape[0]))
                batch_features = normalized[start_idx:end_idx]
                batch_action_labels = action_labels[start_idx:end_idx]
                batch_venue_mask = venue_mask[start_idx:end_idx]
                batch_venue_labels = venue_labels[start_idx:end_idx]
                batch_loss = self._backend.batch_step(
                    state=state,
                    batch_features=batch_features,
                    batch_action_labels=batch_action_labels,
                    batch_venue_mask=batch_venue_mask,
                    batch_venue_labels=batch_venue_labels,
                    config=config,
                )
                batch_rows = end_idx - start_idx
                weighted_loss_total += batch_loss * batch_rows
                seen += batch_rows

        if seen <= 0:
            raise ValueError("train split is empty")
        return float(weighted_loss_total / seen)

    def _train_candidate_from_tensor_cache(
        self,
        *,
        manifest: TrajectoryManifest,
        directory: Path,
        cache_manifest: TensorCacheManifest,
        prepared: _StreamingPreparedData,
        train_window: StreamingWindow,
        validation_window: StreamingWindow,
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
        logger.info(
            "tensor_cache_training_candidate_started candidate_index=%d seed=%d learning_rate=%.6f "
            "l2_weight=%.6f effective_batch_size=%d estimated_batch_bytes=%d "
            "batches_per_epoch=%d batch_target_bytes=%d tensor_cache_format=%s",
            candidate_index,
            candidate_spec.seed,
            candidate_spec.learning_rate,
            candidate_spec.l2_weight,
            prepared.batch_plan.effective_batch_size,
            prepared.batch_plan.estimated_batch_bytes,
            prepared.batch_plan.batches_per_epoch,
            prepared.batch_plan.batch_target_bytes,
            cache_manifest.format_version,
        )
        loss_history: list[float] = []
        validation_history: list[float] = []
        validation_wall_sec_history: list[float] = []
        best_validation_total_net_return: float | None = None
        best_parameters: LinearPolicyParameters | None = None
        best_validation_score: PolicyScore | None = None
        best_epoch = 0
        for epoch in range(1, config.epochs + 1):
            import time as _time

            epoch_started_at = _time.perf_counter()
            total_loss = self._train_tensor_cache_epoch(
                directory=directory,
                cache_manifest=cache_manifest,
                prepared=prepared,
                train_window=train_window,
                state=state,
                config=config,
            )
            epoch_wall_sec = _time.perf_counter() - epoch_started_at
            train_rows_per_sec = prepared.train_step_count / max(epoch_wall_sec, 1e-9)
            logger.info(
                "epoch_timing epoch=%d wall_sec=%.1f total_loss=%.6f train_rows_per_sec=%.2f "
                "tensor_cache_used=true jsonl_fallback_used=false",
                epoch,
                epoch_wall_sec,
                total_loss,
                train_rows_per_sec,
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
            validation_started_at = _time.perf_counter()
            validation_report = self._validation_report_from_tensor_cache(
                manifest=manifest,
                directory=directory,
                cache_manifest=cache_manifest,
                artifact=validation_artifact,
                validation_window=validation_window,
            )
            validation_wall_sec = _time.perf_counter() - validation_started_at
            validation_rows_per_sec = prepared.val_step_count / max(validation_wall_sec, 1e-9)
            logger.info(
                "validation_timing epoch=%d wall_sec=%.1f validation_rows_per_sec=%.2f "
                "tensor_cache_used=true jsonl_fallback_used=false",
                epoch,
                validation_wall_sec,
                validation_rows_per_sec,
            )
            validation_score = PolicyScorer().score(validation_report)
            validation_history.append(validation_report.total_net_return)
            validation_wall_sec_history.append(validation_wall_sec)
            is_best = (
                best_validation_total_net_return is None
                or validation_report.total_net_return > best_validation_total_net_return
            )
            if is_best:
                best_epoch = epoch
                best_parameters = parameters
                best_validation_total_net_return = validation_report.total_net_return
                best_validation_score = validation_score

        assert best_parameters is not None
        assert best_validation_total_net_return is not None
        assert best_validation_score is not None
        logger.info(
            "training_candidate_completed candidate_index=%d seed=%d best_epoch=%d "
            "best_validation_total_net_return=%.6f best_validation_composite_rank=%.6f "
            "effective_batch_size=%d estimated_batch_bytes=%d batches_per_epoch=%d "
            "training_backend=%s tensor_cache_used=true jsonl_fallback_used=false",
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
            validation_wall_sec_history=validation_wall_sec_history,
        )

    def _validation_report_from_tensor_cache(
        self,
        *,
        manifest: TrajectoryManifest,
        directory: Path,
        cache_manifest: TensorCacheManifest,
        artifact: PolicyArtifact,
        validation_window: StreamingWindow,
    ) -> EvaluationReport:
        from quantlab_ml.evaluation import EvaluationEngine

        engine = EvaluationEngine(self._evaluation_boundary(manifest.reward_spec.timestamping))
        return engine.evaluate_directory(
            manifest=manifest,
            directory=directory,
            artifact=artifact,
            cache_manifest=cache_manifest,
            split_name=validation_window.split_name,
            start=validation_window.start,
            end=validation_window.end,
            exclusive_end=validation_window.exclusive_end,
            allow_jsonl_fallback=False,
        )

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
        validation_wall_sec_history: list[float] = []
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
                "epoch_timing epoch=%d wall_sec=%.1f total_loss=%.6f train_rows_per_sec=%.2f",
                epoch,
                _t_epoch1 - _t_epoch0,
                total_loss,
                prepared.train_step_count / max(_t_epoch1 - _t_epoch0, 1e-9),
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
            _t_val0 = _time.perf_counter()
            validation_report = self._validation_report_from_manifest(
                manifest=manifest,
                directory=directory,
                artifact=validation_artifact,
                validation_window=validation_window,
                store_cls=store_cls,
            )
            _t_val1 = _time.perf_counter()
            validation_score = PolicyScorer().score(validation_report)
            validation_history.append(validation_report.total_net_return)
            validation_wall_sec_history.append(_t_val1 - _t_val0)
            logger.info(
                "validation_timing epoch=%d wall_sec=%.1f validation_rows_per_sec=%.2f "
                "tensor_cache_used=false jsonl_fallback_used=true",
                epoch,
                _t_val1 - _t_val0,
                prepared.val_step_count / max(_t_val1 - _t_val0, 1e-9),
            )
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
            validation_wall_sec_history=validation_wall_sec_history,
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
        parameters: LinearPolicyParameters | LinearPolicyV2Parameters,
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
        evaluation_surface_id = build_evaluation_surface_id(
            slice_id=manifest.dataset_spec.slice_id,
            split_version=manifest.split_artifact.split_version,
            reward_version=manifest.reward_spec.reward_version,
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
            f"action_space:{manifest.action_space.action_space_version}",
            f"runtime_contract:{strict_runtime_contract.runtime_contract_version}",
            f"policy_kind:{strict_runtime_contract.policy_kind}",
            f"derived_contract:{strict_runtime_contract.derived_contract_version}",
            f"derived_signature:{strict_runtime_contract.derived_channel_template_signature}",
            f"feature_dim:{strict_runtime_contract.expected_feature_dim}",
            "compat_mode:strict",
        ]
        if strict_runtime_contract.policy_state_feature_version is not None:
            artifact_tags.append(f"policy_state_features:{strict_runtime_contract.policy_state_feature_version}")
        if strict_runtime_contract.joint_action_vocabulary_version is not None:
            artifact_tags.append(f"joint_action_vocabulary:{strict_runtime_contract.joint_action_vocabulary_version}")
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
                action_space_version=manifest.action_space.action_space_version,
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
    best_parameters: LinearPolicyParameters | LinearPolicyV2Parameters
    best_validation_total_net_return: float
    best_validation_score: PolicyScore
    loss_history: list[float]
    validation_history: list[float]
    validation_wall_sec_history: list[float]
    batch_assembly_wall_sec: float = 0.0
    batch_compute_wall_sec: float = 0.0
    numerics: _Phase1ANumericsTelemetry | None = None


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


@dataclass(slots=True)
class _Phase1ANumpyTrainingState:
    joint_action_weight: np.ndarray
    joint_action_bias: np.ndarray
    value_weight: np.ndarray
    value_bias: float


@dataclass(slots=True)
class _Phase1ATorchTrainingState:
    joint_action_weight: Any
    joint_action_bias: Any
    value_weight: Any
    value_bias: Any


def _phase1a_initialize_state(
    *,
    backend: _LinearTrainingBackend,
    seed: int,
    joint_action_count: int,
    feature_dim: int,
    compute_dtype: str,
) -> object:
    numpy_dtype = np.float32 if compute_dtype == "float32" else np.float64
    joint_action_weight, joint_action_bias, value_weight, value_bias = _initial_phase1a_parameter_arrays(
        seed=seed,
        joint_action_count=joint_action_count,
        feature_dim=feature_dim,
        dtype=numpy_dtype,
    )
    if isinstance(backend, _NumpyLinearTrainingBackend):
        return _Phase1ANumpyTrainingState(
            joint_action_weight=joint_action_weight,
            joint_action_bias=joint_action_bias,
            value_weight=value_weight,
            value_bias=value_bias,
        )
    if isinstance(backend, _TorchLinearTrainingBackend):
        torch_module = backend._torch
        torch_module.manual_seed(seed)
        if backend.device_resolution.cuda_available and hasattr(torch_module.cuda, "manual_seed_all"):
            torch_module.cuda.manual_seed_all(seed)
        if hasattr(torch_module, "use_deterministic_algorithms"):
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch_module.use_deterministic_algorithms(True)
        device = backend.device_resolution.compute_device
        torch_dtype = torch_module.float32 if compute_dtype == "float32" else torch_module.float64
        return _Phase1ATorchTrainingState(
            joint_action_weight=torch_module.tensor(
                joint_action_weight,
                dtype=torch_dtype,
                device=device,
            ),
            joint_action_bias=torch_module.tensor(
                joint_action_bias,
                dtype=torch_dtype,
                device=device,
            ),
            value_weight=torch_module.tensor(
                value_weight,
                dtype=torch_dtype,
                device=device,
            ),
            value_bias=torch_module.tensor(
                value_bias,
                dtype=torch_dtype,
                device=device,
            ),
        )
    raise TypeError(f"unsupported backend for phase1a initialization: {type(backend)!r}")


def _phase1a_batch_step(
    *,
    backend: _LinearTrainingBackend,
    state: object,
    batch_features: np.ndarray,
    batch_joint_action_labels: np.ndarray,
    batch_joint_action_masks: np.ndarray,
    batch_value_targets: np.ndarray,
    config: TrainingConfig,
    batch_context: dict[str, object] | None = None,
) -> _Phase1ABatchStepResult:
    numerics = _Phase1ANumericsTelemetry()
    resolved_batch_context = dict(batch_context or {})
    _phase1a_require_finite_array(
        "batch_features",
        batch_features,
        numerics=numerics,
        batch_context=resolved_batch_context,
    )
    _phase1a_require_finite_array(
        "batch_value_targets",
        batch_value_targets,
        numerics=numerics,
        batch_context=resolved_batch_context,
    )
    if isinstance(backend, _NumpyLinearTrainingBackend):
        training_state = _expect_phase1a_numpy_state(state)
        joint_logits = batch_features @ training_state.joint_action_weight.T + training_state.joint_action_bias
        action_logit_abs_max = float(np.max(np.abs(joint_logits))) if joint_logits.size > 0 else 0.0
        masked_joint_logits = np.where(batch_joint_action_masks, joint_logits, -1.0e30)
        _phase1a_require_finite_array(
            "joint_logits",
            masked_joint_logits,
            numerics=numerics,
            batch_context=resolved_batch_context,
        )
        joint_probabilities = _softmax_matrix(masked_joint_logits)
        _phase1a_require_finite_array(
            "joint_probabilities",
            joint_probabilities,
            numerics=numerics,
            batch_context=resolved_batch_context,
        )
        joint_loss = _cross_entropy_loss(joint_probabilities, batch_joint_action_labels)
        action_entropy = _categorical_entropy(joint_probabilities)
        joint_gradient = joint_probabilities.copy()
        joint_gradient[np.arange(len(batch_joint_action_labels)), batch_joint_action_labels] -= 1.0
        joint_gradient /= len(batch_joint_action_labels)
        joint_gradient[~batch_joint_action_masks] = 0.0
        joint_action_weight_gradient = joint_gradient.T @ batch_features
        joint_action_bias_gradient = joint_gradient.sum(axis=0)

        value_prediction = batch_features @ training_state.value_weight + training_state.value_bias
        value_pred_abs_max = float(np.max(np.abs(value_prediction))) if value_prediction.size > 0 else 0.0
        if math.isfinite(value_pred_abs_max):
            numerics.value_pred_abs_max = max(numerics.value_pred_abs_max, value_pred_abs_max)
        _phase1a_require_finite_array(
            "value_prediction",
            value_prediction,
            numerics=numerics,
            batch_context=resolved_batch_context,
        )
        value_error = value_prediction.astype(np.float64, copy=False) - batch_value_targets.astype(
            np.float64,
            copy=False,
        )
        value_loss = _huber_loss_numpy(value_error, delta=_PHASE1A_AUX_VALUE_HUBER_DELTA)
        _phase1a_require_finite_scalar(
            "value_loss",
            value_loss,
            numerics=numerics,
            batch_context=resolved_batch_context,
        )
        value_weight_gradient = np.zeros(training_state.value_weight.shape, dtype=np.float64)
        value_bias_gradient = 0.0
        aux_value_loss_weighted = 0.0
        total_loss = float(joint_loss)
        if config.aux_value_loss_weight > 0.0:
            aux_value_loss_weighted = config.aux_value_loss_weight * value_loss
            total_loss += aux_value_loss_weighted
            huber_gradient = _huber_gradient_numpy(value_error, delta=_PHASE1A_AUX_VALUE_HUBER_DELTA)
            value_scale = config.aux_value_loss_weight / len(batch_value_targets)
            value_weight_gradient = value_scale * (
                huber_gradient[:, None] * batch_features.astype(np.float64, copy=False)
            ).sum(axis=0)
            value_bias_gradient = value_scale * float(huber_gradient.sum())

        if config.l2_weight > 0.0:
            total_loss += config.l2_weight * (
                float(np.sum(training_state.joint_action_weight**2))
                + float(np.sum(np.square(training_state.value_weight.astype(np.float64, copy=False))))
            )
            joint_action_weight_gradient += config.l2_weight * training_state.joint_action_weight
            value_weight_gradient += config.l2_weight * training_state.value_weight.astype(np.float64, copy=False)

        (
            value_weight_gradient,
            value_bias_gradient,
            value_grad_norm_pre_clip,
            value_grad_norm_post_clip,
            clip_applied,
        ) = _phase1a_clip_value_gradients_numpy(
            weight_gradient=value_weight_gradient,
            bias_gradient=value_bias_gradient,
            clip_norm=_PHASE1A_VALUE_GRAD_CLIP_NORM,
        )
        if math.isfinite(value_grad_norm_pre_clip):
            numerics.value_grad_norm_pre_clip = max(
                numerics.value_grad_norm_pre_clip,
                value_grad_norm_pre_clip,
            )
        if math.isfinite(value_grad_norm_post_clip):
            numerics.value_grad_norm_post_clip = max(
                numerics.value_grad_norm_post_clip,
                value_grad_norm_post_clip,
            )
        numerics.clip_applied_count += int(clip_applied)
        _phase1a_require_finite_array(
            "value_weight_gradient",
            value_weight_gradient,
            numerics=numerics,
            batch_context=resolved_batch_context,
        )
        _phase1a_require_finite_scalar(
            "value_bias_gradient",
            value_bias_gradient,
            numerics=numerics,
            batch_context=resolved_batch_context,
        )

        training_state.joint_action_weight -= config.learning_rate * joint_action_weight_gradient
        training_state.joint_action_bias -= config.learning_rate * joint_action_bias_gradient
        next_value_weight = training_state.value_weight - (
            config.learning_rate * value_weight_gradient.astype(training_state.value_weight.dtype, copy=False)
        )
        next_value_bias = float(
            training_state.value_bias
            - (config.learning_rate * np.asarray(value_bias_gradient, dtype=np.float64).item())
        )
        _phase1a_require_finite_array(
            "updated_value_weight",
            next_value_weight,
            numerics=numerics,
            batch_context=resolved_batch_context,
        )
        _phase1a_require_finite_scalar(
            "updated_value_bias",
            next_value_bias,
            numerics=numerics,
            batch_context=resolved_batch_context,
        )
        training_state.value_weight = next_value_weight
        training_state.value_bias = next_value_bias
        return _Phase1ABatchStepResult(
            joint_ce_loss=float(joint_loss),
            aux_value_loss_raw=value_loss,
            aux_value_loss_weighted=float(aux_value_loss_weighted),
            total_loss=float(total_loss),
            action_logit_abs_max=action_logit_abs_max,
            action_entropy=action_entropy,
            numerics=numerics,
        )

    if isinstance(backend, _TorchLinearTrainingBackend):
        training_state = _expect_phase1a_torch_state(state)
        torch_module = backend._torch
        device = backend.device_resolution.compute_device
        batch_size = int(batch_joint_action_labels.shape[0])
        tensor_dtype = training_state.joint_action_weight.dtype
        x_batch = torch_module.tensor(batch_features, dtype=tensor_dtype, device=device)
        labels_batch = torch_module.tensor(batch_joint_action_labels, dtype=torch_module.int64, device=device)
        mask_batch = torch_module.tensor(batch_joint_action_masks, dtype=torch_module.bool, device=device)
        value_targets = torch_module.tensor(batch_value_targets, dtype=tensor_dtype, device=device)

        joint_logits = x_batch @ training_state.joint_action_weight.transpose(0, 1) + training_state.joint_action_bias
        action_logit_abs_max = float(torch_module.max(torch_module.abs(joint_logits)).item()) if batch_size > 0 else 0.0
        masked_joint_logits = torch_module.where(mask_batch, joint_logits, torch_module.full_like(joint_logits, -1.0e30))
        _phase1a_require_finite_tensor(
            torch_module,
            "joint_logits",
            masked_joint_logits,
            numerics=numerics,
            batch_context=resolved_batch_context,
        )
        joint_probabilities = torch_module.softmax(masked_joint_logits, dim=1)
        _phase1a_require_finite_tensor(
            torch_module,
            "joint_probabilities",
            joint_probabilities,
            numerics=numerics,
            batch_context=resolved_batch_context,
        )
        joint_loss = _torch_cross_entropy_loss(torch_module, joint_probabilities, labels_batch)
        action_entropy = _categorical_entropy_torch(torch_module, joint_probabilities)
        joint_gradient = joint_probabilities.clone()
        joint_gradient[
            torch_module.arange(batch_size, device=device),
            labels_batch,
        ] -= 1.0
        joint_gradient /= batch_size
        joint_gradient = torch_module.where(mask_batch, joint_gradient, torch_module.zeros_like(joint_gradient))
        joint_action_weight_gradient = joint_gradient.transpose(0, 1) @ x_batch
        joint_action_bias_gradient = joint_gradient.sum(dim=0)

        value_prediction = x_batch @ training_state.value_weight + training_state.value_bias
        value_pred_abs_max = (
            float(torch_module.max(torch_module.abs(value_prediction)).item()) if batch_size > 0 else 0.0
        )
        if math.isfinite(value_pred_abs_max):
            numerics.value_pred_abs_max = max(numerics.value_pred_abs_max, value_pred_abs_max)
        _phase1a_require_finite_tensor(
            torch_module,
            "value_prediction",
            value_prediction,
            numerics=numerics,
            batch_context=resolved_batch_context,
        )
        value_prediction_f64 = value_prediction.to(dtype=torch_module.float64)
        value_targets_f64 = value_targets.to(dtype=torch_module.float64)
        value_error = value_prediction_f64 - value_targets_f64
        value_loss = _huber_loss_torch(
            torch_module,
            value_error,
            delta=_PHASE1A_AUX_VALUE_HUBER_DELTA,
        )
        _phase1a_require_finite_tensor(
            torch_module,
            "value_loss",
            value_loss,
            numerics=numerics,
            batch_context=resolved_batch_context,
        )
        value_weight_gradient = torch_module.zeros_like(training_state.value_weight, dtype=torch_module.float64)
        value_bias_gradient = torch_module.tensor(0.0, dtype=torch_module.float64, device=device)
        aux_value_loss_weighted = 0.0
        total_loss = float(joint_loss.item())
        if config.aux_value_loss_weight > 0.0:
            aux_value_loss_weighted = config.aux_value_loss_weight * float(value_loss.item())
            total_loss += aux_value_loss_weighted
            huber_gradient = _huber_gradient_torch(
                torch_module,
                value_error,
                delta=_PHASE1A_AUX_VALUE_HUBER_DELTA,
            )
            value_scale = config.aux_value_loss_weight / batch_size
            value_weight_gradient = value_scale * (
                huber_gradient.unsqueeze(1) * x_batch.to(dtype=torch_module.float64)
            ).sum(dim=0)
            value_bias_gradient = value_scale * huber_gradient.sum()

        if config.l2_weight > 0.0:
            total_loss += config.l2_weight * (
                float(torch_module.sum(training_state.joint_action_weight**2).item())
                + float(torch_module.sum(training_state.value_weight.to(dtype=torch_module.float64) ** 2).item())
            )
            joint_action_weight_gradient = joint_action_weight_gradient + (
                config.l2_weight * training_state.joint_action_weight
            )
            value_weight_gradient = value_weight_gradient + (
                config.l2_weight * training_state.value_weight.to(dtype=torch_module.float64)
            )

        (
            value_weight_gradient,
            value_bias_gradient,
            value_grad_norm_pre_clip,
            value_grad_norm_post_clip,
            clip_applied,
        ) = _phase1a_clip_value_gradients_torch(
            torch_module=torch_module,
            weight_gradient=value_weight_gradient,
            bias_gradient=value_bias_gradient,
            clip_norm=_PHASE1A_VALUE_GRAD_CLIP_NORM,
        )
        if math.isfinite(value_grad_norm_pre_clip):
            numerics.value_grad_norm_pre_clip = max(
                numerics.value_grad_norm_pre_clip,
                value_grad_norm_pre_clip,
            )
        if math.isfinite(value_grad_norm_post_clip):
            numerics.value_grad_norm_post_clip = max(
                numerics.value_grad_norm_post_clip,
                value_grad_norm_post_clip,
            )
        numerics.clip_applied_count += int(clip_applied)
        _phase1a_require_finite_tensor(
            torch_module,
            "value_weight_gradient",
            value_weight_gradient,
            numerics=numerics,
            batch_context=resolved_batch_context,
        )
        _phase1a_require_finite_tensor(
            torch_module,
            "value_bias_gradient",
            value_bias_gradient,
            numerics=numerics,
            batch_context=resolved_batch_context,
        )

        training_state.joint_action_weight = training_state.joint_action_weight - (
            config.learning_rate * joint_action_weight_gradient
        )
        training_state.joint_action_bias = training_state.joint_action_bias - (
            config.learning_rate * joint_action_bias_gradient
        )
        next_value_weight = training_state.value_weight - (
            config.learning_rate * value_weight_gradient.to(dtype=tensor_dtype)
        )
        next_value_bias = training_state.value_bias - (
            config.learning_rate * value_bias_gradient.to(dtype=tensor_dtype)
        )
        _phase1a_require_finite_tensor(
            torch_module,
            "updated_value_weight",
            next_value_weight,
            numerics=numerics,
            batch_context=resolved_batch_context,
        )
        _phase1a_require_finite_tensor(
            torch_module,
            "updated_value_bias",
            next_value_bias,
            numerics=numerics,
            batch_context=resolved_batch_context,
        )
        training_state.value_weight = next_value_weight
        training_state.value_bias = next_value_bias
        return _Phase1ABatchStepResult(
            joint_ce_loss=float(joint_loss.item()),
            aux_value_loss_raw=float(value_loss.item()),
            aux_value_loss_weighted=float(aux_value_loss_weighted),
            total_loss=float(total_loss),
            action_logit_abs_max=action_logit_abs_max,
            action_entropy=action_entropy,
            numerics=numerics,
        )

    raise TypeError(f"unsupported backend for phase1a batch step: {type(backend)!r}")


def _phase1a_require_finite_array(
    component: str,
    value: np.ndarray | float,
    *,
    numerics: _Phase1ANumericsTelemetry,
    batch_context: dict[str, object],
) -> None:
    if not np.all(np.isfinite(np.asarray(value, dtype=np.float64))):
        if numerics.first_nonfinite_component is None:
            numerics.first_nonfinite_component = component
            numerics.first_nonfinite_batch_context = dict(batch_context)
        raise _Phase1ANumericsError(
            component=component,
            batch_context=dict(batch_context),
            numerics=_Phase1ANumericsTelemetry.from_mapping(numerics.as_dict()),
        )


def _phase1a_require_finite_scalar(
    component: str,
    value: float,
    *,
    numerics: _Phase1ANumericsTelemetry,
    batch_context: dict[str, object],
) -> None:
    _phase1a_require_finite_array(
        component,
        np.asarray([value], dtype=np.float64),
        numerics=numerics,
        batch_context=batch_context,
    )


def _phase1a_require_finite_tensor(
    torch_module: Any,
    component: str,
    value: Any,
    *,
    numerics: _Phase1ANumericsTelemetry,
    batch_context: dict[str, object],
) -> None:
    if not bool(torch_module.all(torch_module.isfinite(value)).item()):
        if numerics.first_nonfinite_component is None:
            numerics.first_nonfinite_component = component
            numerics.first_nonfinite_batch_context = dict(batch_context)
        raise _Phase1ANumericsError(
            component=component,
            batch_context=dict(batch_context),
            numerics=_Phase1ANumericsTelemetry.from_mapping(numerics.as_dict()),
        )


def _phase1a_clip_value_gradients_numpy(
    *,
    weight_gradient: np.ndarray,
    bias_gradient: float,
    clip_norm: float,
) -> tuple[np.ndarray, float, float, float, bool]:
    bias_value = float(bias_gradient)
    pre_clip_norm = float(
        np.sqrt(np.sum(np.square(weight_gradient, dtype=np.float64)) + (bias_value * bias_value))
    )
    if not math.isfinite(pre_clip_norm):
        return weight_gradient, bias_value, pre_clip_norm, pre_clip_norm, False
    if pre_clip_norm <= clip_norm or pre_clip_norm <= 0.0:
        return weight_gradient, bias_value, pre_clip_norm, pre_clip_norm, False
    scale = clip_norm / pre_clip_norm
    clipped_weight = weight_gradient * scale
    clipped_bias = bias_value * scale
    post_clip_norm = float(
        np.sqrt(np.sum(np.square(clipped_weight, dtype=np.float64)) + (clipped_bias * clipped_bias))
    )
    return clipped_weight, clipped_bias, pre_clip_norm, post_clip_norm, True


def _phase1a_clip_value_gradients_torch(
    *,
    torch_module: Any,
    weight_gradient: Any,
    bias_gradient: Any,
    clip_norm: float,
) -> tuple[Any, Any, float, float, bool]:
    pre_clip_norm = float(
        torch_module.sqrt(torch_module.sum(weight_gradient**2) + (bias_gradient * bias_gradient)).item()
    )
    if not math.isfinite(pre_clip_norm):
        return weight_gradient, bias_gradient, pre_clip_norm, pre_clip_norm, False
    if pre_clip_norm <= clip_norm or pre_clip_norm <= 0.0:
        return weight_gradient, bias_gradient, pre_clip_norm, pre_clip_norm, False
    scale = clip_norm / pre_clip_norm
    clipped_weight = weight_gradient * scale
    clipped_bias = bias_gradient * scale
    post_clip_norm = float(
        torch_module.sqrt(torch_module.sum(clipped_weight**2) + (clipped_bias * clipped_bias)).item()
    )
    return clipped_weight, clipped_bias, pre_clip_norm, post_clip_norm, True


def _huber_loss_numpy(errors: np.ndarray, *, delta: float) -> float:
    abs_error = np.abs(errors)
    quadratic = np.minimum(abs_error, delta)
    linear = abs_error - quadratic
    return float(np.mean(0.5 * np.square(quadratic) + (delta * linear)))


def _huber_gradient_numpy(errors: np.ndarray, *, delta: float) -> np.ndarray:
    abs_error = np.abs(errors)
    return np.where(abs_error <= delta, errors, delta * np.sign(errors))


def _huber_loss_torch(torch_module: Any, errors: Any, *, delta: float) -> Any:
    abs_error = torch_module.abs(errors)
    quadratic = torch_module.minimum(abs_error, torch_module.full_like(abs_error, delta))
    linear = abs_error - quadratic
    return torch_module.mean((0.5 * quadratic * quadratic) + (delta * linear))


def _huber_gradient_torch(torch_module: Any, errors: Any, *, delta: float) -> Any:
    delta_tensor = torch_module.full_like(errors, delta)
    return torch_module.where(torch_module.abs(errors) <= delta_tensor, errors, delta_tensor * torch_module.sign(errors))


def _phase1a_parameters(
    *,
    backend: _LinearTrainingBackend,
    state: object,
    joint_action_keys: list[str],
    venue_choices: list[str],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    config: TrainingConfig,
) -> LinearPolicyV2Parameters:
    if isinstance(backend, _NumpyLinearTrainingBackend):
        training_state = _expect_phase1a_numpy_state(state)
        return LinearPolicyV2Parameters(
            joint_action_keys=joint_action_keys,
            venue_choices=venue_choices,
            feature_mean=feature_mean.tolist(),
            feature_std=feature_std.tolist(),
            joint_action_weight=training_state.joint_action_weight.tolist(),
            joint_action_bias=training_state.joint_action_bias.tolist(),
            value_weight=training_state.value_weight.tolist(),
            value_bias=float(training_state.value_bias),
            preferred_size_band=config.preferred_size_band,
            preferred_leverage_band=config.preferred_leverage_band,
            joint_action_vocabulary_version=(
                config.joint_action_vocabulary_version or JOINT_ACTION_VOCABULARY_VERSION_PHASE1A
            ),
            policy_state_feature_version=(
                config.policy_state_feature_version or POLICY_STATE_FEATURE_VERSION_PHASE1A
            ),
        )
    if isinstance(backend, _TorchLinearTrainingBackend):
        training_state = _expect_phase1a_torch_state(state)
        return LinearPolicyV2Parameters(
            joint_action_keys=joint_action_keys,
            venue_choices=venue_choices,
            feature_mean=feature_mean.tolist(),
            feature_std=feature_std.tolist(),
            joint_action_weight=training_state.joint_action_weight.detach().cpu().tolist(),
            joint_action_bias=training_state.joint_action_bias.detach().cpu().tolist(),
            value_weight=training_state.value_weight.detach().cpu().tolist(),
            value_bias=float(training_state.value_bias.detach().cpu().item()),
            preferred_size_band=config.preferred_size_band,
            preferred_leverage_band=config.preferred_leverage_band,
            joint_action_vocabulary_version=(
                config.joint_action_vocabulary_version or JOINT_ACTION_VOCABULARY_VERSION_PHASE1A
            ),
            policy_state_feature_version=(
                config.policy_state_feature_version or POLICY_STATE_FEATURE_VERSION_PHASE1A
            ),
        )
    raise TypeError(f"unsupported backend for phase1a parameters: {type(backend)!r}")


def _phase1a_parameters_are_finite(parameters: LinearPolicyV2Parameters) -> bool:
    arrays = (
        np.asarray(parameters.feature_mean, dtype=np.float64),
        np.asarray(parameters.feature_std, dtype=np.float64),
        np.asarray(parameters.joint_action_weight, dtype=np.float64),
        np.asarray(parameters.joint_action_bias, dtype=np.float64),
        np.asarray(parameters.value_weight, dtype=np.float64),
        np.asarray([parameters.value_bias], dtype=np.float64),
    )
    return all(np.all(np.isfinite(array)) for array in arrays)


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
    return best_label_from_step(step)


def _softmax_matrix(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis=1, keepdims=True)


def _categorical_entropy(probabilities: np.ndarray) -> float:
    safe = np.clip(probabilities, 1.0e-12, 1.0)
    return float(np.mean(-np.sum(safe * np.log(safe), axis=1)))


def _categorical_entropy_torch(torch_module: Any, probabilities: Any) -> float:
    safe = torch_module.clamp(probabilities, min=1.0e-12, max=1.0)
    return float(torch_module.mean(-torch_module.sum(safe * torch_module.log(safe), dim=1)).item())


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


def _initial_phase1a_parameter_arrays(
    *,
    seed: int,
    joint_action_count: int,
    feature_dim: int,
    dtype: np.dtype[Any] | type[np.generic] = np.float64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    resolved_dtype = np.dtype(dtype)
    joint_action_weight = rng.normal(0.0, 0.01, size=(joint_action_count, feature_dim)).astype(resolved_dtype)
    joint_action_bias = np.zeros(joint_action_count, dtype=resolved_dtype)
    value_weight = np.zeros(feature_dim, dtype=resolved_dtype)
    value_bias = 0.0
    return joint_action_weight, joint_action_bias, value_weight, value_bias


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


def _expect_phase1a_numpy_state(state: object) -> _Phase1ANumpyTrainingState:
    if not isinstance(state, _Phase1ANumpyTrainingState):
        raise TypeError("expected Phase 1A NumPy training state")
    return state


def _expect_phase1a_torch_state(state: object) -> _Phase1ATorchTrainingState:
    if not isinstance(state, _Phase1ATorchTrainingState):
        raise TypeError("expected Phase 1A PyTorch training state")
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


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)

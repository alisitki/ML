from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import gc
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

logger = logging.getLogger(__name__)

TrainingBackendName = Literal["numpy", "pytorch"]


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
        """\u26a0\ufe0f  FIXTURE / TEST COMPAT PATH \u2014 do not call from production.

        Delegates to _finalize_prepared_data so fixture and streaming paths
        share identical normalization/packaging logic.
        """
        train_examples = self._build_examples(bundle, split="train")
        validation_examples = self._build_examples(bundle, split="validation")
        return self._finalize_prepared_data(
            train_examples,
            validation_examples,
            bundle.action_space.action_keys,
            bundle.dataset_spec.exchanges,
        )

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
                prepared=prepared,
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
        prepared: _PreparedTrainingData,
        candidate_run: _CandidateTrainingRun,
        selection_run: _CandidateSelectionRun,
        training_run_id: str,
        candidate_rank: int,
        selected_candidate: bool,
        search_budget_summary: SearchBudgetSummary,
    ) -> dict[str, object]:
        return {
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
            "training_loss_history": candidate_run.loss_history,
            "validation_objective_history": candidate_run.validation_history,
            "search_budget_summary": search_budget_summary.model_dump(mode="json"),
        }

    def _build_examples(self, bundle: TrajectoryBundle, split: str) -> list[_TrainingExample]:
        trajectories = bundle.splits.get(split, [])
        examples: list[_TrainingExample] = []
        for trajectory in trajectories:
            for step in trajectory.steps:
                features = np.asarray(
                    observation_feature_vector(step.observation), dtype=np.float32
                )
                action_key, venue = _best_label(step)
                examples.append(_TrainingExample(features=features, action_key=action_key, venue=venue))
        return examples

    def _validation_report(self, bundle: TrajectoryBundle, artifact: PolicyArtifact) -> EvaluationReport:
        from quantlab_ml.evaluation import EvaluationEngine

        boundary = EvaluationBoundary(
            fee_handling="shared_reward_contract",
            funding_handling="carry_from_funding_stream",
            slippage_handling="fixed_bps",
            fill_assumption_mode=bundle.reward_spec.timestamping,
            timeout_semantics="force_terminal_at_data_end",
            terminal_semantics="trajectory_boundary_is_terminal",
            infeasible_action_treatment="force_abstain",
        )
        return EvaluationEngine(boundary).evaluate(bundle, artifact, split="validation")

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

        # Final refit on full train + validation splits
        prepared = self._prepare_training_data_streaming(
            manifest, directory, TrajectoryDirectoryStore
        )

        candidate_results: list[TrainingCandidateResult] = []
        for candidate_rank, selection_run in enumerate(ranked_selections, start=1):
            selected_candidate = candidate_rank == 1
            candidate_run = self._train_candidate_from_manifest(
                manifest=manifest,
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
                prepared=prepared,
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

    def _build_examples_streaming(
        self,
        directory: Path,
        split_name: str,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        exclusive_end: datetime | None = None,
        store_cls: Any,
    ) -> list[_TrainingExample]:
        """Stream JSONL records, apply optional time window, return _TrainingExample list.

        Feature extraction is done immediately per step; TrajectoryStep objects
        are not accumulated.  The returned list holds only (features, action_key,
        venue) tuples — ~750 floats × 4 bytes = ~3 KB per example.
        """
        examples: list[_TrainingExample] = []
        for record in store_cls.iter_records(directory, split_name):
            for step in record.steps:
                t = step.event_time
                if start is not None and t < start:
                    continue
                if end is not None and t > end:
                    continue
                if exclusive_end is not None and t >= exclusive_end:
                    continue
                features = np.asarray(
                    observation_feature_vector(step.observation), dtype=np.float32
                )
                action_key, venue = _best_label(step)
                examples.append(_TrainingExample(features=features, action_key=action_key, venue=venue))
        return examples

    def _prepare_training_data_streaming(
        self,
        manifest: TrajectoryManifest,
        directory: Path,
        store_cls: type,
    ) -> _PreparedTrainingData:
        """Stream train.jsonl and validation.jsonl to build _PreparedTrainingData."""
        train_examples = self._build_examples_streaming(
            directory, "train", store_cls=store_cls
        )
        validation_examples = self._build_examples_streaming(
            directory, "validation", store_cls=store_cls
        )
        return self._finalize_prepared_data(
            train_examples, validation_examples,
            manifest.action_space.action_keys,
            manifest.dataset_spec.exchanges,
        )

    def _finalize_prepared_data(
        self,
        train_examples: list[_TrainingExample],
        validation_examples: list[_TrainingExample],
        action_keys: list[str],
        venue_choices: list[str],
    ) -> _PreparedTrainingData:
        """Convert lists of _TrainingExample to memory-efficient numpy matrices.

        Strategy (production-scale, e.g. 9590 steps, 680K features):
        1. Extract scalar labels first (tiny arrays).
        2. Pre-allocate float32 matrix and fill row-by-row.
        3. Delete the examples list immediately after filling (frees ~26 GB).
        4. Normalize in-place (no second copy).
        5. Repeat for validation (smaller: ~6.5 GB).

        Peak RAM: ~52 GB during fill (examples + matrix coexist briefly).
        After finalize: ~26 GB train + ~6.5 GB val + small arrays.
        """
        if not train_examples:
            raise ValueError("train split is empty")
        if not validation_examples:
            raise ValueError("validation split is empty")

        n_train = len(train_examples)
        n_val = len(validation_examples)
        feat_dim = int(np.asarray(train_examples[0].features).shape[0])

        # --- train labels (tiny; extract before del) ---
        action_labels = np.asarray(
            [action_keys.index(e.action_key) for e in train_examples], dtype=np.int64
        )
        venue_mask = np.asarray([e.venue is not None for e in train_examples], dtype=np.bool_)
        venue_labels = np.asarray(
            [venue_choices.index(e.venue) if e.venue is not None else 0 for e in train_examples],
            dtype=np.int64,
        )

        # --- train matrix: pre-allocate float32, fill row-by-row, then del ---
        train_matrix = np.empty((n_train, feat_dim), dtype=np.float32)
        for i, ex in enumerate(train_examples):
            train_matrix[i] = ex.features
        del train_examples
        gc.collect()

        # --- normalization stats (from train only; fit on train, never val) ---
        feature_mean = train_matrix.mean(axis=0)
        feature_std = train_matrix.std(axis=0)
        feature_std = np.where(feature_std < 1e-6, 1.0, feature_std).astype(np.float32)
        feature_mean = feature_mean.astype(np.float32)

        # --- normalize train in-place (no third copy) ---
        train_matrix -= feature_mean
        train_matrix /= feature_std
        normalized_train = train_matrix  # same object

        # --- val labels + matrix ---
        val_action_labels = np.asarray(
            [action_keys.index(e.action_key) for e in validation_examples], dtype=np.int64
        )
        val_matrix = np.empty((n_val, feat_dim), dtype=np.float32)
        for i, ex in enumerate(validation_examples):
            val_matrix[i] = ex.features
        del validation_examples
        gc.collect()
        val_matrix -= feature_mean
        val_matrix /= feature_std
        normalized_val = val_matrix  # same object

        logger.info(
            "training_data_prepared train_examples=%d validation_examples=%d "
            "feature_dim=%d action_count=%d venue_count=%d",
            n_train, n_val, feat_dim, len(action_keys), len(venue_choices),
        )
        return _PreparedTrainingData(
            train_step_count=n_train,
            val_step_count=n_val,
            action_keys=action_keys,
            venue_choices=venue_choices,
            normalized_train=normalized_train,
            normalized_val=normalized_val,
            feature_mean=feature_mean,
            feature_std=feature_std,
            action_labels=action_labels,
            val_action_labels=val_action_labels,
            venue_mask=venue_mask,
            venue_labels=venue_labels,
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
            train_examples = self._build_examples_streaming(
                directory,
                "development",
                start=fold.train_window.start,
                end=fold.train_window.end,
                exclusive_end=purge_cutoff if fold.purge_width_steps > 0 else None,
                store_cls=TrajectoryDirectoryStore,
            )
            val_examples = self._build_examples_streaming(
                directory,
                "development",
                start=fold.validation_window.start,
                end=fold.validation_window.end,
                store_cls=TrajectoryDirectoryStore,
            )
            fold_prepared = self._finalize_prepared_data(
                train_examples, val_examples,
                manifest.action_space.action_keys,
                manifest.dataset_spec.exchanges,
            )
            fold_run = self._train_candidate_from_manifest(
                manifest=manifest,
                prepared=fold_prepared,
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
            # Free the large normalized matrices before the next fold allocates
            # its own (fold_k normalized_train is 22-26 GB; without explicit
            # del, Python's refcount may not release it before fold_k+1 alloc).
            del fold_prepared
            gc.collect()
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
        prepared: _PreparedTrainingData,
        candidate_spec: TrainingCandidateSpec,
        candidate_index: int,
        parent_policy_id: str | None,
        training_run_id: str,
        code_commit_hash: str,
    ) -> _CandidateTrainingRun:
        """Train all epochs using streaming-compatible proxy validation.

        Unlike the in-memory path (_train_candidate), this method does NOT do
        per-epoch early stopping via the full EvaluationEngine.  It trains
        config.epochs epochs and scores the final model with a cross-entropy
        proxy on validation features (_compute_proxy_validation_score).

        The proxy metric has the same optimisation direction as actual performance
        and is sufficient for fold candidate ranking.  The final selected policy
        is evaluated accurately by the 'evaluate' CLI command (EvaluationEngine on
        final_untouched_test.jsonl).
        """
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

        for epoch in range(1, config.epochs + 1):
            total_loss = self._backend.step(state=state, prepared=prepared, config=config)
            loss_history.append(total_loss)

        parameters = self._backend.parameters(state=state, prepared=prepared, config=config)
        proxy_return, proxy_score = _compute_proxy_validation_score(prepared, parameters)

        logger.info(
            "training_candidate_completed candidate_index=%d seed=%d "
            "epochs=%d proxy_validation_return=%.6f training_backend=%s",
            candidate_index, candidate_spec.seed, config.epochs,
            proxy_return, self._backend.backend_name,
        )
        return _CandidateTrainingRun(
            config=config,
            candidate_spec=candidate_spec,
            candidate_index=candidate_index,
            best_epoch=config.epochs,
            best_parameters=parameters,
            best_validation_total_net_return=proxy_return,
            best_validation_score=proxy_score,
            loss_history=loss_history,
            validation_history=[],
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
        prepared: _PreparedTrainingData,
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

        validation_step_count = max(prepared.val_step_count, 1)
        expected_return_score = validation_total_net_return / validation_step_count
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
class _TrainingExample:
    features: np.ndarray  # float32, shape (feature_dim,) -- compact; NOT list[float]
    action_key: str
    venue: str | None


@dataclass(slots=True)
class _PreparedTrainingData:
    """Normalized training and validation matrices ready for gradient computation.

    Memory layout (production scale, e.g. 9590 train + 2390 val steps, 680 K features):
      normalized_train : float32  (n_train, feat_dim)  ~26 GB
      normalized_val   : float32  (n_val,   feat_dim)  ~6.5 GB
      feature_mean/std : float32  (feat_dim,)           ~3 MB each
      label arrays     : int64/bool                     tiny

    The raw _TrainingExample lists are intentionally discarded in
    _finalize_prepared_data() after building these matrices to keep peak RAM
    within the 75 GB cgroup limit.
    """

    train_step_count: int
    val_step_count: int
    action_keys: list[str]
    venue_choices: list[str]
    normalized_train: np.ndarray   # float32, shape (n_train, feat_dim)
    normalized_val: np.ndarray     # float32, shape (n_val,   feat_dim)
    feature_mean: np.ndarray       # float32, shape (feat_dim,)
    feature_std: np.ndarray        # float32, shape (feat_dim,)
    action_labels: np.ndarray      # int64,   shape (n_train,)
    val_action_labels: np.ndarray  # int64,   shape (n_val,)
    venue_mask: np.ndarray         # bool,    shape (n_train,)
    venue_labels: np.ndarray       # int64,   shape (n_train,)

    @property
    def feature_dim(self) -> int:
        return int(self.normalized_train.shape[1])


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

    def parameters(
        self,
        *,
        state: object,
        prepared: _PreparedTrainingData,
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
        training_state = _expect_numpy_state(state)
        action_logits = prepared.normalized_train @ training_state.action_weight.T + training_state.action_bias
        action_probabilities = _softmax_matrix(action_logits)
        action_loss = _cross_entropy_loss(action_probabilities, prepared.action_labels)
        action_gradient = action_probabilities.copy()
        action_gradient[np.arange(len(prepared.action_labels)), prepared.action_labels] -= 1.0
        action_gradient /= len(prepared.action_labels)

        action_weight_gradient = action_gradient.T @ prepared.normalized_train
        action_bias_gradient = action_gradient.sum(axis=0)

        venue_loss = 0.0
        venue_weight_gradient = np.zeros_like(training_state.venue_weight)
        venue_bias_gradient = np.zeros_like(training_state.venue_bias)
        if prepared.venue_mask.any():
            masked_inputs = prepared.normalized_train[prepared.venue_mask]
            masked_labels = prepared.venue_labels[prepared.venue_mask]
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
        prepared: _PreparedTrainingData,
        config: TrainingConfig,
    ) -> LinearPolicyParameters:
        training_state = _expect_numpy_state(state)
        return _build_linear_policy_parameters(
            prepared=prepared,
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
        training_state = _expect_torch_state(state)
        torch_module = self._torch
        device = self.device_resolution.compute_device

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
        x_batch = torch_module.tensor(
            prepared.normalized_train[batch_idx].astype(np.float64),
            dtype=torch_module.float64,
            device=device,
        )
        labels_batch = torch_module.tensor(
            prepared.action_labels[batch_idx], dtype=torch_module.int64, device=device,
        )
        venue_mask_batch = torch_module.tensor(
            prepared.venue_mask[batch_idx], dtype=torch_module.bool, device=device,
        )
        venue_labels_batch = torch_module.tensor(
            prepared.venue_labels[batch_idx], dtype=torch_module.int64, device=device,
        )

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
            venue_gradient /= batch_size  # normalize by full batch for gradient consistency
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
        prepared: _PreparedTrainingData,
        config: TrainingConfig,
    ) -> LinearPolicyParameters:
        training_state = _expect_torch_state(state)
        return _build_linear_policy_parameters(
            prepared=prepared,
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


def _normalize(matrix: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (matrix - mean) / std


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


def _compute_proxy_validation_score(
    prepared: _PreparedTrainingData,
    parameters: LinearPolicyParameters,
) -> tuple[float, "PolicyScore"]:
    """Compute a cross-entropy proxy validation score (streaming training path).

    Uses prepared.normalized_val and prepared.val_action_labels (precomputed in
    _finalize_prepared_data) to avoid rebuilding the val matrix every epoch.

    Deliberate trade-off: no fee/slippage/funding accounting.  Used for fold
    candidate RANKING only.  The final selected policy is evaluated accurately
    by the 'evaluate' CLI command (full EvaluationEngine, final_untouched_test).
    """
    if prepared.val_step_count == 0:
        proxy_return = 0.0
        proxy_rank = 0.5
    else:
        W = np.array(parameters.action_weight, dtype=np.float64)   # (A, F)
        b = np.array(parameters.action_bias, dtype=np.float64)      # (A,)
        # normalized_val is float32; upcast to float64 for matrix multiply precision
        norm_val = prepared.normalized_val.astype(np.float64)
        logits = norm_val @ W.T + b                                 # (N, A)
        probs = _softmax_matrix(logits)
        val_labels = prepared.val_action_labels
        ce = _cross_entropy_loss(probs, val_labels)
        proxy_return = -ce
        # Map CE to [0, 1]: lower CE → higher rank; CE=log(A) → 0.5 (random baseline)
        random_ce = float(np.log(max(len(prepared.action_keys), 2)))
        proxy_rank = float(np.clip(1.0 - ce / max(random_ce * 2.0, 1e-6), 0.0, 0.99))

    stub_id = f"proxy-{abs(hash(proxy_return)):012x}"
    score = PolicyScore(
        policy_id=stub_id,
        evaluation_id=stub_id,
        created_at=utcnow(),
        expected_return_score=proxy_return,
        risk_score=0.5,
        turnover_score=0.5,
        stability_score=proxy_rank,
        applicability_score=0.5,
        composite_rank=proxy_rank,
    )
    return proxy_return, score


def _build_linear_policy_parameters(
    *,
    prepared: _PreparedTrainingData,
    config: TrainingConfig,
    action_weight: list[list[float]],
    action_bias: list[float],
    venue_weight: list[list[float]],
    venue_bias: list[float],
) -> LinearPolicyParameters:
    return LinearPolicyParameters(
        action_keys=prepared.action_keys,
        venue_choices=prepared.venue_choices,
        feature_mean=prepared.feature_mean.tolist(),
        feature_std=prepared.feature_std.tolist(),
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

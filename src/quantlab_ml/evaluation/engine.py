from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import Any

import numpy as np

from quantlab_ml.common import utcnow
from quantlab_ml.contracts import (
    DYNAMIC_TARGET_ASSET,
    DatasetSpec,
    EvaluationBoundary,
    EvaluationReport,
    PolicyArtifact,
    PolicyState,
    RewardEventSpec,
    TimeRange,
    TrajectoryBundle,
    TrajectoryManifest,
    TrajectoryRecord,
    TrajectoryStep,
)
from quantlab_ml.models import LinearPolicyParameters, RuntimeDecision
from quantlab_ml.policies import PolicyRuntimeBridge
from quantlab_ml.rewards import RewardEngine
from quantlab_ml.runtime_contract import (
    build_strict_runtime_contract,
    canonical_raw_surface_shapes,
    scale_specs_match,
)
from quantlab_ml.trajectories.tensor_cache import (
    TENSOR_CACHE_FORMAT_VERSION,
    TensorCacheManifest,
    has_tensor_cache,
    load_tensor_cache_shard,
    read_tensor_cache_manifest,
    window_row_indices,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _CompiledLinearPolicy:
    action_keys: list[str]
    venue_choices: list[str]
    feature_mean: np.ndarray
    feature_std: np.ndarray
    action_weight: np.ndarray
    action_bias: np.ndarray
    venue_weight: np.ndarray
    venue_bias: np.ndarray
    preferred_size_band: str
    preferred_leverage_band: str

    @classmethod
    def from_artifact(cls, artifact: PolicyArtifact) -> "_CompiledLinearPolicy":
        if artifact.policy_payload.runtime_adapter != "linear-policy-v1":
            raise ValueError(
                f"unsupported runtime adapter for tensor-cache evaluation: {artifact.policy_payload.runtime_adapter!r}"
            )
        parameters = LinearPolicyParameters.model_validate_json(artifact.policy_payload.blob)
        feature_mean = np.asarray(parameters.feature_mean, dtype=np.float64)
        feature_std = np.asarray(parameters.feature_std, dtype=np.float64)
        safe_feature_std = np.where(np.abs(feature_std) > 1e-9, feature_std, 1.0)
        return cls(
            action_keys=list(parameters.action_keys),
            venue_choices=list(parameters.venue_choices),
            feature_mean=feature_mean,
            feature_std=safe_feature_std,
            action_weight=np.asarray(parameters.action_weight, dtype=np.float64),
            action_bias=np.asarray(parameters.action_bias, dtype=np.float64),
            venue_weight=np.asarray(parameters.venue_weight, dtype=np.float64),
            venue_bias=np.asarray(parameters.venue_bias, dtype=np.float64),
            preferred_size_band=parameters.preferred_size_band,
            preferred_leverage_band=parameters.preferred_leverage_band,
        )

    @property
    def feature_dim(self) -> int:
        return int(self.feature_mean.shape[0])

    def decide_batch(self, raw_features: np.ndarray) -> list[RuntimeDecision]:
        normalized = raw_features.astype(np.float64, copy=False)
        normalized -= self.feature_mean
        normalized /= self.feature_std

        action_logits = normalized @ self.action_weight.transpose(1, 0) + self.action_bias
        action_probabilities = _softmax_matrix(action_logits)
        action_indices = np.argmax(action_probabilities, axis=1)

        venue_logits = normalized @ self.venue_weight.transpose(1, 0) + self.venue_bias
        venue_probabilities = _softmax_matrix(venue_logits)
        venue_indices = np.argmax(venue_probabilities, axis=1)

        decisions: list[RuntimeDecision] = []
        for row_index, action_index in enumerate(action_indices):
            action_key = self.action_keys[int(action_index)]
            action_confidence = float(action_probabilities[row_index, action_index])
            if action_key == "abstain":
                decisions.append(RuntimeDecision(action_key="abstain", confidence=action_confidence))
                continue

            venue_index = int(venue_indices[row_index])
            venue_confidence = float(venue_probabilities[row_index, venue_index])
            decisions.append(
                RuntimeDecision(
                    action_key=action_key,
                    venue=self.venue_choices[venue_index],
                    size_band_key=self.preferred_size_band,
                    leverage_band_key=self.preferred_leverage_band,
                    confidence=(action_confidence + venue_confidence) / 2.0,
                )
            )
        return decisions


class EvaluationEngine:
    def __init__(self, boundary: EvaluationBoundary):
        self.boundary = boundary
        self.runtime_bridge = PolicyRuntimeBridge()

    def evaluate(self, bundle: TrajectoryBundle, artifact: PolicyArtifact, split: str = "validation") -> EvaluationReport:
        return self.evaluate_records(
            bundle.dataset_spec,
            bundle.reward_spec,
            bundle.splits[split],
            artifact,
        )

    def evaluate_records(
        self,
        dataset_spec: DatasetSpec,
        reward_spec: RewardEventSpec,
        trajectories: Iterable[TrajectoryRecord],
        artifact: PolicyArtifact,
    ) -> EvaluationReport:
        self._validate_boundary(reward_spec)
        reward_engine = RewardEngine(reward_spec, artifact.action_space)
        rewards: list[float] = []
        total_steps = 0
        realized_trade_count = 0
        infeasible_action_count = 0
        infeasible_penalty_total = 0.0
        risk_penalty_total = 0.0
        turnover_penalty_total = 0.0
        fee_total = 0.0
        funding_total = 0.0
        slippage_total = 0.0
        action_counts = {action.key: 0 for action in artifact.action_space.actions}
        notes: list[str] = []
        first_step_time = None
        last_step_time = None
        saw_trajectory = False

        for trajectory in trajectories:
            saw_trajectory = True
            current_policy_state = PolicyState()
            for step in trajectory.steps:
                total_steps += 1
                if first_step_time is None:
                    first_step_time = step.event_time
                last_step_time = step.event_time
                decision = self.runtime_bridge.decide(artifact, step.observation)
                applied = reward_engine.apply_decision(
                    snapshot=step.reward_snapshot,
                    requested_action_key=decision.action_key,
                    action_feasibility=step.action_feasibility,
                    infeasible_action_treatment=self.boundary.infeasible_action_treatment,
                    venue=decision.venue,
                    size_band_key=decision.size_band_key,
                    leverage_band_key=decision.leverage_band_key,
                    policy_state=current_policy_state,
                )
                if applied.reward_context is not None:
                    step.reward_context = applied.reward_context
                    step.reward_snapshot.context = applied.reward_context
                if applied.infeasible:
                    infeasible_action_count += 1
                    infeasible_penalty_total += applied.infeasible_penalty
                    notes.append(f"infeasible:{decision.action_key}:{step.event_time.isoformat()}")
                rewards.append(applied.net_reward)
                action_counts[applied.applied_action_key] = action_counts.get(applied.applied_action_key, 0) + 1
                risk_penalty_total += applied.risk_penalty
                turnover_penalty_total += applied.turnover_penalty
                fee_total += applied.fee
                funding_total += applied.funding
                slippage_total += applied.slippage
                if applied.applied_action_key != "abstain":
                    realized_trade_count += 1
                current_policy_state = reward_engine.advance_policy_state(current_policy_state, applied)

        if not saw_trajectory or first_step_time is None or last_step_time is None:
            raise ValueError("evaluation trajectories are empty")

        active_range = TimeRange(
            start=first_step_time,
            end=last_step_time + timedelta(seconds=dataset_spec.sampling_interval_seconds),
        )
        total_net_return = sum(rewards)
        return EvaluationReport(
            policy_id=artifact.policy_id,
            evaluation_id=f"eval-{artifact.policy_id}",
            created_at=utcnow(),
            boundary=self.boundary,
            total_steps=total_steps,
            realized_trade_count=realized_trade_count,
            infeasible_action_count=infeasible_action_count,
            infeasible_penalty_total=infeasible_penalty_total,
            total_net_return=total_net_return,
            average_net_return=total_net_return / max(total_steps, 1),
            risk_penalty_total=risk_penalty_total,
            turnover_penalty_total=turnover_penalty_total,
            fee_total=fee_total,
            funding_total=funding_total,
            slippage_total=slippage_total,
            action_counts=action_counts,
            step_reward_std=np.std(np.asarray(rewards, dtype=np.float64)).item() if len(rewards) > 1 else 0.0,
            coverage_symbols=dataset_spec.symbols,
            coverage_venues=dataset_spec.exchanges,
            coverage_streams=dataset_spec.stream_universe,
            active_date_range=active_range,
            notes=notes,
        )

    def evaluate_directory(
        self,
        *,
        manifest: TrajectoryManifest,
        directory: Path,
        artifact: PolicyArtifact,
        split_name: str,
        start: datetime | None = None,
        end: datetime | None = None,
        exclusive_end: datetime | None = None,
        cache_manifest: TensorCacheManifest | None = None,
        allow_jsonl_fallback: bool = False,
    ) -> EvaluationReport:
        resolved_cache_manifest = cache_manifest
        if resolved_cache_manifest is None and has_tensor_cache(directory):
            resolved_cache_manifest = read_tensor_cache_manifest(directory)
        if resolved_cache_manifest is not None:
            return self._evaluate_tensor_cache(
                manifest=manifest,
                directory=directory,
                artifact=artifact,
                cache_manifest=resolved_cache_manifest,
                split_name=split_name,
                start=start,
                end=end,
                exclusive_end=exclusive_end,
            )
        if not allow_jsonl_fallback:
            raise ValueError(
                "tensor cache manifest missing for trajectory directory; "
                "pass allow_jsonl_fallback=True only for temporary compatibility maintenance"
            )
        from quantlab_ml.trajectories.streaming_store import TrajectoryDirectoryStore

        logger.warning(
            "evaluation_directory_tensor_cache_missing path=%s split=%s tensor_cache_used=false "
            "jsonl_fallback_used=true path_classification=temporary_compatibility_maintenance",
            directory,
            split_name,
        )
        return self.evaluate_records(
            manifest.dataset_spec,
            manifest.reward_spec,
            self._iter_directory_records(
                directory=directory,
                split_name=split_name,
                start=start,
                end=end,
                exclusive_end=exclusive_end,
                store_cls=TrajectoryDirectoryStore,
            ),
            artifact,
        )

    def _evaluate_tensor_cache(
        self,
        *,
        manifest: TrajectoryManifest,
        directory: Path,
        artifact: PolicyArtifact,
        cache_manifest: TensorCacheManifest,
        split_name: str,
        start: datetime | None,
        end: datetime | None,
        exclusive_end: datetime | None,
    ) -> EvaluationReport:
        import time as _time

        self._validate_boundary(manifest.reward_spec)
        compiled_policy = _CompiledLinearPolicy.from_artifact(artifact)
        self._validate_tensor_cache_contract(
            manifest=manifest,
            artifact=artifact,
            cache_manifest=cache_manifest,
            compiled_policy=compiled_policy,
        )
        split_manifest = cache_manifest.splits.get(split_name)
        if split_manifest is None:
            raise ValueError(f"tensor cache is missing split {split_name!r}")

        logger.info(
            "tensor_cache_evaluation_started split=%s tensor_cache_format=%s tensor_cache_used=true "
            "jsonl_fallback_used=false compiled_policy_mode=tensor_cache_linear_policy_batch "
            "tensor_cache_shard_count=%d cache_feature_dim=%d",
            split_name,
            cache_manifest.format_version,
            split_manifest.shard_count,
            cache_manifest.feature_dim,
        )

        reward_engine = RewardEngine(manifest.reward_spec, artifact.action_space)
        rewards: list[float] = []
        total_steps = 0
        realized_trade_count = 0
        infeasible_action_count = 0
        infeasible_penalty_total = 0.0
        risk_penalty_total = 0.0
        turnover_penalty_total = 0.0
        fee_total = 0.0
        funding_total = 0.0
        slippage_total = 0.0
        action_counts = {action.key: 0 for action in artifact.action_space.actions}
        notes: list[str] = []
        first_step_time = None
        last_step_time = None
        current_policy_state = PolicyState()
        saw_rows = False
        started_at = _time.perf_counter()

        for shard in split_manifest.shards:
            loaded = load_tensor_cache_shard(directory, shard)
            row_idx = window_row_indices(
                loaded.event_time_ms,
                start=start,
                end=end,
                exclusive_end=exclusive_end,
            )
            if row_idx.size == 0:
                continue
            selected_rows = [loaded.replay_rows[int(index)] for index in row_idx]
            decisions = compiled_policy.decide_batch(loaded.features[row_idx])

            for local_index, row in enumerate(selected_rows):
                absolute_index = int(row_idx[local_index])
                if bool(loaded.trajectory_start[absolute_index]) != row.trajectory_start:
                    raise ValueError(
                        "tensor cache trajectory_start mismatch between metadata and replay sidecar: "
                        f"split={split_name!r}, shard={shard.shard_index}, row={absolute_index}"
                    )
                if row.trajectory_start:
                    current_policy_state = PolicyState()
                if artifact.target_asset != DYNAMIC_TARGET_ASSET and row.target_symbol != artifact.target_asset:
                    raise ValueError("artifact target_asset is incompatible with tensor-cache target_symbol")

                saw_rows = True
                total_steps += 1
                if first_step_time is None:
                    first_step_time = row.event_time
                last_step_time = row.event_time
                decision = decisions[local_index]
                applied = reward_engine.apply_decision(
                    snapshot=row.reward_snapshot,
                    requested_action_key=decision.action_key,
                    action_feasibility=row.action_feasibility,
                    infeasible_action_treatment=self.boundary.infeasible_action_treatment,
                    venue=decision.venue,
                    size_band_key=decision.size_band_key,
                    leverage_band_key=decision.leverage_band_key,
                    policy_state=current_policy_state,
                )
                if applied.infeasible:
                    infeasible_action_count += 1
                    infeasible_penalty_total += applied.infeasible_penalty
                    notes.append(f"infeasible:{decision.action_key}:{row.event_time.isoformat()}")
                rewards.append(applied.net_reward)
                action_counts[applied.applied_action_key] = action_counts.get(applied.applied_action_key, 0) + 1
                risk_penalty_total += applied.risk_penalty
                turnover_penalty_total += applied.turnover_penalty
                fee_total += applied.fee
                funding_total += applied.funding
                slippage_total += applied.slippage
                if applied.applied_action_key != "abstain":
                    realized_trade_count += 1
                current_policy_state = reward_engine.advance_policy_state(current_policy_state, applied)

        if not saw_rows or first_step_time is None or last_step_time is None:
            raise ValueError("evaluation trajectories are empty")

        wall_sec = _time.perf_counter() - started_at
        logger.info(
            "tensor_cache_evaluation_completed split=%s total_steps=%d wall_sec=%.1f "
            "evaluation_rows_per_sec=%.2f tensor_cache_used=true jsonl_fallback_used=false "
            "compiled_policy_mode=tensor_cache_linear_policy_batch",
            split_name,
            total_steps,
            wall_sec,
            total_steps / max(wall_sec, 1e-9),
        )

        active_range = TimeRange(
            start=first_step_time,
            end=last_step_time + timedelta(seconds=manifest.dataset_spec.sampling_interval_seconds),
        )
        total_net_return = sum(rewards)
        return EvaluationReport(
            policy_id=artifact.policy_id,
            evaluation_id=f"eval-{artifact.policy_id}",
            created_at=utcnow(),
            boundary=self.boundary,
            total_steps=total_steps,
            realized_trade_count=realized_trade_count,
            infeasible_action_count=infeasible_action_count,
            infeasible_penalty_total=infeasible_penalty_total,
            total_net_return=total_net_return,
            average_net_return=total_net_return / max(total_steps, 1),
            risk_penalty_total=risk_penalty_total,
            turnover_penalty_total=turnover_penalty_total,
            fee_total=fee_total,
            funding_total=funding_total,
            slippage_total=slippage_total,
            action_counts=action_counts,
            step_reward_std=np.std(np.asarray(rewards, dtype=np.float64)).item() if len(rewards) > 1 else 0.0,
            coverage_symbols=manifest.dataset_spec.symbols,
            coverage_venues=manifest.dataset_spec.exchanges,
            coverage_streams=manifest.dataset_spec.stream_universe,
            active_date_range=active_range,
            notes=notes,
        )

    def _iter_directory_records(
        self,
        *,
        directory: Path,
        split_name: str,
        start: datetime | None,
        end: datetime | None,
        exclusive_end: datetime | None,
        store_cls: Any,
    ) -> Iterator[TrajectoryRecord]:
        for record in store_cls.iter_records(directory, split_name):
            selected_steps = [
                step
                for step in record.steps
                if self._window_includes(
                    step,
                    start=start,
                    end=end,
                    exclusive_end=exclusive_end,
                )
            ]
            if not selected_steps:
                continue
            trajectory_id = record.trajectory_id
            if record.split != split_name:
                trajectory_id = f"{split_name}-{record.trajectory_id}"
            yield TrajectoryRecord(
                trajectory_id=trajectory_id,
                split=split_name,  # type: ignore[arg-type]
                target_symbol=record.target_symbol,
                start_time=selected_steps[0].event_time,
                end_time=selected_steps[-1].event_time,
                steps=selected_steps,
                terminal=record.terminal,
                terminal_reason=record.terminal_reason,
            )

    def _validate_tensor_cache_contract(
        self,
        *,
        manifest: TrajectoryManifest,
        artifact: PolicyArtifact,
        cache_manifest: TensorCacheManifest,
        compiled_policy: _CompiledLinearPolicy,
    ) -> None:
        if cache_manifest.format_version != TENSOR_CACHE_FORMAT_VERSION:
            raise ValueError(f"unsupported tensor cache format: {cache_manifest.format_version!r}")
        strict_contract = artifact.runtime_metadata.strict_runtime_contract or build_strict_runtime_contract(
            artifact.observation_schema,
            policy_kind=artifact.policy_payload.runtime_adapter,
        )
        expected_contract = build_strict_runtime_contract(
            manifest.observation_schema,
            policy_kind=artifact.policy_payload.runtime_adapter,
        )
        if artifact.reward_version != manifest.reward_spec.reward_version:
            raise ValueError("artifact reward version does not match evaluation reward surface")
        if artifact.runtime_metadata.observation_schema_version != manifest.observation_schema.schema_version:
            raise ValueError("artifact observation schema version does not match evaluation surface")
        if artifact.allowed_venues != manifest.dataset_spec.exchanges:
            raise ValueError("artifact allowed_venues do not match evaluation venue set")
        if artifact.runtime_metadata.required_streams != manifest.dataset_spec.stream_universe:
            raise ValueError("artifact required_streams do not match evaluation stream universe")
        for stream in manifest.dataset_spec.stream_universe:
            if artifact.runtime_metadata.required_field_families.get(stream, []) != manifest.observation_schema.field_axis.get(stream, []):
                raise ValueError(f"artifact required_field_families mismatch for stream {stream!r}")
        if artifact.runtime_metadata.required_scale_preset != [scale.label for scale in manifest.trajectory_spec.scale_preset]:
            raise ValueError("artifact required_scale_preset does not match evaluation scale preset")
        if not scale_specs_match(strict_contract.required_scale_specs, expected_contract.required_scale_specs):
            raise ValueError("artifact strict runtime scale specs do not match evaluation surface")
        if strict_contract.required_raw_surface_shapes != canonical_raw_surface_shapes(manifest.observation_schema):
            raise ValueError("artifact strict runtime raw surface shapes do not match evaluation surface")
        if strict_contract.derived_contract_version != expected_contract.derived_contract_version:
            raise ValueError("artifact strict runtime derived contract does not match evaluation surface")
        if strict_contract.derived_channel_template_signature != expected_contract.derived_channel_template_signature:
            raise ValueError("artifact strict runtime derived channel signature does not match evaluation surface")
        if strict_contract.derived_channel_templates != expected_contract.derived_channel_templates:
            raise ValueError("artifact strict runtime derived channel templates do not match evaluation surface")
        if compiled_policy.feature_dim != strict_contract.expected_feature_dim:
            raise ValueError(
                "artifact payload feature dimension does not match runtime contract: "
                f"payload_feature_dim={compiled_policy.feature_dim}, expected_feature_dim={strict_contract.expected_feature_dim}"
            )
        if cache_manifest.feature_dim != strict_contract.expected_feature_dim:
            raise ValueError(
                "tensor cache feature dimension does not match runtime contract: "
                f"cache_feature_dim={cache_manifest.feature_dim}, expected_feature_dim={strict_contract.expected_feature_dim}"
            )

    def _window_includes(
        self,
        step: TrajectoryStep,
        *,
        start: datetime | None,
        end: datetime | None,
        exclusive_end: datetime | None,
    ) -> bool:
        if start is not None and step.event_time < start:
            return False
        if end is not None and step.event_time > end:
            return False
        if exclusive_end is not None and step.event_time >= exclusive_end:
            return False
        return True

    def _validate_boundary(self, reward_spec: RewardEventSpec) -> None:
        if self.boundary.fill_assumption_mode != reward_spec.timestamping:
            raise ValueError(
                "evaluation fill_assumption_mode must match reward timestamping for replay parity"
            )
        if self.boundary.fee_handling != "shared_reward_contract":
            raise ValueError("unsupported fee_handling for v1 replay surface")
        if self.boundary.funding_handling != "carry_from_funding_stream":
            raise ValueError("unsupported funding_handling for v1 replay surface")
        if self.boundary.slippage_handling != "fixed_bps":
            raise ValueError("unsupported slippage_handling for v1 replay surface")
        if self.boundary.terminal_semantics != "trajectory_boundary_is_terminal":
            raise ValueError("unsupported terminal_semantics for v1 replay surface")
        if self.boundary.timeout_semantics != "force_terminal_at_data_end":
            raise ValueError("unsupported timeout_semantics for v1 replay surface")


def _softmax_matrix(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis=1, keepdims=True)

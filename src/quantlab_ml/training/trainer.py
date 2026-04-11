"""MomentumBaselineTrainer — V2 surface uyumlu; compat adapter üzerinden çalışır."""
from __future__ import annotations

from statistics import quantiles

from quantlab_ml.common import current_code_commit_hash, hash_payload, utcnow
from quantlab_ml.contracts import (
    ACTION_SPACE_VERSION,
    DYNAMIC_TARGET_ASSET,
    OBSERVATION_SCHEMA_VERSION,
    LineagePointer,
    OpaquePolicyPayload,
    PolicyArtifact,
    RuntimeMetadata,
    TrajectoryBundle,
)
from quantlab_ml.models import MomentumBaselineParameters
from quantlab_ml.training.compat_adapter import V2toV1BundleAdapter
from quantlab_ml.training.config import TrainingConfig


class MomentumBaselineTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def train(self, bundle: TrajectoryBundle, parent_policy_id: str | None = None) -> PolicyArtifact:
        adapter = V2toV1BundleAdapter(bundle)
        steps = adapter.train_steps()
        if not steps:
            raise ValueError("train split is empty")

        momentum_samples: list[float] = []
        for step_view in steps:
            series = [v for v in step_view.mark_price_series() if v is not None]
            if len(series) >= 2 and series[-2] != 0.0:
                momentum_samples.append(abs((series[-1] - series[-2]) / series[-2]))

        abstain_threshold = _quantile(momentum_samples, self.config.abstain_threshold_quantile)
        size_band = bundle.action_space.size_bands[0]
        leverage_band = bundle.action_space.leverage_bands[0]

        parameters = MomentumBaselineParameters(
            abstain_threshold=abstain_threshold,
            preferred_exchange=self.config.preferred_exchange,
            preferred_size_band=size_band.key,
            preferred_leverage_band=leverage_band.key,
        )
        payload_blob = parameters.model_dump_json()
        payload = OpaquePolicyPayload(
            runtime_adapter=self.config.runtime_adapter,
            payload_format="json",
            payload_format_version="json-v1",
            blob=payload_blob,
            digest=hash_payload(parameters),
        )
        lineages = LineagePointer(
            parent_policy_id=parent_policy_id,
            generation=0 if parent_policy_id is None else 1,
            notes=["v2 surface — compat adapter momentum baseline"],
        )
        policy_id = f"policy-{payload.digest[:12]}"
        artifact_id = f"artifact-{payload.digest[:12]}"
        confidence = min(0.99, len(steps) / 100.0)
        training_config_hash = hash_payload(self.config)
        training_snapshot_id = f"{bundle.dataset_spec.dataset_hash}:{bundle.dataset_spec.slice_id}"
        evaluation_surface_id = (
            f"{bundle.dataset_spec.slice_id}:{bundle.split_artifact.split_version}:{bundle.reward_spec.reward_version}"
        )
        target_asset = bundle.dataset_spec.symbols[0] if len(bundle.dataset_spec.symbols) == 1 else DYNAMIC_TARGET_ASSET
        required_context = {}
        if target_asset == DYNAMIC_TARGET_ASSET:
            required_context = {"target_symbol_source": "observation.target_symbol"}

        # Compat metric: available venue-specific reward'lar arasında en iyi net_reward
        average_best_reward = sum(
            max(
                (r.net_reward for r in step_view.action_rewards_all() if r.applicable),
                default=0.0,
            )
            for step_view in steps
        ) / len(steps)

        artifact = PolicyArtifact(
            artifact_id=artifact_id,
            artifact_version="policy_artifact_v1",
            policy_id=policy_id,
            policy_family=self.config.trainer_name,
            training_snapshot_id=training_snapshot_id,
            training_config_hash=training_config_hash,
            code_commit_hash=current_code_commit_hash(),
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
                expected_return_score=average_best_reward,
                risk_score=0.0,
                turnover_score=0.0,
                confidence_or_quality_score=confidence,
                min_capital_requirement=500.0,
                size_bounds=size_band,
                leverage_bounds=leverage_band,
                artifact_compatibility_tags=[
                    f"runtime_adapter:{self.config.runtime_adapter}",
                    f"reward:{bundle.reward_spec.reward_version}",
                    f"split:{bundle.split_artifact.split_version}",
                    f"observation:{OBSERVATION_SCHEMA_VERSION}",
                    f"action_space:{ACTION_SPACE_VERSION}",
                ],
                runtime_adapter=self.config.runtime_adapter,
                required_context=required_context,
                lineage_pointer=lineages,
            ),
            training_run_id=f"trainrun-{payload.digest[:12]}",
            parent_artifact_id=parent_policy_id,
            training_summary={
                "trainer_name": self.config.trainer_name,
                "train_step_count": len(steps),
                "abstain_threshold": abstain_threshold,
                "surface_version": "v2",
            },
        )
        return artifact


def _quantile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    buckets = quantiles(values, n=100, method="inclusive")
    index = min(98, max(0, int(probability * 100) - 1))
    return buckets[index]

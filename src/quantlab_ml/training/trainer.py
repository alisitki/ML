"""MomentumBaselineTrainer — V2 surface uyumlu; compat adapter üzerinden çalışır."""
from __future__ import annotations

import json
from statistics import quantiles

from quantlab_ml.common import hash_payload, utcnow
from quantlab_ml.contracts import ExecutorMetadata, LineagePointer, OpaquePolicyPayload, PolicyArtifact, TrajectoryBundle
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
            blob=payload_blob,
            digest=hash_payload(parameters),
        )
        lineages = LineagePointer(
            parent_policy_id=parent_policy_id,
            generation=0 if parent_policy_id is None else 1,
            notes=["v2 surface — compat adapter momentum baseline"],
        )
        policy_id = f"policy-{payload.digest[:12]}"
        confidence = min(0.99, len(steps) / 100.0)

        # Best reward: venue=None fallback ActionReward'ından (compat kaydı)
        average_best_reward = sum(
            max(
                (r.net_reward for r in step_view.action_rewards_all() if r.applicable),
                default=0.0,
            )
            for step_view in steps
        ) / len(steps)

        artifact = PolicyArtifact(
            policy_id=policy_id,
            created_at=utcnow(),
            observation_schema=bundle.observation_schema,
            action_space=bundle.action_space,
            policy_payload=payload,
            executor_metadata=ExecutorMetadata(
                asset_universe=bundle.dataset_spec.symbols,
                venue_compatibility=bundle.dataset_spec.exchanges,
                instrument_compatibility=bundle.dataset_spec.symbols,
                min_capital_requirement=500.0,
                size_bounds=size_band,
                leverage_bounds=leverage_band,
                liquidity_flags={"requires_positive_open_interest": True},
                applicability_flags={"supports_abstain": True, "offline_only": True},
                expected_return=average_best_reward,
                risk_score=0.0,
                turnover_score=0.0,
                confidence_score=confidence,
                artifact_version="v2",
                lineage_pointer=lineages,
            ),
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

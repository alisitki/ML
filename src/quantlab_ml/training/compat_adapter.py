"""training/compat_adapter.py — V2 → V1 Bundle Adapter

MomentumBaselineTrainer ve EvaluationEngine gibi mevcut bileşenler
V2 TrajectoryBundle'ı bu adapter aracılığıyla tüketir.

Adapter:
  - V2 ObservationContext'ten momentum hesabı için gerekli seriyi çıkarır
  - action_mask flat dict'ini ActionFeasibilitySurface'ten türetir
  - reward_snapshot'tan scalar net_reward'ı alır (venue=None fallback kaydından)

Bu modül dışında averaging veya scalar collapse yapılmaz.
"""
from __future__ import annotations

from quantlab_ml.contracts import TrajectoryBundle, TrajectoryStep
from quantlab_ml.contracts.compat import target_stream_series


class V2toV1BundleAdapter:
    """V2 TrajectoryBundle'ı V1-uyumlu arayüzle wrap eder."""

    def __init__(self, bundle: TrajectoryBundle):
        self.bundle = bundle

    def train_steps(self) -> list[_StepView]:
        return [
            _StepView(step)
            for trajectory in self.bundle.splits.get("train", [])
            for step in trajectory.steps
        ]

    def eval_steps(self) -> list[_StepView]:
        return [
            _StepView(step)
            for trajectory in self.bundle.splits.get("eval", [])
            for step in trajectory.steps
        ]


class _StepView:
    """Tek bir TrajectoryStep için V1-uyumlu erişim proxy'si."""

    def __init__(self, step: TrajectoryStep):
        self._step = step

    @property
    def event_time(self):
        return self._step.event_time

    @property
    def reward_snapshot(self):
        return self._step.reward_snapshot

    def mark_price_series(self) -> list[float | None]:
        """V1 uyumlu exchange-ortalamalı mark_price serisi (compat.target_stream_series)."""
        return target_stream_series(self._step.observation, "mark_price")

    def flat_action_mask(self) -> dict[str, bool]:
        """V1 uyumlu flat action mask (herhangi bir venue/band feasible mi?)."""
        return self._step.action_feasibility.to_flat_mask()

    def best_net_reward(self, action_key: str) -> float:
        """Tüm (venue-specific) applicable reward'lar arasında en yüksek net_reward'ı döner."""
        rewards = [
            r.net_reward
            for r in self._step.reward_snapshot.action_rewards
            if r.action_key == action_key and r.applicable
        ]
        # Hiç applicable reward yoksa infeasible_action_penalty döner.
        # reward_spec'e doğrudan erişim yok; penalty sabit negatif sentinel.
        return max(rewards) if rewards else -0.001  # infeasible_action_penalty varsayılanı

    def action_rewards_all(self):
        return self._step.reward_snapshot.action_rewards

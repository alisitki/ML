from __future__ import annotations

from pydantic import BaseModel

from quantlab_ml.contracts import ActionSpaceSpec, ObservationContext


class RuntimeDecision(BaseModel):
    action_key: str
    venue: str | None = None
    size_band_key: str | None = None
    leverage_band_key: str | None = None
    confidence: float = 0.0


class MomentumBaselineParameters(BaseModel):
    stream: str = "mark_price"
    abstain_threshold: float
    preferred_exchange: str
    preferred_size_band: str
    preferred_leverage_band: str


class MomentumBaselineModel:
    def __init__(self, parameters: MomentumBaselineParameters):
        self.parameters = parameters

    def decide(self, observation: ObservationContext, action_space: ActionSpaceSpec) -> RuntimeDecision:
        from quantlab_ml.contracts.compat import target_stream_series as _compat_series
        series = [value for value in _compat_series(observation, self.parameters.stream) if value is not None]
        if len(series) < 2 or series[-2] == 0.0:
            return RuntimeDecision(action_key="abstain")

        momentum = (series[-1] - series[-2]) / series[-2]
        if abs(momentum) <= self.parameters.abstain_threshold:
            return RuntimeDecision(action_key="abstain", confidence=0.5)

        confidence = min(1.0, abs(momentum) / max(self.parameters.abstain_threshold, 1e-9))
        action_key = "enter_long" if momentum > 0.0 else "enter_short"
        return RuntimeDecision(
            action_key=action_key,
            venue=self.parameters.preferred_exchange,
            size_band_key=self.parameters.preferred_size_band,
            leverage_band_key=self.parameters.preferred_leverage_band,
            confidence=confidence,
        )

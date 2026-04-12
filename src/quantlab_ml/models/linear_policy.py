from __future__ import annotations

import math

from pydantic import BaseModel, model_validator

from quantlab_ml.contracts import ActionSpaceSpec, ObservationContext
from quantlab_ml.models.baseline import RuntimeDecision
from quantlab_ml.models.features import observation_feature_vector


class LinearPolicyParameters(BaseModel):
    action_keys: list[str]
    venue_choices: list[str]
    feature_mean: list[float]
    feature_std: list[float]
    action_weight: list[list[float]]
    action_bias: list[float]
    venue_weight: list[list[float]]
    venue_bias: list[float]
    preferred_size_band: str
    preferred_leverage_band: str

    @model_validator(mode="after")
    def validate_shapes(self) -> "LinearPolicyParameters":
        feature_dim = len(self.feature_mean)
        if feature_dim == 0:
            raise ValueError("feature_mean must not be empty")
        if len(self.feature_std) != feature_dim:
            raise ValueError("feature_std length must match feature_mean length")
        if len(self.action_weight) != len(self.action_keys):
            raise ValueError("action_weight rows must match action_keys")
        if len(self.action_bias) != len(self.action_keys):
            raise ValueError("action_bias length must match action_keys")
        if len(self.venue_weight) != len(self.venue_choices):
            raise ValueError("venue_weight rows must match venue_choices")
        if len(self.venue_bias) != len(self.venue_choices):
            raise ValueError("venue_bias length must match venue_choices")
        for row in self.action_weight:
            if len(row) != feature_dim:
                raise ValueError("action_weight columns must match feature dimension")
        for row in self.venue_weight:
            if len(row) != feature_dim:
                raise ValueError("venue_weight columns must match feature dimension")
        return self


class LinearPolicyModel:
    def __init__(self, parameters: LinearPolicyParameters):
        self.parameters = parameters

    def decide(self, observation: ObservationContext, action_space: ActionSpaceSpec) -> RuntimeDecision:
        features = observation_feature_vector(observation)
        normalized = _normalize(features, self.parameters.feature_mean, self.parameters.feature_std)

        action_logits = [
            _dot(row, normalized) + bias
            for row, bias in zip(self.parameters.action_weight, self.parameters.action_bias, strict=True)
        ]
        action_probs = _softmax(action_logits)
        action_index = _argmax(action_probs)
        action_key = self.parameters.action_keys[action_index]
        action_confidence = action_probs[action_index]

        if action_key == "abstain":
            return RuntimeDecision(action_key="abstain", confidence=action_confidence)

        venue_logits = [
            _dot(row, normalized) + bias
            for row, bias in zip(self.parameters.venue_weight, self.parameters.venue_bias, strict=True)
        ]
        venue_probs = _softmax(venue_logits)
        venue_index = _argmax(venue_probs)
        venue_confidence = venue_probs[venue_index]

        return RuntimeDecision(
            action_key=action_key,
            venue=self.parameters.venue_choices[venue_index],
            size_band_key=self.parameters.preferred_size_band,
            leverage_band_key=self.parameters.preferred_leverage_band,
            confidence=(action_confidence + venue_confidence) / 2.0,
        )


def _normalize(values: list[float], mean: list[float], std: list[float]) -> list[float]:
    normalized: list[float] = []
    for value, offset, scale in zip(values, mean, std, strict=True):
        safe_scale = scale if abs(scale) > 1e-9 else 1.0
        normalized.append((value - offset) / safe_scale)
    return normalized


def _softmax(logits: list[float]) -> list[float]:
    max_logit = max(logits)
    exps = [math.exp(logit - max_logit) for logit in logits]
    total = sum(exps)
    if total <= 0.0:
        return [1.0 / len(logits)] * len(logits)
    return [value / total for value in exps]


def _dot(left: list[float], right: list[float]) -> float:
    return sum(lhs * rhs for lhs, rhs in zip(left, right, strict=True))


def _argmax(values: list[float]) -> int:
    best_index = 0
    best_value = values[0]
    for index, value in enumerate(values[1:], start=1):
        if value > best_value:
            best_index = index
            best_value = value
    return best_index

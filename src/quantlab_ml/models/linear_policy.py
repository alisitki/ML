from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from pydantic import BaseModel, model_validator

from quantlab_ml.contracts import (
    ACTION_SPACE_VERSION_V2_PHASE1A,
    ActionFeasibilitySurface,
    ActionSpaceSpec,
    ObservationContext,
    PolicyState,
)
from quantlab_ml.models.baseline import RuntimeDecision
from quantlab_ml.models.features import (
    PHASE1A_POLICY_STATE_FEATURE_DIM,
    observation_feature_vector,
    phase1a_feature_vector,
    policy_state_feature_array,
)


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


class LinearPolicyV2Parameters(BaseModel):
    joint_action_keys: list[str]
    venue_choices: list[str]
    feature_mean: list[float]
    feature_std: list[float]
    joint_action_weight: list[list[float]]
    joint_action_bias: list[float]
    value_weight: list[float]
    value_bias: float
    preferred_size_band: str
    preferred_leverage_band: str
    joint_action_vocabulary_version: str
    policy_state_feature_version: str

    @model_validator(mode="after")
    def validate_shapes(self) -> "LinearPolicyV2Parameters":
        feature_dim = len(self.feature_mean)
        if feature_dim == 0:
            raise ValueError("feature_mean must not be empty")
        if len(self.feature_std) != feature_dim:
            raise ValueError("feature_std length must match feature_mean length")
        if len(self.joint_action_weight) != len(self.joint_action_keys):
            raise ValueError("joint_action_weight rows must match joint_action_keys")
        if len(self.joint_action_bias) != len(self.joint_action_keys):
            raise ValueError("joint_action_bias length must match joint_action_keys")
        if len(self.value_weight) != feature_dim:
            raise ValueError("value_weight length must match feature dimension")
        for row in self.joint_action_weight:
            if len(row) != feature_dim:
                raise ValueError("joint_action_weight columns must match feature dimension")
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


class LinearPolicyV2Model:
    def __init__(self, parameters: LinearPolicyV2Parameters):
        self.parameters = parameters

    def decide(
        self,
        observation: ObservationContext,
        action_space: ActionSpaceSpec,
        *,
        policy_state: PolicyState,
        action_feasibility: ActionFeasibilitySurface,
    ) -> RuntimeDecision:
        if action_space.action_space_version != ACTION_SPACE_VERSION_V2_PHASE1A:
            raise ValueError("linear-policy-v2 requires action_space_v2_phase1a")

        expected_keys = _phase1a_joint_action_keys(self.parameters.venue_choices)
        if self.parameters.joint_action_keys != expected_keys:
            raise ValueError("linear-policy-v2 joint action vocabulary mismatch")

        features = phase1a_feature_vector(
            observation,
            policy_state,
            venue_choices=self.parameters.venue_choices,
        )
        normalized = _normalize(features, self.parameters.feature_mean, self.parameters.feature_std)
        joint_logits = [
            _dot(row, normalized) + bias
            for row, bias in zip(
                self.parameters.joint_action_weight,
                self.parameters.joint_action_bias,
                strict=True,
            )
        ]
        valid_mask = _phase1a_joint_action_mask(
            venue_choices=self.parameters.venue_choices,
            action_feasibility=action_feasibility,
            policy_state=policy_state,
            preferred_size_band=self.parameters.preferred_size_band,
            preferred_leverage_band=self.parameters.preferred_leverage_band,
        )
        if len(valid_mask) != len(joint_logits):
            raise ValueError("linear-policy-v2 joint mask/logit size mismatch")
        if not any(valid_mask):
            raise ValueError("linear-policy-v2 joint action mask contains no valid action")

        masked_logits = [
            logit if valid else float("-inf")
            for logit, valid in zip(joint_logits, valid_mask, strict=True)
        ]
        probabilities = _softmax(masked_logits)
        joint_index = _argmax(probabilities)
        joint_action_key = self.parameters.joint_action_keys[joint_index]
        action_key, venue = _decode_phase1a_joint_action_key(joint_action_key)
        confidence = probabilities[joint_index]
        if action_key in {"abstain", "hold", "exit"}:
            return RuntimeDecision(action_key=action_key, confidence=confidence)
        return RuntimeDecision(
            action_key=action_key,
            venue=venue,
            size_band_key=self.parameters.preferred_size_band,
            leverage_band_key=self.parameters.preferred_leverage_band,
            confidence=confidence,
        )


@dataclass(slots=True)
class CompiledLinearPolicyV2:
    joint_action_keys: list[str]
    venue_choices: list[str]
    observation_feature_mean: np.ndarray
    observation_feature_std: np.ndarray
    policy_state_feature_mean: np.ndarray
    policy_state_feature_std: np.ndarray
    observation_joint_action_weight: np.ndarray
    policy_state_joint_action_weight: np.ndarray
    joint_action_bias: np.ndarray
    preferred_size_band: str
    preferred_leverage_band: str

    @classmethod
    def from_parameters(cls, parameters: LinearPolicyV2Parameters) -> "CompiledLinearPolicyV2":
        expected_keys = _phase1a_joint_action_keys(parameters.venue_choices)
        if parameters.joint_action_keys != expected_keys:
            raise ValueError("linear-policy-v2 joint action vocabulary mismatch")
        feature_mean = np.asarray(parameters.feature_mean, dtype=np.float64)
        feature_std = np.asarray(parameters.feature_std, dtype=np.float64)
        safe_feature_std = np.where(np.abs(feature_std) > 1e-9, feature_std, 1.0)
        observation_feature_dim = feature_mean.shape[0] - PHASE1A_POLICY_STATE_FEATURE_DIM
        if observation_feature_dim <= 0:
            raise ValueError("linear-policy-v2 feature dimension is smaller than policy-state feature dimension")
        joint_action_weight = np.asarray(parameters.joint_action_weight, dtype=np.float64)
        return cls(
            joint_action_keys=list(parameters.joint_action_keys),
            venue_choices=list(parameters.venue_choices),
            observation_feature_mean=feature_mean[:observation_feature_dim],
            observation_feature_std=safe_feature_std[:observation_feature_dim],
            policy_state_feature_mean=feature_mean[observation_feature_dim:],
            policy_state_feature_std=safe_feature_std[observation_feature_dim:],
            observation_joint_action_weight=joint_action_weight[:, :observation_feature_dim],
            policy_state_joint_action_weight=joint_action_weight[:, observation_feature_dim:],
            joint_action_bias=np.asarray(parameters.joint_action_bias, dtype=np.float64),
            preferred_size_band=parameters.preferred_size_band,
            preferred_leverage_band=parameters.preferred_leverage_band,
        )

    @classmethod
    def from_artifact(cls, blob: str) -> "CompiledLinearPolicyV2":
        return cls.from_parameters(LinearPolicyV2Parameters.model_validate_json(blob))

    @property
    def observation_feature_dim(self) -> int:
        return int(self.observation_feature_mean.shape[0])

    @property
    def feature_dim(self) -> int:
        return int(self.observation_feature_dim + self.policy_state_feature_mean.shape[0])

    def observation_logits_batch(self, raw_observation_features: np.ndarray) -> np.ndarray:
        normalized = raw_observation_features.astype(np.float64, copy=False)
        normalized -= self.observation_feature_mean
        normalized /= self.observation_feature_std
        return normalized @ self.observation_joint_action_weight.transpose(1, 0) + self.joint_action_bias

    def decide_from_observation_logits(
        self,
        observation_logits: np.ndarray,
        *,
        policy_state: PolicyState,
        action_feasibility: ActionFeasibilitySurface,
    ) -> RuntimeDecision:
        policy_state_features = policy_state_feature_array(
            policy_state,
            venue_choices=self.venue_choices,
            dtype=np.float64,
        )
        normalized_state = policy_state_features.copy()
        normalized_state -= self.policy_state_feature_mean
        normalized_state /= self.policy_state_feature_std
        joint_logits = observation_logits + (self.policy_state_joint_action_weight @ normalized_state)
        valid_mask = _phase1a_joint_action_mask(
            venue_choices=self.venue_choices,
            action_feasibility=action_feasibility,
            policy_state=policy_state,
            preferred_size_band=self.preferred_size_band,
            preferred_leverage_band=self.preferred_leverage_band,
        )
        if len(valid_mask) != int(joint_logits.shape[0]):
            raise ValueError("linear-policy-v2 joint mask/logit size mismatch")
        if not any(valid_mask):
            raise ValueError("linear-policy-v2 joint action mask contains no valid action")
        masked_logits = [
            float(logit) if valid else float("-inf")
            for logit, valid in zip(joint_logits.tolist(), valid_mask, strict=True)
        ]
        probabilities = _softmax(masked_logits)
        joint_index = _argmax(probabilities)
        joint_action_key = self.joint_action_keys[joint_index]
        action_key, venue = _decode_phase1a_joint_action_key(joint_action_key)
        confidence = probabilities[joint_index]
        if action_key in {"abstain", "hold", "exit"}:
            return RuntimeDecision(action_key=action_key, confidence=confidence)
        return RuntimeDecision(
            action_key=action_key,
            venue=venue,
            size_band_key=self.preferred_size_band,
            leverage_band_key=self.preferred_leverage_band,
            confidence=confidence,
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


def _phase1a_joint_action_keys(venue_choices: list[str]) -> list[str]:
    keys = ["abstain", "hold", "exit"]
    keys.extend(f"enter_long@{venue}" for venue in venue_choices)
    keys.extend(f"enter_short@{venue}" for venue in venue_choices)
    return keys


def _decode_phase1a_joint_action_key(joint_action_key: str) -> tuple[str, str | None]:
    if "@" not in joint_action_key:
        return joint_action_key, None
    action_key, venue = joint_action_key.split("@", 1)
    return action_key, venue


def _phase1a_joint_action_mask(
    *,
    venue_choices: list[str],
    action_feasibility: ActionFeasibilitySurface,
    policy_state: PolicyState,
    preferred_size_band: str,
    preferred_leverage_band: str,
) -> list[bool]:
    flat = policy_state.previous_position_side == "flat"
    mask = [flat, not flat, not flat]
    for venue in venue_choices:
        mask.append(
            flat
            and action_feasibility.is_feasible(
                "enter_long",
                venue,
                preferred_size_band,
                preferred_leverage_band,
            )
        )
    for venue in venue_choices:
        mask.append(
            flat
            and action_feasibility.is_feasible(
                "enter_short",
                venue,
                preferred_size_band,
                preferred_leverage_band,
            )
        )
    return mask

from __future__ import annotations

from pydantic import Field

from quantlab_ml.contracts.common import QuantBaseModel


class CandidateSearchConfig(QuantBaseModel):
    seeds: list[int] = Field(default_factory=list)
    learning_rates: list[float] = Field(default_factory=list)
    l2_weights: list[float] = Field(default_factory=list)


class TrainingConfig(QuantBaseModel):
    trainer_name: str
    runtime_adapter: str
    epochs: int = 8
    learning_rate: float = 0.1
    l2_weight: float = 1e-4
    seed: int = 7
    preferred_size_band: str
    preferred_leverage_band: str
    bootstrap_horizon_steps: int = 1
    aux_value_loss_weight: float = 0.0
    policy_state_feature_version: str | None = None
    joint_action_vocabulary_version: str | None = None
    phase1a_compute_dtype: str = "float32"
    candidate_search: CandidateSearchConfig | None = None

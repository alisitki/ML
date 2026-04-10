from __future__ import annotations

from quantlab_ml.contracts.common import QuantBaseModel


class TrainingConfig(QuantBaseModel):
    trainer_name: str
    runtime_adapter: str
    abstain_threshold_quantile: float = 0.5
    preferred_exchange: str

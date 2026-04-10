from __future__ import annotations

from abc import ABC, abstractmethod

from quantlab_ml.contracts import ActionSpaceSpec, ObservationContext
from quantlab_ml.models.baseline import RuntimeDecision


class PolicyModel(ABC):
    @abstractmethod
    def decide(self, observation: ObservationContext, action_space: ActionSpaceSpec) -> RuntimeDecision:
        raise NotImplementedError

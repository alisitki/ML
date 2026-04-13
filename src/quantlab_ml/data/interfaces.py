from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

from quantlab_ml.contracts import DatasetSpec, NormalizedMarketEvent


class MarketDataSource(ABC):
    @abstractmethod
    def load_events(self, dataset_spec: DatasetSpec) -> Iterable[NormalizedMarketEvent]:
        raise NotImplementedError

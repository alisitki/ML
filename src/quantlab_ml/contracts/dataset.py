from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import AliasChoices, Field, model_validator

from quantlab_ml.contracts.common import QuantBaseModel, TimeRange

# Zorunlu ham field setleri — stream ailesi başına.
# Builder bu katalogu kullanır; eksik field NaN ile doldurulur.
REQUIRED_FIELDS_BY_STREAM: dict[str, list[str]] = {
    "trade": ["price", "qty", "side_or_signed_flow_proxy", "event_delta", "count_or_burst"],
    "bbo": ["bid_price", "ask_price", "bid_size", "ask_size", "spread", "mid", "imbalance_inputs"],
    "mark_price": ["mark_price", "event_delta", "index_price_if_available"],
    "funding": ["funding_rate", "next_funding_time", "funding_update_age"],
    "open_interest": ["open_interest", "oi_delta", "oi_update_age"],
}


class StreamFieldCatalog(QuantBaseModel):
    """Bir stream ailesi için zorunlu ham field isimleri."""

    stream: str
    fields: list[str]

    @model_validator(mode="after")
    def validate_fields(self) -> "StreamFieldCatalog":
        if not self.fields:
            raise ValueError(f"stream {self.stream} field catalog must not be empty")
        return self


class WalkForwardSpec(QuantBaseModel):
    """Deterministic walk-forward fold generation config."""

    train_window_steps: int
    validation_window_steps: int
    step_size_steps: int | None = None

    @model_validator(mode="after")
    def validate_steps(self) -> "WalkForwardSpec":
        if self.train_window_steps <= 0:
            raise ValueError("walkforward train_window_steps must be positive")
        if self.validation_window_steps <= 0:
            raise ValueError("walkforward validation_window_steps must be positive")
        if self.step_size_steps is None:
            object.__setattr__(self, "step_size_steps", self.validation_window_steps)
        if self.step_size_steps <= 0:
            raise ValueError("walkforward step_size_steps must be positive")
        return self


class DatasetSpec(QuantBaseModel):
    dataset_hash: str
    slice_id: str
    exchanges: list[str]
    symbols: list[str]
    stream_universe: list[str]
    available_streams_by_exchange: dict[str, list[str]]
    train_range: TimeRange
    validation_range: TimeRange = Field(
        validation_alias=AliasChoices("validation_range", "eval_range"),
    )
    final_untouched_test_range: TimeRange
    walkforward: WalkForwardSpec
    sampling_interval_seconds: int = 60

    # V2: Per-stream zorunlu field katalogları.
    # Boş bırakılırsa REQUIRED_FIELDS_BY_STREAM sabitinden doldurulur.
    field_catalogs: list[StreamFieldCatalog] = Field(default_factory=list)

    # V2: (exchange, stream) düzeyinde yapısal erişilebilirlik kısıtları.
    # available_streams_by_exchange stream'i listeden dışarıda bırakmakla eşdeğer;
    # bu ek katman daha ayrıntılı future override için yer açar.
    # exchange → stream → bool; belirtilmemiş koordinatlar available_streams_by_exchange'den türetilir.
    availability_by_contract: dict[str, dict[str, bool]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_axes(self) -> "DatasetSpec":
        if not self.exchanges:
            raise ValueError("dataset spec requires at least one exchange")
        if not self.symbols:
            raise ValueError("dataset spec requires at least one symbol")
        if not self.stream_universe:
            raise ValueError("dataset spec requires at least one stream family")
        if set(self.available_streams_by_exchange) != set(self.exchanges):
            raise ValueError("available_streams_by_exchange must define every exchange exactly once")
        for exchange, streams in self.available_streams_by_exchange.items():
            if not streams:
                raise ValueError(f"exchange {exchange} must expose at least one stream family")
            unknown = set(streams) - set(self.stream_universe)
            if unknown:
                raise ValueError(f"exchange {exchange} references unknown streams: {sorted(unknown)}")
        if not self.train_range.end < self.validation_range.start:
            raise ValueError("train_range must end before validation_range starts")
        if not self.validation_range.end < self.final_untouched_test_range.start:
            raise ValueError("validation_range must end before final_untouched_test_range starts")
        if self.sampling_interval_seconds <= 0:
            raise ValueError("sampling interval must be positive")
        development_steps = _inclusive_step_count(
            self.train_range.start,
            self.validation_range.end,
            self.sampling_interval_seconds,
        )
        if self.walkforward.train_window_steps + self.walkforward.validation_window_steps > development_steps:
            raise ValueError("walkforward windows exceed the train+validation development region")
        # availability_by_contract yalnız bilinen exchange / stream'lere başvurabilir
        for exchange, stream_map in self.availability_by_contract.items():
            if exchange not in self.exchanges:
                raise ValueError(f"availability_by_contract references unknown exchange: {exchange}")
            for stream in stream_map:
                if stream not in self.stream_universe:
                    raise ValueError(
                        f"availability_by_contract[{exchange}] references unknown stream: {stream}"
                    )
        return self

    @model_validator(mode="after")
    def fill_default_field_catalogs(self) -> "DatasetSpec":
        """field_catalogs boşsa bilinen stream'ler için sabitten doldur."""
        if not self.field_catalogs:
            object.__setattr__(
                self,
                "field_catalogs",
                [
                    StreamFieldCatalog(stream=stream, fields=list(fields))
                    for stream, fields in REQUIRED_FIELDS_BY_STREAM.items()
                    if stream in self.stream_universe
                ],
            )
        return self

    def stream_available(self, exchange: str, stream: str) -> bool:
        """(exchange, stream) koordinatının yapısal olarak erişilebilir olup olmadığını döner."""
        # availability_by_contract varsa önceliği o alır
        if exchange in self.availability_by_contract:
            contract_map = self.availability_by_contract[exchange]
            if stream in contract_map:
                return contract_map[stream]
        # Yoksa available_streams_by_exchange'e bak
        return stream in self.available_streams_by_exchange.get(exchange, [])

    def fields_for_stream(self, stream: str) -> list[str]:
        """Bir stream ailesi için zorunlu field listesini döner."""
        for catalog in self.field_catalogs:
            if catalog.stream == stream:
                return catalog.fields
        # Katalog yoksa sabitten bak
        return list(REQUIRED_FIELDS_BY_STREAM.get(stream, []))

    @property
    def development_range(self) -> TimeRange:
        return TimeRange(start=self.train_range.start, end=self.validation_range.end)


class NormalizedMarketEvent(QuantBaseModel):
    event_time: datetime
    exchange: str
    symbol: str
    stream_type: str
    # V2'de `fields` dict birincil veri kaynağıdır; builder bu alanı kullanır.
    # `value` alanı yalnızca legacy uyumluluk için tutulmaktadır ve
    # yeni observation builder tarafından giriş olarak kullanılmaz.
    value: float = 0.0  # legacy-compat; yeni kod fields dict'ini kullanır
    fields: dict[str, Any] = Field(default_factory=dict)
    ingest_metadata: dict[str, Any] = Field(default_factory=dict)


def _inclusive_step_count(start: datetime, end: datetime, sampling_interval_seconds: int) -> int:
    total_seconds = (end - start).total_seconds()
    return int(total_seconds // sampling_interval_seconds) + 1

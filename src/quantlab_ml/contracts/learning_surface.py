from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import Field, model_validator

from quantlab_ml.contracts.common import InvalidActionMaskSemantics, NumericBand, QuantBaseModel
from quantlab_ml.contracts.dataset import DatasetSpec, WalkForwardSpec
from quantlab_ml.contracts.rewards import RewardContext, RewardEventSpec, RewardSnapshot, RewardTimeline

OBSERVATION_SCHEMA_VERSION = "observation_schema_v1"
DERIVED_SURFACE_CONTRACT_VERSION = "derived_surface_v1"


# ---------------------------------------------------------------------------
# Scale & Field Schema
# ---------------------------------------------------------------------------


class ScaleSpec(QuantBaseModel):
    """Tek bir zaman ölçeği tanımı."""

    label: str  # örn. "1m", "5m", "15m", "60m"
    resolution_seconds: int  # 60, 300, 900, 3600
    num_buckets: int  # bu ölçekte kaç time bucket

    @model_validator(mode="after")
    def validate_positive(self) -> "ScaleSpec":
        if self.resolution_seconds <= 0:
            raise ValueError("resolution_seconds must be positive")
        if self.num_buckets <= 0:
            raise ValueError("num_buckets must be positive")
        return self

    @property
    def bucket_labels(self) -> list[str]:
        return [f"{self.label}-t-{i}" for i in reversed(range(self.num_buckets))]


# ---------------------------------------------------------------------------
# Observation Tensor Blokları
# ---------------------------------------------------------------------------


class RawScaleTensor(QuantBaseModel):
    """Tek bir zaman ölçeği için ham field-level tensor bloğu.

    Boyut: time × symbol × exchange × stream × field
    Tüm listeler aynı uzunlukta (flat); sıralama time-major.

    Mask öncelik sırası (birbirini dışlar):
      padding → unavailable_by_contract → missing → stale → geçerli değer
    """

    scale_label: str
    # shape bilgisi (time, symbol, exchange, stream, field)
    shape: list[int]  # 5 elemanlı liste
    # Ham değerler — eksik/unavailable/padding koordinatlarda 0.0
    values: list[float]
    # Kaç saniye önce geldiği — her koordinat için event yaşı (saniye)
    age: list[float]
    # True → bu koordinat padded (yetersiz tarihçe)
    padding: list[bool]
    # True → bu koordinat yapısal olarak erişilemez (contract availability=False)
    unavailable_by_contract: list[bool]
    # True → koordinat erişilebilir ama bu adımda veri gelMEdi (runtime missing)
    missing: list[bool]
    # True → koordinat erişilebilir, veri var ama freshness bound'u geçmiş
    stale: list[bool]

    @model_validator(mode="after")
    def validate_flat_size(self) -> "RawScaleTensor":
        if len(self.shape) != 5:
            raise ValueError("shape must have exactly 5 dimensions: (time, symbol, exchange, stream, field)")
        expected = 1
        for dim in self.shape:
            expected *= dim
        for field_name in ("values", "age", "padding", "unavailable_by_contract", "missing", "stale"):
            actual = len(getattr(self, field_name))
            if actual != expected:
                raise ValueError(
                    f"{field_name} length {actual} != flat size {expected} "
                    f"(shape={self.shape})"
                )
        return self

    @property
    def flat_size(self) -> int:
        result = 1
        for dim in self.shape:
            result *= dim
        return result


class DerivedChannel(QuantBaseModel):
    """Tek bir derived sinyal kaydı."""

    key: str  # örn. "bid_ask_spread", "venue_pair_price_spread_binance_bybit"
    description: str
    values: list[float]  # scale × time × symbol shape'inde flat; producer belirler
    shape: list[int]

    @model_validator(mode="after")
    def validate_shape(self) -> "DerivedChannel":
        expected = 1
        for dim in self.shape:
            expected *= dim
        if expected != len(self.values):
            raise ValueError(
                f"derived channel '{self.key}' values length {len(self.values)} "
                f"does not match declared shape {self.shape}"
            )
        return self


class DerivedSurface(QuantBaseModel):
    """Raw'ın yanına eklenen target-centric derived sinyaller.

    V1 kapsamı: spread, imbalance, signed flow proxy, OI delta,
    funding delta, freshness, venue-pair price spread (O(n²) pairwise),
    relative move (target vs diğer symboller).
    Derived sinyaller raw tensorların yerini almaz.
    """

    contract_version: str = DERIVED_SURFACE_CONTRACT_VERSION
    channels: list[DerivedChannel] = Field(default_factory=list)

    def get(self, key: str) -> DerivedChannel | None:
        for ch in self.channels:
            if ch.key == key:
                return ch
        return None


# ---------------------------------------------------------------------------
# Observation Schema
# ---------------------------------------------------------------------------


class ObservationSchema(QuantBaseModel):
    schema_version: str = OBSERVATION_SCHEMA_VERSION
    # V2: Çok ölçekli zaman ekseni — her ölçek bağımsız bucket + mask mantığına sahip
    scale_axis: list[ScaleSpec]
    asset_axis: list[str]  # symbol listesi
    exchange_axis: list[str]  # exchange listesi
    stream_axis: list[str]  # stream ailesi listesi
    # stream → field isimleri listesi
    field_axis: dict[str, list[str]]
    # exchange → stream → bool; yapısal erişilebilirlik
    availability_by_contract: dict[str, dict[str, bool]] = Field(default_factory=dict)

    missing_data_semantics: str = "true means runtime missing — coordinate available by contract but no event arrived"
    padding_semantics: str = "true means left padding due to insufficient history only"
    stale_data_semantics: str = "true means latest event is older than freshness bound"
    unavailable_semantics: str = "true means coordinate structurally unavailable by contract"
    target_asset_pointer_semantics: str = "target_asset_index points into asset_axis"

    @model_validator(mode="after")
    def validate_schema(self) -> "ObservationSchema":
        if not self.scale_axis:
            raise ValueError("scale_axis must not be empty")
        if not self.asset_axis:
            raise ValueError("asset_axis must not be empty")
        if not self.exchange_axis:
            raise ValueError("exchange_axis must not be empty")
        if not self.stream_axis:
            raise ValueError("stream_axis must not be empty")
        for stream in self.stream_axis:
            if stream not in self.field_axis:
                raise ValueError(f"field_axis missing entry for stream: {stream}")
        return self

    def shape_for_scale(self, scale_label: str) -> tuple[int, int, int, int, int]:
        """(time, symbol, exchange, stream, field_total) — field_total = sum(fields per stream)."""
        spec = next((s for s in self.scale_axis if s.label == scale_label), None)
        if spec is None:
            raise KeyError(f"unknown scale label: {scale_label}")
        field_total = sum(len(fields) for fields in self.field_axis.values())
        return (
            spec.num_buckets,
            len(self.asset_axis),
            len(self.exchange_axis),
            len(self.stream_axis),
            field_total,
        )

    def stream_available(self, exchange: str, stream: str) -> bool:
        if exchange in self.availability_by_contract:
            contract_map = self.availability_by_contract[exchange]
            if stream in contract_map:
                return contract_map[stream]
        return True  # varsayılan: mevcut


# ---------------------------------------------------------------------------
# Policy State
# ---------------------------------------------------------------------------


class PolicyState(QuantBaseModel):
    """Geçmiş uygulanmış kararlar ve karar-anı-bilinen state'ten türetilir.

    Future bilgi içermez; önceki step'ten taşınan inventory snapshot'ıdır.
    """

    previous_position_side: Literal["flat", "long", "short"] = "flat"
    previous_venue: str | None = None
    hold_age_steps: int = 0  # mevcut pozisyonun kaç step önce açıldığı
    turnover_accumulator: float = 0.0  # birikimli round-trip hacmi (normalize)


# ---------------------------------------------------------------------------
# Action Feasibility
# ---------------------------------------------------------------------------


class FeasibilityCell(QuantBaseModel):
    """Tek bir (action, venue, size_band, leverage_band) hücresi."""

    feasible: bool
    reason: str = ""  # feasible=False ise açıklama etiketi


class ActionFeasibilitySurface(QuantBaseModel):
    """Action × venue × size_band × leverage_band feasibility matrisi.

    Yalnızca decision-time'da bilinen bilgiden üretilir; future fiyat kullanılmaz.
    """

    # Nested dict: action_key → venue → size_band_key → leverage_band_key → FeasibilityCell
    surface: dict[str, dict[str, dict[str, dict[str, FeasibilityCell]]]] = Field(default_factory=dict)

    def is_feasible(self, action_key: str, venue: str, size_band: str, leverage_band: str) -> bool:
        try:
            return self.surface[action_key][venue][size_band][leverage_band].feasible
        except KeyError:
            return False

    def abstain_feasible(self) -> bool:
        """abstain her zaman uygulanabilir olmalı."""
        return "abstain" in self.surface

    def to_flat_mask(self) -> dict[str, bool]:
        """Geriye dönük compat: action_key → bool (herhangi bir venue/band feasible mi?)."""
        result: dict[str, bool] = {}
        for action_key, venues in self.surface.items():
            result[action_key] = any(
                cell.feasible
                for size_map in venues.values()
                for lev_map in size_map.values()
                for cell in lev_map.values()
            )
        return result


# ---------------------------------------------------------------------------
# Observation Context
# ---------------------------------------------------------------------------


class ObservationContext(QuantBaseModel):
    as_of: datetime
    observation_schema: ObservationSchema
    target_symbol: str
    target_asset_index: int

    # V2: Her scale için ayrı ham tensor bloğu
    raw_surface: dict[str, RawScaleTensor]  # scale_label → tensor

    # V2 opsiyonel derived sinyal katmanı
    derived_surface: DerivedSurface | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_context(self) -> "ObservationContext":
        schema = self.observation_schema
        # target_asset_index sınırı
        if self.target_asset_index >= len(schema.asset_axis):
            raise ValueError("target_asset_index is out of bounds")
        if schema.asset_axis[self.target_asset_index] != self.target_symbol:
            raise ValueError("target_symbol and target_asset_index do not match")
        # Her scale için tensor bloğu mevcut olmalı
        for scale_spec in schema.scale_axis:
            if scale_spec.label not in self.raw_surface:
                raise ValueError(f"raw_surface missing tensor for scale: {scale_spec.label}")
            expected_shape = list(schema.shape_for_scale(scale_spec.label))
            actual_shape = self.raw_surface[scale_spec.label].shape
            if actual_shape != expected_shape:
                raise ValueError(
                    f"raw_surface[{scale_spec.label}] shape {actual_shape} "
                    f"does not match observation schema shape {expected_shape}"
                )
        return self


# ---------------------------------------------------------------------------
# Action Space
# ---------------------------------------------------------------------------


class ActionChoice(QuantBaseModel):
    key: str
    label: str
    category: Literal["abstain", "directional", "risk_management"]
    direction: Literal["flat", "long", "short"]
    requires_venue: bool
    requires_size_band: bool
    requires_leverage_band: bool


class ActionSpaceSpec(QuantBaseModel):
    actions: list[ActionChoice]
    venue_choices: list[str]
    size_bands: list[NumericBand]
    leverage_bands: list[NumericBand]
    invalid_action_mask_semantics: InvalidActionMaskSemantics

    @model_validator(mode="after")
    def validate_abstain(self) -> "ActionSpaceSpec":
        abstain_actions = [action for action in self.actions if action.category == "abstain"]
        if len(abstain_actions) != 1:
            raise ValueError("action space must contain exactly one abstain action")
        if abstain_actions[0].key != "abstain":
            raise ValueError("abstain action key must be 'abstain'")
        return self

    @property
    def action_keys(self) -> list[str]:
        return [action.key for action in self.actions]


# ---------------------------------------------------------------------------
# Trajectory Spec
# ---------------------------------------------------------------------------


class TrajectorySpec(QuantBaseModel):
    # V2: lookback_steps kaldırıldı; yerini scale_preset listesi aldı
    scale_preset: list[ScaleSpec]
    max_episode_steps: int = 128
    stale_after_seconds: int
    terminal_semantics: str
    timeout_semantics: str

    @model_validator(mode="after")
    def validate_preset(self) -> "TrajectorySpec":
        if not self.scale_preset:
            raise ValueError("scale_preset must not be empty")
        labels = [s.label for s in self.scale_preset]
        if len(labels) != len(set(labels)):
            raise ValueError("scale_preset labels must be unique")
        return self


# ---------------------------------------------------------------------------
# Trajectory Step
# ---------------------------------------------------------------------------


class TrajectoryStep(QuantBaseModel):
    event_time: datetime
    decision_timestamp: datetime  # V2: karar anı (= event_time ile eşit veya sonrası)
    observation: ObservationContext
    target_symbol: str  # step düzeyinde de taşınır
    # V2: action_mask → action_feasibility
    action_feasibility: ActionFeasibilitySurface
    reward_snapshot: RewardSnapshot
    # V2 yeni alanlar
    reward_context: RewardContext
    reward_timeline: RewardTimeline
    policy_state: PolicyState | None = None  # ilk step'te None olabilir

    @model_validator(mode="after")
    def validate_abstain(self) -> "TrajectoryStep":
        if not self.action_feasibility.abstain_feasible():
            raise ValueError("trajectory step requires explicit available abstain action")
        return self


# ---------------------------------------------------------------------------
# Trajectory Record & Bundle
# ---------------------------------------------------------------------------


class TrajectoryRecord(QuantBaseModel):
    trajectory_id: str
    split: Literal["train", "validation", "final_untouched_test"]
    target_symbol: str
    start_time: datetime
    end_time: datetime
    steps: list[TrajectoryStep]
    terminal: bool = True
    terminal_reason: str

    @model_validator(mode="after")
    def validate_steps(self) -> "TrajectoryRecord":
        if not self.steps:
            raise ValueError("trajectory must contain at least one step")
        return self


class SplitWindow(QuantBaseModel):
    start: datetime
    end: datetime

    @model_validator(mode="after")
    def validate_bounds(self) -> "SplitWindow":
        if self.end < self.start:
            raise ValueError("split window end must be greater than or equal to start")
        return self


class WalkForwardFold(QuantBaseModel):
    fold_id: str
    train_window: SplitWindow
    validation_window: SplitWindow
    purge_width_steps: int
    embargo_width_steps: int
    horizon_steps: int

    @model_validator(mode="after")
    def validate_fold(self) -> "WalkForwardFold":
        if self.train_window.end >= self.validation_window.start:
            raise ValueError("walk-forward train window must end before validation window starts")
        if self.purge_width_steps < 0 or self.embargo_width_steps < 0:
            raise ValueError("purge/embargo widths must be non-negative")
        if self.horizon_steps <= 0:
            raise ValueError("walk-forward fold horizon_steps must be positive")
        return self


class SplitArtifact(QuantBaseModel):
    split_version: str
    purge_width_steps: int
    embargo_width_steps: int
    fold_generation_config: WalkForwardSpec
    development_window: SplitWindow
    train_window: SplitWindow
    validation_window: SplitWindow
    final_untouched_test_window: SplitWindow
    folds: list[WalkForwardFold]

    @model_validator(mode="after")
    def validate_folds(self) -> "SplitArtifact":
        if self.split_version != "split_v1_walkforward":
            raise ValueError("split artifact must use split_v1_walkforward")
        if not self.folds:
            raise ValueError("split artifact must contain at least one walk-forward fold")
        return self


class TrajectoryBundle(QuantBaseModel):
    dataset_spec: DatasetSpec
    trajectory_spec: TrajectorySpec
    action_space: ActionSpaceSpec
    reward_spec: RewardEventSpec
    observation_schema: ObservationSchema
    split_artifact: SplitArtifact
    splits: dict[str, list[TrajectoryRecord]]

    @model_validator(mode="after")
    def validate_split_keys(self) -> "TrajectoryBundle":
        required_keys = {"train", "validation", "final_untouched_test"}
        actual_keys = set(self.splits)
        if actual_keys != required_keys:
            raise ValueError(
                f"trajectory bundle splits must match canonical keys {sorted(required_keys)}; got {sorted(actual_keys)}"
            )
        return self

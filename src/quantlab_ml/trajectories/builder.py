from __future__ import annotations

import logging
import math
from collections import defaultdict
from datetime import datetime, timedelta
from itertools import combinations
from typing import Iterable, Literal, cast

from quantlab_ml.contracts import (
    ActionFeasibilitySurface,
    ActionSpaceSpec,
    DatasetSpec,
    DerivedChannel,
    DerivedSurface,
    FeasibilityCell,
    NormalizedMarketEvent,
    ObservationContext,
    ObservationSchema,
    PolicyState,
    RawScaleTensor,
    RewardContext,
    RewardEventSpec,
    RewardTimeline,
    ScaleSpec,
    TimeRange,
    SplitArtifact,
    SplitWindow,
    TrajectoryBundle,
    TrajectoryRecord,
    TrajectorySpec,
    TrajectoryStep,
    VenueExecutionRef,
    WalkForwardFold,
)
from quantlab_ml.rewards import RewardEngine

# Indeks tipi: (symbol, exchange, stream) → zaman sıralı event listesi
_Index = dict[tuple[str, str, str], list[NormalizedMarketEvent]]

logger = logging.getLogger(__name__)


class TrajectoryBuilder:
    def __init__(
        self,
        dataset_spec: DatasetSpec,
        trajectory_spec: TrajectorySpec,
        action_space: ActionSpaceSpec,
        reward_spec: RewardEventSpec,
    ):
        self.dataset_spec = dataset_spec
        self.trajectory_spec = trajectory_spec
        self.action_space = action_space
        self.reward_spec = reward_spec
        self.reward_engine = RewardEngine(reward_spec, action_space)

        # ObservationSchema — V2
        field_axis: dict[str, list[str]] = {
            stream: dataset_spec.fields_for_stream(stream)
            for stream in dataset_spec.stream_universe
        }
        availability: dict[str, dict[str, bool]] = {}
        for exchange in dataset_spec.exchanges:
            availability[exchange] = {
                stream: dataset_spec.stream_available(exchange, stream)
                for stream in dataset_spec.stream_universe
            }

        self.observation_schema = ObservationSchema(
            scale_axis=trajectory_spec.scale_preset,
            asset_axis=dataset_spec.symbols,
            exchange_axis=dataset_spec.exchanges,
            stream_axis=dataset_spec.stream_universe,
            field_axis=field_axis,
            availability_by_contract=availability,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, events: list[NormalizedMarketEvent]) -> TrajectoryBundle:
        logger.info(
            "trajectory_build_started slice_id=%s event_count=%d symbol_count=%d exchange_count=%d",
            self.dataset_spec.slice_id,
            len(events),
            len(self.dataset_spec.symbols),
            len(self.dataset_spec.exchanges),
        )
        indexed = self._index_events(events)
        self._validate_split_ranges()
        history_start = min(
            (event.event_time for event in events),
            default=self.dataset_spec.train_range.start,
        )
        split_artifact = self._build_split_artifact()
        splits = {
            "train": self._build_split("train", self.dataset_spec.train_range, indexed, history_start),
            "validation": self._build_split("validation", self.dataset_spec.validation_range, indexed, history_start),
            "final_untouched_test": self._build_split(
                "final_untouched_test",
                self.dataset_spec.final_untouched_test_range,
                indexed,
                history_start,
            ),
        }
        bundle = TrajectoryBundle(
            dataset_spec=self.dataset_spec,
            trajectory_spec=self.trajectory_spec,
            action_space=self.action_space,
            reward_spec=self.reward_spec,
            observation_schema=self.observation_schema,
            split_artifact=split_artifact,
            splits=splits,
        )
        logger.info(
            "trajectory_build_completed slice_id=%s train_records=%d validation_records=%d "
            "final_test_records=%d train_steps=%d validation_steps=%d final_test_steps=%d fold_count=%d",
            self.dataset_spec.slice_id,
            len(splits["train"]),
            len(splits["validation"]),
            len(splits["final_untouched_test"]),
            sum(len(record.steps) for record in splits["train"]),
            sum(len(record.steps) for record in splits["validation"]),
            sum(len(record.steps) for record in splits["final_untouched_test"]),
            len(split_artifact.folds),
        )
        return bundle

    # ------------------------------------------------------------------
    # Split & Step Assembly
    # ------------------------------------------------------------------

    def _build_split(
        self,
        split_name: str,
        split_range: TimeRange,
        indexed: _Index,
        history_start: datetime,
    ) -> list[TrajectoryRecord]:
        timestamps = self._timestamps(split_range)
        if len(timestamps) <= self.reward_spec.horizon_steps:
            return []

        trajectories: list[TrajectoryRecord] = []
        for symbol in self.dataset_spec.symbols:
            symbol_steps: list[TrajectoryStep] = []
            usable_count = len(timestamps) - self.reward_spec.horizon_steps
            prev_step: TrajectoryStep | None = None

            for step_index in range(usable_count):
                event_time = timestamps[step_index]

                observation = self._build_observation(indexed, symbol, event_time, history_start)
                action_feasibility = self._build_action_feasibility(indexed, symbol, event_time)
                reward_context = self._build_reward_context(indexed, symbol, event_time)
                reward_timeline = self._build_reward_timeline(indexed, symbol, timestamps, step_index)
                policy_state = self._build_policy_state(prev_step)
                reward_snapshot = self.reward_engine.build_snapshot(
                    event_time=event_time,
                    reward_context=reward_context,
                    reward_timeline=reward_timeline,
                    action_feasibility=action_feasibility,
                )

                step = TrajectoryStep(
                    event_time=event_time,
                    decision_timestamp=event_time,
                    observation=observation,
                    target_symbol=symbol,
                    action_feasibility=action_feasibility,
                    reward_snapshot=reward_snapshot,
                    reward_context=reward_context,
                    reward_timeline=reward_timeline,
                    policy_state=policy_state,
                )
                symbol_steps.append(step)
                prev_step = step

            for chunk_index, chunk in enumerate(_chunked(symbol_steps, self.trajectory_spec.max_episode_steps)):
                trajectories.append(
                    TrajectoryRecord(
                        trajectory_id=f"{split_name}-{symbol.lower()}-{chunk_index}",
                        split=cast(
                            Literal["train", "validation", "final_untouched_test"],
                            split_name,
                        ),
                        target_symbol=symbol,
                        start_time=chunk[0].event_time,
                        end_time=chunk[-1].event_time,
                        steps=chunk,
                        terminal=True,
                        terminal_reason=self.trajectory_spec.terminal_semantics,
                    )
                )
        return trajectories

    def _build_split_artifact(self) -> SplitArtifact:
        purge_width_steps = self.reward_spec.horizon_steps
        embargo_width_steps = self.reward_spec.horizon_steps
        folds = self._generate_walkforward_folds(
            self._timestamps(self.dataset_spec.development_range),
            purge_width_steps=purge_width_steps,
            embargo_width_steps=embargo_width_steps,
        )
        return SplitArtifact(
            split_version="split_v1_walkforward",
            purge_width_steps=purge_width_steps,
            embargo_width_steps=embargo_width_steps,
            fold_generation_config=self.dataset_spec.walkforward,
            development_window=self._window_from_range(self.dataset_spec.development_range),
            train_window=self._window_from_range(self.dataset_spec.train_range),
            validation_window=self._window_from_range(self.dataset_spec.validation_range),
            final_untouched_test_window=self._window_from_range(self.dataset_spec.final_untouched_test_range),
            folds=folds,
        )

    def _generate_walkforward_folds(
        self,
        development_timestamps: list[datetime],
        purge_width_steps: int,
        embargo_width_steps: int,
    ) -> list[WalkForwardFold]:
        config = self.dataset_spec.walkforward
        if len(development_timestamps) < config.train_window_steps + config.validation_window_steps:
            raise ValueError("development region is too short for configured walk-forward windows")

        folds: list[WalkForwardFold] = []
        validation_start_index = config.train_window_steps
        max_validation_start = len(development_timestamps) - config.validation_window_steps
        advance = max(
            config.step_size_steps if config.step_size_steps is not None else config.validation_window_steps,
            config.validation_window_steps + embargo_width_steps,
        )

        while validation_start_index <= max_validation_start:
            validation_end_index = validation_start_index + config.validation_window_steps - 1
            train_end_index = validation_start_index - 1
            folds.append(
                WalkForwardFold(
                    fold_id=f"wf-{len(folds):02d}",
                    train_window=SplitWindow(
                        start=development_timestamps[0],
                        end=development_timestamps[train_end_index],
                    ),
                    validation_window=SplitWindow(
                        start=development_timestamps[validation_start_index],
                        end=development_timestamps[validation_end_index],
                    ),
                    purge_width_steps=purge_width_steps,
                    embargo_width_steps=embargo_width_steps,
                    horizon_steps=self.reward_spec.horizon_steps,
                )
            )
            validation_start_index += advance

        if not folds:
            raise ValueError("walk-forward generation produced no folds")
        return folds

    def _validate_split_ranges(self) -> None:
        required_timestamp_count = self.reward_spec.horizon_steps + 1
        segment_ranges = {
            "train": self.dataset_spec.train_range,
            "validation": self.dataset_spec.validation_range,
            "final_untouched_test": self.dataset_spec.final_untouched_test_range,
        }
        for split_name, split_range in segment_ranges.items():
            if len(self._timestamps(split_range)) < required_timestamp_count:
                raise ValueError(
                    f"{split_name} split requires at least {required_timestamp_count} timestamps for horizon "
                    f"{self.reward_spec.horizon_steps}"
                )
        if self.dataset_spec.walkforward.validation_window_steps < required_timestamp_count:
            raise ValueError(
                "walk-forward validation_window_steps must be at least horizon_steps + 1 to avoid empty folds"
            )
        if self.dataset_spec.walkforward.train_window_steps < required_timestamp_count:
            raise ValueError(
                "walk-forward train_window_steps must be at least horizon_steps + 1 to avoid empty folds"
            )

    def _window_from_range(self, split_range: TimeRange) -> SplitWindow:
        return SplitWindow(start=split_range.start, end=split_range.end)

    # ------------------------------------------------------------------
    # Observation Assembly
    # ------------------------------------------------------------------

    def _build_observation(
        self,
        indexed: _Index,
        target_symbol: str,
        event_time: datetime,
        history_start: datetime,
    ) -> ObservationContext:
        raw_surface: dict[str, RawScaleTensor] = {}

        for scale_spec in self.trajectory_spec.scale_preset:
            tensor = self._build_scale_tensor(indexed, target_symbol, event_time, history_start, scale_spec)
            raw_surface[scale_spec.label] = tensor

        derived = self._build_derived_surface(indexed, target_symbol, event_time)

        return ObservationContext(
            as_of=event_time,
            observation_schema=self.observation_schema,
            target_symbol=target_symbol,
            target_asset_index=self.dataset_spec.symbols.index(target_symbol),
            raw_surface=raw_surface,
            derived_surface=derived,
            metadata={"slice_id": self.dataset_spec.slice_id},
        )

    def _build_scale_tensor(
        self,
        indexed: _Index,
        target_symbol: str,
        event_time: datetime,
        history_start: datetime,
        scale_spec: ScaleSpec,
    ) -> RawScaleTensor:
        symbols = self.dataset_spec.symbols
        exchanges = self.dataset_spec.exchanges
        streams = self.dataset_spec.stream_universe
        field_axis = self.observation_schema.field_axis

        n_t = scale_spec.num_buckets
        n_sym = len(symbols)
        n_exc = len(exchanges)
        n_str = len(streams)
        total_fields = sum(len(field_axis.get(s, [])) for s in streams)
        flat_size = n_t * n_sym * n_exc * n_str * total_fields

        values: list[float] = [0.0] * flat_size
        age: list[float] = [0.0] * flat_size
        padding: list[bool] = [False] * flat_size
        unavailable_by_contract: list[bool] = [False] * flat_size
        missing: list[bool] = [False] * flat_size
        stale: list[bool] = [False] * flat_size

        interval = timedelta(seconds=scale_spec.resolution_seconds)

        # Field offsets hesabı (stream sırasına göre kümülatif)
        stream_field_offsets: dict[str, int] = {}
        offset = 0
        for s in streams:
            stream_field_offsets[s] = offset
            offset += len(field_axis.get(s, []))

        for t_idx in range(n_t):
            # En yeni bucket t_idx = num_buckets-1 (en sağ / en güncel)
            bucket_offset = n_t - 1 - t_idx  # 0 = en güncel
            slice_time = event_time - interval * bucket_offset
            is_padding = slice_time < history_start

            for sym_idx, symbol in enumerate(symbols):
                for exc_idx, exchange in enumerate(exchanges):
                    for str_idx, stream in enumerate(streams):
                        fields = field_axis.get(stream, [])
                        f_offset = stream_field_offsets[stream]
                        available = self.dataset_spec.stream_available(exchange, stream)

                        # Flat index hesabı
                        base = (
                            t_idx * n_sym * n_exc * n_str * total_fields
                            + sym_idx * n_exc * n_str * total_fields
                            + exc_idx * n_str * total_fields
                            + str_idx * total_fields
                            + f_offset
                        )

                        if is_padding:
                            for fi in range(len(fields)):
                                idx = base + fi
                                padding[idx] = True
                                missing[idx] = False
                                unavailable_by_contract[idx] = False
                            continue

                        if not available:
                            for fi in range(len(fields)):
                                idx = base + fi
                                unavailable_by_contract[idx] = True
                                padding[idx] = False
                                missing[idx] = False
                            continue

                        latest = self._latest_event(indexed, symbol, exchange, stream, slice_time)
                        if latest is None:
                            for fi in range(len(fields)):
                                idx = base + fi
                                missing[idx] = True
                                padding[idx] = False
                                unavailable_by_contract[idx] = False
                            continue

                        event_age = (slice_time - latest.event_time).total_seconds()
                        is_stale = event_age > self.trajectory_spec.stale_after_seconds

                        for fi, field_name in enumerate(fields):
                            idx = base + fi
                            raw_val = latest.fields.get(field_name, math.nan)
                            values[idx] = raw_val if not math.isnan(raw_val) else 0.0
                            age[idx] = event_age
                            stale[idx] = is_stale
                            padding[idx] = False
                            unavailable_by_contract[idx] = False
                            missing[idx] = math.isnan(raw_val)

        shape = [n_t, n_sym, n_exc, n_str, total_fields]
        return RawScaleTensor(
            scale_label=scale_spec.label,
            shape=shape,
            values=values,
            age=age,
            padding=padding,
            unavailable_by_contract=unavailable_by_contract,
            missing=missing,
            stale=stale,
        )

    # ------------------------------------------------------------------
    # Derived Surface (Target-Centric, V1 Scope)
    # ------------------------------------------------------------------

    def _build_derived_surface(
        self,
        indexed: _Index,
        target_symbol: str,
        event_time: datetime,
    ) -> DerivedSurface:
        channels: list[DerivedChannel] = []

        # --- Venue-pair price spread (O(n²) pairwise, target symbol) ---
        exchanges = self.dataset_spec.exchanges
        for exc_a, exc_b in combinations(exchanges, 2):
            for stream in ("mark_price", "bbo"):
                if not (
                    self.dataset_spec.stream_available(exc_a, stream)
                    and self.dataset_spec.stream_available(exc_b, stream)
                ):
                    continue
                ev_a = self._latest_event(indexed, target_symbol, exc_a, stream, event_time)
                ev_b = self._latest_event(indexed, target_symbol, exc_b, stream, event_time)
                if ev_a is None or ev_b is None:
                    spread = math.nan
                else:
                    # mark_price → "mark_price" field; bbo → "mid" field
                    field = "mark_price" if stream == "mark_price" else "mid"
                    price_a = ev_a.fields.get(field, math.nan)
                    price_b = ev_b.fields.get(field, math.nan)
                    spread = price_a - price_b if not (math.isnan(price_a) or math.isnan(price_b)) else math.nan
                key = f"venue_pair_{stream}_spread_{exc_a}_{exc_b}"
                channels.append(
                    DerivedChannel(
                        key=key,
                        description=f"{stream} price spread: {exc_a} - {exc_b} for {target_symbol}",
                        values=[0.0 if math.isnan(spread) else spread],
                        shape=[1],
                    )
                )
                break  # mark_price öncelikli; her çift için tek kanal

        # --- Target vs diğer symbol relative move ---
        target_mark = self._best_price(indexed, target_symbol, event_time)
        for symbol in self.dataset_spec.symbols:
            if symbol == target_symbol:
                continue
            other_mark = self._best_price(indexed, symbol, event_time)
            if target_mark is None or other_mark is None or other_mark == 0.0:
                rel = math.nan
            else:
                rel = (target_mark - other_mark) / other_mark
            channels.append(
                DerivedChannel(
                    key=f"relative_move_{target_symbol}_vs_{symbol}",
                    description=f"price return of {target_symbol} relative to {symbol}",
                    values=[0.0 if math.isnan(rel) else rel],
                    shape=[1],
                )
            )

        return DerivedSurface(channels=sorted(channels, key=lambda channel: channel.key))

    # ------------------------------------------------------------------
    # Action Feasibility
    # ------------------------------------------------------------------

    def _build_action_feasibility(
        self,
        indexed: _Index,
        symbol: str,
        event_time: datetime,
    ) -> ActionFeasibilitySurface:
        """Decision-time bilgisinden venue × size × leverage feasibility matrisi üret.

        Future fiyat veya next_time kullanılmaz.
        """
        exchanges = self.dataset_spec.exchanges
        size_bands = self.action_space.size_bands
        leverage_bands = self.action_space.leverage_bands

        # Her exchange için mark veya bbo fiyatı ve OI varlığını kontrol et
        venue_liquid: dict[str, bool] = {}
        for exchange in exchanges:
            price = self._venue_price(indexed, symbol, exchange, event_time)
            oi_event = self._latest_event(indexed, symbol, exchange, "open_interest", event_time)
            oi_ok = (
                oi_event is None  # OI stream unavailable → pas geç
                or not self.dataset_spec.stream_available(exchange, "open_interest")
                or (oi_event.fields.get("open_interest", 0.0) or 0.0) > 0.0
            )
            venue_liquid[exchange] = price is not None and price > 0.0 and oi_ok

        surface: dict[str, dict[str, dict[str, dict[str, FeasibilityCell]]]] = {}

        for action in self.action_space.actions:
            surface[action.key] = {}
            if action.key == "abstain":
                # abstain her zaman feasible; venue/band gerekmiyor
                for venue in exchanges:
                    surface[action.key][venue] = {}
                    for sb in size_bands:
                        surface[action.key][venue][sb.key] = {}
                        for lb in leverage_bands:
                            surface[action.key][venue][sb.key][lb.key] = FeasibilityCell(feasible=True)
                continue

            for venue in exchanges:
                surface[action.key][venue] = {}
                for sb in size_bands:
                    surface[action.key][venue][sb.key] = {}
                    for lb in leverage_bands:
                        feasible = venue_liquid.get(venue, False)
                        reason = "" if feasible else "no_liquid_price_or_oi"
                        surface[action.key][venue][sb.key][lb.key] = FeasibilityCell(
                            feasible=feasible, reason=reason
                        )

        return ActionFeasibilitySurface(surface=surface)

    # ------------------------------------------------------------------
    # Reward Context & Timeline
    # ------------------------------------------------------------------

    def _build_reward_context(
        self,
        indexed: _Index,
        symbol: str,
        event_time: datetime,
    ) -> RewardContext:
        """Venue-specific execution reference; yalnızca decision-time bilgisi."""
        venues: dict[str, VenueExecutionRef] = {}
        for exchange in self.dataset_spec.exchanges:
            price = self._venue_price(indexed, symbol, exchange, event_time)
            if price is None:
                continue
            funding_event = self._latest_event(indexed, symbol, exchange, "funding", event_time)
            funding_rate = 0.0
            # Use large-but-finite sentinel instead of float("inf") for JSON safety.
            # 86_400 seconds = 24h; clearly beyond any stale_after_seconds threshold.
            _NO_DATA_AGE = 86_400.0
            funding_age: float = _NO_DATA_AGE
            if funding_event is not None:
                funding_rate = funding_event.fields.get("funding_rate", 0.0) or 0.0
                funding_age = (event_time - funding_event.event_time).total_seconds()

            venues[exchange] = VenueExecutionRef(
                exchange=exchange,
                reference_price=price,
                fee_regime_bps=self.reward_spec.fee_bps,
                slippage_proxy_bps=self.reward_spec.slippage_bps,
                funding_rate=funding_rate,
                funding_freshness_seconds=funding_age,
            )

        return RewardContext(
            venues=venues,
            hold_horizon_steps=self.reward_spec.horizon_steps,
            turnover_state=0.0,
            previous_position_state="flat",
            selected_venue=None,
        )

    def _build_reward_timeline(
        self,
        indexed: _Index,
        symbol: str,
        timestamps: list[datetime],
        step_index: int,
    ) -> RewardTimeline:
        """horizon_steps adım için venue-specific referans fiyat serisi."""
        horizon = self.reward_spec.horizon_steps
        venue_series: dict[str, list[float]] = {}

        for exchange in self.dataset_spec.exchanges:
            series: list[float] = []
            for h in range(1, horizon + 1):
                future_idx = step_index + h
                if future_idx < len(timestamps):
                    future_time = timestamps[future_idx]
                    price = self._venue_price(indexed, symbol, exchange, future_time)
                    series.append(price if price is not None else 0.0)
                else:
                    series.append(0.0)
            venue_series[exchange] = series

        return RewardTimeline(horizon_steps=horizon, venue_reference_series=venue_series)

    def _build_policy_state(self, prev_step: TrajectoryStep | None) -> PolicyState | None:
        """Önceki step'ten inventory snapshot; ilk step'te None."""
        if prev_step is None:
            return None
        ctx = prev_step.reward_context
        return PolicyState(
            previous_position_side=cast(
                Literal["flat", "long", "short"],
                ctx.previous_position_state,
            ),
            previous_venue=ctx.selected_venue,
            hold_age_steps=0,
            turnover_accumulator=ctx.turnover_state,
        )

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _venue_price(
        self,
        indexed: _Index,
        symbol: str,
        exchange: str,
        event_time: datetime,
    ) -> float | None:
        """Tek bir exchange için mark veya mid fiyatı; averaging yok."""
        if self.dataset_spec.stream_available(exchange, "mark_price"):
            ev = self._latest_event(indexed, symbol, exchange, "mark_price", event_time)
            if ev is not None:
                v = ev.fields.get("mark_price", math.nan)
                if not math.isnan(v):
                    return v
        if self.dataset_spec.stream_available(exchange, "bbo"):
            ev = self._latest_event(indexed, symbol, exchange, "bbo", event_time)
            if ev is not None:
                v = ev.fields.get("mid", math.nan)
                if not math.isnan(v):
                    return v
        return None

    def _best_price(
        self,
        indexed: _Index,
        symbol: str,
        event_time: datetime,
    ) -> float | None:
        """Tüm exchange'ler arasında ilk bulunan geçerli fiyat (averaging yok)."""
        for exchange in self.dataset_spec.exchanges:
            price = self._venue_price(indexed, symbol, exchange, event_time)
            if price is not None:
                return price
        return None

    def _latest_event(
        self,
        indexed: _Index,
        symbol: str,
        exchange: str,
        stream: str,
        event_time: datetime,
    ) -> NormalizedMarketEvent | None:
        key = (symbol, exchange, stream)
        for event in reversed(indexed.get(key, [])):
            if event.event_time <= event_time:
                return event
        return None

    def _timestamps(self, split_range: TimeRange) -> list[datetime]:
        timestamps: list[datetime] = []
        current = split_range.start
        # En kaba ölçek adımını kullan (en büyük resolution)
        base_interval = timedelta(seconds=self.dataset_spec.sampling_interval_seconds)
        while current <= split_range.end:
            timestamps.append(current)
            current += base_interval
        return timestamps

    def _index_events(
        self, events: list[NormalizedMarketEvent]
    ) -> _Index:
        indexed: _Index = defaultdict(list)
        for event in events:
            indexed[(event.symbol, event.exchange, event.stream_type)].append(event)
        for keyed_events in indexed.values():
            keyed_events.sort(key=lambda item: item.event_time)
        return indexed


def _chunked(items: list[TrajectoryStep], chunk_size: int) -> Iterable[list[TrajectoryStep]]:
    for index in range(0, len(items), chunk_size):
        yield items[index : index + chunk_size]

from __future__ import annotations

import gc
import logging
import math
from collections import defaultdict
from collections.abc import Iterable, Iterator
from datetime import datetime, timedelta, timezone
from itertools import combinations
from pathlib import Path
from typing import Literal, cast

import numpy as np

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
    TrajectoryManifest,
    TrajectoryRecord,
    TrajectorySpec,
    TrajectoryStep,
    VenueExecutionRef,
    WalkForwardFold,
)
from quantlab_ml.rewards import RewardEngine

class _CompactEvent:
    """Minimum event representation for the build-time index.

    Replaces the full NormalizedMarketEvent Pydantic object in the internal
    index, eliminating Pydantic overhead, per-instance __dict__, and duplicate
    symbol/exchange/stream_type storage (those are already encoded in the dict key).

    Fields kept:
        event_time_ts: UNIX timestamp in seconds (float64 scalar).
                       Replaces datetime object (~56 bytes) with a raw float.
        fields:        Raw field dict; semantically identical to
                       NormalizedMarketEvent.fields.

    __slots__ removes the per-instance __dict__ (~64 bytes saved per event).
    Per-event RAM: ~514 bytes vs ~1060 bytes for the full Pydantic object.
    For 43 M events (controlled-remote-day): ~22 GB vs ~45 GB.
    """

    __slots__ = ("event_time_ts", "fields")

    def __init__(self, event_time_ts: float, fields: dict[str, float]) -> None:
        self.event_time_ts: float = event_time_ts
        self.fields: dict[str, float] = fields


# Each index bucket stores a sorted numpy timestamp array paired with the
# compact event list.  The numpy array enables O(log N) binary search vs the
# O(N) reverse linear scan that would be required with a plain Python list.
# Tuple layout: (times: np.ndarray[float64, shape=(N,)], events: list[_CompactEvent])
_IndexBucket = tuple[np.ndarray, list[_CompactEvent]]  # times=float64, events=compact
_Index = dict[tuple[str, str, str], _IndexBucket]

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

    def build(self, events: Iterable[NormalizedMarketEvent]) -> TrajectoryBundle:
        """⚠️  FIXTURE / TEST COMPAT PATH — do not call from production code.

        Builds the entire TrajectoryBundle in memory.  This OOMs for
        production-profile snapshots (tens of thousands of steps).  Use
        build_to_directory() for production builds.
        """
        indexed, event_count, history_start_from_events = self._index_events(events)
        history_start = history_start_from_events or self.dataset_spec.train_range.start
        logger.info(
            "trajectory_build_started slice_id=%s event_count=%d symbol_count=%d exchange_count=%d",
            self.dataset_spec.slice_id,
            event_count,
            len(self.dataset_spec.symbols),
            len(self.dataset_spec.exchanges),
        )
        self._validate_split_ranges()
        split_artifact = self._build_split_artifact()
        development_records = self._build_split(
            "development",
            self.dataset_spec.development_range,
            indexed,
            history_start,
        )
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
            development_records=development_records,
        )
        logger.info(
            "trajectory_build_completed slice_id=%s development_records=%d train_records=%d validation_records=%d "
            "final_test_records=%d development_steps=%d train_steps=%d validation_steps=%d final_test_steps=%d "
            "fold_count=%d",
            self.dataset_spec.slice_id,
            len(development_records),
            len(splits["train"]),
            len(splits["validation"]),
            len(splits["final_untouched_test"]),
            sum(len(record.steps) for record in development_records),
            sum(len(record.steps) for record in splits["train"]),
            sum(len(record.steps) for record in splits["validation"]),
            sum(len(record.steps) for record in splits["final_untouched_test"]),
            len(split_artifact.folds),
        )
        return bundle

    def build_to_directory(
        self,
        events: Iterable[NormalizedMarketEvent],
        output_dir: Path,
    ) -> TrajectoryManifest:
        """PRODUCTION PATH — build trajectories streaming to a JSONL directory.

        Writes one TrajectoryRecord at a time to disk, never holding more than
        one record in memory (plus the event index).  Returns the manifest
        written to output_dir/manifest.json.

        Memory profile:
            - Event index (~40 GB for controlled-remote-day) held throughout.
            - At most max_episode_steps TrajectoryStep objects in RAM at a time.
            - No full TrajectoryBundle assembly.
        """
        # Avoid circular import (streaming_store imports contracts, not builder)
        from quantlab_ml.trajectories.streaming_store import TrajectoryDirectoryStore

        indexed, event_count, history_start_from_events = self._index_events(events)
        history_start = history_start_from_events or self.dataset_spec.train_range.start
        logger.info(
            "trajectory_build_started slice_id=%s event_count=%d symbol_count=%d exchange_count=%d",
            self.dataset_spec.slice_id,
            event_count,
            len(self.dataset_spec.symbols),
            len(self.dataset_spec.exchanges),
        )
        self._validate_split_ranges()
        split_artifact = self._build_split_artifact()

        split_names = ["development", "train", "validation", "final_untouched_test"]
        split_ranges = [
            self.dataset_spec.development_range,
            self.dataset_spec.train_range,
            self.dataset_spec.validation_range,
            self.dataset_spec.final_untouched_test_range,
        ]

        manifest = TrajectoryManifest(
            dataset_spec=self.dataset_spec,
            trajectory_spec=self.trajectory_spec,
            action_space=self.action_space,
            reward_spec=self.reward_spec,
            observation_schema=self.observation_schema,
            split_artifact=split_artifact,
            split_names=split_names,
        )
        TrajectoryDirectoryStore.write_manifest(output_dir, manifest)

        total_records = 0
        total_steps = 0
        for split_name, split_range in zip(split_names, split_ranges):
            logger.info("building_split split=%s", split_name)
            records_iter = self._build_split_iter(split_name, split_range, indexed, history_start)
            rec_count, step_count = TrajectoryDirectoryStore.write_split(
                output_dir, split_name, records_iter
            )
            total_records += rec_count
            total_steps += step_count
            gc.collect()
            logger.info(
                "split_complete split=%s records=%d steps=%d", split_name, rec_count, step_count
            )

        logger.info(
            "trajectory_build_completed slice_id=%s total_records=%d total_steps=%d "
            "fold_count=%d output_dir=%s",
            self.dataset_spec.slice_id,
            total_records,
            total_steps,
            len(split_artifact.folds),
            output_dir,
        )
        return manifest

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
                            Literal["train", "validation", "final_untouched_test", "development"],
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

    def _build_split_iter(
        self,
        split_name: str,
        split_range: TimeRange,
        indexed: _Index,
        history_start: datetime,
    ) -> Iterator[TrajectoryRecord]:
        """Generator: yield TrajectoryRecord objects one at a time (streaming build).

        Records are yielded every max_episode_steps steps within each symbol,
        allowing the caller to write-and-discard each record before the next
        one is constructed.  Peak memory = 1 active chunk (max_episode_steps
        TrajectoryStep objects) per iteration.
        """
        timestamps = self._timestamps(split_range)
        if len(timestamps) <= self.reward_spec.horizon_steps:
            return

        max_chunk = self.trajectory_spec.max_episode_steps
        usable_count = len(timestamps) - self.reward_spec.horizon_steps

        for symbol in self.dataset_spec.symbols:
            chunk: list[TrajectoryStep] = []
            chunk_index = 0
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
                chunk.append(step)
                prev_step = step

                if len(chunk) == max_chunk:
                    yield TrajectoryRecord(
                        trajectory_id=f"{split_name}-{symbol.lower()}-{chunk_index}",
                        split=cast(
                            Literal["train", "validation", "final_untouched_test", "development"],
                            split_name,
                        ),
                        target_symbol=symbol,
                        start_time=chunk[0].event_time,
                        end_time=chunk[-1].event_time,
                        steps=chunk,
                        terminal=True,
                        terminal_reason=self.trajectory_spec.terminal_semantics,
                    )
                    chunk = []
                    chunk_index += 1
                    # prev_step stays set to carry policy_state forward

            # yield remainder
            if chunk:
                yield TrajectoryRecord(
                    trajectory_id=f"{split_name}-{symbol.lower()}-{chunk_index}",
                    split=cast(
                        Literal["train", "validation", "final_untouched_test", "development"],
                        split_name,
                    ),
                    target_symbol=symbol,
                    start_time=chunk[0].event_time,
                    end_time=chunk[-1].event_time,
                    steps=chunk,
                    terminal=True,
                    terminal_reason=self.trajectory_spec.terminal_semantics,
                )

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

        values                = np.zeros(flat_size, dtype=np.float32)
        age                   = np.zeros(flat_size, dtype=np.float32)
        padding               = np.zeros(flat_size, dtype=np.bool_)
        unavailable_by_contract = np.zeros(flat_size, dtype=np.bool_)
        missing               = np.zeros(flat_size, dtype=np.bool_)
        stale                 = np.zeros(flat_size, dtype=np.bool_)

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
            slice_time_ts = slice_time.timestamp()  # float; used for _latest_event + age calc
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

                        latest = self._latest_event(indexed, symbol, exchange, stream, slice_time_ts)
                        if latest is None:
                            for fi in range(len(fields)):
                                idx = base + fi
                                missing[idx] = True
                                padding[idx] = False
                                unavailable_by_contract[idx] = False
                            continue

                        event_age = slice_time_ts - latest.event_time_ts
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
                ev_a = self._latest_event(indexed, target_symbol, exc_a, stream, event_time.timestamp())
                ev_b = self._latest_event(indexed, target_symbol, exc_b, stream, event_time.timestamp())
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
        event_time_ts = event_time.timestamp()
        for exchange in exchanges:
            price = self._venue_price(indexed, symbol, exchange, event_time)
            oi_event = self._latest_event(indexed, symbol, exchange, "open_interest", event_time_ts)
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
        event_time_ts = event_time.timestamp()
        for exchange in self.dataset_spec.exchanges:
            price = self._venue_price(indexed, symbol, exchange, event_time)
            if price is None:
                continue
            funding_event = self._latest_event(indexed, symbol, exchange, "funding", event_time_ts)
            funding_rate = 0.0
            # Use large-but-finite sentinel instead of float("inf") for JSON safety.
            # 86_400 seconds = 24h; clearly beyond any stale_after_seconds threshold.
            _NO_DATA_AGE = 86_400.0
            funding_age: float = _NO_DATA_AGE
            if funding_event is not None:
                funding_rate = funding_event.fields.get("funding_rate", 0.0) or 0.0
                funding_age = event_time_ts - funding_event.event_time_ts

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
        event_time_ts = event_time.timestamp()
        if self.dataset_spec.stream_available(exchange, "mark_price"):
            ev = self._latest_event(indexed, symbol, exchange, "mark_price", event_time_ts)
            if ev is not None:
                v = ev.fields.get("mark_price", math.nan)
                if not math.isnan(v):
                    return v
        if self.dataset_spec.stream_available(exchange, "bbo"):
            ev = self._latest_event(indexed, symbol, exchange, "bbo", event_time_ts)
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
        event_time_ts: float,
    ) -> _CompactEvent | None:
        """Return the most-recent compact event at or before a UNIX timestamp.

        Uses binary search (np.searchsorted) on the pre-built sorted timestamp
        array for O(log N) lookup instead of O(N) reverse linear scan.
        For a trade bucket with 1.44 M events this drops from ~720 K comparisons
        on average to ~21.

        Args:
            event_time_ts: Upper-bound as UNIX seconds (float).  Callers
                           pre-compute this with ``dt.timestamp()`` to avoid
                           repeated object construction inside the hot path.

        Returns:
            The most-recent _CompactEvent with event_time_ts <= query, or None.
        """
        bucket = indexed.get((symbol, exchange, stream))
        if bucket is None:
            return None
        times_arr, events = bucket
        if len(times_arr) == 0:
            return None
        pos = int(np.searchsorted(times_arr, event_time_ts, side="right")) - 1
        if pos < 0:
            return None
        return events[pos]

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
        self, events: Iterable[NormalizedMarketEvent]
    ) -> tuple[_Index, int, datetime | None]:
        """Index events as compact (event_time_ts, fields) objects.

        Each raw NormalizedMarketEvent is converted to a _CompactEvent
        immediately — discarding the heavy Pydantic object — then accumulated
        in per-bucket temporary lists.  After all events are consumed the lists
        are sorted and the timestamp column is frozen into a numpy float64 array
        so that _latest_event() can use np.searchsorted for O(log N) lookup.

        Memory profile:
          Building phase:  ~32 bytes per timestamp (Python float) + 8 bytes per
                           event pointer in the tmp lists (~4 GB at 106 M events).
          After freeze:     numpy column  ~8 bytes per event (~0.85 GB);
                           Python floats GC'd (saves ~3.4 GB).
          CompactEvent:    ~432 bytes per event including fields dict.
          Index total:     ~47 GB for 106 M events (vs ~46 GB before bisect;
                           the numpy column adds only 0.85 GB).

        split / leakage / causality:
          The timestamp comparison semantics are identical to the prior code:
          ``event_time_ts <= query_ts``.  No forward-looking information added.
        """
        # --- pass 1: accumulate per-bucket parallel lists ---
        tmp_times: dict[tuple[str, str, str], list[float]] = defaultdict(list)
        tmp_events: dict[tuple[str, str, str], list[_CompactEvent]] = defaultdict(list)
        count = 0
        min_time_ts: float | None = None
        for event in events:
            et_ts = event.event_time.timestamp()
            compact = _CompactEvent(
                event_time_ts=et_ts,
                fields=dict(event.fields),
            )
            key = (event.symbol, event.exchange, event.stream_type)
            tmp_times[key].append(et_ts)
            tmp_events[key].append(compact)
            count += 1
            if min_time_ts is None or et_ts < min_time_ts:
                min_time_ts = et_ts

        # --- pass 2: sort each bucket + freeze timestamp column into numpy ---
        indexed: _Index = {}
        for key in tmp_times:
            times_list = tmp_times[key]
            evs_list = tmp_events[key]
            # sort ascending by timestamp (in-place via argsort on numpy)
            times_arr = np.array(times_list, dtype=np.float64)
            sort_idx = np.argsort(times_arr, kind="stable")
            frozen_times = times_arr[sort_idx]
            sorted_events = [evs_list[i] for i in sort_idx]
            indexed[key] = (frozen_times, sorted_events)

        del tmp_times, tmp_events
        gc.collect()

        min_time = (
            datetime.fromtimestamp(min_time_ts, tz=timezone.utc)
            if min_time_ts is not None
            else None
        )
        return indexed, count, min_time


def _chunked(items: list[TrajectoryStep], chunk_size: int) -> Iterable[list[TrajectoryStep]]:
    for index in range(0, len(items), chunk_size):
        yield items[index : index + chunk_size]

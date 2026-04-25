#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from quantlab_ml.contracts import DatasetSpec
from quantlab_ml.trajectories.event_token_cache import (
    EventTokenCacheSplitWriter,
    _EventCandidate,
    _InformativeUnit,
    _canonical_payload_identity,
    datetime_to_epoch_millis,
)

PROFILE_VERSION = "ql033_r6_window_base_micro_profile_v1"
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
EXCHANGES = ("binance", "bybit", "okx")
STREAMS = ("trade", "bbo")
R4_PARTIAL_PROFILE_PATH = Path(
    "outputs/ql033-r4-hotpath-20260424-rerun2-thin/run_a/trajectories/"
    "event_token_cache_v1/development_partial_selector_profile.json"
)
R4_BUILD_TIME_PATH = Path("outputs/ql033-r4-hotpath-20260424-rerun2-thin/run_a/build.time.json")
R4_BUILD_GATE_SECONDS = 2207.0


@dataclass(slots=True)
class _ProfileEvent:
    event_time_ts: float
    fields: dict[str, object]
    source_label_id: int
    source_event_index: int


@dataclass(slots=True)
class _WindowedLaneTable:
    table: Any
    ts_ms: np.ndarray


class _ProfileOnlyEventTokenCacheSplitWriter(EventTokenCacheSplitWriter):
    def _write_partial_selector_profile(self, *, status: str) -> None:
        return


class _ReferenceWindowBaseWriter(_ProfileOnlyEventTokenCacheSplitWriter):
    def _compute_window_base_optimized(self, *, decision_time: datetime):
        return self._compute_window_base_slow_reference(decision_time=decision_time)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a bounded QL-033 R6 window_base reference-vs-candidate micro-profile. "
            "Synthetic mode is continuity-only. S3 compact mode uses bounded real proof-slice data."
        )
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--source", choices=("synthetic", "s3-compact"), default="synthetic")
    parser.add_argument("--data-config", type=Path, default=Path("outputs/ql033-r5-windowbase-20260424-rerun2-thin/proof-slice.data.yaml"))
    parser.add_argument("--s3-env-file", type=Path, default=Path(".env"))
    parser.add_argument("--object-list-json", type=Path)
    parser.add_argument("--materialize-s3-cache", action="store_true")
    parser.add_argument("--real-index-mode", choices=("windowed", "full"), default="windowed")
    parser.add_argument("--miss-rows", type=int, default=360)
    parser.add_argument("--cache-hit-rows", type=int, default=360)
    parser.add_argument("--start", default="2026-01-25T16:00:00Z")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.source == "s3-compact":
        profile = run_s3_compact_profile(
            start=_parse_utc(args.start),
            miss_rows=args.miss_rows,
            cache_hit_rows=args.cache_hit_rows,
            output_root=args.output_json.parent,
            data_config=args.data_config,
            s3_env_file=args.s3_env_file,
            object_list_json=args.object_list_json,
            materialize_s3_cache=args.materialize_s3_cache,
            real_index_mode=args.real_index_mode,
        )
    else:
        profile = run_profile(
            start=_parse_utc(args.start),
            miss_rows=args.miss_rows,
            cache_hit_rows=args.cache_hit_rows,
            output_root=args.output_json.parent,
        )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(profile, indent=2, sort_keys=True))
    return 0


def run_profile(
    *,
    start: datetime,
    miss_rows: int,
    cache_hit_rows: int,
    output_root: Path,
) -> dict[str, Any]:
    if miss_rows <= 0:
        raise ValueError("miss_rows must be positive")
    if cache_hit_rows < 0:
        raise ValueError("cache_hit_rows must be non-negative")
    cache_hit_rows = min(cache_hit_rows, miss_rows)
    dataset_spec = _dataset_spec(start=start, minute_count=miss_rows)
    indexed = _synthetic_indexed_events(start=start, minute_count=miss_rows + 1)
    rows = (
        [("BTCUSDT", start + timedelta(minutes=index)) for index in range(miss_rows)]
        + [("ETHUSDT", start + timedelta(minutes=index)) for index in range(cache_hit_rows)]
    )

    reference_writer = _ReferenceWindowBaseWriter(
        directory=output_root / "reference_scratch",
        split_name="development",
        dataset_spec=dataset_spec,
        indexed=indexed,
        source_labels=["synthetic://ql033-r6-window-base-profile"],
    )
    candidate_writer = _ProfileOnlyEventTokenCacheSplitWriter(
        directory=output_root / "candidate_scratch",
        split_name="development",
        dataset_spec=dataset_spec,
        indexed=indexed,
        source_labels=["synthetic://ql033-r6-window-base-profile"],
    )

    reference = _run_mode(reference_writer, rows=rows)
    candidate = _run_mode(candidate_writer, rows=rows)
    semantic_equivalence = _compare_reference_candidate(
        reference_writer=reference_writer,
        candidate_writer=candidate_writer,
        decision_times=[start + timedelta(minutes=index) for index in range(miss_rows)],
    )
    speedup = (
        reference["subphase_timings"]["window_base_total_wall_sec"]
        / candidate["subphase_timings"]["window_base_total_wall_sec"]
        if candidate["subphase_timings"]["window_base_total_wall_sec"] > 0.0
        else None
    )

    return {
        "profile_version": PROFILE_VERSION,
        "generated_at": datetime.now(tz=UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "task_scope": "R6 candidate micro-profile; not full remote proof",
        "data_source": {
            "kind": "synthetic_proof_shape_continuity_fixture",
            "representative_proof_slice_data": False,
            "reason": "local QL-033 mirrors retain manifests/diagnostics only; event shard payloads are absent",
        },
        "same_rows": True,
        "coverage": {
            "symbols": list(SYMBOLS),
            "venues": list(EXCHANGES),
            "streams": list(STREAMS),
            "cadence_seconds": 60,
            "start_around": "2026-01-25T16:00:00Z",
            "first_decision_timestamp": _iso(rows[0][1]),
            "last_decision_timestamp": _iso(rows[-1][1]),
            "latest_unique_decision_timestamp_seen": _iso(start + timedelta(minutes=miss_rows - 1)),
            "first_pass_miss_rows": reference["cache_misses"],
            "cache_reuse_rows": reference["cache_hits"],
            "includes_first_pass_miss_only_region": False,
            "includes_cache_reuse_transition": reference["cache_hits"] > 0,
            "cache_reuse_transition_feasible": True,
        },
        "modes": {
            "reference": reference,
            "candidate": candidate,
        },
        "semantic_equivalence": semantic_equivalence,
        "projection": {
            "synthetic_window_base_speedup": speedup,
            "projected_build_time_multiplier": None,
            "classification": "insufficient_evidence",
            "reason": "projection requires representative proof-slice market data, not synthetic continuity data",
        },
        "proceed_to_full_proof": False,
    }


def run_s3_compact_profile(
    *,
    start: datetime,
    miss_rows: int,
    cache_hit_rows: int,
    output_root: Path,
    data_config: Path,
    s3_env_file: Path,
    object_list_json: Path | None = None,
    materialize_s3_cache: bool = False,
    real_index_mode: str = "windowed",
) -> dict[str, Any]:
    from quantlab_ml.common import load_yaml
    from quantlab_ml.data.adapters import S3CompactedSource

    if miss_rows <= 0:
        raise ValueError("miss_rows must be positive")
    cache_hit_rows = min(max(cache_hit_rows, 0), miss_rows)
    dataset_spec = DatasetSpec.model_validate(load_yaml(data_config)["dataset"])
    unique_decision_times = [start + timedelta(minutes=index) for index in range(miss_rows)]
    rows = (
        [("BTCUSDT", decision_time) for decision_time in unique_decision_times]
        + [("ETHUSDT", decision_time) for decision_time in unique_decision_times[:cache_hit_rows]]
    )
    event_start = start - timedelta(seconds=60)
    event_end = unique_decision_times[-1]
    source = S3CompactedSource.from_env_file(s3_env_file)
    object_refs = _load_object_refs(object_list_json)
    output_root.mkdir(parents=True, exist_ok=True)
    if real_index_mode == "windowed":
        return _run_s3_compact_windowed_profile(
            source=source,
            dataset_spec=dataset_spec,
            source_labels_config={
                "data_config": str(data_config),
                "s3_state_key": source.state_key,
            },
            rows=rows,
            unique_decision_times=unique_decision_times,
            event_start=event_start,
            event_end=event_end,
            output_root=output_root,
            s3_env_file=s3_env_file,
            object_refs=object_refs,
            object_cache_dir=output_root / "s3_object_cache" if materialize_s3_cache else None,
        )

    indexed, source_labels, load_summary = _load_s3_compact_indexed(
        source=source,
        dataset_spec=dataset_spec,
        event_start=event_start,
        event_end=event_end,
        s3_env_file=s3_env_file,
    )
    (output_root / "real_data_load_summary.json").write_text(
        json.dumps(load_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    reference_writer = _ReferenceWindowBaseWriter(
        directory=output_root / "reference_scratch",
        split_name="development",
        dataset_spec=dataset_spec,
        indexed=indexed,
        source_labels=source_labels,
    )
    candidate_writer = _ProfileOnlyEventTokenCacheSplitWriter(
        directory=output_root / "candidate_scratch",
        split_name="development",
        dataset_spec=dataset_spec,
        indexed=indexed,
        source_labels=source_labels,
    )
    reference = _run_mode(reference_writer, rows=rows)
    candidate = _run_mode(candidate_writer, rows=rows)
    semantic_equivalence = _compare_reference_candidate(
        reference_writer=reference_writer,
        candidate_writer=candidate_writer,
        decision_times=unique_decision_times,
    )
    window_base_speedup = (
        reference["subphase_timings"]["window_base_total_wall_sec"]
        / candidate["subphase_timings"]["window_base_total_wall_sec"]
        if candidate["subphase_timings"]["window_base_total_wall_sec"] > 0.0
        else None
    )
    projection = _build_projection(window_base_speedup=window_base_speedup)
    first_pass_region_ok = reference["cache_misses"] >= 359
    cache_transition_ok = reference["cache_hits"] > 0
    return {
        "profile_version": PROFILE_VERSION,
        "generated_at": datetime.now(tz=UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "task_scope": "R6 real-data micro-profile; not full remote proof",
        "data_source": {
            "kind": "s3_compact_real_proof_slice",
            "representative_proof_slice_data": True,
            "data_config": str(data_config),
            "s3_state_key": source.state_key,
            "event_start": _iso(event_start),
            "event_end": _iso(event_end),
            "partition_count": load_summary["partition_count"],
            "object_count": load_summary["object_count"],
            "event_count": load_summary["event_count"],
        },
        "same_rows": True,
        "coverage": {
            "symbols": list(SYMBOLS),
            "venues": list(EXCHANGES),
            "streams": list(STREAMS),
            "cadence_seconds": 60,
            "start_around": "2026-01-25T16:00:00Z",
            "first_decision_timestamp": _iso(rows[0][1]),
            "last_decision_timestamp": _iso(rows[-1][1]),
            "latest_unique_decision_timestamp_seen": _iso(unique_decision_times[-1]),
            "first_pass_miss_rows": reference["cache_misses"],
            "cache_reuse_rows": reference["cache_hits"],
            "includes_first_pass_miss_only_region": first_pass_region_ok,
            "includes_cache_reuse_transition": cache_transition_ok,
            "cache_reuse_transition_feasible": True,
        },
        "modes": {
            "reference": reference,
            "candidate": candidate,
        },
        "semantic_equivalence": semantic_equivalence,
        "projection": projection,
        "proceed_to_full_proof": False,
    }


def _run_s3_compact_windowed_profile(
    *,
    source: Any,
    dataset_spec: DatasetSpec,
    source_labels_config: dict[str, str],
    rows: list[tuple[str, datetime]],
    unique_decision_times: list[datetime],
    event_start: datetime,
    event_end: datetime,
    output_root: Path,
    s3_env_file: Path,
    object_refs: list[dict[str, str]] | None,
    object_cache_dir: Path | None,
) -> dict[str, Any]:
    tables_by_lane, source_labels, load_summary = _load_s3_compact_tables(
        source=source,
        dataset_spec=dataset_spec,
        event_start=event_start,
        event_end=event_end,
        s3_env_file=s3_env_file,
        object_refs=object_refs,
        object_cache_dir=object_cache_dir,
    )
    (output_root / "real_data_load_summary.json").write_text(
        json.dumps(load_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    modes = {
        "reference": _empty_mode_summary(rows=rows),
        "candidate": _empty_mode_summary(rows=rows),
    }
    mismatches: list[str] = []
    cache_hit_rows_by_decision = {
        decision_time: []
        for decision_time in unique_decision_times
    }
    for target_symbol, decision_time in rows[len(unique_decision_times):]:
        cache_hit_rows_by_decision.setdefault(decision_time, []).append(target_symbol)

    for decision_index, decision_time in enumerate(unique_decision_times):
        if decision_index == 0 or (decision_index + 1) % 25 == 0 or decision_index + 1 == len(unique_decision_times):
            _progress(
                "windowed profile "
                f"{decision_index + 1}/{len(unique_decision_times)} decision_time={_iso(decision_time)}"
            )
        indexed = _indexed_from_tables_for_window(
            tables_by_lane=tables_by_lane,
            window_start=decision_time - timedelta(seconds=60),
            decision_time=decision_time,
        )
        row_targets = ["BTCUSDT", *cache_hit_rows_by_decision.get(decision_time, [])]
        reference_writer = _ReferenceWindowBaseWriter(
            directory=output_root / "reference_scratch",
            split_name="development",
            dataset_spec=dataset_spec,
            indexed=indexed,
            source_labels=source_labels,
        )
        candidate_writer = _ProfileOnlyEventTokenCacheSplitWriter(
            directory=output_root / "candidate_scratch",
            split_name="development",
            dataset_spec=dataset_spec,
            indexed=indexed,
            source_labels=source_labels,
        )
        reference = reference_writer._compute_window_base_slow_reference(decision_time=decision_time)
        candidate = candidate_writer._compute_window_base_optimized(decision_time=decision_time)
        if not mismatches:
            if [_candidate_snapshot(item) for item in reference.raw_candidates] != [
                _candidate_snapshot(item) for item in candidate.raw_candidates
            ]:
                mismatches.append(f"{_iso(decision_time)}:raw_candidates")
            elif [_candidate_snapshot(item) for item in reference.deduped_candidates] != [
                _candidate_snapshot(item) for item in candidate.deduped_candidates
            ]:
                mismatches.append(f"{_iso(decision_time)}:deduped_candidates")
            elif _window_base_snapshot(reference.window_base) != _window_base_snapshot(candidate.window_base):
                mismatches.append(f"{_iso(decision_time)}:window_base")
        _append_windowed_rows(
            mode_summary=modes["reference"],
            writer=reference_writer,
            computation=reference,
            row_targets=row_targets,
            decision_time=decision_time,
            first_row_index=decision_index,
        )
        _append_windowed_rows(
            mode_summary=modes["candidate"],
            writer=candidate_writer,
            computation=candidate,
            row_targets=row_targets,
            decision_time=decision_time,
            first_row_index=decision_index,
        )

    for mode_summary in modes.values():
        _finalize_mode_summary(mode_summary)

    reference_mode = modes["reference"]
    candidate_mode = modes["candidate"]
    window_base_speedup = (
        reference_mode["subphase_timings"]["window_base_total_wall_sec"]
        / candidate_mode["subphase_timings"]["window_base_total_wall_sec"]
        if candidate_mode["subphase_timings"]["window_base_total_wall_sec"] > 0.0
        else None
    )
    first_pass_region_ok = reference_mode["cache_misses"] >= 359
    cache_transition_ok = reference_mode["cache_hits"] > 0
    return {
        "profile_version": PROFILE_VERSION,
        "generated_at": datetime.now(tz=UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "task_scope": "R6 real-data windowed micro-profile; not full remote proof",
        "data_source": {
            "kind": "s3_compact_real_proof_slice_windowed",
            "representative_proof_slice_data": True,
            **source_labels_config,
            "event_start": _iso(event_start),
            "event_end": _iso(event_end),
            "partition_count": load_summary["partition_count"],
            "object_count": load_summary["object_count"],
            "event_count": load_summary["event_count"],
            "index_mode": "windowed",
        },
        "same_rows": True,
        "coverage": {
            "symbols": list(SYMBOLS),
            "venues": list(EXCHANGES),
            "streams": list(STREAMS),
            "cadence_seconds": 60,
            "start_around": "2026-01-25T16:00:00Z",
            "first_decision_timestamp": _iso(rows[0][1]),
            "last_decision_timestamp": _iso(max(decision_time for _, decision_time in rows)),
            "latest_unique_decision_timestamp_seen": _iso(unique_decision_times[-1]),
            "first_pass_miss_rows": reference_mode["cache_misses"],
            "cache_reuse_rows": reference_mode["cache_hits"],
            "includes_first_pass_miss_only_region": first_pass_region_ok,
            "includes_cache_reuse_transition": cache_transition_ok,
            "cache_reuse_transition_feasible": True,
        },
        "modes": modes,
        "semantic_equivalence": {
            "ordered_reference_candidate_match": not mismatches,
            "windows_compared": len(unique_decision_times),
            "mismatches": mismatches,
        },
        "projection": _build_projection(window_base_speedup=window_base_speedup),
        "proceed_to_full_proof": False,
    }


def _load_s3_compact_indexed(
    *,
    source: Any,
    dataset_spec: DatasetSpec,
    event_start: datetime,
    event_end: datetime,
    s3_env_file: Path,
) -> tuple[dict[tuple[str, str, str], tuple[np.ndarray, list[_ProfileEvent]]], list[str], dict[str, Any]]:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.fs as fs
    import pyarrow.parquet as pq

    from quantlab_ml.common import load_env_file

    values = load_env_file(s3_env_file)
    endpoint = values["S3_COMPACT_ENDPOINT"].replace("https://", "").replace("http://", "")
    filesystem = fs.S3FileSystem(
        access_key=values["S3_COMPACT_ACCESS_KEY"],
        secret_key=values["S3_COMPACT_SECRET_KEY"],
        region=values["S3_COMPACT_REGION"],
        endpoint_override=endpoint,
        scheme="https" if values["S3_COMPACT_ENDPOINT"].startswith("https") else "http",
    )
    start_ms = int(event_start.timestamp() * 1000)
    end_ms = int(event_end.timestamp() * 1000)
    source_label_to_id: dict[str, int] = {}
    source_labels: list[str] = []
    lanes: dict[tuple[str, str, str], list[_ProfileEvent]] = defaultdict(list)
    object_summaries: list[dict[str, Any]] = []

    partitions = source.list_matching_partitions(dataset_spec)
    _progress(
        "loading s3 compact tables "
        f"partitions={len(partitions)} event_start={_iso(event_start)} event_end={_iso(event_end)}"
    )
    for partition in partitions:
        for key in source.discover_partition_objects(partition):
            if not key.endswith(".parquet"):
                raise ValueError(f"real micro-profile currently requires parquet S3 objects: {key}")
            source_label = f"s3://{source.bucket}/{key}"
            source_label_id = source_label_to_id.setdefault(source_label, len(source_labels))
            if source_label_id == len(source_labels):
                source_labels.append(source_label)
            parquet_path = f"{source.bucket}/{key}"
            parquet_file = pq.ParquetFile(parquet_path, filesystem=filesystem)
            ts_index = parquet_file.schema_arrow.get_field_index("ts_event")
            if ts_index < 0:
                raise ValueError(f"missing ts_event column: {key}")
            _progress(f"loading object {key} row_groups={parquet_file.num_row_groups}")
            row_group_offsets: list[int] = []
            running_offset = 0
            for row_group_index in range(parquet_file.num_row_groups):
                row_group_offsets.append(running_offset)
                running_offset += parquet_file.metadata.row_group(row_group_index).num_rows
            selected_row_groups: list[int] = []
            for row_group_index in range(parquet_file.num_row_groups):
                stats = parquet_file.metadata.row_group(row_group_index).column(ts_index).statistics
                if stats is None or (stats.max >= start_ms and stats.min <= end_ms):
                    selected_row_groups.append(row_group_index)
            object_event_count = 0
            for row_group_index in selected_row_groups:
                if row_group_index == selected_row_groups[0] or row_group_index == selected_row_groups[-1]:
                    _progress(f"reading object {key} row_group={row_group_index}")
                table = parquet_file.read_row_group(row_group_index)
                source_indices = pa.array(
                    range(
                        row_group_offsets[row_group_index],
                        row_group_offsets[row_group_index] + table.num_rows,
                    ),
                    type=pa.int64(),
                )
                table = table.append_column("__source_event_index", source_indices)
                mask = pc.and_(
                    pc.greater_equal(table["ts_event"], start_ms),
                    pc.less_equal(table["ts_event"], end_ms),
                )
                filtered = table.filter(mask)
                object_event_count += filtered.num_rows
                for row in filtered.to_pylist():
                    event_time_ts = float(row["ts_event"]) / 1000.0
                    exchange = str(row["exchange"])
                    symbol = str(row["symbol"]).upper()
                    stream = str(row["stream"])
                    if exchange not in dataset_spec.exchanges:
                        continue
                    if symbol not in dataset_spec.symbols:
                        continue
                    if not dataset_spec.stream_available(exchange, stream):
                        continue
                    fields = {
                        key: value
                        for key, value in row.items()
                        if key
                        not in {
                            "__source_event_index",
                            "event_time",
                            "ts_event",
                            "exchange",
                            "symbol",
                            "stream_type",
                            "stream",
                            "value",
                        }
                    }
                    lanes[(symbol, exchange, stream)].append(
                        _ProfileEvent(
                            event_time_ts=event_time_ts,
                            fields=fields,
                            source_label_id=source_label_id,
                            source_event_index=int(row["__source_event_index"]),
                        )
                    )
            object_summaries.append(
                {
                    "key": key,
                    "selected_row_groups": len(selected_row_groups),
                    "event_count": object_event_count,
                }
            )
            _progress(
                f"loaded object {key} selected_row_groups={len(selected_row_groups)} "
                f"events={object_event_count}"
            )

    indexed = {
        lane_key: _freeze_lane_events(events)
        for lane_key, events in lanes.items()
    }
    event_count = sum(len(events) for events in lanes.values())
    return indexed, source_labels, {
        "partition_count": len(partitions),
        "object_count": len(object_summaries),
        "event_count": event_count,
        "event_start": _iso(event_start),
        "event_end": _iso(event_end),
        "objects": object_summaries,
        "lane_counts": {
            "/".join(lane_key): len(events)
            for lane_key, events in sorted(lanes.items())
        },
    }


def _freeze_lane_events(events: list[_ProfileEvent]) -> tuple[np.ndarray, list[_ProfileEvent]]:
    times_arr = np.asarray([event.event_time_ts for event in events], dtype=np.float64)
    source_label_ids_arr = np.asarray([event.source_label_id for event in events], dtype=np.int64)
    source_event_indices_arr = np.asarray([event.source_event_index for event in events], dtype=np.int64)
    sort_idx = np.lexsort((source_event_indices_arr, source_label_ids_arr, times_arr))
    return times_arr[sort_idx], [events[int(index)] for index in sort_idx]


def _load_s3_compact_tables(
    *,
    source: Any,
    dataset_spec: DatasetSpec,
    event_start: datetime,
    event_end: datetime,
    s3_env_file: Path,
    object_refs: list[dict[str, str]] | None = None,
    object_cache_dir: Path | None = None,
) -> tuple[dict[tuple[str, str, str], Any], list[str], dict[str, Any]]:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.fs as fs
    import pyarrow.parquet as pq

    from quantlab_ml.common import load_env_file

    values = load_env_file(s3_env_file)
    endpoint = values["S3_COMPACT_ENDPOINT"].replace("https://", "").replace("http://", "")
    filesystem = fs.S3FileSystem(
        access_key=values["S3_COMPACT_ACCESS_KEY"],
        secret_key=values["S3_COMPACT_SECRET_KEY"],
        region=values["S3_COMPACT_REGION"],
        endpoint_override=endpoint,
        scheme="https" if values["S3_COMPACT_ENDPOINT"].startswith("https") else "http",
    )
    start_ms = int(event_start.timestamp() * 1000)
    end_ms = int(event_end.timestamp() * 1000)
    source_labels: list[str] = []
    table_chunks_by_lane: defaultdict[tuple[str, str, str], list[Any]] = defaultdict(list)
    object_summaries: list[dict[str, Any]] = []

    if object_refs is None:
        _progress("discovering s3 compact partitions from state")
        partitions = source.list_matching_partitions(dataset_spec)
        object_items = [
            {"partition_id": partition.partition_id, "key": key}
            for partition in partitions
            for key in source.discover_partition_objects(partition)
        ]
    else:
        object_items = object_refs

    partition_count = len({item["partition_id"] for item in object_items})
    _progress(
        "loading s3 compact tables "
        f"partitions={partition_count} objects={len(object_items)} "
        f"event_start={_iso(event_start)} event_end={_iso(event_end)}"
    )
    for item in object_items:
        key = item["key"]
        if not key.endswith(".parquet"):
            raise ValueError(f"real micro-profile currently requires parquet S3 objects: {key}")
        source_label_id = len(source_labels)
        source_labels.append(f"s3://{source.bucket}/{key}")
        parquet_target: str | Path
        parquet_filesystem: fs.S3FileSystem | None
        if object_cache_dir is not None:
            parquet_target = _materialize_s3_object(source=source, key=key, cache_dir=object_cache_dir)
            parquet_filesystem = None
        else:
            parquet_target = f"{source.bucket}/{key}"
            parquet_filesystem = filesystem
        parquet_file = pq.ParquetFile(parquet_target, filesystem=parquet_filesystem)
        ts_index = parquet_file.schema_arrow.get_field_index("ts_event")
        if ts_index < 0:
            raise ValueError(f"missing ts_event column: {key}")
        _progress(f"loading object {key} row_groups={parquet_file.num_row_groups}")
        row_group_offsets: list[int] = []
        running_offset = 0
        for row_group_index in range(parquet_file.num_row_groups):
            row_group_offsets.append(running_offset)
            running_offset += parquet_file.metadata.row_group(row_group_index).num_rows

        selected_row_groups: list[int] = []
        object_event_count = 0
        for row_group_index in range(parquet_file.num_row_groups):
            stats = parquet_file.metadata.row_group(row_group_index).column(ts_index).statistics
            if stats is None or (stats.max >= start_ms and stats.min <= end_ms):
                selected_row_groups.append(row_group_index)
        lane_key: tuple[str, str, str] | None = None
        for row_group_index in selected_row_groups:
            if row_group_index == selected_row_groups[0] or row_group_index == selected_row_groups[-1]:
                _progress(f"reading object {key} row_group={row_group_index}")
            table = parquet_file.read_row_group(row_group_index)
            source_indices = pa.array(
                range(
                    row_group_offsets[row_group_index],
                    row_group_offsets[row_group_index] + table.num_rows,
                ),
                type=pa.int64(),
            )
            source_label_ids = pa.array([source_label_id] * table.num_rows, type=pa.int64())
            table = table.append_column("__source_event_index", source_indices)
            table = table.append_column("__source_label_id", source_label_ids)
            mask = pc.and_(
                pc.greater_equal(table["ts_event"], start_ms),
                pc.less_equal(table["ts_event"], end_ms),
            )
            filtered = table.filter(mask)
            object_event_count += filtered.num_rows
            if filtered.num_rows <= 0:
                continue
            first_row = filtered.slice(0, 1).to_pylist()[0]
            lane_key = (
                str(first_row["symbol"]).upper(),
                str(first_row["exchange"]),
                str(first_row["stream"]),
            )
            table_chunks_by_lane[lane_key].append(filtered)
        object_summaries.append(
            {
                "key": key,
                "selected_row_groups": len(selected_row_groups),
                "event_count": object_event_count,
            }
        )
        _progress(
            f"loaded object {key} selected_row_groups={len(selected_row_groups)} "
            f"events={object_event_count}"
        )

    tables_by_lane: dict[tuple[str, str, str], _WindowedLaneTable] = {}
    unsorted_lanes: list[str] = []
    for lane_key, chunks in table_chunks_by_lane.items():
        table = pa.concat_tables(chunks, promote_options="default")
        ts_ms = np.asarray(table["ts_event"].combine_chunks().to_numpy(zero_copy_only=False), dtype=np.int64)
        if ts_ms.size > 1 and bool(np.any(ts_ms[1:] < ts_ms[:-1])):
            unsorted_lanes.append("/".join(lane_key))
            table = table.sort_by(
                [
                    ("ts_event", "ascending"),
                    ("__source_label_id", "ascending"),
                    ("__source_event_index", "ascending"),
                ]
            )
            ts_ms = np.asarray(table["ts_event"].combine_chunks().to_numpy(zero_copy_only=False), dtype=np.int64)
        tables_by_lane[lane_key] = _WindowedLaneTable(table=table, ts_ms=ts_ms)
    event_count = sum(lane.table.num_rows for lane in tables_by_lane.values())
    return tables_by_lane, source_labels, {
        "partition_count": partition_count,
        "object_count": len(object_summaries),
        "event_count": event_count,
        "event_start": _iso(event_start),
        "event_end": _iso(event_end),
        "objects": object_summaries,
        "lane_sorting": "ts_event_source_order_sorted_for_window_search",
        "lane_sorting_repaired_lanes": unsorted_lanes,
        "lane_counts": {
            "/".join(lane_key): lane.table.num_rows
            for lane_key, lane in sorted(tables_by_lane.items())
        },
    }


def _progress(message: str) -> None:
    print(
        json.dumps(
            {
                "event": "ql033_r6_micro_profile_progress",
                "timestamp": datetime.now(tz=UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
                "message": message,
            },
            sort_keys=True,
        ),
        file=sys.stderr,
        flush=True,
    )


def _load_object_refs(path: Path | None) -> list[dict[str, str]] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    objects = payload.get("objects")
    if not isinstance(objects, list) or not objects:
        raise ValueError(f"object list must contain non-empty objects list: {path}")
    refs: list[dict[str, str]] = []
    for item in objects:
        if not isinstance(item, dict):
            raise ValueError(f"object list item must be an object: {item!r}")
        key = item.get("key")
        partition_id = item.get("partition_id")
        if not isinstance(key, str) or not isinstance(partition_id, str):
            raise ValueError(f"object list item requires key and partition_id strings: {item!r}")
        refs.append({"key": key, "partition_id": partition_id})
    return refs


def _materialize_s3_object(*, source: Any, key: str, cache_dir: Path) -> Path:
    local_path = cache_dir / key
    if local_path.exists():
        return local_path
    local_path.parent.mkdir(parents=True, exist_ok=True)
    _progress(f"materializing s3 object s3://{source.bucket}/{key} -> {local_path}")
    source.client.download_file(source.bucket, key, str(local_path))
    _progress(f"materialized s3 object {key} bytes={local_path.stat().st_size}")
    return local_path


def _indexed_from_tables_for_window(
    *,
    tables_by_lane: dict[tuple[str, str, str], _WindowedLaneTable],
    window_start: datetime,
    decision_time: datetime,
) -> dict[tuple[str, str, str], tuple[np.ndarray, list[_ProfileEvent]]]:
    start_ms = int(window_start.timestamp() * 1000)
    end_ms = int(decision_time.timestamp() * 1000)
    indexed: dict[tuple[str, str, str], tuple[np.ndarray, list[_ProfileEvent]]] = {}
    for lane_key, lane in tables_by_lane.items():
        start_pos = int(np.searchsorted(lane.ts_ms, start_ms, side="left"))
        end_pos = int(np.searchsorted(lane.ts_ms, end_ms, side="right"))
        filtered = lane.table.slice(start_pos, max(end_pos - start_pos, 0))
        events = [
            _ProfileEvent(
                event_time_ts=float(row["ts_event"]) / 1000.0,
                fields={
                    key: value
                    for key, value in row.items()
                    if key
                    not in {
                        "__source_event_index",
                        "__source_label_id",
                        "event_time",
                        "ts_event",
                        "exchange",
                        "symbol",
                        "stream_type",
                        "stream",
                        "value",
                    }
                },
                source_label_id=int(row["__source_label_id"]),
                source_event_index=int(row["__source_event_index"]),
            )
            for row in filtered.to_pylist()
        ]
        indexed[lane_key] = _freeze_lane_events(events)
    return indexed


def _empty_mode_summary(*, rows: list[tuple[str, datetime]]) -> dict[str, Any]:
    return {
        "rows_processed": len(rows),
        "first_decision_timestamp": _iso(rows[0][1]),
        "last_decision_timestamp": _iso(rows[-1][1]),
        "cache_hits": 0,
        "cache_misses": 0,
        "candidate_counters": Counter(),
        "subphase_timings": Counter(),
        "total_selector_wall_sec": 0.0,
        "_per_row_wall_sec": [],
    }


def _append_windowed_rows(
    *,
    mode_summary: dict[str, Any],
    writer: EventTokenCacheSplitWriter,
    computation: Any,
    row_targets: list[str],
    decision_time: datetime,
    first_row_index: int,
) -> None:
    decision_ms = datetime_to_epoch_millis(decision_time)
    writer._window_base_cache[decision_ms] = computation.window_base
    window_base_total = _window_base_total_wall_sec(computation)
    mode_summary["cache_misses"] += 1
    _add_window_base_timings(mode_summary, computation)
    for local_index, target_symbol in enumerate(row_targets):
        row_start = time.perf_counter()
        _, informative_units, selected_units, row_stats = writer._build_window(
            target_symbol=target_symbol,
            decision_time=decision_time,
            row_index=first_row_index + local_index,
        )
        row_wall = time.perf_counter() - row_start
        if local_index == 0:
            row_wall += window_base_total
            mode_summary["total_selector_wall_sec"] += window_base_total
        else:
            mode_summary["cache_hits"] += 1
        mode_summary["_per_row_wall_sec"].append(row_wall)
        mode_summary["total_selector_wall_sec"] += row_stats.total_selector_wall_sec
        _add_row_counters(
            mode_summary=mode_summary,
            row_stats=row_stats,
            informative_units=informative_units,
            selected_units=selected_units,
        )


def _add_window_base_timings(mode_summary: dict[str, Any], computation: Any) -> None:
    window_base = computation.window_base
    timings = mode_summary["subphase_timings"]
    timings["window_base_total_wall_sec"] += _window_base_total_wall_sec(computation)
    timings["lane_range_extraction_wall_sec"] += window_base.lane_range_extraction_wall_sec
    timings["raw_candidate_assembly_wall_sec"] += window_base.raw_candidate_assembly_wall_sec
    timings["deterministic_ordering_wall_sec"] += window_base.deterministic_ordering_wall_sec
    timings["dedupe_wall_sec"] += window_base.dedupe_wall_sec
    timings["bbo_tuple_extraction_wall_sec"] += window_base.bbo_tuple_extraction_wall_sec
    timings["bbo_burst_significance_wall_sec"] += window_base.bbo_burst_significance_wall_sec


def _window_base_total_wall_sec(computation: Any) -> float:
    window_base = computation.window_base
    return (
        window_base.lane_range_extraction_wall_sec
        + window_base.raw_candidate_assembly_wall_sec
        + window_base.deterministic_ordering_wall_sec
        + window_base.dedupe_wall_sec
        + window_base.bbo_significance_wall_sec
    )


def _add_row_counters(
    *,
    mode_summary: dict[str, Any],
    row_stats: Any,
    informative_units: list[_InformativeUnit],
    selected_units: list[_InformativeUnit],
) -> None:
    aggregate = mode_summary["candidate_counters"]
    aggregate["raw_candidates"] += row_stats.candidate_token_count
    aggregate["informative_units"] += row_stats.informative_candidate_count
    aggregate["selected_tokens"] += row_stats.selected_token_count
    aggregate["duplicate_events"] += row_stats.duplicate_event_count
    aggregate["same_timestamp_ties"] += row_stats.same_timestamp_tie_count
    aggregate["source_order_inversions"] += row_stats.source_order_inversion_count
    aggregate["t4_candidates"] += row_stats.t4_candidate_count
    aggregate["t4_anchors"] += row_stats.t4_anchor_count
    aggregate["informative_bbo_units"] += sum(1 for unit in informative_units if unit.stream == "bbo")
    aggregate["selected_bbo_units"] += sum(1 for unit in selected_units if unit.stream == "bbo")
    mode_summary["subphase_timings"]["t4_resolution_wall_sec"] += row_stats.t4_resolution_wall_sec
    mode_summary["subphase_timings"]["quota_fill_wall_sec"] += row_stats.quota_fill_wall_sec


def _finalize_mode_summary(mode_summary: dict[str, Any]) -> None:
    mode_summary["candidate_counters"] = dict(sorted(mode_summary["candidate_counters"].items()))
    mode_summary["subphase_timings"] = {
        key: float(value)
        for key, value in sorted(mode_summary["subphase_timings"].items())
    }
    mode_summary["per_row_timing_sec"] = _percentiles(mode_summary.pop("_per_row_wall_sec"))


def _build_projection(*, window_base_speedup: float | None) -> dict[str, Any]:
    r4_profile = _load_json_if_present(R4_PARTIAL_PROFILE_PATH)
    r4_build_time = _load_json_if_present(R4_BUILD_TIME_PATH)
    if window_base_speedup is None or not r4_profile or not r4_build_time:
        return {
            "classification": "insufficient_evidence",
            "projected_build_time_multiplier": None,
            "reason": "projection requires real speedup plus R4 partial profile/build timing",
        }
    r4_wall = float(r4_build_time["real_seconds"])
    r4_window_base = float(r4_profile["window_base_precompute_wall_sec"])
    projected_saved = r4_window_base * (1.0 - (1.0 / max(window_base_speedup, 1e-12)))
    projected_wall = max(r4_wall - projected_saved, 0.0)
    projected_multiplier = projected_wall / 980.67
    safely_under_gate = projected_wall <= (R4_BUILD_GATE_SECONDS * 0.95)
    return {
        "classification": "projected_clear" if safely_under_gate else "marginal_or_insufficient",
        "window_base_speedup": window_base_speedup,
        "r4_reference_wall_sec": r4_wall,
        "r4_window_base_precompute_wall_sec": r4_window_base,
        "projected_saved_wall_sec": projected_saved,
        "projected_wall_sec": projected_wall,
        "projected_build_time_multiplier": projected_multiplier,
        "build_gate_seconds": R4_BUILD_GATE_SECONDS,
        "safely_under_2_25x_gate": safely_under_gate,
        "assumptions": [
            "R4 partial profile is used only for blocker projection, not acceptance",
            "R5 19-row SIGTERM run is not used as pass evidence",
            "projected saving applies only to window_base precompute share observed in R4",
        ],
    }


def _load_json_if_present(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _run_mode(
    writer: EventTokenCacheSplitWriter,
    *,
    rows: list[tuple[str, datetime]],
) -> dict[str, Any]:
    aggregate = Counter()
    per_row_wall_sec: list[float] = []
    subphase = Counter()
    seen_decision_ms: set[int] = set()
    selector_wall = 0.0

    for row_index, (target_symbol, decision_time) in enumerate(rows):
        decision_ms = datetime_to_epoch_millis(decision_time)
        cache_hit = decision_ms in writer._window_base_cache
        row_start = time.perf_counter()
        _, informative_units, selected_units, row_stats = writer._build_window(
            target_symbol=target_symbol,
            decision_time=decision_time,
            row_index=row_index,
        )
        per_row_wall_sec.append(time.perf_counter() - row_start)
        selector_wall += row_stats.total_selector_wall_sec
        aggregate["raw_candidates"] += row_stats.candidate_token_count
        aggregate["informative_units"] += row_stats.informative_candidate_count
        aggregate["selected_tokens"] += row_stats.selected_token_count
        aggregate["duplicate_events"] += row_stats.duplicate_event_count
        aggregate["same_timestamp_ties"] += row_stats.same_timestamp_tie_count
        aggregate["source_order_inversions"] += row_stats.source_order_inversion_count
        aggregate["t4_candidates"] += row_stats.t4_candidate_count
        aggregate["t4_anchors"] += row_stats.t4_anchor_count
        aggregate["informative_bbo_units"] += sum(1 for unit in informative_units if unit.stream == "bbo")
        aggregate["selected_bbo_units"] += sum(1 for unit in selected_units if unit.stream == "bbo")
        subphase["t4_resolution_wall_sec"] += row_stats.t4_resolution_wall_sec
        subphase["quota_fill_wall_sec"] += row_stats.quota_fill_wall_sec

        if not cache_hit and decision_ms not in seen_decision_ms:
            window_base = writer._window_base_cache[decision_ms]
            subphase["window_base_total_wall_sec"] += (
                window_base.lane_range_extraction_wall_sec
                + window_base.raw_candidate_assembly_wall_sec
                + window_base.deterministic_ordering_wall_sec
                + window_base.dedupe_wall_sec
                + window_base.bbo_significance_wall_sec
            )
            subphase["lane_range_extraction_wall_sec"] += window_base.lane_range_extraction_wall_sec
            subphase["raw_candidate_assembly_wall_sec"] += window_base.raw_candidate_assembly_wall_sec
            subphase["deterministic_ordering_wall_sec"] += window_base.deterministic_ordering_wall_sec
            subphase["dedupe_wall_sec"] += window_base.dedupe_wall_sec
            subphase["bbo_tuple_extraction_wall_sec"] += window_base.bbo_tuple_extraction_wall_sec
            subphase["bbo_burst_significance_wall_sec"] += window_base.bbo_burst_significance_wall_sec
            seen_decision_ms.add(decision_ms)

    return {
        "rows_processed": len(rows),
        "first_decision_timestamp": _iso(rows[0][1]),
        "last_decision_timestamp": _iso(rows[-1][1]),
        "cache_hits": writer._window_base_cache_hit_count,
        "cache_misses": writer._window_base_cache_miss_count,
        "candidate_counters": dict(sorted(aggregate.items())),
        "subphase_timings": {key: float(value) for key, value in sorted(subphase.items())},
        "total_selector_wall_sec": selector_wall,
        "per_row_timing_sec": _percentiles(per_row_wall_sec),
    }


def _compare_reference_candidate(
    *,
    reference_writer: EventTokenCacheSplitWriter,
    candidate_writer: EventTokenCacheSplitWriter,
    decision_times: list[datetime],
) -> dict[str, Any]:
    mismatches: list[str] = []
    for decision_time in decision_times:
        reference = reference_writer._compute_window_base_slow_reference(decision_time=decision_time)
        candidate = candidate_writer._compute_window_base_optimized(decision_time=decision_time)
        if [_candidate_snapshot(item) for item in reference.raw_candidates] != [
            _candidate_snapshot(item) for item in candidate.raw_candidates
        ]:
            mismatches.append(f"{_iso(decision_time)}:raw_candidates")
        if [_candidate_snapshot(item) for item in reference.deduped_candidates] != [
            _candidate_snapshot(item) for item in candidate.deduped_candidates
        ]:
            mismatches.append(f"{_iso(decision_time)}:deduped_candidates")
        if _window_base_snapshot(reference.window_base) != _window_base_snapshot(candidate.window_base):
            mismatches.append(f"{_iso(decision_time)}:window_base")
        if mismatches:
            break
    return {
        "ordered_reference_candidate_match": not mismatches,
        "windows_compared": len(decision_times),
        "mismatches": mismatches,
    }


def _dataset_spec(*, start: datetime, minute_count: int) -> DatasetSpec:
    end = start + timedelta(minutes=minute_count - 1)
    return DatasetSpec.model_validate(
        {
            "dataset_hash": "ql033-r6-synthetic-profile",
            "slice_id": "ql033-r6-synthetic-profile-20260125-1600",
            "exchanges": list(EXCHANGES),
            "symbols": list(SYMBOLS),
            "stream_universe": list(STREAMS),
            "available_streams_by_exchange": {
                exchange: list(STREAMS)
                for exchange in EXCHANGES
            },
            "train_range": {"start": _iso(start), "end": _iso(end)},
            "validation_range": {
                "start": _iso(end + timedelta(minutes=1)),
                "end": _iso(end + timedelta(minutes=2)),
            },
            "final_untouched_test_range": {
                "start": _iso(end + timedelta(minutes=3)),
                "end": _iso(end + timedelta(minutes=4)),
            },
            "walkforward": {
                "train_window_steps": 240,
                "validation_window_steps": 120,
                "step_size_steps": 120,
            },
            "sampling_interval_seconds": 60,
        }
    )


def _synthetic_indexed_events(
    *,
    start: datetime,
    minute_count: int,
) -> dict[tuple[str, str, str], tuple[np.ndarray, list[_ProfileEvent]]]:
    lanes: dict[tuple[str, str, str], list[_ProfileEvent]] = {
        (symbol, exchange, stream): []
        for symbol in SYMBOLS
        for exchange in EXCHANGES
        for stream in STREAMS
    }
    source_event_index = 0
    symbol_base = {"BTCUSDT": 100.0, "ETHUSDT": 50.0, "SOLUSDT": 20.0}
    venue_shift = {"binance": 0.0, "bybit": 0.2, "okx": -0.2}
    event_start = start - timedelta(minutes=1)
    for minute in range(minute_count):
        minute_start = event_start + timedelta(minutes=minute)
        for exchange in EXCHANGES:
            for symbol in SYMBOLS:
                base = symbol_base[symbol] + venue_shift[exchange] + (minute * 0.01)
                for tick in range(10):
                    event_time = minute_start + timedelta(milliseconds=100 + (tick * 35))
                    if tick in {3, 4}:
                        event_time = minute_start + timedelta(milliseconds=210)
                    bbo_fields = _bbo_fields(
                        mid=base + (tick * 0.02),
                        spread=0.10 if tick < 6 else 0.22,
                        bid_size=12.0 if tick < 5 else 2.0,
                        ask_size=8.0 if tick % 2 == 0 else 14.0,
                    )
                    lanes[(symbol, exchange, "bbo")].append(
                        _ProfileEvent(
                            event_time_ts=event_time.timestamp(),
                            fields=bbo_fields,
                            source_label_id=0,
                            source_event_index=source_event_index,
                        )
                    )
                    source_event_index += 1
                    trade_fields = {
                        "price": base + (tick * 0.015),
                        "qty": 1.0 + (tick % 3),
                        "side_or_signed_flow_proxy": 1.0 if tick % 2 == 0 else -1.0,
                        "event_delta": 0.015,
                        "count_or_burst": tick + 1,
                    }
                    lanes[(symbol, exchange, "trade")].append(
                        _ProfileEvent(
                            event_time_ts=event_time.timestamp(),
                            fields=trade_fields,
                            source_label_id=0,
                            source_event_index=source_event_index,
                        )
                    )
                    source_event_index += 1

    return {
        lane_key: (np.asarray([event.event_time_ts for event in events], dtype=float), events)
        for lane_key, events in lanes.items()
    }


def _bbo_fields(
    *,
    mid: float,
    spread: float,
    bid_size: float,
    ask_size: float,
) -> dict[str, object]:
    return {
        "bid_price": mid - (spread / 2.0),
        "ask_price": mid + (spread / 2.0),
        "bid_size": bid_size,
        "ask_size": ask_size,
        "spread": spread,
        "mid": mid,
        "imbalance_inputs": (bid_size - ask_size) / (bid_size + ask_size),
    }


def _candidate_snapshot(
    candidate: _EventCandidate,
) -> tuple[int, str, str, str, int, int, tuple[tuple[str, tuple[str, object]], ...]]:
    return (
        int(candidate.event_time_ts * 1000),
        candidate.exchange,
        candidate.symbol,
        candidate.stream,
        candidate.source_label_id,
        candidate.source_event_index,
        _canonical_payload_identity(candidate.event.fields),
    )


def _window_base_snapshot(window_base) -> dict[str, object]:
    return {
        "units": [
            (
                _candidate_snapshot(unit.candidate),
                unit.source_bucket,
                unit.lane_key,
                unit.burst_id,
                round(unit.salience, 12),
                sorted(unit.emission_tags),
                sorted(unit.matched_reasons),
                unit.canonical_significance_reason,
                _latest_reference_identity(unit),
            )
            for unit in window_base.informative_units
        ],
        "supported_lane_count": window_base.supported_lane_count,
        "latest_reference_by_symbol": dict(window_base.latest_reference_by_symbol),
        "deduped_count": window_base.deduped_count,
        "duplicate_count": window_base.duplicate_count,
        "duplicate_dropped_by_stream": dict(window_base.duplicate_dropped_by_stream),
        "duplicate_dropped_by_venue": dict(window_base.duplicate_dropped_by_venue),
        "same_timestamp_tie_count": window_base.same_timestamp_tie_count,
        "source_order_inversion_count": window_base.source_order_inversion_count,
        "candidate_by_symbol": dict(window_base.candidate_by_symbol),
        "candidate_by_venue": dict(window_base.candidate_by_venue),
        "candidate_by_stream": dict(window_base.candidate_by_stream),
    }


def _latest_reference_identity(unit: _InformativeUnit) -> tuple[object, ...] | None:
    if unit.best_anchor_key is None:
        return None
    return (
        unit.best_anchor_key,
        unit.best_anchor_tier,
        unit.best_anchor_delta_ms,
    )


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())

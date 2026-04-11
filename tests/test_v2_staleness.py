"""test_v2_staleness.py

funding ve OI için missing=False, stale=True durumunu doğrular.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from quantlab_ml.contracts import DatasetSpec, TrajectoryBundle
from quantlab_ml.data import LocalFixtureSource
from quantlab_ml.trajectories import TrajectoryBuilder


def _build_bundle_from_lines(
    lines: list[str],
    tmp_path: Path,
    dataset_spec: DatasetSpec,
    training_bundle,
    reward_spec,
) -> TrajectoryBundle:
    path = tmp_path / "stale_events.ndjson"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    trajectory_spec, action_space, _ = training_bundle
    events = LocalFixtureSource(path).load_events(dataset_spec)
    return TrajectoryBuilder(dataset_spec, trajectory_spec, action_space, reward_spec).build(events)


def _find_stream_coords(schema, tensor, exchange: str, stream: str) -> list[int]:
    """Belirli (exchange, stream) için tüm non-padding koordinatları döner."""
    n_t = schema.scale_axis[0].num_buckets
    n_sym = len(schema.asset_axis)
    n_exc = len(schema.exchange_axis)
    n_str = len(schema.stream_axis)
    field_counts = [len(schema.field_axis.get(s, [])) for s in schema.stream_axis]
    total_fields = sum(field_counts)

    exc_idx = schema.exchange_axis.index(exchange)
    str_idx = schema.stream_axis.index(stream)
    f_offset = sum(field_counts[:str_idx])
    n_fields = field_counts[str_idx]

    result = []
    for t in range(n_t):
        for sym_idx in range(n_sym):
            base = (
                t * n_sym * n_exc * n_str * total_fields
                + sym_idx * n_exc * n_str * total_fields
                + exc_idx * n_str * total_fields
                + str_idx * total_fields
                + f_offset
            )
            for fi in range(n_fields):
                result.append(base + fi)
    return result


def test_stale_funding_missing_false_stale_true(
    tmp_path: Path,
    fixture_path: Path,
    dataset_spec: DatasetSpec,
    training_bundle,
    reward_spec,
) -> None:
    """funding büyük gap sonrası stale=True, missing=False olmalı."""
    all_lines = fixture_path.read_text(encoding="utf-8").splitlines()

    # binance BTCUSDT funding event'lerini t=00:01 .. 00:06 arasından çıkar
    gap_times = {"00:01:00Z", "00:02:00Z", "00:03:00Z", "00:04:00Z", "00:05:00Z", "00:06:00Z"}
    gap_lines = [
        line for line in all_lines
        if not (
            '"BTCUSDT"' in line
            and '"funding"' in line
            and '"binance"' in line
            and any(t in line for t in gap_times)
        )
    ]

    bundle = _build_bundle_from_lines(gap_lines, tmp_path, dataset_spec, training_bundle, reward_spec)
    first_final_test_step = bundle.splits["final_untouched_test"][0].steps[-1]
    schema = first_final_test_step.observation.observation_schema
    tensor = first_final_test_step.observation.raw_surface["1m"]
    coords = _find_stream_coords(schema, tensor, "binance", "funding")

    stale_found = False
    for idx in coords:
        if tensor.padding[idx] or tensor.unavailable_by_contract[idx]:
            continue
        if tensor.stale[idx]:
            assert tensor.missing[idx] is False, "stale coord must not be missing"
            stale_found = True
            break

    assert stale_found, "Expected at least one stale funding coordinate after large gap"

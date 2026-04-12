from __future__ import annotations

# _coerce_event_time typing fix: the function now eagerly parses str→datetime instead of
# returning datetime|str. The tests below verify the accepted input surface is preserved.

from pathlib import Path

from quantlab_ml.common import load_yaml
from quantlab_ml.contracts import DatasetSpec
from quantlab_ml.data import LocalFixtureSource


def test_default_and_fixture_profiles_are_distinct(repo_root: Path) -> None:
    default_spec = DatasetSpec.model_validate(load_yaml(repo_root / "configs" / "data" / "default.yaml")["dataset"])
    fixture_spec = DatasetSpec.model_validate(load_yaml(repo_root / "configs" / "data" / "fixture.yaml")["dataset"])
    s3_current_spec = DatasetSpec.model_validate(
        load_yaml(repo_root / "configs" / "data" / "s3-current.yaml")["dataset"]
    )

    assert len(default_spec.symbols) == 10
    assert fixture_spec.symbols == ["BTCUSDT", "ETHUSDT"]
    assert s3_current_spec.symbols == default_spec.symbols
    assert default_spec.stream_available("binance", "open_interest") is False
    assert default_spec.stream_available("bybit", "open_interest") is True
    assert s3_current_spec.train_range.start.year == 2026
    assert s3_current_spec.validation_range.start.year == 2026
    assert s3_current_spec.final_untouched_test_range.end.year == 2026


def test_adapter_rejects_binance_open_interest(tmp_path: Path, repo_root: Path) -> None:
    spec = DatasetSpec.model_validate(load_yaml(repo_root / "configs" / "data" / "default.yaml")["dataset"])
    path = tmp_path / "events.ndjson"
    path.write_text(
        "\n".join(
            [
                '{"event_time":"2024-01-01T00:00:00Z","exchange":"binance","symbol":"BTCUSDT","stream_type":"open_interest","open_interest":1000}',
                '{"event_time":"2024-01-01T00:00:00Z","exchange":"bybit","symbol":"BTCUSDT","stream_type":"open_interest","open_interest":1001}',
                '{"event_time":"2024-01-01T00:00:00Z","exchange":"binance","symbol":"BTCUSDT","stream_type":"mark_price","price":100.0}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    events = LocalFixtureSource(path).load_events(spec)

    assert ("binance", "open_interest") not in {(event.exchange, event.stream_type) for event in events}
    assert ("bybit", "open_interest") in {(event.exchange, event.stream_type) for event in events}
    assert spec.walkforward.train_window_steps >= 1


# ---------------------------------------------------------------------------
# _coerce_event_time input-surface tests
# Verify typing fix (str→datetime) preserves accepted input surface.
# ---------------------------------------------------------------------------


def test_coerce_event_time_datetime_passthrough(tmp_path: Path, repo_root: Path) -> None:
    """event_time already a datetime object is returned as-is."""
    from datetime import UTC, datetime

    from quantlab_ml.data.adapters import _coerce_event_time

    dt = datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)
    result = _coerce_event_time({"event_time": dt})
    assert isinstance(result, datetime)
    assert result == dt


def test_coerce_event_time_iso_string_with_z(tmp_path: Path, repo_root: Path) -> None:
    """event_time as ISO-8601 string with trailing Z is parsed to datetime."""
    from datetime import UTC, datetime

    from quantlab_ml.data.adapters import _coerce_event_time

    result = _coerce_event_time({"event_time": "2024-06-01T12:00:00Z"})
    assert isinstance(result, datetime)
    assert result == datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)


def test_coerce_event_time_iso_string_with_offset(tmp_path: Path, repo_root: Path) -> None:
    """event_time as ISO-8601 string with explicit +00:00 offset is parsed to datetime."""
    from datetime import datetime

    from quantlab_ml.data.adapters import _coerce_event_time

    result = _coerce_event_time({"event_time": "2024-06-01T12:00:00+00:00"})
    assert isinstance(result, datetime)
    assert result.year == 2024


def test_coerce_event_time_ts_event_milliseconds(tmp_path: Path, repo_root: Path) -> None:
    """ts_event numeric millisecond epoch is converted to datetime."""
    from datetime import datetime

    from quantlab_ml.data.adapters import _coerce_event_time

    # 1717243200000ms = 2024-06-01T12:00:00Z
    ts_ms = 1717243200000
    result = _coerce_event_time({"ts_event": ts_ms})
    assert isinstance(result, datetime)
    assert result.year == 2024

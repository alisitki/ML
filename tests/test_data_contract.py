from __future__ import annotations

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

from __future__ import annotations

from io import BytesIO
import importlib
import json
from pathlib import Path

from typer.testing import CliRunner

from quantlab_ml.cli.app import app
from quantlab_ml.contracts import DatasetSpec
from quantlab_ml.data import S3CompactedSource

cli_module = importlib.import_module("quantlab_ml.cli.app")


class FakeS3Client:
    def __init__(self, objects: dict[str, bytes]):
        self.objects = objects

    def get_object(self, Bucket: str, Key: str) -> dict[str, BytesIO]:
        if Key not in self.objects:
            raise KeyError(Key)
        return {"Body": BytesIO(self.objects[Key])}

    def head_object(self, Bucket: str, Key: str) -> dict[str, str]:
        if Key not in self.objects:
            raise KeyError(Key)
        return {"ETag": "fake"}

    def list_objects_v2(self, Bucket: str, Prefix: str, MaxKeys: int) -> dict[str, list[dict[str, str]]]:
        contents = [{"Key": key} for key in sorted(self.objects) if key.startswith(Prefix)][:MaxKeys]
        return {"Contents": contents}


def test_s3_compacted_source_from_env_file(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "S3_COMPACT_ENDPOINT=https://example.invalid",
                "S3_COMPACT_BUCKET=quantlab-compact",
                "S3_COMPACT_ACCESS_KEY=test-access",
                "S3_COMPACT_SECRET_KEY=test-secret",
                "S3_COMPACT_REGION=us-east-1",
                "S3_COMPACT_STATE_KEY=compacted/_state.json",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    source = S3CompactedSource.from_env_file(env_path)

    assert source.bucket == "quantlab-compact"
    assert source.endpoint_url == "https://example.invalid"
    assert source.state_key == "compacted/_state.json"
    assert source.data_prefix == "compacted"


def test_s3_compacted_source_discovers_root_partition_layout() -> None:
    state = {
        "days": {"20260125": {"status": "success", "updated_at": "2026-01-25T00:00:00Z"}},
        "last_compacted_date": "20260125",
        "partitions": {
            "binance/mark_price/btcusdt/20260125": {
                "status": "success",
                "rows": 2,
                "total_size_bytes": 128,
                "updated_at": "2026-01-25T00:00:00Z",
                "day_quality_post": "ok",
                "post_filter_version": "v1",
            },
            "binance/open_interest/btcusdt/20260125": {
                "status": "success",
                "rows": 1,
                "total_size_bytes": 64,
                "updated_at": "2026-01-25T00:00:00Z",
                "day_quality_post": "ok",
                "post_filter_version": "v1",
            },
        },
        "updated_at": "2026-01-25T00:00:00Z",
    }
    data_rows = "\n".join(
        [
            '{"ts_event":1769299200000,"exchange":"binance","symbol":"BTCUSDT","stream":"mark_price","mark_price":100.0,"index_price":100.2}',
            '{"ts_event":1769299260000,"exchange":"binance","symbol":"BTCUSDT","stream":"mark_price","mark_price":101.0,"index_price":101.1}',
        ]
    )
    client = FakeS3Client(
        {
            "compacted/_state.json": json.dumps(state).encode("utf-8"),
            "exchange=binance/stream=mark_price/symbol=btcusdt/date=20260125/data.ndjson": data_rows.encode("utf-8"),
            "exchange=binance/stream=mark_price/symbol=btcusdt/date=20260125/meta.json": b'{"rows":2}',
        }
    )
    source = S3CompactedSource(
        endpoint_url="https://example.invalid",
        bucket="quantlab-compact",
        access_key_id="test-access",
        secret_access_key="test-secret",
        region_name="us-east-1",
        client=client,
    )
    dataset_spec = DatasetSpec.model_validate(
        {
            "dataset_hash": "s3-test",
            "slice_id": "s3-test-slice",
            "exchanges": ["binance"],
            "symbols": ["BTCUSDT"],
            "stream_universe": ["mark_price", "open_interest"],
            "available_streams_by_exchange": {"binance": ["mark_price"]},
            "train_range": {"start": "2026-01-25T00:00:00Z", "end": "2026-01-25T00:01:00Z"},
            "validation_range": {"start": "2026-01-25T00:02:00Z", "end": "2026-01-25T00:03:00Z"},
            "final_untouched_test_range": {"start": "2026-01-25T00:04:00Z", "end": "2026-01-25T00:05:00Z"},
            "walkforward": {"train_window_steps": 2, "validation_window_steps": 2, "step_size_steps": 1},
            "sampling_interval_seconds": 60,
        }
    )

    discovered = source.discover_partition_objects("binance/mark_price/btcusdt/20260125")
    events = source.load_events(dataset_spec)
    summary = source.summarize_state(dataset_spec)

    assert discovered == ["exchange=binance/stream=mark_price/symbol=btcusdt/date=20260125/data.ndjson"]
    event_list = list(events)
    assert len(event_list) == 2
    assert {event.stream_type for event in event_list} == {"mark_price"}
    assert summary["matched_partition_count"] == 1
    assert summary["storage_layout_hint"].startswith("exchange=<exchange>/stream=<stream>")
    assert summary["successful_day_count"] == 1


def test_inspect_s3_compact_cli(monkeypatch, tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("S3_COMPACT_STATE_KEY=compacted/_state.json\n", encoding="utf-8")

    class DummySource:
        def summarize_state(self, dataset_spec=None) -> dict[str, object]:
            return {
                "bucket": "quantlab-compact",
                "state_key": "compacted/_state.json",
                "matched_partition_count": 3,
            }

    class DummyFactory:
        @classmethod
        def from_env_file(cls, path: Path) -> DummySource:
            assert path == env_path
            return DummySource()

    monkeypatch.setattr(cli_module, "S3CompactedSource", DummyFactory)
    runner = CliRunner()

    result = runner.invoke(app, ["inspect-s3-compact", "--env-file", str(env_path)])

    assert result.exit_code == 0, result.stdout
    assert '"matched_partition_count": 3' in result.stdout


def test_s3_load_events_is_streaming() -> None:
    """load_events must return an Iterator (generator), not a materialised list."""
    import types

    state = {
        "days": {"20260125": {"status": "success", "updated_at": "2026-01-25T00:00:00Z"}},
        "last_compacted_date": "20260125",
        "partitions": {
            "binance/mark_price/btcusdt/20260125": {
                "status": "success",
                "rows": 1,
                "total_size_bytes": 64,
                "updated_at": "2026-01-25T00:00:00Z",
            },
        },
        "updated_at": "2026-01-25T00:00:00Z",
    }
    data_row = '{"ts_event":1769299200000,"exchange":"binance","symbol":"BTCUSDT","stream":"mark_price","mark_price":100.0,"index_price":100.2}\n'
    client = FakeS3Client(
        {
            "compacted/_state.json": json.dumps(state).encode("utf-8"),
            "exchange=binance/stream=mark_price/symbol=btcusdt/date=20260125/data.ndjson": data_row.encode("utf-8"),
        }
    )
    source = S3CompactedSource(
        endpoint_url="https://example.invalid",
        bucket="quantlab-compact",
        access_key_id="test-access",
        secret_access_key="test-secret",
        region_name="us-east-1",
        client=client,
    )
    dataset_spec = DatasetSpec.model_validate(
        {
            "dataset_hash": "streaming-test",
            "slice_id": "streaming-test-slice",
            "exchanges": ["binance"],
            "symbols": ["BTCUSDT"],
            "stream_universe": ["mark_price"],
            "available_streams_by_exchange": {"binance": ["mark_price"]},
            "train_range": {"start": "2026-01-25T00:00:00Z", "end": "2026-01-25T00:01:00Z"},
            "validation_range": {"start": "2026-01-25T00:02:00Z", "end": "2026-01-25T00:03:00Z"},
            "final_untouched_test_range": {"start": "2026-01-25T00:04:00Z", "end": "2026-01-25T00:05:00Z"},
            "walkforward": {"train_window_steps": 2, "validation_window_steps": 2, "step_size_steps": 1},
            "sampling_interval_seconds": 60,
        }
    )

    result = source.load_events(dataset_spec)

    # Must be a generator (Iterator), not a materialised list.
    assert isinstance(result, types.GeneratorType), (
        f"expected GeneratorType, got {type(result).__name__}"
    )
    # Consuming it must still yield the correct events.
    consumed = list(result)
    assert len(consumed) == 1
    assert consumed[0].stream_type == "mark_price"


def test_index_events_accepts_plain_iterator() -> None:
    """_index_events must work with a plain iterator, not just a list."""
    from datetime import UTC, datetime
    from pathlib import Path

    from quantlab_ml.common import load_yaml
    from quantlab_ml.contracts import (
        ActionSpaceSpec,
        DatasetSpec,
        NormalizedMarketEvent,
        RewardEventSpec,
        TrajectorySpec,
    )
    from quantlab_ml.trajectories.builder import TrajectoryBuilder

    repo_root = Path(__file__).parent.parent
    ds_cfg = load_yaml(repo_root / "configs" / "data" / "fixture.yaml")["dataset"]
    tr_cfg = load_yaml(repo_root / "configs" / "training" / "default.yaml")
    rw_cfg = load_yaml(repo_root / "configs" / "reward" / "default.yaml")["reward"]

    dataset_spec = DatasetSpec.model_validate(ds_cfg)
    trajectory_spec = TrajectorySpec.model_validate(tr_cfg["trajectory"])
    action_space = ActionSpaceSpec.model_validate(tr_cfg["action_space"])
    reward_spec = RewardEventSpec.model_validate(rw_cfg)

    events: list[NormalizedMarketEvent] = [
        NormalizedMarketEvent(
            event_time=datetime(2026, 1, 25, 0, 0, tzinfo=UTC),
            exchange="binance",
            symbol="BTCUSDT",
            stream_type="mark_price",
            value=100.0,
            fields={"mark_price": 100.0},
            ingest_metadata={},
        ),
        NormalizedMarketEvent(
            event_time=datetime(2026, 1, 25, 0, 1, tzinfo=UTC),
            exchange="binance",
            symbol="BTCUSDT",
            stream_type="mark_price",
            value=101.0,
            fields={"mark_price": 101.0},
            ingest_metadata={},
        ),
    ]

    builder = TrajectoryBuilder(dataset_spec, trajectory_spec, action_space, reward_spec)

    # Pass a plain iterator (not a list) — must not raise and must index correctly.
    indexed, count, min_time = builder._index_events(iter(events))

    assert count == 2
    assert min_time == datetime(2026, 1, 25, 0, 0, tzinfo=UTC)
    key = ("BTCUSDT", "binance", "mark_price")
    assert key in indexed
    assert len(indexed[key]) == 2

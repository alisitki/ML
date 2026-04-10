from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from io import BytesIO
import json
from pathlib import Path
from typing import Any

from quantlab_ml.common import load_env_file
from quantlab_ml.contracts import DatasetSpec, NormalizedMarketEvent
from quantlab_ml.data.interfaces import MarketDataSource


class LocalFixtureSource(MarketDataSource):
    def __init__(self, path: Path):
        self.path = path

    def load_events(self, dataset_spec: DatasetSpec) -> list[NormalizedMarketEvent]:
        events: list[NormalizedMarketEvent] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            raw = json.loads(line)
            event = _normalize_record(raw, source_label=str(self.path))
            if event.exchange in dataset_spec.exchanges and event.symbol in dataset_spec.symbols:
                if dataset_spec.stream_available(event.exchange, event.stream_type):
                    events.append(event)
        return sorted(events, key=lambda item: item.event_time)


class LocalParquetSource(MarketDataSource):
    def __init__(self, path: Path):
        self.path = path

    def load_events(self, dataset_spec: DatasetSpec) -> list[NormalizedMarketEvent]:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover - optional dependency boundary
            raise RuntimeError("pyarrow is required for parquet adapter support") from exc

        files = list(self.path.glob("*.parquet")) if self.path.is_dir() else [self.path]
        events: list[NormalizedMarketEvent] = []
        for file_path in files:
            table = pq.read_table(file_path)
            for raw in table.to_pylist():
                event = _normalize_record(raw, source_label=str(file_path))
                if event.exchange in dataset_spec.exchanges and event.symbol in dataset_spec.symbols:
                    if dataset_spec.stream_available(event.exchange, event.stream_type):
                        events.append(event)
        return sorted(events, key=lambda item: item.event_time)


class S3CompactedSource(MarketDataSource):
    def __init__(
        self,
        endpoint_url: str,
        bucket: str,
        access_key_id: str,
        secret_access_key: str,
        region_name: str,
        state_key: str = "compacted/_state.json",
        data_prefix: str = "compacted",
        client: Any | None = None,
    ):
        self.endpoint_url = endpoint_url
        self.bucket = bucket
        self.access_key_id = access_key_id
        self.secret_access_key = secret_access_key
        self.region_name = region_name
        self.state_key = state_key
        self.data_prefix = data_prefix.rstrip("/")
        self._client = client
        self._state_cache: dict[str, Any] | None = None

    @classmethod
    def from_env_file(cls, path: Path) -> "S3CompactedSource":
        values = load_env_file(path)
        required = [
            "S3_COMPACT_ENDPOINT",
            "S3_COMPACT_BUCKET",
            "S3_COMPACT_ACCESS_KEY",
            "S3_COMPACT_SECRET_KEY",
            "S3_COMPACT_REGION",
            "S3_COMPACT_STATE_KEY",
        ]
        missing = [key for key in required if key not in values]
        if missing:
            raise ValueError(f"missing S3 compact env vars: {', '.join(sorted(missing))}")
        state_key = values["S3_COMPACT_STATE_KEY"]
        data_prefix = state_key.rsplit("/", 1)[0] if "/" in state_key else "compacted"
        return cls(
            endpoint_url=values["S3_COMPACT_ENDPOINT"],
            bucket=values["S3_COMPACT_BUCKET"],
            access_key_id=values["S3_COMPACT_ACCESS_KEY"],
            secret_access_key=values["S3_COMPACT_SECRET_KEY"],
            region_name=values["S3_COMPACT_REGION"],
            state_key=state_key,
            data_prefix=data_prefix,
        )

    def load_events(self, dataset_spec: DatasetSpec) -> list[NormalizedMarketEvent]:
        matching_partitions = self.list_matching_partitions(dataset_spec)
        object_keys: list[str] = []
        missing_partitions: list[str] = []
        for partition in matching_partitions:
            discovered = self.discover_partition_objects(partition)
            if not discovered:
                missing_partitions.append(partition.partition_id)
                continue
            object_keys.extend(discovered)

        if not object_keys:
            visible = self.list_visible_keys(prefix=f"{self.data_prefix}/", max_keys=25)
            sample_prefixes = [self._partition_storage_prefix(ref) for ref in matching_partitions[:5]]
            raise RuntimeError(
                "matched S3 compact partitions from state metadata but could not discover any "
                f"readable data objects in bucket '{self.bucket}'. matched_partitions={len(matching_partitions)} "
                f"missing_partition_objects={len(missing_partitions)} visible_keys_sample={visible} "
                f"partition_prefix_sample={sample_prefixes}"
            )

        events: list[NormalizedMarketEvent] = []
        for key in sorted(set(object_keys)):
            events.extend(self._read_object_events(key, dataset_spec))
        return sorted(events, key=lambda item: item.event_time)

    def load_state(self, refresh: bool = False) -> dict[str, Any]:
        if self._state_cache is not None and not refresh:
            return self._state_cache
        payload = self.client.get_object(Bucket=self.bucket, Key=self.state_key)["Body"].read()
        self._state_cache = json.loads(payload)
        return self._state_cache

    def summarize_state(self, dataset_spec: DatasetSpec | None = None) -> dict[str, Any]:
        state = self.load_state()
        partitions = state.get("partitions", {})
        day_meta = state.get("days", {})
        partition_refs = (
            self.list_matching_partitions(dataset_spec) if dataset_spec is not None else self._partition_refs(partitions)
        )
        partition_status = Counter(ref.metadata.get("status", "unknown") for ref in partition_refs)
        exchange_counts = Counter(ref.exchange for ref in partition_refs)
        stream_counts = Counter(ref.stream for ref in partition_refs)
        symbol_counts = Counter(ref.symbol for ref in partition_refs)
        day_counts = Counter(ref.day for ref in partition_refs)
        successful_days = sorted({ref.day for ref in partition_refs if ref.metadata.get("status") == "success"})
        return {
            "bucket": self.bucket,
            "endpoint_url": self.endpoint_url,
            "state_key": self.state_key,
            "data_prefix": self.data_prefix,
            "storage_layout_hint": "exchange=<exchange>/stream=<stream>/symbol=<symbol>/date=<YYYYMMDD>/data.parquet",
            "last_compacted_date": state.get("last_compacted_date"),
            "updated_at": state.get("updated_at"),
            "visible_root_keys": self.list_visible_keys(prefix=f"{self.data_prefix}/", max_keys=50),
            "state_day_count": len(day_meta),
            "state_partition_count": len(partitions),
            "matched_partition_count": len(partition_refs),
            "partition_status_counts": dict(partition_status),
            "day_status_counts": dict(Counter(meta.get("status", "unknown") for meta in day_meta.values())),
            "exchange_counts": dict(exchange_counts),
            "stream_counts": dict(stream_counts),
            "symbol_counts_top": symbol_counts.most_common(20),
            "days_sample_first": sorted(day_counts)[:10],
            "days_sample_last": sorted(day_counts)[-10:],
            "successful_day_count": len(successful_days),
            "successful_days_sample_first": successful_days[:10],
            "successful_days_sample_last": successful_days[-10:],
        }

    def list_matching_partitions(self, dataset_spec: DatasetSpec) -> list["S3PartitionRef"]:
        day_filter = _dataset_days(dataset_spec)
        allowed_symbols = {symbol.lower() for symbol in dataset_spec.symbols}
        refs: list[S3PartitionRef] = []
        for ref in self._partition_refs(self.load_state().get("partitions", {})):
            if ref.exchange not in dataset_spec.exchanges:
                continue
            if ref.stream not in dataset_spec.stream_universe:
                continue
            if ref.symbol not in allowed_symbols:
                continue
            if ref.day not in day_filter:
                continue
            if not dataset_spec.stream_available(ref.exchange, ref.stream):
                continue
            if ref.metadata.get("status") != "success":
                continue
            refs.append(ref)
        return refs

    def list_visible_keys(self, prefix: str | None = None, max_keys: int = 1000) -> list[str]:
        effective_prefix = prefix if prefix is not None else f"{self.data_prefix}/"
        response = self.client.list_objects_v2(Bucket=self.bucket, Prefix=effective_prefix, MaxKeys=max_keys)
        return [item["Key"] for item in response.get("Contents", [])]

    def discover_partition_objects(self, partition: "S3PartitionRef | str") -> list[str]:
        partition_ref = partition if isinstance(partition, S3PartitionRef) else self._find_partition_ref(partition)
        storage_prefix = self._partition_storage_prefix(partition_ref)
        candidates = [
            f"{self.data_prefix}/{partition_ref.partition_id}",
            f"{self.data_prefix}/{partition_ref.partition_id}/",
            partition_ref.partition_id,
            f"{partition_ref.partition_id}/",
            storage_prefix,
        ]
        exact_candidates = [
            f"{storage_prefix}data.parquet",
            f"{storage_prefix}data.ndjson",
            f"{storage_prefix}data.jsonl",
            f"{storage_prefix}events.ndjson",
            f"{storage_prefix}events.jsonl",
        ]
        failing_key = partition_ref.metadata.get("failing_key")
        if isinstance(failing_key, str) and _is_supported_data_key(failing_key):
            exact_candidates.append(failing_key)
        discovered: list[str] = []
        for prefix in candidates:
            keys = self.list_visible_keys(prefix=prefix, max_keys=200)
            for key in keys:
                if key.endswith("_state.json"):
                    continue
                if _is_supported_data_key(key):
                    discovered.append(key)
        for key in exact_candidates:
            if self._object_exists(key) and _is_supported_data_key(key):
                discovered.append(key)
        return sorted(set(discovered))

    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                import boto3
            except ImportError as exc:  # pragma: no cover - dependency guard
                raise RuntimeError("boto3 is required for S3 compact support") from exc
            session = boto3.session.Session(
                aws_access_key_id=self.access_key_id,
                aws_secret_access_key=self.secret_access_key,
                region_name=self.region_name,
            )
            self._client = session.client("s3", endpoint_url=self.endpoint_url)
        return self._client

    def _partition_refs(self, partitions: dict[str, Any]) -> list["S3PartitionRef"]:
        refs: list[S3PartitionRef] = []
        for partition_id, metadata in partitions.items():
            try:
                exchange, stream, symbol, day = partition_id.split("/")
            except ValueError:
                continue
            refs.append(
                S3PartitionRef(
                    partition_id=partition_id,
                    exchange=exchange,
                    stream=stream,
                    symbol=symbol,
                    day=day,
                    metadata=metadata,
                )
            )
        return refs

    def _find_partition_ref(self, partition_id: str) -> "S3PartitionRef":
        for ref in self._partition_refs(self.load_state().get("partitions", {})):
            if ref.partition_id == partition_id:
                return ref
        raise KeyError(f"unknown partition_id: {partition_id}")

    def _partition_storage_prefix(self, partition_ref: "S3PartitionRef") -> str:
        return (
            f"exchange={partition_ref.exchange}/"
            f"stream={partition_ref.stream}/"
            f"symbol={partition_ref.symbol}/"
            f"date={partition_ref.day}/"
        )

    def _object_exists(self, key: str) -> bool:
        head_object = getattr(self.client, "head_object", None)
        if callable(head_object):
            try:
                head_object(Bucket=self.bucket, Key=key)
            except Exception:
                return False
            return True
        return any(found_key == key for found_key in self.list_visible_keys(prefix=key, max_keys=1))

    def _read_object_events(self, key: str, dataset_spec: DatasetSpec) -> list[NormalizedMarketEvent]:
        body = self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read()
        if key.endswith(".parquet"):
            try:
                import pyarrow.parquet as pq
                import pyarrow as pa
            except ImportError as exc:  # pragma: no cover - optional parser boundary
                raise RuntimeError("pyarrow is required to read parquet objects from S3") from exc
            table = pq.read_table(pa.BufferReader(body) if hasattr(pa, "BufferReader") else BytesIO(body))
            rows = table.to_pylist()
        else:
            text = _decode_body(body, key)
            rows = []
            for line in text.splitlines():
                if not line.strip():
                    continue
                rows.append(json.loads(line))

        events: list[NormalizedMarketEvent] = []
        for raw in rows:
            event = _normalize_record(raw, source_label=f"s3://{self.bucket}/{key}")
            if event.exchange in dataset_spec.exchanges and event.symbol in dataset_spec.symbols:
                if dataset_spec.stream_available(event.exchange, event.stream_type):
                    events.append(event)
        return events


@dataclass(slots=True)
class S3PartitionRef:
    partition_id: str
    exchange: str
    stream: str
    symbol: str
    day: str
    metadata: dict[str, Any]


def _normalize_record(record: dict[str, Any], source_label: str) -> NormalizedMarketEvent:
    stream_type = record.get("stream_type", record.get("stream"))
    if stream_type is None:
        raise KeyError("stream_type")
    canonical_value = _canonical_value(record)
    fields = {
        key: value
        for key, value in record.items()
        if key not in {"event_time", "ts_event", "exchange", "symbol", "stream_type", "stream", "value"}
    }
    return NormalizedMarketEvent(
        event_time=_coerce_event_time(record),
        exchange=record["exchange"],
        symbol=record["symbol"],
        stream_type=stream_type,
        value=canonical_value,
        fields=fields,
        ingest_metadata={"source": source_label},
    )


def _canonical_value(record: dict[str, Any]) -> float:
    if "value" in record:
        return float(record["value"])
    stream_type = record.get("stream_type", record.get("stream"))
    if stream_type == "bbo":
        bid = float(record.get("bid_price", record.get("bid", 0.0)))
        ask = float(record.get("ask_price", record.get("ask", 0.0)))
        return (bid + ask) / 2.0
    if stream_type == "trade":
        return float(record["price"])
    if stream_type == "mark_price":
        return float(record.get("mark_price", record.get("price", 0.0)))
    if stream_type == "funding":
        return float(record.get("funding_rate", record.get("rate", 0.0)))
    if stream_type == "open_interest":
        return float(record.get("open_interest", 0.0))
    raise ValueError(f"cannot infer canonical value for stream: {stream_type}")


def _coerce_event_time(record: dict[str, Any]) -> datetime | str:
    if "event_time" in record:
        return record["event_time"]
    if "ts_event" in record:
        return datetime.fromtimestamp(float(record["ts_event"]) / 1000.0, tz=UTC)
    raise KeyError("event_time")


def _dataset_days(dataset_spec: DatasetSpec) -> set[str]:
    start = dataset_spec.train_range.start.date()
    end = dataset_spec.eval_range.end.date()
    days: set[str] = set()
    current = start
    while current <= end:
        days.add(current.strftime("%Y%m%d"))
        current = current.fromordinal(current.toordinal() + 1)
    return days


def _is_supported_data_key(key: str) -> bool:
    base_name = key.rsplit("/", 1)[-1]
    if base_name in {"_state.json", "meta.json", "quality_day.json"}:
        return False
    suffixes = (
        ".parquet",
        ".jsonl",
        ".ndjson",
        ".json",
        ".jsonl.gz",
        ".ndjson.gz",
    )
    return key.endswith(suffixes)


def _decode_body(body: bytes, key: str) -> str:
    if key.endswith(".gz"):
        import gzip

        return gzip.decompress(body).decode("utf-8")
    return body.decode("utf-8")

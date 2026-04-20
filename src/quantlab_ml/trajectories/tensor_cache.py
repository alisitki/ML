from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import Field

from quantlab_ml.common import ensure_parent_dir
from quantlab_ml.contracts import (
    ActionFeasibilitySurface,
    PolicyState,
    RewardSnapshot,
    TrajectoryRecord,
    TrajectoryStep,
)
from quantlab_ml.contracts.common import QuantBaseModel
from quantlab_ml.models.features import observation_feature_array, observation_feature_segment_manifest

TENSOR_CACHE_FORMAT_VERSION = "tensor_cache_v1"
TENSOR_CACHE_DIRNAME = TENSOR_CACHE_FORMAT_VERSION
TENSOR_CACHE_MANIFEST_FILENAME = "tensor_cache_manifest.json"
TENSOR_CACHE_DIAGNOSTICS_FORMAT_VERSION = "tensor_cache_diagnostics_v1"
TENSOR_CACHE_DIAGNOSTICS_FILENAME = "tensor_cache_diagnostics.json"
DEFAULT_TENSOR_CACHE_SHARD_TARGET_BYTES = 512 * 1024 * 1024


def tensor_cache_directory(directory: Path) -> Path:
    return directory / TENSOR_CACHE_DIRNAME


def tensor_cache_manifest_path(directory: Path) -> Path:
    return tensor_cache_directory(directory) / TENSOR_CACHE_MANIFEST_FILENAME


def tensor_cache_diagnostics_path(directory: Path) -> Path:
    return tensor_cache_directory(directory) / TENSOR_CACHE_DIAGNOSTICS_FILENAME


def has_tensor_cache_manifest(directory: Path) -> bool:
    return tensor_cache_manifest_path(directory).exists()


class TensorCacheReplayRow(QuantBaseModel):
    event_time: datetime
    target_symbol: str
    trajectory_id: str
    trajectory_start: bool
    reward_snapshot: RewardSnapshot
    action_feasibility: ActionFeasibilitySurface
    policy_state: PolicyState | None = None


class TensorCacheShardManifest(QuantBaseModel):
    split_name: str
    shard_index: int
    row_count: int
    first_event_time: datetime
    last_event_time: datetime
    feature_path: str
    action_label_path: str
    venue_label_path: str
    venue_mask_path: str
    event_time_path: str
    trajectory_start_path: str
    replay_path: str


class TensorCacheSplitManifest(QuantBaseModel):
    split_name: str
    row_count: int
    shard_count: int
    shards: list[TensorCacheShardManifest] = Field(default_factory=list)


class TensorCacheManifest(QuantBaseModel):
    format_version: str = TENSOR_CACHE_FORMAT_VERSION
    feature_dtype: str
    feature_dim: int
    shard_target_bytes: int
    splits: dict[str, TensorCacheSplitManifest]


class TensorCachePayloadStatus(QuantBaseModel):
    manifest_present: bool
    payload_complete: bool
    referenced_payload_count: int = 0
    existing_payload_count: int = 0
    missing_payload_count: int = 0
    missing_payloads: list[str] = Field(default_factory=list)


class TensorCacheFeatureSegmentStats(QuantBaseModel):
    name: str
    start: int
    length: int
    nonzero_ratio: float
    always_zero_feature_count: int
    always_zero_ratio: float


class TensorCacheEmpiricalSparsitySummary(QuantBaseModel):
    row_count: int
    feature_dim: int
    total_nonzero_count: int
    total_value_count: int
    nonzero_ratio: float
    always_zero_feature_count: int
    always_zero_ratio: float
    segments: list[TensorCacheFeatureSegmentStats] = Field(default_factory=list)


class TensorCacheLabelHistograms(QuantBaseModel):
    action_counts: dict[str, int] = Field(default_factory=dict)
    venue_counts: dict[str, int] = Field(default_factory=dict)
    action_venue_counts: dict[str, int] = Field(default_factory=dict)
    trajectory_start_count: int = 0


class TensorCachePolicyStateHistograms(QuantBaseModel):
    previous_position_side_counts: dict[str, int] = Field(default_factory=dict)
    previous_venue_counts: dict[str, int] = Field(default_factory=dict)
    hold_age_steps_counts: dict[str, int] = Field(default_factory=dict)
    turnover_accumulator_counts: dict[str, int] = Field(default_factory=dict)
    missing_policy_state_count: int = 0


class TensorCacheSplitDiagnostics(QuantBaseModel):
    split_name: str
    empirical_sparsity: TensorCacheEmpiricalSparsitySummary
    label_histograms: TensorCacheLabelHistograms
    policy_state_histograms: TensorCachePolicyStateHistograms


class TensorCacheDiagnosticsManifest(QuantBaseModel):
    format_version: str = TENSOR_CACHE_DIAGNOSTICS_FORMAT_VERSION
    splits: dict[str, TensorCacheSplitDiagnostics]


@dataclass(slots=True)
class LoadedTensorCacheShard:
    features: np.ndarray
    action_labels: np.ndarray
    venue_labels: np.ndarray
    venue_mask: np.ndarray
    event_time_ms: np.ndarray
    trajectory_start: np.ndarray
    replay_rows: list[TensorCacheReplayRow]

    @property
    def row_count(self) -> int:
        return int(self.features.shape[0])


class TensorCacheSplitWriter:
    def __init__(
        self,
        *,
        directory: Path,
        split_name: str,
        feature_dim: int,
        action_keys: list[str],
        venue_choices: list[str],
        shard_target_bytes: int = DEFAULT_TENSOR_CACHE_SHARD_TARGET_BYTES,
    ) -> None:
        if feature_dim <= 0:
            raise ValueError("tensor cache feature_dim must be positive")
        self.directory = directory
        self.split_name = split_name
        self.feature_dim = feature_dim
        self.action_keys = action_keys
        self.venue_choices = venue_choices
        self.shard_target_bytes = shard_target_bytes
        self.split_dir = tensor_cache_directory(directory) / split_name
        self.rows_per_shard = max(
            1,
            shard_target_bytes // max(feature_dim * np.dtype(np.float32).itemsize, 1),
        )
        self._feature_buffer = np.empty((self.rows_per_shard, feature_dim), dtype=np.float32)
        self._action_label_buffer = np.empty(self.rows_per_shard, dtype=np.int64)
        self._venue_label_buffer = np.empty(self.rows_per_shard, dtype=np.int64)
        self._venue_mask_buffer = np.empty(self.rows_per_shard, dtype=np.bool_)
        self._event_time_buffer = np.empty(self.rows_per_shard, dtype=np.int64)
        self._trajectory_start_buffer = np.empty(self.rows_per_shard, dtype=np.bool_)
        self._replay_rows: list[TensorCacheReplayRow] = []
        self._shards: list[TensorCacheShardManifest] = []
        self._pending_rows = 0
        self._total_rows = 0
        self._next_shard_index = 0
        self._feature_nonzero_counts = np.zeros(feature_dim, dtype=np.int64)
        self._feature_segments: list[dict[str, int | str]] | None = None
        self._action_counts: Counter[str] = Counter()
        self._venue_counts: Counter[str] = Counter()
        self._action_venue_counts: Counter[str] = Counter()
        self._previous_position_side_counts: Counter[str] = Counter()
        self._previous_venue_counts: Counter[str] = Counter()
        self._hold_age_steps_counts: Counter[str] = Counter()
        self._turnover_accumulator_counts: Counter[str] = Counter()
        self._missing_policy_state_count = 0
        self._trajectory_start_count = 0
        self._final_diagnostics: TensorCacheSplitDiagnostics | None = None

    def consume_record(self, record: TrajectoryRecord) -> None:
        trajectory_start = True
        for step in record.steps:
            self._append_step(record=record, step=step, trajectory_start=trajectory_start)
            trajectory_start = False

    def finalize(self) -> TensorCacheSplitManifest:
        if self._pending_rows > 0:
            self._flush()
        self._final_diagnostics = self._build_split_diagnostics()
        return TensorCacheSplitManifest(
            split_name=self.split_name,
            row_count=self._total_rows,
            shard_count=len(self._shards),
            shards=self._shards,
        )

    def diagnostics(self) -> TensorCacheSplitDiagnostics:
        if self._final_diagnostics is None:
            raise ValueError("tensor cache diagnostics requested before finalize()")
        return self._final_diagnostics

    def _append_step(
        self,
        *,
        record: TrajectoryRecord,
        step: TrajectoryStep,
        trajectory_start: bool,
    ) -> None:
        features = observation_feature_array(step.observation, dtype=np.float32)
        if features.shape[0] != self.feature_dim:
            raise ValueError(
                f"tensor cache feature_dim mismatch for split={self.split_name!r}: "
                f"expected={self.feature_dim}, got={features.shape[0]}"
            )
        if self._feature_segments is None:
            self._feature_segments = observation_feature_segment_manifest(step.observation)
        action_key, venue = best_label_from_step(step)
        row = self._pending_rows
        self._feature_buffer[row] = features
        self._action_label_buffer[row] = self.action_keys.index(action_key)
        self._venue_mask_buffer[row] = venue is not None
        self._venue_label_buffer[row] = self.venue_choices.index(venue) if venue is not None else 0
        self._event_time_buffer[row] = datetime_to_epoch_millis(step.event_time)
        self._trajectory_start_buffer[row] = trajectory_start
        self._feature_nonzero_counts += np.not_equal(features, 0.0).astype(np.int64, copy=False)
        self._action_counts[action_key] += 1
        self._venue_counts[venue or "<none>"] += 1
        self._action_venue_counts[f"{action_key}::{venue or '<none>'}"] += 1
        if trajectory_start:
            self._trajectory_start_count += 1
        policy_state = step.policy_state
        if policy_state is None:
            self._missing_policy_state_count += 1
        else:
            self._previous_position_side_counts[policy_state.previous_position_side] += 1
            self._previous_venue_counts[policy_state.previous_venue or "<none>"] += 1
            self._hold_age_steps_counts[str(policy_state.hold_age_steps)] += 1
            self._turnover_accumulator_counts[_format_histogram_float(policy_state.turnover_accumulator)] += 1
        self._replay_rows.append(
            TensorCacheReplayRow(
                event_time=step.event_time,
                target_symbol=record.target_symbol,
                trajectory_id=record.trajectory_id,
                trajectory_start=trajectory_start,
                reward_snapshot=step.reward_snapshot.model_copy(deep=True),
                action_feasibility=step.action_feasibility.model_copy(deep=True),
                policy_state=step.policy_state.model_copy(deep=True) if step.policy_state is not None else None,
            )
        )
        self._pending_rows += 1
        self._total_rows += 1
        if self._pending_rows == self.rows_per_shard:
            self._flush()

    def _flush(self) -> None:
        if self._pending_rows <= 0:
            return
        shard_index = self._next_shard_index
        self._next_shard_index += 1
        ensure_parent_dir(self.split_dir / "placeholder")
        shard_prefix = f"shard_{shard_index:05d}"
        feature_path = self.split_dir / f"{shard_prefix}_X.pt"
        action_label_path = self.split_dir / f"{shard_prefix}_action_y.pt"
        venue_label_path = self.split_dir / f"{shard_prefix}_venue_y.pt"
        venue_mask_path = self.split_dir / f"{shard_prefix}_venue_mask.pt"
        event_time_path = self.split_dir / f"{shard_prefix}_event_time.pt"
        trajectory_start_path = self.split_dir / f"{shard_prefix}_trajectory_start.pt"
        replay_path = self.split_dir / f"{shard_prefix}_replay.jsonl"

        _torch_save(feature_path, self._feature_buffer[: self._pending_rows])
        _torch_save(action_label_path, self._action_label_buffer[: self._pending_rows])
        _torch_save(venue_label_path, self._venue_label_buffer[: self._pending_rows])
        _torch_save(venue_mask_path, self._venue_mask_buffer[: self._pending_rows])
        _torch_save(event_time_path, self._event_time_buffer[: self._pending_rows])
        _torch_save(trajectory_start_path, self._trajectory_start_buffer[: self._pending_rows])
        _write_replay_jsonl(replay_path, self._replay_rows)

        self._shards.append(
            TensorCacheShardManifest(
                split_name=self.split_name,
                shard_index=shard_index,
                row_count=self._pending_rows,
                first_event_time=self._replay_rows[0].event_time,
                last_event_time=self._replay_rows[-1].event_time,
                feature_path=_relative_cache_path(self.directory, feature_path),
                action_label_path=_relative_cache_path(self.directory, action_label_path),
                venue_label_path=_relative_cache_path(self.directory, venue_label_path),
                venue_mask_path=_relative_cache_path(self.directory, venue_mask_path),
                event_time_path=_relative_cache_path(self.directory, event_time_path),
                trajectory_start_path=_relative_cache_path(self.directory, trajectory_start_path),
                replay_path=_relative_cache_path(self.directory, replay_path),
            )
        )
        self._pending_rows = 0
        self._replay_rows = []

    def _build_split_diagnostics(self) -> TensorCacheSplitDiagnostics:
        if self._feature_segments is None:
            self._feature_segments = []
        total_value_count = self._total_rows * self.feature_dim
        total_nonzero_count = int(self._feature_nonzero_counts.sum())
        always_zero_feature_count = int(np.count_nonzero(self._feature_nonzero_counts == 0))
        segment_stats: list[TensorCacheFeatureSegmentStats] = []
        for segment in self._feature_segments:
            start = int(segment["start"])
            length = int(segment["length"])
            if length <= 0:
                segment_stats.append(
                    TensorCacheFeatureSegmentStats(
                        name=str(segment["name"]),
                        start=start,
                        length=length,
                        nonzero_ratio=0.0,
                        always_zero_feature_count=0,
                        always_zero_ratio=0.0,
                    )
                )
                continue
            segment_counts = self._feature_nonzero_counts[start : start + length]
            segment_nonzero = int(segment_counts.sum())
            segment_total_values = self._total_rows * length
            segment_always_zero = int(np.count_nonzero(segment_counts == 0))
            segment_stats.append(
                TensorCacheFeatureSegmentStats(
                    name=str(segment["name"]),
                    start=start,
                    length=length,
                    nonzero_ratio=(segment_nonzero / segment_total_values) if segment_total_values else 0.0,
                    always_zero_feature_count=segment_always_zero,
                    always_zero_ratio=(segment_always_zero / length) if length else 0.0,
                )
            )

        return TensorCacheSplitDiagnostics(
            split_name=self.split_name,
            empirical_sparsity=TensorCacheEmpiricalSparsitySummary(
                row_count=self._total_rows,
                feature_dim=self.feature_dim,
                total_nonzero_count=total_nonzero_count,
                total_value_count=total_value_count,
                nonzero_ratio=(total_nonzero_count / total_value_count) if total_value_count else 0.0,
                always_zero_feature_count=always_zero_feature_count,
                always_zero_ratio=(always_zero_feature_count / self.feature_dim) if self.feature_dim else 0.0,
                segments=segment_stats,
            ),
            label_histograms=TensorCacheLabelHistograms(
                action_counts=dict(self._action_counts),
                venue_counts=dict(self._venue_counts),
                action_venue_counts=dict(self._action_venue_counts),
                trajectory_start_count=self._trajectory_start_count,
            ),
            policy_state_histograms=TensorCachePolicyStateHistograms(
                previous_position_side_counts=dict(self._previous_position_side_counts),
                previous_venue_counts=dict(self._previous_venue_counts),
                hold_age_steps_counts=dict(self._hold_age_steps_counts),
                turnover_accumulator_counts=dict(self._turnover_accumulator_counts),
                missing_policy_state_count=self._missing_policy_state_count,
            ),
        )


def write_tensor_cache_manifest_atomic(directory: Path, manifest: TensorCacheManifest) -> None:
    path = tensor_cache_manifest_path(directory)
    ensure_parent_dir(path)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    tmp_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    tmp_path.replace(path)


def read_tensor_cache_manifest(directory: Path) -> TensorCacheManifest:
    return TensorCacheManifest.model_validate_json(
        tensor_cache_manifest_path(directory).read_text(encoding="utf-8")
    )


def tensor_cache_payload_status(directory: Path) -> TensorCachePayloadStatus:
    manifest_path = tensor_cache_manifest_path(directory)
    if not manifest_path.exists():
        return TensorCachePayloadStatus(
            manifest_present=False,
            payload_complete=False,
        )
    cache_manifest = read_tensor_cache_manifest(directory)
    referenced_paths: list[str] = []
    for split_manifest in cache_manifest.splits.values():
        for shard in split_manifest.shards:
            referenced_paths.extend(
                [
                    shard.feature_path,
                    shard.action_label_path,
                    shard.venue_label_path,
                    shard.venue_mask_path,
                    shard.event_time_path,
                    shard.trajectory_start_path,
                    shard.replay_path,
                ]
            )
    missing_payloads = sorted(
        {
            relative_path
            for relative_path in referenced_paths
            if not (directory / relative_path).exists()
        }
    )
    referenced_payload_count = len(referenced_paths)
    existing_payload_count = referenced_payload_count - len(missing_payloads)
    return TensorCachePayloadStatus(
        manifest_present=True,
        payload_complete=len(missing_payloads) == 0,
        referenced_payload_count=referenced_payload_count,
        existing_payload_count=existing_payload_count,
        missing_payload_count=len(missing_payloads),
        missing_payloads=missing_payloads,
    )


def has_tensor_cache(directory: Path) -> bool:
    return tensor_cache_payload_status(directory).payload_complete


def write_tensor_cache_diagnostics_atomic(
    directory: Path,
    diagnostics: TensorCacheDiagnosticsManifest,
) -> None:
    path = tensor_cache_diagnostics_path(directory)
    ensure_parent_dir(path)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    tmp_path.write_text(diagnostics.model_dump_json(indent=2), encoding="utf-8")
    tmp_path.replace(path)


def read_tensor_cache_diagnostics(directory: Path) -> TensorCacheDiagnosticsManifest:
    return TensorCacheDiagnosticsManifest.model_validate_json(
        tensor_cache_diagnostics_path(directory).read_text(encoding="utf-8")
    )


def load_tensor_cache_shard(directory: Path, shard: TensorCacheShardManifest) -> LoadedTensorCacheShard:
    features = _torch_load_numpy(directory / shard.feature_path, np.float32)
    action_labels = _torch_load_numpy(directory / shard.action_label_path, np.int64)
    venue_labels = _torch_load_numpy(directory / shard.venue_label_path, np.int64)
    venue_mask = _torch_load_numpy(directory / shard.venue_mask_path, np.bool_)
    event_time_ms = _torch_load_numpy(directory / shard.event_time_path, np.int64)
    trajectory_start = _torch_load_numpy(directory / shard.trajectory_start_path, np.bool_)
    replay_rows = list(_read_replay_jsonl(directory / shard.replay_path))
    row_count = int(features.shape[0])
    if len(replay_rows) != row_count:
        raise ValueError(
            f"tensor cache replay row count mismatch for split={shard.split_name!r} shard={shard.shard_index}: "
            f"tensor_rows={row_count}, replay_rows={len(replay_rows)}"
        )
    return LoadedTensorCacheShard(
        features=features,
        action_labels=action_labels,
        venue_labels=venue_labels,
        venue_mask=venue_mask,
        event_time_ms=event_time_ms,
        trajectory_start=trajectory_start,
        replay_rows=replay_rows,
    )


def window_row_indices(
    event_time_ms: np.ndarray,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
    exclusive_end: datetime | None = None,
) -> np.ndarray:
    mask = np.ones(int(event_time_ms.shape[0]), dtype=np.bool_)
    if start is not None:
        mask &= event_time_ms >= datetime_to_epoch_millis(start)
    if end is not None:
        mask &= event_time_ms <= datetime_to_epoch_millis(end)
    if exclusive_end is not None:
        mask &= event_time_ms < datetime_to_epoch_millis(exclusive_end)
    return np.flatnonzero(mask)


def datetime_to_epoch_millis(value: datetime) -> int:
    return int(value.timestamp() * 1000)


def epoch_millis_to_datetime(value: int) -> datetime:
    return datetime.fromtimestamp(value / 1000.0, tz=UTC)


def best_label_from_step(step: TrajectoryStep) -> tuple[str, str | None]:
    abstain_reward = step.reward_snapshot.for_action("abstain").net_reward
    best_directional = None
    for reward in step.reward_snapshot.action_rewards:
        if reward.action_key == "abstain" or not reward.applicable:
            continue
        if best_directional is None or reward.net_reward > best_directional.net_reward:
            best_directional = reward
    if best_directional is None or best_directional.net_reward <= abstain_reward:
        return "abstain", None
    return best_directional.action_key, best_directional.venue


def _format_histogram_float(value: float) -> str:
    return f"{value:.6g}"


def _relative_cache_path(directory: Path, path: Path) -> str:
    return str(path.relative_to(directory))


def _write_replay_jsonl(path: Path, rows: list[TensorCacheReplayRow]) -> None:
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(row.model_dump_json())
            handle.write("\n")


def _read_replay_jsonl(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.rstrip("\n")
            if not stripped:
                continue
            yield TensorCacheReplayRow.model_validate_json(stripped)


def _torch_save(path: Path, values: np.ndarray) -> None:
    ensure_parent_dir(path)
    torch_module = _require_torch()
    tensor = torch_module.from_numpy(np.ascontiguousarray(values))
    torch_module.save(tensor, path)


def _torch_load_numpy(path: Path, dtype: type[np.generic]) -> np.ndarray:
    torch_module = _require_torch()
    loaded = torch_module.load(path, map_location="cpu")
    if hasattr(loaded, "detach"):
        array = loaded.detach().cpu().numpy()
    else:
        array = np.asarray(loaded)
    return np.asarray(array, dtype=dtype)


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("torch is required for tensor cache support") from exc
    return torch


__all__ = [
    "DEFAULT_TENSOR_CACHE_SHARD_TARGET_BYTES",
    "LoadedTensorCacheShard",
    "TENSOR_CACHE_DIRNAME",
    "TENSOR_CACHE_DIAGNOSTICS_FILENAME",
    "TENSOR_CACHE_DIAGNOSTICS_FORMAT_VERSION",
    "TENSOR_CACHE_FORMAT_VERSION",
    "TENSOR_CACHE_MANIFEST_FILENAME",
    "TensorCacheDiagnosticsManifest",
    "TensorCacheEmpiricalSparsitySummary",
    "TensorCacheFeatureSegmentStats",
    "TensorCacheLabelHistograms",
    "TensorCacheManifest",
    "TensorCachePayloadStatus",
    "TensorCachePolicyStateHistograms",
    "TensorCacheReplayRow",
    "TensorCacheShardManifest",
    "TensorCacheSplitDiagnostics",
    "TensorCacheSplitManifest",
    "TensorCacheSplitWriter",
    "best_label_from_step",
    "datetime_to_epoch_millis",
    "epoch_millis_to_datetime",
    "has_tensor_cache",
    "has_tensor_cache_manifest",
    "load_tensor_cache_shard",
    "read_tensor_cache_diagnostics",
    "read_tensor_cache_manifest",
    "tensor_cache_directory",
    "tensor_cache_diagnostics_path",
    "tensor_cache_manifest_path",
    "tensor_cache_payload_status",
    "window_row_indices",
    "write_tensor_cache_diagnostics_atomic",
    "write_tensor_cache_manifest_atomic",
]

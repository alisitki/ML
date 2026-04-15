from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import Field

from quantlab_ml.common import ensure_parent_dir
from quantlab_ml.contracts import (
    ActionFeasibilitySurface,
    RewardSnapshot,
    TrajectoryRecord,
    TrajectoryStep,
)
from quantlab_ml.contracts.common import QuantBaseModel
from quantlab_ml.models.features import observation_feature_array

TENSOR_CACHE_FORMAT_VERSION = "tensor_cache_v1"
TENSOR_CACHE_DIRNAME = TENSOR_CACHE_FORMAT_VERSION
TENSOR_CACHE_MANIFEST_FILENAME = "tensor_cache_manifest.json"
DEFAULT_TENSOR_CACHE_SHARD_TARGET_BYTES = 512 * 1024 * 1024


def tensor_cache_directory(directory: Path) -> Path:
    return directory / TENSOR_CACHE_DIRNAME


def tensor_cache_manifest_path(directory: Path) -> Path:
    return tensor_cache_directory(directory) / TENSOR_CACHE_MANIFEST_FILENAME


def has_tensor_cache(directory: Path) -> bool:
    return tensor_cache_manifest_path(directory).exists()


class TensorCacheReplayRow(QuantBaseModel):
    event_time: datetime
    target_symbol: str
    trajectory_id: str
    trajectory_start: bool
    reward_snapshot: RewardSnapshot
    action_feasibility: ActionFeasibilitySurface


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

    def consume_record(self, record: TrajectoryRecord) -> None:
        trajectory_start = True
        for step in record.steps:
            self._append_step(record=record, step=step, trajectory_start=trajectory_start)
            trajectory_start = False

    def finalize(self) -> TensorCacheSplitManifest:
        if self._pending_rows > 0:
            self._flush()
        return TensorCacheSplitManifest(
            split_name=self.split_name,
            row_count=self._total_rows,
            shard_count=len(self._shards),
            shards=self._shards,
        )

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
        action_key, venue = best_label_from_step(step)
        row = self._pending_rows
        self._feature_buffer[row] = features
        self._action_label_buffer[row] = self.action_keys.index(action_key)
        self._venue_mask_buffer[row] = venue is not None
        self._venue_label_buffer[row] = self.venue_choices.index(venue) if venue is not None else 0
        self._event_time_buffer[row] = datetime_to_epoch_millis(step.event_time)
        self._trajectory_start_buffer[row] = trajectory_start
        self._replay_rows.append(
            TensorCacheReplayRow(
                event_time=step.event_time,
                target_symbol=record.target_symbol,
                trajectory_id=record.trajectory_id,
                trajectory_start=trajectory_start,
                reward_snapshot=step.reward_snapshot.model_copy(deep=True),
                action_feasibility=step.action_feasibility.model_copy(deep=True),
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
    "TENSOR_CACHE_FORMAT_VERSION",
    "TENSOR_CACHE_MANIFEST_FILENAME",
    "TensorCacheManifest",
    "TensorCacheReplayRow",
    "TensorCacheShardManifest",
    "TensorCacheSplitManifest",
    "TensorCacheSplitWriter",
    "best_label_from_step",
    "datetime_to_epoch_millis",
    "epoch_millis_to_datetime",
    "has_tensor_cache",
    "load_tensor_cache_shard",
    "read_tensor_cache_manifest",
    "tensor_cache_directory",
    "tensor_cache_manifest_path",
    "window_row_indices",
    "write_tensor_cache_manifest_atomic",
]

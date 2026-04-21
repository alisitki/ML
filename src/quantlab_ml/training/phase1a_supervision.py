from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import Field

from quantlab_ml.common import ensure_parent_dir, hash_payload
from quantlab_ml.contracts import (
    ActionSpaceSpec,
    JOINT_ACTION_VOCABULARY_VERSION_PHASE1A,
    POLICY_STATE_FEATURE_VERSION_PHASE1A,
    PolicyState,
    RewardEventSpec,
    TrajectoryManifest,
)
from quantlab_ml.contracts.common import QuantBaseModel
from quantlab_ml.models.features import policy_state_feature_array
from quantlab_ml.rewards import RewardEngine
from quantlab_ml.trajectories.streaming_store import TrajectoryDirectoryStore
from quantlab_ml.trajectories.tensor_cache import (
    TensorCacheManifest,
    datetime_to_epoch_millis,
    read_tensor_cache_manifest,
    tensor_cache_payload_status,
)
from .config import TrainingConfig
from .phase1a_oracle import (
    apply_phase1a_joint_action,
    phase1a_joint_action_keys,
    phase1a_joint_action_mask,
    phase1a_label_available,
    solve_phase1a_oracle,
)

PHASE1A_SUPERVISION_FORMAT_VERSION = "phase1a_supervision_v1"
PHASE1A_SUPERVISION_DIRNAME = PHASE1A_SUPERVISION_FORMAT_VERSION
PHASE1A_SUPERVISION_MANIFEST_FILENAME = "manifest.json"


class Phase1ASupervisionShardManifest(QuantBaseModel):
    split_name: str
    shard_index: int
    row_count: int
    first_event_time_ms: int
    last_event_time_ms: int
    policy_state_path: str
    joint_mask_path: str
    joint_label_path: str
    value_target_path: str
    supervised_mask_path: str


class Phase1ASupervisionSplitManifest(QuantBaseModel):
    split_name: str
    row_count: int
    shard_count: int
    shards: list[Phase1ASupervisionShardManifest] = Field(default_factory=list)


class Phase1ASupervisionManifest(QuantBaseModel):
    format_version: str = PHASE1A_SUPERVISION_FORMAT_VERSION
    tensor_cache_manifest_hash: str
    training_config_hash: str
    reward_version: str
    action_space_version: str
    bootstrap_horizon_steps: int
    policy_state_feature_version: str
    joint_action_vocabulary_version: str
    phase1a_compute_dtype: str
    policy_state_dtype: str
    value_target_dtype: str
    splits: dict[str, Phase1ASupervisionSplitManifest]


class Phase1ASupervisionPayloadStatus(QuantBaseModel):
    manifest_present: bool
    payload_complete: bool
    missing_payloads: list[str] = Field(default_factory=list)


@dataclass(slots=True, frozen=True)
class LoadedPhase1ASupervisionShard:
    policy_state_features: np.ndarray
    joint_mask: np.ndarray
    joint_labels: np.ndarray
    value_targets: np.ndarray
    supervised_mask: np.ndarray


@dataclass(slots=True, frozen=True)
class Phase1ASupervisionMaterializationReport:
    manifest: Phase1ASupervisionManifest
    output_dir: Path
    materialization_wall_sec: float
    materialization_reused: bool
    tensor_cache_used: bool
    phase1a_supervision_used: bool
    tensor_cache_manifest_hash: str
    phase1a_supervision_manifest_hash: str
    training_config_hash: str
    action_space_version: str
    policy_state_feature_version: str
    phase1a_compute_dtype: str

    def profile_payload(self) -> dict[str, object]:
        return {
            "materialization_wall_sec": self.materialization_wall_sec,
            "materialization_reused": self.materialization_reused,
            "tensor_cache_used": self.tensor_cache_used,
            "phase1a_supervision_used": self.phase1a_supervision_used,
            "tensor_cache_manifest_hash": self.tensor_cache_manifest_hash,
            "phase1a_supervision_manifest_hash": self.phase1a_supervision_manifest_hash,
            "training_config_hash": self.training_config_hash,
            "action_space_version": self.action_space_version,
            "policy_state_feature_version": self.policy_state_feature_version,
            "phase1a_compute_dtype": self.phase1a_compute_dtype,
        }


def phase1a_supervision_directory(directory: Path) -> Path:
    return directory / PHASE1A_SUPERVISION_DIRNAME


def phase1a_supervision_manifest_path(directory: Path) -> Path:
    return phase1a_supervision_directory(directory) / PHASE1A_SUPERVISION_MANIFEST_FILENAME


def read_phase1a_supervision_manifest(directory: Path) -> Phase1ASupervisionManifest:
    return Phase1ASupervisionManifest.model_validate_json(
        phase1a_supervision_manifest_path(directory).read_text(encoding="utf-8")
    )


def phase1a_supervision_payload_status(directory: Path) -> Phase1ASupervisionPayloadStatus:
    manifest_path = phase1a_supervision_manifest_path(directory)
    if not manifest_path.exists():
        return Phase1ASupervisionPayloadStatus(
            manifest_present=False,
            payload_complete=False,
        )
    manifest = read_phase1a_supervision_manifest(directory)
    missing_payloads: list[str] = []
    for split_manifest in manifest.splits.values():
        for shard in split_manifest.shards:
            for relative_path in (
                shard.policy_state_path,
                shard.joint_mask_path,
                shard.joint_label_path,
                shard.value_target_path,
                shard.supervised_mask_path,
            ):
                if not (directory / relative_path).exists():
                    missing_payloads.append(relative_path)
    return Phase1ASupervisionPayloadStatus(
        manifest_present=True,
        payload_complete=len(missing_payloads) == 0,
        missing_payloads=sorted(set(missing_payloads)),
    )


def load_phase1a_supervision_shard(
    directory: Path,
    shard: Phase1ASupervisionShardManifest,
) -> LoadedPhase1ASupervisionShard:
    return LoadedPhase1ASupervisionShard(
        policy_state_features=_torch_load_numpy(directory / shard.policy_state_path, np.float32),
        joint_mask=_torch_load_numpy(directory / shard.joint_mask_path, np.bool_),
        joint_labels=_torch_load_numpy(directory / shard.joint_label_path, np.int64),
        value_targets=_torch_load_numpy(directory / shard.value_target_path, np.float32),
        supervised_mask=_torch_load_numpy(directory / shard.supervised_mask_path, np.bool_),
    )


def materialize_phase1a_supervision(
    *,
    trajectories_directory: Path,
    output_directory: Path,
    manifest: TrajectoryManifest,
    training_config: TrainingConfig,
) -> Phase1ASupervisionMaterializationReport:
    started_at = time.perf_counter()
    tensor_cache_status = tensor_cache_payload_status(trajectories_directory)
    if not tensor_cache_status.payload_complete:
        raise ValueError("phase1a supervision materialization requires payload-complete tensor_cache_v1")
    if training_config.runtime_adapter != "linear-policy-v2":
        raise ValueError("phase1a supervision materialization requires linear-policy-v2 training config")
    if training_config.policy_state_feature_version != POLICY_STATE_FEATURE_VERSION_PHASE1A:
        raise ValueError("phase1a supervision materialization requires policy_state_features_v2_phase1a")
    if training_config.joint_action_vocabulary_version != JOINT_ACTION_VOCABULARY_VERSION_PHASE1A:
        raise ValueError("phase1a supervision materialization requires joint_action_vocabulary_v2_phase1a")

    tensor_cache_manifest = read_tensor_cache_manifest(trajectories_directory)
    tensor_cache_manifest_hash = hash_payload(tensor_cache_manifest)
    training_config_hash = hash_payload(training_config)
    expected_manifest = Phase1ASupervisionManifest(
        tensor_cache_manifest_hash=tensor_cache_manifest_hash,
        training_config_hash=training_config_hash,
        reward_version=manifest.reward_spec.reward_version,
        action_space_version=manifest.action_space.action_space_version,
        bootstrap_horizon_steps=training_config.bootstrap_horizon_steps,
        policy_state_feature_version=training_config.policy_state_feature_version or POLICY_STATE_FEATURE_VERSION_PHASE1A,
        joint_action_vocabulary_version=training_config.joint_action_vocabulary_version
        or JOINT_ACTION_VOCABULARY_VERSION_PHASE1A,
        phase1a_compute_dtype=training_config.phase1a_compute_dtype,
        policy_state_dtype="float32",
        value_target_dtype="float32",
        splits={},
    )
    existing_status = phase1a_supervision_payload_status(trajectories_directory)
    if existing_status.payload_complete:
        existing_manifest = read_phase1a_supervision_manifest(trajectories_directory)
        if _compatibility_payload(existing_manifest) == _compatibility_payload(expected_manifest):
            return Phase1ASupervisionMaterializationReport(
                manifest=existing_manifest,
                output_dir=phase1a_supervision_directory(trajectories_directory),
                materialization_wall_sec=time.perf_counter() - started_at,
                materialization_reused=True,
                tensor_cache_used=True,
                phase1a_supervision_used=True,
                tensor_cache_manifest_hash=tensor_cache_manifest_hash,
                phase1a_supervision_manifest_hash=hash_payload(existing_manifest),
                training_config_hash=training_config_hash,
                action_space_version=manifest.action_space.action_space_version,
                policy_state_feature_version=training_config.policy_state_feature_version
                or POLICY_STATE_FEATURE_VERSION_PHASE1A,
                phase1a_compute_dtype=training_config.phase1a_compute_dtype,
            )

    resolved_output_directory = output_directory.expanduser().resolve()
    expected_output_directory = phase1a_supervision_directory(trajectories_directory.expanduser().resolve())
    if resolved_output_directory != expected_output_directory:
        raise ValueError(
            "phase1a supervision output directory must be the canonical sibling path "
            f"{expected_output_directory}"
        )
    if resolved_output_directory.exists():
        shutil.rmtree(resolved_output_directory)
    venue_choices = list(manifest.dataset_spec.exchanges)
    reward_engine = RewardEngine(manifest.reward_spec, manifest.action_space)
    split_manifests: dict[str, Phase1ASupervisionSplitManifest] = {}
    joint_action_keys = phase1a_joint_action_keys(venue_choices)
    joint_action_to_index = {key: index for index, key in enumerate(joint_action_keys)}

    for split_name, cache_split in tensor_cache_manifest.splits.items():
        shard_manifests: list[Phase1ASupervisionShardManifest] = []
        record_iter = TrajectoryDirectoryStore.iter_records(trajectories_directory, split_name)
        current_record = next(record_iter, None)
        current_record_row_index = 0
        current_policy_state = PolicyState()
        current_rows: list[tuple[np.ndarray, np.ndarray, int, float, bool, int]] = []

        for shard in cache_split.shards:
            current_rows.clear()
            first_event_time_ms: int | None = None
            last_event_time_ms: int | None = None
            while len(current_rows) < shard.row_count:
                if current_record is None:
                    raise ValueError(
                        f"phase1a supervision row underflow for split={split_name!r} shard={shard.shard_index}"
                    )
                row = current_record.steps[current_record_row_index]
                if current_record_row_index == 0:
                    current_policy_state = PolicyState()
                policy_state_features = policy_state_feature_array(
                    current_policy_state,
                    venue_choices=venue_choices,
                    dtype=np.float32,
                )
                joint_mask = phase1a_joint_action_mask(
                    venue_choices=venue_choices,
                    action_feasibility=row.action_feasibility,
                    policy_state=current_policy_state,
                    preferred_size_band=training_config.preferred_size_band,
                    preferred_leverage_band=training_config.preferred_leverage_band,
                )
                supervised = phase1a_label_available(
                    row_count=len(current_record.steps),
                    row_index=current_record_row_index,
                    horizon_steps=training_config.bootstrap_horizon_steps,
                )
                label_index = 0
                oracle_return = 0.0
                if supervised:
                    oracle = solve_phase1a_oracle(
                        rows=current_record.steps,
                        row_index=current_record_row_index,
                        horizon_steps=training_config.bootstrap_horizon_steps,
                        venue_choices=venue_choices,
                        reward_engine=reward_engine,
                        policy_state=current_policy_state,
                        preferred_size_band=training_config.preferred_size_band,
                        preferred_leverage_band=training_config.preferred_leverage_band,
                    )
                    label_index = joint_action_to_index[oracle.joint_action_key]
                    oracle_return = float(oracle.oracle_return)
                    applied = apply_phase1a_joint_action(
                        reward_engine=reward_engine,
                        row=row,
                        joint_action_key=oracle.joint_action_key,
                        policy_state=current_policy_state,
                        preferred_size_band=training_config.preferred_size_band,
                        preferred_leverage_band=training_config.preferred_leverage_band,
                    )
                    current_policy_state = reward_engine.advance_policy_state(current_policy_state, applied)
                event_time_ms = datetime_to_epoch_millis(row.event_time)
                if first_event_time_ms is None:
                    first_event_time_ms = event_time_ms
                last_event_time_ms = event_time_ms
                current_rows.append(
                    (
                        policy_state_features,
                        joint_mask.astype(np.bool_, copy=False),
                        label_index,
                        oracle_return,
                        supervised,
                        event_time_ms,
                    )
                )
                current_record_row_index += 1
                if current_record_row_index >= len(current_record.steps):
                    current_record = next(record_iter, None)
                    current_record_row_index = 0

            shard_prefix = f"shard_{shard.shard_index:05d}"
            split_dir = resolved_output_directory / split_name
            policy_state_path = split_dir / f"{shard_prefix}_policy_state_X.pt"
            joint_mask_path = split_dir / f"{shard_prefix}_joint_mask.pt"
            joint_label_path = split_dir / f"{shard_prefix}_joint_y.pt"
            value_target_path = split_dir / f"{shard_prefix}_value_y.pt"
            supervised_mask_path = split_dir / f"{shard_prefix}_supervised_mask.pt"

            policy_state_array = np.stack([row[0] for row in current_rows], axis=0).astype(np.float32, copy=False)
            joint_mask_array = np.stack([row[1] for row in current_rows], axis=0).astype(np.bool_, copy=False)
            joint_label_array = np.asarray([row[2] for row in current_rows], dtype=np.int64)
            value_target_array = np.asarray([row[3] for row in current_rows], dtype=np.float32)
            supervised_mask_array = np.asarray([row[4] for row in current_rows], dtype=np.bool_)

            _torch_save(policy_state_path, policy_state_array)
            _torch_save(joint_mask_path, joint_mask_array)
            _torch_save(joint_label_path, joint_label_array)
            _torch_save(value_target_path, value_target_array)
            _torch_save(supervised_mask_path, supervised_mask_array)
            shard_manifests.append(
                Phase1ASupervisionShardManifest(
                    split_name=split_name,
                    shard_index=shard.shard_index,
                    row_count=shard.row_count,
                    first_event_time_ms=int(first_event_time_ms or 0),
                    last_event_time_ms=int(last_event_time_ms or 0),
                    policy_state_path=_relative_path(trajectories_directory, policy_state_path),
                    joint_mask_path=_relative_path(trajectories_directory, joint_mask_path),
                    joint_label_path=_relative_path(trajectories_directory, joint_label_path),
                    value_target_path=_relative_path(trajectories_directory, value_target_path),
                    supervised_mask_path=_relative_path(trajectories_directory, supervised_mask_path),
                )
            )

        split_manifests[split_name] = Phase1ASupervisionSplitManifest(
            split_name=split_name,
            row_count=cache_split.row_count,
            shard_count=cache_split.shard_count,
            shards=shard_manifests,
        )

    finalized_manifest = expected_manifest.model_copy(update={"splits": split_manifests})
    write_phase1a_supervision_manifest_atomic(trajectories_directory, finalized_manifest)
    return Phase1ASupervisionMaterializationReport(
        manifest=finalized_manifest,
        output_dir=resolved_output_directory,
        materialization_wall_sec=time.perf_counter() - started_at,
        materialization_reused=False,
        tensor_cache_used=True,
        phase1a_supervision_used=True,
        tensor_cache_manifest_hash=tensor_cache_manifest_hash,
        phase1a_supervision_manifest_hash=hash_payload(finalized_manifest),
        training_config_hash=training_config_hash,
        action_space_version=manifest.action_space.action_space_version,
        policy_state_feature_version=training_config.policy_state_feature_version or POLICY_STATE_FEATURE_VERSION_PHASE1A,
        phase1a_compute_dtype=training_config.phase1a_compute_dtype,
    )


def write_phase1a_supervision_manifest_atomic(
    directory: Path,
    manifest: Phase1ASupervisionManifest,
) -> None:
    path = phase1a_supervision_manifest_path(directory)
    ensure_parent_dir(path)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    tmp_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _compatibility_payload(manifest: Phase1ASupervisionManifest) -> dict[str, object]:
    return {
        "format_version": manifest.format_version,
        "tensor_cache_manifest_hash": manifest.tensor_cache_manifest_hash,
        "training_config_hash": manifest.training_config_hash,
        "reward_version": manifest.reward_version,
        "action_space_version": manifest.action_space_version,
        "bootstrap_horizon_steps": manifest.bootstrap_horizon_steps,
        "policy_state_feature_version": manifest.policy_state_feature_version,
        "joint_action_vocabulary_version": manifest.joint_action_vocabulary_version,
        "phase1a_compute_dtype": manifest.phase1a_compute_dtype,
    }


def _relative_path(directory: Path, path: Path) -> str:
    return str(path.resolve().relative_to(directory.expanduser().resolve()))


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("torch is required for phase1a supervision support") from exc
    return torch


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

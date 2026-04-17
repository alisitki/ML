"""contracts/manifest.py — Streaming-compatible trajectory metadata.

TrajectoryManifest is the lightweight counterpart to TrajectoryBundle.
It carries dataset/training/reward/observation metadata but contains NO
step records and NO trajectory data.

Usage contract:
    - Production build path   → writes manifest.json + per-split JSONL files
    - Production train path   → reads manifest.json only, streams JSONL per split
    - Test/fixture compat path → uses TrajectoryBundle (see trajectories.storage)
"""
from __future__ import annotations

from pydantic import Field

from quantlab_ml.contracts.common import QuantBaseModel
from quantlab_ml.contracts.dataset import DatasetSpec
from quantlab_ml.contracts.learning_surface import (
    ActionSpaceSpec,
    ObservationSchema,
    SplitArtifact,
    TrajectorySpec,
)
from quantlab_ml.contracts.rewards import RewardEventSpec

TRAJECTORY_STREAMING_FORMAT_VERSION = "trajectories_streaming_v1"


class TrajectorySplitStats(QuantBaseModel):
    """Persisted split counts for retained build/accounting evidence."""

    record_count: int
    step_count: int


class TrajectoryManifest(QuantBaseModel):
    """Metadata-only manifest for streaming trajectory directories.

    Contains everything needed to configure training and evaluation,
    but no trajectory steps or records.  Step data lives exclusively
    in per-split JSONL files alongside this manifest.
    """

    format_version: str = TRAJECTORY_STREAMING_FORMAT_VERSION
    dataset_spec: DatasetSpec
    trajectory_spec: TrajectorySpec
    action_space: ActionSpaceSpec
    reward_spec: RewardEventSpec
    observation_schema: ObservationSchema
    split_artifact: SplitArtifact
    # Names of splits that were written (in build order)
    split_names: list[str]
    split_write_stats: dict[str, TrajectorySplitStats] = Field(default_factory=dict)

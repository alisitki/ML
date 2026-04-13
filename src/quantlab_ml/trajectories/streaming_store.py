"""trajectories/streaming_store.py — Production streaming trajectory store.

PRODUCTION PATH: build → JSONL directory, train/evaluate → stream JSONL.

The directory layout written by TrajectoryDirectoryStore:

    <output_dir>/
    ├── manifest.json                 # TrajectoryManifest (no step data)
    ├── development.jsonl             # one TrajectoryRecord JSON per line
    ├── train.jsonl
    ├── validation.jsonl
    └── final_untouched_test.jsonl

Each JSONL line is the complete JSON serialization of one TrajectoryRecord,
including ndarray fields encoded as base64 dicts (from contracts/numpy_types.py).

Record line-size guard:
    Lines are checked against WARN_RECORD_LINE_MB and FAIL_RECORD_LINE_MB.
    Large lines are logged as WARNING; lines exceeding the fail threshold raise
    RuntimeError to make oversized records immediately visible.

For backward-compatible in-memory builds (small fixture / test data), see:
    quantlab_ml.trajectories.storage.TrajectoryStore  ← FIXTURE/TEST COMPAT ONLY
"""
from __future__ import annotations

import gc
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

from quantlab_ml.common import dump_model, load_model
from quantlab_ml.contracts import TrajectoryManifest, TrajectoryRecord

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Line-size thresholds
# ---------------------------------------------------------------------------

# Warn if a single serialised record line exceeds this size.
# Default: 512 MB.  Typical: 128 steps × ~1.9 MB/step ≈ 243 MB.
WARN_RECORD_LINE_MB: float = 512.0

# Fail hard if a single line exceeds this size — data error or config problem.
FAIL_RECORD_LINE_MB: float = 2048.0

# Explicit GC collect every N records during write/read.
_GC_EVERY_N_RECORDS: int = 10

# File names
MANIFEST_FILENAME = "manifest.json"


def _split_filename(split_name: str) -> str:
    return f"{split_name}.jsonl"


class TrajectoryDirectoryStore:
    """Streaming read/write for the trajectory directory format.

    This is the PRODUCTION trajectory store.
    Use TrajectoryStore (storage.py) ONLY for fixture/test compat.
    """

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    @staticmethod
    def write_manifest(output_dir: Path, manifest: TrajectoryManifest) -> None:
        """Write manifest.json to output_dir (creates dir if absent)."""
        output_dir.mkdir(parents=True, exist_ok=True)
        dump_model(output_dir / MANIFEST_FILENAME, manifest)
        logger.info("streaming_store_wrote_manifest path=%s", output_dir / MANIFEST_FILENAME)

    @staticmethod
    def write_split(
        output_dir: Path,
        split_name: str,
        records: Iterator[TrajectoryRecord],
        *,
        warn_line_mb: float = WARN_RECORD_LINE_MB,
        fail_line_mb: float = FAIL_RECORD_LINE_MB,
    ) -> tuple[int, int]:
        """Stream-write one split to a JSONL file.

        Returns (record_count, total_step_count).

        Line-size guard:
            Logs WARNING if any line exceeds warn_line_mb.
            Raises RuntimeError if any line exceeds fail_line_mb.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / _split_filename(split_name)
        warn_bytes = int(warn_line_mb * 1024 * 1024)
        fail_bytes = int(fail_line_mb * 1024 * 1024)

        record_count = 0
        total_steps = 0

        with path.open("w", encoding="utf-8") as f:
            for record in records:
                line = record.model_dump_json()
                line_bytes = len(line.encode("utf-8"))

                # ------- line-size guard -------
                if line_bytes > fail_bytes:
                    raise RuntimeError(
                        f"record_line_too_large: split={split_name!r} "
                        f"trajectory_id={record.trajectory_id!r} "
                        f"size_mb={line_bytes / 1_048_576:.1f} "
                        f"fail_threshold_mb={fail_line_mb:.0f} — "
                        "check max_episode_steps and tensor dimensions"
                    )
                if line_bytes > warn_bytes:
                    logger.warning(
                        "large_record_line split=%s trajectory_id=%s "
                        "size_mb=%.1f warn_threshold_mb=%.0f",
                        split_name,
                        record.trajectory_id,
                        line_bytes / 1_048_576,
                        warn_line_mb,
                    )
                # ------- write -------
                f.write(line)
                f.write("\n")

                record_count += 1
                total_steps += len(record.steps)
                del line  # release serialised string immediately

                if record_count % _GC_EVERY_N_RECORDS == 0:
                    gc.collect()

        logger.info(
            "streaming_store_wrote_split split=%s records=%d steps=%d path=%s",
            split_name,
            record_count,
            total_steps,
            path,
        )
        return record_count, total_steps

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    @staticmethod
    def read_manifest(directory: Path) -> TrajectoryManifest:
        """Read manifest.json from a trajectory directory."""
        return load_model(directory / MANIFEST_FILENAME, TrajectoryManifest)

    @staticmethod
    def iter_records(
        directory: Path,
        split_name: str,
    ) -> Iterator[TrajectoryRecord]:
        """Yield TrajectoryRecord objects one at a time from a split JSONL file.

        Records are deserialised and yielded individually.  Callers MUST
        process and discard each record before the next one is loaded to
        keep memory footprint bounded.
        """
        path = directory / _split_filename(split_name)
        if not path.exists():
            raise FileNotFoundError(
                f"split file not found: {path} — "
                f"expected split_name to be one of the names in manifest.split_names"
            )

        yielded = 0
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                record = TrajectoryRecord.model_validate_json(line)
                yield record
                del record
                yielded += 1
                if yielded % _GC_EVERY_N_RECORDS == 0:
                    gc.collect()

    @staticmethod
    def split_exists(directory: Path, split_name: str) -> bool:
        """Return True if the JSONL file for split_name exists and is non-empty."""
        path = directory / _split_filename(split_name)
        return path.exists() and path.stat().st_size > 0

    @staticmethod
    def is_trajectory_directory(path: Path) -> bool:
        """Return True if path looks like a streaming trajectory directory."""
        return path.is_dir() and (path / MANIFEST_FILENAME).exists()

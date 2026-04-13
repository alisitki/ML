"""trajectories/storage.py — Fixture and test-only in-memory trajectory store.

!!! FIXTURE / TEST COMPAT PATH ONLY !!!

TrajectoryStore loads the entire TrajectoryBundle into memory.
This is acceptable for small fixture datasets used in tests, but will
OOM for production-profile snapshots (tens of thousands of steps with
multi-scale tensor observations).

PRODUCTION PATH: use TrajectoryDirectoryStore (streaming_store.py).
    - build: builder.build_to_directory() → JSONL directory
    - train:  TrajectoryDirectoryStore.read_manifest() + .iter_records()
    - evaluate: TrajectoryDirectoryStore.iter_records("final_untouched_test")

This module is intentionally kept for backward compat with:
    - pytest fixtures (tests/conftest.py::trajectory_bundle)
    - CLI backward compat (--trajectories <file.json>)
    - Registry smoke tests
Do NOT expand its usage to production code paths.
"""
from __future__ import annotations

from pathlib import Path

from quantlab_ml.common import dump_model, load_model
from quantlab_ml.contracts import TrajectoryBundle

_FIXTURE_TEST_COMPAT_ONLY = True  # sentinel — grep-able marker


class TrajectoryStore:
    """In-memory trajectory read/write.

    FIXTURE / TEST COMPAT PATH — do not call from production build or training code.
    Production code must use TrajectoryDirectoryStore (streaming_store.py).
    """

    @staticmethod
    def write(path: Path, bundle: TrajectoryBundle) -> None:
        """Write TrajectoryBundle to a single JSON file.

        TEST / FIXTURE COMPAT — not for production-profile snapshots.
        """
        dump_model(path, bundle)

    @staticmethod
    def read(path: Path) -> TrajectoryBundle:
        """Read TrajectoryBundle from a single JSON file into memory.

        TEST / FIXTURE COMPAT — not for production-profile snapshots.
        """
        return load_model(path, TrajectoryBundle)

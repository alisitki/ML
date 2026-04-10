from __future__ import annotations

from pathlib import Path

from quantlab_ml.common import dump_model, load_model
from quantlab_ml.contracts import TrajectoryBundle


class TrajectoryStore:
    @staticmethod
    def write(path: Path, bundle: TrajectoryBundle) -> None:
        dump_model(path, bundle)

    @staticmethod
    def read(path: Path) -> TrajectoryBundle:
        return load_model(path, TrajectoryBundle)

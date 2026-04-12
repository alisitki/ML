from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel

ModelT = TypeVar("ModelT", bound=BaseModel)


def utcnow() -> datetime:
    return datetime.now(tz=UTC)


def configure_logging(level: str | None = None) -> None:
    configured_level = level if level is not None else os.getenv("QUANTLAB_ML_LOG_LEVEL", "INFO")
    resolved_level = configured_level.upper()
    numeric_level = getattr(logging, resolved_level, logging.INFO)
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
        root_logger.addHandler(handler)
    root_logger.setLevel(numeric_level)


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return raw


def load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key] = value
    return values


def dump_json_data(path: Path, data: Any) -> None:
    ensure_parent_dir(path)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def dump_model(path: Path, model: BaseModel) -> None:
    ensure_parent_dir(path)
    path.write_text(model.model_dump_json(indent=2), encoding="utf-8")


def load_model(path: Path, model_type: type[ModelT]) -> ModelT:
    return model_type.model_validate_json(path.read_text(encoding="utf-8"))


def hash_payload(payload: Any) -> str:
    if isinstance(payload, BaseModel):
        builtins = payload.model_dump(mode="json", exclude_none=False)
    else:
        builtins = payload
    serialized = json.dumps(builtins, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def current_code_commit_hash() -> str:
    repo_root = Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"

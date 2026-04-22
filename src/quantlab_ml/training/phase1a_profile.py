from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from quantlab_ml.common import ensure_parent_dir

Phase1AProfileStage = Literal["materialize", "train", "evaluate"]

_PROFILE_VERSION = "phase1a_profile_v1"
_SUMMARY_KEYS = (
    "materialization_wall_sec",
    "materialization_reused",
    "tensor_cache_used",
    "phase1a_supervision_used",
    "compiled_v2_eval_used",
    "jsonl_fallback_used",
    "candidate_wall_sec",
    "fold_wall_sec",
    "batch_assembly_wall_sec",
    "batch_compute_wall_sec",
    "batch_compute_share",
    "evaluation_rows_per_sec",
    "joint_ce_loss",
    "aux_value_loss_raw",
    "aux_value_loss_weighted",
    "total_loss",
    "action_logit_abs_max",
    "action_entropy",
    "value_pred_abs_max",
    "value_grad_norm_pre_clip",
    "value_grad_norm_post_clip",
    "clip_applied_count",
    "first_nonfinite_component",
    "first_nonfinite_batch_context",
)


def merge_phase1a_profile(
    profile_path: Path,
    stage: Phase1AProfileStage,
    payload: dict[str, object],
) -> None:
    resolved_path = profile_path.expanduser().resolve()
    ensure_parent_dir(resolved_path)
    if resolved_path.exists():
        current = json.loads(resolved_path.read_text(encoding="utf-8"))
    else:
        current = {
            "profile_version": _PROFILE_VERSION,
            "stages": {},
            "summary": {},
        }
    if current.get("profile_version") != _PROFILE_VERSION:
        raise ValueError(
            "phase1a_profile.json profile_version mismatch: "
            f"expected {_PROFILE_VERSION!r}, got {current.get('profile_version')!r}"
        )
    stages = dict(current.get("stages", {}))
    stages[stage] = dict(payload)
    merged = {
        "profile_version": _PROFILE_VERSION,
        "stages": stages,
        "summary": _build_summary(stages),
    }
    temp_path = resolved_path.with_suffix(f"{resolved_path.suffix}.tmp")
    temp_path.write_text(json.dumps(merged, indent=2, sort_keys=True), encoding="utf-8")
    temp_path.replace(resolved_path)


def _build_summary(stages: dict[str, dict[str, object]]) -> dict[str, object]:
    summary: dict[str, object] = {key: None for key in _SUMMARY_KEYS}
    materialize = stages.get("materialize", {})
    train = stages.get("train", {})
    evaluate = stages.get("evaluate", {})

    if "materialization_wall_sec" in materialize:
        summary["materialization_wall_sec"] = materialize["materialization_wall_sec"]
    if "materialization_reused" in materialize:
        summary["materialization_reused"] = materialize["materialization_reused"]
    summary["tensor_cache_used"] = _all_present_true(stages, "tensor_cache_used")
    summary["phase1a_supervision_used"] = _all_present_true(stages, "phase1a_supervision_used")
    if "compiled_v2_eval_used" in evaluate:
        summary["compiled_v2_eval_used"] = evaluate["compiled_v2_eval_used"]
    summary["jsonl_fallback_used"] = _any_present_true(stages, "jsonl_fallback_used")
    if "candidate_wall_sec" in train:
        summary["candidate_wall_sec"] = train["candidate_wall_sec"]
    if "fold_wall_sec" in train:
        summary["fold_wall_sec"] = train["fold_wall_sec"]
    if "batch_assembly_wall_sec" in train:
        summary["batch_assembly_wall_sec"] = train["batch_assembly_wall_sec"]
    if "batch_compute_wall_sec" in train:
        summary["batch_compute_wall_sec"] = train["batch_compute_wall_sec"]
    if "batch_compute_share" in train:
        summary["batch_compute_share"] = train["batch_compute_share"]
    if "evaluation_rows_per_sec" in evaluate:
        summary["evaluation_rows_per_sec"] = evaluate["evaluation_rows_per_sec"]
    for key in (
        "joint_ce_loss",
        "aux_value_loss_raw",
        "aux_value_loss_weighted",
        "total_loss",
        "action_logit_abs_max",
        "action_entropy",
        "value_pred_abs_max",
        "value_grad_norm_pre_clip",
        "value_grad_norm_post_clip",
        "clip_applied_count",
        "first_nonfinite_component",
        "first_nonfinite_batch_context",
    ):
        if key in train:
            summary[key] = train[key]
    return summary


def _all_present_true(stages: dict[str, dict[str, object]], key: str) -> bool | None:
    present = [stage[key] for stage in stages.values() if key in stage]
    if not present:
        return None
    return all(bool(value) for value in present)


def _any_present_true(stages: dict[str, dict[str, object]], key: str) -> bool | None:
    present = [stage[key] for stage in stages.values() if key in stage]
    if not present:
        return None
    return any(bool(value) for value in present)

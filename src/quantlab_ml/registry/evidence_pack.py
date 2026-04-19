from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from quantlab_ml.common import hash_payload, utcnow
from quantlab_ml.registry.audit import audit_registry_continuity
from quantlab_ml.registry.store import LocalRegistryStore

_ACTIVE_STATUSES = {"candidate", "challenger", "champion"}


def build_offline_evidence_pack(
    *,
    registry_roots: list[Path],
    inspected_evidence_kinds: list[str],
    authority_statuses: list[str | None],
) -> dict[str, Any]:
    if not registry_roots:
        raise ValueError("offline evidence pack requires at least one registry_root")
    if not (
        len(registry_roots) == len(inspected_evidence_kinds) == len(authority_statuses)
    ):
        raise ValueError("offline evidence pack inputs must align one-for-one by source")

    sources: list[dict[str, Any]] = []
    grouped_by_snapshot: dict[str, dict[str, list[str]]] = defaultdict(lambda: {"policy_ids": [], "registry_roots": []})
    grouped_by_surface: dict[str, dict[str, list[str]]] = defaultdict(lambda: {"policy_ids": [], "registry_roots": []})
    grouped_by_window: dict[str, dict[str, list[str]]] = defaultdict(lambda: {"policy_ids": [], "registry_roots": []})
    grouped_by_slice: dict[str, dict[str, list[str]]] = defaultdict(lambda: {"policy_ids": [], "registry_roots": []})

    for registry_root, inspected_evidence_kind, authority_status in zip(
        registry_roots,
        inspected_evidence_kinds,
        authority_statuses,
        strict=True,
    ):
        store = LocalRegistryStore(registry_root)
        audit = audit_registry_continuity(
            store,
            inspected_evidence_kind=inspected_evidence_kind,
            authority_status=authority_status,
        )
        records = store.list_records()
        active_records = [record for record in records if record.status in _ACTIVE_STATUSES]
        comparison_reports = store.list_comparison_reports()
        paper_sim_records = store.list_paper_sim_evidence()
        source_limitations = _source_limitations(
            audit=audit,
            active_records=active_records,
            comparison_report_count=len(comparison_reports),
            paper_sim_evidence_count=len(paper_sim_records),
        )
        source_summary = {
            "registry_root": str(registry_root),
            "inspected_evidence_kind": audit["inspected_evidence_kind"],
            "authority_status": audit["authority_status"],
            "audit_scope_verdict": audit["audit_scope_verdict"],
            "closeout_decision_allowed": audit["closeout_decision_allowed"],
            "record_count": len(records),
            "active_record_count": len(active_records),
            "scored_record_count": sum(1 for record in active_records if record.score_history),
            "comparison_report_count": len(comparison_reports),
            "paper_sim_evidence_count": len(paper_sim_records),
            "training_snapshot_ids": sorted({record.training_snapshot_id for record in active_records}),
            "evaluation_surface_ids": sorted({record.evaluation_surface_id for record in active_records}),
            "slice_ids": sorted({record.slice_id for record in active_records}),
            "train_windows": sorted({_format_range(record.train_window) for record in active_records}),
            "eval_windows": sorted(
                {_format_range(record.eval_window) for record in active_records if record.eval_window is not None}
            ),
            "missing_comparison_policy_ids": sorted(
                record.policy_id
                for record in active_records
                if record.status == "challenger" and record.score_history and record.comparison_report_id is None
            ),
            "missing_paper_sim_policy_ids": sorted(
                record.policy_id
                for record in active_records
                if record.score_history and record.paper_sim_evidence_id is None
            ),
            "limitations": source_limitations,
            "policy_records": [
                {
                    "policy_id": record.policy_id,
                    "status": record.status,
                    "training_snapshot_id": record.training_snapshot_id,
                    "evaluation_surface_id": record.evaluation_surface_id,
                    "slice_id": record.slice_id,
                    "train_window": _format_range(record.train_window),
                    "eval_window": _format_range(record.eval_window) if record.eval_window is not None else None,
                    "comparison_report_id": record.comparison_report_id,
                    "paper_sim_evidence_id": record.paper_sim_evidence_id,
                }
                for record in active_records
            ],
        }
        sources.append(source_summary)

        for record in active_records:
            _append_group_entry(grouped_by_snapshot[record.training_snapshot_id], record.policy_id, registry_root)
            _append_group_entry(grouped_by_surface[record.evaluation_surface_id], record.policy_id, registry_root)
            _append_group_entry(grouped_by_window[_format_range(record.train_window)], record.policy_id, registry_root)
            _append_group_entry(grouped_by_slice[record.slice_id], record.policy_id, registry_root)

    overall_limitations = sorted(
        {
            limitation
            for source in sources
            for limitation in source["limitations"]
        }
    )
    pack_payload = {
        "generated_at": utcnow().isoformat(),
        "pack_id": f"offline-evidence-pack-{hash_payload(sources)[:12]}",
        "source_count": len(sources),
        "sources": sources,
        "grouped_by_training_snapshot": _sorted_group_map(grouped_by_snapshot),
        "grouped_by_evaluation_surface": _sorted_group_map(grouped_by_surface),
        "grouped_by_train_window": _sorted_group_map(grouped_by_window),
        "grouped_by_slice": _sorted_group_map(grouped_by_slice),
        "overall_limitations": overall_limitations,
    }
    return pack_payload


def render_offline_evidence_pack_markdown(pack: dict[str, Any]) -> str:
    lines = [
        "# Offline Evidence Pack",
        "",
        f"- Generated at: `{pack['generated_at']}`",
        f"- Pack ID: `{pack['pack_id']}`",
        f"- Source count: `{pack['source_count']}`",
        "",
        "## Sources",
        "",
        "| Registry Root | Evidence Class | Authority | Audit Verdict | Active Policies | Comparison Reports | Paper/Sim Evidence |",
        "| --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for source in pack["sources"]:
        lines.append(
            "| "
            f"`{source['registry_root']}` | "
            f"`{source['inspected_evidence_kind']}` | "
            f"`{source['authority_status']}` | "
            f"`{source['audit_scope_verdict']}` | "
            f"{source['active_record_count']} | "
            f"{source['comparison_report_count']} | "
            f"{source['paper_sim_evidence_count']} |"
        )
    lines.extend(["", "## Grouped By Evaluation Surface", ""])
    lines.extend(_render_group_section(pack["grouped_by_evaluation_surface"]))
    lines.extend(["", "## Grouped By Training Snapshot", ""])
    lines.extend(_render_group_section(pack["grouped_by_training_snapshot"]))
    lines.extend(["", "## Grouped By Train Window", ""])
    lines.extend(_render_group_section(pack["grouped_by_train_window"]))
    lines.extend(["", "## Grouped By Slice", ""])
    lines.extend(_render_group_section(pack["grouped_by_slice"]))
    lines.extend(["", "## Limitations", ""])
    if pack["overall_limitations"]:
        for limitation in pack["overall_limitations"]:
            lines.append(f"- {limitation}")
    else:
        lines.append("- No additional pack-level limitations were detected.")
    return "\n".join(lines) + "\n"


def _append_group_entry(bucket: dict[str, list[str]], policy_id: str, registry_root: Path) -> None:
    if policy_id not in bucket["policy_ids"]:
        bucket["policy_ids"].append(policy_id)
    root_text = str(registry_root)
    if root_text not in bucket["registry_roots"]:
        bucket["registry_roots"].append(root_text)


def _sorted_group_map(grouped: dict[str, dict[str, list[str]]]) -> dict[str, dict[str, list[str]]]:
    return {
        key: {
            "policy_ids": sorted(value["policy_ids"]),
            "registry_roots": sorted(value["registry_roots"]),
        }
        for key, value in sorted(grouped.items())
    }


def _format_range(time_range: Any) -> str:
    return f"{time_range.start.isoformat()} -> {time_range.end.isoformat()}"


def _source_limitations(
    *,
    audit: dict[str, Any],
    active_records: list[Any],
    comparison_report_count: int,
    paper_sim_evidence_count: int,
) -> list[str]:
    limitations: list[str] = []
    if audit["inspected_evidence_kind"] != "authoritative_evidence":
        limitations.append(
            "This source remains non-authoritative evidence and must not be relabeled as authoritative evidence."
        )
    if audit["authority_status"] != "confirmed":
        limitations.append("Authority remains unconfirmed for this source.")
    if comparison_report_count == 0:
        limitations.append("No registry-backed comparison reports were found for this source.")
    if paper_sim_evidence_count == 0:
        limitations.append("No paper/sim evidence was linked for this source.")
    if any(record.status == "challenger" and record.score_history for record in active_records):
        limitations.append("Scored challengers still require explicit comparison and paper/sim linkage review.")
    return sorted(set(limitations))


def _render_group_section(grouped: dict[str, dict[str, list[str]]]) -> list[str]:
    if not grouped:
        return ["- No grouped entries found."]
    lines = [
        "| Key | Policy IDs | Registry Roots |",
        "| --- | --- | --- |",
    ]
    for key, value in grouped.items():
        policy_ids = ", ".join(f"`{policy_id}`" for policy_id in value["policy_ids"])
        registry_roots = ", ".join(f"`{registry_root}`" for registry_root in value["registry_roots"])
        lines.append(f"| `{key}` | {policy_ids} | {registry_roots} |")
    return lines

from __future__ import annotations

from pathlib import Path


LEGACY_DOC_STEMS = (
    ("arch", "itecture"),
    ("arti", "facts"),
    ("data", "-contract"),
    ("learning", "-surface"),
)


def _legacy_name(parts: tuple[str, str]) -> str:
    return "".join(parts) + ".md"


LEGACY_DOCS = tuple(str(Path("docs") / _legacy_name(parts)) for parts in LEGACY_DOC_STEMS)
LEGACY_REFERENCES = LEGACY_DOCS + tuple(_legacy_name(parts) for parts in LEGACY_DOC_STEMS)


def test_readme_and_canonical_docs_are_aligned(repo_root: Path) -> None:
    readme = (repo_root / "README.md").read_text(encoding="utf-8")
    project_state = (repo_root / "docs" / "PROJECT_STATE.md").read_text(encoding="utf-8")
    backlog = (repo_root / "docs" / "BACKLOG.md").read_text(encoding="utf-8")
    roadmap = (repo_root / "docs" / "ROADMAP.md").read_text(encoding="utf-8")
    docs_index = (repo_root / "docs" / "DOCS_INDEX.md").read_text(encoding="utf-8")
    runtime_model = (repo_root / "docs" / "ONLINE_RUNTIME_MODEL.md").read_text(encoding="utf-8")
    commercialization_gates = (
        repo_root / "docs" / "COMMERCIALIZATION_GATES.md"
    ).read_text(encoding="utf-8")
    offline_closure = (repo_root / "docs" / "OFFLINE_CLOSURE_CRITERIA.md").read_text(encoding="utf-8")
    continuity_runbook = (repo_root / "docs" / "CONTINUITY_AUDIT_RUNBOOK.md").read_text(encoding="utf-8")
    continuity_discovery_runbook = (
        repo_root / "docs" / "CONTINUITY_AUTHORITY_DISCOVERY_RUNBOOK.md"
    ).read_text(encoding="utf-8")
    closeout_records = (repo_root / "docs" / "CONTINUITY_CLOSEOUT_RECORDS.md").read_text(encoding="utf-8")
    remote_gpu_runbook = (repo_root / "docs" / "REMOTE_GPU_RUNBOOK.md").read_text(encoding="utf-8")
    continuity_authority_decision = (
        repo_root / "docs" / "history" / "2026Q2" / "CONTINUITY_AUTHORITY_DECISION.md"
    ).read_text(encoding="utf-8")
    continuity_discovery_record = (
        repo_root / "docs" / "history" / "2026Q2" / "CONTINUITY_AUTHORITY_DISCOVERY_RUN_2026-04-18.md"
    ).read_text(encoding="utf-8")
    minimum_evidence_pack = (
        repo_root / "docs" / "history" / "2026Q2" / "OFFLINE_CLOSURE_MINIMUM_EVIDENCE_PACK.md"
    ).read_text(encoding="utf-8")
    market_data_contract = (repo_root / "docs" / "CANONICAL_MARKET_DATA_CONTRACT.md").read_text(
        encoding="utf-8"
    )
    observation_schema = (repo_root / "docs" / "OBSERVATION_SCHEMA.md").read_text(encoding="utf-8")

    assert "QuantLab ML" in readme
    assert "configs/data/default.yaml" in readme
    assert "configs/data/fixture.yaml" in readme
    assert "configs/data/s3-current.yaml" in readme
    assert "configs/data/controlled-remote-day.yaml" in readme
    assert "configs/training/production.yaml" in readme
    assert "compare-policies" in readme
    assert "record-paper-sim" in readme
    assert "build-offline-evidence-pack" in readme
    assert "inspect-s3-compact" in readme
    assert "audit-continuity" in readme
    assert "docs/CANONICAL_MARKET_DATA_CONTRACT.md" in readme
    assert "docs/OBSERVATION_SCHEMA.md" in readme
    assert "docs/POLICY_ARTIFACT_SCHEMA.md" in readme
    assert "docs/EXECUTION_INTENT_SCHEMA.md" in readme
    assert "docs/REMOTE_GPU_RUNBOOK.md" in readme
    assert "docs/CONTINUITY_CLOSEOUT_RECORDS.md" in readme
    assert "## Current implemented scope" in readme
    assert "## Current closure verdict" in readme
    assert "## Not yet implemented as current repo reality" in readme
    assert "## Current focus before live/runtime work" in readme
    assert "## Next build phase" in readme
    assert "Current repository boundary" in readme
    assert "fresh authoritative evidence" in readme.lower()
    assert "broader offline evidence expansion" in readme.lower()
    assert "historical local authority-discovery loop" in readme.lower()
    assert "registry-backed comparison reports" in readme.lower()
    assert "offline evidence-pack summaries" in readme.lower()

    assert "## Current phase" in project_state
    assert "## Current verdict" in project_state
    assert "## Current implemented strengths" in project_state
    assert "## Current missing layers" in project_state
    assert "## Current focus" in project_state
    assert "## Blocked before live-path focus" in project_state
    assert "## Not started / not main focus yet" in project_state
    assert "minimum offline-closure evidence pack" in project_state.lower()
    assert "outputs/registry" in project_state
    assert "historical authoritative continuity roots are unavailable in this workspace" in project_state.lower()
    assert "ql-031" in project_state.lower()
    assert "single active next batch in this workspace" in project_state.lower()
    assert "comparison-report and offline evidence-pack tooling exist" in project_state.lower()

    assert "ql-016" in backlog.lower()
    assert "ql-031" in backlog.lower()
    assert "single active next batch in this workspace" in backlog.lower()
    assert "fresh authoritative evidence" in backlog.lower()
    assert "compare-policies" in backlog
    assert "build-offline-evidence-pack" in backlog
    assert "discover_continuity_authority.py --search-root" not in backlog

    assert "## Target runtime architecture" in runtime_model
    assert "## Current implemented runtime-related surface" in runtime_model
    assert "## Missing components before QuantLab has a live-operating runtime" in runtime_model
    assert "## Next implementation steps" in runtime_model

    assert "## Target gates vs. current evidence" in commercialization_gates
    assert "| Gate | Target meaning | Current status | Missing evidence |" in commercialization_gates
    assert "offline closure criteria" in commercialization_gates.lower()

    assert "## Phase 2 - Runtime / Live Parity Foundation" in roadmap
    assert "## Phase 3 - Shadow / Paper + Thin Executor Operating Loop" in roadmap
    assert "Entry criteria:" in roadmap
    assert "Exit criteria:" in roadmap

    assert "## Authority map" in docs_index
    assert "docs/OFFLINE_CLOSURE_CRITERIA.md" in docs_index
    assert "docs/CONTINUITY_AUDIT_RUNBOOK.md" in docs_index
    assert "docs/CONTINUITY_AUTHORITY_DISCOVERY_RUNBOOK.md" in docs_index
    assert "docs/CONTINUITY_CLOSEOUT_RECORDS.md" in docs_index

    assert "PASS" in offline_closure
    assert "PARTIAL" in offline_closure
    assert "FAIL" in offline_closure
    assert "artifact / registry / compatibility truth" in offline_closure
    assert "repo-tracked artifact" in offline_closure.lower()
    assert "external retained evidence" in offline_closure.lower()
    assert "authoritative evidence" in offline_closure.lower()

    assert "RETIRE" in continuity_runbook
    assert "FREEZE" in continuity_runbook
    assert "KEEP-TEMPORARY-WITH-EXPLICIT-SCOPE" in continuity_runbook
    assert "closeout_decision_allowed" in continuity_runbook
    assert "zero active records" in continuity_runbook.lower()
    assert "authoritative registry root is unknown" in continuity_runbook.lower()
    assert "broken or non-relocatable retained artifact paths" in continuity_runbook.lower()
    assert "repo-tracked artifact" in continuity_runbook.lower()
    assert "external retained evidence" in continuity_runbook.lower()
    assert "authoritative evidence" in continuity_runbook.lower()
    assert "do not relabel the retained bundle as `authoritative_evidence`" in continuity_runbook

    assert "/workspace/runs/*/registry" in continuity_discovery_runbook
    assert "/root/runs/*/registry" in continuity_discovery_runbook
    assert "0 eligible external candidates => blocked" in continuity_discovery_runbook
    assert ">1 eligible external candidates => blocked as ambiguity" in continuity_discovery_runbook
    assert "1 eligible external candidate => authority confirmation step allowed" in (
        continuity_discovery_runbook
    )
    assert "retained_bundle_only" in continuity_discovery_runbook
    assert "must not run `authoritative-evidence` reruns" in continuity_discovery_runbook

    assert "decision_status" in closeout_records
    assert "pending_authoritative_evidence" in closeout_records
    assert "repo-tracked artifact" in closeout_records.lower()
    assert "external retained evidence" in closeout_records.lower()
    assert "authoritative evidence" in closeout_records.lower()

    assert "retained evidence" in remote_gpu_runbook.lower()
    assert "audit-continuity" in remote_gpu_runbook
    assert "quantlab-ml compare-policies" in (repo_root / "docs" / "EVALUATION_RUNBOOK.md").read_text(encoding="utf-8")
    assert "comparison report record" in (repo_root / "docs" / "REGISTRY_SCHEMA.md").read_text(encoding="utf-8")

    assert "only a confirmed external active registry root" in continuity_authority_decision.lower()
    assert "remains `external_retained_evidence` only" in continuity_authority_decision
    assert "no rerun of `audit-continuity` may use `authoritative-evidence` on the retained bundle alone" in (
        continuity_authority_decision.lower()
    )
    assert "do not continue the historical external-root discovery loop" in continuity_authority_decision.lower()
    assert "broader offline evidence expansion" in continuity_authority_decision.lower()

    assert "blocked_no_eligible_external_candidates" in continuity_discovery_record
    assert "/workspace/runs" in continuity_discovery_record
    assert "/root/runs" in continuity_discovery_record
    assert "authoritative rerun command/output" in continuity_discovery_record.lower()
    assert "not run because zero eligible external candidates were found" in continuity_discovery_record.lower()
    assert "outputs/analysis/continuity-authority-discovery-2026-04-18.json" in continuity_discovery_record
    assert "no mounted external run roots were visible" in continuity_discovery_record.lower()
    assert "closes the historical local authority-discovery loop" in continuity_discovery_record.lower()
    assert "broader offline evidence expansion" in continuity_discovery_record.lower()
    assert (
        "authoritative scope still blocked; closeout remains pending with sharper blockers"
        in continuity_discovery_record
    )

    assert "Sprint closeout sentence" in minimum_evidence_pack
    assert "clear_in_inspected_scope" in minimum_evidence_pack
    assert "comparison_report_id" in minimum_evidence_pack
    assert "multi-window or multi-slice" in minimum_evidence_pack
    assert "must remain `external_retained_evidence`" in minimum_evidence_pack
    assert "fresh authoritative evidence" in minimum_evidence_pack.lower()
    assert "ql-031" in minimum_evidence_pack.lower()

    assert "fields` as the primary field carrier" in market_data_contract
    assert "compacted/_state.json" in market_data_contract
    assert "exchange=<exchange>/stream=<stream>/symbol=<symbol>/date=<YYYYMMDD>/data.parquet" in (
        market_data_contract
    )
    assert "explicit contract override wins" in market_data_contract
    assert "contract-unavailable (availability_by_contract = False)" in observation_schema


def test_legacy_docs_are_deleted_and_unreferenced(repo_root: Path) -> None:
    for rel_path in LEGACY_DOCS:
        assert not (repo_root / rel_path).exists(), f"{rel_path} should be deleted"

    surviving_docs = [
        repo_root / "AGENTS.md",
        repo_root / "README.md",
        *sorted((repo_root / "docs").glob("*.md")),
    ]

    for path in surviving_docs:
        text = path.read_text(encoding="utf-8")
        for legacy_ref in LEGACY_REFERENCES:
            assert legacy_ref not in text, f"{path} still references {legacy_ref}"

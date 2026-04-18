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
    roadmap = (repo_root / "docs" / "ROADMAP.md").read_text(encoding="utf-8")
    docs_index = (repo_root / "docs" / "DOCS_INDEX.md").read_text(encoding="utf-8")
    runtime_model = (repo_root / "docs" / "ONLINE_RUNTIME_MODEL.md").read_text(encoding="utf-8")
    commercialization_gates = (
        repo_root / "docs" / "COMMERCIALIZATION_GATES.md"
    ).read_text(encoding="utf-8")
    offline_closure = (repo_root / "docs" / "OFFLINE_CLOSURE_CRITERIA.md").read_text(encoding="utf-8")
    continuity_runbook = (repo_root / "docs" / "CONTINUITY_AUDIT_RUNBOOK.md").read_text(encoding="utf-8")
    remote_gpu_runbook = (repo_root / "docs" / "REMOTE_GPU_RUNBOOK.md").read_text(encoding="utf-8")
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
    assert "inspect-s3-compact" in readme
    assert "audit-continuity" in readme
    assert "docs/CANONICAL_MARKET_DATA_CONTRACT.md" in readme
    assert "docs/OBSERVATION_SCHEMA.md" in readme
    assert "docs/POLICY_ARTIFACT_SCHEMA.md" in readme
    assert "docs/EXECUTION_INTENT_SCHEMA.md" in readme
    assert "docs/REMOTE_GPU_RUNBOOK.md" in readme
    assert "## Current implemented scope" in readme
    assert "## Current closure verdict" in readme
    assert "## Not yet implemented as current repo reality" in readme
    assert "## Current focus before live/runtime work" in readme
    assert "## Next build phase" in readme
    assert "Current repository boundary" in readme

    assert "## Current phase" in project_state
    assert "## Current verdict" in project_state
    assert "## Current implemented strengths" in project_state
    assert "## Current missing layers" in project_state
    assert "## Current focus" in project_state
    assert "## Blocked before live-path focus" in project_state
    assert "## Not started / not main focus yet" in project_state

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

    assert "PASS" in offline_closure
    assert "PARTIAL" in offline_closure
    assert "FAIL" in offline_closure
    assert "artifact / registry / compatibility truth" in offline_closure

    assert "RETIRE" in continuity_runbook
    assert "FREEZE" in continuity_runbook
    assert "KEEP-TEMPORARY-WITH-EXPLICIT-SCOPE" in continuity_runbook
    assert "zero active records" in continuity_runbook.lower()
    assert "authoritative registry root is unknown" in continuity_runbook.lower()
    assert "broken or non-relocatable retained artifact paths" in continuity_runbook.lower()

    assert "retained evidence" in remote_gpu_runbook.lower()
    assert "audit-continuity" in remote_gpu_runbook

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

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
    runtime_model = (repo_root / "docs" / "ONLINE_RUNTIME_MODEL.md").read_text(encoding="utf-8")
    commercialization_gates = (
        repo_root / "docs" / "COMMERCIALIZATION_GATES.md"
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
    assert "inspect-s3-compact" in readme
    assert "audit-continuity" in readme
    assert "docs/CANONICAL_MARKET_DATA_CONTRACT.md" in readme
    assert "docs/OBSERVATION_SCHEMA.md" in readme
    assert "docs/POLICY_ARTIFACT_SCHEMA.md" in readme
    assert "docs/EXECUTION_INTENT_SCHEMA.md" in readme
    assert "docs/REMOTE_GPU_RUNBOOK.md" in readme
    assert "## Current implemented scope" in readme
    assert "## Not yet implemented as current repo reality" in readme
    assert "## Next build phase" in readme
    assert "Current repository boundary" in readme

    assert "## Current phase" in project_state
    assert "## Current implemented strengths" in project_state
    assert "## Current missing layers" in project_state
    assert "## Next phase after active focus" in project_state

    assert "## Target runtime architecture" in runtime_model
    assert "## Current implemented runtime-related surface" in runtime_model
    assert "## Missing components before QuantLab has a live-operating runtime" in runtime_model
    assert "## Next implementation steps" in runtime_model

    assert "## Target gates vs. current evidence" in commercialization_gates
    assert "| Gate | Target meaning | Current status | Missing evidence |" in commercialization_gates

    assert "## Phase 2 - Runtime/live parity foundation" in roadmap

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

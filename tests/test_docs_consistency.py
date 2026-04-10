from __future__ import annotations

from pathlib import Path


def test_docs_and_readme_match_profile_split(repo_root: Path) -> None:
    readme = (repo_root / "README.md").read_text(encoding="utf-8")
    data_contract = (repo_root / "docs" / "data-contract.md").read_text(encoding="utf-8")
    architecture = (repo_root / "docs" / "architecture.md").read_text(encoding="utf-8")

    assert "QuantLab ML" in readme
    assert "configs/data/default.yaml" in readme
    assert "configs/data/fixture.yaml" in readme
    assert "configs/data/s3-current.yaml" in readme
    assert "inspect-s3-compact" in readme
    assert "binance | yes | yes | yes | yes | no" in data_contract
    assert "compacted/_state.json" in data_contract
    assert "exchange=<exchange>/stream=<stream>/symbol=<symbol>/date=<YYYYMMDD>/data.parquet" in data_contract
    assert "metadata-first" in architecture

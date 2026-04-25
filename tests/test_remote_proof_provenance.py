from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_provenance_module(repo_root: Path):
    script_path = repo_root / "scripts" / "remote_proof_provenance.py"
    spec = importlib.util.spec_from_file_location("remote_proof_provenance", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_provenance_fails_current_head_mismatch_without_override(
    repo_root: Path,
    tmp_path: Path,
) -> None:
    provenance = _load_provenance_module(repo_root)
    runner = tmp_path / "run_ql033_r6.sh"
    validator = tmp_path / "validate_ql033_r4.py"
    runner.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    validator.write_text("print('validator')\n", encoding="utf-8")

    report = provenance.build_provenance_report(
        repo_root=repo_root,
        expected_local_head="local-head",
        remote_head="remote-head",
        baseline_import_proof_commit="baseline-head",
        runner_script=runner,
        validator_script=validator,
    )

    assert report["status"] == "fail"
    assert "current_local_remote_head_mismatch" in report["blocking_reasons"]
    assert report["current_local_head_expected"] == "local-head"
    assert report["current_remote_head_actual"] == "remote-head"
    assert report["baseline_import_proof_commit"] == "baseline-head"
    assert report["baseline_import_proof_is_current_code_provenance"] is False


def test_provenance_records_override_and_script_checksums(
    repo_root: Path,
    tmp_path: Path,
) -> None:
    provenance = _load_provenance_module(repo_root)
    runner = tmp_path / "run_ql033_r6.sh"
    validator = tmp_path / "validate_ql033_r4.py"
    runner.write_text("#!/usr/bin/env bash\necho run\n", encoding="utf-8")
    validator.write_text("print('validator')\n", encoding="utf-8")

    report = provenance.build_provenance_report(
        repo_root=repo_root,
        expected_local_head="local-head",
        remote_head="remote-head",
        baseline_import_proof_commit="baseline-head",
        runner_script=runner,
        validator_script=validator,
        allow_head_mismatch=True,
        override_reason="operator approved testing fork",
    )

    assert report["status"] == "pass"
    assert report["override"]["override_recorded"] is True
    assert report["runner_script"]["sha256"]
    assert report["validator_script"]["sha256"]
    assert report["runner_script"]["sha256"] != report["validator_script"]["sha256"]

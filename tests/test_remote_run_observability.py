from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


def test_remote_run_stage_writes_started_line_immediately(repo_root: Path, tmp_path: Path) -> None:
    script_path = repo_root / "scripts" / "remote_run_stage.py"
    log_path = tmp_path / "build.log"
    exit_path = tmp_path / "build.exit"
    process = subprocess.Popen(
        [
            sys.executable,
            str(script_path),
            "--run-id",
            "ql031-observability",
            "--phase",
            "BUILD_STARTED",
            "--log",
            str(log_path),
            "--exit-file",
            str(exit_path),
            "--heartbeat-interval-seconds",
            "0.5",
            "--",
            sys.executable,
            "-c",
            "import time; time.sleep(1.5)",
        ],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        deadline = time.time() + 2.0
        first_line = None
        while time.time() < deadline:
            if log_path.exists():
                lines = log_path.read_text(encoding="utf-8").splitlines()
                if lines:
                    first_line = lines[0]
                    break
            time.sleep(0.05)
        assert first_line is not None
        assert first_line.startswith("[STARTED]")
        assert "run_id=ql031-observability" in first_line
    finally:
        process.wait(timeout=10)
    assert exit_path.read_text(encoding="utf-8").strip() == "0"


def test_remote_run_stage_emits_heartbeat_and_completion_markers(repo_root: Path, tmp_path: Path) -> None:
    script_path = repo_root / "scripts" / "remote_run_stage.py"
    log_path = tmp_path / "train.log"
    exit_path = tmp_path / "train.exit"
    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--run-id",
            "ql031-heartbeat",
            "--phase",
            "TRAIN_STARTED",
            "--log",
            str(log_path),
            "--exit-file",
            str(exit_path),
            "--heartbeat-interval-seconds",
            "0.5",
            "--",
            sys.executable,
            "-c",
            "import time; time.sleep(1.2); print('done')",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    payload = log_path.read_text(encoding="utf-8")
    assert "[STARTED]" in payload
    assert "[TRAIN_STARTED]" in payload
    assert "[HEARTBEAT]" in payload
    assert "[COMPLETED]" in payload
    assert exit_path.read_text(encoding="utf-8").strip() == "0"


def test_remote_run_stage_emits_failed_marker(repo_root: Path, tmp_path: Path) -> None:
    script_path = repo_root / "scripts" / "remote_run_stage.py"
    log_path = tmp_path / "evaluate.log"
    exit_path = tmp_path / "evaluate.exit"
    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--run-id",
            "ql031-failure",
            "--phase",
            "EVAL_STARTED",
            "--log",
            str(log_path),
            "--exit-file",
            str(exit_path),
            "--",
            sys.executable,
            "-c",
            "import sys; sys.exit(3)",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 3
    payload = log_path.read_text(encoding="utf-8")
    assert "[STARTED]" in payload
    assert "[EVAL_STARTED]" in payload
    assert "[FAILED]" in payload
    assert exit_path.read_text(encoding="utf-8").strip() == "3"


def test_remote_run_stage_reemits_progress_markers(repo_root: Path, tmp_path: Path) -> None:
    script_path = repo_root / "scripts" / "remote_run_stage.py"
    log_path = tmp_path / "materialize.log"
    exit_path = tmp_path / "materialize.exit"
    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--run-id",
            "ql031-progress",
            "--phase",
            "MATERIALIZE_STARTED",
            "--log",
            str(log_path),
            "--exit-file",
            str(exit_path),
            "--",
            sys.executable,
            "-c",
            "print('[PROGRESS] marker=materialization_completed rows=42')",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    payload = log_path.read_text(encoding="utf-8")
    assert "[PROGRESS] marker=materialization_completed rows=42" in payload
    assert "run_id=ql031-progress" in payload
    assert "phase=MATERIALIZE_STARTED" in payload
    assert exit_path.read_text(encoding="utf-8").strip() == "0"

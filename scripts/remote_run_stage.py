#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

_PROGRESS_MARKER_RE = re.compile(r"\[PROGRESS\]\s+marker=(?P<marker>[A-Za-z0-9_.-]+)(?P<rest>.*)")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wrap a remote run stage with immediate log creation, explicit start markers, "
            "heartbeats, unbuffered child execution, and terminal/log streaming."
        )
    )
    parser.add_argument("--log", type=Path, required=True, help="Stage log path.")
    parser.add_argument("--run-id", required=True, help="Run identifier written into log markers.")
    parser.add_argument("--phase", required=True, help="Stage marker, e.g. BUILD_STARTED or EVAL_STARTED.")
    parser.add_argument(
        "--exit-file",
        type=Path,
        default=None,
        help="Optional exit-code sidecar path.",
    )
    parser.add_argument(
        "--heartbeat-interval-seconds",
        type=float,
        default=90.0,
        help="Heartbeat interval during silent phases.",
    )
    parser.add_argument(
        "--cwd",
        type=Path,
        default=None,
        help="Optional working directory for the child process.",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command to execute after '--'.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("remote_run_stage requires a command after '--'")

    log_path = args.log.expanduser().resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start_wall = _timestamp()
    start_monotonic = time.monotonic()
    activity_lock = threading.Lock()
    last_output = {"value": start_monotonic}
    host = socket.gethostname()
    phase = args.phase
    exit_file = args.exit_file.expanduser().resolve() if args.exit_file is not None else None

    prepared_command = _prepare_command(command)
    child_env = os.environ.copy()
    child_env["PYTHONUNBUFFERED"] = "1"

    with log_path.open("a", encoding="utf-8", buffering=1) as log_handle:
        process: subprocess.Popen[str] | None = None
        stop_heartbeat = threading.Event()
        heartbeat_thread: threading.Thread | None = None
        try:
            process = subprocess.Popen(
                prepared_command,
                cwd=str(args.cwd.expanduser().resolve()) if args.cwd is not None else None,
                env=child_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            _emit(
                log_handle,
                (
                    f"[STARTED] ts={start_wall} run_id={args.run_id} host={host} "
                    f"pid={process.pid} command={shlex.join(command)}"
                ),
            )
            _emit(
                log_handle,
                f"[{phase}] ts={_timestamp()} run_id={args.run_id} host={host} pid={process.pid} elapsed_s=0",
            )
            heartbeat_thread = threading.Thread(
                target=_heartbeat_loop,
                args=(
                    log_handle,
                    activity_lock,
                    last_output,
                    stop_heartbeat,
                    args.heartbeat_interval_seconds,
                    args.run_id,
                    phase,
                    start_monotonic,
                ),
                daemon=True,
            )
            heartbeat_thread.start()
            assert process.stdout is not None
            for line in process.stdout:
                with activity_lock:
                    last_output["value"] = time.monotonic()
                stripped = line.rstrip("\n")
                _emit(log_handle, stripped)
                progress_match = _PROGRESS_MARKER_RE.search(stripped)
                if progress_match is not None:
                    marker = progress_match.group("marker")
                    rest = progress_match.group("rest").strip()
                    suffix = f" {rest}" if rest else ""
                    _emit(
                        log_handle,
                        (
                            f"[PROGRESS] ts={_timestamp()} run_id={args.run_id} phase={phase} "
                            f"elapsed_s={_elapsed_seconds(start_monotonic)} marker={marker}{suffix}"
                        ),
                    )
            return_code = process.wait()
        except FileNotFoundError as exc:
            return_code = 127
            _emit(
                log_handle,
                (
                    f"[FAILED] ts={_timestamp()} run_id={args.run_id} host={host} "
                    f"phase={phase} elapsed_s={_elapsed_seconds(start_monotonic)} "
                    f"exit_code={return_code} detail={exc}"
                ),
            )
            _write_exit_file(exit_file, return_code)
            return return_code
        except KeyboardInterrupt:
            return_code = 130
            if process is not None and process.poll() is None:
                process.terminate()
            _emit(
                log_handle,
                (
                    f"[FAILED] ts={_timestamp()} run_id={args.run_id} host={host} "
                    f"phase={phase} elapsed_s={_elapsed_seconds(start_monotonic)} "
                    f"exit_code={return_code} detail=keyboard_interrupt"
                ),
            )
            _write_exit_file(exit_file, return_code)
            return return_code
        finally:
            stop_heartbeat.set()
            if heartbeat_thread is not None:
                heartbeat_thread.join(timeout=max(args.heartbeat_interval_seconds, 1.0) + 1.0)

        marker = "COMPLETED" if return_code == 0 else "FAILED"
        _emit(
            log_handle,
            (
                f"[{marker}] ts={_timestamp()} run_id={args.run_id} host={host} "
                f"phase={phase} elapsed_s={_elapsed_seconds(start_monotonic)} exit_code={return_code}"
            ),
        )
        _write_exit_file(exit_file, return_code)
        return return_code


def _prepare_command(command: list[str]) -> list[str]:
    normalized = list(command)
    executable = Path(normalized[0]).name.lower()
    if executable.startswith("python") and "-u" not in normalized[1:]:
        normalized.insert(1, "-u")
    if shutil.which("stdbuf") is not None and normalized[0] != "stdbuf":
        normalized = ["stdbuf", "-oL", "-eL", *normalized]
    return normalized


def _heartbeat_loop(
    log_handle,
    activity_lock: threading.Lock,
    last_output: dict[str, float],
    stop_heartbeat: threading.Event,
    interval_seconds: float,
    run_id: str,
    phase: str,
    start_monotonic: float,
) -> None:
    while not stop_heartbeat.wait(interval_seconds):
        now = time.monotonic()
        with activity_lock:
            silent_for = now - last_output["value"]
        if silent_for < interval_seconds:
            continue
        _emit(
            log_handle,
            (
                f"[HEARTBEAT] ts={_timestamp()} run_id={run_id} "
                f"phase={phase} elapsed_s={_elapsed_seconds(start_monotonic)}"
            ),
        )


def _write_exit_file(path: Path | None, exit_code: int) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{exit_code}\n", encoding="utf-8")


def _timestamp() -> str:
    return datetime.now(tz=UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _elapsed_seconds(start_monotonic: float) -> int:
    return int(round(time.monotonic() - start_monotonic))


def _emit(log_handle, line: str) -> None:
    if not line:
        return
    log_handle.write(f"{line}\n")
    log_handle.flush()
    print(line, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())

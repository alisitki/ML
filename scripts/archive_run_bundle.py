#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ARCHIVE_BASE_URI = "s3://quantlab-archive/quantlab"
DEFAULT_MAX_THIN_FILE_BYTES = 10 * 1024 * 1024

DENYLIST_DIR_NAMES = {
    ".aws",
    ".cache",
    ".config",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".ssh",
    ".venv",
    "__pycache__",
}
DENYLIST_FILE_NAMES = {
    ".DS_Store",
    ".env",
}
DENYLIST_FILE_PATTERNS = (
    "*.key",
    "*.pem",
    "id_*",
)
THIN_KEEP_NAMES = {
    "SHA256SUMS",
    "acceptance_evidence.json",
    "archive_manifest.json",
    "archive_receipt.json",
    "bundle_file_inventory.txt",
    "bundle_manifest.json",
    "bundle_size_summary.txt",
    "continuity_audit_authoritative.json",
    "continuity_authority_discovery.json",
    "disk_inventory.txt",
    "evaluation.json",
    "golden_raw_window_audit.jsonl",
    "inspect_s3.json",
    "local_prune_receipt.json",
    "manifest.json",
    "normalization_receipt.json",
    "offline_evidence_pack.json",
    "offline_evidence_pack.md",
    "ql031_status.json",
    "retained_root_discovery.json",
    "score.json",
    "stage-a-summary.json",
    "validation_report.json",
    "validation_report.md",
    "workspace_blocker_inventory.json",
    "workspace_blocker_inventory.md",
    "workspace_plus_diagnostic_blocker_inventory.json",
    "workspace_plus_diagnostic_blocker_inventory.md",
}
THIN_KEEP_SUFFIXES = (
    ".csv",
    ".exit",
    ".log",
    ".md",
    ".summary.json",
    ".txt",
    ".yaml",
    ".yml",
)
HEAVY_SUFFIXES = (
    ".jsonl",
    ".npy",
    ".npz",
    ".parquet",
    ".pt",
)


@dataclass(frozen=True)
class FileRecord:
    path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class ArchivePlan:
    source_root: Path
    size_bytes: int
    retained_class: str
    replayable: bool
    destination_prefix: str
    file_count: int
    denied_entries: tuple[str, ...]
    tracked_entries: tuple[str, ...]
    thin_keep_files: tuple[str, ...]
    prune_candidate_files: tuple[str, ...]
    prune_candidate_bytes: int

    @property
    def blocked(self) -> bool:
        return bool(self.denied_entries or self.tracked_entries)

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_root": str(self.source_root),
            "size_bytes": self.size_bytes,
            "size_human": human_size(self.size_bytes),
            "retained_class": self.retained_class,
            "replayable": self.replayable,
            "destination_prefix": self.destination_prefix,
            "file_count": self.file_count,
            "classification": "blocked" if self.blocked else "archive_then_prune",
            "blocked_denylisted_entries": list(self.denied_entries),
            "blocked_tracked_entries": list(self.tracked_entries),
            "thin_local_mirror": {
                "file_count": len(self.thin_keep_files),
                "files": list(self.thin_keep_files),
                "sample": list(self.thin_keep_files[:20]),
            },
            "proposed_prune": {
                "file_count": len(self.prune_candidate_files),
                "size_bytes": self.prune_candidate_bytes,
                "size_human": human_size(self.prune_candidate_bytes),
                "files": list(self.prune_candidate_files),
                "sample": list(self.prune_candidate_files[:20]),
            },
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inventory and archive QuantLab run/proof roots to S3. The default mode is a "
            "dry-run inventory; --execute is required before any upload or receipt write."
        )
    )
    parser.add_argument(
        "--source-root",
        action="append",
        type=Path,
        default=[],
        help="Run/proof root to inventory or archive. Repeat for multiple roots.",
    )
    parser.add_argument(
        "--inventory-root",
        type=Path,
        default=None,
        help="Discover local output candidates under this root, usually outputs.",
    )
    parser.add_argument(
        "--archive-base-uri",
        default=DEFAULT_ARCHIVE_BASE_URI,
        help="S3 base URI. Defaults to s3://quantlab-archive/quantlab.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help=(
            "Optional env file with S3_ARCHIVE_*, S3_COMPACT_*, or AWS_* credentials. "
            "Dedicated S3_ARCHIVE_* credentials are preferred; shared compact credentials "
            "are allowed only if the quantlab-archive verification check succeeds."
        ),
    )
    parser.add_argument(
        "--verify-credentials",
        action="store_true",
        help="Run a non-mutating S3 bucket/list credential check before inventory/upload.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Upload and write local receipts. Omit for dry-run inventory only.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path for the dry-run or execution report. Not written unless provided.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.env_file is not None:
        load_env_file(args.env_file)

    credential_status: dict[str, Any] | None = None
    if args.verify_credentials:
        credential_status = verify_archive_credentials(args.archive_base_uri)
        if not credential_status["ok"]:
            report = {
                "mode": "execute" if args.execute else "dry_run",
                "archive_base_uri": args.archive_base_uri,
                "credential_status": credential_status,
                "plans": [],
                "blocked": True,
            }
            emit_report(report, args.output_json)
            return 2

    source_roots = list(args.source_root)
    if args.inventory_root is not None:
        source_roots.extend(discover_inventory_roots(args.inventory_root, repo_root=REPO_ROOT))
    if not source_roots:
        raise SystemExit("provide --source-root or --inventory-root")

    plans = [
        build_archive_plan(
            source_root=root,
            archive_base_uri=args.archive_base_uri,
            repo_root=REPO_ROOT,
        )
        for root in source_roots
    ]
    report = {
        "mode": "execute" if args.execute else "dry_run",
        "archive_base_uri": args.archive_base_uri,
        "credential_status": credential_status or {"ok": None, "checked": False},
        "plans": [plan.as_dict() for plan in plans],
        "blocked": any(plan.blocked for plan in plans),
    }
    if not args.execute:
        emit_report(report, args.output_json)
        return 0

    blocked = [plan for plan in plans if plan.blocked]
    if blocked:
        report["execution_error"] = "refusing upload because one or more roots are blocked"
        emit_report(report, args.output_json)
        return 3
    if credential_status is None:
        report["execution_error"] = "--execute requires --verify-credentials"
        emit_report(report, args.output_json)
        return 4

    execution_results = [archive_plan(plan) for plan in plans]
    report["execution_results"] = execution_results
    emit_report(report, args.output_json)
    return 0


def load_env_file(path: Path) -> None:
    resolved = path.expanduser().resolve()
    if resolved.name == ".env":
        # The file itself is still never archived. This only lets operators load credentials.
        pass
    if not resolved.exists():
        raise FileNotFoundError(f"env file does not exist: {resolved}")
    for raw_line in resolved.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def verify_archive_credentials(archive_base_uri: str) -> dict[str, Any]:
    try:
        import boto3
        from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError
    except ImportError as exc:
        return {"ok": False, "checked": True, "error": f"boto3 unavailable: {exc}"}

    bucket, prefix = parse_s3_uri(archive_base_uri)
    credential_resolution = resolve_s3_client_config()
    if credential_resolution.get("error"):
        return {
            "ok": False,
            "checked": True,
            "bucket": bucket,
            "credential_source": credential_resolution["credential_source"],
            "error": credential_resolution["error"],
        }
    try:
        client = build_s3_client(boto3, credential_resolution)
        client.head_bucket(Bucket=bucket)
        client.list_objects_v2(Bucket=bucket, Prefix=prefix.rstrip("/") + "/", MaxKeys=1)
    except (NoCredentialsError, BotoCoreError, ClientError) as exc:
        return {
            "ok": False,
            "checked": True,
            "bucket": bucket,
            "credential_source": credential_resolution["credential_source"],
            "error": str(exc),
        }
    return {
        "ok": True,
        "checked": True,
        "bucket": bucket,
        "prefix": prefix,
        "credential_source": credential_resolution["credential_source"],
    }


def resolve_s3_client_config() -> dict[str, Any]:
    archive_access = os.environ.get("S3_ARCHIVE_ACCESS_KEY")
    archive_secret = os.environ.get("S3_ARCHIVE_SECRET_KEY")
    archive_session = os.environ.get("S3_ARCHIVE_SESSION_TOKEN")
    archive_region = os.environ.get("S3_ARCHIVE_REGION")
    archive_endpoint = os.environ.get("S3_ARCHIVE_ENDPOINT")

    compact_access = os.environ.get("S3_COMPACT_ACCESS_KEY")
    compact_secret = os.environ.get("S3_COMPACT_SECRET_KEY")
    compact_session = os.environ.get("S3_COMPACT_SESSION_TOKEN")
    compact_region = os.environ.get("S3_COMPACT_REGION")
    compact_endpoint = os.environ.get("S3_COMPACT_ENDPOINT")

    aws_region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")

    if archive_access or archive_secret:
        if not archive_access or not archive_secret:
            return {
                "credential_source": "S3_ARCHIVE",
                "kwargs": {},
                "error": "partial S3_ARCHIVE credentials; both access and secret key are required",
            }
        return {
            "credential_source": "S3_ARCHIVE",
            "kwargs": _client_kwargs(
                access_key=archive_access,
                secret_key=archive_secret,
                session_token=archive_session,
                region=archive_region or aws_region,
                endpoint=archive_endpoint,
            ),
        }

    if compact_access or compact_secret:
        if not compact_access or not compact_secret:
            return {
                "credential_source": "S3_COMPACT",
                "kwargs": {},
                "error": "partial S3_COMPACT credentials; both access and secret key are required",
            }
        return {
            "credential_source": "S3_COMPACT",
            "kwargs": _client_kwargs(
                access_key=compact_access,
                secret_key=compact_secret,
                session_token=compact_session,
                region=compact_region or aws_region,
                endpoint=archive_endpoint or compact_endpoint,
            ),
        }

    return {
        "credential_source": "AWS_DEFAULT",
        "kwargs": _client_kwargs(
            access_key=os.environ.get("AWS_ACCESS_KEY_ID"),
            secret_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            session_token=os.environ.get("AWS_SESSION_TOKEN"),
            region=aws_region,
            endpoint=archive_endpoint or compact_endpoint,
        ),
    }


def _client_kwargs(
    *,
    access_key: str | None,
    secret_key: str | None,
    session_token: str | None,
    region: str | None,
    endpoint: str | None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if access_key and secret_key:
        kwargs["aws_access_key_id"] = access_key
        kwargs["aws_secret_access_key"] = secret_key
    if session_token:
        kwargs["aws_session_token"] = session_token
    if region:
        kwargs["region_name"] = region
    if endpoint:
        kwargs["endpoint_url"] = endpoint
    return kwargs


def build_s3_client(boto3_module: Any, credential_resolution: dict[str, Any] | None = None) -> Any:
    resolution = credential_resolution or resolve_s3_client_config()
    if resolution.get("error"):
        raise RuntimeError(resolution["error"])
    return boto3_module.client("s3", **resolution["kwargs"])


def archive_plan(plan: ArchivePlan) -> dict[str, Any]:
    import boto3

    bucket, prefix = parse_s3_uri(plan.destination_prefix)
    credential_resolution = resolve_s3_client_config()
    if credential_resolution.get("error"):
        raise RuntimeError(credential_resolution["error"])
    client = build_s3_client(boto3, credential_resolution)
    file_records = inventory_files(plan.source_root)
    generated_at = utc_now()
    manifest = build_archive_manifest(plan, file_records, generated_at=generated_at)
    receipt = {
        "receipt_version": "archive_receipt_v1",
        "source_root": str(plan.source_root),
        "archive_destination_prefix": plan.destination_prefix,
        "timestamp": generated_at,
        "retained_class": plan.retained_class,
        "replayable": plan.replayable,
        "what_was_kept_locally": list(plan.thin_keep_files),
        "what_was_pruned_locally": [],
        "what_was_pruned_remotely": [],
        "verification_status": "pending",
    }
    generated_files = {
        "archive_manifest.json": json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        "SHA256SUMS": render_sha256sums(file_records),
        "archive_receipt.json": json.dumps(receipt, indent=2, sort_keys=True) + "\n",
    }

    uploaded: list[dict[str, Any]] = []
    for record in file_records:
        object_key = f"{prefix.rstrip('/')}/{record.path}"
        source_path = plan.source_root / record.path
        client.upload_file(
            str(source_path),
            bucket,
            object_key,
            ExtraArgs={"Metadata": {"sha256": record.sha256}},
        )
        head = client.head_object(Bucket=bucket, Key=object_key)
        if head["ContentLength"] != record.size_bytes:
            raise RuntimeError(f"archive size verification failed for {record.path}")
        if head.get("Metadata", {}).get("sha256") != record.sha256:
            raise RuntimeError(f"archive checksum metadata verification failed for {record.path}")
        uploaded.append({"path": record.path, "size_bytes": record.size_bytes, "sha256": record.sha256})

    for relative_path, text in generated_files.items():
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        object_key = f"{prefix.rstrip('/')}/_archive/{relative_path}"
        client.put_object(
            Bucket=bucket,
            Key=object_key,
            Body=text.encode("utf-8"),
            Metadata={"sha256": digest},
        )
        head = client.head_object(Bucket=bucket, Key=object_key)
        if head["ContentLength"] != len(text.encode("utf-8")):
            raise RuntimeError(f"archive metadata size verification failed for {relative_path}")
        if head.get("Metadata", {}).get("sha256") != digest:
            raise RuntimeError(f"archive metadata checksum verification failed for {relative_path}")

    verified_at = utc_now()
    receipt["verification_status"] = "verified"
    receipt["verified_at"] = verified_at
    receipt["file_inventory"] = uploaded
    receipt["checksum_manifest"] = "SHA256SUMS"
    write_json(plan.source_root / "archive_manifest.json", manifest)
    write_text(plan.source_root / "SHA256SUMS", generated_files["SHA256SUMS"])
    write_json(plan.source_root / "archive_receipt.json", receipt)
    return {
        "source_root": str(plan.source_root),
        "archive_destination_prefix": plan.destination_prefix,
        "uploaded_file_count": len(uploaded),
        "verification_status": "verified",
        "verified_at": verified_at,
        "receipt_path": str(plan.source_root / "archive_receipt.json"),
    }


def discover_inventory_roots(inventory_root: Path, *, repo_root: Path = REPO_ROOT) -> list[Path]:
    resolved = inventory_root.expanduser().resolve()
    outputs_root = (repo_root / "outputs").resolve()
    roots: list[Path] = []
    if resolved != outputs_root:
        return [child for child in sorted(resolved.iterdir()) if child.is_dir()]
    for child in sorted(resolved.iterdir()):
        if child.name == "analysis":
            continue
        if child.is_dir():
            roots.append(child)
    analysis_root = resolved / "analysis"
    if analysis_root.exists():
        roots.extend(child for child in sorted(analysis_root.iterdir()) if child.is_dir())
    return roots


def build_archive_plan(
    *,
    source_root: Path,
    archive_base_uri: str = DEFAULT_ARCHIVE_BASE_URI,
    repo_root: Path = REPO_ROOT,
) -> ArchivePlan:
    resolved = source_root.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"source root does not exist: {resolved}")
    if not resolved.is_dir():
        raise ValueError(f"source root must be a directory: {resolved}")
    if not source_root_allowed(resolved, repo_root=repo_root):
        raise ValueError(f"source root is outside archive allowlist: {resolved}")

    files = list(iter_files(resolved))
    denied_entries = tuple(find_denylisted_entries(resolved))
    tracked_entries = tuple(find_tracked_entries(resolved, repo_root=repo_root))
    retained_class, replayable = infer_retained_class(resolved)
    thin_keep_files = tuple(
        path.relative_to(resolved).as_posix()
        for path in files
        if should_keep_in_thin_mirror(path, resolved)
    )
    thin_keep_set = set(thin_keep_files)
    prune_candidate_files = tuple(
        path.relative_to(resolved).as_posix()
        for path in files
        if path.relative_to(resolved).as_posix() not in thin_keep_set
    )
    prune_candidate_bytes = sum((resolved / path).stat().st_size for path in prune_candidate_files)
    return ArchivePlan(
        source_root=resolved,
        size_bytes=sum(path.stat().st_size for path in files),
        retained_class=retained_class,
        replayable=replayable,
        destination_prefix=derive_destination_prefix(resolved, archive_base_uri, repo_root=repo_root),
        file_count=len(files),
        denied_entries=denied_entries,
        tracked_entries=tracked_entries,
        thin_keep_files=thin_keep_files,
        prune_candidate_files=prune_candidate_files,
        prune_candidate_bytes=prune_candidate_bytes,
    )


def source_root_allowed(source_root: Path, *, repo_root: Path = REPO_ROOT) -> bool:
    outputs_root = (repo_root / "outputs").resolve()
    remote_roots = (Path("/workspace/runs").resolve(), Path("/root/runs").resolve())
    return path_is_relative_to(source_root, outputs_root) or any(
        path_is_relative_to(source_root, remote_root) for remote_root in remote_roots
    )


def derive_destination_prefix(
    source_root: Path,
    archive_base_uri: str = DEFAULT_ARCHIVE_BASE_URI,
    *,
    repo_root: Path = REPO_ROOT,
) -> str:
    base = archive_base_uri.rstrip("/")
    outputs_root = (repo_root / "outputs").resolve()
    resolved = source_root.resolve()
    if path_is_relative_to(resolved, outputs_root):
        relative = resolved.relative_to(outputs_root).as_posix()
        return f"{base}/local-outputs/{relative}/"
    for remote_root in (Path("/workspace/runs").resolve(), Path("/root/runs").resolve()):
        if path_is_relative_to(resolved, remote_root):
            relative = resolved.relative_to(remote_root).as_posix()
            return f"{base}/remote-runs/{relative}/"
    raise ValueError(f"cannot derive archive prefix for source root: {source_root}")


def find_denylisted_entries(source_root: Path) -> list[str]:
    denied: list[str] = []
    for path in iter_all_entries(source_root):
        relative = path.relative_to(source_root).as_posix()
        if is_denylisted(path):
            denied.append(relative)
    return sorted(denied)


def is_denylisted(path: Path) -> bool:
    name = path.name
    if path.is_dir() and name in DENYLIST_DIR_NAMES:
        return True
    if name in DENYLIST_FILE_NAMES:
        return True
    return any(fnmatch.fnmatch(name, pattern) for pattern in DENYLIST_FILE_PATTERNS)


def find_tracked_entries(source_root: Path, *, repo_root: Path = REPO_ROOT) -> list[str]:
    if not (repo_root / ".git").exists():
        return []
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "ls-files", "--", str(source_root)],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return []
    if result.returncode != 0:
        return []
    return sorted(line.strip() for line in result.stdout.splitlines() if line.strip())


def infer_retained_class(source_root: Path) -> tuple[str, bool]:
    manifest_path = source_root / "bundle_manifest.json"
    if manifest_path.exists():
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
        if payload.get("run_completion_state") == "partial" or payload.get("known_partial") is True:
            return "partial", bool(payload.get("replayable", False))
    has_split_jsonl = any((source_root / "trajectories").glob("*.jsonl"))
    has_tensor_payload = any((source_root / "trajectories" / "tensor_cache_v1").glob("**/*.pt"))
    has_event_payload = any((source_root / "trajectories" / "event_token_cache_v1").glob("**/*.pt"))
    replayable = has_split_jsonl or has_tensor_payload
    if replayable:
        return "full", True
    if manifest_path.exists() or (source_root / "trajectories" / "manifest.json").exists():
        return "slim", False
    if has_event_payload:
        return "partial", False
    return "partial", False


def should_keep_in_thin_mirror(path: Path, source_root: Path) -> bool:
    relative = path.relative_to(source_root).as_posix()
    name = path.name
    if is_denylisted(path):
        return False
    if name in THIN_KEEP_NAMES:
        return True
    if name.endswith(THIN_KEEP_SUFFIXES):
        return True
    if "manifest" in name and name.endswith(".json"):
        return True
    if "receipt" in name and name.endswith(".json"):
        return True
    if name.endswith(".json") and path.stat().st_size <= DEFAULT_MAX_THIN_FILE_BYTES:
        return True
    if relative.startswith("configs/") and path.stat().st_size <= DEFAULT_MAX_THIN_FILE_BYTES:
        return True
    return False


def inventory_files(source_root: Path) -> list[FileRecord]:
    records: list[FileRecord] = []
    for path in iter_files(source_root):
        records.append(
            FileRecord(
                path=path.relative_to(source_root).as_posix(),
                size_bytes=path.stat().st_size,
                sha256=sha256_file(path),
            )
        )
    return records


def build_archive_manifest(
    plan: ArchivePlan,
    file_records: list[FileRecord],
    *,
    generated_at: str,
) -> dict[str, Any]:
    return {
        "manifest_version": "archive_manifest_v1",
        "generated_at": generated_at,
        "source_root": str(plan.source_root),
        "archive_destination_prefix": plan.destination_prefix,
        "retained_class": plan.retained_class,
        "replayable": plan.replayable,
        "file_inventory": [record.__dict__ for record in file_records],
        "thin_local_mirror": list(plan.thin_keep_files),
        "proposed_prune": list(plan.prune_candidate_files),
    }


def render_sha256sums(file_records: list[FileRecord]) -> str:
    lines = [f"{record.sha256}  {record.path}" for record in file_records]
    return "\n".join(lines) + ("\n" if lines else "")


def iter_files(root: Path) -> list[Path]:
    return [path for path in sorted(root.rglob("*")) if path.is_file()]


def iter_all_entries(root: Path) -> list[Path]:
    return [path for path in sorted(root.rglob("*"))]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"expected s3:// URI, got {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def path_is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def human_size(size_bytes: int) -> str:
    units = ("B", "K", "M", "G", "T")
    size = float(size_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{int(size)}{unit}" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1024
    return f"{size_bytes}B"


def utc_now() -> str:
    return datetime.now(tz=UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def emit_report(report: dict[str, Any], output_json: Path | None) -> None:
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    raise SystemExit(main())

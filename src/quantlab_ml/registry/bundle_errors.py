from __future__ import annotations

from pathlib import Path


class BundlePayloadError(ValueError):
    def __init__(
        self,
        *,
        error_code: str,
        detail: str,
        bundle_payload_class: str | None = None,
        bundle_root: Path | None = None,
    ) -> None:
        self.error_code = error_code
        self.detail = detail
        self.bundle_payload_class = bundle_payload_class
        self.bundle_root = bundle_root
        super().__init__(self.to_message())

    def to_message(self) -> str:
        parts = [self.error_code]
        if self.bundle_payload_class is not None:
            parts.append(f"bundle_payload_class={self.bundle_payload_class}")
        parts.append(self.detail)
        if self.bundle_root is not None:
            parts.append(f"path={self.bundle_root}")
        return " ".join(parts)


class DanglingTensorCacheManifestError(BundlePayloadError):
    def __init__(
        self,
        *,
        detail: str,
        bundle_payload_class: str = "slim",
        bundle_root: Path | None = None,
    ) -> None:
        super().__init__(
            error_code="dangling_tensor_cache_manifest",
            detail=detail,
            bundle_payload_class=bundle_payload_class,
            bundle_root=bundle_root,
        )


class Phase0EmpiricalClosureUnsupportedError(BundlePayloadError):
    def __init__(
        self,
        *,
        detail: str,
        bundle_payload_class: str = "slim",
        bundle_root: Path | None = None,
    ) -> None:
        super().__init__(
            error_code="phase0_empirical_closure_unsupported",
            detail=detail,
            bundle_payload_class=bundle_payload_class,
            bundle_root=bundle_root,
        )


__all__ = [
    "BundlePayloadError",
    "DanglingTensorCacheManifestError",
    "Phase0EmpiricalClosureUnsupportedError",
]

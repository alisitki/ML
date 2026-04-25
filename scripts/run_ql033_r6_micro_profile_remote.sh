#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${1:?usage: $0 RUN_ROOT [MISS_ROWS] [CACHE_HIT_ROWS]}"
MISS_ROWS="${2:-359}"
CACHE_HIT_ROWS="${3:-120}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
OBJECT_LIST_JSON="${OBJECT_LIST_JSON:-outputs/analysis/ql033-r6-real-micro-profile/s3_compact_objects.json}"

mkdir -p "${RUN_ROOT}"

PROFILE_JSON="${RUN_ROOT}/ql033_r6_real_window_base_micro_profile.json"
VALIDATION_JSON="${RUN_ROOT}/ql033_r6_real_window_base_micro_profile.validation.json"
PROFILE_LOG="${RUN_ROOT}/profile.log"
VALIDATION_LOG="${RUN_ROOT}/profile_validation.log"

"${PYTHON_BIN}" -u scripts/profile_ql033_r6_window_base.py \
  --source s3-compact \
  --real-index-mode windowed \
  --data-config outputs/ql033-r5-windowbase-20260424-rerun2-thin/proof-slice.data.yaml \
  --s3-env-file .env \
  --object-list-json "${OBJECT_LIST_JSON}" \
  --materialize-s3-cache \
  --miss-rows "${MISS_ROWS}" \
  --cache-hit-rows "${CACHE_HIT_ROWS}" \
  --output-json "${PROFILE_JSON}" \
  > >(tee "${PROFILE_LOG}") \
  2>&1

set +e
"${PYTHON_BIN}" -u scripts/validate_ql033_r6_micro_profile.py \
  --profile-json "${PROFILE_JSON}" \
  --report-json "${VALIDATION_JSON}" \
  > >(tee "${VALIDATION_LOG}") \
  2>&1
VALIDATION_EXIT=$?
set -e

{
  find "${RUN_ROOT}" -maxdepth 1 -type f \
    \( -name '*.json' -o -name '*.log' \) \
    -print0 | sort -z | xargs -0 sha256sum
} > "${RUN_ROOT}/SHA256SUMS"

echo "{\"validation_exit\": ${VALIDATION_EXIT}, \"profile_json\": \"${PROFILE_JSON}\", \"validation_json\": \"${VALIDATION_JSON}\"}" \
  > "${RUN_ROOT}/runner_status.json"

exit 0

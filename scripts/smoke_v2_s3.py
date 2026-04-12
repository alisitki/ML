#!/usr/bin/env python3
"""scripts/smoke_v2_s3.py — V2 Learning Surface gerçek veri smoke testi.

Kullanım:
    python scripts/smoke_v2_s3.py [--symbols BTCUSDT,ETHUSDT] [--dry-run]

Kontrol ettikleri:
  1. Tensor shape — beklenen (n_t, n_sym, n_exc, n_str, n_fld)
  2. Memory boyutu — 1 step tensor float64 bytes
  3. Mask dağılımı — padding / unavailable / missing / stale / valid oranları
  4. Venue spread dolululğu — DerivedSurface venue_pair_* kanalları NaN mı?
  5. Reward context farklılaşması — exchange'ler arası reference_price ve funding farkı
  6. Field coverage — mark_price / funding_rate / open_interest alanları dolu mu?
  7. Stale/missing oranı — funding ve OI stream'leri için beklenen aralıkta mı?

Çıktı: JSON rapor (stdout) + konsolda özet tablo.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# ------------------------------------------------------------------
# Path setup
# ------------------------------------------------------------------
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from quantlab_ml.common import load_yaml  # noqa: E402
from quantlab_ml.contracts import (  # noqa: E402
    ActionSpaceSpec,
    DatasetSpec,
    RewardEventSpec,
    TrajectorySpec,
    TrajectoryBundle,
)
from quantlab_ml.data import S3CompactedSource  # noqa: E402
from quantlab_ml.trajectories import TrajectoryBuilder  # noqa: E402

# ------------------------------------------------------------------
# Config yolları
# ------------------------------------------------------------------
DATA_CONFIG = REPO / "configs" / "data" / "s3-current.yaml"
TRAINING_CONFIG = REPO / "configs" / "training" / "default.yaml"
REWARD_CONFIG = REPO / "configs" / "reward" / "default.yaml"
ENV_FILE = REPO / ".env"


def _load_configs() -> tuple[DatasetSpec, TrajectorySpec, ActionSpaceSpec, RewardEventSpec]:
    ds = DatasetSpec.model_validate(load_yaml(DATA_CONFIG)["dataset"])
    raw_train = load_yaml(TRAINING_CONFIG)
    ts = TrajectorySpec.model_validate(raw_train["trajectory"])
    acs = ActionSpaceSpec.model_validate(raw_train["action_space"])
    rs = RewardEventSpec.model_validate(load_yaml(REWARD_CONFIG)["reward"])
    return ds, ts, acs, rs


# ------------------------------------------------------------------
# Report section builders
# ------------------------------------------------------------------

def _check_tensor_shape(bundle: TrajectoryBundle) -> dict[str, Any]:
    schema = bundle.observation_schema
    results: dict[str, Any] = {}
    for scale_spec in schema.scale_axis:
        expected = schema.shape_for_scale(scale_spec.label)
        n_t, n_sym, n_exc, n_str, n_fld = expected
        flat_expected = n_t * n_sym * n_exc * n_str * n_fld

        # İlk train step'ini kontrol et
        first_step = bundle.splits["train"][0].steps[0]
        tensor = first_step.observation.raw_surface[scale_spec.label]
        match = tensor.flat_size == flat_expected and tensor.shape == list(expected)
        results[scale_spec.label] = {
            "expected_shape": list(expected),
            "actual_shape": tensor.shape,
            "expected_flat": flat_expected,
            "actual_flat": tensor.flat_size,
            "ok": match,
        }
    return results


def _check_memory(bundle: TrajectoryBundle) -> dict[str, Any]:
    schema = bundle.observation_schema
    first_step = bundle.splits["train"][0].steps[0]
    results: dict[str, Any] = {}
    total_bytes = 0
    for scale_spec in schema.scale_axis:
        tensor = first_step.observation.raw_surface[scale_spec.label]
        # float64 = 8 bytes per value, bool = 1 byte per mask cell
        values_bytes = tensor.flat_size * 8  # values + age
        age_bytes = tensor.flat_size * 8
        mask_bytes = tensor.flat_size * 4  # 4 masks × 1 byte
        total_scale = values_bytes + age_bytes + mask_bytes
        results[scale_spec.label] = {
            "flat_size": tensor.flat_size,
            "values_bytes": values_bytes,
            "age_bytes": age_bytes,
            "total_bytes_approx": total_scale,
            "total_kbytes": round(total_scale / 1024, 2),
        }
        total_bytes += total_scale
    results["_total_kbytes"] = round(total_bytes / 1024, 2)
    return results


def _check_mask_distribution(bundle: TrajectoryBundle) -> dict[str, Any]:
    schema = bundle.observation_schema
    results: dict[str, Any] = {}

    for scale_spec in schema.scale_axis:
        padding_count = 0
        unavail_count = 0
        missing_count = 0
        stale_count = 0
        valid_count = 0
        total_cells = 0

        for split_records in bundle.splits.values():
            for record in split_records:
                for step in record.steps:
                    tensor = step.observation.raw_surface[scale_spec.label]
                    for i in range(tensor.flat_size):
                        total_cells += 1
                        if tensor.padding[i]:
                            padding_count += 1
                        elif tensor.unavailable_by_contract[i]:
                            unavail_count += 1
                        elif tensor.missing[i]:
                            missing_count += 1
                        elif tensor.stale[i]:
                            stale_count += 1
                        else:
                            valid_count += 1

        def pct(n: int) -> str:
            return f"{100 * n / max(total_cells, 1):.1f}%"

        results[scale_spec.label] = {
            "total_cells": total_cells,
            "padding": {"count": padding_count, "pct": pct(padding_count)},
            "unavailable_by_contract": {"count": unavail_count, "pct": pct(unavail_count)},
            "missing": {"count": missing_count, "pct": pct(missing_count)},
            "stale": {"count": stale_count, "pct": pct(stale_count)},
            "valid": {"count": valid_count, "pct": pct(valid_count)},
        }
    return results


def _check_venue_spread(bundle: TrajectoryBundle) -> dict[str, Any]:
    results: dict[str, Any] = {}
    first_step = bundle.splits["train"][-1].steps[-1]  # En istikrarlı step
    derived = first_step.observation.derived_surface
    if derived is None:
        return {"error": "derived_surface is None"}

    spread_channels = [ch for ch in derived.channels if "venue_pair" in ch.key and "spread" in ch.key]
    for ch in spread_channels:
        has_value = ch.values and ch.values[0] != 0.0
        results[ch.key] = {
            "value": ch.values[0] if ch.values else None,
            "filled": has_value,
            "description": ch.description,
        }
    return results


def _check_reward_differentiation(bundle: TrajectoryBundle) -> dict[str, Any]:
    results: dict[str, Any] = {}
    # Birden fazla step için venue farklılığı kontrol et
    for split_name, records in bundle.splits.items():
        if not records:
            continue
        prices: dict[str, list[float]] = {}
        fundings: dict[str, list[float]] = {}
        for record in records[:2]:  # ilk 2 trajectory yeterli
            for step in record.steps[:5]:  # ilk 5 step
                ctx = step.reward_context
                for exchange, ref in ctx.venues.items():
                    prices.setdefault(exchange, []).append(ref.reference_price)
                    fundings.setdefault(exchange, []).append(ref.funding_rate)

        exchange_summary = {}
        for exchange in prices:
            ps = prices[exchange]
            fs = fundings[exchange]
            exchange_summary[exchange] = {
                "price_min": min(ps),
                "price_max": max(ps),
                "price_range": max(ps) - min(ps),
                "funding_min": min(fs),
                "funding_max": max(fs),
            }

        # Venue'lar arası ortalama fiyat farkı
        exchanges = list(exchange_summary.keys())
        venue_diffs: dict[str, float] = {}
        for i, exc_a in enumerate(exchanges):
            for exc_b in exchanges[i + 1 :]:
                avg_a = sum(prices[exc_a]) / len(prices[exc_a])
                avg_b = sum(prices[exc_b]) / len(prices[exc_b])
                venue_diffs[f"{exc_a}_vs_{exc_b}"] = abs(avg_a - avg_b)

        results[split_name] = {
            "per_exchange": exchange_summary,
            "avg_price_diffs": venue_diffs,
            "venues_differentiated": any(v > 0.0 for v in venue_diffs.values()),
        }
    return results


def _check_field_coverage(bundle: TrajectoryBundle) -> dict[str, Any]:
    schema = bundle.observation_schema
    results: dict[str, Any] = {}

    check_field_pairs = [
        ("mark_price", "mark_price"),
        ("funding", "funding_rate"),
        ("open_interest", "open_interest"),
        ("bbo", "mid"),
        ("trade", "price"),
    ]

    first_step = bundle.splits["train"][-1].steps[-1]
    scale_label = schema.scale_axis[0].label
    tensor = first_step.observation.raw_surface[scale_label]

    n_t, n_sym, n_exc, n_str, n_fld_total = tensor.shape
    stream_axis = schema.stream_axis
    field_counts = [len(schema.field_axis.get(s, [])) for s in stream_axis]

    for stream, field_name in check_field_pairs:
        if stream not in stream_axis:
            results[f"{stream}.{field_name}"] = {"skip": "stream not in axis"}
            continue
        if field_name not in schema.field_axis.get(stream, []):
            results[f"{stream}.{field_name}"] = {"skip": "field not in catalog"}
            continue

        str_idx = stream_axis.index(stream)
        fi = schema.field_axis[stream].index(field_name)
        f_total = sum(field_counts)
        f_offset = sum(field_counts[:str_idx])

        filled = []
        for t_idx in range(n_t):
            for sym_idx in range(n_sym):
                for exc_idx in range(n_exc):
                    idx = (
                        t_idx * n_sym * n_exc * n_str * f_total
                        + sym_idx * n_exc * n_str * f_total
                        + exc_idx * n_str * f_total
                        + str_idx * f_total
                        + f_offset + fi
                    )
                    if idx >= tensor.flat_size:
                        continue
                    if tensor.padding[idx] or tensor.unavailable_by_contract[idx] or tensor.missing[idx]:
                        continue
                    filled.append(tensor.values[idx])

        results[f"{stream}.{field_name}"] = {
            "filled_coords": len(filled),
            "sample_values": [round(v, 6) for v in filled[:5]],
            "all_zero": all(v == 0.0 for v in filled) if filled else True,
        }
    return results


def _check_stale_by_stream(bundle: TrajectoryBundle) -> dict[str, Any]:
    schema = bundle.observation_schema
    scale_label = schema.scale_axis[0].label
    stream_axis = schema.stream_axis
    field_counts = [len(schema.field_axis.get(s, [])) for s in stream_axis]
    f_total = sum(field_counts)
    n_exc = len(schema.exchange_axis)
    n_str = len(stream_axis)
    n_sym = len(schema.asset_axis)

    stale_counts: dict[str, int] = {s: 0 for s in stream_axis}
    valid_counts: dict[str, int] = {s: 0 for s in stream_axis}

    for split_records in bundle.splits.values():
        for record in split_records:
            for step in record.steps:
                tensor = step.observation.raw_surface[scale_label]
                n_t = schema.scale_axis[0].num_buckets
                f_offsets = []
                off = 0
                for fc in field_counts:
                    f_offsets.append(off)
                    off += fc

                for t_idx in range(n_t):
                    for sym_idx in range(n_sym):
                        for exc_idx in range(n_exc):
                            for str_idx, stream in enumerate(stream_axis):
                                first_fi = f_offsets[str_idx]
                                idx = (
                                    t_idx * n_sym * n_exc * n_str * f_total
                                    + sym_idx * n_exc * n_str * f_total
                                    + exc_idx * n_str * f_total
                                    + str_idx * f_total
                                    + first_fi
                                )
                                if idx >= tensor.flat_size:
                                    continue
                                if tensor.padding[idx] or tensor.unavailable_by_contract[idx] or tensor.missing[idx]:
                                    continue
                                if tensor.stale[idx]:
                                    stale_counts[stream] += 1
                                else:
                                    valid_counts[stream] += 1

    results: dict[str, Any] = {}
    for stream in stream_axis:
        total = stale_counts[stream] + valid_counts[stream]
        results[stream] = {
            "stale": stale_counts[stream],
            "valid": valid_counts[stream],
            "stale_pct": f"{100 * stale_counts[stream] / max(total, 1):.1f}%",
        }
    return results


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def _print_section(title: str, data: dict[str, Any]) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(json.dumps(data, indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(description="V2 S3 smoke test")
    parser.add_argument("--dry-run", action="store_true", help="S3'e bağlanma; sadece config'i doğrula")
    parser.add_argument(
        "--symbols",
        default="",
        help="Sembol listesi (virgülle ayrılmış, boşsa config'den gelir)",
    )
    parser.add_argument(
        "--max-symbols",
        type=int,
        default=3,
        help="Config'den alınan sembol listesini bu sayıyla kırp (default=3, download süresini kısaltır)",
    )
    parser.add_argument(
        "--skip-streams",
        default="bbo",
        help="İndirilmeyecek stream'ler (virgülle, default='bbo'). BBO günlük ~200MB olduğundan smoke'ta atlanır.",
    )
    args = parser.parse_args()

    print("Loading configs...")
    ds, ts, acs, rs = _load_configs()

    # Optional symbol override / kırpma
    if args.symbols:
        override = [s.strip() for s in args.symbols.split(",") if s.strip()]
        ds = ds.model_copy(update={"symbols": override})
        print(f"Symbol override: {ds.symbols}")
    elif args.max_symbols and len(ds.symbols) > args.max_symbols:
        ds = ds.model_copy(update={"symbols": ds.symbols[: args.max_symbols]})
        print(f"Symbol kırpıldı (--max-symbols={args.max_symbols}): {ds.symbols}")

    print(f"Dataset: {ds.dataset_hash}")
    print(f"Symbols: {ds.symbols}")
    print(f"Exchanges: {ds.exchanges}")
    print(f"Scale preset: {[s.label for s in ts.scale_preset]}")

    if args.dry_run:
        print("\n[DRY RUN] Config validated. S3 bağlantısı atlandı.")
        print(json.dumps({"status": "dry_run_ok", "symbols": ds.symbols}, indent=2))
        return

    print("\nConnecting to S3...")
    t0 = time.time()
    source = S3CompactedSource.from_env_file(ENV_FILE)

    # Boto3'e bağlantı ve okuma timeout'u ayarla — silent hang'leri önler.
    try:
        from botocore.config import Config as BotocoreConfig
        timeout_config = BotocoreConfig(connect_timeout=10, read_timeout=120, retries={"max_attempts": 2})
        import boto3
        session = boto3.session.Session(
            aws_access_key_id=source.access_key_id,
            aws_secret_access_key=source.secret_access_key,
            region_name=source.region_name,
        )
        source._client = session.client("s3", endpoint_url=source.endpoint_url, config=timeout_config)
        print("Boto3 client ready (connect_timeout=10s, read_timeout=120s)")
    except ImportError:
        print("WARN: botocore not available — using default timeouts")

    # Partition listesini state'ten al — S3 discover loop yok
    skip_streams = {s.strip() for s in args.skip_streams.split(",") if s.strip()}
    partitions = source.list_matching_partitions(ds)
    if skip_streams:
        before = len(partitions)
        partitions = [p for p in partitions if p.stream not in skip_streams]
        print(f"Skipping streams {skip_streams}: {before} → {len(partitions)} partitions")
    print(f"Matched {len(partitions)} partitions (after stream filter)")
    if not partitions:
        print("ERROR: Hiç partition kalmadı. skip-streams veya data config'i kontrol edin.")
        sys.exit(1)

    # S3 bucket layout: exchange=.../stream=.../symbol=.../date=.../data.parquet (prefix yok)
    direct_keys = [
        f"exchange={p.exchange}/stream={p.stream}/symbol={p.symbol}/date={p.day}/data.parquet"
        for p in partitions
    ]

    # Toplam boyut tahmini (head_object x5 örnek)
    try:
        sample_keys = direct_keys[:5]
        total_size_est = 0
        for sk in sample_keys:
            r = source.client.head_object(Bucket=source.bucket, Key=sk)
            total_size_est += r["ContentLength"]
        avg_mb = total_size_est / len(sample_keys) / 1024 / 1024
        est_total_mb = avg_mb * len(direct_keys)
        print(f"Using {len(direct_keys)} keys — avg {avg_mb:.1f} MB/file, ~{est_total_mb:.0f} MB total")
    except Exception:
        print(f"Using {len(direct_keys)} direct object keys")

    # Her partition'u direkt oku — ilerleme göster
    events = []
    read_errors = 0
    for i, key in enumerate(direct_keys, 1):
        if i % 20 == 0 or i == 1 or i == len(direct_keys):
            print(f"  [{i}/{len(direct_keys)}] {key}")
        try:
            batch = source._read_object_events(key, ds)
            events.extend(batch)
        except Exception as exc:
            read_errors += 1
            print(f"  WARN [{i}]: {key}: {exc}")
    elapsed_load = time.time() - t0
    print(f"Loaded {len(events)} events in {elapsed_load:.1f}s ({read_errors} read errors)")

    if not events:
        print("ERROR: No events loaded. S3 bağlantı veya config sorununu kontrol edin.")
        sys.exit(1)

    print("Building trajectory bundle...")
    t1 = time.time()
    bundle = TrajectoryBuilder(ds, ts, acs, rs).build(events)
    elapsed_build = time.time() - t1
    train_steps = sum(len(r.steps) for r in bundle.splits.get("train", []))
    eval_steps = sum(len(r.steps) for r in bundle.splits.get("eval", []))
    print(f"Built in {elapsed_build:.1f}s — train={train_steps} steps, eval={eval_steps} steps")

    if train_steps == 0:
        print("ERROR: Train split boş. Data range veya field mapping sorununu kontrol edin.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Kontroller
    # ------------------------------------------------------------------
    report: dict[str, Any] = {
        "status": "ok",
        "dataset_hash": ds.dataset_hash,
        "events_loaded": len(events),
        "elapsed_load_s": round(elapsed_load, 2),
        "elapsed_build_s": round(elapsed_build, 2),
        "train_steps": train_steps,
        "eval_steps": eval_steps,
    }

    print("\nRunning checks...")

    shape_results = _check_tensor_shape(bundle)
    report["tensor_shape"] = shape_results
    all_shapes_ok = all(v["ok"] for v in shape_results.values())
    _print_section("1. Tensor Shape", shape_results)

    memory_results = _check_memory(bundle)
    report["memory"] = memory_results
    _print_section("2. Memory (approx, per step)", memory_results)

    mask_results = _check_mask_distribution(bundle)
    report["mask_distribution"] = mask_results
    _print_section("3. Mask Distribution (all steps)", mask_results)

    spread_results = _check_venue_spread(bundle)
    report["venue_spread"] = spread_results
    any_spread_filled = any(v.get("filled", False) for v in spread_results.values() if isinstance(v, dict))
    _print_section("4. Venue Spread (DerivedSurface)", spread_results)

    reward_results = _check_reward_differentiation(bundle)
    report["reward_differentiation"] = reward_results
    venues_differ = any(
        split.get("venues_differentiated", False)
        for split in reward_results.values()
    )
    _print_section("5. Reward Context Differentiation", reward_results)

    field_results = _check_field_coverage(bundle)
    report["field_coverage"] = field_results
    all_fields_filled = all(
        v.get("filled_coords", 0) > 0
        for v in field_results.values()
        if "skip" not in v
    )
    _print_section("6. Field Coverage (last train step)", field_results)

    stale_results = _check_stale_by_stream(bundle)
    report["stale_by_stream"] = stale_results
    _print_section("7. Stale/Missing by Stream", stale_results)

    # ------------------------------------------------------------------
    # Özet
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("  SMOKE TEST SUMMARY")
    print(f"{'='*60}")
    checks = {
        "tensor_shape_ok": all_shapes_ok,
        "venue_spread_filled": any_spread_filled,
        "reward_venues_differentiated": venues_differ,
        "field_coverage_ok": all_fields_filled,
    }
    for check, result in checks.items():
        icon = "✅" if result else "❌"
        print(f"  {icon} {check}: {result}")

    report["checks"] = checks
    all_passed = all(checks.values())
    report["status"] = "pass" if all_passed else "fail"

    print(f"\n  {'PASS ✅' if all_passed else 'FAIL ❌'} — {sum(checks.values())}/{len(checks)} checks passed")
    print()

    # JSON raporu stdout'a
    print("\n--- JSON REPORT ---")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()

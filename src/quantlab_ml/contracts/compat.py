"""contracts/compat.py — Legacy Reduction Adapter

Bu modül V1 davranışını emüle eden yardımcı fonksiyonlar içerir.
Yalnızca training/compat_adapter.py ve legacy test kodu buradan import eder.
Çekirdek contracts public API'si bu modülü dışarıya açmaz.

V1 davranışı:
  - Tüm exchange'lerin ortalaması alınır (scalar collapse)
  - İlk scale (en küçük çözünürlük) kullanılır
  - value_cube düz liste olarak üretilir
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quantlab_ml.contracts.learning_surface import ObservationContext
    from quantlab_ml.contracts.rewards import RewardSnapshot


def target_stream_series(obs: "ObservationContext", stream: str) -> list[float | None]:
    """V1 uyumlu: target symbol için stream'in exchange-ortalamalı zaman serisini döner.

    Yalnızca ilk scale (en küçük çözünürlük) kullanılır.
    - padding veya missing → None
    - unavailable_by_contract → None (exchange sayımına katılmaz)
    - stale → değer kullanılır (V1 davranışı korunuyor)
    """
    schema = obs.observation_schema
    if not schema.scale_axis:
        return []

    first_scale = schema.scale_axis[0]
    tensor = obs.raw_surface.get(first_scale.label)
    if tensor is None:
        return []

    num_buckets = first_scale.num_buckets
    symbols = schema.asset_axis
    exchanges = schema.exchange_axis
    streams = schema.stream_axis
    fields_by_stream = schema.field_axis

    if stream not in streams:
        raise KeyError(f"stream '{stream}' not in observation schema stream_axis")

    stream_idx = streams.index(stream)
    asset_idx = obs.target_asset_index

    # Her stream'in kaç field'ı var (önceki stream'lerin toplamı = field offset)
    field_counts = [len(fields_by_stream.get(s, [])) for s in streams]
    field_offset_for_stream = sum(field_counts[:stream_idx])
    total_fields = sum(field_counts)

    n_sym = len(symbols)
    n_exc = len(exchanges)
    n_str = len(streams)

    series: list[float | None] = []
    for t in range(num_buckets):
        values: list[float] = []
        for e_idx in range(n_exc):
            # İlk field'ı al (stream'in birincil değeri olarak kabul edilir)
            flat = (
                t * n_sym * n_exc * n_str * total_fields
                + asset_idx * n_exc * n_str * total_fields
                + e_idx * n_str * total_fields
                + stream_idx * total_fields
                + field_offset_for_stream
            )
            if flat >= len(tensor.values):
                continue
            if tensor.padding[flat] or tensor.missing[flat] or tensor.unavailable_by_contract[flat]:
                continue
            v = tensor.values[flat]
            if not math.isnan(v):
                values.append(v)
        series.append(sum(values) / len(values) if values else None)
    return series


def flat_value_cube(obs: "ObservationContext") -> list[float]:
    """V1 uyumlu: ilk scale'in values listesini döner.

    Gerçek V1 value_cube ile birebir format uyumunu garantilemez;
    yalnızca trainer'ın momentum hesabı için yeterli.
    """
    schema = obs.observation_schema
    if not schema.scale_axis:
        return []
    first_scale = schema.scale_axis[0]
    tensor = obs.raw_surface.get(first_scale.label)
    if tensor is None:
        return []
    return list(tensor.values)


def snapshot_reference_price(snapshot: "RewardSnapshot") -> float:
    """V1 uyumlu: RewardSnapshot'tan referans fiyatı çeker.

    Önce V2 context'e, yoksa legacy scalar'a bakar.
    """
    if snapshot.context is not None and snapshot.context.selected_venue is not None:
        venue_ref = snapshot.context.venues.get(snapshot.context.selected_venue)
        if venue_ref is not None:
            return venue_ref.reference_price
    # Legacy fallback
    return snapshot.reference_price


def flat_missing_mask(obs: "ObservationContext") -> list[bool]:
    """V1 uyumlu: missing | padding | unavailable_by_contract OR birleşimi."""
    schema = obs.observation_schema
    if not schema.scale_axis:
        return []
    first_scale = schema.scale_axis[0]
    tensor = obs.raw_surface.get(first_scale.label)
    if tensor is None:
        return []
    return [
        p or u or m
        for p, u, m in zip(tensor.padding, tensor.unavailable_by_contract, tensor.missing)
    ]


def flat_padding_mask(obs: "ObservationContext") -> list[bool]:
    """V1 uyumlu: yalnızca padding flag'ını döner."""
    schema = obs.observation_schema
    if not schema.scale_axis:
        return []
    first_scale = schema.scale_axis[0]
    tensor = obs.raw_surface.get(first_scale.label)
    if tensor is None:
        return []
    return list(tensor.padding)


def flat_stale_mask(obs: "ObservationContext") -> list[bool]:
    """V1 uyumlu: yalnızca stale flag'ını döner."""
    schema = obs.observation_schema
    if not schema.scale_axis:
        return []
    first_scale = schema.scale_axis[0]
    tensor = obs.raw_surface.get(first_scale.label)
    if tensor is None:
        return []
    return list(tensor.stale)


def flat_action_mask(obs: "ObservationContext") -> dict[str, bool]:  # noqa: ARG001
    """V1 uyumlu stub — ObservationContext'ten action mask üretemez.

    Gerçek mask için TrajectoryStep.action_feasibility.to_flat_mask() kullanın.
    Bu fonksiyon yalnızca import uyumluluğu için mevcuttur.
    """
    raise NotImplementedError(
        "flat_action_mask cannot be derived from ObservationContext alone. "
        "Use TrajectoryStep.action_feasibility.to_flat_mask() instead."
    )


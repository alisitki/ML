"""test_v2_contract_availability.py

binance open_interest koordinatlarının availability_by_contract=False olduğunu,
bunun stale veya runtime missing ile karışmadığını doğrular.
"""
from __future__ import annotations

from quantlab_ml.contracts import TrajectoryBundle


def _find_unavailable_coords(tensor, schema, exchange: str, stream: str) -> list[int]:
    """Belirli bir (exchange, stream) için unavailable_by_contract=True indexlerini döner."""
    n_t = schema.scale_axis[0].num_buckets
    n_sym = len(schema.asset_axis)
    n_exc = len(schema.exchange_axis)
    n_str = len(schema.stream_axis)
    field_counts = [len(schema.field_axis.get(s, [])) for s in schema.stream_axis]
    total_fields = sum(field_counts)

    exc_idx = schema.exchange_axis.index(exchange)
    str_idx = schema.stream_axis.index(stream)
    f_offset = sum(field_counts[:str_idx])
    n_fields = field_counts[str_idx]

    result = []
    for t in range(n_t):
        for sym_idx in range(n_sym):
            base = (
                t * n_sym * n_exc * n_str * total_fields
                + sym_idx * n_exc * n_str * total_fields
                + exc_idx * n_str * total_fields
                + str_idx * total_fields
                + f_offset
            )
            for fi in range(n_fields):
                result.append(base + fi)
    return result


def test_binance_open_interest_is_unavailable_by_contract(
    trajectory_bundle: TrajectoryBundle,
) -> None:
    schema = trajectory_bundle.observation_schema

    assert not schema.stream_available("binance", "open_interest"), (
        "binance open_interest should be structurally unavailable by contract"
    )

    # Use a later train step where there's sufficient history (less padding)
    train_records = trajectory_bundle.splits["train"]
    target_step = train_records[-1].steps[-1]
    tensor = target_step.observation.raw_surface["1m"]

    unavail_coords = _find_unavailable_coords(tensor, schema, "binance", "open_interest")
    assert len(unavail_coords) > 0

    # Among non-padding coords, ALL should be unavailable_by_contract
    non_padding_unavail_found = False
    for idx in unavail_coords:
        if tensor.padding[idx]:
            # padding coords have all flags False (padding takes priority in builder)
            continue
        assert tensor.unavailable_by_contract[idx] is True, (
            f"idx={idx} should be unavailable_by_contract (non-padding)"
        )
        assert tensor.stale[idx] is False
        assert tensor.missing[idx] is False
        non_padding_unavail_found = True

    assert non_padding_unavail_found, (
        "Expected at least one non-padding unavailable coord for binance open_interest"
    )


def test_bybit_open_interest_is_available_by_contract(
    trajectory_bundle: TrajectoryBundle,
) -> None:
    schema = trajectory_bundle.observation_schema
    assert schema.stream_available("bybit", "open_interest"), (
        "bybit open_interest should be structurally available by contract"
    )


def test_unavailable_mask_is_distinct_from_missing_and_stale(
    trajectory_bundle: TrajectoryBundle,
) -> None:
    """unavailable_by_contract hiçbir zaman stale veya missing ile aynı anda True olmamalı."""
    for split_records in trajectory_bundle.splits.values():
        for record in split_records:
            for step in record.steps:
                for tensor in step.observation.raw_surface.values():
                    for i in range(tensor.flat_size):
                        if tensor.unavailable_by_contract[i]:
                            assert not tensor.stale[i], f"unavailable coord {i} is also stale"
                            assert not tensor.missing[i], f"unavailable coord {i} is also missing"

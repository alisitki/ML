from __future__ import annotations

from itertools import combinations

from quantlab_ml.common import hash_payload
from quantlab_ml.contracts.learning_surface import DERIVED_SURFACE_CONTRACT_VERSION, ObservationSchema, ScaleSpec
from quantlab_ml.contracts.policies import (
    DERIVED_CHANNEL_TARGET_PLACEHOLDER,
    DerivedChannelTemplate,
    StrictRuntimeContract,
)

_DERIVED_STREAM_PRIORITY = ("mark_price", "bbo")


def build_strict_runtime_contract(
    observation_schema: ObservationSchema,
    *,
    policy_kind: str,
) -> StrictRuntimeContract:
    templates = canonical_derived_channel_templates(observation_schema)
    return StrictRuntimeContract(
        policy_kind=policy_kind,
        required_scale_specs=[scale.model_copy(deep=True) for scale in observation_schema.scale_axis],
        required_raw_surface_shapes=canonical_raw_surface_shapes(observation_schema),
        derived_contract_version=DERIVED_SURFACE_CONTRACT_VERSION,
        derived_channel_templates=templates,
        derived_channel_template_signature=derived_channel_template_signature(templates),
        expected_feature_dim=expected_feature_dim(observation_schema, templates),
    )


def canonical_raw_surface_shapes(observation_schema: ObservationSchema) -> dict[str, list[int]]:
    return {
        scale.label: list(observation_schema.shape_for_scale(scale.label))
        for scale in observation_schema.scale_axis
    }


def canonical_derived_channel_templates(observation_schema: ObservationSchema) -> list[DerivedChannelTemplate]:
    templates: list[DerivedChannelTemplate] = []
    available_streams = set(observation_schema.stream_axis)

    for exchange_a, exchange_b in combinations(observation_schema.exchange_axis, 2):
        selected_stream = _select_pairwise_stream(observation_schema, available_streams, exchange_a, exchange_b)
        if selected_stream is None:
            continue
        templates.append(
            DerivedChannelTemplate(
                key_template=f"venue_pair_{selected_stream}_spread_{exchange_a}_{exchange_b}",
                shape=[1],
            )
        )

    for symbol in observation_schema.asset_axis:
        templates.append(
            DerivedChannelTemplate(
                key_template=f"relative_move_{DERIVED_CHANNEL_TARGET_PLACEHOLDER}_vs_{symbol}",
                shape=[1],
                skip_if_target_symbol_equals=symbol,
            )
        )

    return sorted(templates, key=lambda item: item.key_template)


def resolve_derived_channel_templates(
    templates: list[DerivedChannelTemplate],
    *,
    target_symbol: str,
) -> list[DerivedChannelTemplate]:
    resolved: list[DerivedChannelTemplate] = []
    for template in templates:
        resolved_key = template.resolve_key(target_symbol)
        if resolved_key is None:
            continue
        resolved.append(
            DerivedChannelTemplate(
                key_template=resolved_key,
                shape=list(template.shape),
                skip_if_target_symbol_equals=template.skip_if_target_symbol_equals,
            )
        )
    return resolved


def expected_feature_dim(
    observation_schema: ObservationSchema,
    templates: list[DerivedChannelTemplate] | None = None,
) -> int:
    canonical_templates = templates or canonical_derived_channel_templates(observation_schema)
    raw_dim = sum(_shape_size(list(observation_schema.shape_for_scale(scale.label))) * 6 for scale in observation_schema.scale_axis)
    derived_dims = {
        target_symbol: sum(_shape_size(template.shape) for template in resolve_derived_channel_templates(canonical_templates, target_symbol=target_symbol))
        for target_symbol in observation_schema.asset_axis
    }
    if len(set(derived_dims.values())) > 1:
        raise ValueError(
            "derived channel contract must produce invariant feature dimensions across target symbols; "
            f"got {derived_dims}"
        )
    derived_dim = next(iter(derived_dims.values()), 0)
    return raw_dim + derived_dim + 1


def derived_channel_template_signature(templates: list[DerivedChannelTemplate]) -> str:
    return hash_payload(
        [
            {
                "key_template": template.key_template,
                "shape": list(template.shape),
                "skip_if_target_symbol_equals": template.skip_if_target_symbol_equals,
            }
            for template in templates
        ]
    )


def scale_specs_match(left: list[ScaleSpec], right: list[ScaleSpec]) -> bool:
    return [scale.model_dump(mode="json") for scale in left] == [scale.model_dump(mode="json") for scale in right]


def _select_pairwise_stream(
    observation_schema: ObservationSchema,
    available_streams: set[str],
    exchange_a: str,
    exchange_b: str,
) -> str | None:
    for stream in _DERIVED_STREAM_PRIORITY:
        if stream not in available_streams:
            continue
        if observation_schema.stream_available(exchange_a, stream) and observation_schema.stream_available(exchange_b, stream):
            return stream
    return None


def _shape_size(shape: list[int]) -> int:
    total = 1
    for dim in shape:
        total *= dim
    return total

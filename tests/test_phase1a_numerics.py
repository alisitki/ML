from __future__ import annotations

import numpy as np
import pytest

from quantlab_ml.training import trainer as trainer_module


def test_phase1a_batch_step_clips_aux_value_gradient_and_records_telemetry(
    phase1a_training_bundle,
) -> None:
    _, _, training_config = phase1a_training_bundle
    config = training_config.model_copy(update={"phase1a_compute_dtype": "float32"})
    backend = trainer_module._resolve_training_backend("numpy")
    state = trainer_module._phase1a_initialize_state(
        backend=backend,
        seed=7,
        joint_action_count=3,
        feature_dim=8,
        compute_dtype="float32",
    )
    state.joint_action_weight.fill(0.0)
    state.joint_action_bias.fill(0.0)
    state.value_weight.fill(0.0)
    state.value_bias = 0.0

    result = trainer_module._phase1a_batch_step(
        backend=backend,
        state=state,
        batch_features=np.ones((4, 8), dtype=np.float32),
        batch_joint_action_labels=np.zeros(4, dtype=np.int64),
        batch_joint_action_masks=np.ones((4, 3), dtype=np.bool_),
        batch_value_targets=np.full(4, 1.0e9, dtype=np.float32),
        config=config,
        batch_context={
            "candidate_index": 0,
            "epoch": 1,
            "split_name": "development",
            "shard_index": 0,
            "row_index_min": 0,
            "row_index_max": 3,
            "row_count": 4,
        },
    )

    assert np.isfinite(result.total_loss)
    assert np.all(np.isfinite(state.value_weight))
    assert np.isfinite(state.value_bias)
    assert result.numerics.value_pred_abs_max == 0.0
    assert result.numerics.value_grad_norm_pre_clip > result.numerics.value_grad_norm_post_clip > 0.0
    assert result.numerics.value_grad_norm_post_clip <= (
        trainer_module._PHASE1A_VALUE_GRAD_CLIP_NORM * 1.000001
    )
    assert result.numerics.clip_applied_count == 1
    assert result.numerics.first_nonfinite_component is None
    assert result.numerics.first_nonfinite_batch_context is None


def test_phase1a_batch_step_reports_first_nonfinite_batch_context(
    phase1a_training_bundle,
) -> None:
    _, _, training_config = phase1a_training_bundle
    config = training_config.model_copy(update={"phase1a_compute_dtype": "float32"})
    backend = trainer_module._resolve_training_backend("numpy")
    state = trainer_module._phase1a_initialize_state(
        backend=backend,
        seed=7,
        joint_action_count=3,
        feature_dim=8,
        compute_dtype="float32",
    )
    batch_context = {
        "candidate_index": 3,
        "epoch": 2,
        "split_name": "development",
        "shard_index": 2,
        "row_index_min": 99,
        "row_index_max": 122,
        "row_count": 24,
    }

    with pytest.raises(trainer_module._Phase1ANumericsError) as excinfo:
        trainer_module._phase1a_batch_step(
            backend=backend,
            state=state,
            batch_features=np.asarray([[np.inf] + [1.0] * 7], dtype=np.float32),
            batch_joint_action_labels=np.zeros(1, dtype=np.int64),
            batch_joint_action_masks=np.ones((1, 3), dtype=np.bool_),
            batch_value_targets=np.zeros(1, dtype=np.float32),
            config=config,
            batch_context=batch_context,
        )

    assert excinfo.value.component == "batch_features"
    assert excinfo.value.batch_context == batch_context
    assert excinfo.value.numerics is not None
    assert excinfo.value.numerics.first_nonfinite_component == "batch_features"
    assert excinfo.value.numerics.first_nonfinite_batch_context == batch_context

from __future__ import annotations

import pytest

from quantlab_ml.training import LinearPolicyTrainer, MomentumBaselineTrainer


def test_momentum_baseline_trainer_warns_and_preserves_training_behavior(
    trajectory_bundle,
    training_bundle,
) -> None:
    _, _, training_config = training_bundle

    with pytest.warns(DeprecationWarning, match="MomentumBaselineTrainer is deprecated"):
        trainer = MomentumBaselineTrainer(training_config)

    assert isinstance(trainer, LinearPolicyTrainer)
    artifact = trainer.train(trajectory_bundle)
    assert artifact.policy_family == training_config.trainer_name
    assert artifact.training_summary["training_backend"] == "numpy"

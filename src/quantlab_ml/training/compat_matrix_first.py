"""FIXTURE / TEST COMPAT ONLY matrix-first training helpers.

This module intentionally keeps the legacy matrix-first preparation flow alive
only for fixture tests and narrow continuity checks. Production code must use
the streaming trainer path in ``trainer.py`` and must not import these helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
import gc
import logging
from typing import Any

import numpy as np

from quantlab_ml.contracts import TrajectoryBundle
from quantlab_ml.models.features import observation_feature_vector

logger = logging.getLogger(__name__)

_FIXTURE_TEST_COMPAT_ONLY = True


@dataclass(slots=True)
class CompatTrainingExample:
    """FIXTURE / TEST COMPAT ONLY `_FIXTURE_TEST_COMPAT_ONLY`."""

    features: np.ndarray
    action_key: str
    venue: str | None


@dataclass(slots=True)
class CompatPreparedTrainingData:
    """FIXTURE / TEST COMPAT ONLY `_FIXTURE_TEST_COMPAT_ONLY`.

    Matrix-first prepared training state. Production code must not use this
    dense in-memory representation.
    """

    train_step_count: int
    val_step_count: int
    action_keys: list[str]
    venue_choices: list[str]
    normalized_train: np.ndarray
    normalized_val: np.ndarray
    feature_mean: np.ndarray
    feature_std: np.ndarray
    action_labels: np.ndarray
    val_action_labels: np.ndarray
    venue_mask: np.ndarray
    venue_labels: np.ndarray

    @property
    def feature_dim(self) -> int:
        return int(self.normalized_train.shape[1])


def build_examples(bundle: TrajectoryBundle, split: str) -> list[CompatTrainingExample]:
    """FIXTURE / TEST COMPAT ONLY `_FIXTURE_TEST_COMPAT_ONLY`."""

    trajectories = bundle.splits.get(split, [])
    examples: list[CompatTrainingExample] = []
    for trajectory in trajectories:
        for step in trajectory.steps:
            features = np.asarray(observation_feature_vector(step.observation), dtype=np.float32)
            action_key, venue = _best_label(step)
            examples.append(CompatTrainingExample(features=features, action_key=action_key, venue=venue))
    return examples


def prepare_training_data(bundle: TrajectoryBundle) -> CompatPreparedTrainingData:
    """FIXTURE / TEST COMPAT ONLY `_FIXTURE_TEST_COMPAT_ONLY`."""

    train_examples = build_examples(bundle, split="train")
    validation_examples = build_examples(bundle, split="validation")
    return finalize_prepared_data(
        train_examples,
        validation_examples,
        bundle.action_space.action_keys,
        bundle.dataset_spec.exchanges,
    )


def finalize_prepared_data(
    train_examples: list[CompatTrainingExample],
    validation_examples: list[CompatTrainingExample],
    action_keys: list[str],
    venue_choices: list[str],
) -> CompatPreparedTrainingData:
    """FIXTURE / TEST COMPAT ONLY `_FIXTURE_TEST_COMPAT_ONLY`."""

    if not train_examples:
        raise ValueError("train split is empty")
    if not validation_examples:
        raise ValueError("validation split is empty")

    n_train = len(train_examples)
    n_val = len(validation_examples)
    feat_dim = int(np.asarray(train_examples[0].features).shape[0])

    action_labels = np.asarray([action_keys.index(e.action_key) for e in train_examples], dtype=np.int64)
    venue_mask = np.asarray([e.venue is not None for e in train_examples], dtype=np.bool_)
    venue_labels = np.asarray(
        [venue_choices.index(e.venue) if e.venue is not None else 0 for e in train_examples],
        dtype=np.int64,
    )

    train_matrix = np.empty((n_train, feat_dim), dtype=np.float32)
    for i, example in enumerate(train_examples):
        train_matrix[i] = example.features
    del train_examples
    gc.collect()

    feature_mean = train_matrix.mean(axis=0).astype(np.float32)
    feature_std = train_matrix.std(axis=0)
    feature_std = np.where(feature_std < 1e-6, 1.0, feature_std).astype(np.float32)

    train_matrix -= feature_mean
    train_matrix /= feature_std

    val_action_labels = np.asarray([action_keys.index(e.action_key) for e in validation_examples], dtype=np.int64)
    val_matrix = np.empty((n_val, feat_dim), dtype=np.float32)
    for i, example in enumerate(validation_examples):
        val_matrix[i] = example.features
    del validation_examples
    gc.collect()
    val_matrix -= feature_mean
    val_matrix /= feature_std

    logger.info(
        "training_data_prepared_compat train_examples=%d validation_examples=%d "
        "feature_dim=%d action_count=%d venue_count=%d",
        n_train,
        n_val,
        feat_dim,
        len(action_keys),
        len(venue_choices),
    )
    return CompatPreparedTrainingData(
        train_step_count=n_train,
        val_step_count=n_val,
        action_keys=action_keys,
        venue_choices=venue_choices,
        normalized_train=train_matrix,
        normalized_val=val_matrix,
        feature_mean=feature_mean,
        feature_std=feature_std,
        action_labels=action_labels,
        val_action_labels=val_action_labels,
        venue_mask=venue_mask,
        venue_labels=venue_labels,
    )


def _best_label(step: Any) -> tuple[str, str | None]:
    abstain_reward = step.reward_snapshot.for_action("abstain").net_reward
    best_directional = None
    for reward in step.reward_snapshot.action_rewards:
        if reward.action_key == "abstain" or not reward.applicable:
            continue
        if best_directional is None or reward.net_reward > best_directional.net_reward:
            best_directional = reward
    if best_directional is None or best_directional.net_reward <= abstain_reward:
        return "abstain", None
    return best_directional.action_key, best_directional.venue


__all__ = [
    "_FIXTURE_TEST_COMPAT_ONLY",
    "CompatPreparedTrainingData",
    "CompatTrainingExample",
    "build_examples",
    "finalize_prepared_data",
    "prepare_training_data",
]

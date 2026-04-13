"""tests/test_streaming_trajectory.py — Streaming trajectory store + builder tests.

Verifies:
    1. TrajectoryDirectoryStore writes JSONL files and manifest correctly.
    2. Record line-size guard fires warn / fail as expected.
    3. iter_records() round-trips records correctly (data identity).
    4. build_to_directory() produces valid JSONL directory from fixture events.
    5. TrajectoryManifest is written alongside split files.
    6. Train streaming path (train_search_from_directory) runs without error.
    7. TrajectoryStore (compat) is NOT used in any of the above.

All tests use fixture data only — no production-profile snapshot.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from quantlab_ml.contracts import (
    TrajectoryManifest,
    TrajectoryRecord,
)
from quantlab_ml.data import LocalFixtureSource
from quantlab_ml.training import LinearPolicyTrainer
from quantlab_ml.trajectories import TrajectoryBuilder, TrajectoryDirectoryStore
from quantlab_ml.trajectories.streaming_store import (
    MANIFEST_FILENAME,
    FAIL_RECORD_LINE_MB,
    _split_filename,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_builder(fixture_path, dataset_spec, training_bundle, reward_spec):
    trajectory_spec, action_space, _ = training_bundle
    source = LocalFixtureSource(fixture_path)
    events = source.load_events(dataset_spec)
    builder = TrajectoryBuilder(dataset_spec, trajectory_spec, action_space, reward_spec)
    return builder, events


# ---------------------------------------------------------------------------
# TrajectoryDirectoryStore Unit Tests
# ---------------------------------------------------------------------------


class TestTrajectoryDirectoryStore:
    """Unit tests for streaming store write / read with synthetic minimal records."""

    def _minimal_record(self, trajectory_bundle, split_name: str = "train") -> TrajectoryRecord:
        """Extract one real record from the fixture bundle."""
        records = trajectory_bundle.splits.get(split_name) or trajectory_bundle.development_records
        assert records, f"no records in split {split_name!r}"
        return records[0]

    def test_write_and_read_roundtrip(self, tmp_path: Path, trajectory_bundle) -> None:
        """iter_records() retrieves the same record that was written."""
        record = self._minimal_record(trajectory_bundle)
        manifest = TrajectoryManifest(
            dataset_spec=trajectory_bundle.dataset_spec,
            trajectory_spec=trajectory_bundle.trajectory_spec,
            action_space=trajectory_bundle.action_space,
            reward_spec=trajectory_bundle.reward_spec,
            observation_schema=trajectory_bundle.observation_schema,
            split_artifact=trajectory_bundle.split_artifact,
            split_names=["train"],
        )

        TrajectoryDirectoryStore.write_manifest(tmp_path, manifest)
        TrajectoryDirectoryStore.write_split(tmp_path, "train", iter([record]))

        retrieved = list(TrajectoryDirectoryStore.iter_records(tmp_path, "train"))
        assert len(retrieved) == 1
        assert retrieved[0].trajectory_id == record.trajectory_id
        assert retrieved[0].target_symbol == record.target_symbol
        assert len(retrieved[0].steps) == len(record.steps)

    def test_manifest_json_written(self, tmp_path: Path, trajectory_bundle) -> None:
        """manifest.json must exist after write_manifest."""
        manifest = TrajectoryManifest(
            dataset_spec=trajectory_bundle.dataset_spec,
            trajectory_spec=trajectory_bundle.trajectory_spec,
            action_space=trajectory_bundle.action_space,
            reward_spec=trajectory_bundle.reward_spec,
            observation_schema=trajectory_bundle.observation_schema,
            split_artifact=trajectory_bundle.split_artifact,
            split_names=["train"],
        )
        TrajectoryDirectoryStore.write_manifest(tmp_path, manifest)
        mpath = tmp_path / MANIFEST_FILENAME
        assert mpath.exists()
        raw = json.loads(mpath.read_text())
        assert raw["format_version"] == "trajectories_streaming_v1"

    def test_split_exists(self, tmp_path: Path, trajectory_bundle) -> None:
        """split_exists() returns True only after writing."""
        record = self._minimal_record(trajectory_bundle)
        assert not TrajectoryDirectoryStore.split_exists(tmp_path, "train")
        TrajectoryDirectoryStore.write_split(tmp_path, "train", iter([record]))
        assert TrajectoryDirectoryStore.split_exists(tmp_path, "train")

    def test_is_trajectory_directory(self, tmp_path: Path, trajectory_bundle) -> None:
        """is_trajectory_directory() returns True only after manifest is written."""
        assert not TrajectoryDirectoryStore.is_trajectory_directory(tmp_path)
        manifest = TrajectoryManifest(
            dataset_spec=trajectory_bundle.dataset_spec,
            trajectory_spec=trajectory_bundle.trajectory_spec,
            action_space=trajectory_bundle.action_space,
            reward_spec=trajectory_bundle.reward_spec,
            observation_schema=trajectory_bundle.observation_schema,
            split_artifact=trajectory_bundle.split_artifact,
            split_names=["train"],
        )
        TrajectoryDirectoryStore.write_manifest(tmp_path, manifest)
        assert TrajectoryDirectoryStore.is_trajectory_directory(tmp_path)

    def test_multiple_records_roundtrip(self, tmp_path: Path, trajectory_bundle) -> None:
        """Multiple records written are all retrievable in order."""
        all_records = [
            r for split_records in trajectory_bundle.splits.values()
            for r in split_records
        ] + list(trajectory_bundle.development_records)
        assert len(all_records) >= 2, "Need at least 2 records for this test"
        records_to_write = all_records[:3]

        TrajectoryDirectoryStore.write_split(tmp_path, "train", iter(records_to_write))
        retrieved = list(TrajectoryDirectoryStore.iter_records(tmp_path, "train"))
        assert len(retrieved) == len(records_to_write)
        for written, read in zip(records_to_write, retrieved):
            assert written.trajectory_id == read.trajectory_id

    def test_missing_split_raises(self, tmp_path: Path) -> None:
        """iter_records() on nonexistent split raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="split file not found"):
            list(TrajectoryDirectoryStore.iter_records(tmp_path, "nonexistent"))

    def test_write_returns_counts(self, tmp_path: Path, trajectory_bundle) -> None:
        """write_split() returns correct (record_count, total_step_count)."""
        record = self._minimal_record(trajectory_bundle)
        rec_count, step_count = TrajectoryDirectoryStore.write_split(
            tmp_path, "train", iter([record])
        )
        assert rec_count == 1
        assert step_count == len(record.steps)


# ---------------------------------------------------------------------------
# Line-size guard tests
# ---------------------------------------------------------------------------


class TestLineSizeGuard:
    """Line-size guard: warn_line_mb / fail_line_mb thresholds."""

    def _minimal_record(self, trajectory_bundle) -> TrajectoryRecord:
        records = trajectory_bundle.splits.get("train") or trajectory_bundle.development_records
        return records[0]

    def test_no_warning_for_normal_size(
        self, tmp_path: Path, trajectory_bundle, caplog
    ) -> None:
        """Normal-size records (fixture) must not trigger any size warning."""
        record = self._minimal_record(trajectory_bundle)
        with caplog.at_level(logging.WARNING, logger="quantlab_ml.trajectories.streaming_store"):
            TrajectoryDirectoryStore.write_split(tmp_path, "train", iter([record]))
        size_warnings = [r for r in caplog.records if "large_record_line" in r.message]
        assert len(size_warnings) == 0

    def test_warn_threshold_triggers_log(
        self, tmp_path: Path, trajectory_bundle, caplog
    ) -> None:
        """Setting an artificially low warn threshold triggers the warning log."""
        record = self._minimal_record(trajectory_bundle)
        with caplog.at_level(logging.WARNING, logger="quantlab_ml.trajectories.streaming_store"):
            TrajectoryDirectoryStore.write_split(
                tmp_path, "train", iter([record]),
                warn_line_mb=0.001,  # 1 KB — will always trigger
                fail_line_mb=FAIL_RECORD_LINE_MB,
            )
        warnings = [r for r in caplog.records if "large_record_line" in r.message]
        assert len(warnings) >= 1
        assert "train" in warnings[0].message
        assert record.trajectory_id in warnings[0].message

    def test_fail_threshold_raises(self, tmp_path: Path, trajectory_bundle) -> None:
        """Setting fail threshold to 0 raises RuntimeError for any record."""
        record = self._minimal_record(trajectory_bundle)
        with pytest.raises(RuntimeError, match="record_line_too_large"):
            TrajectoryDirectoryStore.write_split(
                tmp_path, "train", iter([record]),
                warn_line_mb=0.0,
                fail_line_mb=0.0,  # always fails
            )

    def test_fail_message_mentions_trajectory_id(self, tmp_path: Path, trajectory_bundle) -> None:
        """RuntimeError message includes trajectory_id for diagnostics."""
        record = self._minimal_record(trajectory_bundle)
        with pytest.raises(RuntimeError) as exc_info:
            TrajectoryDirectoryStore.write_split(
                tmp_path, "train", iter([record]),
                warn_line_mb=0.0,
                fail_line_mb=0.0,
            )
        assert record.trajectory_id in str(exc_info.value)
        assert "size_mb" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Integration: build_to_directory()
# ---------------------------------------------------------------------------


class TestBuildToDirectory:
    """Integration tests for TrajectoryBuilder.build_to_directory()."""

    def test_build_creates_manifest_and_splits(
        self,
        tmp_path: Path,
        fixture_path: Path,
        dataset_spec,
        training_bundle,
        reward_spec,
    ) -> None:
        """build_to_directory() must write manifest.json + all 4 split JSONL files."""
        builder, events = _make_builder(fixture_path, dataset_spec, training_bundle, reward_spec)
        manifest = builder.build_to_directory(events, tmp_path)

        assert isinstance(manifest, TrajectoryManifest)
        assert (tmp_path / MANIFEST_FILENAME).exists()
        for split_name in ["development", "train", "validation", "final_untouched_test"]:
            assert (tmp_path / _split_filename(split_name)).exists(), \
                f"expected {split_name}.jsonl to exist"

    def test_build_is_trajectory_directory(
        self,
        tmp_path: Path,
        fixture_path: Path,
        dataset_spec,
        training_bundle,
        reward_spec,
    ) -> None:
        """Output directory is recognised by is_trajectory_directory()."""
        builder, events = _make_builder(fixture_path, dataset_spec, training_bundle, reward_spec)
        builder.build_to_directory(events, tmp_path)
        assert TrajectoryDirectoryStore.is_trajectory_directory(tmp_path)

    def test_build_records_match_bundle(
        self,
        tmp_path: Path,
        fixture_path: Path,
        dataset_spec,
        training_bundle,
        reward_spec,
        trajectory_bundle,
    ) -> None:
        """Records in train.jsonl have the same trajectory_ids as bundle.splits['train']."""
        builder, events = _make_builder(fixture_path, dataset_spec, training_bundle, reward_spec)
        builder.build_to_directory(events, tmp_path)

        dir_ids = {r.trajectory_id for r in TrajectoryDirectoryStore.iter_records(tmp_path, "train")}
        bundle_ids = {r.trajectory_id for r in trajectory_bundle.splits["train"]}
        assert dir_ids == bundle_ids

    def test_build_step_count_preserved(
        self,
        tmp_path: Path,
        fixture_path: Path,
        dataset_spec,
        training_bundle,
        reward_spec,
        trajectory_bundle,
    ) -> None:
        """Total step count in streamed train split matches in-memory bundle."""
        builder, events = _make_builder(fixture_path, dataset_spec, training_bundle, reward_spec)
        builder.build_to_directory(events, tmp_path)

        dir_steps = sum(
            len(r.steps) for r in TrajectoryDirectoryStore.iter_records(tmp_path, "train")
        )
        bundle_steps = sum(len(r.steps) for r in trajectory_bundle.splits["train"])
        assert dir_steps == bundle_steps

    def test_final_untouched_test_not_empty(
        self,
        tmp_path: Path,
        fixture_path: Path,
        dataset_spec,
        training_bundle,
        reward_spec,
    ) -> None:
        """final_untouched_test split must exist and have at least one record."""
        builder, events = _make_builder(fixture_path, dataset_spec, training_bundle, reward_spec)
        builder.build_to_directory(events, tmp_path)
        records = list(TrajectoryDirectoryStore.iter_records(tmp_path, "final_untouched_test"))
        assert len(records) >= 1, "final_untouched_test must have at least one record"

    def test_manifest_read_back(
        self,
        tmp_path: Path,
        fixture_path: Path,
        dataset_spec,
        training_bundle,
        reward_spec,
    ) -> None:
        """Manifest written by build can be read back and has correct split_names."""
        builder, events = _make_builder(fixture_path, dataset_spec, training_bundle, reward_spec)
        builder.build_to_directory(events, tmp_path)
        manifest = TrajectoryDirectoryStore.read_manifest(tmp_path)
        assert set(manifest.split_names) == {
            "development", "train", "validation", "final_untouched_test"
        }
        assert manifest.format_version == "trajectories_streaming_v1"


# ---------------------------------------------------------------------------
# Integration: train_search_from_directory()
# ---------------------------------------------------------------------------


class TestTrainSearchFromDirectory:
    """Integration: full streaming train path (fixture data)."""

    def test_train_produces_policy_artifact(
        self,
        tmp_path: Path,
        fixture_path: Path,
        dataset_spec,
        training_bundle,
        reward_spec,
    ) -> None:
        """train_search_from_directory() returns a valid PolicyArtifact."""
        trajectory_spec, action_space, training_config = training_bundle
        builder, events = _make_builder(
            fixture_path, dataset_spec,
            (trajectory_spec, action_space, training_config),
            reward_spec,
        )
        builder.build_to_directory(events, tmp_path)

        manifest = TrajectoryDirectoryStore.read_manifest(tmp_path)
        trainer = LinearPolicyTrainer(training_config)
        result = trainer.train_search_from_directory(manifest, tmp_path)

        from quantlab_ml.contracts import PolicyArtifact
        assert isinstance(result.selected_artifact, PolicyArtifact)
        assert result.selected_artifact.policy_id.startswith("policy-")

    def test_train_proxy_score_bounded(
        self,
        tmp_path: Path,
        fixture_path: Path,
        dataset_spec,
        training_bundle,
        reward_spec,
    ) -> None:
        """Proxy composite_rank must be in [0, 1)."""
        trajectory_spec, action_space, training_config = training_bundle
        builder, events = _make_builder(
            fixture_path, dataset_spec,
            (trajectory_spec, action_space, training_config),
            reward_spec,
        )
        builder.build_to_directory(events, tmp_path)

        manifest = TrajectoryDirectoryStore.read_manifest(tmp_path)
        trainer = LinearPolicyTrainer(training_config)
        result = trainer.train_search_from_directory(manifest, tmp_path)
        cr = result.candidate_results[0].best_validation_composite_rank
        assert 0.0 <= cr < 1.0, f"composite_rank out of bounds: {cr}"

    def test_streaming_artifact_ids_differ_between_runs(
        self,
        tmp_path: Path,
        fixture_path: Path,
        dataset_spec,
        training_bundle,
        reward_spec,
    ) -> None:
        """Two separate build+train runs with different seeds produce different policy_ids."""
        trajectory_spec, action_space, training_config = training_bundle
        builder, events = _make_builder(
            fixture_path, dataset_spec,
            (trajectory_spec, action_space, training_config),
            reward_spec,
        )
        build_dir_a = tmp_path / "a"
        build_dir_b = tmp_path / "b"
        builder.build_to_directory(events, build_dir_a)
        builder.build_to_directory(events, build_dir_b)

        manifest_a = TrajectoryDirectoryStore.read_manifest(build_dir_a)
        manifest_b = TrajectoryDirectoryStore.read_manifest(build_dir_b)

        config_a = training_config.model_copy(update={"seed": 1})
        config_b = training_config.model_copy(update={"seed": 42})
        trainer_a = LinearPolicyTrainer(config_a)
        trainer_b = LinearPolicyTrainer(config_b)

        result_a = trainer_a.train_search_from_directory(manifest_a, build_dir_a)
        result_b = trainer_b.train_search_from_directory(manifest_b, build_dir_b)

        assert result_a.selected_artifact.policy_id != result_b.selected_artifact.policy_id


# ---------------------------------------------------------------------------
# Fixture/test compat path marker tests
# ---------------------------------------------------------------------------


class TestCompatPathMarker:
    """Verify that TrajectoryStore is correctly marked as test/fixture compat only."""

    def test_trajectorystore_has_compat_marker(self) -> None:
        """TrajectoryStore module must expose _FIXTURE_TEST_COMPAT_ONLY sentinel."""
        from quantlab_ml.trajectories.storage import _FIXTURE_TEST_COMPAT_ONLY
        assert _FIXTURE_TEST_COMPAT_ONLY is True

    def test_trajectorystore_roundtrip_still_works(
        self, tmp_path: Path, trajectory_bundle
    ) -> None:
        """TrajectoryStore still works for fixture use (backward compat)."""
        from quantlab_ml.trajectories import TrajectoryStore
        p = tmp_path / "bundle.json"
        TrajectoryStore.write(p, trajectory_bundle)
        loaded = TrajectoryStore.read(p)
        assert loaded.dataset_spec.slice_id == trajectory_bundle.dataset_spec.slice_id

    def test_build_docstring_mentions_compat(self) -> None:
        """builder.build() docstring must mention FIXTURE / TEST COMPAT."""
        import inspect
        doc = inspect.getdoc(TrajectoryBuilder.build) or ""
        assert "COMPAT" in doc or "TEST" in doc or "fixture" in doc.lower(), \
            "build() docstring must mention fixture/test compat path"

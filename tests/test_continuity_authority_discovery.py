from __future__ import annotations

from pathlib import Path

from quantlab_ml.common import hash_payload
from quantlab_ml.registry.authority_discovery import discover_continuity_authority
from quantlab_ml.registry.store import LocalRegistryStore


def test_discovery_marks_repo_outputs_registry_as_retained_bundle_only(
    tmp_path: Path,
    trajectory_bundle,
    policy_artifact,
    training_bundle: tuple,
) -> None:
    _, _, training_config = training_bundle
    repo_root = tmp_path / "repo"
    registry_root = repo_root / "outputs" / "retained-scope" / "registry"
    store = LocalRegistryStore(registry_root)
    store.register_candidate(
        policy_artifact,
        trajectory_bundle,
        reward_config_hash=hash_payload(trajectory_bundle.reward_spec),
        training_config_hash=hash_payload(training_config),
    )

    summary = discover_continuity_authority(
        search_roots=[repo_root / "outputs"],
        repo_root=repo_root,
        include_default_search_roots=False,
    )

    assert summary["eligible_external_candidate_count"] == 0
    candidate = summary["candidates"][0]
    assert candidate["candidate_classification"] == "retained_bundle_only"
    assert candidate["is_repo_outputs_retained_bundle"] is True
    assert "candidate_is_repo_local_retained_bundle" in candidate["non_eligibility_reasons"]


def test_discovery_marks_readable_external_root_as_eligible(
    tmp_path: Path,
    trajectory_bundle,
    policy_artifact,
    training_bundle: tuple,
) -> None:
    _, _, training_config = training_bundle
    repo_root = tmp_path / "repo"
    search_root = tmp_path / "external-runs"
    run_root = search_root / "run-a"
    registry_root = run_root / "registry"
    store = LocalRegistryStore(registry_root)
    store.register_candidate(
        policy_artifact,
        trajectory_bundle,
        reward_config_hash=hash_payload(trajectory_bundle.reward_spec),
        training_config_hash=hash_payload(training_config),
    )
    for stage in ("build", "train", "evaluate", "score", "export"):
        (run_root / f"{stage}.exit").write_text("0\n", encoding="utf-8")

    summary = discover_continuity_authority(
        search_roots=[search_root],
        repo_root=repo_root,
        include_default_search_roots=False,
    )

    assert summary["decision"] == "authority_confirmation_step_allowed"
    assert summary["eligible_external_candidate_count"] == 1
    candidate = summary["candidates"][0]
    assert candidate["candidate_classification"] == "eligible_for_authority_confirmation"
    assert candidate["non_eligibility_reasons"] == []
    assert candidate["stage_exit_codes"] == {
        "build": 0,
        "train": 0,
        "evaluate": 0,
        "score": 0,
        "export": 0,
    }


def test_discovery_marks_unreadable_external_root_as_not_eligible(
    tmp_path: Path,
    trajectory_bundle,
    policy_artifact,
    training_bundle: tuple,
) -> None:
    _, _, training_config = training_bundle
    repo_root = tmp_path / "repo"
    search_root = tmp_path / "external-runs"
    run_root = search_root / "run-a"
    registry_root = run_root / "registry"
    store = LocalRegistryStore(registry_root)
    store.register_candidate(
        policy_artifact,
        trajectory_bundle,
        reward_config_hash=hash_payload(trajectory_bundle.reward_spec),
        training_config_hash=hash_payload(training_config),
    )
    artifact_path = store.artifacts_dir / f"{policy_artifact.policy_id}.json"
    artifact_path.unlink()

    summary = discover_continuity_authority(
        search_roots=[search_root],
        repo_root=repo_root,
        include_default_search_roots=False,
    )

    assert summary["decision"] == "blocked_no_eligible_external_candidates"
    candidate = summary["candidates"][0]
    assert candidate["candidate_classification"] == "not_eligible"
    assert "unreadable_active_artifact_paths" in candidate["non_eligibility_reasons"]
    assert "active_record_artifact_readability_mismatch" in candidate["non_eligibility_reasons"]


def test_discovery_blocks_when_multiple_external_candidates_are_eligible(
    tmp_path: Path,
    trajectory_bundle,
    policy_artifact,
    training_bundle: tuple,
) -> None:
    _, _, training_config = training_bundle
    repo_root = tmp_path / "repo"
    search_root = tmp_path / "external-runs"
    for run_name in ("run-a", "run-b"):
        run_root = search_root / run_name
        store = LocalRegistryStore(run_root / "registry")
        store.register_candidate(
            policy_artifact.model_copy(
                update={
                    "policy_id": f"{policy_artifact.policy_id}-{run_name}",
                    "artifact_id": f"{policy_artifact.artifact_id}-{run_name}",
                    "training_run_id": f"{policy_artifact.training_run_id}-{run_name}",
                },
                deep=True,
            ),
            trajectory_bundle,
            reward_config_hash=hash_payload(trajectory_bundle.reward_spec),
            training_config_hash=hash_payload(training_config),
        )
        for stage in ("build", "train", "evaluate", "score", "export"):
            (run_root / f"{stage}.exit").write_text("0\n", encoding="utf-8")

    summary = discover_continuity_authority(
        search_roots=[search_root],
        repo_root=repo_root,
        include_default_search_roots=False,
    )

    assert summary["decision"] == "blocked_ambiguous_external_candidates"
    assert summary["authority_confirmation_step_allowed"] is False
    assert summary["eligible_external_candidate_count"] == 2


def test_discovery_marks_nonzero_stage_exit_as_not_eligible(
    tmp_path: Path,
    trajectory_bundle,
    policy_artifact,
    training_bundle: tuple,
) -> None:
    _, _, training_config = training_bundle
    repo_root = tmp_path / "repo"
    search_root = tmp_path / "external-runs"
    run_root = search_root / "run-a"
    store = LocalRegistryStore(run_root / "registry")
    store.register_candidate(
        policy_artifact,
        trajectory_bundle,
        reward_config_hash=hash_payload(trajectory_bundle.reward_spec),
        training_config_hash=hash_payload(training_config),
    )
    (run_root / "train.exit").write_text("1\n", encoding="utf-8")

    summary = discover_continuity_authority(
        search_roots=[search_root],
        repo_root=repo_root,
        include_default_search_roots=False,
    )

    assert summary["decision"] == "blocked_no_eligible_external_candidates"
    candidate = summary["candidates"][0]
    assert candidate["candidate_classification"] == "not_eligible"
    assert "nonzero_stage_exit_code:train=1" in candidate["non_eligibility_reasons"]
    assert candidate["stage_exit_codes"] == {"train": 1}

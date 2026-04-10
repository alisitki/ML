from __future__ import annotations

from quantlab_ml.contracts import RegistryRecord


class CandidateSelector:
    def choose_champion(self, records: list[RegistryRecord]) -> str | None:
        ranked = self.rank(records)
        return ranked[0].policy_id if ranked else None

    def rank(self, records: list[RegistryRecord]) -> list[RegistryRecord]:
        scored_records = [record for record in records if record.score_history]
        return sorted(scored_records, key=_record_rank, reverse=True)


def _record_rank(record: RegistryRecord) -> float:
    if not record.score_history:
        return float("-inf")
    return record.score_history[-1].score.composite_rank

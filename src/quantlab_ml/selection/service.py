from __future__ import annotations

from quantlab_ml.contracts import RegistryRecord


class CandidateSelector:
    def rank(self, records: list[RegistryRecord]) -> list[RegistryRecord]:
        scored_records = [
            record
            for record in records
            if record.score_history and record.status not in {"rejected", "retired", "archived"}
        ]
        return sorted(scored_records, key=_record_rank, reverse=True)


def _record_rank(record: RegistryRecord) -> float:
    if not record.score_history:
        return float("-inf")
    return record.score_history[-1].composite_rank

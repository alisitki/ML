from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class QuantBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class TimeRange(QuantBaseModel):
    start: datetime
    end: datetime

    @model_validator(mode="after")
    def validate_bounds(self) -> "TimeRange":
        if self.end <= self.start:
            raise ValueError("time range end must be after start")
        return self


class NumericBand(QuantBaseModel):
    key: str
    lower: float
    upper: float

    @model_validator(mode="after")
    def validate_bounds(self) -> "NumericBand":
        if self.upper <= self.lower:
            raise ValueError("numeric band upper must be greater than lower")
        return self


class InvalidActionMaskSemantics(QuantBaseModel):
    true_means_available: bool = True
    false_reason_label: str = "constraint_violation"


class LineagePointer(QuantBaseModel):
    parent_policy_id: str | None = None
    generation: int = 0
    notes: list[str] = Field(default_factory=list)


class ChampionStatus(QuantBaseModel):
    status: Literal["candidate", "challenger", "champion", "retired"] = "candidate"

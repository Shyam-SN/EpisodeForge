"""Models for Input Validation / Dataset-level QA."""

from __future__ import annotations

import enum
from typing import Any

from pydantic import BaseModel, Field

from episodeforge.canonical.schema import QAVerdict


class CheckSeverity(enum.StrEnum):
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"


class CheckResult(BaseModel):
    check_name: str
    passed: bool
    severity: CheckSeverity = CheckSeverity.ERROR
    message: str = ""
    details: dict[str, Any] = Field(default_factory=dict)


class InputQAResult(BaseModel):
    episode_id: str
    verdict: QAVerdict
    passed: bool
    checks: list[CheckResult] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)

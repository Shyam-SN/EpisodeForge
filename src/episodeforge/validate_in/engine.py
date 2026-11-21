"""Central Input QA Engine executing all dataset validation checks on CanonicalEpisode."""

from __future__ import annotations

from episodeforge.canonical.schema import CanonicalEpisode, QAVerdict
from episodeforge.validate_in.checks.images import check_image_streams
from episodeforge.validate_in.checks.kinematics import (
    check_kinematic_limits,
    check_numeric_integrity,
    check_velocity_and_acceleration,
)
from episodeforge.validate_in.checks.timestamps import (
    check_cross_stream_alignment,
    check_monotonic_timestamps,
    check_sampling_jitter,
)
from episodeforge.validate_in.models import CheckResult, CheckSeverity, InputQAResult


class InputQAEngine:
    """Executes automated quality assurance checks on incoming raw episodes."""

    def __init__(self, reject_on_failure_flag: bool = True) -> None:
        self.reject_on_failure_flag = reject_on_failure_flag

    def validate(self, episode: CanonicalEpisode) -> InputQAResult:
        checks: list[CheckResult] = []

        # 1. Numeric integrity (NaN / Inf)
        checks.append(check_numeric_integrity(episode))

        # 2. Kinematic limits
        checks.append(check_kinematic_limits(episode))

        # 3. Dynamic bounds (velocity, acceleration)
        checks.append(check_velocity_and_acceleration(episode))

        # 4. Timestamp monotonicity
        checks.append(check_monotonic_timestamps(episode))

        # 5. Sampling jitter
        checks.append(check_sampling_jitter(episode))

        # 6. Cross-stream temporal alignment
        checks.append(check_cross_stream_alignment(episode))

        # 7. Image perceptual and visual stream health
        checks.append(check_image_streams(episode))

        # 8. Task success check (if simulated demonstration marked failed)
        if self.reject_on_failure_flag and not episode.success:
            checks.append(
                CheckResult(
                    check_name="task_success",
                    passed=False,
                    severity=CheckSeverity.ERROR,
                    message="Demonstration terminated unsuccessfully (task success criteria not met).",
                    details={"task": episode.task, "success": episode.success},
                )
            )

        # Aggregate verdict
        has_error = any(c.severity == CheckSeverity.ERROR and not c.passed for c in checks)
        has_warn = any(c.severity == CheckSeverity.WARN and not c.passed for c in checks)

        if has_error:
            verdict = QAVerdict.FAIL
            passed = False
        elif has_warn:
            verdict = QAVerdict.WARN
            passed = True
        else:
            verdict = QAVerdict.PASS
            passed = True

        reasons = [c.message for c in checks if not c.passed]

        # Extract numerical metrics
        metrics: dict[str, float] = {}
        for c in checks:
            for k, v in c.details.items():
                if isinstance(v, (int, float)):
                    metrics[k] = float(v)

        return InputQAResult(
            episode_id=episode.episode_id,
            verdict=verdict,
            passed=passed,
            checks=checks,
            reasons=reasons,
            metrics=metrics,
        )

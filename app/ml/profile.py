from __future__ import annotations


def estimate_baseline_lung(age: int, asthma: int, smoker: int, activity: str) -> float:
    score = 1.0

    if age < 18:
        score -= 0.05
    elif age > 60:
        score -= 0.10

    if asthma:
        score -= 0.12

    if smoker:
        score -= 0.10

    if activity == "exercise":
        score += 0.04
    elif activity == "jog":
        score += 0.02

    return max(0.70, min(1.15, round(score, 2)))


def sensitivity_label(baseline_lung: float) -> str:
    if baseline_lung >= 0.98:
        return "low"
    elif baseline_lung >= 0.86:
        return "moderate"
    return "high"

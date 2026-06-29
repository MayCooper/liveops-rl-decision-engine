from __future__ import annotations

import json
import logging
import time
from typing import Callable

from app.cloud_io import log_agent_audit, settings
from app.core import PlayerState, PolicyAuditFinding, PolicyAuditReport, StressScenario, safety_gate

logger = logging.getLogger(__name__)
AUDIT_AGENT_INSTRUCTION = (
    "You are the audit/explanation console for a LiveOps RL Policy Engine. You explain policy recommendations, "
    "raw vs served scores, blocked actions, safety notes, policy metrics, and rollout audit results. You do not "
    "choose player actions. You do not override the Q-learning policy. You do not override the deterministic safety "
    "mask. Keep the explanation concise, technical, and grounded only in the provided context. If the question is "
    "outside the project scope, say that the console is scoped to recommendation, safety, metrics, and rollout audit explanation."
)


def gemini_error_message(exc: Exception) -> str:
    text = str(exc).lower()
    if "resource_exhausted" in text or "prepayment" in text or "quota" in text or "429" in text:
        return "Gemini quota or billing is unavailable: credits are depleted or quota is exhausted. Deterministic fallback used."
    if "not_found" in text or "is not found" in text or "404" in text:
        return f"Gemini model is unavailable for this key. Try GEMINI_MODEL_FAST=gemini-2.5-flash. Deterministic fallback used."
    if "permission" in text or "api key" in text or "403" in text:
        return "Gemini credentials are not authorized for this API. Check the API key restrictions and enabled Gemini API. Deterministic fallback used."
    if "connection" in text or "proxy" in text or "timeout" in text:
        return "Gemini network call failed. Check proxy settings and internet access. Deterministic fallback used."
    return "Gemini call failed; deterministic fallback used."

def _is_retryable_gemini_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "resource_exhausted" in text or "429" in text or "rate" in text or "timeout" in text


def generate_gemini_text(user_message: str, context: dict, max_context_chars: int = 6000) -> str:
    if not settings.use_gemini:
        raise RuntimeError("Gemini is not configured.")

    safe_context = json.dumps(context, default=str)[:max_context_chars]
    prompt = f"{AUDIT_AGENT_INSTRUCTION}\n\nUser message:\n{user_message}\n\nProvided context JSON:\n{safe_context}"
    from app.rag_chat import _gemini_model, _message_text

    last_exc = None
    for attempt in range(3):
        try:
            return _message_text(_gemini_model().invoke(prompt))
        except Exception as exc:
            last_exc = exc
            if attempt == 2 or not _is_retryable_gemini_error(exc):
                break
            time.sleep(2 ** attempt)
    raise last_exc or RuntimeError("Gemini call failed.")


def generate_audit_summary(report: PolicyAuditReport, metrics: dict) -> dict:
    if not settings.use_gemini:
        return {"use_gemini": False}
    try:
        context = {
            "rollout_decision": report.rollout_decision,
            "human_approval_required": report.human_approval_required,
            "risky_segments": report.risky_segments,
            "allowed_segments": report.allowed_segments,
            "blocked_segments": report.blocked_segments,
            "monitoring_metrics": report.monitoring_metrics,
            "rollback_conditions": report.rollback_conditions,
            "findings": [finding.model_dump() for finding in report.findings],
            "safety_gate": metrics.get("safety_gate"),
        }
        summary = generate_gemini_text(
            "Write one concise audit_summary for this deterministic rollout audit. Do not change or reinterpret the decision fields.",
            context,
            max_context_chars=5000,
        )
        return {"use_gemini": True, "audit_summary": summary[:1200]}
    except Exception as exc:
        logger.warning("Gemini audit summary failed: %s", exc)
        return {"use_gemini": False, "gemini_error": gemini_error_message(exc)}


def fallback_stress_scenarios() -> list[StressScenario]:
    """Deterministic scenarios used locally and as Gemini fallback."""
    scenarios = [
        ("struggling_new_rewards", "New player, low skill, high frustration, repeated rewards.",
         {"segment": "new", "skill": 0.18, "frustration": 0.86, "engagement": 0.32, "churn_risk": 0.78, "economy_balance": 0.38, "recent_losses": 5, "recent_rewards": 4, "day": 6}, "bonus overuse and difficulty risk"),
        ("bored_advanced", "Advanced player with high skill and low engagement.",
         {"segment": "advanced", "skill": 0.88, "frustration": 0.22, "engagement": 0.31, "churn_risk": 0.48, "economy_balance": 0.66, "recent_losses": 1, "recent_rewards": 1, "day": 8}, "under-challenge"),
        ("mid_churn_low_economy", "Mid-skill player, high churn risk, low economy balance.",
         {"segment": "mid_skill", "skill": 0.52, "frustration": 0.63, "engagement": 0.35, "churn_risk": 0.71, "economy_balance": 0.18, "recent_losses": 3, "recent_rewards": 1, "day": 5}, "economy-sensitive intervention"),
        ("stable_player", "Stable player likely needing no intervention.",
         {"segment": "mid_skill", "skill": 0.58, "frustration": 0.24, "engagement": 0.72, "churn_risk": 0.22, "economy_balance": 0.62, "recent_losses": 0, "recent_rewards": 1, "day": 4}, "over-intervention"),
        ("reward_exposed", "Player with repeated reward exposure.",
         {"segment": "new", "skill": 0.42, "frustration": 0.49, "engagement": 0.5, "churn_risk": 0.47, "economy_balance": 0.3, "recent_losses": 2, "recent_rewards": 6, "day": 9}, "economy inflation"),
        ("difficulty_risk", "High-frustration player where increasing difficulty is risky.",
         {"segment": "new", "skill": 0.36, "frustration": 0.81, "engagement": 0.44, "churn_risk": 0.74, "economy_balance": 0.57, "recent_losses": 4, "recent_rewards": 0, "day": 3}, "difficulty escalation"),
    ]
    return [StressScenario(name=n, description=d, player=PlayerState(**p), expected_risk=r) for n, d, p, r in scenarios]


def generate_stress_scenarios(metrics: dict) -> list[StressScenario]:
    return fallback_stress_scenarios()


def _deterministic_report(metrics: dict, scenarios: list[StressScenario], recommendations: list[dict], gate: dict) -> PolicyAuditReport:
    findings = []
    risky_segments = set()
    blocked_segments = set()
    for item in recommendations:
        player = item["scenario"].player
        action = item["recommendation"]
        severity = "low"
        finding = "Recommendation is consistent with scenario risk."
        mitigation = "Monitor normal rollout metrics."
        if player.segment == "new" and action == "increase_difficulty":
            severity, finding, mitigation = "high", "Increasing difficulty for a frustrated new player can raise churn.",
            "Block this action for high-frustration new players."
            risky_segments.add("new")
            blocked_segments.add("new")
        elif action == "grant_bonus_resources" and player.recent_rewards >= 4:
            severity, finding, mitigation = "medium", "Reward-heavy player received another resource grant.",
            "Cap resource grants and watch economy penalty."
            risky_segments.add(player.segment)
        elif player.churn_risk > 0.65 and action == "do_nothing":
            severity, finding, mitigation = "medium", "High churn-risk player received no intervention.",
            "Review churn-risk thresholds before rollout."
            risky_segments.add(player.segment)
        findings.append(PolicyAuditFinding(scenario=item["scenario"].name,
                                           recommendation=action, severity=severity, finding=finding, mitigation=mitigation))

    decision = gate["decision"]
    allowed = [s for s in ["new", "mid_skill", "advanced"] if s not in blocked_segments]
    return PolicyAuditReport(
        rollout_decision=decision,
        rationale="Deterministic safety gate is the hard control. " \
        "Agent audit highlights scenario-level risks without choosing actions.",
        risky_segments=sorted(risky_segments),
        allowed_segments=allowed if decision != "reject" else [],
        blocked_segments=sorted(blocked_segments) if decision != "reject" else ["new", "mid_skill", "advanced"],
        monitoring_metrics=["avg_reward", "retention_rate", "reward_by_segment", "economy_penalty_avg", "frustration_final"],
        rollback_conditions=["new_player_reward_delta_vs_baseline < -0.02", "economy_penalty_delta_vs_baseline > 0.08", "retention_rate drops for any segment"],
        human_approval_required=decision != "approve" or any(f.severity == "high" for f in findings),
        findings=findings,
    )


def run_policy_audit(metrics: dict, recommend_fn: Callable[[dict], str]) -> PolicyAuditReport:
    """Run two-agent style stress generation and rollout audit."""
    scenarios = generate_stress_scenarios(metrics)
    recommendations = [{"scenario": s, "recommendation": recommend_fn(s.player.model_dump())} for s in scenarios]
    gate = metrics.get("safety_gate") or safety_gate(metrics.get("q_learning", metrics))
    report = _deterministic_report(metrics, scenarios, recommendations, gate)
    log_agent_audit(report.model_dump())
    return report





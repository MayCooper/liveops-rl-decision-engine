from __future__ import annotations

import copy
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np
from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.cloud_io import log_arena_event, settings
from app.core import PlayerState, clamp
from app.policies import QLearningPolicy

router = APIRouter(prefix="/arena", tags=["arena"])

ArenaAction = Literal[
    "do_nothing",
    "recommend_training_match",
    "grant_upgrade_currency",
    "offer_temporary_power_boost",
    "reduce_enemy_power",
    "recommend_recovery_match",
    "unlock_elite_match",
]

ARENA_ACTIONS: list[str] = [
    "do_nothing",
    "recommend_training_match",
    "grant_upgrade_currency",
    "offer_temporary_power_boost",
    "reduce_enemy_power",
    "recommend_recovery_match",
    "unlock_elite_match",
]

LIVEOPS_TO_ARENA = {
    "do_nothing": "do_nothing",
    "recommend_quest": "recommend_training_match",
    "grant_bonus_resources": "grant_upgrade_currency",
    "offer_cosmetic_reward": "offer_temporary_power_boost",
    "decrease_difficulty": "reduce_enemy_power",
    "increase_difficulty": "unlock_elite_match",
}

ACTION_COST = {
    "do_nothing": 0.0,
    "recommend_training_match": 0.04,
    "grant_upgrade_currency": 0.22,
    "offer_temporary_power_boost": 0.10,
    "reduce_enemy_power": 0.14,
    "recommend_recovery_match": 0.03,
    "unlock_elite_match": 0.02,
}

POLICY_RULES_PATH = Path("policies/policy_rules.json")


class ArenaState(BaseModel):
    """Game-like state used by the browser simulator.

    This is not a second agent or separate game engine. It is a controlled
    environment adapter that maps visible arena telemetry into the existing
    LiveOps PlayerState used by the Q-learning decision policy.
    """

    scenario_name: str = "custom"
    player_power: float = Field(default=800, ge=100, le=2500)
    enemy_power: float = Field(default=1000, ge=100, le=3000)
    fatigue: float = Field(default=0.25, ge=0, le=1)
    skill_score: float = Field(default=0.45, ge=0, le=1)
    consecutive_losses: int = Field(default=4, ge=0, le=20)
    attempts_on_enemy: int = Field(default=5, ge=0, le=40)
    best_completion_pct: float = Field(default=78, ge=0, le=100)
    upgrade_currency: int = Field(default=120, ge=0, le=5000)
    upgrade_cost: int = Field(default=300, ge=1, le=5000)
    recent_rewards_24h: int = Field(default=1, ge=0, le=10)
    training_affinity: float = Field(default=0.75, ge=0, le=1)
    new_player: bool = False
    advanced_player: bool = False
    temporary_power_boost_pct: float = Field(default=0.0, ge=0, le=0.50)
    enemy_power_modifier_pct: float = Field(default=0.0, ge=-0.50, le=0.50)
    elite_match_unlocked: bool = False
    day: int = Field(default=0, ge=0)


class RecommendRequest(BaseModel):
    state: ArenaState
    request_id: str | None = None


class ApplyActionRequest(BaseModel):
    state: ArenaState
    action: ArenaAction | None = None
    request_id: str | None = None


class PlayMatchRequest(BaseModel):
    state: ArenaState
    seed: int | None = 42
    request_id: str | None = None


class AutoRolloutRequest(BaseModel):
    state: ArenaState
    horizon: int = Field(default=10, ge=1, le=200)
    seed: int = 42
    request_id: str | None = None


class BenchmarkRequest(BaseModel):
    scenarios: list[str] | None = None
    episodes_per_scenario: int = Field(default=30, ge=1, le=300)
    steps_per_episode: int = Field(default=6, ge=1, le=20)
    seed: int = 42


class DerivedArenaFeatures(BaseModel):
    effective_player_power: float
    effective_enemy_power: float
    power_gap_pct: float
    win_probability: float
    frustration_score: float
    churn_risk: float
    economy_pressure: float
    reward_saturation: float
    fatigue_pressure: float
    progression_stall: float
    cold_start: bool
    history_confidence: float


class ArenaRecommendation(BaseModel):
    recommended_action: ArenaAction
    raw_recommended_action: ArenaAction
    arena_action_scores: dict[str, float]
    liveops_state: dict[str, Any]
    liveops_recommendation: dict[str, Any]
    blocked_actions: list[dict[str, str]]
    safety_notes: list[str]
    derived_features: DerivedArenaFeatures
    expected_effect: dict[str, Any]
    frustration_derivation: dict[str, Any] = Field(default_factory=dict)
    agent_trace: list[dict[str, Any]] = Field(default_factory=list)
    policy_context: dict[str, Any] = Field(default_factory=dict)
    architecture_note: str
    logged: bool = False


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, x))))


def _bounded_int(value: float, low: int = 0, high: int = 10) -> int:
    return int(max(low, min(high, round(value))))


def _copy_state(state: ArenaState) -> ArenaState:
    return ArenaState.model_validate(copy.deepcopy(state.model_dump()))


def effective_player_power(state: ArenaState) -> float:
    fatigue_penalty = 1.0 - 0.22 * state.fatigue
    boost = 1.0 + state.temporary_power_boost_pct
    return float(state.player_power * fatigue_penalty * boost)


def effective_enemy_power(state: ArenaState) -> float:
    return float(state.enemy_power * (1.0 + state.enemy_power_modifier_pct))


def estimate_win_probability(state: ArenaState) -> float:
    player = effective_player_power(state)
    enemy = max(1.0, effective_enemy_power(state))
    power_ratio = player / enemy
    near_win_signal = clamp((state.best_completion_pct - 55.0) / 45.0)
    loss_penalty = min(state.consecutive_losses, 8) * 0.055
    fatigue_penalty = state.fatigue * 0.38
    skill_bonus = (state.skill_score - 0.5) * 0.85
    logit = 2.65 * (power_ratio - 1.0) + skill_bonus + 0.45 * near_win_signal - loss_penalty - fatigue_penalty
    return float(clamp(_sigmoid(logit)))


def derive_features(state: ArenaState) -> DerivedArenaFeatures:
    player = effective_player_power(state)
    enemy = max(1.0, effective_enemy_power(state))
    power_gap_pct = (player - enemy) / enemy
    win_probability = estimate_win_probability(state)
    economy_pressure = clamp(1.0 - state.upgrade_currency / max(1.0, float(state.upgrade_cost)))
    reward_saturation = clamp(state.recent_rewards_24h / 4.0)
    fatigue_pressure = clamp(state.fatigue)
    progression_stall = clamp(
        0.45 * min(state.consecutive_losses / 6.0, 1.0)
        + 0.30 * min(state.attempts_on_enemy / 8.0, 1.0)
        + 0.25 * (1.0 - state.best_completion_pct / 100.0)
    )
    frustration_score = clamp(
        0.15
        + 0.28 * progression_stall
        + 0.25 * max(0.0, -power_gap_pct)
        + 0.18 * fatigue_pressure
        + 0.14 * min(state.consecutive_losses / 5.0, 1.0)
    )
    cold_start = state.new_player and state.attempts_on_enemy <= 1 and state.consecutive_losses == 0
    history_confidence = clamp((state.attempts_on_enemy + min(state.consecutive_losses, 6) + state.recent_rewards_24h) / 10.0)
    if cold_start:
        history_confidence = min(history_confidence, 0.2)
    engagement_proxy = clamp(0.25 + 0.32 * state.training_affinity + 0.28 * (state.best_completion_pct / 100.0) - 0.18 * state.fatigue)
    churn_risk = clamp(0.50 * frustration_score + 0.25 * (1.0 - engagement_proxy) + 0.18 * max(0.0, -power_gap_pct) + 0.07 * economy_pressure)
    return DerivedArenaFeatures(
        effective_player_power=round(player, 2),
        effective_enemy_power=round(enemy, 2),
        power_gap_pct=round(power_gap_pct, 4),
        win_probability=round(win_probability, 4),
        frustration_score=round(frustration_score, 4),
        churn_risk=round(churn_risk, 4),
        economy_pressure=round(economy_pressure, 4),
        reward_saturation=round(reward_saturation, 4),
        fatigue_pressure=round(fatigue_pressure, 4),
        progression_stall=round(progression_stall, 4),
        cold_start=cold_start,
        history_confidence=round(history_confidence, 4),
    )


def arena_to_liveops_state(state: ArenaState) -> PlayerState:
    features = derive_features(state)
    if state.new_player or features.cold_start:
        segment = "new"
    elif state.advanced_player or state.skill_score >= 0.72:
        segment = "advanced"
    else:
        segment = "mid_skill"
    engagement = clamp(0.25 + 0.35 * state.training_affinity + 0.30 * (state.best_completion_pct / 100.0) - 0.20 * state.fatigue)
    return PlayerState(
        segment=segment,
        skill=clamp(state.skill_score),
        frustration=features.frustration_score,
        engagement=engagement,
        churn_risk=features.churn_risk,
        economy_balance=clamp(1.0 - features.economy_pressure),
        recent_losses=_bounded_int(state.consecutive_losses, 0, 7),
        recent_rewards=_bounded_int(state.recent_rewards_24h, 0, 7),
        day=state.day,
    )


@lru_cache(maxsize=1)
def _load_policy() -> QLearningPolicy:
    path = Path(settings.POLICY_ARTIFACT_PATH)
    if path.exists():
        return QLearningPolicy.load(path)
    trained = QLearningPolicy().train(n_episodes=500, days=10)
    trained.save(path)
    return trained


def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    finite = {k: float(v) for k, v in scores.items() if float(v) > -1e8 and math.isfinite(float(v))}
    if not finite:
        return {action: 0.0 for action in scores}
    lo, hi = min(finite.values()), max(finite.values())
    span = hi - lo
    out: dict[str, float] = {}
    for action, value in scores.items():
        v = float(value)
        if v <= -1e8 or not math.isfinite(v):
            out[action] = -999.0
        elif span < 1e-9:
            out[action] = 0.5
        else:
            out[action] = round((v - lo) / span, 4)
    return out


def _arena_action_scores(state: ArenaState, liveops_scores: dict[str, float]) -> dict[str, float]:
    f = derive_features(state)
    mapped = {LIVEOPS_TO_ARENA[k]: float(v) for k, v in liveops_scores.items() if k in LIVEOPS_TO_ARENA}
    scores = {action: mapped.get(action, 0.0) for action in ARENA_ACTIONS}

    # Arena shaping makes the same RL policy visibly respond to game telemetry.
    scores["recommend_recovery_match"] += 0.55 * f.fatigue_pressure + 0.12 * f.churn_risk
    scores["recommend_training_match"] += 0.25 * state.training_affinity + 0.18 * f.progression_stall
    scores["offer_temporary_power_boost"] += 0.38 * max(0.0, -f.power_gap_pct) + 0.16 * max(0.0, state.best_completion_pct - 80.0) / 20.0
    scores["reduce_enemy_power"] += 0.22 * max(0.0, -f.power_gap_pct) + 0.18 * f.frustration_score + 0.08 * min(state.consecutive_losses / 5.0, 1.0)
    scores["grant_upgrade_currency"] += 0.30 * f.economy_pressure + 0.12 * f.churn_risk
    scores["unlock_elite_match"] += 0.45 * float(state.advanced_player or state.skill_score > 0.75) * max(0.0, 0.55 - f.frustration_score)

    if f.cold_start:
        scores["recommend_training_match"] += 0.55
        scores["do_nothing"] += 0.12
        scores["unlock_elite_match"] -= 0.65
        scores["reduce_enemy_power"] -= 0.35
        scores["grant_upgrade_currency"] -= 0.30
    return scores


def _policy_rules() -> dict[str, Any]:
    if POLICY_RULES_PATH.exists():
        try:
            return json.loads(POLICY_RULES_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "rules": [
            {"id": "COLD-001", "action": "unlock_elite_match", "condition": "cold_start", "decision": "block", "reason": "Do not unlock elite content for cold-start players."},
            {"id": "ECON-001", "action": "grant_upgrade_currency", "condition": "recent_rewards_24h >= 3", "decision": "block", "reason": "Reward saturation is high; avoid economy inflation."},
            {"id": "DIFF-001", "action": "unlock_elite_match", "condition": "frustration_score >= 0.65", "decision": "block", "reason": "Do not increase challenge while player frustration is high."},
            {"id": "REC-001", "action": "reduce_enemy_power", "condition": "not stuck", "decision": "block", "reason": "Difficulty reductions require repeated failure or low win probability."},
        ]
    }


def _apply_arena_safety_gate(state: ArenaState, scores: dict[str, float]) -> tuple[dict[str, float], list[dict[str, str]], list[str]]:
    f = derive_features(state)
    served = dict(scores)
    blocked: list[dict[str, str]] = []
    notes = ["Single RL decision agent: Q-learning scores the action; deterministic policy gate constrains serving."]

    def block(action: str, rule_id: str, reason: str) -> None:
        served[action] = -1e9
        blocked.append({"action": action, "rule_id": rule_id, "reason": reason})

    if f.cold_start:
        block("unlock_elite_match", "COLD-001", "Cold-start players need low-risk onboarding before elite content.")
        if state.recent_rewards_24h > 0:
            block("grant_upgrade_currency", "COLD-002", "Cold-start users should not receive repeated resource grants before behavior is observed.")
        served["recommend_training_match"] += 0.35
        notes.append("Cold-start confidence is low; safe onboarding action receives a serving boost.")

    if state.recent_rewards_24h >= 3:
        block("grant_upgrade_currency", "ECON-001", "Reward saturation is high; resource grant blocked to control economy cost.")
        block("offer_temporary_power_boost", "ECON-002", "Temporary boost is blocked when recent reward exposure is already high.")

    if f.frustration_score >= 0.65 and (state.new_player or f.churn_risk >= 0.55):
        block("unlock_elite_match", "DIFF-001", "Difficulty escalation is blocked for frustrated or high-risk players.")

    if f.win_probability >= 0.55 and state.consecutive_losses < 3:
        block("reduce_enemy_power", "DIFF-002", "Enemy reduction requires repeated failure or low win probability.")

    if f.fatigue_pressure >= 0.65:
        served["recommend_recovery_match"] += 0.35
        notes.append("High fatigue detected; recovery intervention receives a serving boost.")

    if state.advanced_player and f.frustration_score < 0.45 and f.win_probability > 0.55:
        served["unlock_elite_match"] += 0.28
        notes.append("Advanced low-frustration player can receive harder content.")

    if all(value <= -1e8 for value in served.values()):
        served["do_nothing"] = 0.0
        notes.append("All actions were blocked; deterministic fallback served do_nothing.")

    return served, blocked, notes


def _expected_effect(state: ArenaState, action: str) -> dict[str, Any]:
    before = derive_features(state)
    after_state = _apply_action_to_state(state, action, mutate_day=False)
    after = derive_features(after_state)
    return {
        "action": action,
        "win_probability_before": before.win_probability,
        "win_probability_after": after.win_probability,
        "win_probability_delta": round(after.win_probability - before.win_probability, 4),
        "frustration_before": before.frustration_score,
        "frustration_after": after.frustration_score,
        "frustration_delta": round(after.frustration_score - before.frustration_score, 4),
        "churn_risk_before": before.churn_risk,
        "churn_risk_after": after.churn_risk,
        "churn_risk_delta": round(after.churn_risk - before.churn_risk, 4),
        "estimated_cost": ACTION_COST.get(action, 0.0),
        "state_after_action": after_state.model_dump(),
    }



def _frustration_derivation(state: ArenaState, features: DerivedArenaFeatures | None = None) -> dict[str, Any]:
    """Explain how the visible frustration score is derived from telemetry.

    Frustration is intentionally not treated as a raw event in the arena demo.
    It is calculated from behavior signals that a real game backend could log.
    """
    f = features or derive_features(state)
    telemetry_signals = {
        "consecutive_losses": state.consecutive_losses,
        "attempts_on_enemy": state.attempts_on_enemy,
        "best_completion_pct": state.best_completion_pct,
        "fatigue": state.fatigue,
        "power_gap_pct": f.power_gap_pct,
        "recent_rewards_24h": state.recent_rewards_24h,
        "training_affinity": state.training_affinity,
        "history_confidence": f.history_confidence,
    }
    components = [
        {
            "name": "progression_stall",
            "value": f.progression_stall,
            "weight": 0.28,
            "explanation": "Repeated attempts, losses, and low completion indicate the player is stuck.",
        },
        {
            "name": "negative_power_gap",
            "value": round(max(0.0, -f.power_gap_pct), 4),
            "weight": 0.25,
            "explanation": "The enemy is materially stronger than the effective player power.",
        },
        {
            "name": "fatigue_pressure",
            "value": f.fatigue_pressure,
            "weight": 0.18,
            "explanation": "Fatigue/debuff pressure increases difficulty even when base power is unchanged.",
        },
        {
            "name": "loss_streak_pressure",
            "value": round(min(state.consecutive_losses / 5.0, 1.0), 4),
            "weight": 0.14,
            "explanation": "A visible loss streak increases intervention pressure.",
        },
    ]
    return {
        "derived_frustration_score": f.frustration_score,
        "base_offset": 0.15,
        "telemetry_signals": telemetry_signals,
        "components": components,
        "formula_summary": "0.15 + 0.28*progression_stall + 0.25*negative_power_gap + 0.18*fatigue + 0.14*loss_streak_pressure, clipped to [0, 1]",
    }


def _build_agent_trace(
    state: ArenaState,
    features: DerivedArenaFeatures,
    raw_scores: dict[str, float],
    served_scores: dict[str, float],
    raw_recommended: str,
    recommended: str,
    blocked: list[dict[str, Any]],
    expected: dict[str, Any],
) -> list[dict[str, Any]]:
    top_raw = sorted(raw_scores.items(), key=lambda item: item[1], reverse=True)[:3]
    top_served = sorted(served_scores.items(), key=lambda item: item[1], reverse=True)[:3]
    return [
        {
            "step": 1,
            "component": "Telemetry intake",
            "color": "blue",
            "summary": "Read player/enemy state from sliders or preset scenario.",
            "details": {
                "scenario_name": state.scenario_name,
                "player_power": state.player_power,
                "enemy_power": state.enemy_power,
                "losses": state.consecutive_losses,
                "attempts": state.attempts_on_enemy,
                "fatigue": state.fatigue,
            },
        },
        {
            "step": 2,
            "component": "Derived feature builder",
            "color": "cyan",
            "summary": "Convert raw telemetry into win probability, frustration, churn risk, and history confidence.",
            "details": features.model_dump(),
        },
        {
            "step": 3,
            "component": "RL Q-policy scoring",
            "color": "purple",
            "summary": f"Raw RL-preferred action before guardrails: {raw_recommended}.",
            "details": {"top_raw_scores": top_raw},
        },
        {
            "step": 4,
            "component": "RED safety/risk gate",
            "color": "red",
            "summary": "Deterministic guardrails block risky actions and boost safer fallbacks. This is part of the same RL decision agent flow, not a second decision agent.",
            "details": {"blocked_actions": blocked, "top_served_scores": top_served},
        },
        {
            "step": 5,
            "component": "Served action + expected effect",
            "color": "green",
            "summary": f"Served action: {recommended}.",
            "details": expected,
        },
    ]

def recommend_for_state(state: ArenaState, request_id: str | None = None, log: bool = True) -> ArenaRecommendation:
    policy = _load_policy()
    liveops_state = arena_to_liveops_state(state)
    explanation = policy.recommend_with_explanation(liveops_state.model_dump())
    source_scores = explanation.get("served_action_scores") or explanation.get("raw_action_scores") or {}
    raw_scores = _arena_action_scores(state, source_scores)
    raw_recommended = max(raw_scores, key=raw_scores.get)
    served_scores, arena_blocked, arena_notes = _apply_arena_safety_gate(state, raw_scores)
    recommended = max(served_scores, key=served_scores.get)
    expected = _expected_effect(state, recommended)
    blocked = list(explanation.get("blocked_actions", [])) + arena_blocked
    notes = list(explanation.get("safety_notes", [])) + arena_notes
    response = ArenaRecommendation(
        recommended_action=recommended,  # type: ignore[arg-type]
        raw_recommended_action=raw_recommended,  # type: ignore[arg-type]
        arena_action_scores=_normalize_scores(served_scores),
        liveops_state=liveops_state.model_dump(),
        liveops_recommendation=explanation,
        blocked_actions=blocked,
        safety_notes=notes,
        derived_features=derive_features(state),
        expected_effect=expected,
        frustration_derivation=_frustration_derivation(state, derive_features(state)),
        agent_trace=_build_agent_trace(
            state=state,
            features=derive_features(state),
            raw_scores=raw_scores,
            served_scores=served_scores,
            raw_recommended=raw_recommended,
            recommended=recommended,
            blocked=blocked,
            expected=expected,
        ),
        policy_context={
            "architecture": "one_rl_decision_agent_with_internal_policy_gate",
            "policy_rules_path": str(POLICY_RULES_PATH),
            "policy_rule_count": len(_policy_rules().get("rules", [])),
            "red_component": "deterministic safety/risk gate",
        },
        architecture_note="One learned RL decision agent serves actions. The red safety/risk gate is a deterministic component inside the serving path; explanations do not choose actions.",
        logged=False,
    )
    if log:
        response.logged = log_arena_event(
            {
                "event_type": "arena_recommendation",
                "request_id": request_id,
                "state": state.model_dump(),
                "recommended_action": response.recommended_action,
                "raw_recommended_action": response.raw_recommended_action,
                "derived_features": response.derived_features.model_dump(),
                "expected_effect": response.expected_effect,
                "blocked_actions": response.blocked_actions,
            }
        )
    return response


def _apply_action_to_state(state: ArenaState, action: str, mutate_day: bool = True) -> ArenaState:
    n = _copy_state(state)
    if action == "recommend_training_match":
        gain = int(40 + 120 * n.training_affinity)
        n.upgrade_currency = min(5000, n.upgrade_currency + gain)
        n.skill_score = clamp(n.skill_score + 0.025 * n.training_affinity)
        n.fatigue = clamp(n.fatigue + 0.025)
    elif action == "grant_upgrade_currency":
        grant = max(60, int(0.65 * n.upgrade_cost))
        n.upgrade_currency = min(5000, n.upgrade_currency + grant)
        n.recent_rewards_24h = min(10, n.recent_rewards_24h + 1)
    elif action == "offer_temporary_power_boost":
        n.temporary_power_boost_pct = max(n.temporary_power_boost_pct, 0.14)
        n.recent_rewards_24h = min(10, n.recent_rewards_24h + 1)
    elif action == "reduce_enemy_power":
        n.enemy_power_modifier_pct = min(n.enemy_power_modifier_pct, -0.12)
    elif action == "recommend_recovery_match":
        n.fatigue = clamp(n.fatigue - 0.28)
        n.consecutive_losses = max(0, n.consecutive_losses - 1)
    elif action == "unlock_elite_match":
        n.elite_match_unlocked = True
        n.enemy_power_modifier_pct = max(n.enemy_power_modifier_pct, 0.12)
    elif action == "do_nothing":
        pass
    else:
        raise HTTPException(status_code=400, detail=f"Unknown arena action: {action}")

    if n.upgrade_currency >= n.upgrade_cost and action in {"grant_upgrade_currency", "recommend_training_match"}:
        n.upgrade_currency -= n.upgrade_cost
        n.player_power = min(2500, round(n.player_power * 1.075, 2))
    if mutate_day:
        n.day += 1
    return n


def _play_match(state: ArenaState, seed: int | None = 42) -> dict[str, Any]:
    """Simulate one arena fight and return both outcome data and cinematic replay events.

    The replay event stream is intentionally richer than the minimum state update.
    The browser uses these events to render a lightweight mini-battle: attacks,
    dodges, status effects, damage numbers, HP bars, progress, and result banners.
    """
    rng = np.random.default_rng(seed)
    before = derive_features(state)
    win_probability = before.win_probability
    won = bool(rng.random() < win_probability)

    if won:
        completion = 100.0
        final_player_health = float(rng.uniform(18, 72))
        final_enemy_health = 0.0
    else:
        completion_mean = 42.0 + 55.0 * win_probability + 0.15 * state.best_completion_pct
        completion = float(clamp(rng.normal(completion_mean, 12.0), 5.0, 99.0))
        final_player_health = 0.0
        final_enemy_health = float(clamp(100.0 - completion + rng.normal(0, 8), 1.0, 95.0))

    n = _copy_state(state)
    n.attempts_on_enemy += 1
    n.best_completion_pct = round(max(n.best_completion_pct, completion), 2)
    n.day += 1
    if won:
        n.consecutive_losses = 0
        n.upgrade_currency = min(5000, n.upgrade_currency + 90 + int(60 * n.training_affinity))
        n.recent_rewards_24h = min(10, n.recent_rewards_24h + 1)
        n.skill_score = clamp(n.skill_score + 0.018)
        n.fatigue = clamp(n.fatigue + 0.035)
    else:
        n.consecutive_losses += 1
        n.upgrade_currency = min(5000, n.upgrade_currency + (25 if completion >= 80 else 10))
        n.skill_score = clamp(n.skill_score + 0.006)
        n.fatigue = clamp(n.fatigue + 0.075 + 0.025 * min(n.consecutive_losses, 5))

    n.attempts_on_enemy = min(40, n.attempts_on_enemy)
    n.consecutive_losses = min(20, n.consecutive_losses)
    n.temporary_power_boost_pct = 0.0
    n.enemy_power_modifier_pct = 0.0
    after = derive_features(n)

    # Cinematic event stream. HP values are percentages, not game-balance stats.
    player_hp = 100.0
    enemy_hp = 100.0
    player_damage_total = max(0.0, 100.0 - final_player_health)
    enemy_damage_total = max(0.0, 100.0 - final_enemy_health)
    rounds = 5 if won else 4
    replay_events: list[dict[str, Any]] = [
        {
            "t": 0.0,
            "type": "match_started",
            "message": "Boss encounter started.",
            "player_health": 100,
            "enemy_health": 100,
            "completion_pct": 0,
            "win_probability": win_probability,
            "effective_player_power": before.effective_player_power,
            "effective_enemy_power": before.effective_enemy_power,
        }
    ]

    if state.temporary_power_boost_pct > 0:
        replay_events.append(
            {
                "t": 0.06,
                "type": "boost_proc",
                "message": f"Temporary power boost active: +{round(state.temporary_power_boost_pct * 100)}% power.",
                "player_health": round(player_hp, 1),
                "enemy_health": round(enemy_hp, 1),
                "completion_pct": 2,
                "effect": "boost",
            }
        )
    if state.enemy_power_modifier_pct < 0:
        replay_events.append(
            {
                "t": 0.08,
                "type": "enemy_weakened",
                "message": f"Enemy difficulty reduced for this attempt: {round(abs(state.enemy_power_modifier_pct) * 100)}% lower power.",
                "player_health": round(player_hp, 1),
                "enemy_health": round(enemy_hp, 1),
                "completion_pct": 3,
                "effect": "debuff_enemy",
            }
        )
    if state.fatigue >= 0.65:
        replay_events.append(
            {
                "t": 0.10,
                "type": "fatigue_drag",
                "message": "High fatigue is slowing the player during the encounter.",
                "player_health": round(player_hp, 1),
                "enemy_health": round(enemy_hp, 1),
                "completion_pct": 4,
                "effect": "fatigue",
            }
        )

    for i in range(1, rounds + 1):
        progress = float(round((completion / rounds) * i, 1))
        player_share = player_damage_total / rounds
        enemy_share = enemy_damage_total / rounds

        # The hero sometimes dodges or lands a stronger hit; this changes only replay shape,
        # not the already-computed outcome state.
        critical = bool(rng.random() < clamp(0.12 + 0.18 * state.skill_score + 0.10 * state.temporary_power_boost_pct))
        dodge = bool(rng.random() < clamp(0.06 + 0.14 * state.skill_score - 0.08 * state.fatigue))
        hero_damage = float(clamp(rng.normal(enemy_share * (1.18 if critical else 1.0), 5.5), 3.0, 38.0))
        enemy_hp = max(final_enemy_health if i == rounds else 0.0, enemy_hp - hero_damage)
        replay_events.append(
            {
                "t": round(0.12 + i * 0.15, 2),
                "type": "player_attack",
                "message": "Critical strike lands." if critical else "Player attacks the boss.",
                "player_health": round(player_hp, 1),
                "enemy_health": round(enemy_hp if i < rounds else max(final_enemy_health, 0.0), 1),
                "completion_pct": round(min(progress, completion), 1),
                "damage": int(round(hero_damage)),
                "critical": critical,
                "effect": "slash",
            }
        )

        if won and i == rounds:
            break

        if dodge:
            replay_events.append(
                {
                    "t": round(0.15 + i * 0.15, 2),
                    "type": "dodge",
                    "message": "Player dodges the counterattack.",
                    "player_health": round(player_hp, 1),
                    "enemy_health": round(enemy_hp, 1),
                    "completion_pct": round(min(progress + 2, completion), 1),
                    "damage": 0,
                    "effect": "dodge",
                }
            )
            continue

        boss_damage = float(clamp(rng.normal(player_share * (1.12 if state.fatigue > 0.6 else 1.0), 5.5), 4.0, 42.0))
        player_hp = max(final_player_health if i == rounds else 0.0, player_hp - boss_damage)
        replay_events.append(
            {
                "t": round(0.19 + i * 0.15, 2),
                "type": "enemy_attack",
                "message": "Boss counterattacks." if player_hp > 0 else "Boss lands a finishing hit.",
                "player_health": round(player_hp if i < rounds else max(final_player_health, 0.0), 1),
                "enemy_health": round(enemy_hp, 1),
                "completion_pct": round(min(progress + 4, completion), 1),
                "damage": int(round(boss_damage)),
                "effect": "impact",
            }
        )

    replay_events.append(
        {
            "t": 0.94,
            "type": "match_result",
            "message": "Victory — boss defeated" if won else f"Loss at {round(completion, 1)}% completion",
            "won": won,
            "completion_pct": round(completion, 1),
            "player_health": round(final_player_health, 1),
            "enemy_health": round(final_enemy_health, 1),
            "effect": "victory" if won else "defeat",
        }
    )
    replay_events.append(
        {
            "t": 1.0,
            "type": "state_update",
            "message": "Telemetry updated: state, frustration, churn risk, and future recommendation are recalculated.",
            "frustration_before": before.frustration_score,
            "frustration_after": after.frustration_score,
            "churn_risk_before": before.churn_risk,
            "churn_risk_after": after.churn_risk,
            "win_probability_before": before.win_probability,
            "win_probability_after": after.win_probability,
        }
    )

    return {
        "won": won,
        "completion_pct": round(completion, 2),
        "player_health": round(final_player_health, 2),
        "enemy_health": round(final_enemy_health, 2),
        "win_probability_used": win_probability,
        "state_before": state.model_dump(),
        "state_after": n.model_dump(),
        "derived_before": before.model_dump(),
        "derived_after": after.model_dump(),
        "replay_events": replay_events,
    }


def _feature_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, float]:
    return {
        "win_probability": round(float(after.get("win_probability", 0)) - float(before.get("win_probability", 0)), 4),
        "frustration": round(float(after.get("frustration_score", 0)) - float(before.get("frustration_score", 0)), 4),
        "churn_risk": round(float(after.get("churn_risk", 0)) - float(before.get("churn_risk", 0)), 4),
    }


def _rollout_reason(rec: ArenaRecommendation) -> str:
    features = rec.derived_features
    action = rec.recommended_action
    if action in {"reduce_enemy_power", "recommend_recovery_match"}:
        return f"Frustration {features.frustration_score:.2f} and churn risk {features.churn_risk:.2f} made a recovery-oriented intervention preferable."
    if action == "recommend_training_match":
        return "The player still has training affinity, so the policy used a lower-risk engagement/progression action."
    if action == "grant_upgrade_currency":
        return "The policy found economy pressure high enough for a resource grant and the safety mask allowed it."
    if action == "offer_temporary_power_boost":
        return "The policy chose a short-lived boost to improve the next match without permanently changing economy balance."
    if action == "unlock_elite_match":
        return "The player appears stable or under-challenged, so the policy escalated difficulty."
    return "The player state looked stable enough that the policy did not intervene."


def _rollout_explanation(steps: list[dict[str, Any]], initial: dict[str, Any], final: dict[str, Any], final_rec: ArenaRecommendation) -> str:
    if not steps:
        return "No automatic rollout steps were run."
    first = steps[0]
    actions = [str(step.get("recommended_action", "do_nothing")) for step in steps]
    unique_actions = []
    for action in actions:
        if action not in unique_actions:
            unique_actions.append(action)
    blocked = sum(len(step.get("blocked_actions") or []) for step in steps)
    initial_features = derive_features(ArenaState.model_validate(initial))
    final_features = derive_features(ArenaState.model_validate(final))
    direction = "decreased" if final_features.churn_risk <= initial_features.churn_risk else "increased"
    return (
        f"The RL policy began with {first['recommended_action']} because {first['why'].lower()} "
        f"Across {len(steps)} automatic match-policy cycles it used {', '.join(unique_actions)} as telemetry changed after each simulated match. "
        f"The safety mask blocked {blocked} risky action(s) before serving. "
        f"By the final state, churn risk {direction} from {initial_features.churn_risk:.2f} to {final_features.churn_risk:.2f}, "
        f"frustration moved from {initial_features.frustration_score:.2f} to {final_features.frustration_score:.2f}, "
        f"and the next recommended action is {final_rec.recommended_action}."
    )


def run_auto_rl_rollout(state: ArenaState, horizon: int = 10, seed: int = 42, request_id: str | None = None) -> dict[str, Any]:
    current = _copy_state(state)
    initial_state = current.model_dump()
    steps: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    serving_policy = _load_policy().name
    for idx in range(1, horizon + 1):
        rec = recommend_for_state(current, request_id=request_id, log=False)
        telemetry_before_action = rec.derived_features.model_dump()
        action = rec.recommended_action
        action_state = _apply_action_to_state(current, action, mutate_day=False)
        telemetry_after_action = derive_features(action_state).model_dump()
        match = _play_match(action_state, seed=int(rng.integers(0, 1_000_000)))
        telemetry_after_match = dict(match["derived_after"])
        next_state = ArenaState.model_validate(match["state_after"])
        next_rec = recommend_for_state(next_state, request_id=request_id, log=False)
        steps.append(
            {
                "step": idx,
                "match_label": f"Auto match {idx}",
                "recommended_action": action,
                "applied_action": action,
                "why": _rollout_reason(rec),
                "raw_recommended_action": rec.raw_recommended_action,
                "raw_action_values": rec.liveops_recommendation.get("raw_action_scores", {}),
                "served_action_values": rec.liveops_recommendation.get("served_action_scores", {}),
                "arena_served_action_values": rec.arena_action_scores,
                "uncertainty_values": rec.liveops_recommendation.get("action_uncertainty", {}),
                "blocked_actions": rec.blocked_actions,
                "safety_notes": rec.safety_notes,
                "match_outcome": {
                    "won": match["won"],
                    "completion_pct": match["completion_pct"],
                    "win_probability_used": match["win_probability_used"],
                },
                "state_before_action": current.model_dump(),
                "state_after_action": action_state.model_dump(),
                "state_after_match": next_state.model_dump(),
                "telemetry_before_action": telemetry_before_action,
                "telemetry_after_action": telemetry_after_action,
                "telemetry_after_match": telemetry_after_match,
                "predicted_delta_from_action": _feature_delta(telemetry_before_action, telemetry_after_action),
                "actual_delta_after_match": _feature_delta(telemetry_before_action, telemetry_after_match),
                "next_recommendation_after_match": {
                    "recommended_action": next_rec.recommended_action,
                    "raw_recommended_action": next_rec.raw_recommended_action,
                    "why": _rollout_reason(next_rec),
                    "blocked_actions": next_rec.blocked_actions,
                    "safety_notes": next_rec.safety_notes,
                    "arena_action_scores": next_rec.arena_action_scores,
                    "derived_features": next_rec.derived_features.model_dump(),
                    "expected_effect": next_rec.expected_effect,
                },
            }
        )
        current = next_state
    final_rec = recommend_for_state(current, request_id=request_id, log=False)
    final_state = current.model_dump()
    explanation = _rollout_explanation(steps, initial_state, final_state, final_rec)
    blocked_total = sum(len(step.get("blocked_actions") or []) for step in steps)
    stabilized = final_rec.derived_features.churn_risk <= derive_features(ArenaState.model_validate(initial_state)).churn_risk
    response = {
        "initial_state": initial_state,
        "final_state": final_state,
        "rollout_horizon": horizon,
        "serving_policy_name": serving_policy,
        "steps": steps,
        "final_recommendation": final_rec.model_dump(),
        "final_rl_insight": {
            "stabilized_player": stabilized,
            "blocked_action_count": blocked_total,
            "next_recommended_action": final_rec.recommended_action,
            "final_churn_risk": final_rec.derived_features.churn_risk,
            "final_frustration": final_rec.derived_features.frustration_score,
        },
        "explanation": explanation,
        "architecture_note": "Inference-time rollout only: the fixed RL policy is not retrained; it adapts recommendations from updated telemetry.",
    }
    response["logged"] = False
    return response
PRESET_SCENARIOS: dict[str, ArenaState] = {
    "cold_start": ArenaState(scenario_name="cold_start", player_power=500, enemy_power=520, fatigue=0.0, skill_score=0.50, consecutive_losses=0, attempts_on_enemy=0, best_completion_pct=0, upgrade_currency=0, upgrade_cost=250, recent_rewards_24h=0, training_affinity=0.50, new_player=True),
    "new_player_stuck": ArenaState(scenario_name="new_player_stuck", player_power=780, enemy_power=1050, fatigue=0.28, skill_score=0.38, consecutive_losses=5, attempts_on_enemy=6, best_completion_pct=74, upgrade_currency=120, upgrade_cost=300, recent_rewards_24h=1, training_affinity=0.82, new_player=True),
    "underpowered_engaged": ArenaState(scenario_name="underpowered_engaged", player_power=850, enemy_power=1120, fatigue=0.18, skill_score=0.55, consecutive_losses=3, attempts_on_enemy=4, best_completion_pct=86, upgrade_currency=210, upgrade_cost=350, recent_rewards_24h=0, training_affinity=0.88),
    "fatigued_player": ArenaState(scenario_name="fatigued_player", player_power=980, enemy_power=1020, fatigue=0.78, skill_score=0.58, consecutive_losses=4, attempts_on_enemy=5, best_completion_pct=69, upgrade_currency=180, upgrade_cost=320, recent_rewards_24h=1, training_affinity=0.62),
    "advanced_bored": ArenaState(scenario_name="advanced_bored", player_power=1450, enemy_power=1050, fatigue=0.12, skill_score=0.86, consecutive_losses=0, attempts_on_enemy=2, best_completion_pct=100, upgrade_currency=640, upgrade_cost=500, recent_rewards_24h=1, training_affinity=0.35, advanced_player=True),
    "high_reward_saturation": ArenaState(scenario_name="high_reward_saturation", player_power=760, enemy_power=980, fatigue=0.32, skill_score=0.46, consecutive_losses=4, attempts_on_enemy=5, best_completion_pct=76, upgrade_currency=130, upgrade_cost=300, recent_rewards_24h=4, training_affinity=0.70, new_player=True),
    "near_win_repeated_failure": ArenaState(scenario_name="near_win_repeated_failure", player_power=920, enemy_power=1000, fatigue=0.36, skill_score=0.61, consecutive_losses=5, attempts_on_enemy=7, best_completion_pct=94, upgrade_currency=240, upgrade_cost=360, recent_rewards_24h=1, training_affinity=0.76),
}


def _rule_based_arena_action(state: ArenaState) -> str:
    f = derive_features(state)
    if f.cold_start:
        return "recommend_training_match"
    if f.fatigue_pressure > 0.65:
        return "recommend_recovery_match"
    if state.advanced_player and state.skill_score > 0.72 and f.frustration_score < 0.45:
        return "unlock_elite_match"
    if state.consecutive_losses >= 4 and f.frustration_score > 0.6:
        return "reduce_enemy_power"
    if f.economy_pressure > 0.55 and state.recent_rewards_24h < 2 and f.churn_risk > 0.5:
        return "grant_upgrade_currency"
    if f.progression_stall > 0.5 and state.training_affinity > 0.55:
        return "recommend_training_match"
    return "do_nothing"


def _choose_policy_action(policy_name: str, state: ArenaState, rng: np.random.Generator) -> str:
    if policy_name == "do_nothing":
        return "do_nothing"
    if policy_name == "random":
        return str(rng.choice(ARENA_ACTIONS))
    if policy_name == "rule_based":
        return _rule_based_arena_action(state)
    rec = recommend_for_state(state, log=False)
    if policy_name == "raw_rl":
        return rec.raw_recommended_action
    if policy_name == "safety_gated_rl":
        return rec.recommended_action
    raise ValueError(f"Unknown policy_name: {policy_name}")


def _policy_violation(state: ArenaState, action: str) -> bool:
    _, blocked, _ = _apply_arena_safety_gate(state, {a: 0.0 for a in ARENA_ACTIONS})
    return any(b["action"] == action for b in blocked)


def _episode_reward(before: DerivedArenaFeatures, after: DerivedArenaFeatures, action: str, won: bool, violation: bool) -> float:
    return float(
        0.45 * (1.0 - after.churn_risk)
        + 0.22 * after.win_probability
        + 0.18 * (before.frustration_score - after.frustration_score)
        + 0.08 * float(won)
        - ACTION_COST.get(action, 0.0)
        - (0.30 if violation else 0.0)
    )


def _run_policy_episode(policy_name: str, initial_state: ArenaState, steps: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    state = _copy_state(initial_state)
    rows: list[dict[str, Any]] = []
    total_cost = 0.0
    violations = 0
    wins = 0
    reward_sum = 0.0
    for step in range(steps):
        before_features = derive_features(state)
        action = _choose_policy_action(policy_name, state, rng)
        violation = _policy_violation(state, action)
        if violation and policy_name == "safety_gated_rl":
            action = recommend_for_state(state, log=False).recommended_action
            violation = False
        action_state = _apply_action_to_state(state, action, mutate_day=False)
        match = _play_match(action_state, seed=int(rng.integers(0, 1_000_000)))
        next_state = ArenaState.model_validate(match["state_after"])
        after_features = derive_features(next_state)
        reward = _episode_reward(before_features, after_features, action, bool(match["won"]), violation)
        total_cost += ACTION_COST.get(action, 0.0)
        violations += int(violation)
        wins += int(bool(match["won"]))
        reward_sum += reward
        rows.append(
            {
                "step": step,
                "policy_name": policy_name,
                "scenario_name": initial_state.scenario_name,
                "action": action,
                "policy_violation": violation,
                "won": bool(match["won"]),
                "completion_pct": match["completion_pct"],
                "reward": round(reward, 4),
                "win_probability_before": before_features.win_probability,
                "win_probability_after": after_features.win_probability,
                "frustration_before": before_features.frustration_score,
                "frustration_after": after_features.frustration_score,
                "churn_risk_before": before_features.churn_risk,
                "churn_risk_after": after_features.churn_risk,
                "cost": ACTION_COST.get(action, 0.0),
            }
        )
        state = next_state
    final_features = derive_features(state)
    return {
        "policy_name": policy_name,
        "scenario_name": initial_state.scenario_name,
        "rows": rows,
        "win_rate": wins / max(1, steps),
        "avg_reward": reward_sum / max(1, steps),
        "avg_cost": total_cost / max(1, steps),
        "policy_violations": violations,
        "final_frustration": final_features.frustration_score,
        "final_churn_risk": final_features.churn_risk,
        "final_win_probability": final_features.win_probability,
        "final_state": state.model_dump(),
    }


@router.get("/dashboard", include_in_schema=False)
def arena_dashboard() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "arena.html")


@router.get("/presets")
def presets() -> dict[str, Any]:
    return {
        "architecture_note": "One RL decision agent recommends interventions. The deterministic safety gate and optional explanation layer are not second decision agents.",
        "actions": ARENA_ACTIONS,
        "presets": {name: state.model_dump() for name, state in PRESET_SCENARIOS.items()},
    }


@router.get("/policy_rules")
def policy_rules() -> dict[str, Any]:
    return _policy_rules()


@router.get("/dataset_profile")
def arena_dataset_profile() -> dict[str, Any]:
    path = Path("data/arena_dataset_profile.json")
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"available": False, "message": "data/arena_dataset_profile.json not found."}


@router.post("/recommend", response_model=ArenaRecommendation)
def recommend(request: RecommendRequest) -> ArenaRecommendation:
    return recommend_for_state(request.state, request_id=request.request_id, log=True)


@router.post("/agent_trace")
def agent_trace(request: RecommendRequest) -> dict[str, Any]:
    """Return the single-agent decision path for a given arena state."""
    rec = recommend_for_state(request.state, request_id=request.request_id, log=False)
    return {
        "architecture": "one_rl_decision_agent",
        "red_component": "deterministic safety/risk gate",
        "recommended_action": rec.recommended_action,
        "raw_recommended_action": rec.raw_recommended_action,
        "derived_features": rec.derived_features.model_dump(),
        "frustration_derivation": rec.frustration_derivation,
        "agent_trace": rec.agent_trace,
        "policy_context": rec.policy_context,
        "blocked_actions": rec.blocked_actions,
    }


@router.post("/apply_action")
def apply_action(request: ApplyActionRequest) -> dict[str, Any]:
    action = request.action or recommend_for_state(request.state, request_id=request.request_id, log=False).recommended_action
    before = derive_features(request.state)
    after_state = _apply_action_to_state(request.state, action)
    after = derive_features(after_state)
    logged = log_arena_event(
        {
            "event_type": "arena_action_applied",
            "request_id": request.request_id,
            "action": action,
            "state_before": request.state.model_dump(),
            "state_after": after_state.model_dump(),
            "derived_before": before.model_dump(),
            "derived_after": after.model_dump(),
        }
    )
    return {"action": action, "state_before": request.state.model_dump(), "state_after": after_state.model_dump(), "derived_before": before.model_dump(), "derived_after": after.model_dump(), "logged": logged}


@router.post("/play_match")
def play_match(request: PlayMatchRequest) -> dict[str, Any]:
    result = _play_match(request.state, seed=request.seed)
    result["logged"] = log_arena_event({"event_type": "arena_match_result", "request_id": request.request_id, **result})
    return result


@router.post("/auto_rl_rollout")
def auto_rl_rollout(request: AutoRolloutRequest) -> dict[str, Any]:
    return run_auto_rl_rollout(request.state, horizon=request.horizon, seed=request.seed, request_id=request.request_id)


@router.post("/benchmark")
def benchmark(request: BenchmarkRequest = Body(default_factory=BenchmarkRequest)) -> dict[str, Any]:
    scenario_names = request.scenarios or list(PRESET_SCENARIOS.keys())
    unknown = [name for name in scenario_names if name not in PRESET_SCENARIOS]
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown scenarios: {', '.join(unknown)}")
    policies = ["do_nothing", "random", "rule_based", "raw_rl", "safety_gated_rl"]
    rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for s_idx, scenario_name in enumerate(scenario_names):
        initial_state = PRESET_SCENARIOS[scenario_name]
        for p_idx, policy_name in enumerate(policies):
            episode_results = []
            for ep in range(request.episodes_per_scenario):
                seed = request.seed + 100_000 * s_idx + 1_000 * p_idx + ep
                episode_results.append(_run_policy_episode(policy_name, initial_state, request.steps_per_episode, seed))
            rows_for_policy = [r for ep_result in episode_results for r in ep_result["rows"]]
            summary = {
                "scenario_name": scenario_name,
                "policy_name": policy_name,
                "episodes": request.episodes_per_scenario,
                "steps_per_episode": request.steps_per_episode,
                "win_rate": round(float(np.mean([e["win_rate"] for e in episode_results])), 4),
                "avg_reward": round(float(np.mean([e["avg_reward"] for e in episode_results])), 4),
                "avg_cost": round(float(np.mean([e["avg_cost"] for e in episode_results])), 4),
                "policy_violations": int(sum(e["policy_violations"] for e in episode_results)),
                "final_frustration": round(float(np.mean([e["final_frustration"] for e in episode_results])), 4),
                "final_churn_risk": round(float(np.mean([e["final_churn_risk"] for e in episode_results])), 4),
                "final_win_probability": round(float(np.mean([e["final_win_probability"] for e in episode_results])), 4),
            }
            rows.append(summary)
            details.append({"scenario_name": scenario_name, "policy_name": policy_name, "trace_sample": rows_for_policy[: min(8, len(rows_for_policy))]})
    aggregate = []
    for policy_name in policies:
        subset = [r for r in rows if r["policy_name"] == policy_name]
        aggregate.append(
            {
                "policy_name": policy_name,
                "win_rate": round(float(np.mean([r["win_rate"] for r in subset])), 4),
                "avg_reward": round(float(np.mean([r["avg_reward"] for r in subset])), 4),
                "avg_cost": round(float(np.mean([r["avg_cost"] for r in subset])), 4),
                "policy_violations": int(sum(r["policy_violations"] for r in subset)),
                "final_frustration": round(float(np.mean([r["final_frustration"] for r in subset])), 4),
                "final_churn_risk": round(float(np.mean([r["final_churn_risk"] for r in subset])), 4),
                "final_win_probability": round(float(np.mean([r["final_win_probability"] for r in subset])), 4),
            }
        )
    result = {"aggregate": aggregate, "by_scenario": rows, "details": details, "scenarios": scenario_names, "policies": policies}
    result["logged"] = log_arena_event({"event_type": "arena_benchmark_run", "request": request.model_dump(), "aggregate": aggregate})
    return result


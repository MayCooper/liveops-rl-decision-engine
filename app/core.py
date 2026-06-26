from __future__ import annotations

from typing import Any, Callable, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


# Schemas

Action = Literal[
    "do_nothing",
    "decrease_difficulty",
    "increase_difficulty",
    "recommend_quest",
    "grant_bonus_resources",
    "offer_cosmetic_reward",
]
Segment = Literal["new", "mid_skill", "advanced"]


class PlayerState(BaseModel):
    segment: Segment
    skill: float = Field(ge=0, le=1)
    frustration: float = Field(ge=0, le=1)
    engagement: float = Field(ge=0, le=1)
    churn_risk: float = Field(ge=0, le=1)
    economy_balance: float = Field(ge=0, le=1)
    recent_losses: int = Field(ge=0)
    recent_rewards: int = Field(ge=0)
    day: int = Field(ge=0)


class RecommendationRequest(BaseModel):
    player: PlayerState
    request_id: str | None = None


class RecommendationResponse(BaseModel):
    recommended_action: Action
    policy_version: str
    raw_action_scores: dict[str, float] = Field(default_factory=dict)
    served_action_scores: dict[str, float] = Field(default_factory=dict)
    blocked_actions: list[dict[str, str]] = Field(default_factory=list)
    safety_notes: list[str] = Field(default_factory=list)
    logged: bool


class PolicyMetrics(BaseModel):
    rule: dict[str, Any]
    bandit: dict[str, Any]
    q_learning: dict[str, Any]
    safety_gate: dict[str, Any]


class StressScenario(BaseModel):
    name: str
    description: str
    player: PlayerState
    expected_risk: str


class PolicyAuditFinding(BaseModel):
    scenario: str
    recommendation: Action | None = None
    severity: Literal["low", "medium", "high"]
    finding: str
    mitigation: str


class PolicyAuditReport(BaseModel):
    rollout_decision: Literal["approve", "limited_rollout", "reject"]
    rationale: str
    risky_segments: list[str]
    allowed_segments: list[str]
    blocked_segments: list[str]
    monitoring_metrics: list[str]
    rollback_conditions: list[str]
    human_approval_required: bool
    findings: list[PolicyAuditFinding]


# Simulation

ACTIONS = [
    "do_nothing",
    "decrease_difficulty",
    "increase_difficulty",
    "recommend_quest",
    "grant_bonus_resources",
    "offer_cosmetic_reward",
]


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return float(max(low, min(high, value)))


def create_initial_player(seed: int | None = None) -> dict:
    """Create a plausible initial player state."""
    rng = np.random.default_rng(seed)
    segment = rng.choice(["new", "mid_skill", "advanced"], p=[0.35, 0.4, 0.25])
    if segment == "new":
        skill, engagement, frustration = rng.beta(2, 5), rng.beta(3, 3), rng.beta(3, 3)
    elif segment == "mid_skill":
        skill, engagement, frustration = rng.beta(4, 4), rng.beta(4, 3), rng.beta(2, 4)
    else:
        skill, engagement, frustration = rng.beta(6, 2), rng.beta(3, 3), rng.beta(2, 5)
    churn_risk = clamp(0.35 * frustration + 0.35 * (1 - engagement) + rng.normal(0, 0.08))
    return {
        "segment": str(segment),
        "skill": clamp(skill),
        "frustration": clamp(frustration),
        "engagement": clamp(engagement),
        "churn_risk": churn_risk,
        "economy_balance": clamp(rng.beta(3, 3)),
        "recent_losses": int(rng.integers(0, 4)),
        "recent_rewards": int(rng.integers(0, 3)),
        "day": 0,
    }


def compute_reward(before: dict, after: dict, action: str, economy_penalty: float) -> float:
    """Reward balances retention, engagement, frustration, economy health, and intervention cost."""
    retention_benefit = 1.0 - after["churn_risk"]
    engagement_benefit = after["engagement"]
    frustration_reduction = before["frustration"] - after["frustration"]
    intervention_penalty = 0.025 if action != "do_nothing" else 0.0
    if action == "grant_bonus_resources":
        if before["recent_rewards"] >= 3:
            intervention_penalty += 0.1
        if before["churn_risk"] < 0.55 or before["engagement"] > 0.65:
            intervention_penalty += 0.14
    return float(
        0.45 * retention_benefit
        + 0.35 * engagement_benefit
        + 0.25 * frustration_reduction
        - 0.65 * economy_penalty
        - intervention_penalty
    )


def step_player(state: dict | PlayerState, action: str, rng: np.random.Generator) -> tuple[dict, float, dict]:
    """Advance one simulated day after applying a LiveOps action."""
    s = state.model_dump() if isinstance(state, PlayerState) else dict(state)
    n = dict(s)
    noise = lambda scale=0.025: float(rng.normal(0, scale))
    economy_penalty = 0.0

    if action == "decrease_difficulty":
        n["frustration"] -= 0.2 if s["segment"] == "new" else 0.05
        n["engagement"] += 0.07 if s["segment"] == "new" else -0.03
        if s["segment"] == "advanced":
            n["engagement"] -= 0.1
    elif action == "increase_difficulty":
        n["engagement"] += 0.11 if s["segment"] == "advanced" else -0.06
        if s["segment"] == "new":
            n["frustration"] += 0.12
        elif s["frustration"] > 0.6:
            n["frustration"] += 0.025
        else:
            n["frustration"] += 0.02
        n["skill"] += 0.03
    elif action == "recommend_quest":
        n["engagement"] += 0.1 if s["segment"] == "mid_skill" else 0.03
        n["frustration"] -= 0.04
    elif action == "grant_bonus_resources":
        n["engagement"] += 0.09
        n["frustration"] -= 0.05
        n["economy_balance"] -= 0.08 + 0.03 * min(s["recent_rewards"], 4)
        n["recent_rewards"] += 1
        economy_penalty = 0.12 + 0.07 * min(s["recent_rewards"], 5)
    elif action == "offer_cosmetic_reward":
        n["engagement"] += 0.08 if s["engagement"] > 0.45 and s["churn_risk"] < 0.75 else 0.015
        n["recent_rewards"] += 1
        economy_penalty = 0.015
    else:
        if s["churn_risk"] > 0.65:
            n["engagement"] -= 0.07
            n["frustration"] += 0.05

    loss_pressure = 0.012 * s["recent_losses"] + (0.03 if s["skill"] < 0.35 else 0)
    recovery = 0.04 * max(0, s["engagement"] - 0.6) + 0.035 * max(0, s["skill"] - 0.65)
    if action in {"decrease_difficulty", "recommend_quest", "grant_bonus_resources"}:
        recovery += 0.03
    n["frustration"] = clamp(n["frustration"] + loss_pressure - recovery + noise())
    n["engagement"] = clamp(n["engagement"] + 0.04 * (n["skill"] - 0.5) - 0.03 * n["frustration"] + noise())
    n["skill"] = clamp(n["skill"] + 0.015 + noise(0.012))
    n["economy_balance"] = clamp(n["economy_balance"] + 0.02 - 0.015 * n["recent_rewards"] + noise(0.015))
    n["churn_risk"] = clamp(0.48 * n["frustration"] + 0.42 * (1 - n["engagement"]) + 0.1 * (1 - n["economy_balance"]) + noise())
    n["recent_losses"] = int(max(0, min(7, round(s["recent_losses"] + rng.choice([-1, 0, 1], p=[0.35, 0.4, 0.25]) + n["frustration"]))))
    n["recent_rewards"] = int(max(0, min(7, round(n["recent_rewards"] - 0.25 + rng.choice([0, 1], p=[0.8, 0.2])))))
    n["day"] = int(s["day"]) + 1

    reward = compute_reward(s, n, action, economy_penalty)
    retained = bool(rng.random() > n["churn_risk"] * 0.08)
    return n, reward, {"retained": retained, "economy_penalty": float(economy_penalty)}


def simulate_episode(policy_fn: Callable[[dict], str], days: int = 14, seed: int | None = None, player_id: int = 0, policy_name: str = "policy") -> list[dict]:
    """Simulate one player episode."""
    rng = np.random.default_rng(seed)
    state = create_initial_player(seed)
    rows = []
    for _ in range(days):
        before = dict(state)
        action = policy_fn(before)
        next_state, reward, info = step_player(before, action, rng)
        rows.append(
            {
                "player_id": player_id,
                "day": before["day"],
                **before,
                "action": action,
                "reward": reward,
                **{f"next_{k}": v for k, v in next_state.items() if k != "day"},
                "retained": info["retained"],
                "economy_penalty": info["economy_penalty"],
                "policy_name": policy_name,
            }
        )
        state = next_state
        if not info["retained"]:
            break
    return rows


def simulate_dataset(policy_fn: Callable[[dict], str], n_players: int = 500, days: int = 14, seed: int = 42, policy_name: str = "policy") -> pd.DataFrame:
    """Simulate multiple player episodes into a flat table."""
    rows = []
    for i in range(n_players):
        rows.extend(simulate_episode(policy_fn, days=days, seed=seed + i, player_id=i, policy_name=policy_name))
    return pd.DataFrame(rows)


# Evaluation


def _policy_fn(policy):
    return policy if callable(policy) else policy.recommend


def segment_metrics(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Summarize reward, retention, and economy by player segment."""
    out = {}
    for segment, g in df.groupby("segment"):
        last = g.sort_values("day").groupby("player_id").tail(1)
        out[segment] = {
            "avg_reward": float(g["reward"].mean()),
            "retention_rate": float(last["retained"].mean()),
            "economy_penalty_avg": float(g["economy_penalty"].mean()),
        }
    return out


def evaluate_policy(policy, n_players: int = 300, days: int = 14, seed: int = 123) -> dict:
    """Run an offline simulator evaluation for one policy."""
    name = getattr(policy, "name", "policy")
    df = simulate_dataset(_policy_fn(policy), n_players=n_players, days=days, seed=seed, policy_name=name)
    last = df.sort_values("day").groupby("player_id").tail(1)
    return {
        "policy_name": name,
        "avg_reward": float(df["reward"].mean()),
        "retention_rate": float(last["retained"].mean()),
        "avg_engagement_final": float(last["next_engagement"].mean()),
        "avg_frustration_final": float(last["next_frustration"].mean()),
        "economy_penalty_avg": float(df["economy_penalty"].mean()),
        "reward_by_segment": {k: v["avg_reward"] for k, v in segment_metrics(df).items()},
        "segment_metrics": segment_metrics(df),
    }


def _with_deltas(candidate: dict, baseline: dict) -> dict:
    result = dict(candidate)
    result["reward_delta_vs_baseline"] = candidate["avg_reward"] - baseline["avg_reward"]
    result["new_player_reward_delta_vs_baseline"] = candidate["reward_by_segment"].get("new", 0) - baseline["reward_by_segment"].get("new", 0)
    result["advanced_reward_delta_vs_baseline"] = candidate["reward_by_segment"].get("advanced", 0) - baseline["reward_by_segment"].get("advanced", 0)
    result["economy_penalty_delta_vs_baseline"] = candidate["economy_penalty_avg"] - baseline["economy_penalty_avg"]
    return result


def safety_gate(metrics: dict) -> dict:
    """Hard rollout gate for the Q-learning policy."""
    reasons = []
    decision = "approve"
    if metrics["new_player_reward_delta_vs_baseline"] < -0.02:
        decision = "reject"
        reasons.append("New-player reward delta is below -0.02.")
    if metrics["economy_penalty_delta_vs_baseline"] > 0.08:
        decision = "reject"
        reasons.append("Economy penalty delta is above 0.08.")
    worsened = [s for s, delta in {
        "new": metrics.get("new_player_reward_delta_vs_baseline", 0),
        "advanced": metrics.get("advanced_reward_delta_vs_baseline", 0),
    }.items() if delta < -0.01]
    if decision != "reject" and metrics["reward_delta_vs_baseline"] > 0 and worsened:
        decision = "limited_rollout"
        reasons.append(f"Overall reward improved but segment worsened: {', '.join(worsened)}.")
    if decision != "reject" and metrics["reward_delta_vs_baseline"] <= 0:
        decision = "limited_rollout"
        reasons.append("Q-learning did not clearly improve overall reward.")
    if not reasons:
        reasons.append("Overall reward improved with no material segment or economy regression.")
    return {"decision": decision, "allow": decision == "approve", "limited": decision == "limited_rollout", "reject": decision == "reject", "reasons": reasons}



def _empty_ope(message: str, dataset: str | None = None, warnings: list[str] | None = None) -> dict:
    return {
        "available": False,
        "message": message,
        "candidate_policy": "q_learning",
        "dataset": dataset,
        "warnings": warnings or [],
        "trace_examples": [],
    }


def _safe_float(value) -> float:
    if pd.isna(value):
        raise ValueError("NaN value")
    out = float(value)
    if not np.isfinite(out):
        raise ValueError("Non-finite value")
    return out


def _safe_int(value) -> int:
    return int(_safe_float(value))


def _segment_ope(group: pd.DataFrame, clip_weight: float) -> dict:
    weights = group.loc[group["match"], "importance_weight"].astype(float)
    weighted_rewards = group.loc[group["match"], "ips_contribution"].astype(float)
    matched_rows = int(group["match"].sum())
    total_rows = int(len(group))
    ess = float((weights.sum() ** 2) / (weights.pow(2).sum())) if matched_rows and weights.pow(2).sum() > 0 else 0.0
    return {
        "snips_estimate": float(weighted_rewards.sum() / weights.sum()) if weights.sum() > 0 else 0.0,
        "match_rate": float(matched_rows / total_rows) if total_rows else 0.0,
        "effective_sample_size": ess,
        "matched_rows": matched_rows,
        "total_rows": total_rows,
    }


def evaluate_off_policy(candidate_policy, logged_df: pd.DataFrame, clip_weight: float = 10.0, trace_limit: int = 8) -> dict:
    """Estimate served candidate-policy value from logged behavior-policy data."""
    dataset_name = "data/arena_liveops_episodes.csv"
    required = [
        "segment",
        "skill",
        "frustration",
        "engagement",
        "churn_risk",
        "economy_balance",
        "recent_losses",
        "recent_rewards",
        "day",
        "action",
        "action_probability",
        "reward",
    ]
    if logged_df is None or logged_df.empty:
        return _empty_ope("OPE could not be computed because the logged dataset is empty.", dataset_name)
    missing = [col for col in required if col not in logged_df.columns]
    if missing:
        return _empty_ope("OPE could not be computed because required columns are missing.", dataset_name, [f"Missing columns: {', '.join(missing)}"])

    rows = []
    trace_examples = []
    warnings: list[str] = []
    invalid_rows = 0
    trace_limit = max(0, int(trace_limit))

    for _, row in logged_df.iterrows():
        try:
            action_probability = _safe_float(row["action_probability"])
            reward = _safe_float(row["reward"])
            if action_probability <= 0:
                invalid_rows += 1
                continue
            state = {
                "segment": str(row["segment"]),
                "skill": _safe_float(row["skill"]),
                "frustration": _safe_float(row["frustration"]),
                "engagement": _safe_float(row["engagement"]),
                "churn_risk": _safe_float(row["churn_risk"]),
                "economy_balance": _safe_float(row["economy_balance"]),
                "recent_losses": _safe_int(row["recent_losses"]),
                "recent_rewards": _safe_int(row["recent_rewards"]),
                "day": _safe_int(row["day"]),
            }
            logged_action = str(row["action"])
            candidate_action = candidate_policy.recommend(state)
            match = candidate_action == logged_action
            weight = float(1.0 / action_probability) if match else 0.0
            clipped_weight = float(min(weight, clip_weight)) if match else 0.0
            ips_contribution = float(weight * reward) if match else 0.0
            clipped_ips_contribution = float(clipped_weight * reward) if match else 0.0
            record = {
                "player_id": str(row.get("player_id", "")),
                "day": state["day"],
                "segment": state["segment"],
                "logged_action": logged_action,
                "candidate_action": candidate_action,
                "match": bool(match),
                "action_probability": action_probability,
                "importance_weight": weight,
                "clipped_weight": clipped_weight,
                "reward": reward,
                "ips_contribution": ips_contribution,
                "clipped_ips_contribution": clipped_ips_contribution,
            }
            rows.append(record)
            if len(trace_examples) < trace_limit:
                trace_examples.append(record)
        except Exception:
            invalid_rows += 1
            continue

    if invalid_rows:
        warnings.append(f"Skipped {invalid_rows} rows with missing, non-finite, or non-positive propensity values.")
    if not rows:
        return _empty_ope("OPE could not be computed because no valid logged rows were available.", dataset_name, warnings)

    out = pd.DataFrame(rows)
    total_rows = int(len(out))
    matched_rows = int(out["match"].sum())
    weights = out.loc[out["match"], "importance_weight"].astype(float)
    ips_sum = float(out["ips_contribution"].sum())
    clipped_ips_sum = float(out["clipped_ips_contribution"].sum())
    weight_sum = float(weights.sum())
    weight_sq_sum = float(weights.pow(2).sum()) if matched_rows else 0.0
    ess = float((weight_sum**2) / weight_sq_sum) if weight_sq_sum > 0 else 0.0
    by_segment = {segment: _segment_ope(group, clip_weight) for segment, group in out.groupby("segment")}

    return {
        "available": True,
        "candidate_policy": getattr(candidate_policy, "name", "q_learning"),
        "dataset": dataset_name,
        "ips_estimate": float(ips_sum / total_rows) if total_rows else 0.0,
        "snips_estimate": float(ips_sum / weight_sum) if weight_sum > 0 else 0.0,
        "clipped_ips_estimate": float(clipped_ips_sum / total_rows) if total_rows else 0.0,
        "match_rate": float(matched_rows / total_rows) if total_rows else 0.0,
        "effective_sample_size": ess,
        "matched_rows": matched_rows,
        "total_rows": total_rows,
        "avg_logged_reward": float(out["reward"].mean()),
        "clip_weight": float(clip_weight),
        "ope_by_segment": by_segment,
        "trace_examples": trace_examples,
        "warnings": warnings,
    }

def compare_policies(rule_policy, bandit_policy, q_policy) -> dict:
    """Evaluate rule, bandit, and Q-learning policies and attach baseline deltas."""
    rule = evaluate_policy(rule_policy)
    bandit = _with_deltas(evaluate_policy(bandit_policy), rule)
    q_learning = _with_deltas(evaluate_policy(q_policy), rule)
    return {"rule": rule, "bandit": bandit, "q_learning": q_learning, "safety_gate": safety_gate(q_learning)}


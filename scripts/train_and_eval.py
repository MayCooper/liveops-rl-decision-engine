from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.cloud_io import log_eval_results, settings
from app.core import compare_policies, evaluate_off_policy, simulate_dataset
from app.policies import ACTIONS, BanditPolicy, QLearningPolicy, RuleBasedPolicy

ARENA_TO_LIVEOPS = {
    "do_nothing": "do_nothing",
    "recommend_training_match": "recommend_quest",
    "grant_upgrade_currency": "grant_bonus_resources",
    "offer_temporary_power_boost": "offer_cosmetic_reward",
    "reduce_enemy_power": "decrease_difficulty",
    "recommend_recovery_match": "recommend_quest",
    "unlock_elite_match": "increase_difficulty",
}


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return float(max(low, min(high, value)))


def _engagement(row: pd.Series, prefix: str = "") -> float:
    best = float(row.get(f"{prefix}best_completion_pct", row.get("best_completion_pct", 0))) / 100.0
    fatigue = float(row.get(f"{prefix}fatigue", row.get("fatigue", 0)))
    affinity = float(row.get("training_affinity", 0.5))
    return _clip(0.25 + 0.35 * affinity + 0.30 * best - 0.20 * fatigue)


def _segment(row: pd.Series) -> str:
    if bool(row.get("new_player", False)):
        return "new"
    if bool(row.get("advanced_player", False)) or float(row.get("skill_score", 0)) >= 0.72:
        return "advanced"
    return "mid_skill"


def arena_to_liveops_df(df: pd.DataFrame) -> pd.DataFrame:
    """Map arena telemetry into the compact LiveOps schema used by the tabular policies."""
    rows = []
    for _, row in df.iterrows():
        action = ARENA_TO_LIVEOPS.get(str(row.get("action")))
        if action not in ACTIONS:
            continue
        upgrade_cost = max(1.0, float(row.get("upgrade_cost", 1)))
        next_upgrade_currency = float(row.get("next_upgrade_currency", row.get("upgrade_currency", 0)))
        rows.append(
            {
                "player_id": row.get("player_id"),
                "day": int(row.get("episode_step", row.get("day", 0))),
                "segment": _segment(row),
                "skill": _clip(float(row.get("skill_score", 0.5))),
                "frustration": _clip(float(row.get("frustration_score", row.get("model_feature_frustration_score", 0.5)))),
                "engagement": _engagement(row),
                "churn_risk": _clip(float(row.get("churn_risk", row.get("observed_churn_risk", 0.5)))),
                "economy_balance": _clip(1.0 - float(row.get("economy_pressure", 0.5))),
                "recent_losses": int(max(0, min(7, round(float(row.get("consecutive_losses", 0)))))),
                "recent_rewards": int(max(0, min(7, round(float(row.get("recent_rewards_24h", 0)))))),
                "action": action,
                "action_probability": float(row.get("action_probability", 1.0 / len(ACTIONS))),
                "reward": float(row.get("reward", 0.0)),
                "next_skill": _clip(float(row.get("next_skill_score", row.get("skill_score", 0.5)))),
                "next_frustration": _clip(float(row.get("next_model_feature_frustration_score", row.get("frustration_score", 0.5)))),
                "next_engagement": _engagement(row, "next_"),
                "next_churn_risk": _clip(float(row.get("next_churn_risk", row.get("churn_risk", 0.5)))),
                "next_economy_balance": _clip(next_upgrade_currency / upgrade_cost),
                "next_recent_losses": int(max(0, min(7, round(float(row.get("next_consecutive_losses", row.get("consecutive_losses", 0))))))),
                "next_recent_rewards": int(max(0, min(7, round(float(row.get("next_recent_rewards_24h", row.get("recent_rewards_24h", 0))))))),
                "retained": bool(row.get("retained_proxy", True)),
                "economy_penalty": float(row.get("action_cost", 0.0)),
                "policy_name": str(row.get("behavior_policy", "arena_behavior")),
            }
        )
    return pd.DataFrame(rows)


def player_split(df: pd.DataFrame, seed: int = 2026) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    players = np.array(sorted(df["player_id"].dropna().unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(players)
    n = len(players)
    train_ids = set(players[: int(n * 0.70)])
    val_ids = set(players[int(n * 0.70) : int(n * 0.85)])
    test_ids = set(players[int(n * 0.85) :])
    train = df[df["player_id"].isin(train_ids)].copy()
    val = df[df["player_id"].isin(val_ids)].copy()
    test = df[df["player_id"].isin(test_ids)].copy()
    manifest = {
        "split_key": "player_id",
        "seed": seed,
        "players": {"train": len(train_ids), "validation": len(val_ids), "test": len(test_ids)},
        "rows": {"train": int(len(train)), "validation": int(len(val)), "test": int(len(test))},
        "leakage_checks": {
            "train_validation_overlap": len(train_ids & val_ids),
            "train_test_overlap": len(train_ids & test_ids),
            "validation_test_overlap": len(val_ids & test_ids),
        },
    }
    return train, val, test, manifest


def load_bandit_training_data(rule: RuleBasedPolicy) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    arena_csv = Path("data/arena_liveops_episodes.csv")
    if arena_csv.exists():
        print("Training source: repo arena_liveops_episodes.csv")
        liveops_df = arena_to_liveops_df(pd.read_csv(arena_csv))
        train, _, test, manifest = player_split(liveops_df)
        Path("data/dataset_split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return train, test, manifest

    print("Training source: generated fallback simulation")
    generated = simulate_dataset(rule.recommend, n_players=900, days=14, seed=20, policy_name=rule.name)
    train, _, test, manifest = player_split(generated)
    Path("data/dataset_split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return train, test, manifest


def compute_ope_metrics(q_policy: QLearningPolicy, ope_df: pd.DataFrame) -> dict:
    if ope_df is not None and not ope_df.empty:
        result = evaluate_off_policy(q_policy, ope_df)
        result["dataset"] = "data/arena_liveops_episodes.csv:test_split"
        return result
    generated = simulate_dataset(RuleBasedPolicy().recommend, n_players=300, days=14, seed=20, policy_name="rule")
    result = evaluate_off_policy(q_policy, generated)
    result["dataset"] = "generated_fallback"
    result.setdefault("warnings", []).append("Arena test split was missing; used deterministic generated fallback data.")
    return result


def main() -> None:
    Path("artifacts").mkdir(exist_ok=True)
    rule = RuleBasedPolicy()
    train_df, test_df, split_manifest = load_bandit_training_data(rule)
    bandit = BanditPolicy().fit(train_df)
    q_policy = QLearningPolicy().train_from_logged(train_df, passes=3).train(n_episodes=6000, days=14, seed=7)

    metrics = compare_policies(rule, bandit, q_policy)
    metrics["dataset_split"] = split_manifest
    metrics["off_policy_evaluation"] = compute_ope_metrics(q_policy, test_df)
    q_policy.save(settings.POLICY_ARTIFACT_PATH)
    Path(settings.POLICY_METRICS_PATH).write_text(json.dumps(metrics, indent=2))
    log_eval_results(metrics)

    print("LiveOps Policy Lab training summary")
    print(f"Rule baseline reward: {metrics['rule']['avg_reward']:.3f}")
    print(f"Bandit reward:        {metrics['bandit']['avg_reward']:.3f}")
    print(f"Q-learning reward:    {metrics['q_learning']['avg_reward']:.3f}")
    print(f"Safety gate:          {metrics['safety_gate']['decision']}")
    print(f"Split players:        {split_manifest['players']}")
    ope = metrics.get("off_policy_evaluation", {})
    if ope.get("available"):
        print(f"OPE SNIPS estimate:   {ope['snips_estimate']:.3f}")
        print(f"OPE match rate:       {ope['match_rate']:.3f}")
        print(f"OPE effective sample size: {ope['effective_sample_size']:.1f}")
    else:
        print(f"OPE unavailable:      {ope.get('message', 'not computed')}")
    for reason in metrics["safety_gate"]["reasons"]:
        print(f"- {reason}")


if __name__ == "__main__":
    main()

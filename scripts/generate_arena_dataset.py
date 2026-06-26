from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from app.arena import (
    ACTION_COST,
    ARENA_ACTIONS,
    ArenaState,
    PRESET_SCENARIOS,
    _apply_action_to_state,
    _choose_policy_action,
    _episode_reward,
    _play_match,
    _policy_violation,
    derive_features,
)


def clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(v)))


def jitter_state(base: ArenaState, rng: np.random.Generator) -> ArenaState:
    s = base.model_dump()
    s["player_power"] = round(max(100, s["player_power"] * rng.normal(1.0, 0.08)), 2)
    s["enemy_power"] = round(max(100, s["enemy_power"] * rng.normal(1.0, 0.09)), 2)
    s["fatigue"] = clamp(s["fatigue"] + rng.normal(0, 0.09))
    s["skill_score"] = clamp(s["skill_score"] + rng.normal(0, 0.08))
    s["consecutive_losses"] = int(max(0, min(20, round(s["consecutive_losses"] + rng.normal(0, 1.5)))))
    s["attempts_on_enemy"] = int(max(0, min(45, round(s["attempts_on_enemy"] + rng.normal(0, 2.2)))))
    s["best_completion_pct"] = round(max(0, min(100, s["best_completion_pct"] + rng.normal(0, 8.0))), 2)
    s["upgrade_currency"] = int(max(0, min(5000, round(s["upgrade_currency"] + rng.normal(0, 90)))))
    s["upgrade_cost"] = int(max(1, min(5000, round(s["upgrade_cost"] + rng.normal(0, 50)))))
    s["recent_rewards_24h"] = int(max(0, min(12, round(s["recent_rewards_24h"] + rng.normal(0, 1.1)))))
    s["training_affinity"] = clamp(s["training_affinity"] + rng.normal(0, 0.12))
    s["temporary_power_boost_pct"] = 0.0
    s["enemy_power_modifier_pct"] = 0.0
    s["elite_match_unlocked"] = False
    s["day"] = 0
    return ArenaState.model_validate(s)


def derive_behavioral_telemetry(state: ArenaState, before, match: dict, rng: np.random.Generator) -> dict:
    """Create raw behavioral telemetry. Frustration is derived from these fields, not stored as a direct input."""
    completion = float(match.get("completion_pct", state.best_completion_pct))
    won = bool(match.get("won"))
    power_gap_pressure = clamp((-before.power_gap_pct + 0.05) / 0.75)
    failure_pressure = clamp(state.consecutive_losses / 10.0)
    fatigue_pressure = clamp(state.fatigue)

    session_time_minutes = max(3.0, rng.normal(18 + 4.5 * state.attempts_on_enemy + 5.0 * failure_pressure + 8.0 * fatigue_pressure, 5.0))
    time_on_current_enemy_minutes = max(1.0, rng.normal(4.0 + 3.2 * state.attempts_on_enemy + 6.0 * failure_pressure, 3.5))
    attempts_last_24h = int(max(0, round(state.attempts_on_enemy + rng.normal(4.0 + 6.0 * failure_pressure, 2.8))))
    failed_challenge_count = int(max(0, round(state.consecutive_losses + (0 if won else 1) + rng.normal(1.0, 1.2))))
    challenge_retry_count = int(max(0, round(state.attempts_on_enemy * rng.uniform(0.45, 0.95) + failed_challenge_count * 0.35)))
    optional_challenges_tried = int(max(0, round(rng.normal(1.0 + 3.0 * state.training_affinity, 1.2))))
    side_quests_tried = int(max(0, round(optional_challenges_tried * rng.uniform(0.35, 0.90))))
    raid_attempts = int(max(0, round(rng.normal(0.4 + (2.2 if state.advanced_player else 0.0), 0.9))))
    quit_after_failure = bool((not won) and rng.random() < clamp(0.08 + 0.38 * failure_pressure + 0.20 * power_gap_pressure + 0.16 * fatigue_pressure))
    retry_latency_minutes = max(0.2, rng.normal(1.5 + 3.0 * fatigue_pressure + 4.0 * failure_pressure + (7.0 if quit_after_failure else 0.0), 1.2))
    booster_uses = int(max(0, round(rng.normal(0.35 + 1.8 * power_gap_pressure + 0.8 * failure_pressure, 0.8))))
    revive_uses = int(max(0, round(rng.normal(0.20 + 1.6 * failure_pressure + 0.6 * fatigue_pressure, 0.7))))
    currency_spent_last_24h = int(max(0, round(rng.normal(45 + 120 * power_gap_pressure + 80 * failure_pressure, 65))))
    asset_losses_last_24h = int(max(0, round(rng.normal(0.2 + 2.8 * failure_pressure + 1.4 * power_gap_pressure, 1.1))))
    rewards_claimed_last_24h = int(max(0, round(state.recent_rewards_24h + rng.normal(0.4, 1.0))))
    challenge_switches = int(max(0, round(rng.normal(0.5 + 4.0 * failure_pressure + 1.4 * power_gap_pressure, 1.4))))
    idle_seconds_after_loss = int(max(0, round(rng.normal(35 + 120 * failure_pressure + 85 * fatigue_pressure + (180 if quit_after_failure else 0), 55))))
    help_screen_opens = int(max(0, round(rng.normal(0.4 + 2.2 * failure_pressure + 1.1 * power_gap_pressure, 0.9))))
    chat_negative_signal_count = int(max(0, round(rng.normal(0.1 + 1.4 * failure_pressure + 0.8 * fatigue_pressure, 0.7))))
    near_miss_count = int(max(0, round((1 if 70 <= completion < 96 and not won else 0) + rng.normal(0.6 * failure_pressure, 0.5))))

    # Derived, explainable frustration feature. This is intentionally built from observable telemetry.
    time_pressure = clamp(time_on_current_enemy_minutes / 80.0)
    retry_pressure = clamp(challenge_retry_count / 18.0)
    failure_count_pressure = clamp(failed_challenge_count / 12.0)
    quit_pressure = 1.0 if quit_after_failure else 0.0
    resource_loss_pressure = clamp((asset_losses_last_24h + revive_uses + booster_uses) / 10.0)
    idle_pressure = clamp(idle_seconds_after_loss / 420.0)
    help_pressure = clamp(help_screen_opens / 7.0)
    near_miss_pressure = clamp(near_miss_count / 5.0)
    optional_content_relief = clamp((optional_challenges_tried + side_quests_tried) / 12.0)

    frustration_from_telemetry = clamp(
        0.18 * time_pressure
        + 0.20 * retry_pressure
        + 0.18 * failure_count_pressure
        + 0.12 * quit_pressure
        + 0.11 * resource_loss_pressure
        + 0.08 * idle_pressure
        + 0.07 * help_pressure
        + 0.05 * near_miss_pressure
        + 0.09 * fatigue_pressure
        + 0.08 * power_gap_pressure
        - 0.08 * optional_content_relief
    )

    return {
        "session_time_minutes": round(session_time_minutes, 2),
        "time_on_current_enemy_minutes": round(time_on_current_enemy_minutes, 2),
        "attempts_last_24h": attempts_last_24h,
        "failed_challenge_count": failed_challenge_count,
        "challenge_retry_count": challenge_retry_count,
        "optional_challenges_tried": optional_challenges_tried,
        "side_quests_tried": side_quests_tried,
        "raid_attempts": raid_attempts,
        "quit_after_failure": quit_after_failure,
        "retry_latency_minutes": round(retry_latency_minutes, 2),
        "booster_uses": booster_uses,
        "revive_uses": revive_uses,
        "currency_spent_last_24h": currency_spent_last_24h,
        "asset_losses_last_24h": asset_losses_last_24h,
        "rewards_claimed_last_24h": rewards_claimed_last_24h,
        "challenge_switches": challenge_switches,
        "idle_seconds_after_loss": idle_seconds_after_loss,
        "help_screen_opens": help_screen_opens,
        "chat_negative_signal_count": chat_negative_signal_count,
        "near_miss_count": near_miss_count,
        "time_pressure": round(time_pressure, 4),
        "retry_pressure": round(retry_pressure, 4),
        "failure_count_pressure": round(failure_count_pressure, 4),
        "resource_loss_pressure": round(resource_loss_pressure, 4),
        "idle_pressure": round(idle_pressure, 4),
        "optional_content_relief": round(optional_content_relief, 4),
        "frustration_score": round(frustration_from_telemetry, 4),
        "frustration_source": "derived_from_behavioral_telemetry_v2",
    }


def generate(n_players: int = 3200, steps: int = 10, seed: int = 2026) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    scenarios = list(PRESET_SCENARIOS.keys())
    scenario_weights = np.array([0.12, 0.18, 0.16, 0.15, 0.13, 0.13, 0.13])
    scenario_weights = scenario_weights / scenario_weights.sum()
    behavior_policies = ["rule_based", "raw_rl", "safety_gated_rl", "random", "do_nothing"]
    behavior_weights = np.array([0.34, 0.22, 0.30, 0.09, 0.05])
    rows = []

    for player_id in range(n_players):
        scenario = str(rng.choice(scenarios, p=scenario_weights))
        state = jitter_state(PRESET_SCENARIOS[scenario], rng)
        behavior_policy = str(rng.choice(behavior_policies, p=behavior_weights))
        cohort = str(rng.choice(["new", "returning", "spender", "advanced"], p=[0.28, 0.42, 0.16, 0.14]))
        install_age_days = int(max(0, rng.gamma(2.0, 18.0))) if not state.new_player else int(rng.integers(0, 3))

        for step in range(steps):
            before = derive_features(state)
            if rng.random() < 0.08:
                action = str(rng.choice(ARENA_ACTIONS))
                behavior_source = "exploration"
                action_probability = 1 / len(ARENA_ACTIONS)
            else:
                action = _choose_policy_action(behavior_policy, state, rng)
                behavior_source = behavior_policy
                action_probability = 0.72 if behavior_policy != "random" else 1 / len(ARENA_ACTIONS)

            violation = _policy_violation(state, action)
            action_state = _apply_action_to_state(state, action, mutate_day=False)
            match = _play_match(action_state, seed=int(rng.integers(0, 1_000_000)))
            telemetry = derive_behavioral_telemetry(state, before, match, rng)
            next_state = ArenaState.model_validate(match["state_after"])
            after = derive_features(next_state)

            # Churn proxy uses the derived telemetry frustration, not a direct frustration label.
            derived_frustration = float(telemetry["frustration_score"])
            observed_churn_risk = clamp(0.38 * derived_frustration + 0.26 * after.churn_risk + 0.18 * before.progression_stall + 0.10 * before.fatigue_pressure + 0.08 * before.reward_saturation)
            retained_proxy = bool(rng.random() > observed_churn_risk * 0.10)
            reward = _episode_reward(before, after, action, bool(match["won"]), violation) - 0.12 * derived_frustration

            rows.append({
                "player_id": player_id,
                "cohort": cohort,
                "install_age_days": install_age_days,
                "episode_step": step,
                "scenario_name": scenario,
                "behavior_policy": behavior_policy,
                "behavior_source": behavior_source,
                "action": action,
                "action_probability": round(action_probability, 4),
                "action_cost": ACTION_COST.get(action, 0.0),
                "policy_violation": violation,
                "won": bool(match["won"]),
                "retained_proxy": retained_proxy,
                "observed_churn_risk": round(observed_churn_risk, 4),
                "completion_pct": match["completion_pct"],
                "reward": round(reward, 6),
                **{k: v for k, v in state.model_dump().items()},
                **telemetry,
                "model_feature_frustration_score": before.frustration_score,
                "effective_player_power": before.effective_player_power,
                "effective_enemy_power": before.effective_enemy_power,
                "power_gap_pct": before.power_gap_pct,
                "win_probability": before.win_probability,
                "churn_risk": before.churn_risk,
                "economy_pressure": before.economy_pressure,
                "reward_saturation": before.reward_saturation,
                "fatigue_pressure": before.fatigue_pressure,
                "progression_stall": before.progression_stall,
                "cold_start": before.cold_start,
                "history_confidence": before.history_confidence,
                "next_player_power": next_state.player_power,
                "next_enemy_power": next_state.enemy_power,
                "next_fatigue": next_state.fatigue,
                "next_skill_score": next_state.skill_score,
                "next_consecutive_losses": next_state.consecutive_losses,
                "next_attempts_on_enemy": next_state.attempts_on_enemy,
                "next_best_completion_pct": next_state.best_completion_pct,
                "next_upgrade_currency": next_state.upgrade_currency,
                "next_recent_rewards_24h": next_state.recent_rewards_24h,
                "next_win_probability": after.win_probability,
                "next_model_feature_frustration_score": after.frustration_score,
                "next_churn_risk": after.churn_risk,
            })
            state = next_state
            install_age_days += 1 if step % 5 == 0 else 0
            if not retained_proxy and rng.random() < 0.50:
                break
    return pd.DataFrame(rows)


def write_outputs(df: pd.DataFrame) -> None:
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    csv_path = data_dir / "arena_liveops_episodes.csv"
    df.to_csv(csv_path, index=False)
    profile = {
        "dataset": str(csv_path),
        "rows": int(len(df)),
        "players": int(df["player_id"].nunique()),
        "scenarios": df["scenario_name"].value_counts().to_dict(),
        "actions": df["action"].value_counts().to_dict(),
        "behavior_policies": df["behavior_policy"].value_counts().to_dict(),
        "win_rate": float(df["won"].mean()),
        "retention_proxy_rate": float(df["retained_proxy"].mean()),
        "avg_reward": float(df["reward"].mean()),
        "avg_action_cost": float(df["action_cost"].mean()),
        "policy_violation_rate": float(df["policy_violation"].mean()),
        "avg_win_probability": float(df["win_probability"].mean()),
        "avg_derived_frustration": float(df["frustration_score"].mean()),
        "avg_observed_churn_risk": float(df["observed_churn_risk"].mean()),
        "frustration_derivation": {
            "source": "derived_from_behavioral_telemetry_v2",
            "target_column": "frustration_score",
            "raw_signal_columns": [
                "session_time_minutes",
                "time_on_current_enemy_minutes",
                "attempts_last_24h",
                "failed_challenge_count",
                "challenge_retry_count",
                "optional_challenges_tried",
                "side_quests_tried",
                "raid_attempts",
                "quit_after_failure",
                "retry_latency_minutes",
                "booster_uses",
                "revive_uses",
                "currency_spent_last_24h",
                "asset_losses_last_24h",
                "rewards_claimed_last_24h",
                "challenge_switches",
                "idle_seconds_after_loss",
                "help_screen_opens",
                "chat_negative_signal_count",
                "near_miss_count",
            ],
            "formula_summary": "Weighted pressure score from time spent, retries, failures, quit-after-failure, resource loss, idle time, help opens, near misses, fatigue, and power gap; optional content engagement provides relief.",
        },
        "columns": list(df.columns),
        "notes": [
            "Synthetic arena telemetry for the LiveOps RL simulator.",
            "Frustration is not a manually supplied input; frustration_score is derived from observable behavioral telemetry.",
            "Includes cold-start, fatigue, reward saturation, near-win failure, and advanced-player boredom scenarios.",
            "One learned RL decision agent is evaluated against do-nothing, random, rule-based, raw RL, and safety-gated RL baselines. Guardrails are deterministic, not a second agent.",
        ],
    }
    (data_dir / "arena_dataset_profile.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")
    players = np.array(sorted(df["player_id"].unique()))
    split_rng = np.random.default_rng(2026)
    split_rng.shuffle(players)
    n_players = len(players)
    train_ids = set(players[: int(n_players * 0.70)])
    val_ids = set(players[int(n_players * 0.70) : int(n_players * 0.85)])
    test_ids = set(players[int(n_players * 0.85) :])
    split_manifest = {
        "split_key": "player_id",
        "seed": 2026,
        "players": {"train": len(train_ids), "validation": len(val_ids), "test": len(test_ids)},
        "rows": {
            "train": int(df[df["player_id"].isin(train_ids)].shape[0]),
            "validation": int(df[df["player_id"].isin(val_ids)].shape[0]),
            "test": int(df[df["player_id"].isin(test_ids)].shape[0]),
        },
        "leakage_checks": {
            "train_validation_overlap": len(train_ids & val_ids),
            "train_test_overlap": len(train_ids & test_ids),
            "validation_test_overlap": len(val_ids & test_ids),
        },
    }
    (data_dir / "dataset_split_manifest.json").write_text(json.dumps(split_manifest, indent=2), encoding="utf-8")
    sample_path = data_dir / "arena_liveops_episodes_sample.ndjson"
    sample_path.write_text("\n".join(json.dumps(r, default=str) for r in df.head(150).to_dict(orient="records")), encoding="utf-8")


if __name__ == "__main__":
    df = generate()
    write_outputs(df)
    print(f"Wrote {len(df)} rows to data/arena_liveops_episodes.csv")


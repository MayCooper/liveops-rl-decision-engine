# LiveOps Policy Context

The RL policy recommends one of the arena LiveOps actions: do_nothing, recommend_training_match, grant_upgrade_currency, offer_temporary_power_boost, reduce_enemy_power, recommend_recovery_match, or unlock_elite_match.

The Q-learning policy scores candidate actions. A deterministic safety gate then blocks or redirects risky actions before serving. The safety gate is not a second decision-making agent; it is a hard guardrail layer.

Important safety rules include blocking elite unlocks for cold-start players, blocking repeated currency grants when reward saturation is high, blocking challenge escalation for frustrated or high-risk players, and requiring repeated failure or low win probability before reducing enemy power.

The LLM explanation layer can explain the recommendation, scores, guardrail blocks, expected state effect, metrics, OPE evidence, and rollout audit. It must not choose actions, mutate simulator state, override the policy, or bypass the safety gate.

## Arena game context

The Arena demo is a synthetic LiveOps environment for a boss-battle game. A player has visible telemetry such as player power, enemy power, fatigue, skill score, consecutive losses, attempts on the current enemy, best completion percentage, upgrade currency, recent rewards, and training affinity.

The simulator derives win probability, frustration score, churn-risk proxy, economy pressure, reward saturation, fatigue pressure, progression stall, cold-start status, and history confidence from that telemetry.

The included arena dataset has 3,200 players and 29,293 decision episodes across seven scenarios: cold_start, new_player_stuck, underpowered_engaged, fatigued_player, advanced_bored, high_reward_saturation, and near_win_repeated_failure.

Frustration is a derived feature. It is not a raw label typed by a user. It is calculated from behavioral signals such as time on enemy, retries, failed challenges, quit-after-failure, resource loss, idle time after loss, help-screen opens, near misses, fatigue, and power gap.

## Evaluation and rollout context

Benchmarking compares do-nothing, random, rule-based, raw RL, and safety-gated RL policies. Raw RL shows what the learned policy would prefer before guardrails. Safety-gated RL shows what is safe to serve after policy constraints.

Offline Policy Evaluation estimates candidate policy value from logged behavior-policy data. Match rate shows how often logged behavior actions match the candidate action. Effective sample size summarizes how much useful weighted evidence remains after propensity weighting.

Run events from the interactive demo are logged as append-only JSONL records in artifacts/local_logs.jsonl. Event types include arena_recommendation, arena_action_applied, arena_match_result, arena_benchmark_run, and agent_message.

Interactive battle results are not added as new columns to the static training CSV. They are session/run telemetry. If these events need to become training data, add a deliberate offline export step that converts JSONL events into rows and then regenerates dataset profiles, train/validation/test split manifests, and evaluation artifacts.

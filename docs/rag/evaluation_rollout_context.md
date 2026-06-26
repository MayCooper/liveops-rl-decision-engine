# Evaluation And Rollout Context

Benchmarking compares do-nothing, random, rule-based, raw RL, and safety-gated RL policies. Raw RL shows what the learned policy would prefer before guardrails. Safety-gated RL shows what is safe to serve after policy constraints.

Offline Policy Evaluation estimates candidate policy value from logged behavior-policy data. Match rate shows how often logged behavior actions match the candidate action. Effective sample size summarizes how much useful weighted evidence remains after propensity weighting.

Run events from the interactive demo are logged as append-only JSONL records in artifacts/local_logs.jsonl. Event types include arena_recommendation, arena_action_applied, arena_match_result, arena_benchmark_run, and agent_message.

Interactive battle results are not added as new columns to the static training CSV. They are session/run telemetry. If these events need to become training data, add a deliberate offline export step that converts JSONL events into rows and then regenerates dataset profiles, train/validation/test split manifests, and evaluation artifacts.

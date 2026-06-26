# LiveOps Policy Context

The RL policy recommends one of the arena LiveOps actions: do_nothing, recommend_training_match, grant_upgrade_currency, offer_temporary_power_boost, reduce_enemy_power, recommend_recovery_match, or unlock_elite_match.

The Q-learning policy scores candidate actions. A deterministic safety gate then blocks or redirects risky actions before serving. The safety gate is not a second decision-making agent; it is a hard guardrail layer.

Important safety rules include blocking elite unlocks for cold-start players, blocking repeated currency grants when reward saturation is high, blocking challenge escalation for frustrated or high-risk players, and requiring repeated failure or low win probability before reducing enemy power.

The LLM explanation layer can explain the recommendation, scores, guardrail blocks, expected state effect, metrics, OPE evidence, and rollout audit. It must not choose actions, mutate simulator state, override the policy, or bypass the safety gate.

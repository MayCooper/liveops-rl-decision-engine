# LiveOps Arena Policy

This project uses one learned RL decision agent. The RL policy recommends the intervention. Deterministic policy rules then allow, block, or boost actions before serving. The optional explanation layer summarizes what happened; it is not a second decision agent.

## Allowed intervention categories

- `recommend_training_match`: low-risk progression support.
- `grant_upgrade_currency`: economic intervention with strict reward-saturation limits.
- `offer_temporary_power_boost`: short-lived assist for underpowered or near-win players.
- `reduce_enemy_power`: temporary difficulty relief for repeated failure.
- `recommend_recovery_match`: fatigue recovery path.
- `unlock_elite_match`: harder content for advanced, low-frustration players.
- `do_nothing`: valid action when intervention is unnecessary or unsafe.

## Hard constraints

Cold-start users should not receive elite difficulty. Reward-saturated players should not receive more currency or boost rewards. Frustrated or high-risk users should not receive harder content. Difficulty reductions should be reserved for repeated failure or low win probability.

## Evaluation

The system should be judged against do-nothing, random, rule-based, raw RL, and safety-gated RL baselines. A successful policy improves win rate and recovery while keeping economy cost and policy violations controlled.

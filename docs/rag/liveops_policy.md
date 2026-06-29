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

## Economy policy

The economy policy prevents the RL decision agent from overusing resource grants or power boosts.

Resource grants are blocked when recent reward exposure is high. Temporary power boosts are also blocked under high reward saturation because they function as value-bearing interventions. The benchmark reports average intervention cost and policy violations so that improved win rate is not accepted if it comes from excessive reward spending.

## Cold-start policy

Cold-start users have little or no behavioral history. The system should prefer safe onboarding actions, such as training matches or doing nothing, until there is enough history to estimate player pressure more confidently.

Cold-start users should not receive elite difficulty, repeated grants, or aggressive difficulty changes. The simulator exposes `cold_start` and `history_confidence` so the UI can show why the served action becomes more conservative.

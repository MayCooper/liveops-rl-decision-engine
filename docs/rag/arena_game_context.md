# Arena Game Context

The Arena demo is a synthetic LiveOps environment for a boss-battle game. A player has visible telemetry such as player power, enemy power, fatigue, skill score, consecutive losses, attempts on the current enemy, best completion percentage, upgrade currency, recent rewards, and training affinity.

The simulator derives win probability, frustration score, churn-risk proxy, economy pressure, reward saturation, fatigue pressure, progression stall, cold-start status, and history confidence from that telemetry.

The included arena dataset has 3,200 players and 29,293 decision episodes across seven scenarios: cold_start, new_player_stuck, underpowered_engaged, fatigued_player, advanced_bored, high_reward_saturation, and near_win_repeated_failure.

Frustration is a derived feature. It is not a raw label typed by a user. It is calculated from behavioral signals such as time on enemy, retries, failed challenges, quit-after-failure, resource loss, idle time after loss, help-screen opens, near misses, fatigue, and power gap.

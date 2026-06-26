# Arena RL Simulator

The arena simulator is an interactive LiveOps decision sandbox. It is designed to show how a learned RL policy reacts when game/player conditions change, without building a full game.

## Architecture

There is one learned decision agent: the existing Q-learning policy. It recommends the player intervention. The deterministic safety gate constrains which actions can be served. The optional audit/explanation layer only explains decisions; it does not choose or override actions.

```
UI sliders
→ /arena/recommend
→ arena telemetry is mapped to PlayerState
→ Q-learning policy scores actions
→ deterministic policy gate blocks/boosts actions
→ UI displays recommendation, scores, expected effect, and blocked actions
```

## What the UI changes

The Arena simulator is now integrated into the main browser screen:

```
http://127.0.0.1:8014/
```

The standalone route remains available at `/arena/dashboard`, but the intended demo entry point is the main page.

The UI includes sliders for player power, enemy power, fatigue, skill score, consecutive losses, attempts on enemy, best completion %, upgrade currency, upgrade cost, recent rewards, and training affinity. When a slider changes, the frontend calls `/arena/recommend` after a short debounce and updates the recommendation immediately.

## Replay behavior

The simulator is not a real-time game engine. During setup, slider changes update metrics live. During replay, the current state is frozen, `/arena/play_match` returns replay events, and the UI animates those events with simple HTML/CSS/JavaScript.

## Benchmarking

The `/arena/benchmark` endpoint compares:

- `do_nothing`
- `random`
- `rule_based`
- `raw_rl`
- `safety_gated_rl`

The benchmark runs across cold-start, stuck-player, underpowered, fatigue, advanced-player, reward-saturation, and near-win scenarios. It reports win rate, average reward, average cost, final frustration, final churn risk, final win probability, and policy violations.

## Dataset

The updated repo includes:

```
data/arena_liveops_episodes.csv
data/arena_dataset_profile.json
data/arena_liveops_episodes_sample.ndjson
```

This dataset is synthetic, but it is more realistic than a flat toy dataset because it includes observable behavioral telemetry rather than directly supplied frustration labels. The `frustration_score` column is derived from fields such as session time, time spent on the current enemy, failed challenge count, retry count, optional challenges tried, quit-after-failure, retry latency, booster/revive usage, asset losses, reward claims, challenge switching, idle time after loss, help-screen opens, negative chat signal count, near-miss count, fatigue, and power gap.

The raw telemetry columns support a realistic explanation: the system does not magically know frustration; it infers a pressure score from player behavior.

## Policies

The policy files are in:

```
policies/policy_rules.json
policies/liveops_policy.md
policies/economy_policy.md
policies/cold_start_policy.md
```

These policies document the deterministic guardrails used by the simulator.

## Cinematic battle replay update

The arena dashboard now uses a richer event-driven battle replay instead of symbolic avatar movement. The backend returns a sequence of replay events from `/arena/play_match`, including match start, player attacks, boss counterattacks, dodges, critical hits, active boosts, enemy debuffs, fatigue pressure, match result, and post-match telemetry update. The frontend renders these events as a lightweight mini battle using HTML/CSS/JavaScript: hero and boss sprites, HP bars, progress, slash effects, impact flashes, damage numbers, recovery orbs, currency effects, boss rage/weakened states, and replay speed control.

The animation is still deterministic and explainable: it is generated from simulator telemetry, not from a video model or a separate game engine. This keeps the demo fast, free, inspectable, and directly tied to the RL policy decision and safety-gated outcome.


## Main-screen integration update

The previous symbolic/static simulation panel was removed from the main screen. The main `/` page now uses the richer arena simulator UI: sliders, preset scenarios below the slider section, manual controls, live metric cards, action-score cards, safety gate cards, benchmark charts, and the cinematic battle replay.

## Roman-style hero update

The player avatar is styled as a Roman/gladiator-like fighter using CSS: red cape, gold/bronze armor, crested helmet, shield marked SPQR, sword, and sandals. This keeps the replay free and deterministic while making the scene look more like an actual battle.

## Robust telemetry dataset update

The regenerated dataset contains roughly thirty thousand arena episode rows and more than eighty columns. The important change is that `frustration_score` is derived from behavioral telemetry rather than supplied as a raw slider or label. The profile file records the source columns and formula summary under `frustration_derivation`.

## Merged UI behavior

The main page now combines the arena simulator and the original operations console. The top simulator is used for interactive scenario testing: choose a preset, adjust sliders, recalculate the recommendation, apply the served action, and play the next match. Below the simulator, the operations console restores the health checks, policy metrics, OPE trace, logs, manual recommendation tools, audit/explanation console, and 7-day progress cards.

The UI is framed as one RL Decision Agent. The red safety/risk gate is a deterministic component inside the same serving path. It can block or replace actions before serving, but it is not a second decision agent. The explanation console answers questions about what happened; it does not choose actions.

Frustration is displayed as a derived signal. It is calculated from telemetry such as retries, attempts, completion percentage, fatigue, power gap, and loss streak. The supporting feature dictionary is in `data/frustration_feature_dictionary.json`.

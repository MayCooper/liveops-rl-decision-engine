# Presentation Talk Track

## One-sentence framing

This is a governed RL decision sandbox for LiveOps interventions: it lets a team test how a learned policy reacts to player/game telemetry before deploying it.

## What changed

I added an arena simulator around the existing LiveOps RL policy. The simulator exposes realistic game telemetry through sliders, maps those values into the existing player-state schema, asks the Q-learning policy for an intervention, applies deterministic safety guardrails, and shows the expected and simulated outcome.

## Key design decision

I intentionally kept this as one RL decision agent. The RL policy is the only component that chooses actions. The safety gate is deterministic. The optional audit/explanation layer explains what happened but does not choose or override actions.

## Why the simulator matters

The simulator makes the policy behavior visible. If I increase enemy difficulty, fatigue, or loss streak, the derived frustration, churn risk, and win probability change. The policy rescoring changes as well. The UI then shows the served action, blocked actions, before/after expected effect, replay outcome, and benchmark comparison.

## Evaluation

The benchmark compares do-nothing, random, rule-based, raw RL, and safety-gated RL policies across fixed scenario seeds. This shows whether RL adds value beyond hand-written rules and whether the safety gate controls cost and policy violations.

## Practical MLE signal

The project demonstrates system design, API design, simulation, policy evaluation, safety gating, explainability, frontend observability, and cloud-ready packaging. It avoids overclaiming: this is not a production game engine; it is a controlled decision-policy test harness.

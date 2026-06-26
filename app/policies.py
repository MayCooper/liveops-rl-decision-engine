from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from app.core import ACTIONS, create_initial_player, simulate_episode, step_player


ACTION_TO_INDEX = {action: idx for idx, action in enumerate(ACTIONS)}
INDEX_TO_ACTION = {idx: action for action, idx in ACTION_TO_INDEX.items()}
SEGMENTS = ["new", "mid_skill", "advanced"]
STATE_VECTOR_DIM = 11


def state_to_key(state: dict) -> tuple:
    """Discretize state into a compact tabular key for the bandit/Q-learning baselines."""
    bucket = lambda x, n=5: int(max(0, min(n - 1, np.floor(float(x) * n))))
    return (
        state["segment"],
        bucket(state["skill"]),
        bucket(state["frustration"]),
        bucket(state["engagement"]),
        bucket(state["churn_risk"]),
        bucket(state["economy_balance"]),
        min(int(state.get("recent_losses", 0)), 4),
        min(int(state.get("recent_rewards", 0)), 4),
    )


def heuristic_action_prior(state: dict) -> dict[str, float]:
    """Small prior for unseen states; learned policies overwrite it through training."""
    values = {action: 0.02 for action in ACTIONS}
    values["do_nothing"] = 0.08 if state["churn_risk"] < 0.45 and state["engagement"] > 0.55 else 0.0

    if state["segment"] == "new":
        values["decrease_difficulty"] += 0.22 * state["frustration"]
        values["increase_difficulty"] -= 0.12 * state["frustration"]

    if state["segment"] == "advanced":
        values["increase_difficulty"] += 0.16 * (1 - state["engagement"])
        values["decrease_difficulty"] -= 0.08

    if state["segment"] == "mid_skill":
        values["recommend_quest"] += 0.16 * state["churn_risk"]

    if state["economy_balance"] < 0.3 and state["recent_rewards"] < 2:
        values["grant_bonus_resources"] += 0.12

    if state["recent_rewards"] >= 4:
        values["grant_bonus_resources"] -= 0.2

    if state["engagement"] > 0.5 and state["churn_risk"] < 0.7:
        values["offer_cosmetic_reward"] += 0.06

    return values


def apply_liveops_safety_mask(
    state: dict,
    values: dict[str, float],
    value_label: str = "Q-values",
) -> tuple[dict[str, float], list[dict[str, str]], list[str]]:
    """Apply deterministic serving constraints shared by all policies.

    The learned policy estimates action value. This function is the deployment
    guardrail: it blocks or boosts actions according to explicit LiveOps rules
    before the recommendation is served.
    """
    served = {action: float(values.get(action, 0.0)) for action in ACTIONS}
    blocked_actions: list[dict[str, str]] = []
    safety_notes = [
        "Safety mask applied before serving.",
        f"{value_label} are constrained by deterministic rollout rules.",
    ]

    stable = state["churn_risk"] < 0.35 and state["frustration"] < 0.35 and state["engagement"] > 0.65
    if stable:
        safety_notes.insert(0, "Stable player: do_nothing preferred.")
        for action in ACTIONS:
            if action != "do_nothing":
                served[action] = -1e9
                blocked_actions.append(
                    {
                        "action": action,
                        "reason": "Stable players are served do_nothing to avoid unnecessary intervention.",
                    }
                )
        return served, blocked_actions, safety_notes

    grant_allowed = state["economy_balance"] < 0.35 and state["recent_rewards"] < 2 and state["churn_risk"] > 0.5
    if not grant_allowed:
        served["grant_bonus_resources"] = -1e9
        blocked_actions.append(
            {
                "action": "grant_bonus_resources",
                "reason": "Resource grants are blocked unless economy_balance is low, recent_rewards < 2, and churn_risk is elevated.",
            }
        )

    if state["segment"] == "new" and state["frustration"] > 0.75:
        served["increase_difficulty"] = -1e9
        blocked_actions.append(
            {
                "action": "increase_difficulty",
                "reason": "Difficulty increases are blocked for highly frustrated new players.",
            }
        )
        if state["economy_balance"] >= 0.25:
            served["grant_bonus_resources"] = -1e9
        served["decrease_difficulty"] += 0.2
        safety_notes.append("High-frustration new player: decrease_difficulty receives a serving boost.")

    if state["segment"] == "advanced" and state["engagement"] < 0.45 and state["frustration"] < 0.5:
        served["decrease_difficulty"] = -1e9
        served["increase_difficulty"] += 0.25
        blocked_actions.append(
            {
                "action": "decrease_difficulty",
                "reason": "Difficulty decreases are blocked for bored advanced players unless frustration is high.",
            }
        )
        safety_notes.append("Bored advanced player: increase_difficulty receives a serving boost.")

    return served, blocked_actions, safety_notes


def encode_state_vector(state: dict) -> np.ndarray:
    """Encode a LiveOps state as continuous features for deep value learning."""
    segment = str(state.get("segment", "mid_skill"))
    day = float(state.get("day", 0))
    recent_losses = float(state.get("recent_losses", 0))
    recent_rewards = float(state.get("recent_rewards", 0))
    return np.array(
        [
            1.0 if segment == "new" else 0.0,
            1.0 if segment == "mid_skill" else 0.0,
            1.0 if segment == "advanced" else 0.0,
            float(state.get("skill", 0.5)),
            float(state.get("frustration", 0.5)),
            float(state.get("engagement", 0.5)),
            float(state.get("churn_risk", 0.5)),
            float(state.get("economy_balance", 0.5)),
            min(max(recent_losses, 0.0) / 7.0, 1.0),
            min(max(recent_rewards, 0.0) / 7.0, 1.0),
            min(max(day, 0.0) / 14.0, 1.0),
        ],
        dtype=np.float32,
    )


def _state_from_row(row: pd.Series) -> dict:
    return {
        "segment": str(row["segment"]),
        "skill": float(row["skill"]),
        "frustration": float(row["frustration"]),
        "engagement": float(row["engagement"]),
        "churn_risk": float(row["churn_risk"]),
        "economy_balance": float(row["economy_balance"]),
        "recent_losses": int(row["recent_losses"]),
        "recent_rewards": int(row["recent_rewards"]),
        "day": int(row["day"]),
    }


def _next_state_from_row(row: pd.Series) -> dict:
    state = _state_from_row(row)
    return {
        **state,
        "skill": float(row["next_skill"]),
        "frustration": float(row["next_frustration"]),
        "engagement": float(row["next_engagement"]),
        "churn_risk": float(row["next_churn_risk"]),
        "economy_balance": float(row["next_economy_balance"]),
        "recent_losses": int(row["next_recent_losses"]),
        "recent_rewards": int(row["next_recent_rewards"]),
        "day": int(row["day"]) + 1,
    }


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        return torch, nn, F
    except Exception as exc:
        raise RuntimeError(
            "ConservativeDQNPolicy requires PyTorch. Add `torch` to requirements.txt and reinstall dependencies."
        ) from exc


def _build_q_network(input_dim: int = STATE_VECTOR_DIM, hidden_dim: int = 64):
    _, nn, _ = _require_torch()
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, len(ACTIONS)),
    )


class RuleBasedPolicy:
    name = "rule"

    def recommend(self, state: dict) -> str:
        if state["segment"] == "new" and state["frustration"] > 0.65:
            return "decrease_difficulty"
        if state["segment"] == "advanced" and state["engagement"] < 0.45:
            return "increase_difficulty"
        if state["segment"] == "mid_skill" and state["churn_risk"] > 0.55:
            return "recommend_quest"
        if state["economy_balance"] < 0.25 and state["recent_rewards"] < 2:
            return "grant_bonus_resources"
        return "do_nothing"


class BanditPolicy:
    name = "bandit"

    def __init__(self) -> None:
        self.bucket_rewards: dict[tuple, dict[str, float]] = {}
        self.global_rewards = {action: 0.0 for action in ACTIONS}

    def fit(self, df: pd.DataFrame) -> "BanditPolicy":
        df = df.copy()
        df["state_key"] = df.apply(lambda r: state_to_key(r.to_dict()), axis=1)
        grouped = df.groupby(["state_key", "action"])["reward"].mean()
        self.bucket_rewards = {}
        for (key, action), reward in grouped.items():
            self.bucket_rewards.setdefault(key, {})[action] = float(reward)
        self.global_rewards = {a: float(v) for a, v in df.groupby("action")["reward"].mean().to_dict().items()}
        for action in ACTIONS:
            self.global_rewards.setdefault(action, 0.0)
        return self

    def recommend(self, state: dict) -> str:
        estimates = self.bucket_rewards.get(state_to_key(state), self.global_rewards)
        return max(ACTIONS, key=lambda a: estimates.get(a, self.global_rewards.get(a, 0.0)))


class QLearningPolicy:
    name = "q_learning"

    def __init__(self, alpha: float = 0.15, gamma: float = 0.9) -> None:
        self.alpha = alpha
        self.gamma = gamma
        self.q: dict[str, dict[str, float]] = {}
        self.policy_version = "q_policy_v2"

    def _key(self, state: dict) -> str:
        return json.dumps(state_to_key(state))

    def _values_for_update(self, state: dict) -> dict[str, float]:
        return self.q.setdefault(self._key(state), heuristic_action_prior(state))

    def action_values(self, state: dict) -> dict[str, float]:
        values = self.q.get(self._key(state), heuristic_action_prior(state))
        return {a: float(values.get(a, 0.0)) for a in ACTIONS}

    def _apply_action_mask(
        self,
        state: dict,
        values: dict[str, float],
    ) -> tuple[dict[str, float], list[dict[str, str]], list[str]]:
        return apply_liveops_safety_mask(state, values, value_label="Q-values")

    def choose_action(self, state: dict, epsilon: float = 0.1) -> str:
        if random.random() < epsilon:
            return random.choice(ACTIONS)
        values = self.action_values(state)
        best = max(values.values())
        tied = [action for action, value in values.items() if value == best]
        return sorted(tied)[0]

    def update(self, state: dict, action: str, reward: float, next_state: dict) -> None:
        values = self._values_for_update(state)
        next_best = max(self.action_values(next_state).values())
        old = values[action]
        values[action] = old + self.alpha * (reward + self.gamma * next_best - old)

    def train_from_logged(self, df: pd.DataFrame, passes: int = 3) -> "QLearningPolicy":
        """Warm-start Q buckets from logged transition rows when available."""
        required = {
            "segment",
            "skill",
            "frustration",
            "engagement",
            "churn_risk",
            "economy_balance",
            "recent_losses",
            "recent_rewards",
            "day",
            "action",
            "reward",
            "next_skill",
            "next_frustration",
            "next_engagement",
            "next_churn_risk",
            "next_economy_balance",
            "next_recent_losses",
            "next_recent_rewards",
        }
        if df is None or df.empty or not required.issubset(df.columns):
            return self
        for _ in range(max(1, int(passes))):
            for _, row in df.iterrows():
                state = _state_from_row(row)
                next_state = _next_state_from_row(row)
                action = str(row["action"])
                if action in ACTIONS:
                    self.update(state, action, float(row["reward"]), next_state)
        return self

    def train(
        self,
        n_episodes: int = 2500,
        days: int = 14,
        simulator: Callable = simulate_episode,
        seed: int = 7,
    ) -> "QLearningPolicy":
        rng = np.random.default_rng(seed)
        random.seed(seed)
        for episode in range(n_episodes):
            epsilon = max(0.05, 0.3 * (1 - episode / n_episodes))
            state_seed = int(rng.integers(0, 1_000_000))
            state = create_initial_player(state_seed)
            step_rng = np.random.default_rng(state_seed)
            for _ in range(days):
                # Use the simulator as a tiny model so each visited bucket sees all actions.
                for candidate in ACTIONS:
                    trial_rng = np.random.default_rng(int(rng.integers(0, 1_000_000)))
                    candidate_next, candidate_reward, _ = step_player(state, candidate, trial_rng)
                    self.update(state, candidate, candidate_reward, candidate_next)
                action = self.choose_action(state, epsilon=epsilon)
                next_state, _, info = step_player(state, action, step_rng)
                state = next_state
                if not info["retained"]:
                    break
        return self

    def recommend(self, state: dict) -> str:
        return self.recommend_with_explanation(state)["recommended_action"]

    def recommend_with_explanation(self, state: dict) -> dict:
        raw_scores = self.action_values(state)
        served_scores, blocked_actions, safety_notes = self._apply_action_mask(state, raw_scores)
        return {
            "recommended_action": max(served_scores, key=served_scores.get),
            "raw_action_scores": raw_scores,
            "served_action_scores": served_scores,
            "blocked_actions": blocked_actions,
            "safety_notes": safety_notes,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "policy_version": self.policy_version,
                    "alpha": self.alpha,
                    "gamma": self.gamma,
                    "q": self.q,
                },
                indent=2,
            )
        )

    @classmethod
    def load(cls, path: str | Path) -> "QLearningPolicy":
        data = json.loads(Path(path).read_text())
        policy = cls(alpha=data.get("alpha", 0.15), gamma=data.get("gamma", 0.9))
        policy.policy_version = data.get("policy_version", "q_policy_v1")
        policy.q = {k: {a: float(v) for a, v in vals.items()} for k, vals in data.get("q", {}).items()}
        return policy


class ConservativeDQNPolicy:
    """Offline conservative deep Q policy for the same LiveOps action space.

    This class deliberately keeps the same serving contract as QLearningPolicy:
    - action_values(state) returns one score per LiveOps action
    - recommend(state) returns the best safety-gated action
    - recommend_with_explanation(state) returns the same fields used by FastAPI/UI

    Improvements over the first deep policy version:
    - CQL-style conservative penalty for offline RL support constraints
    - Double DQN target selection to reduce Q-value overestimation
    - Small bootstrapped ensemble for uncertainty-aware pessimistic scoring
    - Optional behavior-cloning support regularizer to stay near logged behavior
    """

    name = "conservative_dqn"

    def __init__(
        self,
        gamma: float = 0.90,
        cql_alpha: float = 0.10,
        hidden_dim: int = 64,
        seed: int = 7,
        ensemble_size: int = 3,
        uncertainty_beta: float = 0.50,
        behavior_clone_alpha: float = 0.02,
        uncertainty_fallback_threshold: float | None = None,
        use_double_dqn: bool = True,
    ) -> None:
        self.gamma = float(gamma)
        self.cql_alpha = float(cql_alpha)
        self.hidden_dim = int(hidden_dim)
        self.seed = int(seed)
        self.ensemble_size = max(1, int(ensemble_size))
        self.uncertainty_beta = float(uncertainty_beta)
        self.behavior_clone_alpha = float(behavior_clone_alpha)
        self.uncertainty_fallback_threshold = uncertainty_fallback_threshold
        self.use_double_dqn = bool(use_double_dqn)
        self.policy_version = "conservative_dqn_ensemble_v2"
        self.model = None
        self.models: list = []
        self.training_summary: dict = {
            "trained": False,
            "message": "Policy has not been trained; heuristic prior will be used.",
        }

    def _ensure_models(self) -> None:
        if self.models:
            self.model = self.models[0]
            return
        self.models = [
            _build_q_network(input_dim=STATE_VECTOR_DIM, hidden_dim=self.hidden_dim)
            for _ in range(self.ensemble_size)
        ]
        self.model = self.models[0]

    def _ensure_model(self) -> None:
        """Backward-compatible helper for code that expects a single model."""
        self._ensure_models()

    def encode_state(self, state: dict) -> np.ndarray:
        return encode_state_vector(state)

    def _q_matrix(self, state: dict) -> np.ndarray:
        if not self.models:
            prior = heuristic_action_prior(state)
            return np.array([[prior[action] for action in ACTIONS]], dtype=np.float32)

        torch, _, _ = _require_torch()
        x = torch.tensor(self.encode_state(state), dtype=torch.float32).unsqueeze(0)
        outputs = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                outputs.append(model(x).squeeze(0).detach().cpu().numpy())
        return np.stack(outputs).astype(np.float32)

    def ensemble_action_statistics(self, state: dict) -> dict[str, dict[str, float]]:
        q_matrix = self._q_matrix(state)
        mean = q_matrix.mean(axis=0)
        std = q_matrix.std(axis=0)
        pessimistic = mean - self.uncertainty_beta * std
        return {
            "mean": {action: float(mean[i]) for i, action in enumerate(ACTIONS)},
            "uncertainty": {action: float(std[i]) for i, action in enumerate(ACTIONS)},
            "pessimistic": {action: float(pessimistic[i]) for i, action in enumerate(ACTIONS)},
        }

    def action_values(self, state: dict) -> dict[str, float]:
        """Return the deployment score: pessimistic ensemble Q-values.

        Existing evaluation code can keep calling action_values(state), while the
        policy internally uses mean(Q) - beta * std(Q) instead of raw Q-values.
        """
        return self.ensemble_action_statistics(state)["pessimistic"]

    def recommend(self, state: dict) -> str:
        return self.recommend_with_explanation(state)["recommended_action"]

    def recommend_with_explanation(self, state: dict) -> dict:
        stats = self.ensemble_action_statistics(state)
        raw_scores = stats["mean"]
        uncertainty = stats["uncertainty"]
        pessimistic_scores = stats["pessimistic"]

        max_uncertainty = max(uncertainty.values()) if uncertainty else 0.0
        fallback_triggered = False
        scores_for_serving = pessimistic_scores
        if (
            self.uncertainty_fallback_threshold is not None
            and self.models
            and max_uncertainty > float(self.uncertainty_fallback_threshold)
        ):
            # Keep the API behavior unchanged while avoiding high-disagreement model outputs.
            scores_for_serving = heuristic_action_prior(state)
            fallback_triggered = True

        served_scores, blocked_actions, safety_notes = apply_liveops_safety_mask(
            state,
            scores_for_serving,
            value_label="Pessimistic ensemble Q-values",
        )
        safety_notes.append(
            "Candidate policy: conservative offline DQN with Double DQN targets, behavior-support regularization, and ensemble uncertainty."
        )
        if fallback_triggered:
            safety_notes.append(
                "Uncertainty fallback triggered: model disagreement exceeded the configured threshold, so heuristic support scores were used before safety masking."
            )

        return {
            "recommended_action": max(served_scores, key=served_scores.get),
            "raw_action_scores": raw_scores,
            "served_action_scores": served_scores,
            "blocked_actions": blocked_actions,
            "safety_notes": safety_notes,
            # Extra fields are useful in logs and direct debugging. The current
            # RecommendationResponse schema will ignore them unless it is extended.
            "pessimistic_action_scores": pessimistic_scores,
            "action_uncertainty": uncertainty,
            "uncertainty_beta": self.uncertainty_beta,
            "max_action_uncertainty": float(max_uncertainty),
            "fallback_triggered": fallback_triggered,
            "policy_name": self.name,
            "training_summary": self.training_summary,
        }

    def _build_training_tensors(self, df: pd.DataFrame):
        required = {
            "segment",
            "skill",
            "frustration",
            "engagement",
            "churn_risk",
            "economy_balance",
            "recent_losses",
            "recent_rewards",
            "day",
            "action",
            "reward",
            "next_skill",
            "next_frustration",
            "next_engagement",
            "next_churn_risk",
            "next_economy_balance",
            "next_recent_losses",
            "next_recent_rewards",
        }
        missing = sorted(required - set(df.columns)) if df is not None else sorted(required)
        if df is None or df.empty or missing:
            return None, missing

        states: list[np.ndarray] = []
        actions: list[int] = []
        rewards: list[float] = []
        next_states: list[np.ndarray] = []
        dones: list[float] = []
        skipped = 0

        for _, row in df.iterrows():
            action = str(row.get("action"))
            if action not in ACTION_TO_INDEX:
                skipped += 1
                continue
            try:
                state = _state_from_row(row)
                next_state = _next_state_from_row(row)
                states.append(encode_state_vector(state))
                actions.append(ACTION_TO_INDEX[action])
                rewards.append(float(row["reward"]))
                next_states.append(encode_state_vector(next_state))
                retained = bool(row.get("retained", True))
                dones.append(0.0 if retained else 1.0)
            except Exception:
                skipped += 1

        if not states:
            return None, ["No valid rows after filtering logged transitions."]

        torch, _, _ = _require_torch()
        tensors = {
            "states": torch.tensor(np.stack(states), dtype=torch.float32),
            "actions": torch.tensor(actions, dtype=torch.long),
            "rewards": torch.tensor(rewards, dtype=torch.float32),
            "next_states": torch.tensor(np.stack(next_states), dtype=torch.float32),
            "dones": torch.tensor(dones, dtype=torch.float32),
            "skipped_rows": skipped,
        }
        return tensors, []

    def fit_from_logged(
        self,
        df: pd.DataFrame,
        epochs: int = 25,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        gamma: float | None = None,
        cql_alpha: float | None = None,
        behavior_clone_alpha: float | None = None,
        ensemble_size: int | None = None,
        uncertainty_beta: float | None = None,
        target_update_every: int = 5,
        seed: int | None = None,
        bootstrap_ensemble: bool = True,
        use_double_dqn: bool | None = None,
    ) -> "ConservativeDQNPolicy":
        """Train the conservative offline DQN from logged LiveOps transitions.

        The model remains plug-compatible with the existing application. Training
        is offline: logged transitions are converted into tensors and used to
        learn Q-values without direct online exploration.
        """
        if gamma is not None:
            self.gamma = float(gamma)
        if cql_alpha is not None:
            self.cql_alpha = float(cql_alpha)
        if behavior_clone_alpha is not None:
            self.behavior_clone_alpha = float(behavior_clone_alpha)
        if ensemble_size is not None:
            self.ensemble_size = max(1, int(ensemble_size))
            self.models = []
            self.model = None
        if uncertainty_beta is not None:
            self.uncertainty_beta = float(uncertainty_beta)
        if use_double_dqn is not None:
            self.use_double_dqn = bool(use_double_dqn)
        if seed is not None:
            self.seed = int(seed)

        if not _torch_available():
            self.training_summary = {
                "trained": False,
                "message": "PyTorch is not installed; conservative DQN was skipped.",
            }
            return self

        torch, nn, F = _require_torch()
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        tensors, missing = self._build_training_tensors(df)
        if tensors is None:
            self.training_summary = {
                "trained": False,
                "message": "Conservative DQN skipped because logged transition data was unavailable or incomplete.",
                "missing_columns": missing,
            }
            return self

        self._ensure_models()
        n = int(tensors["states"].shape[0])
        batch_size = max(1, min(int(batch_size), n))
        epochs = max(1, int(epochs))
        target_update_every = max(1, int(target_update_every))

        model_summaries = []
        for model_idx, model in enumerate(self.models):
            model_seed = self.seed + model_idx * 9973
            torch.manual_seed(model_seed)
            np.random.seed(model_seed)
            random.seed(model_seed)

            target_model = _build_q_network(input_dim=STATE_VECTOR_DIM, hidden_dim=self.hidden_dim)
            target_model.load_state_dict(model.state_dict())
            target_model.eval()

            optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
            mse = nn.MSELoss()
            last_total_loss = 0.0
            last_td_loss = 0.0
            last_cql_penalty = 0.0
            last_bc_loss = 0.0

            for epoch in range(epochs):
                if bootstrap_ensemble and len(self.models) > 1:
                    permutation = torch.randint(low=0, high=n, size=(n,))
                else:
                    permutation = torch.randperm(n)

                epoch_total = 0.0
                epoch_td = 0.0
                epoch_cql = 0.0
                epoch_bc = 0.0
                batches = 0

                model.train()
                for start in range(0, n, batch_size):
                    idx = permutation[start : start + batch_size]
                    states = tensors["states"][idx]
                    action_idx = tensors["actions"][idx]
                    rewards = tensors["rewards"][idx]
                    next_states = tensors["next_states"][idx]
                    dones = tensors["dones"][idx]

                    q_all = model(states)
                    q_logged = q_all.gather(1, action_idx.unsqueeze(1)).squeeze(1)

                    with torch.no_grad():
                        if self.use_double_dqn:
                            next_actions = model(next_states).argmax(dim=1)
                            next_q_all = target_model(next_states)
                            next_q = next_q_all.gather(1, next_actions.unsqueeze(1)).squeeze(1)
                        else:
                            next_q = target_model(next_states).max(dim=1).values
                        target = rewards + self.gamma * (1.0 - dones) * next_q

                    td_loss = mse(q_logged, target)
                    cql_penalty = torch.logsumexp(q_all, dim=1).mean() - q_logged.mean()
                    bc_loss = F.cross_entropy(q_all, action_idx)
                    loss = td_loss + self.cql_alpha * cql_penalty + self.behavior_clone_alpha * bc_loss

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()

                    epoch_total += float(loss.detach().cpu())
                    epoch_td += float(td_loss.detach().cpu())
                    epoch_cql += float(cql_penalty.detach().cpu())
                    epoch_bc += float(bc_loss.detach().cpu())
                    batches += 1

                if (epoch + 1) % target_update_every == 0:
                    target_model.load_state_dict(model.state_dict())

                last_total_loss = epoch_total / max(1, batches)
                last_td_loss = epoch_td / max(1, batches)
                last_cql_penalty = epoch_cql / max(1, batches)
                last_bc_loss = epoch_bc / max(1, batches)

            model.eval()
            model_summaries.append(
                {
                    "model_index": model_idx,
                    "seed": model_seed,
                    "last_loss": last_total_loss,
                    "last_td_loss": last_td_loss,
                    "last_cql_penalty": last_cql_penalty,
                    "last_behavior_clone_loss": last_bc_loss,
                }
            )

        self.model = self.models[0] if self.models else None
        self.training_summary = {
            "trained": True,
            "algorithm": "conservative_dqn_ensemble",
            "objective": "TD loss + cql_alpha * CQL penalty + behavior_clone_alpha * CE(logged_action)",
            "targeting": "Double DQN" if self.use_double_dqn else "max target DQN",
            "uncertainty_scoring": "mean(Q) - uncertainty_beta * std(Q)",
            "rows": n,
            "skipped_rows": int(tensors.get("skipped_rows", 0)),
            "epochs": epochs,
            "batch_size": batch_size,
            "gamma": self.gamma,
            "cql_alpha": self.cql_alpha,
            "behavior_clone_alpha": self.behavior_clone_alpha,
            "learning_rate": float(learning_rate),
            "hidden_dim": self.hidden_dim,
            "ensemble_size": len(self.models),
            "uncertainty_beta": self.uncertainty_beta,
            "uncertainty_fallback_threshold": self.uncertainty_fallback_threshold,
            "bootstrap_ensemble": bool(bootstrap_ensemble),
            "model_summaries": model_summaries,
        }
        return self

    def train_from_logged(self, df: pd.DataFrame, **kwargs) -> "ConservativeDQNPolicy":
        """Alias for fit_from_logged so training code can mirror QLearningPolicy."""
        return self.fit_from_logged(df, **kwargs)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.models:
            path.write_text(
                json.dumps(
                    {
                        "policy_version": self.policy_version,
                        "gamma": self.gamma,
                        "cql_alpha": self.cql_alpha,
                        "hidden_dim": self.hidden_dim,
                        "seed": self.seed,
                        "ensemble_size": self.ensemble_size,
                        "uncertainty_beta": self.uncertainty_beta,
                        "behavior_clone_alpha": self.behavior_clone_alpha,
                        "uncertainty_fallback_threshold": self.uncertainty_fallback_threshold,
                        "use_double_dqn": self.use_double_dqn,
                        "training_summary": self.training_summary,
                        "model_state_dict": None,
                        "model_state_dicts": None,
                    },
                    indent=2,
                )
            )
            return

        torch, _, _ = _require_torch()
        torch.save(
            {
                "policy_version": self.policy_version,
                "gamma": self.gamma,
                "cql_alpha": self.cql_alpha,
                "hidden_dim": self.hidden_dim,
                "seed": self.seed,
                "ensemble_size": len(self.models),
                "uncertainty_beta": self.uncertainty_beta,
                "behavior_clone_alpha": self.behavior_clone_alpha,
                "uncertainty_fallback_threshold": self.uncertainty_fallback_threshold,
                "use_double_dqn": self.use_double_dqn,
                "input_dim": STATE_VECTOR_DIM,
                "actions": ACTIONS,
                "training_summary": self.training_summary,
                "model_state_dict": self.models[0].state_dict(),
                "model_state_dicts": [model.state_dict() for model in self.models],
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "ConservativeDQNPolicy":
        path = Path(path)
        try:
            torch, _, _ = _require_torch()
            data = torch.load(path, map_location="cpu")
        except Exception:
            # Backward-compatible fallback for JSON metadata written when PyTorch was unavailable.
            data = json.loads(path.read_text())

        policy = cls(
            gamma=float(data.get("gamma", 0.90)),
            cql_alpha=float(data.get("cql_alpha", 0.10)),
            hidden_dim=int(data.get("hidden_dim", 64)),
            seed=int(data.get("seed", 7)),
            ensemble_size=int(data.get("ensemble_size", 1)),
            uncertainty_beta=float(data.get("uncertainty_beta", 0.50)),
            behavior_clone_alpha=float(data.get("behavior_clone_alpha", 0.02)),
            uncertainty_fallback_threshold=data.get("uncertainty_fallback_threshold"),
            use_double_dqn=bool(data.get("use_double_dqn", True)),
        )
        policy.policy_version = data.get("policy_version", "conservative_dqn_ensemble_v2")
        policy.training_summary = data.get("training_summary", {})

        model_states = data.get("model_state_dicts")
        single_state = data.get("model_state_dict")
        if model_states is None and single_state is not None:
            model_states = [single_state]

        if model_states:
            policy.models = []
            for state_dict in model_states:
                model = _build_q_network(input_dim=STATE_VECTOR_DIM, hidden_dim=policy.hidden_dim)
                model.load_state_dict(state_dict)
                model.eval()
                policy.models.append(model)
            policy.ensemble_size = len(policy.models)
            policy.model = policy.models[0]

        return policy

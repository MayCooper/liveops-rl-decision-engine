import json
import logging
import os
import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.agents import (
generate_audit_summary,
generate_gemini_text,
gemini_error_message,
run_policy_audit,
)
from app.arena import AutoRolloutRequest, run_auto_rl_rollout, router as arena_router
from app.cloud_io import (
log_jsonl,
log_recommendation,
read_arena_episodes_dataframe,
read_recent_arena_events,
read_recent_audits,
read_recent_policy_metrics,
read_recent_recommendations,
read_synthetic_dataset_profile,
read_synthetic_episode_sample,
settings,
upload_arena_csv_to_bigquery,
)
from app.core import (
ACTIONS,
PlayerState,
RecommendationRequest,
RecommendationResponse,
compare_policies,
evaluate_off_policy,
simulate_dataset,
simulate_episode,
step_player,
)
from app.policies import ConservativeDQNPolicy, QLearningPolicy
from app.rag_chat import answer_agent_message, ollama_status

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
app = FastAPI(title="LiveOps Policy Lab", version="0.1.0")
app.include_router(arena_router)
policy: QLearningPolicy | None = None


PROJECT_SCOPE_MESSAGE = "This console is scoped to explaining the LiveOps policy recommendation, safety mask, metrics, and rollout audit."

def _safe_number(value: Any) -> Any:
    try:
        if value is None:
            return None
        if isinstance(value, (int, float, str, bool)):
            return value
        return float(value)
    except Exception:
        return value


def _artifact_status(path: str) -> dict:
    p = Path(path)
    return {"path": str(p), "exists": p.exists(), "size_bytes": p.stat().st_size if p.exists() else None}


def _latest_model_summary(summary: dict) -> dict:
    models = summary.get("model_summaries") or []
    last = models[-1] if models else {}
    return {
        "final_training_loss": _safe_number(last.get("last_loss")),
        "td_loss": _safe_number(last.get("last_td_loss")),
        "cql_penalty": _safe_number(last.get("last_cql_penalty")),
        "behavior_cloning_loss": _safe_number(last.get("last_behavior_clone_loss")),
    }


def _sample_liveops_state() -> dict:
    return {
        "segment": "new",
        "skill": 0.35,
        "frustration": 0.82,
        "engagement": 0.42,
        "churn_risk": 0.68,
        "economy_balance": 0.30,
        "recent_losses": 5,
        "recent_rewards": 1,
        "day": 3,
    }

def load_policy() -> QLearningPolicy:
    path = Path(settings.POLICY_ARTIFACT_PATH)
    if path.exists():
        loaded = QLearningPolicy.load(path)
        ensure_metrics(loaded)
        return loaded
    logging.warning("Policy artifact missing at %s; training a small local policy.", path)
    trained = QLearningPolicy().train(n_episodes=600, days=10)
    trained.save(path)
    ensure_metrics(trained)
    return trained


def ensure_metrics(q_policy: QLearningPolicy) -> None:
    path = Path(settings.POLICY_METRICS_PATH)
    if path.exists():
        metrics = json.loads(path.read_text())
        if metrics.get("off_policy_evaluation"):
            return
        logging.warning("Policy metrics missing OPE section at %s; adding deterministic OPE metrics.", path)
        metrics["off_policy_evaluation"] = _compute_ope_for_policy(q_policy)
        path.write_text(json.dumps(metrics, indent=2))
        return
    logging.warning("Policy metrics missing at %s; generating fallback metrics.", path)
    from app.policies import BanditPolicy, RuleBasedPolicy

    rule = RuleBasedPolicy()
    rule_df = simulate_dataset(rule.recommend, n_players=250, days=14, seed=20, policy_name=rule.name)
    bandit = BanditPolicy().fit(rule_df)
    metrics = compare_policies(rule, bandit, q_policy)
    metrics["off_policy_evaluation"] = _compute_ope_for_policy(q_policy)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2))


def _compute_ope_for_policy(q_policy: QLearningPolicy) -> dict:
    data_path = Path("data/arena_liveops_episodes.csv")
    try:
        logged_source = str(data_path)
        if settings.use_bigquery or data_path.exists():
            try:
                from scripts.train_and_eval import arena_to_liveops_df

                raw_logged, logged_source = read_arena_episodes_dataframe()
                logged = arena_to_liveops_df(raw_logged)
            except Exception:
                logged, logged_source = read_arena_episodes_dataframe()
            result = evaluate_off_policy(q_policy, logged)
            result["dataset"] = logged_source
            return result
        from app.policies import RuleBasedPolicy

        generated = simulate_dataset(RuleBasedPolicy().recommend, n_players=300, days=14, seed=20, policy_name="rule")
        result = evaluate_off_policy(q_policy, generated)
        result["dataset"] = "generated_fallback"
        result.setdefault("warnings", []).append("data/arena_liveops_episodes.csv was missing; used deterministic generated fallback data.")
        return result
    except Exception as exc:
        logging.warning("OPE fallback computation failed: %s", exc)
        return {
            "available": False,
            "message": "OPE metrics could not be computed.",
            "candidate_policy": "q_learning",
            "dataset": str(data_path),
            "warnings": [str(exc)],
            "trace_examples": [],
        }

@app.on_event("startup")
def startup() -> None:
    global policy
    policy = load_policy()


@app.get("/", include_in_schema=False)
def browser_demo() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/agent-ops", include_in_schema=False)
def agent_ops_console() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "ops_console.html")


@app.get("/ops/console", include_in_schema=False)
def ops_console() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "ops_console.html")


@app.get("/health")
def health() -> dict:
    service_name = os.getenv("K_SERVICE")
    repo_csv = Path("data/arena_liveops_episodes.csv")
    repo_metrics = Path(settings.POLICY_METRICS_PATH)
    repo_policy = Path(settings.POLICY_ARTIFACT_PATH)
    effective_data_source = "bigquery" if settings.use_bigquery else "repo"
    profile_source = "bigquery" if settings.use_bigquery else "repo_json"
    episode_source = "bigquery" if settings.use_bigquery else "repo_csv" if repo_csv.exists() else "generated_fallback"
    logging_sink = "bigquery" if settings.use_bigquery else "local_jsonl"
    explanation_mode = "gemini" if settings.use_gemini else "deterministic_fallback"
    cloud_available = bool(settings.cloud_available())
    cloud_enabled = bool(settings.RUNTIME_MODE == "cloud" and (settings.use_bigquery or settings.use_gemini))
    return {
        "status": "ok",
        "policy_loaded": policy is not None,
        "runtime_mode": settings.RUNTIME_MODE,
        "requested_data_source": settings.DATA_SOURCE,
        "data_source": effective_data_source,
        "data_path": "BigQuery" if settings.use_bigquery else "local CSV/JSON",
        "logging_sink": logging_sink,
        "explanation_mode": explanation_mode,
        "chat_provider": settings.CHAT_PROVIDER,
        "chat_providers": ["offline", "ollama", "gemini"],
        "ollama_model": settings.OLLAMA_MODEL,
        "use_bigquery": settings.use_bigquery,
        "use_gemini": settings.use_gemini,
        "bigquery_configured": settings.bigquery_configured,
        "gemini_configured": settings.gemini_configured,
        "gemini_provider": settings.GEMINI_PROVIDER,
        "gemini_model": settings.GEMINI_MODEL_FAST,
        "gemini_location": settings.GEMINI_LOCATION,
        "cloud_available": cloud_available,
        "cloud_features_enabled": cloud_enabled,
        "cloud_runtime_enabled": cloud_enabled,
        "vertex_rl_serving": False,
        "policy_serving": "local artifact: " + str(repo_policy),
        "cloud_message": "Cloud path active for configured BigQuery/Gemini features." if cloud_enabled else "Local path active: CSV/JSON artifacts, local JSONL logs, deterministic explanation fallback.",
        "env_file_loaded": settings.ENV_FILE_LOADED,
        "repo_data_available": repo_csv.exists(),
        "repo_artifacts_available": repo_policy.exists() and repo_metrics.exists(),
        "gcp_project": settings.GCP_PROJECT or os.getenv("GOOGLE_CLOUD_PROJECT"),
        "bq_dataset": settings.BQ_DATASET,
        "runtime": "cloud_run" if service_name else "local",
        "service_name": service_name,
        "revision": os.getenv("K_REVISION"),
        "region": settings.REGION or os.getenv("CLOUD_RUN_REGION"),
        "dataset_profile_source": profile_source,
        "synthetic_episode_source": episode_source,
    }

@app.post("/runtime_mode")
def runtime_mode(payload: dict = Body(default_factory=dict)) -> dict:
    requested = str(payload.get("mode", "local")).strip().lower()
    try:
        before = settings.RUNTIME_MODE
        actual = settings.set_runtime_mode(requested)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    result = health()
    result["requested_runtime_mode"] = requested

    if requested in {"cloud", "auto"} and actual != "cloud":
        result["runtime_warning"] = (
            "Cloud mode was requested, but this app does not have configured cloud features. "
            "It stayed in local mode. Set USE_BIGQUERY=true with GCP_PROJECT and BQ_DATASET, "
            "and/or ENABLE_GEMINI=true with Gemini credentials."
        )
    elif before != actual:
        result["runtime_message"] = f"Runtime mode changed from {before} to {actual}."
    else:
        result["runtime_message"] = f"Runtime mode remains {actual}."
    return result


@app.post("/recommend_action", response_model=RecommendationResponse)
def recommend_action(request: RecommendationRequest) -> RecommendationResponse:
    if policy is None:
        raise HTTPException(status_code=503, detail="Policy not loaded. Run python scripts/train_and_eval.py.")
    state = request.player.model_dump()
    explanation = policy.recommend_with_explanation(state)
    logged = log_recommendation({"request_id": request.request_id, "player": state, **explanation})
    return RecommendationResponse(policy_version=policy.policy_version, logged=logged, **explanation)


@app.get("/dataset_profile")
def dataset_profile() -> dict:
    return read_synthetic_dataset_profile()


@app.get("/synthetic_episode_sample")
def synthetic_episode_sample(limit: int = 20) -> dict:
    return read_synthetic_episode_sample(limit)


@app.post("/sync_arena_csv_to_bigquery")
def sync_arena_csv_to_bigquery(payload: dict = Body(default_factory=dict)) -> dict:
    replace = bool(payload.get("replace", False))
    return upload_arena_csv_to_bigquery(replace=replace)


@app.post("/auto_rl_rollout")
def auto_rl_rollout_alias(request: AutoRolloutRequest) -> dict:
    return run_auto_rl_rollout(request.state, horizon=request.horizon, seed=request.seed, request_id=request.request_id)

@app.get("/rl_policy_trace")
def rl_policy_trace() -> dict:
    metrics: dict[str, Any] = {}
    metrics_path = Path(settings.POLICY_METRICS_PATH)
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            metrics = {}

    q_artifact = _artifact_status(settings.POLICY_ARTIFACT_PATH)
    dqn_candidates = [
        "artifacts/conservative_dqn_policy.pt",
        "artifacts/conservative_dqn.pt",
        "artifacts/dqn_policy.pt",
        "artifacts/conservative_dqn_policy.json",
    ]
    dqn_artifacts = [_artifact_status(path) for path in dqn_candidates]
    dqn_artifact = next((item for item in dqn_artifacts if item["exists"]), dqn_artifacts[0])

    dqn_config = ConservativeDQNPolicy()
    dqn_training_summary: dict[str, Any] = dict(dqn_config.training_summary or {})
    dqn_loaded = False
    dqn_load_error = None
    if dqn_artifact.get("exists"):
        try:
            loaded_dqn = ConservativeDQNPolicy.load(dqn_artifact["path"])
            dqn_config = loaded_dqn
            dqn_training_summary = dict(getattr(loaded_dqn, "training_summary", {}) or {})
            dqn_loaded = bool(getattr(loaded_dqn, "models", []))
        except Exception as exc:
            dqn_load_error = str(exc)

    serving_name = getattr(policy, "name", "unloaded") if policy is not None else "unloaded"
    sample_state = _sample_liveops_state()
    example_trace: dict[str, Any] = {"available": False, "state": sample_state}
    if policy is not None:
        try:
            explanation = policy.recommend_with_explanation(sample_state)
            raw_scores = explanation.get("raw_action_scores") or {}
            served_scores = explanation.get("served_action_scores") or {}
            pessimistic = explanation.get("pessimistic_action_scores") or raw_scores
            top_raw = max(raw_scores, key=raw_scores.get) if raw_scores else None
            top_served = max(served_scores, key=served_scores.get) if served_scores else None
            example_trace = {
                "available": True,
                "encoded_state_summary": {
                    "segment": sample_state["segment"],
                    "frustration": sample_state["frustration"],
                    "churn_risk": sample_state["churn_risk"],
                    "economy_balance": sample_state["economy_balance"],
                    "recent_losses": sample_state["recent_losses"],
                },
                "raw_top_action": top_raw,
                "top_action_after_uncertainty_penalty": max(pessimistic, key=pessimistic.get) if pessimistic else top_raw,
                "served_action": explanation.get("recommended_action") or top_served,
                "blocked_actions": explanation.get("blocked_actions", []),
                "raw_q_values": raw_scores,
                "served_q_values": served_scores,
                "action_uncertainty": explanation.get("action_uncertainty") or {},
            }
        except Exception as exc:
            example_trace = {"available": False, "state": sample_state, "error": str(exc)}

    q_metrics = metrics.get("q_learning") or {}
    ope = metrics.get("off_policy_evaluation") or {}
    safety_gate = metrics.get("safety_gate") or {}
    training_summary = dqn_training_summary or {}
    loss_summary = _latest_model_summary(training_summary)

    return {
        "title": "RL Policy Trace",
        "decision_pipeline": "player telemetry -> state encoder -> conservative DQN ensemble -> Double DQN value estimate -> uncertainty penalty -> safety mask -> final LiveOps action",
        "serving_policy": {
            "current_serving_policy": serving_name,
            "policy_version": getattr(policy, "policy_version", None) if policy is not None else None,
            "model_type": "conservative offline DQN ensemble" if serving_name == "conservative_dqn" else "tabular Q-learning fallback",
        },
        "model_artifact_status": {
            "tabular_q_policy": q_artifact,
            "conservative_dqn_artifact": dqn_artifact,
            "all_dqn_candidates": dqn_artifacts,
            "dqn_loaded": dqn_loaded,
            "dqn_load_error": dqn_load_error,
            "torch_installed": importlib.util.find_spec("torch") is not None,
        },
        "rl_training_method": {
            "conservative_dqn_available": True,
            "tabular_q_learning_fallback_available": q_artifact["exists"] or policy is not None,
            "method": training_summary.get("algorithm") or "conservative_dqn_ensemble when artifact exists; tabular Q-learning otherwise",
            "targeting": training_summary.get("targeting") or "Double DQN when ConservativeDQNPolicy is trained",
            "objective": training_summary.get("objective") or "TD loss + CQL penalty + behavior cloning regularizer",
            "train_episodes_or_logged_transitions": training_summary.get("rows") or q_metrics.get("training_rows"),
            "gamma": getattr(dqn_config, "gamma", None),
            "cql_alpha": getattr(dqn_config, "cql_alpha", None),
            "behavior_cloning_weight": getattr(dqn_config, "behavior_clone_alpha", None),
            **loss_summary,
        },
        "action_space": {"number_of_actions": len(ACTIONS), "actions": list(ACTIONS)},
        "ensemble_uncertainty": {
            "ensemble_size": getattr(dqn_config, "ensemble_size", None),
            "uncertainty_beta": getattr(dqn_config, "uncertainty_beta", None),
            "uncertainty_aware_scoring_active": True,
            "scoring": "mean_q - beta * std_q",
            "uncertainty_fallback_threshold": getattr(dqn_config, "uncertainty_fallback_threshold", None),
        },
        "safety_mask": {
            "deterministic_safety_mask_active": True,
            "status": safety_gate.get("decision") or "active before serving",
            "allow": safety_gate.get("allow"),
            "reasons": safety_gate.get("reasons", []),
            "shared_with_tabular_and_dqn": True,
        },
        "evaluation_metrics": {
            "latest_policy_metrics": q_metrics,
            "ope": {
                "available": ope.get("available"),
                "ips": ope.get("ips_estimate"),
                "snips": ope.get("snips_estimate"),
                "clipped_ips": ope.get("clipped_ips_estimate"),
                "effective_sample_size": ope.get("effective_sample_size"),
                "matched_rows": ope.get("matched_rows"),
                "total_rows": ope.get("total_rows"),
                "dataset": ope.get("dataset"),
            },
        },
        "fallback_behavior": {
            "if_torch_missing": "Conservative DQN is skipped; tabular Q-learning artifact or heuristic prior remains available.",
            "if_dqn_artifact_missing": "Current app serves the tabular Q-learning policy and reports DQN artifact as missing.",
            "if_metrics_missing": "Endpoint returns N/A/null fields instead of failing.",
        },
        "example_trace": example_trace,
    }
@app.get("/policy_metrics")
def policy_metrics() -> dict:
    path = Path(settings.POLICY_METRICS_PATH)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Metrics artifact missing. Run python scripts/train_and_eval.py.")
    return json.loads(path.read_text())



@app.get("/ope_metrics")
def ope_metrics() -> dict:
    metrics = policy_metrics()
    ope = metrics.get("off_policy_evaluation")
    if not ope:
        return {"available": False, "message": "OPE metrics not found. Run python scripts/train_and_eval.py first."}
    return ope
@app.post("/simulate_episode")
def simulate_episode_endpoint(request: Any = Body(default=None), days: int = 14, seed: int | None = 42) -> dict:
    if policy is None:
        raise HTTPException(status_code=503, detail="Policy not loaded.")
    if isinstance(request, dict) and request.get("player"):
        request = RecommendationRequest.model_validate(request)
    if isinstance(request, RecommendationRequest):
        state = request.player.model_dump()
        rows = []
        import numpy as np

        rng = np.random.default_rng(seed)
        for _ in range(days):
            before = dict(state)
            action = policy.recommend(before)
            state, reward, info = step_player(before, action, rng)
            rows.append(
                {
                    "day": before["day"],
                    "segment": before["segment"],
                    "action": action,
                    "reward": reward,
                    "retained": info["retained"],
                    "engagement": before["engagement"],
                    "frustration": before["frustration"],
                    "churn_risk": before["churn_risk"],
                    "economy_balance": before["economy_balance"],
                    "next_engagement": state["engagement"],
                    "next_frustration": state["frustration"],
                    "next_churn_risk": state["churn_risk"],
                    "next_economy_balance": state["economy_balance"],
                }
            )
            if not info["retained"]:
                break
    else:
        rows = simulate_episode(policy.recommend, days=days, seed=seed, policy_name=policy.name)
    return {"rows": rows[:days]}


@app.post("/run_agent_audit")
def run_agent_audit_endpoint() -> dict:
    if policy is None:
        raise HTTPException(status_code=503, detail="Policy not loaded.")
    metrics = policy_metrics()
    report = run_policy_audit(metrics, policy.recommend)
    response = report.model_dump()
    response.update(generate_audit_summary(report, metrics))
    return response


@app.get("/chat_provider_status")
def chat_provider_status() -> dict:
    ollama = ollama_status()
    return {
        "runtime_mode": settings.RUNTIME_MODE,
        "offline": {"available": settings.RUNTIME_MODE != "cloud", "message": "Disabled in cloud mode; use Gemini or Ollama for cloud-path chat." if settings.RUNTIME_MODE == "cloud" else "Available in local mode; deterministic RAG fallback."},
        "ollama": {**ollama, "available": bool(ollama.get("available") and settings.RUNTIME_MODE != "cloud"), "message": "Disabled in cloud mode; Ollama is a local runtime." if settings.RUNTIME_MODE == "cloud" else ollama.get("message")},
        "gemini": {"available": settings.use_gemini, "message": "Available in cloud mode when Gemini credentials are configured." if settings.use_gemini else "Disabled unless cloud/Gemini credentials are active."},
    }

@app.post("/agent_message")
def agent_message(payload: dict) -> dict:
    message = str(payload.get("message", "")).strip()
    context = payload.get("context") or {}
    provider = payload.get("provider") or context.get("chat_provider")
    session_id = str(payload.get("session_id") or context.get("session_id") or "default")
    result = answer_agent_message(message, context, provider=provider, session_id=session_id)
    try:
        log_jsonl(
            {
                "event": "agent_message",
                "message": message[:300],
                "provider": result.get("provider"),
                "use_gemini": result.get("use_gemini", False),
                "sources": result.get("sources", []),
                "response": str(result.get("response", ""))[:800],
            }
        )
    except Exception as exc:
        logging.warning("Agent message logging failed: %s", exc)
    return result


@app.get("/recent_local_logs")
def recent_local_logs(limit: int = 20) -> dict:
    path = Path(settings.LOCAL_LOG_PATH)
    if not path.exists():
        return {"logs": []}
    lines = path.read_text(encoding="utf-8").splitlines()[-limit:]
    return {"logs": [json.loads(line) for line in lines if line.strip()]}


@app.get("/recent_logs")
def recent_logs(limit: int = 20) -> dict:
    if settings.use_bigquery:
        return {
            "source": "bigquery",
            "recommendations": read_recent_recommendations(limit),
            "audits": read_recent_audits(limit),
            "policy_metrics": read_recent_policy_metrics(limit),
            "arena_events": read_recent_arena_events(limit),
        }
    local = recent_local_logs(limit)
    local["source"] = "local_jsonl"
    return local


@app.get("/recent_recommendations")
def recent_recommendations(limit: int = 5) -> dict:
    return read_recent_recommendations(limit)


@app.get("/recent_audits")
def recent_audits(limit: int = 5) -> dict:
    return read_recent_audits(limit)


@app.get("/recent_policy_metrics")
def recent_policy_metrics(limit: int = 5) -> dict:
    return read_recent_policy_metrics(limit)


def _deterministic_agent_response(message: str, context: dict) -> str:
    m = message.lower()
    last_recommendation = context.get("last_recommendation") or {}
    last_audit = context.get("last_audit") or {}
    last_metrics = context.get("last_metrics") or {}
    last_ope = context.get("last_ope") or (last_metrics.get("off_policy_evaluation") or {})

    if "explain this agent run" in m or "explain the agent run" in m:
        action = last_recommendation.get("recommended_action", "unknown")
        blocked = len(last_recommendation.get("blocked_actions") or [])
        decision = last_audit.get("rollout_decision", "unknown")
        human = last_audit.get("human_approval_required", "unknown")
        match_rate = last_ope.get("match_rate", "unknown") if last_ope else "unknown"
        ess = last_ope.get("effective_sample_size", "unknown") if last_ope else "unknown"
        snips = last_ope.get("snips_estimate", "unknown") if last_ope else "unknown"
        return (
            f"The single RL decision agent served `{action}` after applying the deterministic safety mask, with {blocked} actions blocked before rollout. "
            f"The simulation replay then used that served policy action rather than Gemini. The deterministic audit/explanation layer reviewed the computed OPE and policy metrics, including "
            f"SNIPS={snips}, match_rate={match_rate}, and effective_sample_size={ess}, and returned a rollout decision of `{decision}` with human approval required set to `{human}`. "
            "This explanation summarizes the computed run; it does not change the action, safety mask, OPE metrics, or rollout decision."
        )
    if "explain ope" in m or "offline policy evaluation" in m:
        return "Offline Policy Evaluation estimates candidate policy value from logged behavior-policy data. A row provides direct evidence when the logged action matches the served candidate action; action probabilities convert those matched rewards into IPS, SNIPS, and clipped IPS estimates."

    if "match rate" in m:
        if last_ope:
            return f"OPE match rate is {last_ope.get('match_rate', 'unknown')}. It matters because only logged rows where the behavior action matches the served candidate action directly contribute evidence to importance-weighted estimates."
        return "Match rate matters because only logged rows where the behavior action matches the served candidate action directly contribute evidence to OPE estimates."

    if "ips vs snips" in m or ("ips" in m and "snips" in m):
        return "IPS averages propensity-corrected rewards for matched actions. SNIPS normalizes by the sum of matched importance weights, which can reduce scale sensitivity when propensities vary. Clipped IPS caps large weights to reduce variance."

    if "effective sample size" in m or "ess" in m:
        if last_ope:
            return f"Effective sample size is {last_ope.get('effective_sample_size', 'unknown')}. It summarizes how much useful weighted evidence remains after propensity weighting; concentrated high weights reduce ESS."
        return "Effective sample size summarizes how much useful weighted evidence remains after propensity weighting. If a few high-weight rows dominate, ESS is low even when the raw dataset is large."

    if "confident" in m and "ope" in m:
        match_rate = float(last_ope.get("match_rate") or 0) if last_ope else 0.0
        ess = float(last_ope.get("effective_sample_size") or 0) if last_ope else 0.0
        if not last_ope:
            return "No OPE context is loaded yet. Click Load OPE Metrics first."
        if match_rate >= 0.25 and ess >= 100:
            label = "stronger evidence"
        elif match_rate >= 0.10 and ess >= 25:
            label = "moderate evidence"
        else:
            label = "low evidence"
        return f"This OPE result has {label}: match_rate={match_rate:.3f}, effective_sample_size={ess:.1f}. Treat it as one logged-data signal, not the hard rollout decision."

    if "stable player" in m or ("stable" in m and "engag" in m):
        return (
            "For a stable player, avoid heavy intervention. Keep engagement with low-risk variety: recommend a fresh quest, offer a cosmetic reward, "
            "or hold steady if engagement and churn risk are healthy. Do not grant economy-changing bonuses unless churn or frustration rises. "
            "Watch engagement trend, churn risk, recent losses, and economy balance before escalating."
        )

    if "gemini" in m and ("hook" in m or "used" in m or "enabled" in m or "configured" in m):
        health = context.get("health") or {}
        if health.get("use_gemini"):
            return "Gemini is configured for explanation text. It can explain recommendations, metrics, OPE, and audit results, but it does not choose player actions or override the safety mask."
        return "Gemini support is wired in, but this running app reports use_gemini=false. Set ENABLE_GEMINI=true plus GEMINI_API_KEY or GOOGLE_API_KEY in the environment to enable Gemini explanations; deterministic fallback remains active without them."

    if "bigquery" in m or "big query" in m:
        health = context.get("health") or {}
        if health.get("use_bigquery"):
            return f"BigQuery is enabled for this runtime. Dataset: {health.get('bq_dataset', 'unknown')}. Recommendation, audit, and policy metric logs can use BigQuery."
        return "BigQuery support is wired in, but this running app reports use_bigquery=false and data_source=repo. Repo CSV/JSON and local JSONL logs are being used instead."
    if "explain last recommendation" in m or ("recommendation" in m and last_recommendation):
        if not last_recommendation:
            return "No recommendation context is available yet. Run one of the recommendation scenarios first."
        action = last_recommendation.get("recommended_action", "unknown")
        blocked = last_recommendation.get("blocked_actions") or []
        notes = last_recommendation.get("safety_notes") or []
        return (
            f"The policy served `{action}`. Raw scores are the Q-policy values before rollout constraints; served scores are after the deterministic safety mask. "
            f"{len(blocked)} actions were blocked. "
            f"Safety notes: {'; '.join(notes) if notes else 'none reported'}."
        )

    if "why were actions blocked" in m or "blocked" in m:
        blocked = last_recommendation.get("blocked_actions") or []
        if not blocked:
            return "No blocked-action context is available. Run a recommendation scenario to see safety-mask decisions."
        return "Blocked actions: " + " ".join(f"{b.get('action')}: {b.get('reason')}" for b in blocked)

    if "summarize rollout" in m or "rollout decision" in m:
        if not last_audit:
            return "No audit context is available yet. Run Agent Audit first."
        return (
            f"Rollout decision: {last_audit.get('rollout_decision', 'unknown')}. "
            f"Risky segments: {', '.join(last_audit.get('risky_segments') or []) or 'none'}. "
            f"Human approval required: {last_audit.get('human_approval_required', 'unknown')}."
        )

    if "monitor" in m:
        metrics = last_audit.get("monitoring_metrics") or ["avg_reward", "retention_rate", "reward_by_segment", "economy_penalty_avg", "frustration_final"]
        return "Monitor these rollout metrics: " + ", ".join(metrics) + "."

    if "rollback" in m:
        conditions = last_audit.get("rollback_conditions") or []
        if not conditions:
            return "No audit rollback conditions are loaded yet. Run Agent Audit first."
        return "Rollback triggers: " + "; ".join(conditions) + "."

    if "raw vs served" in m or "raw" in m and "served" in m:
        return "Raw scores are learned Q-policy scores for all candidate LiveOps actions. Served scores are the same candidates after deterministic rollout constraints block unsafe or inappropriate actions. The agent console explains this behavior but does not choose player actions."

    if "metric" in m or "reward" in m or "retention" in m:
        q = (last_metrics.get("q_learning") or {}) if last_metrics else {}
        if not q:
            return "No policy metrics are loaded yet. Click Policy Metrics first."
        return (
            f"Q-learning average reward is {q.get('avg_reward', 'unknown')}; retention rate is {q.get('retention_rate', 'unknown')}; "
            f"economy penalty average is {q.get('economy_penalty_avg', 'unknown')}."
        )

    return PROJECT_SCOPE_MESSAGE























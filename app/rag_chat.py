from __future__ import annotations

import json
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Iterator

import requests

from app.agents import gemini_error_message
from app.cloud_io import settings

SYSTEM_INSTRUCTION = (
    "You are the LiveOps Policy Lab explanation assistant. Explain only the RL policy, arena game context, "
    "safety gate, OPE/evaluation, rollout audit, and current simulator run. The LLM does not choose actions, "
    "alter the policy, or write training data. Ground answers in retrieved documents and live run context."
)

_MEMORY: dict[str, deque[dict[str, str]]] = defaultdict(lambda: deque(maxlen=8))
_DOC_CACHE: list[dict[str, str]] | None = None


def _read_json(path: str) -> Any:
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _load_docs() -> list[dict[str, str]]:
    global _DOC_CACHE
    if _DOC_CACHE is not None:
        return _DOC_CACHE
    docs: list[dict[str, str]] = []
    rag_dir = Path("docs/rag")
    if rag_dir.exists():
        for path in sorted(rag_dir.glob("*.md")):
            docs.append({"source": str(path), "text": path.read_text(encoding="utf-8")})
    artifacts = {
        "data/arena_dataset_profile.json": _read_json("data/arena_dataset_profile.json"),
        "data/frustration_feature_dictionary.json": _read_json("data/frustration_feature_dictionary.json"),
        "artifacts/policy_metrics.json": _read_json("artifacts/policy_metrics.json"),
    }
    for source, data in artifacts.items():
        if data is not None:
            docs.append({"source": source, "text": json.dumps(data, indent=2)[:12000]})
    _DOC_CACHE = docs
    return docs


def _retrieve(message: str, limit: int = 5) -> list[dict[str, str]]:
    terms = {t for t in re.findall(r"[a-zA-Z_]{3,}", message.lower())}
    scored: list[tuple[int, dict[str, str]]] = []
    for doc in _load_docs():
        text = doc["text"].lower()
        score = sum(text.count(term) for term in terms)
        if score:
            scored.append((score, doc))
    if not scored:
        return _load_docs()[:limit]
    return [doc for _, doc in sorted(scored, key=lambda item: item[0], reverse=True)[:limit]]


def _latest_run_events(limit: int = 8) -> list[dict[str, Any]]:
    path = Path(settings.LOCAL_LOG_PATH)
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines()[-120:]:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        event_type = row.get("event_type") or row.get("payload_json", "")
        if any(name in str(event_type) for name in ["arena_recommendation", "arena_action_applied", "arena_match_result", "arena_benchmark_run"]):
            events.append(row)
    return events[-limit:]


def _compact_context(context: dict[str, Any]) -> dict[str, Any]:
    return {
        "scenario": context.get("state", {}).get("scenario_name") or context.get("current_scenario"),
        "current_state": context.get("state"),
        "last_recommendation": context.get("last_recommendation"),
        "timeline": context.get("timeline"),
        "last_match": context.get("last_match"),
        "last_benchmark": context.get("last_benchmark"),
        "health": context.get("health"),
        "recent_logged_run_events": _latest_run_events(),
    }


def _prompt(message: str, context: dict[str, Any], docs: list[dict[str, str]], memory: list[dict[str, str]]) -> str:
    retrieved = "\n\n".join(f"Source: {d['source']}\n{d['text'][:3500]}" for d in docs)
    return (
        f"{SYSTEM_INSTRUCTION}\n\n"
        f"Recent chat memory JSON:\n{json.dumps(memory, default=str)[:3000]}\n\n"
        f"Live run context JSON:\n{json.dumps(_compact_context(context), default=str)[:8000]}\n\n"
        f"Retrieved policy/game/evaluation context:\n{retrieved}\n\n"
        f"User question:\n{message}\n\n"
        "Answer in plain text with short hyphen bullets. Do not use asterisks, markdown tables, or raw JSON. If the question asks about churn risk, use churn_risk/observed_churn_risk as a proxy, not a real next-week production forecast."
    )


def _ollama_prompt(message: str, context: dict[str, Any], docs: list[dict[str, str]], memory: list[dict[str, str]]) -> str:
    retrieved = "\n\n".join(f"Source: {d['source']}\n{d['text'][:900]}" for d in docs[:2])
    compact = {
        "scenario": context.get("state", {}).get("scenario_name") or context.get("current_scenario"),
        "last_recommendation": context.get("last_recommendation"),
        "last_match": context.get("last_match"),
    }
    return (
        "You explain a LiveOps RL demo. Be concise and factual. The LLM explains only; it never chooses actions.\n\n"
        f"Recent memory:\n{json.dumps(memory[-4:], default=str)[:700]}\n\n"
        f"Run context:\n{json.dumps(compact, default=str)[:1800]}\n\n"
        f"Retrieved context:\n{retrieved}\n\n"
        f"Question: {message}\n\n"
        "Answer in at most 8 short bullets. Avoid markdown tables and raw JSON."
    )


def _offline_answer(message: str, context: dict[str, Any], docs: list[dict[str, str]]) -> str:
    rec = context.get("last_recommendation") or {}
    state = context.get("state") or {}
    timeline = context.get("timeline") or []
    m = message.lower()
    features = rec.get("derived_features") or {}
    if ("explain" in m and "game" in m) or "what is the game" in m:
        return (
            "Game overview:\n\n"
            "- The demo is a boss-match LiveOps simulator. Each preset represents a player state, such as a new player stuck after repeated losses or an advanced player who may need harder content.\n"
            "- The RL policy scores possible interventions before the next match: recovery match, training match, reduce enemy power, grant currency, temporary boost, elite match, or no intervention.\n"
            "- The red safety gate blocks actions that violate policy rules, such as escalating difficulty for a frustrated new player or granting resources when economy conditions are not safe.\n"
            "- After you play a match, the app logs before/after telemetry, updates win probability, frustration, churn-risk proxy, and then recalculates the next recommendation.\n"
            "- The chat uses RAG context from the policy docs plus the current run/preset/results, but the chat explanation layer does not choose the action."
        )
    if "rule" in m and "game" in m:
        return (
            "Game rules:\n\n"
            "- Pick a preset or adjust sliders to define the player, enemy, fatigue, losses, completion, rewards, and training affinity.\n"
            "- The policy recommends one intervention for the next decision point.\n"
            "- Safety rules can block risky interventions before serving. Blocked actions appear in the safety gate panel.\n"
            "- Play Next Match records two timeline points for each match: before and after. The vertical chart marker groups those two points under the same match number.\n"
            "- Benchmark bars compare safety-gated RL against four baselines: do nothing, random actions, hand-coded rules, and raw RL before guardrails."
        )
    if any(term in m for term in ["rl model", "model works", "q-learning", "q learning", "learned policy", "how does the policy work"]):
        action = rec.get("recommended_action") or rec.get("served_action") or "unknown"
        blocked = rec.get("blocked_actions") or []
        raw_scores = rec.get("raw_action_scores") or rec.get("arena_action_scores") or {}
        top_raw = max(raw_scores, key=raw_scores.get) if raw_scores else action
        scenario = state.get("scenario_name") or context.get("current_scenario") or "current scenario"
        current_run = (
            f"Current simulator run:\n"
            f"- For `{scenario}`, the RL Q-policy top action was `{top_raw}`.\n"
            f"- After the safety gate, the served action is `{action}`.\n"
            f"- Blocked actions in the current context: {', '.join(str(b.get('action')) for b in blocked[:4]) if blocked else 'none reported'}."
        )
        return (
            "RL policy:\n"
            "- The RL policy is a Q-learning agent that scores candidate LiveOps interventions.\n"
            "- It recommends one of seven actions: do_nothing, recommend_training_match, grant_upgrade_currency, offer_temporary_power_boost, reduce_enemy_power, recommend_recovery_match, or unlock_elite_match.\n"
            "- The LLM does not choose actions, alter the policy, or write training data.\n\n"
            "Arena game context:\n"
            "- The RL policy operates within a boss-match LiveOps simulator.\n"
            "- It considers player telemetry like power, fatigue, skill, losses, attempts, recent rewards, and training affinity.\n"
            "- The simulator derives features like win probability, frustration, economy pressure, reward saturation, and churn-risk proxy from this telemetry.\n\n"
            "Safety gate:\n"
            "- After the RL policy scores actions, a deterministic safety gate applies hard guardrails.\n"
            "- It blocks or redirects risky actions before they are served.\n"
            "- For example, it prevents escalating difficulty for frustrated players or granting resources when reward saturation is high.\n\n"
            "OPE/Evaluation:\n"
            "- The policy is evaluated against do-nothing, random, rule-based, raw RL, and safety-gated RL baselines.\n"
            "- Offline Policy Evaluation estimates policy value from logged behavior data.\n"
            "- Key metrics include win rate, reward, recovery, economy cost, churn-risk proxy, and policy violations.\n\n"
            "Rollout audit:\n"
            "- Run events from the demo, including recommendations and match results, are logged as append-only JSONL records.\n"
            "- This supports auditability without letting the chat layer change decisions.\n\n"
            f"{current_run}"
        )
    if "safety" in m and ("gate" in m or "mask" in m or "blocked" in m):
        return (
            "Safety gate overview:\n\n"
            "- The RL policy first scores actions, then deterministic rules block unsafe or economy-risky choices before serving.\n"
            "- Typical blocks include elite difficulty for cold-start or frustrated players, resource grants under high reward saturation, and boosts when recent reward exposure is already high.\n"
            "- Blocked actions are audit evidence, not model failures; they show the guardrail layer constraining the learned policy."
        )
    if "benchmark" in m or "baseline" in m:
        return (
            "Benchmark overview:\n\n"
            "- The benchmark replays the same scenario against do-nothing, random, rule-based, raw RL, and safety-gated RL policies.\n"
            "- Raw RL shows the learned scores before guardrails. Safety-gated RL is the deployable candidate after deterministic blocks.\n"
            "- Compare win rate, average reward, frustration, churn-risk proxy, intervention cost, and policy violations.\n"
            "- The endpoint is capped for demo safety, so it gives quick directional evidence rather than a full production validation run."
        )
    if "ope" in m or "offline policy evaluation" in m or "effective sample" in m or "match rate" in m:
        return (
            "OPE overview:\n\n"
            "- Offline Policy Evaluation estimates how the candidate policy would have performed using logged behavior-policy data.\n"
            "- Match rate is the share of logged rows where the logged action matches the candidate policy action; only those rows provide direct importance-weighted evidence.\n"
            "- IPS uses propensity-corrected matched rewards. SNIPS normalizes those weights to reduce sensitivity to scale. Clipped IPS caps very large weights to reduce variance.\n"
            "- Effective sample size summarizes how much useful weighted evidence remains after propensity weighting. Low ESS means the estimate should be treated as directional.\n"
            "- OPE supports the rollout decision, but it does not override the deterministic safety gate."
        )
    if "data" in m or "saved" in m or "column" in m:
        return "Battle data is saved as run telemetry, not as new dataset columns.\n\n- Local mode: arena telemetry is written to `artifacts/local_logs.jsonl`.\n- Cloud mode: arena telemetry is written to the BigQuery `arena_match_telemetry` table, with JSONL fallback if BigQuery is unavailable.\n- Seed dataset: `data/arena_liveops_episodes.csv` can be uploaded to BigQuery with `/sync_arena_csv_to_bigquery`; the app reads that BigQuery table in cloud mode.\n- The CSV is not appended during live matches; match results are event telemetry for evaluation/export, not direct training-row mutation."
    if "scenario" in m or "preset" in m or "pick" in m:
        return "You can pick one of seven preset arena situations:\n\n- `cold_start`: almost no history; safest onboarding actions matter.\n- `new_player_stuck`: repeated losses and low win chance; recovery or difficulty relief may be needed.\n- `underpowered_engaged`: engaged but weaker than the enemy; progression help is plausible.\n- `fatigued_player`: fatigue is driving risk; recovery actions should score well.\n- `advanced_bored`: strong advanced player; harder content may be safe.\n- `high_reward_saturation`: rewards are already high; economy guardrails should block more grants.\n- `near_win_repeated_failure`: close losses; the policy weighs small help against cost."
    if any(term in m for term in ["churn", "current scenario", "player", "run", "what next", "next", "result"]):
        action = rec.get("recommended_action", "unknown")
        blocked = rec.get("blocked_actions") or []
        churn = features.get("churn_risk", state.get("churn_risk", "unknown"))
        frustration = features.get("frustration_score", state.get("frustration_score", "unknown"))
        win = features.get("win_probability", state.get("win_probability", "unknown"))
        last = timeline[-1] if timeline else None
        lines = [
            f"Current run: `{state.get('scenario_name', 'unknown')}`",
            "",
            f"- Served action: `{action}`",
            f"- Churn-risk proxy: `{churn}`",
            f"- Frustration: `{frustration}`",
            f"- Win probability: `{win}`",
        ]
        if last:
            outcome = ("won" if last.get("won") else "lost") if last.get("won") is not None else "state"
            lines.append(f"- Latest match: `{outcome}` at `{last.get('completion')}` completion after `{last.get('action')}`")
        if blocked:
            lines.append("")
            lines.append("Safety gate blocked:")
            lines.extend(f"- `{b.get('action')}`: {b.get('reason')}" for b in blocked[:3])
        lines.append("")
        lines.append("This is demo telemetry, not a production next-week churn forecast.")
        return "\n".join(lines)
    return "I can explain the current run, available scenarios, safety-gate blocks, benchmark baselines, OPE evidence, and where run telemetry is saved."


def _message_text(result: Any) -> str:
    content = getattr(result, "content", result)
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or item))
            else:
                parts.append(str(item))
        return "\n".join(parts).strip()
    return str(content).strip()


def _normalize_provider(provider: str | None) -> str:
    normalized = (provider or settings.CHAT_PROVIDER or "offline").strip().lower()
    if normalized in {"cloud", "vertex"}:
        return "gemini"
    if normalized in {"offline", "ollama", "gemini"}:
        return normalized
    return "offline"


def _gemini_model():
    from langchain_google_genai import ChatGoogleGenerativeAI

    kwargs: dict[str, Any] = {"model": settings.GEMINI_MODEL_FAST, "temperature": 0.2, "request_timeout": 20}
    if settings.GEMINI_PROVIDER == "vertex":
        kwargs.update({"vertexai": True, "project": settings.GCP_PROJECT, "location": settings.GEMINI_LOCATION})
    else:
        kwargs["api_key"] = settings.GEMINI_API_KEY
    return ChatGoogleGenerativeAI(**kwargs)


def _ollama_model():
    from langchain_ollama import ChatOllama

    kwargs: dict[str, Any] = {
        "model": _selected_ollama_model(),
        "base_url": settings.OLLAMA_BASE_URL,
        "temperature": 0.2,
        "num_ctx": settings.OLLAMA_NUM_CTX,
        "num_predict": settings.OLLAMA_NUM_PREDICT,
        "keep_alive": settings.OLLAMA_KEEP_ALIVE,
        "sync_client_kwargs": {"timeout": settings.OLLAMA_TIMEOUT_SECONDS},
    }
    if settings.OLLAMA_NUM_GPU is not None:
        kwargs["num_gpu"] = settings.OLLAMA_NUM_GPU
    return ChatOllama(**kwargs)


def _langchain_model(provider: str, message: str, context: dict[str, Any], docs: list[dict[str, str]]):
    if provider == "gemini":
        return _gemini_model()
    if provider == "ollama":
        return _ollama_model()
    from langchain_core.runnables import RunnableLambda

    return RunnableLambda(lambda _: _offline_answer(message, context, docs))


def _gemini_answer(prompt: str) -> str:
    return _message_text(_gemini_model().invoke(prompt))



def ollama_status() -> dict[str, Any]:
    status = {"available": False, "model": settings.OLLAMA_MODEL, "selected_model": settings.OLLAMA_MODEL, "timeout_seconds": settings.OLLAMA_TIMEOUT_SECONDS, "num_gpu": settings.OLLAMA_NUM_GPU, "num_ctx": settings.OLLAMA_NUM_CTX, "num_predict": settings.OLLAMA_NUM_PREDICT, "keep_alive": settings.OLLAMA_KEEP_ALIVE, "fast_known_answers": settings.OLLAMA_FAST_KNOWN_ANSWERS, "installed_models": [], "message": "Ollama is local-only and requires a pulled model."}
    try:
        response = requests.get(f"{settings.OLLAMA_BASE_URL}/api/tags", timeout=0.8)
        response.raise_for_status()
        names = [m.get("name") for m in response.json().get("models", []) if m.get("name")]
        selected = settings.OLLAMA_MODEL if settings.OLLAMA_MODEL in names else (names[0] if names else settings.OLLAMA_MODEL)
        status.update({
            "available": bool(names),
            "selected_model": selected,
            "installed_models": names[:12],
            "message": "Ready." if settings.OLLAMA_MODEL in names else (f"Using installed model `{selected}`. Configure OLLAMA_MODEL or run: ollama pull {settings.OLLAMA_MODEL}" if names else f"No local models found. Run: ollama pull {settings.OLLAMA_MODEL}"),
        })
    except Exception as exc:
        status["message"] = f"Ollama unavailable: {exc}"
    return status


def _selected_ollama_model() -> str:
    status = ollama_status()
    if not status.get("available"):
        raise RuntimeError(status.get("message") or "Ollama unavailable.")
    return str(status.get("selected_model") or settings.OLLAMA_MODEL)

def _ollama_answer(prompt: str) -> str:
    return _message_text(_ollama_model().invoke(prompt))


def _append_memory(session_id: str, message: str, response: str) -> None:
    _MEMORY[session_id].append({"role": "user", "content": message[:800]})
    _MEMORY[session_id].append({"role": "assistant", "content": response[:1200]})


def _fallback_error(provider: str, exc: Exception) -> str:
    if provider == "gemini":
        return gemini_error_message(exc)
    text = str(exc).lower()
    if provider == "ollama" and ("cuda" in text or "unsupported toolchain" in text or "0xc0000409" in text):
        return "Ollama is reachable, but the local model runner crashed with a CUDA/toolchain compatibility error. Offline fallback used."
    if provider == "ollama" and "timed out" in text:
        return f"Ollama is reachable, but the model did not answer within {settings.OLLAMA_TIMEOUT_SECONDS:g} seconds. Offline fallback used."
    return f"{provider} chat failed; offline fallback used: {exc}"


def answer_agent_message(message: str, context: dict[str, Any], provider: str | None = None, session_id: str | None = None) -> dict[str, Any]:
    provider = _normalize_provider(provider)
    session_id = session_id or str(context.get("session_id") or "default")
    docs = _retrieve(message)
    memory = list(_MEMORY[session_id])
    prompt = _ollama_prompt(message, context, docs, memory) if provider == "ollama" else _prompt(message, context, docs, memory)

    error = None
    used_provider = provider
    try:
        if provider == "ollama" and settings.OLLAMA_FAST_KNOWN_ANSWERS:
            response = _offline_answer(message, context, docs)
            if not response.startswith("I can explain the current run"):
                used_provider = "offline_fast_path"
                raise StopIteration
        response = _message_text(_langchain_model(provider, message, context, docs).invoke(prompt))
    except StopIteration:
        pass
    except Exception as exc:
        error = _fallback_error(provider, exc)
        used_provider = "offline"
        response = _offline_answer(message, context, docs)
    _append_memory(session_id, message, response)
    result = {
        "provider": used_provider,
        "use_gemini": used_provider == "gemini",
        "used_rag": True,
        "used_langchain": True,
        "model_router": "langchain",
        "memory_turns": len(_MEMORY[session_id]),
        "sources": [d["source"] for d in docs],
        "response": response,
    }
    if error:
        result["provider_error"] = error
        if provider == "gemini":
            result["gemini_error"] = error
    return result


def _word_chunks(text: str) -> Iterator[str]:
    for chunk in re.findall(r"\S+\s*|\n+", text):
        yield chunk


def stream_agent_message(message: str, context: dict[str, Any], provider: str | None = None, session_id: str | None = None) -> Iterator[dict[str, Any]]:
    provider = _normalize_provider(provider)
    session_id = session_id or str(context.get("session_id") or "default")
    docs = _retrieve(message)
    memory = list(_MEMORY[session_id])
    prompt = _ollama_prompt(message, context, docs, memory) if provider == "ollama" else _prompt(message, context, docs, memory)

    used_provider = provider
    error = None
    chunks: list[str] = []
    try:
        if provider == "ollama" and settings.OLLAMA_FAST_KNOWN_ANSWERS:
            response = _offline_answer(message, context, docs)
            if not response.startswith("I can explain the current run"):
                used_provider = "offline_fast_path"
                for chunk in _word_chunks(response):
                    chunks.append(chunk)
                    yield {"type": "token", "text": chunk}
                raise StopIteration
        model = _langchain_model(provider, message, context, docs)
        if provider == "offline":
            response = _message_text(model.invoke(prompt))
            for chunk in _word_chunks(response):
                chunks.append(chunk)
                yield {"type": "token", "text": chunk}
        else:
            for item in model.stream(prompt):
                chunk = _message_text(item)
                if not chunk:
                    continue
                chunks.append(chunk)
                yield {"type": "token", "text": chunk}
            response = "".join(chunks).strip()
    except StopIteration:
        pass
    except Exception as exc:
        error = _fallback_error(provider, exc)
        used_provider = "offline"
        yield {"type": "error", "text": error, "provider_error": error}
        response = _offline_answer(message, context, docs)
        chunks = []
        for chunk in _word_chunks(response):
            chunks.append(chunk)
            yield {"type": "token", "text": chunk}

    response = "".join(chunks).strip() if chunks else response.strip()
    _append_memory(session_id, message, response)
    done = {
        "type": "done",
        "provider": used_provider,
        "use_gemini": used_provider == "gemini",
        "used_rag": True,
        "used_langchain": True,
        "model_router": "langchain",
        "memory_turns": len(_MEMORY[session_id]),
        "sources": [d["source"] for d in docs],
        "response": response,
    }
    if error:
        done["provider_error"] = error
        if provider == "gemini":
            done["gemini_error"] = error
    yield done






















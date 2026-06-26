from __future__ import annotations

import json
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

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


def _gemini_answer(prompt: str) -> str:
    from langchain_google_genai import ChatGoogleGenerativeAI

    kwargs: dict[str, Any] = {"model": settings.GEMINI_MODEL_FAST, "temperature": 0.2, "request_timeout": 20}
    if settings.GEMINI_PROVIDER == "vertex":
        kwargs.update({"vertexai": True, "project": settings.GCP_PROJECT, "location": settings.GEMINI_LOCATION})
    else:
        kwargs["api_key"] = settings.GEMINI_API_KEY
    model = ChatGoogleGenerativeAI(**kwargs)
    return _message_text(model.invoke(prompt))



def ollama_status() -> dict[str, Any]:
    status = {"available": False, "model": settings.OLLAMA_MODEL, "selected_model": settings.OLLAMA_MODEL, "installed_models": [], "message": "Ollama is local-only and requires a pulled model."}
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
    from langchain_ollama import ChatOllama

    model = ChatOllama(
        model=_selected_ollama_model(),
        base_url=settings.OLLAMA_BASE_URL,
        temperature=0.2,
        sync_client_kwargs={"timeout": 8},
    )
    return _message_text(model.invoke(prompt))

def answer_agent_message(message: str, context: dict[str, Any], provider: str | None = None, session_id: str | None = None) -> dict[str, Any]:
    provider = (provider or settings.CHAT_PROVIDER or "offline").strip().lower()
    if provider == "cloud":
        provider = "gemini"
    if provider not in {"offline", "ollama", "gemini"}:
        provider = "offline"
    session_id = session_id or str(context.get("session_id") or "default")
    docs = _retrieve(message)
    memory = list(_MEMORY[session_id])
    prompt = _prompt(message, context, docs, memory)

    error = None
    used_provider = provider
    try:
        if provider == "gemini":
            response = _gemini_answer(prompt)
        elif provider == "ollama":
            response = _ollama_answer(prompt)
        else:
            from langchain_core.runnables import RunnableLambda

            chain = RunnableLambda(lambda _: _offline_answer(message, context, docs))
            response = str(chain.invoke(prompt)).strip()
    except Exception as exc:
        error = gemini_error_message(exc) if provider == "gemini" else f"{provider} chat failed; offline fallback used: {exc}"
        used_provider = "offline"
        response = _offline_answer(message, context, docs)
    _MEMORY[session_id].append({"role": "user", "content": message[:800]})
    _MEMORY[session_id].append({"role": "assistant", "content": response[:1200]})
    result = {
        "provider": used_provider,
        "use_gemini": used_provider == "gemini",
        "used_rag": True,
        "used_langchain": True,
        "memory_turns": len(_MEMORY[session_id]),
        "sources": [d["source"] for d in docs],
        "response": response,
    }
    if error:
        result["provider_error"] = error
        if provider == "gemini":
            result["gemini_error"] = error
    return result






















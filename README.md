---
title: "LiveOps RL Decision Engine"
emoji: "🎮"
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---

# LiveOps RL Decision Engine

Interactive LiveOps RL decision engine with safety-gated recommendations, rollout forecasting, benchmark comparisons, and lightweight LangChain RAG policy analysis.

## Hosted Demo

- GitHub: https://github.com/MayCooper/liveops-rl-decision-engine
- Hugging Face Space: https://huggingface.co/spaces/maycooper/liveops-rl-decision-engine
- Live app: https://maycooper-liveops-rl-decision-engine.hf.space

The hosted demo runs in local deterministic mode by default. It does not require GCP, BigQuery, Gemini, Ollama, or paid API credentials to open the simulator.

Hosted chat behavior:

- Offline LangChain RAG fallback is enabled for the public demo.
- Gemini is supported by the code path, but disabled on the public Space by default to avoid API-key cost exposure.
- Ollama is local-only unless a separate remote Ollama server is configured. The public Space does not bundle or run local Ollama models.

## What This Demonstrates

This project is a compact end-to-end LiveOps decisioning demo:

- A tabular Q-learning policy scores possible player interventions.
- A deterministic red safety gate blocks risky or policy-violating actions before serving.
- A browser arena simulates boss-match outcomes, player frustration, churn-risk proxy, and win probability.
- Manual mode lets you inspect one recommendation, apply it, and play the next match.
- Auto rollout mode simulates future match-policy cycles for 1-200 matches and summarizes long runs in paged cards.
- Final rollout cards provide scenario-specific forecasts based on the current preset or manual slider state.
- Benchmark charts compare safety-gated RL against do-nothing, random, rule-based, and raw-RL baselines.
- A lightweight LangChain RAG assistant retrieves policy, game, evaluation, run-state, and memory context; hosted mode uses the offline fallback unless Gemini secrets are explicitly configured.

## Current UI

The main UI is served at `/` and combines the simulator with the operations console. It includes:

- preset scenarios and manual sliders
- live win/frustration/churn metrics
- action-score bars and expected effect cards
- deterministic safety-gate blocks
- cinematic battle replay
- actual match telemetry charts
- benchmark comparison charts
- auto-rollout cards with previous/next pagination after five cards
- final forecast cards for the selected scenario or manual state
- policy, audit, OPE, dataset, and recent-log panels
- agent chat backed by lightweight LangChain RAG retrieval, session memory, and offline/Gemini/Ollama provider fallback

## RAG and LangChain Scope

The chat assistant is RAG-backed, but intentionally lightweight rather than production vector search.

It retrieves from:

- `docs/rag/*.md`
- `data/arena_dataset_profile.json`
- `data/frustration_feature_dictionary.json`
- `artifacts/policy_metrics.json`
- recent local run telemetry when available
- current browser run context passed into `/agent_message`
- short session memory

It uses LangChain integrations for:

- Offline hosted fallback: `langchain_core.runnables.RunnableLambda`
- Gemini, optional and disabled by default on the public demo: `langchain_google_genai.ChatGoogleGenerativeAI`
- Ollama, local-only unless a separate remote Ollama server is configured: `langchain_ollama.ChatOllama`

The public Hugging Face demo intentionally defaults to offline RAG responses so it can be shared without exposing a paid Gemini key. Gemini can be enabled with Space secrets for private demos. Ollama should be treated as a local development option, not a free hosted Space dependency.

The chat explains decisions and policy context. It does not choose player actions, update the Q-policy, bypass the safety gate, or rewrite training data.

## Runtime Modes

Default hosted/local mode:

```text
RUNTIME_MODE=local
DATA_SOURCE=repo
USE_BIGQUERY=false
ENABLE_GEMINI=false
```

In this mode the app uses repo-bundled data and artifacts:

- `artifacts/q_policy.json`
- `artifacts/policy_metrics.json`
- `data/arena_liveops_episodes.csv`
- `data/arena_dataset_profile.json`
- `data/frustration_feature_dictionary.json`

Optional cloud mode can use BigQuery and Gemini if environment variables and credentials are configured. The hosted public demo does not enable Gemini by default because public API keys can incur cost. Do not commit service-account JSON, API keys, tokens, or `.env` files.

## Local Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts\train_and_eval.py
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000/
```

API docs:

```text
http://127.0.0.1:8000/docs
```

## Docker

Build and run locally:

```bash
docker build -t liveops-rl-decision-engine .
docker run --rm -p 7860:7860 liveops-rl-decision-engine
```

Open:

```text
http://127.0.0.1:7860/
```

The Docker image defaults to local deterministic mode:

```text
RUNTIME_MODE=local
DATA_SOURCE=repo
USE_BIGQUERY=false
ENABLE_GEMINI=false
PORT=7860
```

## Hugging Face Spaces Deployment

This repo is prepared for a free Hugging Face Docker Space:

- `README.md` includes `sdk: docker` and `app_port: 7860`.
- `Dockerfile` runs `uvicorn app.main:app` on `${PORT:-7860}`.
- `.dockerignore` excludes local secrets, logs, caches, and virtual environments.

To deploy manually:

1. Create a new Hugging Face Space named `liveops-rl-decision-engine`.
2. Choose **Docker** as the Space SDK.
3. Push this repository to the Space Git remote:

```bash
git remote add space https://huggingface.co/spaces/maycooper/liveops-rl-decision-engine
git push space main
```

If cloud features are needed later, configure them as Hugging Face Space secrets or variables, not committed files.

## Cloud Path, Optional

Cloud path mode is opt-in:

```text
RUNTIME_MODE=cloud
DATA_SOURCE=bigquery
USE_BIGQUERY=true
ENABLE_GEMINI=false
GCP_PROJECT=your-gcp-project-id
BQ_DATASET=your_bigquery_dataset
REGION=us-central1
```

Gemini can be enabled with `ENABLE_GEMINI=true` and a private `GEMINI_API_KEY` or `GOOGLE_API_KEY`. Gemini explains results only; it does not serve actions or alter policy decisions.

## Data and Logging

Local mode reads bundled repo artifacts and data. Live UI telemetry is written to local JSONL logs when running locally, but those logs are ignored by Git.

Cloud mode can read/write BigQuery tables when configured. Live match telemetry is event telemetry, not automatic mutation of the training CSV.

## Project Structure

```text
app/
  main.py              FastAPI app and public API routes
  arena.py             arena simulator, recommendations, rollout, benchmark routes
  core.py              LiveOps state and simulation primitives
  policies.py          rule, bandit, Q-learning, and optional DQN policy code
  rag_chat.py          lightweight LangChain RAG assistant
  cloud_io.py          local/BigQuery logging and data access
  static/              browser UI
artifacts/
  q_policy.json        bundled learned tabular Q-policy
  policy_metrics.json  bundled evaluation metrics
data/
  arena_liveops_episodes.csv
  arena_dataset_profile.json
  frustration_feature_dictionary.json
docs/rag/
  policy/game/evaluation context used by the assistant
scripts/
  training, dataset generation, and demo request helpers
```

## Safety Notes

- The Q-policy scores candidate actions.
- The deterministic safety gate constrains what can be served.
- The LLM/RAG layer explains decisions only.
- Churn-risk values are demo proxies derived from telemetry, not production churn forecasts.
- Rollout forecasting is inference-time simulation with a fixed policy, not retraining.
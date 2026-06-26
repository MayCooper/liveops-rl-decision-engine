---
title: LiveOps Policy Lab
emoji: game_die
colorFrom: blue
colorTo: cyan
sdk: docker
app_port: 7860
---

# LiveOps Policy Lab

This project trains a simple RL-style policy for game LiveOps decisions and provides a Gemini-compatible policy audit workflow with deterministic fallback.

It is intentionally compact: a synthetic multi-day player simulator, a rule baseline, a one-step bandit baseline, a tabular Q-learning policy, a FastAPI serving layer, local/BigQuery logging, and an audit workflow that stress-tests rollout risk but never chooses player actions.


## Current UI and Runtime Notes

The main UI is now one continuous page at `/`. It combines the Policy Replay Simulator with the operations console. The operations tools use a fixed left-button/right-results layout so Health Check, Policy Metrics, Safety / Policy Audit, OPE Metrics, Recent Logs, Dataset Profile, and Synthetic Episode Sample update one stable results panel instead of shifting the page.

The app exposes two honest runtime paths:

- `local`: local CSV/JSON data, local `artifacts/q_policy.json`, local JSONL logs, and deterministic explanation fallback.
- `cloud`: BigQuery/Gemini features only when the required environment variables and credentials are configured. The RL policy is still loaded from the local artifact unless a real Vertex AI RL serving endpoint is explicitly added.

The top status cards are populated from `/health`, not hard-coded UI assumptions. If cloud mode is requested without configuration, the backend keeps the app in local mode and returns a warning.

## Architecture

```text
Synthetic player simulator
        |
        v
Rule baseline -> Bandit baseline
        |              |
        +------ compare +------ Tabular Q-learning policy
                              |
                              v
                         FastAPI web app
                              |
           +------------------+------------------+
           v                                     v
 BigQuery or local JSONL logs        Policy audit workflow
                                      stress scenarios -> policy recommendations
                                      -> deterministic safety gate -> report
```


## Project Structure

```text
app/
  __init__.py
  main.py
  core.py
  policies.py
  agents.py
  cloud_io.py
  static/
    index.html
scripts/
  train_and_eval.py
  demo_requests.py
deploy/
  setup_bigquery.sql
  deploy_cloud_run.sh
artifacts/
  .gitkeep
```
## Local Setup

```bash
cd liveops-policy-lab
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/train_and_eval.py
uvicorn app.main:app --reload
python scripts/demo_requests.py
```

Windows activation:

```powershell
cd liveops-policy-lab
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts\train_and_eval.py
uvicorn app.main:app --reload
python scripts\demo_requests.py
```

Local mode works without GCP or Gemini credentials. Artifacts are written to `artifacts/q_policy.json` and `artifacts/policy_metrics.json`; logs go to `artifacts/local_logs.jsonl`.


## Runtime Modes

The default deterministic fallback mode uses the dataset and artifacts included in the repository, disables BigQuery, disables Gemini, and requires no cloud credentials.

Deterministic fallback mode:

```text
RUNTIME_MODE=local
DATA_SOURCE=repo
USE_BIGQUERY=false
ENABLE_GEMINI=false
```

In this mode, users open the hosted URL and use the UI without downloading the repo, authenticating to GCP, or providing a Gemini key. The app reads `data/synthetic_liveops_episodes.csv`, `artifacts/policy_metrics.json`, and `artifacts/q_policy.json`. Recommendation, safety mask, OPE, visual replay, and audit all run from bundled repo data and deterministic fallback.

Cloud path mode is opt-in. Set `RUNTIME_MODE=cloud`, `DATA_SOURCE=bigquery`, `USE_BIGQUERY=true`, `GCP_PROJECT`, and `BQ_DATASET` to use BigQuery. Set `ENABLE_GEMINI=true` and `GEMINI_API_KEY` or `GOOGLE_API_KEY` to enable Gemini explanations.

```text
RUNTIME_MODE=cloud
DATA_SOURCE=bigquery
USE_BIGQUERY=true
ENABLE_GEMINI=false
GCP_PROJECT=your-gcp-project-id
BQ_DATASET=your_bigquery_dataset
REGION=us-central1
```

The user-facing modes are deterministic fallback and cloud path. Deterministic fallback uses repo files. Cloud path is the optional integration path; cloud features do not activate from project variables alone.

Gemini may explain results only. Gemini does not choose player actions, change the Q-learning policy, override the deterministic safety mask, alter OPE metrics, or alter rollout decision fields.

Use `.env.example` as a template only. Your local `.env` is private, must not be committed, and should contain only environment-specific values. Do not store service account JSON files, access tokens, API keys, or other secrets in this repo.

For local development, copy the template and edit the private file:

```powershell
Copy-Item .env.example .env
notepad .env
```

The app loads `.env` automatically at startup. `.env` is ignored by git.

## Configuration and Credentials

Local deterministic fallback mode works without GCP credentials. For local GCP testing only, authenticate with Google Application Default Credentials:

```bash
gcloud auth application-default login
```

BigQuery logging and reads use Google Application Default Credentials locally, or the Cloud Run service account when deployed. For Cloud Run, attach a service account with the needed IAM permissions instead of committing credentials. Expected GCP services are Cloud Run, Cloud Build, Artifact Registry, and BigQuery. Secret Manager is optional for managing `GEMINI_API_KEY`; Vertex AI or the Gemini API is optional if you enable Gemini-backed explanation text.

## Public Demo Without GCP

The app can be deployed to Cloud Run or another container-friendly host as a public deterministic fallback demo without using GCP data services. Use the existing Dockerfile or run `uvicorn app.main:app --host 0.0.0.0 --port $PORT`. In deterministic fallback mode, the app uses local repo files inside the deployed container.

Public deterministic fallback demo on Cloud Run:

```powershell
gcloud run deploy liveops-rl-decision-engine --source . --region us-central1 --allow-unauthenticated --set-env-vars RUNTIME_MODE=local,DATA_SOURCE=repo,USE_BIGQUERY=false,ENABLE_GEMINI=false
```

This can use Cloud Run for hosting, but it does not use BigQuery or Gemini. For a non-GCP host, set the same environment variables in that host's dashboard.

## Cloud Auto Deployment

Cloud path deployment with BigQuery enabled and Gemini disabled:

```powershell
gcloud run deploy liveops-rl-decision-engine --source . --region us-central1 --allow-unauthenticated --service-account liveops-rl-policy-sa@liveops-rl-decision-engine.iam.gserviceaccount.com --set-env-vars RUNTIME_MODE=cloud,DATA_SOURCE=bigquery,USE_BIGQUERY=true,ENABLE_GEMINI=false,GCP_PROJECT=liveops-rl-decision-engine,BQ_DATASET=liveops_rl_decision_engine,REGION=us-central1
```

Do not put Gemini API keys directly in deployment commands. Use Secret Manager or private local environment variables.

Secret Manager example:

```powershell
echo PASTE_PRIVATE_KEY_HERE | gcloud secrets create gemini-api-key --data-file=-

gcloud secrets add-iam-policy-binding gemini-api-key `
  --member="serviceAccount:liveops-rl-policy-sa@liveops-rl-decision-engine.iam.gserviceaccount.com" `
  --role="roles/secretmanager.secretAccessor"

gcloud run deploy liveops-rl-decision-engine `
  --source . `
  --region us-central1 `
  --allow-unauthenticated `
  --service-account liveops-rl-policy-sa@liveops-rl-decision-engine.iam.gserviceaccount.com `
  --set-env-vars RUNTIME_MODE=cloud,DATA_SOURCE=bigquery,USE_BIGQUERY=true,ENABLE_GEMINI=true,GCP_PROJECT=liveops-rl-decision-engine,BQ_DATASET=liveops_rl_decision_engine,REGION=us-central1 `
  --set-secrets GEMINI_API_KEY=gemini-api-key:latest
```

## Robust Arena Dataset

The repo includes `data/arena_liveops_episodes.csv`, `data/arena_dataset_profile.json`, and `data/arena_liveops_episodes_sample.ndjson`. The arena dataset contains observable gameplay telemetry such as time spent, challenge retries, failed challenge counts, optional challenge usage, quit-after-failure, booster/revive usage, asset losses, reward claims, challenge switching, help-screen opens, idle time after loss, and near-miss counts.

`frustration_score` is derived from those telemetry fields. It is not treated as a direct user-provided value. This makes the demo more defensible as a practical ML engineering system.

## Browser Demo

Start the API locally:

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/`.

The main browser UI now opens directly into the Arena RL simulator. It includes player/enemy sliders, preset scenarios below the slider section, manual controls, live metric cards, action-score cards, deterministic safety-gate cards, benchmark charts, and a cinematic Roman-style battle replay. The old symbolic/static simulation panel has been removed from the main page.

There is one learned RL decision agent. The deterministic safety gate constrains serving, and the explanation/audit layer explains the decision; it does not act as a second decision agent.

The same UI is served at `/` after deployment on any container-friendly host. Interactive API docs remain available at `/docs`.
## Offline Policy Evaluation

The simulator gives direct controlled evaluation, while Offline Policy Evaluation, OPE, gives a logged-data estimate before rollout. The synthetic dataset includes behavior-policy action probabilities, so the project reports IPS, SNIPS, clipped IPS, match rate, and effective sample size.

OPE can have high variance, so it is one signal, not the only rollout decision. The deterministic safety gate remains the hard rollout control.

The browser UI includes an OPE Calculation Trace showing matched logged rows, behavior action probabilities, importance weights, and estimate contributions.
## BigQuery Data Source

`synthetic_liveops_episodes` stores the offline synthetic telemetry dataset. `synthetic_dataset_profile` stores the dataset profile.

In cloud-enabled mode, the app reads the offline synthetic telemetry sample and dataset profile from BigQuery only when `DATA_SOURCE=bigquery` and `USE_BIGQUERY=true`. In demo mode, the app uses files in `data/` and deterministic generated fallback if the CSV is missing.

The serving path remains low-latency: the API loads the trained Q-policy artifact and applies the deterministic safety mask in memory. BigQuery is used for offline telemetry, dataset profile, policy evaluation records, recommendation logs, and audit logs.

Runtime recommendation calls still use the trained Q-policy artifact; BigQuery is not queried for every served recommendation.
## Demo Flow

1. Run `python scripts/train_and_eval.py` and show the printed rule, bandit, and Q-learning rewards.
2. Open `artifacts/policy_metrics.json`.
3. Start `uvicorn app.main:app --reload`.
4. Call `/recommend_action` for a struggling new player.
5. Call `/recommend_action` for a bored advanced player.
6. Call `/run_agent_audit`.
7. Show BigQuery rows when enabled, or local JSONL logs in demo mode.

## GCP Setup

```bash
export PROJECT_ID=your-gcp-project
gcloud config set project ${PROJECT_ID}
gcloud services enable run.googleapis.com bigquery.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com storage.googleapis.com secretmanager.googleapis.com
bq mk --dataset ${PROJECT_ID}:liveops_policy_lab
sed "s/PROJECT_ID/${PROJECT_ID}/g" deploy/setup_bigquery.sql | bq query --use_legacy_sql=false
```

On Windows, replace `PROJECT_ID` in `deploy/setup_bigquery.sql` manually or use PowerShell replacement before running `bq query`.

Deploy:

```bash
PROJECT_ID=your-gcp-project REGION=us-central1 bash deploy/deploy_cloud_run.sh
```

PowerShell example with explicit region and Cloud Run service account:

```powershell
gcloud run deploy liveops-rl-decision-engine --source . --region us-central1 --allow-unauthenticated --service-account liveops-rl-policy-sa@liveops-rl-decision-engine.iam.gserviceaccount.com --set-env-vars GCP_PROJECT=liveops-rl-decision-engine,BQ_DATASET=liveops_rl_decision_engine,REGION=us-central1
```

Optional Gemini:

```bash
export GEMINI_API_KEY="..."
PROJECT_ID=your-gcp-project bash deploy/deploy_cloud_run.sh
```

## API

- `GET /health`
- `POST /recommend_action`
- `GET /policy_metrics`
- `GET /ope_metrics`
- `GET /dataset_profile`
- `GET /synthetic_episode_sample`
- `GET /recent_recommendations`
- `GET /recent_audits`
- `GET /recent_policy_metrics`
- `POST /simulate_episode`
- `POST /run_agent_audit`
- `GET /recent_local_logs`

Example recommendation request:

```json
{
  "request_id": "demo-new",
  "player": {
    "segment": "new",
    "skill": 0.2,
    "frustration": 0.86,
    "engagement": 0.3,
    "churn_risk": 0.78,
    "economy_balance": 0.42,
    "recent_losses": 5,
    "recent_rewards": 1,
    "day": 4
  }
}
```

`/recommend_action` returns both raw Q-table scores and served scores. The served scores include deterministic safety constraints applied before rollout.

## Design Decisions

- Synthetic environment: keeps the project runnable without private game telemetry while making assumptions visible.
- Tabular Q-learning: demonstrates sequential decision-making without a full RL framework.
- Baseline comparison: the learned policy is judged against a rule policy and a one-step reward model.
- Serving safety mask: the Q-learning table scores candidate LiveOps actions. Before serving, a deterministic safety mask blocks actions that are obviously unsafe or inappropriate for the player state, such as increasing difficulty for highly frustrated new players or repeatedly granting resources when economy risk is high. This mirrors a production pattern where learned policies are constrained by hard rollout rules.
- Agent audit: the current MVP uses deterministic fallback scenarios and can optionally call Gemini for concise explanation text when credentials are present. The audit reviews rollout risk; it does not choose LiveOps actions.
- Deterministic safety gate: rollout approval cannot be overridden by an LLM.

## Limitations

- The simulator is synthetic and simplified.
- Q-learning is tabular, not deep RL.
- The audit workflow is advisory; deterministic gates are the control.
- The current MVP does not implement a LangGraph graph, though it can be extended that way without changing policy-serving responsibilities.
- A production version would use real telemetry, online experiments, stronger off-policy evaluation, Vertex AI Pipelines, and stricter governance.







## Arena RL Simulator update

This version includes an additional arena simulator dashboard:

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8014
```

Open:

```text
http://127.0.0.1:8014/arena/dashboard
```

The simulator uses one learned RL decision agent. The Q-learning policy recommends the LiveOps intervention, while deterministic safety rules constrain serving. The optional audit/explanation layer summarizes decisions but does not choose or override actions.

New files:

```text
app/arena.py
app/static/arena.html
data/arena_liveops_episodes.csv
data/arena_dataset_profile.json
data/arena_liveops_episodes_sample.ndjson
policies/policy_rules.json
policies/liveops_policy.md
policies/economy_policy.md
policies/cold_start_policy.md
docs/ARENA_SIMULATOR.md
docs/PRESENTATION_TALK_TRACK.md
scripts/generate_arena_dataset.py
```

New endpoints:

```text
GET  /arena/dashboard
GET  /arena/presets
GET  /arena/policy_rules
GET  /arena/dataset_profile
POST /arena/recommend
POST /arena/apply_action
POST /arena/play_match
POST /arena/benchmark
```

The UI includes live scenario graphs, replay animation, action scores, safety-gate blocks, and benchmark charts comparing do-nothing, random, rule-based, raw RL, and safety-gated RL policies.

### Arena battle replay

Open `http://127.0.0.1:8014/arena/dashboard` to use the cinematic arena simulator. The replay panel now renders a lightweight mini battle with hero/boss sprites, HP bars, attack effects, damage numbers, critical hits, dodges, boost/recovery/debuff visuals, and replay speed control. The animation is driven by `/arena/play_match` replay events, so visual changes are tied to the simulator state and RL policy response.

## Merged arena + operations UI

The root page `/` now combines the arena battle simulator with the original LiveOps operations console. Use the guided workflow in the left panel:

1. Load a preset or adjust sliders.
2. Recalculate the RL recommendation.
3. Apply the served action after the red safety/risk gate.
4. Play the next match to update telemetry and replay the outcome.
5. Run benchmark to compare baselines.

The lower operations section restores the original health checks, policy metrics, OPE evidence, logs, manual recommendation tools, audit/explanation console, and day-by-day progress cards.

The project uses one RL Decision Agent. The red safety/risk gate is an internal deterministic component of the serving path, not a second agent. The explanation console describes decisions and answers questions but does not choose actions.

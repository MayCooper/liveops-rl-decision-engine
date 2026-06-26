from __future__ import annotations

import csv
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Any

logger = logging.getLogger(__name__)
_BQ_EXECUTOR = ThreadPoolExecutor(max_workers=2)
_BQ_SKIP_UNTIL = 0.0


from app.config import settings

ARENA_EPISODES_TABLE = "arena_liveops_episodes"
ARENA_EVENTS_TABLE = "arena_match_telemetry"


def _bigquery_client():
    from google.cloud import bigquery

    return bigquery.Client(project=settings.GCP_PROJECT)


def _ensure_dataset(client: Any) -> None:
    from google.cloud import bigquery

    dataset_id = f"{settings.GCP_PROJECT}.{settings.BQ_DATASET}"
    dataset = bigquery.Dataset(dataset_id)
    dataset.location = settings.REGION
    client.create_dataset(dataset, exists_ok=True)


def upload_arena_csv_to_bigquery(csv_path: str = "data/arena_liveops_episodes.csv", replace: bool = False) -> dict:
    """Seed the arena CSV into BigQuery for cloud-mode reads.

    If the table already has rows, the default is to skip instead of duplicating data.
    Use replace=True when intentionally refreshing the seed table from the repo CSV.
    """
    if not settings.use_bigquery:
        return {"uploaded": False, "source": "local", "message": "BigQuery is disabled for this runtime."}
    future = _BQ_EXECUTOR.submit(_upload_arena_csv_to_bigquery, csv_path, replace)
    try:
        return future.result(timeout=12)
    except TimeoutError:
        return {"uploaded": False, "source": csv_path, "message": "BigQuery CSV upload timed out."}
    except Exception as exc:
        return {"uploaded": False, "source": csv_path, "error": str(exc)}


def _upload_arena_csv_to_bigquery(csv_path: str, replace: bool) -> dict:
    path = Path(csv_path)
    if not path.exists():
        return {"uploaded": False, "source": str(path), "message": "CSV file not found."}
    from google.cloud import bigquery

    client = _bigquery_client()
    _ensure_dataset(client)
    table_id = _table_id(ARENA_EPISODES_TABLE)
    try:
        existing = client.get_table(table_id)
        if existing.num_rows and not replace:
            return {"uploaded": False, "table": table_id, "rows": existing.num_rows, "message": "BigQuery seed table already has rows."}
    except Exception:
        pass
    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.CSV,
        skip_leading_rows=1,
        autodetect=True,
        write_disposition="WRITE_TRUNCATE" if replace else "WRITE_APPEND",
    )
    with path.open("rb") as f:
        job = client.load_table_from_file(f, table_id, job_config=job_config)
    job.result(timeout=10)
    table = client.get_table(table_id)
    return {"uploaded": True, "table": table_id, "rows": table.num_rows, "source": str(path)}


def read_arena_episodes_dataframe(limit: int | None = None):
    import pandas as pd

    if settings.use_bigquery:
        try:
            table = _table_id(ARENA_EPISODES_TABLE)
            suffix = f" LIMIT {int(limit)}" if limit else ""
            sql = f"SELECT * FROM `{table}`{suffix}"
            return pd.DataFrame(_query_rows(sql)), "bigquery"
        except Exception as exc:
            logger.warning("BigQuery arena episodes read failed, falling back to local CSV: %s", exc)
    path = Path("data/arena_liveops_episodes.csv")
    if path.exists():
        df = pd.read_csv(path)
        return (df.head(limit) if limit else df), "repo_arena_csv"
    return pd.DataFrame(), "missing"


def log_arena_event(record: dict) -> bool:
    row = {
        "event_type": record.get("event_type") or record.get("event") or "arena_event",
        "request_id": record.get("request_id"),
        "payload_json": json.dumps(_json_safe(record)),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    return maybe_write_bigquery(ARENA_EVENTS_TABLE, [row])


def read_recent_arena_events(limit: int = 20) -> dict:
    limit = _bounded_limit(limit, default=20, maximum=100)
    if settings.use_bigquery:
        return _read_recent_table(
            ARENA_EVENTS_TABLE,
            "SELECT created_at, event_type, request_id, payload_json FROM `{table}` ORDER BY created_at DESC LIMIT {limit}",
            limit,
        )
    path = Path(settings.LOCAL_LOG_PATH)
    if not path.exists():
        return {"configured": False, "table": ARENA_EVENTS_TABLE, "rows": [], "source": "local_jsonl"}
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines()[-limit * 4:]:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("event_type", "").startswith("arena_"):
            rows.append(row)
    return {"configured": False, "table": ARENA_EVENTS_TABLE, "rows": rows[-limit:], "source": "local_jsonl"}

def _json_safe(record: dict) -> dict:
    return json.loads(json.dumps(record, default=str))


def log_jsonl(record: dict, path: str | None = None) -> bool:
    path = path or settings.LOCAL_LOG_PATH
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    enriched = {"created_at": datetime.now(timezone.utc).isoformat(), **_json_safe(record)}
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(enriched) + "\n")
    return True


def maybe_write_bigquery(table_name: str, rows: Iterable[dict]) -> bool:
    global _BQ_SKIP_UNTIL
    rows = list(rows)
    if not settings.use_bigquery or time.monotonic() < _BQ_SKIP_UNTIL:
        for row in rows:
            log_jsonl({"table": table_name, **row})
        return False
    future = _BQ_EXECUTOR.submit(_write_bigquery_rows, table_name, rows)
    try:
        return future.result(timeout=1.5)
    except TimeoutError:
        _BQ_SKIP_UNTIL = time.monotonic() + 60
        logger.warning("BigQuery logging timed out, falling back to JSONL.")
    except Exception as exc:
        _BQ_SKIP_UNTIL = time.monotonic() + 60
        logger.warning("BigQuery logging failed, falling back to JSONL: %s", exc)
    for row in rows:
        log_jsonl({"table": table_name, **row})
    return False


def _write_bigquery_rows(table_name: str, rows: list[dict]) -> bool:
    client = _bigquery_client()
    _ensure_dataset(client)
    table_id = f"{settings.GCP_PROJECT}.{settings.BQ_DATASET}.{table_name}"
    errors = client.insert_rows_json(table_id, [_json_safe(r) for r in rows], timeout=2)
    if errors:
        raise RuntimeError(errors)
    return True


def log_recommendation(record: dict) -> bool:
    return maybe_write_bigquery("recommendation_logs", [{"request_json": json.dumps(_json_safe(record)), "created_at": datetime.now(timezone.utc).isoformat()}])


def log_eval_results(record: dict) -> bool:
    return maybe_write_bigquery("policy_eval_results", [{"metrics_json": json.dumps(_json_safe(record)), "created_at": datetime.now(timezone.utc).isoformat()}])


def log_agent_audit(record: dict) -> bool:
    return maybe_write_bigquery("agent_audit_logs", [{"report_json": json.dumps(_json_safe(record)), "created_at": datetime.now(timezone.utc).isoformat()}])
def _bounded_limit(limit: int, default: int = 5, maximum: int = 20) -> int:
    try:
        value = int(limit)
    except (TypeError, ValueError):
        value = default
    return max(1, min(value, maximum))


def _table_id(table_name: str) -> str:
    return f"{settings.GCP_PROJECT}.{settings.BQ_DATASET}.{table_name}"


def _query_rows(sql: str) -> list[dict]:
    def _run_query() -> list[dict]:
        client = _bigquery_client()
        return [_json_safe(dict(row)) for row in client.query(sql).result(timeout=3)]

    future = _BQ_EXECUTOR.submit(_run_query)
    return future.result(timeout=4)

def _read_local_profile(error: str | None = None) -> dict:
    path = Path("data/arena_dataset_profile.json")
    if not path.exists():
        response = {"source": "repo_json", "profile": {}}
    else:
        response = {"source": "repo_json", "profile": json.loads(path.read_text(encoding="utf-8"))}
    if error:
        response["error"] = error
    return response


def _read_local_episode_sample(limit: int, error: str | None = None) -> dict:
    path = Path("data/arena_liveops_episodes.csv")
    rows: list[dict] = []
    source = "repo_arena_csv"
    if path.exists():
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
                if len(rows) >= limit:
                    break
    else:
        from app.core import simulate_dataset
        from app.policies import RuleBasedPolicy

        rows = simulate_dataset(RuleBasedPolicy().recommend, n_players=2, days=max(1, limit), seed=11, policy_name="rule").head(limit).to_dict("records")
        source = "generated_fallback"
    response = {"source": source, "rows": rows}
    if error:
        response["error"] = error
    return response

def read_synthetic_dataset_profile() -> dict:
    if not settings.use_bigquery:
        return _read_local_profile()
    try:
        rows = _query_rows(f"SELECT profile_json FROM `{_table_id('synthetic_dataset_profile')}` LIMIT 1")
        profile = json.loads(rows[0].get("profile_json") or "{}") if rows else {}
        return {"source": "bigquery", "profile": profile}
    except Exception as exc:
        logger.warning("BigQuery synthetic dataset profile read failed, falling back to local JSON: %s", exc)
        return _read_local_profile(str(exc))


def read_synthetic_episode_sample(limit: int = 20) -> dict:
    limit = _bounded_limit(limit, default=20, maximum=20)
    if not settings.use_bigquery:
        return _read_local_episode_sample(limit)
    try:
        table = _table_id("synthetic_liveops_episodes")
        sql = f"""
            SELECT player_id, day, segment, skill, frustration, engagement, churn_risk,
                   economy_balance, recent_losses, recent_rewards, action, action_probability,
                   reward, next_skill, next_frustration, next_engagement, next_churn_risk,
                   next_economy_balance, next_recent_losses, next_recent_rewards, retained,
                   economy_penalty, policy_name, dataset_version
            FROM `{table}`
            ORDER BY player_id, day
            LIMIT {limit}
        """
        return {"source": "bigquery", "rows": _query_rows(sql)}
    except Exception as exc:
        logger.warning("BigQuery synthetic episode sample read failed, falling back to local CSV: %s", exc)
        return _read_local_episode_sample(limit, str(exc))


def _read_recent_table(table_name: str, sql_template: str, limit: int = 5) -> dict:
    limit = _bounded_limit(limit, default=5, maximum=20)
    if not settings.use_bigquery:
        return {"configured": False, "table": table_name, "rows": [], "message": "BigQuery is not configured for this runtime."}
    try:
        table = _table_id(table_name)
        rows = _query_rows(sql_template.format(table=table, limit=limit))
        return {"configured": True, "table": table_name, "rows": rows}
    except Exception as exc:
        logger.warning("Recent BigQuery query failed for %s: %s", table_name, exc)
        return {"configured": True, "table": table_name, "rows": [], "error": str(exc)}


def read_recent_recommendations(limit: int = 5) -> dict:
    return _read_recent_table(
        "recommendation_logs",
        "SELECT created_at, request_json, response_json FROM `{table}` ORDER BY created_at DESC LIMIT {limit}",
        limit,
    )


def read_recent_audits(limit: int = 5) -> dict:
    return _read_recent_table(
        "agent_audit_logs",
        "SELECT created_at, report_json FROM `{table}` ORDER BY created_at DESC LIMIT {limit}",
        limit,
    )


def read_recent_policy_metrics(limit: int = 5) -> dict:
    return _read_recent_table(
        "policy_eval_results",
        "SELECT created_at, metrics_json FROM `{table}` ORDER BY created_at DESC LIMIT {limit}",
        limit,
    )


















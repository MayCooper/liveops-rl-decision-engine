from __future__ import annotations

import json

import requests

BASE_URL = "http://127.0.0.1:8000"


def show(title: str, response: requests.Response) -> None:
    print(f"\n== {title} ==")
    print(json.dumps(response.json(), indent=2)[:3000])


def main() -> None:
    show("health", requests.get(f"{BASE_URL}/health", timeout=10))
    examples = [
        {
            "request_id": "demo-new",
            "player": {"segment": "new", "skill": 0.2, "frustration": 0.86, "engagement": 0.3, "churn_risk": 0.78, "economy_balance": 0.42, "recent_losses": 5, "recent_rewards": 1, "day": 4},
        },
        {
            "request_id": "demo-advanced",
            "player": {"segment": "advanced", "skill": 0.9, "frustration": 0.2, "engagement": 0.32, "churn_risk": 0.46, "economy_balance": 0.7, "recent_losses": 1, "recent_rewards": 0, "day": 7},
        },
        {
            "request_id": "demo-stable",
            "player": {"segment": "mid_skill", "skill": 0.6, "frustration": 0.22, "engagement": 0.75, "churn_risk": 0.2, "economy_balance": 0.66, "recent_losses": 0, "recent_rewards": 1, "day": 5},
        },
    ]
    for item in examples:
        show(item["request_id"], requests.post(f"{BASE_URL}/recommend_action", json=item, timeout=10))
    show("policy_metrics", requests.get(f"{BASE_URL}/policy_metrics", timeout=10))
    show("run_agent_audit", requests.post(f"{BASE_URL}/run_agent_audit", timeout=30))


if __name__ == "__main__":
    main()


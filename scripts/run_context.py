import os
import subprocess
from datetime import datetime, timezone


def _git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def make_run_context(engine_version="MAX-v1.1"):
    now = datetime.now(timezone.utc)
    return {
        "run_id": now.strftime("%Y%m%d_%H%M%S") + "_" + _git_sha(),
        "git_sha": _git_sha(),
        "engine_version": engine_version,
        "data_schema_version": "1.0",
        "created_at_utc": now.isoformat(),
        "data_sources": {
            "fixtures": "wencai + api-football",
            "odds": "wencai + the-odds-api-budgeted",
        },
        "odds_api_policy": {
            "monthly_limit": int(os.environ.get("ODDS_API_MONTHLY_LIMIT", "100") or 100),
            "daily_budget": int(os.environ.get("ODDS_API_DAILY_BUDGET", "3") or 3),
        },
    }

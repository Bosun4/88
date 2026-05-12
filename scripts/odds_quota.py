import json
import os
from datetime import datetime, timezone

STATE_FILE = os.environ.get("ODDS_QUOTA_STATE_FILE", "data/odds_quota_state.json")
MONTHLY_LIMIT = int(os.environ.get("ODDS_API_MONTHLY_LIMIT", "100") or 100)
DAILY_BUDGET = int(os.environ.get("ODDS_API_DAILY_BUDGET", "3") or 3)


def _month_key(now=None):
    now = now or datetime.now(timezone.utc)
    return now.strftime("%Y-%m")


def _day_key(now=None):
    now = now or datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%d")


def load_state(path=STATE_FILE):
    if not os.path.exists(path):
        return {"month": _month_key(), "days": {}, "month_used": 0}
    try:
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
    except Exception:
        state = {"month": _month_key(), "days": {}, "month_used": 0}
    if state.get("month") != _month_key():
        state = {"month": _month_key(), "days": {}, "month_used": 0}
    state.setdefault("days", {})
    state.setdefault("month_used", 0)
    return state


def save_state(state, path=STATE_FILE):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def can_spend(cost=1, path=STATE_FILE):
    state = load_state(path)
    day = _day_key()
    day_used = int(state.get("days", {}).get(day, 0))
    month_used = int(state.get("month_used", 0))
    if day_used + cost > DAILY_BUDGET:
        return False, f"daily budget exceeded: {day_used}+{cost}>{DAILY_BUDGET}", state
    if month_used + cost > MONTHLY_LIMIT:
        return False, f"monthly budget exceeded: {month_used}+{cost}>{MONTHLY_LIMIT}", state
    return True, "ok", state


def record_spend(cost=1, path=STATE_FILE, response_headers=None):
    state = load_state(path)
    day = _day_key()
    state["days"][day] = int(state.get("days", {}).get(day, 0)) + cost
    state["month_used"] = int(state.get("month_used", 0)) + cost
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    if response_headers:
        # The Odds API commonly returns x-requests-remaining/x-requests-used headers.
        state["last_response_quota_headers"] = {
            k: v for k, v in response_headers.items()
            if k.lower() in {"x-requests-remaining", "x-requests-used", "x-requests-last"}
        }
    save_state(state, path)
    return state


def quota_summary(path=STATE_FILE):
    state = load_state(path)
    day = _day_key()
    return {
        "month": state.get("month"),
        "month_used_local": int(state.get("month_used", 0)),
        "month_limit": MONTHLY_LIMIT,
        "today_used_local": int(state.get("days", {}).get(day, 0)),
        "daily_budget": DAILY_BUDGET,
        "last_response_quota_headers": state.get("last_response_quota_headers", {}),
    }

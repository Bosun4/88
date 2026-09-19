"""Shared kickoff gate for inference and publication; preserve analysis on expiry."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def parse_aware_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        return parsed.astimezone(timezone.utc) if parsed.tzinfo else None
    except ValueError:
        return None


def prematch_status(match: dict, now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("Current time must include timezone")
    kickoff = parse_aware_time(match.get("kickoff_at"))
    if kickoff is None:
        return "kickoff_unverified"
    return "eligible" if now < kickoff else "already_started"


def enforce_publication_gate(match: dict, now: datetime) -> str:
    status = prematch_status(match, now)
    pred = match.setdefault("prediction", {})
    pred["prematch_status"] = status
    pred["published_at"] = now.isoformat()
    pred.setdefault("prediction_completed_at", now.isoformat())
    if status != "eligible":
        pred["recommend_gate_pass"] = False
        pred["bet_action"] = "no_bet"
        pred.setdefault("recommendation", {}).update(is_recommended=False, bet_action="no_bet")
        pred.setdefault("recommend_gate_reasons", []).append(status)
        pred["recommend_gate_reasons"] = list(dict.fromkeys(pred["recommend_gate_reasons"]))
        pred["bet_recommendation"] = {
            "available": False, "no_bet": True, "reason": status,
            "steady": {"legs": [], "total_stake": 0},
            "aggressive": {"legs": [], "total_stake": 0},
        }
        for key in ("is_recommended", "is_strict_recommended", "is_top4_candidate"):
            match[key] = False
    return status

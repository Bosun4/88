"""Minimal schema helpers for historical match snapshots.

The key rule for valid backtesting: every prediction must be evaluated against
what was known at prediction time. Do not rebuild features from post-match data.
"""
from datetime import datetime, timezone

SCHEMA_VERSION = "1.0"


def make_historical_match_snapshot(match, prediction=None, odds_snapshot=None, features_snapshot=None):
    return {
        "schema_version": SCHEMA_VERSION,
        "match_id": match.get("match_id") or match.get("fixture_id") or match.get("id"),
        "date": match.get("date"),
        "league": match.get("league"),
        "season": match.get("season") or match.get("stats_season"),
        "home_team": match.get("home_team"),
        "away_team": match.get("away_team"),
        "actual_score": match.get("actual_score"),
        "actual_result": match.get("actual_result"),
        "odds_snapshot": odds_snapshot or match.get("odds_snapshot") or match.get("odds_api") or {},
        "features_snapshot": features_snapshot or {
            "home_stats": match.get("home_stats", {}),
            "away_stats": match.get("away_stats", {}),
            "h2h": match.get("h2h", []),
            "data_quality_score": match.get("data_quality_score"),
            "data_warnings": match.get("data_warnings", []),
        },
        "prediction_snapshot": prediction or match.get("prediction") or {},
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def validate_historical_match_snapshot(row):
    required = ["match_id", "date", "home_team", "away_team", "actual_score", "prediction_snapshot"]
    missing = [k for k in required if not row.get(k)]
    return {"valid": not missing, "missing": missing}

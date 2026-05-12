#!/usr/bin/env python3
"""Conservative historical fixture collector skeleton.

This module intentionally defaults to dry-run. It is designed to fetch small
API-Football result windows first, then let the prediction/backtest pipeline
consume immutable snapshots.
"""
import argparse
import json
import os
from datetime import datetime, timedelta

API_FOOTBALL_BASE = os.environ.get("API_FOOTBALL_BASE", "https://v3.football.api-sports.io")
API_FOOTBALL_KEY = os.environ.get("API_FOOTBALL_KEY", "")


def date_range(start, end):
    d = datetime.strptime(start, "%Y-%m-%d").date()
    e = datetime.strptime(end, "%Y-%m-%d").date()
    while d <= e:
        yield d.strftime("%Y-%m-%d")
        d += timedelta(days=1)


def build_plan(start, end, leagues=None, max_days=7):
    days = list(date_range(start, end))[:max_days]
    return {
        "source": "api-football",
        "days": days,
        "leagues": leagues or [],
        "requests_estimate": len(days),
        "dry_run_safe": True,
        "notes": [
            "Start with <=7 days to validate schema.",
            "Persist raw responses under data/raw/api_football/YYYY-MM-DD.json.",
            "Do not mix post-match stats into pre-match feature snapshots.",
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--league", action="append", default=[])
    ap.add_argument("--max-days", type=int, default=7)
    ap.add_argument("--dry-run", action="store_true", default=True)
    args = ap.parse_args()
    print(json.dumps(build_plan(args.start, args.end, args.league, args.max_days), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

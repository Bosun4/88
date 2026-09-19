"""Season-scoped, JSON-safe league evidence for the prediction boundary.

Input: match.league_evidence contains sourced fixture, standings, rounds and
home/away_schedule snapshots from fetch_data. External rotation evidence may use
home/away_rotation with team_id and statement. Every snapshot must include source,
source_url, aware captured_at, league_id and season. No network or rank narratives.
"""
from __future__ import annotations

import re

try:
    from .fixture_identity import parse_time
except ImportError:
    from fixture_identity import parse_time


def _fact(value=None, *sources, reason=None, derived=False):
    known = value not in (None, "unknown")
    return {"value": value, "status": ("derived" if derived else "observed") if known else "unknown",
            "sources": [_source(s) for s in sources if isinstance(s, dict)],
            "reason": reason}


def _source(snapshot):
    fields = ("source", "source_url", "captured_at", "league_id", "season", "fixture_id")
    return {key: snapshot[key] for key in fields
            if type(snapshot.get(key)) in (str, int, float, bool)}


def _valid(snapshot, match, max_age_hours=48):
    if not isinstance(snapshot, dict) or not snapshot.get("source") or not snapshot.get("source_url"):
        return False
    if snapshot.get("estimated") or snapshot.get("quality") in ("unavailable", "research_only"):
        return False
    kickoff = parse_time(match.get("kickoff_at"))
    captured = parse_time(snapshot.get("captured_at"))
    # match.captured_at can precede enrichment: use the explicit analysis cutoff
    # if provided, otherwise kickoff is the strict upper bound for stored evidence.
    cutoff = parse_time(match.get("prediction_started_at")) or kickoff
    return bool(kickoff and captured and cutoff and captured < kickoff and captured <= cutoff
                and 0 <= (cutoff - captured).total_seconds() <= max_age_hours * 3600
                and all(match.get(k) is not None and str(snapshot.get(k)) == str(match[k])
                        for k in ("league_id", "season")))


def _round_number(label):
    found = re.fullmatch(r"Regular Season - (\d+)", str(label), re.IGNORECASE)
    return int(found[1]) if found else None


def _standings(snapshot, match):
    if not _valid(snapshot, match):
        return []
    groups = snapshot.get("groups")
    if not isinstance(groups, list):
        return []
    eligible = []
    for rows in groups:
        if not isinstance(rows, list) or not rows:
            continue
        if not all(isinstance(r, dict) and isinstance(r.get("team"), dict)
                   and r["team"].get("id") and type(r.get("rank")) is int
                   and r["rank"] > 0 and type(r.get("points")) is int
                   and isinstance(r.get("all"), dict)
                   and type(r["all"].get("played")) is int and r["all"]["played"] >= 0
                   for r in rows):
            continue
        ids = [r["team"]["id"] for r in rows]
        if len(ids) != len(set(ids)) or len({r["rank"] for r in rows}) != len(rows):
            continue
        if not all(match.get(f"{side}_id") in ids for side in ("home", "away")):
            continue
        captured = parse_time(snapshot["captured_at"])
        if any(r.get("update") and (not parse_time(r["update"])
               or not 0 <= (captured - parse_time(r["update"])).total_seconds() <= 48 * 3600)
               for r in rows):
            continue
        eligible.append(rows)
    return eligible[0] if len(eligible) == 1 else []


def _schedule_context(team, snapshot, rotation, match, side):
    """Kickoff-to-kickoff days, not claimed recovery or inferred rotation."""
    team_id = match.get(f"{side}_id")
    if (_valid(rotation, match) and rotation.get("team_id") == team_id
            and isinstance(rotation.get("statement"), str) and rotation["statement"].strip()):
        team["rotation"] = _fact(rotation["statement"], rotation)
    if not _valid(snapshot, match) or snapshot.get("team_id") != team_id:
        return
    fixtures = snapshot.get("fixtures")
    if not isinstance(fixtures, list):
        return
    kickoff = parse_time(match.get("kickoff_at"))
    captured = parse_time(snapshot["captured_at"])
    before, after, seen = [], [], set()
    for row in fixtures:
        if not isinstance(row, dict):
            continue
        fixture, teams = row.get("fixture") or {}, row.get("teams") or {}
        when = parse_time(fixture.get("date"))
        identifier = fixture.get("id")
        if (not identifier or not when or identifier in seen
                or identifier == match.get("api_football_fixture_id")
                or team_id not in [(teams.get(s) or {}).get("id") for s in ("home", "away")]):
            continue
        seen.add(identifier)
        status = (fixture.get("status") or {}).get("short")
        info = {"fixture_id": identifier, "kickoff_at": when.isoformat(),
                "league_id": (row.get("league") or {}).get("id"), "status": status}
        if when < min(kickoff, captured) and status in ("FT", "AET", "PEN"):
            before.append((when, info))
        elif when > kickoff and status == "NS":
            after.append((when, info))
    if before:
        when, info = max(before, key=lambda pair: pair[0])
        team["previous_fixture"] = _fact(info, snapshot)
        team["rest_days"] = _fact(round((kickoff - when).total_seconds() / 86400, 3),
                                   snapshot, derived=True, reason="kickoff_to_kickoff_days")
    if after:
        when, info = min(after, key=lambda pair: pair[0])
        team["next_fixture"] = _fact(info, snapshot)
        team["next_match_in_days"] = _fact(round((when - kickoff).total_seconds() / 86400, 3),
                                            snapshot, derived=True)
    if before or after:
        count = sum(abs((when - kickoff).total_seconds()) <= 7 * 86400
                    for when, _ in before + after)
        team["fixtures_within_7_days"] = _fact(count, snapshot, derived=True,
                                               reason="observed_neighbours_excluding_target_not_complete_schedule")


def build_league_context(match: dict) -> dict:
    """Return facts/derived gaps with provenance, or explicit unknowns.

    Stage uses verified regular-round progress (first/last quarter), not calendar
    month or ranking. Gaps are signed: boundary points minus team points; the
    relegation cushion reverses that sign. They do not assert motivation, safety,
    qualification or title lock: tiebreak rules and all remaining games are needed.
    """
    match = match if isinstance(match, dict) else {}
    evidence = match.get("league_evidence") or {}
    if not isinstance(evidence, dict):
        evidence = {}
    fixture = evidence.get("fixture") or {}
    valid_fixture = (_valid(fixture, match)
                     and fixture.get("fixture_id") == match.get("api_football_fixture_id")
                     and bool(fixture.get("fixture_id"))
                     and parse_time(fixture.get("kickoff_at")) == parse_time(match.get("kickoff_at"))
                     and all(fixture.get(f"{s}_id") == match.get(f"{s}_id")
                             and match.get(f"{s}_id") for s in ("home", "away")))
    result = {"version": 1, "league": str(match.get("league") or "unknown"),
              "season": _fact(match.get("season") if valid_fixture else None, *([fixture] if valid_fixture else [])),
              "stage": _fact("unknown", reason="missing_verified_rounds"),
              "round": _fact(), "round_label": _fact(), "total_rounds": _fact(),
              "note": "分差仅为积分榜快照；不凭固定排名推断战意、卖分、提前夺冠或保级。"}
    number, total = None, None
    rounds = evidence.get("rounds") or {}
    if valid_fixture:
        label = fixture.get("round")
        result["round_label"] = _fact(label if isinstance(label, str) else None, fixture)
        number = _round_number(label)
        result["round"] = _fact(number, fixture, derived=True)
        if _valid(rounds, match) and isinstance(rounds.get("rounds"), list):
            numbers = [_round_number(r) for r in rounds["rounds"]]
            if numbers and all(n is not None for n in numbers):
                ordered = sorted(set(numbers))
                if ordered == list(range(1, max(ordered) + 1)) and number in ordered:
                    total = max(ordered)
                    result["total_rounds"] = _fact(total, rounds, derived=True)
                    stage = "early" if number / total <= .25 else "late" if number / total >= .75 else "mid"
                    result["stage"] = _fact(stage, fixture, rounds, derived=True,
                                             reason="regular_round_fraction_quarters")
        if number is None and isinstance(label, str):
            low = label.casefold()
            if any(x in low for x in ("quarter-final", "semi-final", "round of", "决赛", "淘汰")) or low == "final":
                result["stage"] = _fact("knockout", fixture, derived=True)
            elif "group" in low or "小组" in low:
                result["stage"] = _fact("group", fixture, derived=True)
    table = evidence.get("standings") or {}
    rows = _standings(table, match) if valid_fixture else []
    for side in ("home", "away"):
        team = {key: _fact() for key in ("rank", "points", "played", "remaining_matches",
                                       "title_gap", "europe_gap", "relegation_cushion")}
        team.update({"motivation": _fact("unknown", reason="points_do_not_prove_intent"),
                     "rotation": _fact("unknown", reason="no_sourced_rotation_evidence"),
                     "rest_days": _fact(), "next_match_in_days": _fact(),
                     "previous_fixture": _fact(), "next_fixture": _fact(),
                     "fixtures_within_7_days": _fact()})
        own = next((r for r in rows if r["team"]["id"] == match.get(f"{side}_id")), None)
        if own:
            for field in ("rank", "points"):
                team[field] = _fact(own[field], table)
            played = own["all"]["played"]
            team["played"] = _fact(played, table)
            # Rounds can include byes/splits. Only a complete regular round-robin
            # table whose round count agrees with team count establishes games.
            complete = sorted(r["rank"] for r in rows) == list(range(1, len(rows) + 1))
            if (total is not None and played <= total and complete
                    and total in (len(rows) - 1, 2 * (len(rows) - 1))):
                team["remaining_matches"] = _fact(total - played, table, rounds, derived=True)
            leader = next((r for r in rows if r["rank"] == 1), None)
            if leader:
                team["title_gap"] = _fact(leader["points"] - own["points"], table, derived=True)
            europe = [r for r in rows if re.search(r"champions league|europa league|conference league",
                                                   str(r.get("description") or ""), re.I)]
            relegation = [r for r in rows if re.search(r"^relegation(?:\s|$)",
                                                       str(r.get("description") or ""), re.I)]
            if europe:
                boundary = max(europe, key=lambda r: r["rank"])
                team["europe_gap"] = _fact(boundary["points"] - own["points"], table, derived=True)
            if relegation:
                boundary = min(relegation, key=lambda r: r["rank"])
                team["relegation_cushion"] = _fact(own["points"] - boundary["points"], table, derived=True)
        if valid_fixture:
            _schedule_context(team, evidence.get(f"{side}_schedule") or {},
                              evidence.get(f"{side}_rotation") or {}, match, side)
        result[side] = team
    return result

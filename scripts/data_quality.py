def assess_match_quality(match):
    """Return a simple, visible data-quality score for each match."""
    score = 100
    missing = []
    warnings = []

    required = ["home_team", "away_team", "league", "sp_home", "sp_draw", "sp_away"]
    for key in required:
        if match.get(key) in (None, "", 0, 0.0):
            missing.append(key)
            score -= 10

    if not match.get("home_id") or not match.get("away_id"):
        warnings.append("api_football_team_id_missing")
        score -= 8

    for side in ("home", "away"):
        stats = match.get(f"{side}_stats") or {}
        if not stats or str(stats.get("played", "0")) in ("0", "?", ""):
            warnings.append(f"{side}_stats_missing_or_fallback")
            score -= 8

    if not match.get("odds_api"):
        warnings.append("the_odds_api_missing_or_budget_skipped")
        score -= 12

    if not match.get("h2h"):
        warnings.append("h2h_missing")
        score -= 4

    if match.get("stats_season") is None:
        warnings.append("stats_season_missing")
        score -= 4

    level = "high" if score >= 80 else "medium" if score >= 60 else "low"
    return {
        "data_quality_score": max(0, min(100, score)),
        "data_quality_level": level,
        "missing_fields": missing,
        "data_warnings": warnings,
    }


def attach_quality(matches):
    for match in matches:
        match.update(assess_match_quality(match))
    return matches


def summarize_quality(matches):
    if not matches:
        return {"avg_score": 0, "low_quality_matches": 0, "count": 0}
    scores = [int(m.get("data_quality_score", 0)) for m in matches]
    return {
        "avg_score": round(sum(scores) / len(scores), 1),
        "low_quality_matches": sum(1 for m in matches if m.get("data_quality_level") == "low"),
        "count": len(matches),
    }

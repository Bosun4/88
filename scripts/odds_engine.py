# odds_engine.py - Core odds-based score prediction
# This replaces AI guessing with mathematical odds analysis

def calc_ttg_profile(v2):
    """Calculate total goals profile from TTG odds"""
    ttg_map = {"a0":0,"a1":1,"a2":2,"a3":3,"a4":4,"a5":5,"a6":6,"a7":7}
    raw = {}
    for k, g in ttg_map.items():
        ov = float(v2.get(k, 0) or 0)
        if ov > 1:
            raw[g] = ov
    if not raw:
        return {"exp": 2.5, "best": 2, "second": 3, "probs": {}, "zero_odds": 99, "seven_odds": 99}
    probs = {k: 1/v for k, v in raw.items()}
    ti = sum(probs.values())
    probs = {k: round(v/ti*100, 1) for k, v in probs.items()}
    exp = sum(k*v/100 for k, v in probs.items())
    sorted_ttg = sorted(raw.items(), key=lambda x: x[1])
    best = sorted_ttg[0][0]
    second = sorted_ttg[1][0] if len(sorted_ttg) > 1 else best
    return {
        "exp": round(exp, 1),
        "best": best,
        "second": second,
        "probs": probs,
        "zero_odds": raw.get(0, 99),
        "seven_odds": raw.get(7, 99),
        "raw": raw
    }


def calc_had_profile(had_dict):
    """Analyze HAD odds for direction and confidence"""
    w = float(had_dict.get("w", had_dict.get("win", 2.5)))
    d = float(had_dict.get("d", had_dict.get("same", had_dict.get("draw", 3.2))))
    l = float(had_dict.get("l", had_dict.get("lose", 3.0)))
    vals = sorted([("H", w), ("D", d), ("A", l)], key=lambda x: x[1])
    direction = vals[0][0]
    gap = (vals[1][1] - vals[0][1]) / max(vals[0][1], 0.01)
    return {
        "direction": direction,
        "confident": gap > 0.15,
        "super_fav_h": w < 1.25,
        "super_fav_a": l < 1.25,
        "strong_fav_h": w < 1.60,
        "strong_fav_a": l < 1.60,
        "draw_lean": d < min(w, l) * 1.08,
        "uncertain": gap < 0.12,
        "w": w, "d": d, "l": l,
        "gap": round(gap, 3)
    }


def parse_crs(v2):
    """Parse all CRS odds into structured data"""
    crs_map = {
        "w10":"1-0","w20":"2-0","w21":"2-1","w30":"3-0","w31":"3-1","w32":"3-2",
        "w40":"4-0","w41":"4-1","w42":"4-2",
        "s00":"0-0","s11":"1-1","s22":"2-2","s33":"3-3",
        "l01":"0-1","l02":"0-2","l12":"1-2","l03":"0-3","l13":"1-3","l23":"2-3"
    }
    scores = {}
    for k, sc in crs_map.items():
        ov = float(v2.get(k, 0) or 0)
        if ov > 1:
            h, a = int(sc.split("-")[0]), int(sc.split("-")[1])
            scores[sc] = {
                "odds": ov,
                "prob": round(1/ov*100, 1),
                "total": h + a,
                "dir": "H" if h > a else "D" if h == a else "A",
                "h": h, "a": a
            }
    return scores


def predict_score(match_data):
    """
    Core prediction engine using layered odds analysis.
    Returns: {"score": "2-1", "reason": "...", "candidates": [...], "confidence": 70}
    """
    v2 = match_data.get("v2_odds_dict", {})
    if not v2:
        v2 = match_data

    sp_h = float(match_data.get("sp_home", match_data.get("win", 2.5)) or 2.5)
    sp_d = float(match_data.get("sp_draw", match_data.get("same", 3.2)) or 3.2)
    sp_a = float(match_data.get("sp_away", match_data.get("lose", 3.0)) or 3.0)

    had = {"w": sp_h, "d": sp_d, "l": sp_a}
    ttg = calc_ttg_profile(v2)
    had_p = calc_had_profile(had)
    crs = parse_crs(v2)

    if not crs:
        return {"score": "1-1", "reason": "no CRS data", "candidates": [], "confidence": 30}

    best_total = ttg["best"]
    second_total = ttg["second"]
    exp_goals = ttg["exp"]
    zero_odds = ttg["zero_odds"]

    # Helper: filter CRS by direction
    def dir_filter(d):
        return {s: c for s, c in crs.items() if c["dir"] == d}

    # Helper: filter by total goals range
    def total_filter(scores, lo, hi):
        return {s: c for s, c in scores.items() if lo <= c["total"] <= hi}

    # Helper: best in filtered set (lowest odds)
    def best_in(scores):
        if not scores:
            return None
        return min(scores, key=lambda s: scores[s]["odds"])

    # ================================================================
    # LAYER 0: Super favorite (odds < 1.25)
    # ================================================================
    if had_p["super_fav_h"]:
        pool = dir_filter("H")
        pool = total_filter(pool, max(best_total, 2), best_total + 2) or pool
        pick = best_in(pool)
        if pick:
            return {"score": pick, "reason": "SUPER_FAV_HOME @%.2f" % sp_h,
                    "candidates": sorted(pool, key=lambda s: pool[s]["odds"])[:3],
                    "confidence": 80, "layer": 0}

    if had_p["super_fav_a"]:
        pool = dir_filter("A")
        pool = total_filter(pool, max(best_total, 2), best_total + 2) or pool
        pick = best_in(pool)
        if pick:
            return {"score": pick, "reason": "SUPER_FAV_AWAY @%.2f" % sp_a,
                    "candidates": sorted(pool, key=lambda s: pool[s]["odds"])[:3],
                    "confidence": 80, "layer": 0}

    # ================================================================
    # LAYER 1: Zero-goal signal (0-goal odds < 8.5)
    # Only trigger if draw is not the worst outcome
    # ================================================================
    if zero_odds < 8.5 and sp_d <= max(sp_h, sp_a):
        return {"score": "0-0", "reason": "ZERO_SIGNAL @%.1f" % zero_odds,
                "candidates": ["0-0", "1-0", "0-1"],
                "confidence": 65, "layer": 1}

    # ================================================================
    # LAYER 2: One-goal signal (1-goal odds < 4.0 + clear direction)
    # ================================================================
    if ttg["raw"].get(1, 99) < 4.0 and had_p["confident"]:
        if had_p["direction"] == "H":
            return {"score": "1-0", "reason": "ONE_GOAL_HOME 1g@%.1f" % ttg["raw"].get(1, 99),
                    "candidates": ["1-0", "0-0", "2-0"],
                    "confidence": 60, "layer": 2}
        elif had_p["direction"] == "A":
            return {"score": "0-1", "reason": "ONE_GOAL_AWAY 1g@%.1f" % ttg["raw"].get(1, 99),
                    "candidates": ["0-1", "0-0", "0-2"],
                    "confidence": 60, "layer": 2}

    # ================================================================
    # LAYER 3: Strong favorite (odds < 1.60)
    # ================================================================
    if had_p["strong_fav_h"]:
        pool = dir_filter("H")
        target = max(best_total, 2)
        pool = total_filter(pool, target - 1, target + 1) or pool
        pick = best_in(pool)
        if pick:
            return {"score": pick, "reason": "STRONG_FAV_HOME @%.2f ttg=%d" % (sp_h, best_total),
                    "candidates": sorted(pool, key=lambda s: pool[s]["odds"])[:3],
                    "confidence": 70, "layer": 3}

    if had_p["strong_fav_a"]:
        pool = dir_filter("A")
        target = max(best_total, 2)
        pool = total_filter(pool, target - 1, target + 1) or pool
        pick = best_in(pool)
        if pick:
            return {"score": pick, "reason": "STRONG_FAV_AWAY @%.2f ttg=%d" % (sp_a, best_total),
                    "candidates": sorted(pool, key=lambda s: pool[s]["odds"])[:3],
                    "confidence": 70, "layer": 3}

    # ================================================================
    # LAYER 4: Draw signal (all odds close + draw competitive)
    # ================================================================
    if had_p["draw_lean"] or (had_p["uncertain"] and sp_d < 3.3):
        pool = dir_filter("D")
        pool = total_filter(pool, max(best_total - 1, 0), best_total + 1) or pool
        pick = best_in(pool)
        if pick:
            return {"score": pick, "reason": "DRAW_SIGNAL d=%.2f gap=%.3f" % (sp_d, had_p["gap"]),
                    "candidates": sorted(pool, key=lambda s: pool[s]["odds"])[:3],
                    "confidence": 55, "layer": 4}

    # ================================================================
    # LAYER 5: General - direction + total goals crossfilter
    # ================================================================
    d = had_p["direction"]
    pool = dir_filter(d)

    # Filter by total goals range (best ± 1)
    filtered = total_filter(pool, max(best_total - 1, 0), best_total + 1)
    if filtered:
        pick = best_in(filtered)
        return {"score": pick, "reason": "DIR_%s_TTG%d" % (d, best_total),
                "candidates": sorted(filtered, key=lambda s: filtered[s]["odds"])[:3],
                "confidence": 55, "layer": 5}

    # No total match, just direction
    if pool:
        pick = best_in(pool)
        return {"score": pick, "reason": "DIR_%s_ONLY" % d,
                "candidates": sorted(pool, key=lambda s: pool[s]["odds"])[:3],
                "confidence": 45, "layer": 5}

    # ================================================================
    # FALLBACK: absolute lowest CRS odds
    # ================================================================
    pick = min(crs, key=lambda s: crs[s]["odds"])
    return {"score": pick, "reason": "FALLBACK",
            "candidates": sorted(crs, key=lambda s: crs[s]["odds"])[:3],
            "confidence": 35, "layer": 6}


def get_over_25_from_odds(v2):
    """Calculate over 2.5 probability from TTG odds"""
    ttg = calc_ttg_profile(v2)
    probs = ttg["probs"]
    if not probs:
        return 50.0
    return round(sum(p for g, p in probs.items() if g >= 3), 1)


def get_btts_estimate(v2, crs_data=None):
    """Estimate BTTS from CRS odds"""
    crs = parse_crs(v2) if not crs_data else crs_data
    btts_scores = {s: c for s, c in crs.items() if c["h"] > 0 and c["a"] > 0}
    total_prob = sum(c["prob"] for c in crs.values())
    btts_prob = sum(c["prob"] for c in btts_scores.values())
    if total_prob > 0:
        return round(btts_prob / total_prob * 100, 1)
    return 45.0


def build_odds_summary(match_data):
    """Build human-readable odds analysis for AI prompts"""
    v2 = match_data.get("v2_odds_dict", {})
    if not v2: v2 = match_data

    ttg = calc_ttg_profile(v2)
    had = calc_had_profile({
        "w": float(match_data.get("sp_home", match_data.get("win", 2.5)) or 2.5),
        "d": float(match_data.get("sp_draw", match_data.get("same", 3.2)) or 3.2),
        "l": float(match_data.get("sp_away", match_data.get("lose", 3.0)) or 3.0)
    })
    crs = parse_crs(v2)
    result = predict_score(match_data)

    lines = []
    lines.append("[ODDS ENGINE ANALYSIS]")
    lines.append("Direction: %s (confidence: %s, gap: %.1f%%)" % (
        had["direction"], "HIGH" if had["confident"] else "LOW", had["gap"]*100))
    lines.append("Expected goals: %.1f | Most likely: %d goals" % (ttg["exp"], ttg["best"]))

    if ttg["zero_odds"] < 8.5:
        lines.append("WARNING: 0-goal @%.1f = high 0-0 chance!" % ttg["zero_odds"])
    if ttg["seven_odds"] < 15:
        lines.append("WARNING: 7+goal @%.1f = HIGH scoring game!" % ttg["seven_odds"])

    # TTG distribution
    parts = []
    for g in sorted(ttg["probs"]):
        parts.append("%dg=%.0f%%" % (g, ttg["probs"][g]))
    lines.append("Goals dist: %s" % " ".join(parts))

    # Top CRS
    sorted_crs = sorted(crs.items(), key=lambda x: x[1]["odds"])[:7]
    lines.append("CRS TOP7: %s" % ", ".join(["%s@%.1f(%.1f%%)" % (s, c["odds"], c["prob"]) for s, c in sorted_crs]))

    # Engine recommendation
    lines.append("")
    lines.append("[ENGINE RECOMMENDATION]")
    lines.append("Score: %s (Layer %d: %s)" % (result["score"], result.get("layer", 9), result["reason"]))
    lines.append("Candidates: %s" % ", ".join(result["candidates"][:3]))
    lines.append("Confidence: %d%%" % result["confidence"])

    return "\n".join(lines)
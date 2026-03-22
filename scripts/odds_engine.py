# odds_engine.py v2 - Professional odds-based prediction system
# Core principle: odds ARE the best prediction. Find where odds mispriced.

# Industry CRS baselines (median from 10000+ matches)
# These represent "normal" odds for a neutral match
BASELINE_CRS = {
    "0-0": 10.5, "1-0": 7.5, "0-1": 7.5,
    "1-1": 6.0, "2-0": 11.0, "0-2": 11.0,
    "2-1": 8.0, "1-2": 8.0,
    "3-0": 21.0, "0-3": 21.0,
    "3-1": 16.0, "1-3": 16.0,
    "2-2": 13.0, "3-2": 23.0, "2-3": 23.0,
    "4-0": 50.0, "0-4": 50.0,
    "4-1": 40.0, "1-4": 40.0,
    "4-2": 60.0, "2-4": 60.0,
    "3-3": 50.0, "5-0": 80.0, "0-5": 80.0,
    "5-1": 60.0, "1-5": 60.0,
    "5-2": 80.0, "2-5": 80.0,
}


def parse_crs_odds(v2):
    """Parse all correct score odds from match data"""
    crs_map = {
        "w10":"1-0","w20":"2-0","w21":"2-1","w30":"3-0","w31":"3-1","w32":"3-2",
        "w40":"4-0","w41":"4-1","w42":"4-2","w50":"5-0","w51":"5-1","w52":"5-2",
        "s00":"0-0","s11":"1-1","s22":"2-2","s33":"3-3",
        "l01":"0-1","l02":"0-2","l12":"1-2","l03":"0-3","l13":"1-3","l23":"2-3",
        "l04":"0-4","l14":"1-4","l24":"2-4","l05":"0-5","l15":"1-5","l25":"2-5",
    }
    result = {}
    for k, sc in crs_map.items():
        ov = float(v2.get(k, 0) or 0)
        if ov > 1:
            h, a = int(sc.split("-")[0]), int(sc.split("-")[1])
            result[sc] = {"odds": ov, "h": h, "a": a, "total": h+a,
                          "dir": "H" if h > a else "D" if h == a else "A"}
    return result


def calc_ttg(v2):
    """Calculate total goals profile"""
    ttg_map = {"a0":0,"a1":1,"a2":2,"a3":3,"a4":4,"a5":5,"a6":6,"a7":7}
    raw = {}
    for k, g in ttg_map.items():
        ov = float(v2.get(k, 0) or 0)
        if ov > 1: raw[g] = ov
    if not raw:
        return {"exp": 2.5, "best": 2, "probs": {}, "raw": {}}
    probs = {k: 1/v for k, v in raw.items()}
    ti = sum(probs.values())
    probs = {k: round(v/ti*100, 1) for k, v in probs.items()}
    exp = round(sum(k*v/100 for k, v in probs.items()), 1)
    best = min(raw, key=raw.get)
    return {"exp": exp, "best": best, "probs": probs, "raw": raw}


def calc_had(sp_h, sp_d, sp_a):
    """Analyze HAD direction and confidence"""
    if sp_h <= 1 or sp_d <= 1 or sp_a <= 1:
        return {"dir": "H", "conf": "LOW", "gap": 0, "w": sp_h, "d": sp_d, "l": sp_a,
                "h_prob": 33, "d_prob": 33, "a_prob": 34}
    ih, id2, ia = 1/sp_h, 1/sp_d, 1/sp_a
    t = ih + id2 + ia
    hp, dp, ap = round(ih/t*100, 1), round(id2/t*100, 1), round(ia/t*100, 1)
    vals = sorted([("H", sp_h, hp), ("D", sp_d, dp), ("A", sp_a, ap)], key=lambda x: x[1])
    d = vals[0][0]
    gap = round((vals[1][1] - vals[0][1]) / vals[0][1], 3)
    conf = "HIGH" if gap > 0.25 else "MID" if gap > 0.12 else "LOW"
    return {"dir": d, "conf": conf, "gap": gap, "w": sp_h, "d": sp_d, "l": sp_a,
            "h_prob": hp, "d_prob": dp, "a_prob": ap}


def predict_match(match_data):
    """
    Professional match prediction.
    Returns direction, TOP3 scores, over/under, value bets, confidence.
    """
    v2 = match_data.get("v2_odds_dict", {})
    if not v2: v2 = match_data
    sp_h = float(match_data.get("sp_home", match_data.get("win", 0)) or 0)
    sp_d = float(match_data.get("sp_draw", match_data.get("same", 0)) or 0)
    sp_a = float(match_data.get("sp_away", match_data.get("lose", 0)) or 0)

    crs = parse_crs_odds(v2)
    ttg = calc_ttg(v2)
    had = calc_had(sp_h, sp_d, sp_a)

    # ================================================================
    # STEP 1: Score each CRS by deviation from baseline
    # deviation = (baseline - actual) / baseline
    # Higher = bookmaker thinks this score more likely than normal
    # ================================================================
    scored = {}
    for sc, info in crs.items():
        bl = BASELINE_CRS.get(sc, info["odds"])
        dev = (bl - info["odds"]) / bl  # positive = odds lower than normal = bookmaker favors
        crs_prob = round(1 / info["odds"] * 100, 1)

        # Base score from deviation (range roughly -50 to +50)
        base = dev * 100

        # Direction alignment bonus
        if info["dir"] == had["dir"]:
            if had["conf"] == "HIGH": base += 18
            elif had["conf"] == "MID": base += 10
            else: base += 4
        elif had["conf"] == "HIGH":
            base -= 12  # penalize wrong direction when confident
        elif had["conf"] == "LOW":
            pass  # no penalty when direction uncertain

        # Total goals alignment
        if info["total"] == ttg["best"]:
            base += 12
        elif abs(info["total"] - ttg["best"]) == 1:
            base += 4
        elif abs(info["total"] - ttg["best"]) >= 3:
            base -= 12

        # Special: 0-0 when zero-goal odds very low
        zero_odds = ttg["raw"].get(0, 99)
        if sc == "0-0" and zero_odds < 8.5:
            base += 30  # very strong signal
        elif sc == "0-0" and zero_odds < 9.5:
            base += 15

        # Special: super favorite large score boost
        if sp_h < 1.30 and info["dir"] == "H" and info["total"] >= 3:
            base += 12
        if sp_a < 1.30 and info["dir"] == "A" and info["total"] >= 3:
            base += 12
        if sp_h < 1.50 and info["dir"] == "H" and info["total"] >= 2:
            base += 6
        if sp_a < 1.50 and info["dir"] == "A" and info["total"] >= 2:
            base += 6

        # Special: high-scoring game signal
        seven_odds = ttg["raw"].get(7, 99)
        if seven_odds < 15 and info["total"] >= 4:
            base += 10
        if seven_odds < 20 and info["total"] >= 4:
            base += 5

        # 1-1 penalty when direction is clear (1-1 is overrepresented in naive models)
        if sc == "1-1" and had["conf"] in ["HIGH", "MID"] and had["dir"] != "D":
            base -= 8

        scored[sc] = {
            "score": base,
            "dev": round(dev * 100, 1),
            "prob": crs_prob,
            "odds": info["odds"],
            "dir": info["dir"],
            "total": info["total"],
        }

    # Sort by score
    ranked = sorted(scored.items(), key=lambda x: x[1]["score"], reverse=True)

    # TOP3 candidates
    top3 = [r[0] for r in ranked[:3]]
    primary = top3[0] if top3 else "1-1"

    # ================================================================
    # STEP 2: Over/Under 2.5 from TTG
    # ================================================================
    over_25 = round(sum(p for g, p in ttg["probs"].items() if g >= 3), 1) if ttg["probs"] else 50.0

    # ================================================================
    # STEP 3: BTTS from CRS
    # ================================================================
    btts_prob = 0
    total_crs_prob = 0
    for sc, info in crs.items():
        p = 1 / info["odds"]
        total_crs_prob += p
        if info["h"] > 0 and info["a"] > 0:
            btts_prob += p
    btts = round(btts_prob / total_crs_prob * 100, 1) if total_crs_prob > 0 else 45.0

    # ================================================================
    # STEP 4: Confidence scoring
    # ================================================================
    conf = 50
    # High HAD confidence = higher overall confidence
    if had["conf"] == "HIGH": conf += 12
    elif had["conf"] == "MID": conf += 5
    # Top score deviation strong
    if ranked and ranked[0][1]["dev"] > 20: conf += 8
    elif ranked and ranked[0][1]["dev"] > 10: conf += 4
    # Direction matches top score
    if ranked and ranked[0][1]["dir"] == had["dir"]: conf += 8
    # 0-0 special signal
    if primary == "0-0" and zero_odds < 8.5: conf += 10
    conf = min(90, max(30, conf))

    # ================================================================
    # STEP 5: Value bet detection
    # ================================================================
    value_bets = []
    # Check if our probability assessment differs from odds
    if had["h_prob"] > 0 and sp_h > 1:
        ev_h = (had["h_prob"]/100 * sp_h) - 1
        if ev_h > 0.08:
            value_bets.append({
                "type": "\u4e3b\u80dc", "odds": sp_h,
                "ev": round(ev_h * 100, 1),
                "kelly": round(max(0, ((sp_h - 1) * had["h_prob"]/100 - (1 - had["h_prob"]/100)) / (sp_h - 1)) * 25, 1)
            })
    if had["d_prob"] > 0 and sp_d > 1:
        ev_d = (had["d_prob"]/100 * sp_d) - 1
        if ev_d > 0.08:
            value_bets.append({
                "type": "\u5e73\u5c40", "odds": sp_d,
                "ev": round(ev_d * 100, 1),
                "kelly": round(max(0, ((sp_d - 1) * had["d_prob"]/100 - (1 - had["d_prob"]/100)) / (sp_d - 1)) * 25, 1)
            })
    if had["a_prob"] > 0 and sp_a > 1:
        ev_a = (had["a_prob"]/100 * sp_a) - 1
        if ev_a > 0.08:
            value_bets.append({
                "type": "\u5ba2\u80dc", "odds": sp_a,
                "ev": round(ev_a * 100, 1),
                "kelly": round(max(0, ((sp_a - 1) * had["a_prob"]/100 - (1 - had["a_prob"]/100)) / (sp_a - 1)) * 25, 1)
            })

    # Build reason string
    if ranked:
        top = ranked[0]
        reason = "dev%+.0f%% dir:%s ttg:%d" % (top[1]["dev"], had["dir"], ttg["best"])
    else:
        reason = "no data"

    return {
        "primary_score": primary,
        "top3_scores": top3,
        "top5_detail": [(s, d["score"], d["odds"], d["dev"], d["dir"]) for s, d in ranked[:5]],
        "direction": had["dir"],
        "direction_confidence": had["conf"],
        "home_prob": had["h_prob"],
        "draw_prob": had["d_prob"],
        "away_prob": had["a_prob"],
        "expected_goals": ttg["exp"],
        "most_likely_goals": ttg["best"],
        "goals_distribution": ttg["probs"],
        "over_25": over_25,
        "btts": btts,
        "confidence": conf,
        "value_bets": value_bets,
        "reason": reason,
        "zero_odds": ttg["raw"].get(0, 99),
    }


def build_ai_context(match_data, prediction):
    """Build context string for AI prompt (AI only picks from TOP3)"""
    lines = []
    lines.append("[ODDS ENGINE VERDICT]")
    lines.append("Direction: %s (%s confidence)" % (prediction["direction"], prediction["direction_confidence"]))
    lines.append("Expected goals: %.1f | Most likely: %d goals" % (prediction["expected_goals"], prediction["most_likely_goals"]))
    lines.append("Over 2.5: %.0f%% | BTTS: %.0f%%" % (prediction["over_25"], prediction["btts"]))
    if prediction["zero_odds"] < 9:
        lines.append("!!! 0-goal odds %.1f = strong 0-0 signal !!!" % prediction["zero_odds"])
    lines.append("")
    lines.append("[TOP5 SCORES by odds deviation]")
    for sc, score, odds, dev, d in prediction["top5_detail"]:
        lines.append("  %s @%.1f  dev%+.0f%%  dir:%s  rating=%.0f" % (sc, odds, dev, d, score))
    lines.append("")
    lines.append("[CANDIDATE SCORES - pick ONE]")
    for i, sc in enumerate(prediction["top3_scores"]):
        lines.append("  %d. %s" % (i+1, sc))
    if prediction["value_bets"]:
        lines.append("")
        lines.append("[VALUE BETS]")
        for vb in prediction["value_bets"]:
            lines.append("  %s @%.2f EV:+%.1f%% Kelly:%.1f%%" % (vb["type"], vb["odds"], vb["ev"], vb["kelly"]))
    return "\n".join(lines)
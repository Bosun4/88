# odds_engine.py v3 - Professional odds-based prediction
# Core: find CRS odds that deviate most from baseline + league adjustment
# Integrated with league_intel for Serie A draws, Ligue 2 low scoring, etc.

from league_intel import detect_league_key, LEAGUE_FACTORS, LEAGUE_PROFILES

# Industry CRS baselines (median from 10000+ matches, neutral match)
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
    if sp_h <= 1 or sp_d <= 1 or sp_a <= 1:
        return {"dir": "H", "conf": "LOW", "gap": 0, "w": sp_h, "d": sp_d, "l": sp_a,
                "h_prob": 33, "d_prob": 33, "a_prob": 34}
    ih, id2, ia = 1/sp_h, 1/sp_d, 1/sp_a
    t = ih + id2 + ia
    hp, dp, ap = round(ih/t*100, 1), round(id2/t*100, 1), round(ia/t*100, 1)
    vals = sorted([("H", sp_h, hp), ("D", sp_d, dp), ("A", sp_a, ap)], key=lambda x: x[1])
    d = vals[0][0]
    gap = round((vals[1][1] - vals[0][1]) / max(vals[0][1], 0.01), 3)
    conf = "HIGH" if gap > 0.25 else "MID" if gap > 0.12 else "LOW"
    return {"dir": d, "conf": conf, "gap": gap, "w": sp_h, "d": sp_d, "l": sp_a,
            "h_prob": hp, "d_prob": dp, "a_prob": ap}


def predict_match(match_data):
    v2 = match_data.get("v2_odds_dict", {})
    if not v2: v2 = match_data
    sp_h = float(match_data.get("sp_home", match_data.get("win", 0)) or 0)
    sp_d = float(match_data.get("sp_draw", match_data.get("same", 0)) or 0)
    sp_a = float(match_data.get("sp_away", match_data.get("lose", 0)) or 0)

    crs = parse_crs_odds(v2)
    ttg = calc_ttg(v2)
    had = calc_had(sp_h, sp_d, sp_a)

    # Detect league for factor adjustment
    league = str(match_data.get("league", ""))
    lk = detect_league_key(league)
    factors = LEAGUE_FACTORS.get(lk, {})
    profile = LEAGUE_PROFILES.get(lk, LEAGUE_PROFILES.get("default", ()))
    league_avg_goals = profile[0] if profile else 2.5

    draw_boost = factors.get("draw_boost", 1.0)
    under_boost = factors.get("under_boost", 1.0)

    zero_odds = ttg["raw"].get(0, 99)
    seven_odds = ttg["raw"].get(7, 99)
    best_total = ttg["best"]
    exp_goals = ttg["exp"]

    # ================================================================
    # Score each CRS by: deviation + direction + total + league factors
    # ================================================================
    scored = {}
    for sc, info in crs.items():
        bl = BASELINE_CRS.get(sc, info["odds"])
        dev = (bl - info["odds"]) / bl  # positive = bookmaker favors this score
        base = dev * 100

        # Direction alignment
        # Exception: 0-0 with strong zero signal bypasses direction penalty
        is_zero_protected = (sc == "0-0" and zero_odds < 9.0)
        if info["dir"] == had["dir"]:
            if had["conf"] == "HIGH": base += 18
            elif had["conf"] == "MID": base += 10
            else: base += 4
        elif is_zero_protected:
            pass  # 0-0 with low zero odds: don't penalize direction mismatch
        elif had["conf"] == "HIGH":
            base -= 12
        # LOW confidence: no penalty for wrong direction

        # Total goals alignment
        if info["total"] == best_total:
            base += 12
        elif abs(info["total"] - best_total) == 1:
            base += 4
        elif abs(info["total"] - best_total) >= 3:
            base -= 12

        # --- LEAGUE FACTORS ---
        # Draw boost (e.g. Serie A 1.3x, Ligue 2 1.2x)
        if info["dir"] == "D":
            base *= draw_boost

        # Under boost for low-scoring leagues (e.g. Ligue 2 1.5x)
        if info["total"] <= 1:
            base *= under_boost
        elif info["total"] >= 4 and under_boost > 1.1:
            base *= (1.0 / under_boost)  # penalize high-scoring in low-scoring leagues

        # League avg goals adjustment:
        # If league avg < 2.2 (very defensive) boost 0-0 and 1-0/0-1
        if league_avg_goals < 2.2 and info["total"] <= 1:
            base += 8
        # If league avg > 3.0 (attacking) boost 3+ goal scores
        if league_avg_goals > 3.0 and info["total"] >= 3:
            base += 6

        # --- SPECIAL SIGNALS ---
        # 0-0: boost when 0-goal odds very low
        if sc == "0-0":
            if zero_odds < 8.0:
                base += 35
            elif zero_odds < 9.0:
                base += 20
            elif zero_odds < 10.0:
                base += 10

        # Super favorite: large score boost
        if sp_h < 1.25 and info["dir"] == "H" and info["total"] >= 3:
            base += 15
        if sp_a < 1.25 and info["dir"] == "A" and info["total"] >= 3:
            base += 15
        if sp_h < 1.50 and info["dir"] == "H" and info["total"] >= 2:
            base += 8
        if sp_a < 1.50 and info["dir"] == "A" and info["total"] >= 2:
            base += 8

        # High-scoring game signal
        if seven_odds < 15 and info["total"] >= 4:
            base += 12
        elif seven_odds < 20 and info["total"] >= 4:
            base += 6

        # 1-1 penalty when direction is clear (anti-1-1 bias)
        if sc == "1-1" and had["conf"] in ["HIGH", "MID"] and had["dir"] != "D":
            base -= 10

        # 0-0 penalty in attacking leagues when direction is clear
        if sc == "0-0" and league_avg_goals > 2.8 and had["conf"] == "HIGH" and zero_odds > 10:
            base -= 8

        scored[sc] = {
            "score": round(base, 1),
            "dev": round(dev * 100, 1),
            "prob": round(1/info["odds"]*100, 1),
            "odds": info["odds"],
            "dir": info["dir"],
            "total": info["total"],
        }

    ranked = sorted(scored.items(), key=lambda x: x[1]["score"], reverse=True)
    top3 = [r[0] for r in ranked[:3]]
    primary = top3[0] if top3 else "1-1"

    # Over/Under 2.5
    over_25 = round(sum(p for g, p in ttg["probs"].items() if g >= 3), 1) if ttg["probs"] else 50.0

    # BTTS
    btts_prob = 0; total_crs_prob = 0
    for sc, info in crs.items():
        p = 1 / info["odds"]
        total_crs_prob += p
        if info["h"] > 0 and info["a"] > 0: btts_prob += p
    btts = round(btts_prob / total_crs_prob * 100, 1) if total_crs_prob > 0 else 45.0

    # Confidence
    conf = 50
    if had["conf"] == "HIGH": conf += 12
    elif had["conf"] == "MID": conf += 5
    if ranked and ranked[0][1]["dev"] > 20: conf += 10
    elif ranked and ranked[0][1]["dev"] > 10: conf += 5
    if ranked and ranked[0][1]["dir"] == had["dir"]: conf += 8
    if primary == "0-0" and zero_odds < 8.5: conf += 10
    # League confidence: well-known patterns add confidence
    if lk in ["ita_top", "fra2"] and primary in ["0-0", "1-0", "0-1", "1-1"]:
        conf += 5  # defensive leagues, low scores more predictable
    conf = min(90, max(30, conf))

    # Value bets
    value_bets = []
    for label, prob, odds in [("\u4e3b\u80dc", had["h_prob"], sp_h), ("\u5e73\u5c40", had["d_prob"], sp_d), ("\u5ba2\u80dc", had["a_prob"], sp_a)]:
        if prob > 0 and odds > 1:
            ev = (prob/100 * odds) - 1
            if ev > 0.08:
                kelly = max(0, ((odds-1)*prob/100 - (1-prob/100))/(odds-1)) * 25
                value_bets.append({"type": label, "odds": odds, "ev": round(ev*100, 1), "kelly": round(kelly, 1)})

    reason = "dev%+.0f%% dir:%s(%s) ttg:%d lg:%s" % (
        ranked[0][1]["dev"] if ranked else 0, had["dir"], had["conf"][:1], best_total, lk)

    return {
        "primary_score": primary,
        "top3_scores": top3,
        "top5_detail": [(s, d["score"], d["odds"], d["dev"], d["dir"]) for s, d in ranked[:5]],
        "direction": had["dir"],
        "direction_confidence": had["conf"],
        "home_prob": had["h_prob"],
        "draw_prob": had["d_prob"],
        "away_prob": had["a_prob"],
        "expected_goals": exp_goals,
        "most_likely_goals": best_total,
        "goals_distribution": ttg["probs"],
        "over_25": over_25,
        "btts": btts,
        "confidence": conf,
        "value_bets": value_bets,
        "reason": reason,
        "zero_odds": zero_odds,
        "league_key": lk,
    }


def build_ai_context(match_data, prediction):
    lines = []
    lines.append("[ENGINE] Dir:%s(%s) Exp:%.1fg O2.5:%.0f%% BTTS:%.0f%%" % (
        prediction["direction"], prediction["direction_confidence"],
        prediction["expected_goals"], prediction["over_25"], prediction["btts"]))
    if prediction["zero_odds"] < 9:
        lines.append("[!] 0-goal @%.1f = strong 0-0 signal!" % prediction["zero_odds"])
    lines.append("[TOP5] %s" % ", ".join(["%s(dev%+.0f%%)" % (s[0], s[3]) for s in prediction["top5_detail"]]))
    lines.append("[PICK] %s" % ", ".join(prediction["top3_scores"]))
    if prediction["value_bets"]:
        lines.append("[VALUE] %s" % " | ".join(["%s@%.2f EV+%.1f%%" % (v["type"], v["odds"], v["ev"]) for v in prediction["value_bets"]]))
    return "\n".join(lines)
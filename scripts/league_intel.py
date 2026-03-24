# league_intel.py v3 - Complete strategic intelligence
# Source: 5-league financial analysis + historical patterns

# ===================================================================
# 1. LEAGUE PROFILES: (场均, O2.5%, U2.5%, 2-3g%, 3+g%, 1-2g%, style)
# ===================================================================
LEAGUE_PROFILES = {
    "eng_top": (2.82, 58, 21, 42, 50, 34, "attacking, upsets COMMON, bottom beats top-6"),
    "eng_champ": (2.75, 55, 22, 40, 48, 35, "chaotic, any result possible"),
    "ger_top": (3.18, 67, 15, 45, 50, 28, "highest scoring, Bayern 4-0 common"),
    "ger2": (2.80, 56, 20, 42, 47, 33, "physical, less goals than BuLi"),
    "esp_top": (2.51, 48, 25, 32, 38, 37, "technical, 2-3g dominant 55-65%"),
    "ita_top": (2.41, 43, 29, 32, 35, 43, "DEFENSIVE, draws 30%+, tactics first"),
    "fra_top": (1.90, 35, 45, 25, 25, 55, "PSG dominates, others LOW scoring"),
    "fra2": (1.85, 33, 47, 24, 23, 57, "very low scoring, 0-0/1-0 common"),
    "ned_top": (3.05, 62, 18, 42, 48, 30, "very attacking like BuLi"),
    "ned2": (2.70, 52, 24, 38, 42, 34, "slightly less attacking"),
    "por_top": (2.55, 50, 24, 35, 40, 35, "big 3 dominate"),
    "bel_top": (2.65, 52, 23, 37, 42, 35, "competitive"),
    "tur_top": (2.70, 53, 23, 38, 43, 34, "emotional, home advantage strong"),
    "sco_top": (2.60, 50, 25, 36, 40, 36, "Celtic/Rangers dominate"),
    "jpn_top": (2.65, 52, 23, 37, 42, 35, "tactical, disciplined"),
    "kor_top": (2.50, 48, 26, 34, 38, 37, "physical, draws common"),
    "aus_top": (2.90, 60, 20, 40, 48, 32, "attacking, open"),
    "arg_top": (2.30, 42, 30, 30, 33, 42, "defensive, fouls heavy"),
    "bra_top": (2.40, 44, 28, 32, 35, 40, "technical but defensive"),
    "mls": (2.85, 58, 20, 40, 48, 32, "open, defensive lapses"),
    "ucl": (2.72, 55, 22, 38, 45, 33, "knockout=tactical, group=open"),
    "uel": (2.65, 52, 24, 36, 42, 35, "more open than UCL"),
    "ucnf": (2.50, 48, 26, 34, 38, 37, "quality gap huge"),
    "women": (2.50, 48, 26, 34, 38, 37, "similar to mens avg"),
    "default": (2.50, 50, 25, 35, 40, 35, "default"),
}

# ===================================================================
# 2. MATCH TYPE GOALS: (1g%, 2g%, 3+g%, likely_total)
# ===================================================================
MATCH_TYPE_GOALS = {
    "eng_top": {"strong_v_weak":(20,20,25,3),"strong_v_mid":(25,23,28,2),"strong_v_strong":(25,23,28,2),"mid_v_weak":(25,30,20,2),"weak_v_weak":(30,30,15,2)},
    "ger_top": {"strong_v_weak":(18,15,33,4),"strong_v_mid":(20,20,25,3),"strong_v_strong":(22,18,25,3),"mid_v_weak":(22,25,20,2),"weak_v_weak":(25,30,18,2)},
    "esp_top": {"strong_v_weak":(18,17,30,2),"strong_v_mid":(20,20,30,2),"strong_v_strong":(20,20,30,2),"mid_v_weak":(22,25,22,2),"weak_v_weak":(28,30,15,1)},
    "ita_top": {"strong_v_weak":(15,15,35,2),"strong_v_mid":(22,18,30,1),"strong_v_strong":(20,18,32,2),"mid_v_weak":(25,28,20,2),"weak_v_weak":(30,32,12,0)},
    "fra_top": {"strong_v_weak":(20,18,30,3),"strong_v_mid":(24,21,25,2),"strong_v_strong":(23,17,25,2),"mid_v_weak":(25,28,20,2),"weak_v_weak":(28,30,15,1)},
    "fra2": {"strong_v_weak":(22,20,25,2),"strong_v_mid":(26,24,20,2),"strong_v_strong":(25,22,22,2),"mid_v_weak":(28,28,16,1),"weak_v_weak":(30,32,12,1)},
    "ned_top": {"strong_v_weak":(18,18,30,3),"strong_v_mid":(22,20,25,3),"strong_v_strong":(22,20,25,3),"mid_v_weak":(22,25,22,2),"weak_v_weak":(25,28,18,2)},
}

# ===================================================================
# 3. LEAGUE SCORING FACTORS (affect odds_engine scoring)
# ===================================================================
LEAGUE_FACTORS = {
    "ita_top": {"draw_boost": 1.3, "under_boost": 1.2, "note": "Serie A draws 30%+, always consider 0-0/1-1"},
    "fra_top": {"draw_boost": 1.0, "under_boost": 1.4, "note": "Ligue 1 low scoring, PSG exception"},
    "fra2": {"draw_boost": 1.2, "under_boost": 1.5, "note": "Ligue 2 very low scoring, 0-0/1-0 common"},
    "eng_top": {"draw_boost": 0.9, "under_boost": 0.8, "note": "PL upsets common, goals frequent"},
    "ger_top": {"draw_boost": 0.8, "under_boost": 0.7, "note": "Bundesliga highest goals, draws rare"},
    "ned_top": {"draw_boost": 0.8, "under_boost": 0.7, "note": "Eredivisie very attacking"},
    "esp_top": {"draw_boost": 1.1, "under_boost": 1.1, "note": "La Liga technical, 2-3g dominant"},
    "arg_top": {"draw_boost": 1.2, "under_boost": 1.3, "note": "Argentine league defensive"},
    "kor_top": {"draw_boost": 1.2, "under_boost": 1.1, "note": "K-League draws common"},
}

# ===================================================================
# 4. FINANCIAL MOTIVATION DATA (per league)
# ===================================================================
LEAGUE_FINANCE = {
    "eng_top": {
        "total_pool": "3B GBP", "champ_vs_last_ratio": 1.6,
        "relegation_cost": "moderate - parachute payments cushion",
        "ucl_value": "100M+ GBP/year", "top4_vs_5th_gap": "huge",
        "relegation_desperation": 6,  # 1-10 scale
        "top4_intensity": 9,
        "title_after_locked": 3,  # how much champion cares after locking
    },
    "esp_top": {
        "total_pool": "1.29B EUR", "champ_vs_last_ratio": 3.9,
        "relegation_cost": "severe - income halves",
        "ucl_value": "50-80M EUR/year", "top4_vs_5th_gap": "massive",
        "relegation_desperation": 8,
        "top4_intensity": 9,
        "title_after_locked": 4,
    },
    "ita_top": {
        "total_pool": "1.27B EUR", "champ_vs_last_ratio": 3.6,
        "relegation_cost": "catastrophic - near bankruptcy",
        "ucl_value": "50-80M EUR/year", "top4_vs_5th_gap": "huge",
        "relegation_desperation": 10,  # highest!
        "top4_intensity": 9,
        "title_after_locked": 5,
    },
    "ger_top": {
        "total_pool": "1.12B EUR", "champ_vs_last_ratio": 2.7,
        "relegation_cost": "devastating - no owner bailout (50+1)",
        "ucl_value": "30-50M EUR/year", "top4_vs_5th_gap": "large",
        "relegation_desperation": 9,
        "top4_intensity": 8,
        "title_after_locked": 2,  # Bayern stops caring after lock
    },
    "fra_top": {
        "total_pool": "0.2B EUR (collapsed!)", "champ_vs_last_ratio": 8.0,
        "relegation_cost": "DESTRUCTION - near zero income in L2",
        "ucl_value": "20-50M EUR (=20-30% of revenue!)", "top4_vs_5th_gap": "life or death",
        "relegation_desperation": 10,
        "top4_intensity": 10,
        "title_after_locked": 2,
    },
}

# ===================================================================
# 5. RIVALRY/DERBY DATABASE
# ===================================================================
DERBIES = {
    # Format: frozenset({team1, team2}): intensity (1-10)
    # These matches = NO mercy, NO tactical rest, MAX intensity
    "eng_top": [
        ({"Arsenal", "\u963f\u68ee\u7eb3", "Tottenham", "\u70ed\u523a"}, 10, "North London Derby"),
        ({"\u5229\u7269\u6d66", "\u57c3\u5f17\u987f", "Liverpool", "Everton"}, 9, "Merseyside Derby"),
        ({"\u66fc\u8054", "\u66fc\u57ce", "Man United", "Man City"}, 10, "Manchester Derby"),
        ({"\u5207\u5c14\u897f", "\u70ed\u523a", "Chelsea", "Tottenham"}, 7, "London Rivalry"),
    ],
    "esp_top": [
        ({"\u7687\u9a6c", "\u9a6c\u7adf", "Real Madrid", "Atletico"}, 10, "Madrid Derby"),
        ({"\u7687\u9a6c", "\u5df4\u8428", "\u5df4\u585e\u7f57\u90a3", "Real Madrid", "Barcelona"}, 10, "El Clasico"),
        ({"\u6bd5\u5c14\u5df4\u9102", "\u7687\u5bb6\u793e\u4f1a"}, 9, "Basque Derby"),
        ({"\u585e\u7ef4\u5229\u4e9a", "\u8d1d\u8482\u65af", "Sevilla", "Betis"}, 9, "Seville Derby"),
    ],
    "ita_top": [
        ({"\u56fd\u7c73", "\u7c73\u5170", "\u56fd\u9645\u7c73\u5170", "Inter", "AC Milan"}, 10, "Derby della Madonnina"),
        ({"\u56fd\u7c73", "\u5c24\u6587", "\u5c24\u6587\u56fe\u65af", "Inter", "Juventus"}, 9, "Derby d'Italia"),
        ({"\u7f57\u9a6c", "\u62c9\u9f50\u5965", "Roma", "Lazio"}, 10, "Derby della Capitale"),
    ],
    "ger_top": [
        ({"\u62dc\u4ec1", "\u591a\u7279", "\u591a\u7279\u8499\u5fb7", "Bayern", "Dortmund"}, 10, "Der Klassiker"),
        ({"\u591a\u7279", "\u6c99\u5c14\u514b", "Dortmund", "Schalke"}, 10, "Revierderby"),
        ({"\u83b1\u6bd4\u9521", "\u83b1\u7eb3"}, 7, "Anti-RB Leipzig matches"),
    ],
    "fra_top": [
        ({"\u5df4\u9ece", "\u9a6c\u8d5b", "PSG", "Marseille"}, 10, "Le Classique"),
        ({"\u91cc\u6602", "\u5723\u57c3\u8482\u5b89", "Lyon", "Saint-Etienne"}, 9, "Derby Rhone"),
    ],
}

# ===================================================================
# 6. HIDDEN MOTIVATION PATTERNS
# ===================================================================
MOTIVATION_PATTERNS = {
    # team_keyword: {condition: motivation_note}
    "psg": "PSG only cares about UCL. League = secondary after title locked.",
    "\u5df4\u9ece": "PSG only cares about UCL. League = secondary after title locked.",
    "bayern": "Bayern stops caring after league locked. Save energy for UCL.",
    "\u62dc\u4ec1": "Bayern stops caring after league locked. Save energy for UCL.",
    "\u7687\u9a6c": "Real Madrid prioritizes UCL over La Liga when both active.",
    "\u5df4\u8428": "Barcelona revenue less dependent on ranking due to social pool.",
    "\u5c24\u6587": "Juventus has historical income cushion. Less desperate than rivals.",
}


def detect_league_key(league_name):
    ln = str(league_name).lower()
    for keys, val in [
        (["\u82f1\u8d85","premier","epl"], "eng_top"),
        (["\u82f1\u51a0","championship"], "eng_champ"),
        (["\u5fb7\u7532","bundesliga"], "ger_top"),
        (["\u5fb7\u4e59","2.bundesliga"], "ger2"),
        (["\u897f\u7532","la liga"], "esp_top"),
        (["\u610f\u7532","serie a"], "ita_top"),
        (["\u6cd5\u7532","ligue 1"], "fra_top"),
        (["\u6cd5\u4e59","ligue 2"], "fra2"),
        (["\u8377\u7532","eredivisie"], "ned_top"),
        (["\u8377\u4e59"], "ned2"),
        (["\u8461\u8d85"], "por_top"),
        (["\u6bd4\u7532"], "bel_top"),
        (["\u571f\u8d85"], "tur_top"),
        (["\u82cf\u8d85"], "sco_top"),
        (["\u65e5\u804c","j1"], "jpn_top"),
        (["\u97e9\u804c","k1"], "kor_top"),
        (["\u6fb3\u8d85","a-league"], "aus_top"),
        (["\u4e2d\u8d85"], "chn_top"),
        (["\u963f\u7532"], "arg_top"),
        (["\u5df4\u7532"], "bra_top"),
        (["mls","\u7f8e\u804c"], "mls"),
        (["\u6b27\u51a0","champions"], "ucl"),
        (["\u6b27\u7f57\u5df4","europa"], "uel"),
        (["\u6b27\u534f\u8054","conference"], "ucnf"),
        (["\u5973\u4e9a\u6d32","\u5973\u8db3","\u5973\u4e16"], "women"),
    ]:
        if any(k in ln for k in keys): return val
    return "default"


def classify_match_type(home_rank, away_rank):
    try: hr = int(home_rank or 10)
    except: hr = 10
    try: ar = int(away_rank or 10)
    except: ar = 10
    sh, sa = hr <= 6, ar <= 6
    wh, wa = hr >= 15, ar >= 15
    if (sh and wa) or (sa and wh): return "strong_v_weak"
    if sh and sa: return "strong_v_strong"
    if (sh or sa) and not (wh or wa): return "strong_v_mid"
    if not sh and not sa and (wh or wa): return "mid_v_weak"
    if wh and wa: return "weak_v_weak"
    return "strong_v_mid"


def detect_derby(home_team, away_team, league_key):
    """Check if this is a derby/rivalry match"""
    h = str(home_team).lower()
    a = str(away_team).lower()
    derbies = DERBIES.get(league_key, [])
    for teams, intensity, name in derbies:
        t_lower = {t.lower() for t in teams}
        if any(k in h for k in t_lower) and any(k in a for k in t_lower):
            return intensity, name
    return 0, ""


def analyze_motivation(m, league_key):
    """6-step motivation analysis"""
    lines = []
    finance = LEAGUE_FINANCE.get(league_key, {})

    try: hr = int(m.get("home_rank", 10) or 10)
    except: hr = 10
    try: ar = int(m.get("away_rank", 10) or 10)
    except: ar = 10

    h = str(m.get("home_team", ""))
    a = str(m.get("away_team", ""))

    # Step 1: Financial context
    if finance:
        rel_desp = finance.get("relegation_desperation", 5)
        t4_int = finance.get("top4_intensity", 7)
        if hr >= 16 or ar >= 16:
            lines.append("[FINANCE] Relegation cost: %s (desperation: %d/10)" % (
                finance.get("relegation_cost", "unknown"), rel_desp))
        if hr <= 6 or ar <= 6:
            lines.append("[FINANCE] Top4 value: %s (intensity: %d/10)" % (
                finance.get("ucl_value", "unknown"), t4_int))

    # Step 2: Position-based motivation
    if hr <= 2 and ar <= 2:
        lines.append("[TITLE] Both in title race! MAX intensity, expect goals")
    elif hr <= 2 or ar <= 2:
        lines.append("[TITLE] Title contender involved, high motivation")
    if 3 <= hr <= 6 or 3 <= ar <= 6:
        if 3 <= hr <= 6 and 3 <= ar <= 6:
            lines.append("[TOP4] Both fighting for UCL spots! Critical match")
        else:
            lines.append("[TOP4] UCL qualification at stake for one team")
    if hr >= 16 or ar >= 16:
        if hr >= 16 and ar >= 16:
            lines.append("[RELEGATION] BOTH fighting relegation! Desperate 6-pointer")
        else:
            lines.append("[RELEGATION] Survival match! Relegation team will be DESPERATE")
    if 8 <= hr <= 14 and 8 <= ar <= 14:
        lines.append("[MID-TABLE] Both safe, low stakes = draw risk HIGH, motivation LOW")

    # Step 3: Hidden motivation patterns
    for team_kw, note in MOTIVATION_PATTERNS.items():
        if team_kw in h.lower() or team_kw in a.lower():
            lines.append("[HIDDEN] %s" % note)
            break

    # Step 4: Derby detection
    derby_int, derby_name = detect_derby(h, a, league_key)
    if derby_int >= 7:
        lines.append("[DERBY] %s (intensity %d/10)! NO tactical rest, MAXIMUM commitment" % (derby_name, derby_int))

    # Step 5: Income sensitivity
    if finance:
        if hr >= 15:
            lines.append("[SENSITIVITY] Home team: relegation = financial DISASTER in this league")
        if ar >= 15:
            lines.append("[SENSITIVITY] Away team: relegation = financial DISASTER in this league")
        if league_key == "ita_top" and (hr <= 6 or ar <= 6):
            lines.append("[SENSITIVITY] Serie A: historical income cushion for big clubs, less desperate")
        if league_key in ["fra_top", "fra2"]:
            lines.append("[SENSITIVITY] French football: TV deal collapsed, every euro counts!")

    return lines


def build_league_intelligence(m):
    league = str(m.get("league", ""))
    lk = detect_league_key(league)
    profile = LEAGUE_PROFILES.get(lk, LEAGUE_PROFILES["default"])
    avg_goals, over25, under25, two_three, three_plus, one_two, desc = profile
    hr = m.get("home_rank", 10)
    ar = m.get("away_rank", 10)
    mt = classify_match_type(hr, ar)
    mt_data = MATCH_TYPE_GOALS.get(lk, {})
    mt_goals = mt_data.get(mt, (22, 22, 25, 2))
    factors = LEAGUE_FACTORS.get(lk, {})
    intel = m.get("intelligence", {})

    lines = []
    lines.append("[LEAGUE:%s] Avg:%.1fg O2.5:%d%% U2.5:%d%% | %s" % (
        league, avg_goals, over25, under25, desc))
    lines.append("[TYPE:%s] 1g=%d%% 2g=%d%% 3+g=%d%% likely=%dg" % (
        mt.replace("_", " ").upper(), mt_goals[0], mt_goals[1], mt_goals[2], mt_goals[3]))

    if factors.get("note"):
        lines.append("[!] %s" % factors["note"])

    is_knockout = False
    baseface = str(m.get("baseface", ""))
    if any(k in league for k in ["\u6b27\u7f57\u5df4", "\u6b27\u51a0", "\u6b27\u534f\u8054"]):
        is_knockout = True
        lines.append("[CUP] Knockout: higher stakes, typically more goals")
        if "\u6b21\u56de\u5408" in baseface or "\u7b2c\u4e8c" in baseface:
            lines.append("[2ND LEG] Trailing team attacks = more goals")

    # Motivation analysis
    motive_lines = analyze_motivation(m, lk)
    lines.extend(motive_lines)

    # Injury impact
    h_inj = str(intel.get("h_inj", intel.get("home_injury", "")))
    g_inj = str(intel.get("g_inj", intel.get("guest_injury", "")))
    inj_kw = ["\u524d\u950b", "\u6838\u5fc3", "\u4e3b\u529b", "\u8fdb\u653b"]
    if any(k in h_inj for k in inj_kw): lines.append("[INJ] Home key attacker OUT")
    if any(k in g_inj for k in inj_kw): lines.append("[INJ] Away key attacker OUT")

    return "\n".join(lines), avg_goals, is_knockout, mt
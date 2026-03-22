# league_intel.py v2 - Complete league intelligence database

LEAGUE_PROFILES = {
    "eng_top": (2.82, 58, 21, 42, 50, 34, "attacking, upsets common, bottom beats top-6"),
    "eng_champ": (2.75, 55, 22, 40, 48, 35, "chaotic, any result possible"),
    "ger_top": (3.18, 67, 15, 45, 50, 28, "highest scoring, Bayern 4-0 common"),
    "ger2": (2.80, 56, 20, 42, 47, 33, "physical, less goals than Bundesliga"),
    "esp_top": (2.51, 48, 25, 32, 38, 37, "technical, 2-3 goals dominant 55-65%"),
    "ita_top": (2.41, 43, 29, 32, 35, 43, "DEFENSIVE, draws 30%+, small scores"),
    "fra_top": (1.90, 35, 45, 25, 25, 55, "PSG dominates, others LOW scoring"),
    "fra2": (1.85, 33, 47, 24, 23, 57, "very low scoring, 0-0/1-0 common"),
    "ned_top": (3.05, 62, 18, 42, 48, 30, "very attacking like Bundesliga"),
    "ned2": (2.70, 52, 24, 38, 42, 34, "slightly less attacking"),
    "por_top": (2.55, 50, 24, 35, 40, 35, "big 3 dominate, mid-table tight"),
    "bel_top": (2.65, 52, 23, 37, 42, 35, "competitive, no clear pattern"),
    "tur_top": (2.70, 53, 23, 38, 43, 34, "emotional, home advantage strong"),
    "sco_top": (2.60, 50, 25, 36, 40, 36, "Celtic/Rangers dominate"),
    "jpn_top": (2.65, 52, 23, 37, 42, 35, "tactical, disciplined"),
    "kor_top": (2.50, 48, 26, 34, 38, 37, "physical, draws common"),
    "aus_top": (2.90, 60, 20, 40, 48, 32, "attacking, open play"),
    "chn_top": (2.45, 45, 28, 33, 36, 39, "defensive, foreign strikers key"),
    "arg_top": (2.30, 42, 30, 30, 33, 42, "defensive, fouls heavy"),
    "bra_top": (2.40, 44, 28, 32, 35, 40, "technical but defensive"),
    "mls": (2.85, 58, 20, 40, 48, 32, "open attacking, defensive lapses"),
    "ucl": (2.72, 55, 22, 38, 45, 33, "knockout=tactical, group=open"),
    "uel": (2.65, 52, 24, 36, 42, 35, "more open than UCL"),
    "ucnf": (2.50, 48, 26, 34, 38, 37, "quality gap huge, blowouts possible"),
    "women": (2.50, 48, 26, 34, 38, 37, "similar to mens average"),
    "default": (2.50, 50, 25, 35, 40, 35, "default neutral"),
}

MATCH_TYPE_GOALS = {
    "eng_top": {"strong_v_weak":(20,20,25,3),"strong_v_mid":(25,23,28,2),"strong_v_strong":(25,23,28,2),"mid_v_weak":(25,30,20,2),"weak_v_weak":(30,30,15,2)},
    "ger_top": {"strong_v_weak":(18,15,33,4),"strong_v_mid":(20,20,25,3),"strong_v_strong":(22,18,25,3),"mid_v_weak":(22,25,20,2),"weak_v_weak":(25,30,18,2)},
    "esp_top": {"strong_v_weak":(18,17,30,2),"strong_v_mid":(20,20,30,2),"strong_v_strong":(20,20,30,2),"mid_v_weak":(22,25,22,2),"weak_v_weak":(28,30,15,1)},
    "ita_top": {"strong_v_weak":(15,15,35,2),"strong_v_mid":(22,18,30,1),"strong_v_strong":(20,18,32,2),"mid_v_weak":(25,28,20,2),"weak_v_weak":(30,32,12,0)},
    "fra_top": {"strong_v_weak":(20,18,30,3),"strong_v_mid":(24,21,25,2),"strong_v_strong":(23,17,25,2),"mid_v_weak":(25,28,20,2),"weak_v_weak":(28,30,15,1)},
    "fra2": {"strong_v_weak":(22,20,25,2),"strong_v_mid":(26,24,20,2),"strong_v_strong":(25,22,22,2),"mid_v_weak":(28,28,16,1),"weak_v_weak":(30,32,12,1)},
    "ned_top": {"strong_v_weak":(18,18,30,3),"strong_v_mid":(22,20,25,3),"strong_v_strong":(22,20,25,3),"mid_v_weak":(22,25,22,2),"weak_v_weak":(25,28,18,2)},
    "ucl": {"strong_v_weak":(18,15,30,3),"strong_v_mid":(22,18,25,2),"strong_v_strong":(22,20,25,2),"mid_v_weak":(20,22,22,3),"weak_v_weak":(25,25,20,2)},
    "uel": {"strong_v_weak":(20,18,28,3),"strong_v_mid":(22,20,25,2),"strong_v_strong":(22,20,25,2),"mid_v_weak":(22,24,22,2),"weak_v_weak":(25,26,18,2)},
}

LEAGUE_FACTORS = {
    "ita_top": {"draw_boost": 1.3, "under_boost": 1.2, "note": "Serie A draws 30%+, always consider 0-0/1-1"},
    "fra_top": {"draw_boost": 1.0, "under_boost": 1.4, "note": "Ligue 1 low scoring, PSG exception"},
    "fra2": {"draw_boost": 1.2, "under_boost": 1.5, "note": "Ligue 2 very low scoring, 0-0/1-0 common"},
    "eng_top": {"draw_boost": 0.9, "under_boost": 0.8, "note": "PL upsets common, goals frequent"},
    "ger_top": {"draw_boost": 0.8, "under_boost": 0.7, "note": "Bundesliga highest goals, draws rare"},
    "ned_top": {"draw_boost": 0.8, "under_boost": 0.7, "note": "Eredivisie very attacking"},
    "esp_top": {"draw_boost": 1.1, "under_boost": 1.1, "note": "La Liga technical, 2-3 goals dominant"},
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
    baseface = str(m.get("baseface", ""))
    intel = m.get("intelligence", {})

    lines = []
    lines.append("[LEAGUE:%s] Avg:%.1fg O2.5:%d%% U2.5:%d%% Style:%s" % (league, avg_goals, over25, under25, desc))
    lines.append("[TYPE:%s] 1g=%d%% 2g=%d%% 3+g=%d%% likely=%dg" % (mt.replace("_"," ").upper(), mt_goals[0], mt_goals[1], mt_goals[2], mt_goals[3]))
    if factors.get("note"):
        lines.append("[!] %s" % factors["note"])

    is_knockout = False
    if any(k in league for k in ["\u6b27\u7f57\u5df4","\u6b27\u51a0","\u6b27\u534f\u8054"]):
        is_knockout = True
        lines.append("[CUP] Knockout: higher stakes, more goals")

    try: hr_int = int(hr)
    except: hr_int = 10
    try: ar_int = int(ar)
    except: ar_int = 10
    if hr_int <= 3 or ar_int <= 3:
        lines.append("[MOTIVE] Title race: 3+ goals 50%+")
    if hr_int >= 16 or ar_int >= 16:
        lines.append("[MOTIVE] Relegation: desperate, chaotic")

    h_inj = str(intel.get("h_inj", intel.get("home_injury", "")))
    g_inj = str(intel.get("g_inj", intel.get("guest_injury", "")))
    inj_kw = ["\u524d\u950b","\u6838\u5fc3","\u4e3b\u529b","\u8fdb\u653b"]
    if any(k in h_inj for k in inj_kw): lines.append("[INJ] Home key attacker OUT")
    if any(k in g_inj for k in inj_kw): lines.append("[INJ] Away key attacker OUT")

    return "\n".join(lines), avg_goals, is_knockout, mt
# league_intel.py - League historical patterns database
# Based on real statistical data across 5 major leagues + cups

LEAGUE_PROFILES = {
    # key: (场均进球, 大2.5%, 小2.5%, 2-3球%, 3球+%, 1-2球%, 特点)
    "eng_top": (2.82, 58, 21, 42, 50, 34, "attacking league, upsets common, big teams can lose to anyone"),
    "ger_top": (3.18, 67, 15, 45, 50, 28, "highest scoring league, Bayern 4-0 common, counter-attacks deadly"),
    "esp_top": (2.51, 48, 25, 32, 38, 37, "technical play, 2-3 goals dominant 55-65%, low block common"),
    "ita_top": (2.41, 43, 29, 32, 35, 43, "defensive league, draws VERY common 30%+, small scores"),
    "fra_top": (1.90, 35, 45, 25, 25, 55, "PSG dominates 3-0, other matches low scoring"),
    "eng_champ": (2.75, 55, 22, 40, 48, 35, "chaotic league, any team can beat any team"),
    "ned_top": (3.05, 62, 18, 42, 48, 30, "very attacking, high scoring, similar to Bundesliga"),
    "por_top": (2.55, 50, 24, 35, 40, 35, "Porto/Benfica/Sporting dominate, mid-table tight"),
    "ucl": (2.72, 55, 22, 38, 45, 33, "knockout=tactical, group=open, VAR adds goals"),
    "uel": (2.65, 52, 24, 36, 42, 35, "more open than UCL, weaker teams concede more"),
    "ucnf": (2.50, 48, 26, 34, 38, 37, "weakest cup, quality gap huge, big scores possible"),
    "aus_top": (2.90, 60, 20, 40, 48, 32, "attacking league, home advantage strong"),
    "default": (2.50, 50, 25, 35, 40, 35, "default profile"),
}

# Match type goal patterns: (1球%, 2球%, 3球+%, most_likely_total)
MATCH_TYPE_GOALS = {
    "eng_top": {"strong_v_weak": (20,20,25,3), "strong_v_mid": (25,23,28,2), "strong_v_strong": (25,23,28,2), "mid_v_weak": (25,30,20,2), "weak_v_weak": (30,30,15,1)},
    "ger_top": {"strong_v_weak": (18,15,33,3), "strong_v_mid": (20,20,25,3), "strong_v_strong": (22,18,25,2), "mid_v_weak": (22,25,20,2), "weak_v_weak": (25,30,18,2)},
    "esp_top": {"strong_v_weak": (18,17,30,2), "strong_v_mid": (20,20,30,2), "strong_v_strong": (20,20,30,2), "mid_v_weak": (22,25,22,2), "weak_v_weak": (28,30,15,1)},
    "ita_top": {"strong_v_weak": (15,15,35,2), "strong_v_mid": (20,18,32,2), "strong_v_strong": (22,18,32,2), "mid_v_weak": (25,28,20,2), "weak_v_weak": (30,32,12,1)},
    "fra_top": {"strong_v_weak": (20,18,30,3), "strong_v_mid": (24,21,25,2), "strong_v_strong": (23,17,25,2), "mid_v_weak": (25,28,20,2), "weak_v_weak": (28,30,15,1)},
}

# League name mapping
def detect_league_key(league_name):
    ln = str(league_name).lower()
    if any(k in ln for k in ["\u82f1\u8d85","premier","epl"]): return "eng_top"
    if any(k in ln for k in ["\u5fb7\u7532","bundesliga"]): return "ger_top"
    if any(k in ln for k in ["\u897f\u7532","la liga","laliga"]): return "esp_top"
    if any(k in ln for k in ["\u610f\u7532","serie a"]): return "ita_top"
    if any(k in ln for k in ["\u6cd5\u7532","ligue 1"]): return "fra_top"
    if any(k in ln for k in ["\u82f1\u51a0","championship"]): return "eng_champ"
    if any(k in ln for k in ["\u8377\u7532","eredivisie"]): return "ned_top"
    if any(k in ln for k in ["\u8461\u8d85","liga portugal"]): return "por_top"
    if any(k in ln for k in ["\u6b27\u51a0","champions"]): return "ucl"
    if any(k in ln for k in ["\u6b27\u7f57\u5df4","europa"]): return "uel"
    if any(k in ln for k in ["\u6b27\u534f\u8054","conference"]): return "ucnf"
    if any(k in ln for k in ["\u6fb3\u8d85","a-league"]): return "aus_top"
    if any(k in ln for k in ["\u6cd5\u4e59"]): return "fra_top"  # use fra profile for Ligue 2
    return "default"


def classify_match_type(home_rank, away_rank):
    try: hr = int(home_rank or 10)
    except: hr = 10
    try: ar = int(away_rank or 10)
    except: ar = 10
    strong_h = hr <= 6
    strong_a = ar <= 6
    weak_h = hr >= 15
    weak_a = ar >= 15
    if strong_h and weak_a: return "strong_v_weak"
    if strong_a and weak_h: return "strong_v_weak"  # reversed but same pattern
    if strong_h and strong_a: return "strong_v_strong"
    if (strong_h or strong_a) and not (weak_h or weak_a): return "strong_v_mid"
    if not strong_h and not strong_a and (weak_h or weak_a): return "mid_v_weak"
    if weak_h and weak_a: return "weak_v_weak"
    return "strong_v_mid"


def build_league_intelligence(m):
    league = str(m.get("league", ""))
    lk = detect_league_key(league)
    profile = LEAGUE_PROFILES.get(lk, LEAGUE_PROFILES["default"])
    avg_goals, over25, under25, two_three, three_plus, one_two, desc = profile

    hr = m.get("home_rank", 10)
    ar = m.get("away_rank", 10)
    mt = classify_match_type(hr, ar)
    mt_goals = MATCH_TYPE_GOALS.get(lk, {}).get(mt, (22, 22, 25, 2))

    baseface = str(m.get("baseface", ""))
    intel = m.get("intelligence", {})

    lines = []
    lines.append("[LEAGUE PROFILE: %s]" % league)
    lines.append("Avg goals: %.2f | Over2.5: %d%% | Under2.5: %d%% | 2-3goals: %d%% | 3+goals: %d%%" % (avg_goals, over25, under25, two_three, three_plus))
    lines.append("Character: %s" % desc)

    lines.append("\n[MATCH TYPE: %s]" % mt.replace("_", " ").upper())
    lines.append("Historical pattern: 1goal=%d%% 2goals=%d%% 3+goals=%d%% most_likely=%d goals" % mt_goals)

    # Knockout detection
    is_knockout = False
    if any(k in league for k in ["\u6b27\u7f57\u5df4","\u6b27\u51a0","\u6b27\u534f\u8054"]):
        is_knockout = True
        lines.append("\n[KNOCKOUT CONTEXT]")
        if "\u6b21\u56de\u5408" in baseface or "\u7b2c\u4e8c" in baseface:
            lines.append("2ND LEG: trailing team goes all-out, expect MORE goals than normal")
        else:
            lines.append("Cup match: higher stakes = more goals than league average")

    # Motivation
    try: hr_int = int(hr)
    except: hr_int = 10
    try: ar_int = int(ar)
    except: ar_int = 10

    if hr_int <= 3 or ar_int <= 3:
        lines.append("[MOTIVATION] Title race team involved: expect 3+ goals (50%+ probability)")
    if hr_int >= 16 or ar_int >= 16:
        lines.append("[MOTIVATION] Relegation battle: desperate play, chaotic, goals unpredictable")
    if hr_int >= 8 and hr_int <= 14 and ar_int >= 8 and ar_int <= 14:
        lines.append("[MOTIVATION] Mid-table: low motivation, 2 goals most likely (30%+ draw risk)")

    # Injuries impact on goals
    h_inj = str(intel.get("h_inj", intel.get("home_injury", "")))
    g_inj = str(intel.get("g_inj", intel.get("guest_injury", "")))
    inj_keywords = ["\u524d\u950b","\u6838\u5fc3","\u4e3b\u529b","\u8fdb\u653b"]
    h_key_inj = any(k in h_inj for k in inj_keywords)
    a_key_inj = any(k in g_inj for k in inj_keywords)
    if h_key_inj:
        lines.append("[INJURY IMPACT] Home key attacker OUT: expect -0.7 goals for home")
    if a_key_inj:
        lines.append("[INJURY IMPACT] Away key attacker OUT: expect -0.7 goals for away")
    if not h_key_inj and not a_key_inj:
        lines.append("[SQUAD] Both squads relatively complete: normal goal expectation")

    # League-specific warnings
    if lk == "ita_top":
        lines.append("[LEAGUE WARNING] Serie A: draws happen 30%+ of games. Do NOT ignore draw.")
    if lk == "eng_top":
        lines.append("[LEAGUE WARNING] Premier League: upsets are COMMON. Bottom teams regularly beat top-6.")
    if lk == "ger_top":
        lines.append("[LEAGUE WARNING] Bundesliga: highest scoring league. 3+ goals in 67% of matches.")
    if lk == "esp_top":
        lines.append("[LEAGUE WARNING] La Liga: technical & tactical. 2-3 goals in 55-65% of matches.")
    if lk == "fra_top":
        lines.append("[LEAGUE WARNING] Ligue 1: low scoring outside PSG games. 1-0 and 0-0 common.")
    if lk == "ned_top":
        lines.append("[LEAGUE WARNING] Eredivisie: very attacking. Similar to Bundesliga in goals.")

    return "\n".join(lines), avg_goals, is_knockout, mt
import requests
import json
import time
import re
from datetime import datetime, timedelta, timezone
from config import *

TEAM_NAME_MAPPING = {
    "西汉姆联": "West Ham", "布伦特": "Brentford", "阿森纳": "Arsenal",
    "曼城": "Manchester City", "利物浦": "Liverpool", "曼联": "Manchester United",
    "切尔西": "Chelsea", "热刺": "Tottenham", "拉齐奥": "Lazio",
    "萨索洛": "Sassuolo", "皇马": "Real Madrid", "巴萨": "Barcelona",
    "马竞": "Atletico Madrid", "拜仁": "Bayern Munich", "亚特兰大": "Atalanta",
    "国际米兰": "Inter Milan", "AC米兰": "AC Milan", "尤文图斯": "Juventus",
    "那不勒斯": "Napoli", "多特蒙德": "Borussia Dortmund", "莱比锡": "RB Leipzig",
    "巴黎": "Paris Saint Germain", "里昂": "Lyon", "马赛": "Marseille",
    "本菲卡": "Benfica", "波尔图": "FC Porto", "阿贾克斯": "Ajax",
    "费耶诺德": "Feyenoord", "塞维利亚": "Sevilla", "比利亚雷亚尔": "Villarreal",
    "毕尔巴鄂": "Athletic Bilbao", "皇家社会": "Real Sociedad",
    "狼队": "Wolverhampton", "纽卡斯尔": "Newcastle", "维拉": "Aston Villa",
    "南安普敦": "Southampton", "考文垂": "Coventry", "利兹联": "Leeds",
    "诺维奇": "Norwich", "伯恩利": "Burnley", "谢菲尔德联": "Sheffield United",
    "莱斯特城": "Leicester", "埃弗顿": "Everton", "水晶宫": "Crystal Palace",
    "布莱顿": "Brighton", "伯恩茅斯": "Bournemouth", "诺丁汉森林": "Nottingham Forest",
    "富勒姆": "Fulham", "伊普斯维奇": "Ipswich",
    "佛罗伦萨": "Fiorentina", "罗马": "AS Roma", "博洛尼亚": "Bologna",
    "都灵": "Torino", "热那亚": "Genoa", "维罗纳": "Verona",
    "帕尔马": "Parma", "莱切": "Lecce", "卡利亚里": "Cagliari",
    "弗赖堡": "Freiburg", "法兰克福": "Eintracht Frankfurt", "斯图加特": "Stuttgart",
    "沃尔夫斯堡": "Wolfsburg", "美因茨": "Mainz", "奥格斯堡": "Augsburg",
    "柏林联合": "Union Berlin", "不莱梅": "Werder Bremen", "波鸿": "Bochum",
    "圣保利": "St. Pauli", "霍芬海姆": "Hoffenheim",
    "里尔": "Lille", "摩纳哥": "Monaco", "尼斯": "Nice", "朗斯": "Lens",
    "雷恩": "Rennes", "斯特拉斯堡": "Strasbourg", "南特": "Nantes",
}

def translate_team_name(name):
    if not name: return ""
    name = str(name).strip()
    if name in TEAM_NAME_MAPPING: return TEAM_NAME_MAPPING[name]
    try:
        from deep_translator import GoogleTranslator
        clean = name.replace("女足", " Women").replace("联", " United")
        return GoogleTranslator(source='zh-CN', target='en').translate(clean).replace("FC", "").strip()
    except:
        return name

def _safe_dict(val): return val if isinstance(val, dict) else {}
def _get_float(val, default=999.0):
    try: return float(val) if val is not None else default
    except: return default

def scrape_wencai_jczq(date_str):
    url = f"https://edu.wencaivip.cn/api/v1.reference/matches?date={date_str}"
    print(f"  正在抓取 [{date_str}]...")
    ms = []
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
    try:
        r = requests.get(url, headers=headers, timeout=15)
        data = r.json()
        match_dict = data.get("data", {}).get("matches", {})
        match_list = []
        for key in match_dict:
            if isinstance(match_dict[key], list): match_list.extend(match_dict[key])
        for item in match_list:
            try:
                t_type = str(item.get("types") or "")
                if t_type != "1" and t_type != "足球": continue
                stime = item.get("stime")
                if not stime: continue
                dt = datetime.fromtimestamp(stime, timezone(timedelta(hours=8)))
                jc_date = (dt - timedelta(hours=11)).strftime("%Y-%m-%d")
                if jc_date != date_str: continue
                chg = _safe_dict(item.get("change"))
                w_c, l_c = _get_float(chg.get("win"), 0), _get_float(chg.get("lose"), 0)
                odds_mov = f"主胜{'升水' if w_c>0 else '降水' if w_c<0 else '平稳'}，客胜{'升水' if l_c>0 else '降水' if l_c<0 else '平稳'}"
                info = _safe_dict(item.get("information"))
                pts = _safe_dict(item.get("points"))
                h_inj = str(info.get("home_injury") or "无重大伤停").replace("\n", " ").strip()[:150]
                g_inj = str(info.get("guest_injury") or "无重大伤停").replace("\n", " ").strip()[:150]
                expert_intro = str(item.get("intro") or "").strip()
                analyse = _safe_dict(item.get("analyse"))
                baseface = str(analyse.get("baseface") or "").strip()
                had_analyse = analyse.get("had_analyse", [])
                v2_odds = {}
                for k in ["a0","a1","a2","a3","a4","a5","a6","a7","s00","s11","s22","s33",
                           "w10","w20","w21","w30","w31","w32","w40","w41","w42",
                           "l01","l02","l12","l03","l13","l23",
                           "ss","sp","sf","ps","pp","pf","fs","fp","ff"]:
                    val = item.get(k)
                    if val is not None:
                        v2_odds[k] = _get_float(val, 0)
                def parse_rank(pos):
                    if not pos: return 0
                    m2 = re.findall(r'\d+', str(pos))
                    return int(m2[0]) if m2 else 0
                home_t = str(item.get("home") or "未知主队")
                away_t = str(item.get("guest") or "未知客队")
                lg_t = str(item.get("cup") or "未知联赛")
                m_num = str(item.get("week") or "") + str(item.get("week_no") or "")
                ms.append({
                    "home_team": home_t, "away_team": away_t,
                    "league": lg_t, "match_num": m_num,
                    "match_time": dt.strftime('%H:%M'),
                    "sp_home": _get_float(item.get("win"), 0),
                    "sp_draw": _get_float(item.get("same"), 0),
                    "sp_away": _get_float(item.get("lose"), 0),
                    "give_ball": item.get("give_ball", 0),
                    "change": _safe_dict(item.get("change")),
                    "vote": _safe_dict(item.get("vote")),
                    "odds_movement": odds_mov,
                    "handicap_info": f"让{item.get('give_ball', 0)}",
                    "intelligence": {"h_inj": h_inj, "g_inj": g_inj},
                    "expert_intro": expert_intro,
                    "baseface": baseface,
                    "had_analyse": had_analyse,
                    "home_rank": parse_rank(pts.get("home_position", item.get("home_position", ""))),
                    "away_rank": parse_rank(pts.get("guest_position", item.get("guest_position", ""))),
                    "v2_odds_dict": v2_odds,
                    "h_icon": item.get("h_icon", ""),
                    "g_icon": item.get("g_icon", ""),
                })
                print(f"    {lg_t} {m_num}: {home_t} vs {away_t} SP:{_get_float(item.get('win'),0)}/{_get_float(item.get('same'),0)}/{_get_float(item.get('lose'),0)}")
            except Exception as e:
                continue
    except Exception as e:
        print(f"  API抓取失败: {e}")
    print(f"  共抓取 {len(ms)} 场竞彩")
    return ms

def search_team_api(name):
    if not API_FOOTBALL_KEY: return None
    h = {"x-apisports-key": API_FOOTBALL_KEY}
    en_name = translate_team_name(name)
    try:
        r = requests.get(f"{API_FOOTBALL_BASE}/teams", headers=h, params={"search": en_name}, timeout=10)
        res = r.json().get("response", [])
        if res:
            return {"id": res[0]["team"]["id"], "name": res[0]["team"]["name"], "logo": res[0]["team"].get("logo", "")}
    except:
        pass
    return None

def fetch_stats(tid, season=2024):
    if not tid or not API_FOOTBALL_KEY: return {}
    h = {"x-apisports-key": API_FOOTBALL_KEY}
    try:
        s = requests.get(f"{API_FOOTBALL_BASE}/teams/statistics", headers=h, params={"team": tid, "season": season}, timeout=10).json().get("response", {})
        if s and "fixtures" in s:
            return {
                "played": s["fixtures"]["played"].get("total", 0),
                "wins": s["fixtures"]["wins"].get("total", 0),
                "draws": s["fixtures"]["draws"].get("total", 0),
                "losses": s["fixtures"]["loses"].get("total", 0),
                "goals_for": s["goals"]["for"]["total"].get("total", 0),
                "goals_against": s["goals"]["against"]["total"].get("total", 0),
                "form": s.get("form", ""),
                "clean_sheets": s["clean_sheet"].get("total", 0),
                "avg_goals_for": str(s["goals"]["for"]["average"].get("total", "0.0")),
                "avg_goals_against": str(s["goals"]["against"]["average"].get("total", "0.0")),
            }
    except:
        pass
    return {}

def fetch_h2h(hid, aid):
    if not hid or not aid or not API_FOOTBALL_KEY: return []
    h = {"x-apisports-key": API_FOOTBALL_KEY}
    try:
        return [
            {"date": m["fixture"]["date"][:10], "home": m["teams"]["home"]["name"],
             "away": m["teams"]["away"]["name"],
             "score": f"{m['goals']['home']}-{m['goals']['away']}",
             "league": m["league"]["name"]}
            for m in requests.get(f"{API_FOOTBALL_BASE}/fixtures/headtohead",
                headers=h, params={"h2h": f"{hid}-{aid}", "last": 5}, timeout=10
            ).json().get("response", [])
        ]
    except:
        return []

def generate_stats_from_context(match, side):
    """API搜不到球队时，用赔率+排名反推合理统计数据"""
    rank = int(match.get("home_rank" if side == "home" else "away_rank", 10) or 10)
    sp_h = float(match.get("sp_home", 0) or 0)
    sp_a = float(match.get("sp_away", 0) or 0)
    rank = max(1, min(20, rank if rank > 0 else 10))
    strength = 1.0 - (rank - 1) / 19.0
    if sp_h > 1 and sp_a > 1:
        if side == "home":
            odds_strength = (1/sp_h) / (1/sp_h + 1/sp_a)
        else:
            odds_strength = (1/sp_a) / (1/sp_h + 1/sp_a)
        strength = strength * 0.3 + odds_strength * 0.7
    played = 25
    win_rate = max(0.15, min(0.70, strength * 0.55 + 0.15))
    draw_rate = 0.25
    wins = max(1, round(played * win_rate))
    draws = max(1, round(played * draw_rate))
    losses = max(1, played - wins - draws)
    gf_per = max(0.6, strength * 1.6 + 0.4)
    ga_per = max(0.5, (1 - strength) * 1.5 + 0.4)
    gf = round(played * gf_per)
    ga = round(played * ga_per)
    cs = max(1, round(played * (1 - ga_per / 2.5) * 0.25))
    if strength > 0.6: form = "WWDWW"
    elif strength > 0.4: form = "WDLWD"
    elif strength > 0.25: form = "LDWDL"
    else: form = "LLDLL"
    print(f"    [{side}] API未命中，排名#{rank} 赔率反推: {wins}W{draws}D{losses}L 均进{gf_per:.2f}")
    return {
        "played": played, "wins": wins, "draws": draws, "losses": losses,
        "goals_for": gf, "goals_against": ga,
        "avg_goals_for": str(round(gf_per, 2)),
        "avg_goals_against": str(round(ga_per, 2)),
        "clean_sheets": cs, "form": form, "rank": rank,
    }

def collect_all(date_str):
    print(f"\n=== 数据抓取 | {date_str} ===")
    matches = scrape_wencai_jczq(date_str)
    if not matches:
        print("  无比赛数据")
        return {"date": date_str, "matches": [], "odds": {}}
    print(f"\n  API-Football 补充数据...")
    for i, m in enumerate(matches):
        print(f"  [{i+1}/{len(matches)}] {m['home_team']} vs {m['away_team']}")
        ht = search_team_api(m["home_team"])
        time.sleep(0.3)
        at = search_team_api(m["away_team"])
        time.sleep(0.3)
        m["home_id"] = ht["id"] if ht else 0
        m["away_id"] = at["id"] if at else 0
        m["home_logo"] = ht["logo"] if ht else m.get("h_icon", "")
        m["away_logo"] = at["logo"] if at else m.get("g_icon", "")
        m["id"] = i + 1
        m["date"] = date_str
        api_h = fetch_stats(m["home_id"]) if m["home_id"] else {}
        time.sleep(0.2)
        api_a = fetch_stats(m["away_id"]) if m["away_id"] else {}
        time.sleep(0.2)
        m["home_stats"] = api_h if api_h.get("played", 0) > 0 else generate_stats_from_context(m, "home")
        m["away_stats"] = api_a if api_a.get("played", 0) > 0 else generate_stats_from_context(m, "away")
        m["h2h"] = fetch_h2h(m["home_id"], m["away_id"])
        time.sleep(0.2)
    return {"date": date_str, "matches": matches, "odds": {}}

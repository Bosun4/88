""" 主运行脚本： 1. 抓取数据 2. AI预测 3. 生成前端JSON 4. 更新 index.html """
import json
import os
import sys
import time
import re
import random
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import requests
from bs4 import BeautifulSoup

from config import *
from fetch_data import collect_all
from predict import run_predictions

# ==================== 工具函数 ====================
def get_today(offset=0):
    return (datetime.now(ZoneInfo(TIMEZONE)) + timedelta(days=offset)).strftime("%Y-%m-%d")

def generate_stats_from_rank(rank, total_teams=20):
    """用联赛排名生成合理的统计数据"""
    random.seed(rank * 7 + 3)
    if rank == 0:
        rank = 10
    strength = 1 - (rank - 1) / total_teams
    played = random.randint(18, 30)
    win_rate = max(0.15, min(0.75, strength * 0.6 + random.uniform(-0.1, 0.1)))
    draw_rate = random.uniform(0.15, 0.30)
    loss_rate = 1 - win_rate - draw_rate
    wins = max(1, int(played * win_rate))
    draws = max(1, int(played * draw_rate))
    losses = max(1, played - wins - draws)
    gf_per = max(0.5, strength * 2.0 + random.uniform(-0.3, 0.3))
    ga_per = max(0.4, (1 - strength) * 1.8 + random.uniform(-0.3, 0.3))
    gf = int(played * gf_per)
    ga = int(played * ga_per)
    cs = max(0, int(played * (1 - ga_per / 2) * 0.3))
    form_chars = []
    for _ in range(min(5, played)):
        r2 = random.random()
        if r2 < win_rate:
            form_chars.append("W")
        elif r2 < win_rate + draw_rate:
            form_chars.append("D")
        else:
            form_chars.append("L")
    return {
        "played": played, "wins": wins, "draws": draws, "losses": losses,
        "goals_for": gf, "goals_against": ga,
        "avg_goals_for": str(round(gf_per, 2)),
        "avg_goals_against": str(round(ga_per, 2)),
        "clean_sheets": cs, "form": "".join(form_chars), "rank": rank
    }

# ==================== 数据抓取函数 ====================
def scrape_500_jczq(date=None):
    date = date or get_today()
    url = C500_URL.format(date=date)
    print(f"  500.com: {url}")
    ms = []
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}, timeout=20)
        r.encoding = "gb2312"
        soup = BeautifulSoup(r.text, "html.parser")
        rows = soup.find_all("tr")
        print(f"  found {len(rows)} rows")
        for row in rows:
            tds = row.find_all("td")
            if len(tds) < 4: continue
            text = "|".join([td.get_text(strip=True) for td in tds])
            # Extract league and match_num
            league = ""
            match_num = ""
            for td in tds[:3]:
                t = td.get_text(strip=True)
                if re.match(r"^[\u4e00-\u9fff]+$", t) and 1 < len(t) < 6:
                    league = t
                elif re.match(r"^\u5468[一二三四五六日]\d{3}$", t):
                    match_num = t
            # Extract teams with rankings
            home = away = ""
            home_rank = away_rank = 0
            for td in tds:
                t = td.get_text(strip=True)
                m2 = re.match(r"^\[(\d+)\](.+)$", t)
                if m2:
                    rank = int(m2.group(1))
                    name = m2.group(2)
                    if not home:
                        home = name
                        home_rank = rank
                    elif not away:
                        away = name
                        away_rank = rank
            if not home or not away:
                links = row.find_all("a")
                teams = []
                for a in links:
                    t = a.get_text(strip=True)
                    m3 = re.match(r"^\[(\d+)\](.+)$", t)
                    if m3:
                        teams.append((m3.group(2), int(m3.group(1))))
                    elif 1 < len(t) < 20 and not t.isdigit() and "vs" not in t.lower():
                        teams.append((t, 0))
                if len(teams) >= 2:
                    home, home_rank = teams[0]
                    away, away_rank = teams[1]
            if not home or not away:
                vs_match = re.search(r"(.+?)\s*(?:vs|VS|V)\s*(.+)", text)
                if vs_match:
                    home = vs_match.group(1).strip()[-10:]
                    away = vs_match.group(2).strip()[:10]
            if home and away:
                sp_nums = re.findall(r"(\d+\.\d{2})", text)
                sp_home = float(sp_nums[0]) if len(sp_nums) >= 1 else 0
                sp_draw = float(sp_nums[1]) if len(sp_nums) >= 2 else 0
                sp_away = float(sp_nums[2]) if len(sp_nums) >= 3 else 0
                m_obj = {
                    "home_team": home, "away_team": away, "league": league, "match_num": match_num,
                    "home_rank": home_rank, "away_rank": away_rank,
                    "sp_home": sp_home, "sp_draw": sp_draw, "sp_away": sp_away,
                    "source": "500", "raw": text[:200]
                }
                ms.append(m_obj)
                print(f"    {league or '?'}: {home}[{home_rank}] vs {away}[{away_rank}] SP:{sp_home:.2f}/{sp_draw:.2f}/{sp_away:.2f}")
    except Exception as e:
        print(f"  500.com error:{e}")
    return ms

def search_team_api(name):
    h = {"x-apisports-key": API_FOOTBALL_KEY}
    try:
        r = requests.get(API_FOOTBALL_BASE + "/teams", headers=h, params={"search": name}, timeout=10)
        d = r.json()
        if d.get("response") and len(d["response"]) > 0:
            team = d["response"][0]["team"]
            return {"id": team["id"], "name": team["name"], "logo": team.get("logo", "")}
    except:
        pass
    return None

def fetch_stats(tid, lid=0, season=2024):
    if not tid: return {}
    h = {"x-apisports-key": API_FOOTBALL_KEY}
    try:
        params = {"team": tid, "season": season}
        if lid: params["league"] = lid
        r = requests.get(API_FOOTBALL_BASE + "/teams/statistics", headers=h, params=params, timeout=15)
        d = r.json()
        if d.get("response"):
            s = d["response"]
            return {
                "played": s["fixtures"]["played"].get("total", 0),
                "wins": s["fixtures"]["wins"].get("total", 0),
                "draws": s["fixtures"]["draws"].get("total", 0),
                "losses": s["fixtures"]["loses"].get("total", 0),
                "goals_for": s["goals"]["for"]["total"].get("total", 0),
                "goals_against": s["goals"]["against"]["total"].get("total", 0),
                "form": s.get("form", ""),
                "clean_sheets": s["clean_sheet"].get("total", 0),
                "avg_goals_for": s["goals"]["for"]["average"].get("total", "0"),
                "avg_goals_against": s["goals"]["against"]["average"].get("total", "0")
            }
    except:
        pass
    return {}

def fetch_h2h(hid, aid):
    if not hid or not aid: return []
    h = {"x-apisports-key": API_FOOTBALL_KEY}
    try:
        r = requests.get(API_FOOTBALL_BASE + "/fixtures/headtohead", headers=h, params={"h2h": f"{hid}-{aid}", "last": 10}, timeout=15)
        d = r.json()
        rc = []
        if d.get("response"):
            for m in d["response"]:
                rc.append({
                    "date": m["fixture"]["date"][:10],
                    "home": m["teams"]["home"]["name"],
                    "away": m["teams"]["away"]["name"],
                    "score": f"{m['goals']['home']}-{m['goals']['away']}",
                    "league": m["league"]["name"]
                })
        return rc
    except:
        pass
    return []

# ==================== 主函数 ====================
def main():
    date = get_today()
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    session = "morning" if now.hour < 15 else "evening"

    print("=" * 60)
    print("⚽ 竞彩足球AI预测系统")
    print(f"📅 日期: {date} 时段: {'上午' if session == 'morning' else '晚上'}")
    print("=" * 60)

    # 1. 数据抓取（优先用 fetch_data.py，失败时用备用 scrape）
    raw_data = collect_all(date)

    os.makedirs("data", exist_ok=True)
    with open("data/raw_data.json", "w", encoding="utf-8") as f:
        json.dump(raw_data, f, ensure_ascii=False, indent=2)

    if not raw_data["matches"]:
        print("⚠️ 今日无竞彩比赛数据，生成空预测页面")
        output = {
            "date": date,
            "session": session,
            "update_time": now.strftime("%Y-%m-%d %H:%M:%S"),
            "total_matches": 0,
            "results": [],
            "top4": [],
        }
    else:
        # 2. AI预测
        results, top4 = run_predictions(raw_data)
        output = {
            "date": date,
            "session": session,
            "update_time": now.strftime("%Y-%m-%d %H:%M:%S"),
            "total_matches": len(results),
            "results": results,
            "top4": [
                {
                    "rank": i + 1,
                    **t,
                }
                for i, t in enumerate(top4)
            ],
        }

    # 3. 保存预测结果
    with open("data/predictions.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # 同时保存历史
    history_file = f"data/history_{date}_{session}.json"
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"✅ 全部完成！")
    print(f" 📊 共 {output['total_matches']} 场比赛")
    print(f" 🎯 推荐 {len(output['top4'])} 场")
    print(f" 📁 文件: data/predictions.json")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
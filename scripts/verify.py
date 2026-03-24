import json
import os
import requests
import time
import re
from datetime import datetime, timedelta
from config import *

def get_yesterday():
    from zoneinfo import ZoneInfo
    return (datetime.now(ZoneInfo(TIMEZONE)) - timedelta(days=1)).strftime("%Y-%m-%d")

def fetch_actual_results(target_date):
    h = {"x-apisports-key": API_FOOTBALL_KEY}
    results = {}
    d = datetime.strptime(target_date, "%Y-%m-%d")
    dates_to_fetch = [
        target_date,
        (d + timedelta(days=1)).strftime("%Y-%m-%d")
    ]
    
    for date_str in dates_to_fetch:
        try:
            r = requests.get(API_FOOTBALL_BASE + "/fixtures", headers=h, params={"date": date_str}, timeout=15)
            data = r.json()
            if data.get("response"):
                for m in data["response"]:
                    if m["fixture"]["status"]["short"] in ["FT", "AET", "PEN"]:
                        hid = str(m["teams"]["home"]["id"])
                        aid = str(m["teams"]["away"]["id"])
                        h_name = m["teams"]["home"]["name"].lower()
                        a_name = m["teams"]["away"]["name"].lower()
                        
                        res_data = {
                            "home_goals": m["goals"]["home"], 
                            "away_goals": m["goals"]["away"], 
                            "status": "FT"
                        }
                        results[f"{hid}_{aid}"] = res_data
                        results[f"{h_name}_vs_{a_name}"] = res_data
        except Exception as e:
            pass
    return results

def verify_and_learn():
    yesterday = get_yesterday()
    print(f"\n🧠 [AI 自我复盘引擎] 启动... 对账日期: {yesterday}")
    
    pred_file = f"data/history_{yesterday.replace('-','')}_evening.json" 
    if not os.path.exists(pred_file):
        pred_file = "data/predictions.json"
        
    if not os.path.exists(pred_file):
        print("  ⚠️ 无预测记录，跳过对账。")
        return
        
    with open(pred_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    matches = data.get("matches", {}).get("today", data.get("results", []))
    if not matches:
        print("  ⚠️ 预测记录中无赛事数据。")
        return

    actual = fetch_actual_results(yesterday)
    review_log = []
    correct, total = 0, 0
    verified = []

    for match in matches:
        hid = str(match.get("home_id", ""))
        aid = str(match.get("away_id", ""))
        home = match["home_team"].lower()
        away = match["away_team"].lower()
        pred = match.get("prediction", {})
        
        found = actual.get(f"{hid}_{aid}")
        if not found:
            for key, res in actual.items():
                if home in key and away in key:
                    found = res
                    break
                    
        if found:
            total += 1
            hg, ag = found["home_goals"], found["away_goals"]
            actual_res = "主胜" if hg > ag else ("平局" if hg == ag else "客胜")
            is_correct = pred.get("result", "") == actual_res
            if is_correct: correct += 1
            
            verified.append({
                "match": f"{match['home_team']} vs {match['away_team']}",
                "result_correct": is_correct,
                "was_recommended": match.get("is_recommended", False)
            })
            
            review_log.append({
                "match": f"{match['home_team']} vs {match['away_team']}",
                "pred": f"{pred.get('result', '')} ({pred.get('predicted_score', '')})",
                "actual": f"{actual_res} ({hg}-{ag})",
                "correct": is_correct
            })

    if total == 0:
        print("  ⏳ 比赛尚未全部结束或获取不到结果。")
        return

    win_rate = (correct / total) * 100
    
    rec_matches = [v for v in verified if v["was_recommended"]]
    rec_correct = sum(1 for v in rec_matches if v["result_correct"])
    rec_win_rate = (rec_correct / len(rec_matches) * 100) if rec_matches else 0
    
    print(f"  🎯 昨日总体胜率: {win_rate:.1f}% ({correct}/{total})")
    print(f"  🔥 核心推荐胜率: {rec_win_rate:.1f}% ({rec_correct}/{len(rec_matches)})")

    prompt = f"你是量化足球AI。昨日总体胜率 {win_rate:.1f}%，精选推荐胜率 {rec_win_rate:.1f}%。复盘记录：{json.dumps(review_log, ensure_ascii=False)}。请深度反思预测失败的比赛，输出一段精炼的调参建议。严格返回纯JSON: {{\"reflection\": \"反思与今日策略(80字以内)\", \"risk_adjustment\": \"稳健/防冷/激进\"}}"
    
    try:
        h = {"Authorization": f"Bearer {GPT_API_KEY}", "Content-Type": "application/json"}
        payload = {"model": "gpt-5.4", "messages": [{"role": "user", "content": prompt}], "temperature": 0.5}
        r = requests.post(GPT_API_URL, headers=h, json=payload, timeout=20)
        
        clean = re.sub(r"```\w*", "", r.json()["choices"][0]["message"]["content"]).strip()
        start = clean.find("{")
        end = clean.rfind("}") + 1
        diary_data = json.loads(clean[start:end])
        diary_data["yesterday_win_rate"] = f"{win_rate:.1f}%"
        diary_data["yesterday_rec_win_rate"] = f"{rec_win_rate:.1f}%"
        diary_data["update_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        os.makedirs("data", exist_ok=True)
        with open("data/ai_diary.json", "w", encoding="utf-8") as f:
            json.dump(diary_data, f, ensure_ascii=False, indent=2)
        print(f"  ✅ 自我进化完成! 今日AI策略: {diary_data['reflection']}")
    except Exception as e:
        print(f"  ❌ AI反思请求失败: {e}")

if __name__ == "__main__":
    verify_and_learn()

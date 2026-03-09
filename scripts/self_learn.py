import json
import os
import requests
import time
from config import *

PRED_FILE = "../data/predictions.json"
DIARY_FILE = "../data/ai_diary.json"

def fetch_actual_results(date_str):
    """去 API 查指定日期的真实赛果"""
    h = {"x-apisports-key": API_FOOTBALL_KEY}
    try:
        r = requests.get(f"{API_FOOTBALL_BASE}/fixtures", headers=h, params={"date": date_str}, timeout=15)
        return r.json().get("response", [])
    except: return []

def self_learn():
    print("\n🧠 [AI 自我复盘引擎] 启动...")
    if not os.path.exists(PRED_FILE):
        print("  ⚠️ 暂无历史预测文件，跳过复盘。")
        return

    # 1. 加载昨天的预测记录
    with open(PRED_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    matches = data.get("matches", data.get("results", []))
    if not matches: return
    
    # 这里的 date 是上一次预测时的日期
    pred_date = data.get("date", matches[0].get("date", "")) 
    if not pred_date: return

    print(f"  📅 正在获取 {pred_date} 的真实赛果进行对账...")
    actuals = fetch_actual_results(pred_date)
    actual_dict = {f"{m['teams']['home']['id']}_{m['teams']['away']['id']}": m for m in actuals}

    review_log = []
    correct_count, total_finished = 0, 0

    # 2. 对账：预测 vs 真实
    for m in matches:
        hid, aid = m.get("home_id"), m.get("away_id")
        key = f"{hid}_{aid}"
        if key in actual_dict:
            fix = actual_dict[key]
            # 确保比赛已经踢完 (FT = Full Time)
            if fix["fixture"]["status"]["short"] in ["FT", "AET", "PEN"]: 
                total_finished += 1
                gh, ga = fix["goals"]["home"], fix["goals"]["away"]
                actual_result = "主胜" if gh > ga else ("平局" if gh == ga else "客胜")
                actual_score = f"{gh}-{ga}"

                pred = m.get("prediction", {})
                pred_result = pred.get("result", "")
                
                is_correct = (actual_result == pred_result)
                if is_correct: correct_count += 1

                review_log.append({
                    "match": f"{m['home_team']} vs {m['away_team']}",
                    "predicted": f"{pred_result} ({pred.get('predicted_score', '')})",
                    "actual": f"{actual_result} ({actual_score})",
                    "correct": is_correct
                })

    if total_finished == 0:
        print("  ⏳ 昨天的比赛尚未全部结束，暂不生成复盘日记。")
        return

    win_rate = (correct_count / total_finished) * 100
    print(f"  🎯 昨日预测胜率: {win_rate:.1f}% ({correct_count}/{total_finished})")

    # 3. 让 GPT 核心写“错题本”和“调参策略”
    prompt = f"你是量化系统的AI核心。以下是你昨天的预测复盘报告：\n昨日胜率: {win_rate:.1f}%\n详情："
    prompt += json.dumps(review_log, ensure_ascii=False)
    prompt += "\n请反思预测失败的比赛，分析是否高估了主队或低估了爆冷概率。输出一段精炼的【今日调参建议】，指导你今天的分析。\n"
    prompt += '请严格返回纯JSON: {"reflection": "昨天的教训与今日策略(100字以内)", "risk_adjustment": "稳健"}'

    print("  🤖 正在请求 GPT 深度反思...")
    try:
        h = {"Authorization": f"Bearer {GPT_API_KEY}", "Content-Type": "application/json"}
        payload = {"model": "gpt-5.4", "messages": [{"role": "user", "content": prompt}], "temperature": 0.5}
        r = requests.post(GPT_API_URL, headers=h, json=payload, timeout=20)
        
        t = r.json()["choices"][0]["message"]["content"].strip()
        start = t.find("{"); end = t.rfind("}") + 1
        diary_data = json.loads(t[start:end])
        diary_data["yesterday_win_rate"] = f"{win_rate:.1f}%"

        # 把复盘结果存为日记
        with open(DIARY_FILE, "w", encoding="utf-8") as f:
            json.dump(diary_data, f, ensure_ascii=False, indent=2)
            
        print(f"  ✅ 复盘完成！今日AI策略: {diary_data.get('reflection', '')}")
    except Exception as e:
        print(f"  ❌ AI反思失败: {e}")

if __name__ == "__main__":
    self_learn()

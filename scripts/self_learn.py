import json
import os
import requests
import time
from config import *
import metrics_ledger as ml

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
    settled = []  # 双账本结算明细

    # 2. 对账：预测 vs 真实 —— 同时做双账本结算
    for m in matches:
        hid, aid = m.get("home_id"), m.get("away_id")
        key = f"{hid}_{aid}"
        if key in actual_dict:
            fix = actual_dict[key]
            # 确保比赛已经踢完 (FT = Full Time)
            if fix["fixture"]["status"]["short"] in ["FT", "AET", "PEN"]:
                gh, ga = fix["goals"]["home"], fix["goals"]["away"]
                pred = m.get("prediction", {})

                s = ml.settle_one(pred, gh, ga)
                settled.append(s)

                review_log.append({
                    "match": f"{m['home_team']} vs {m['away_team']}",
                    "predicted": f"{s['pred_dir']} ({pred.get('predicted_score', '')})",
                    "actual": f"{s['actual_dir']} ({s['actual_score']})",
                    "correct": s["hit"],
                    "bettable": s["bettable"],
                    "upset": s["upset"],
                    "odds": s["odds"],
                    "profit": round(s["profit"], 3),
                })

    if not settled:
        print("  ⏳ 昨天的比赛尚未全部结束，暂不生成复盘日记。")
        return

    # 3. 双账本汇总（ROI 为主，胜率仅作参考）
    agg = ml.aggregate(settled)
    summary = ml.coaching_summary(agg)
    print("  🎯 昨日双账本复盘：")
    for ln in summary.splitlines():
        print("     " + ln)

    # 4. 让 GPT 核心写“错题本”和“调参策略”——以 ROI 为准，不拿胜率惩罚博冷
    prompt = (
        "你是一个混合型足球预测系统的AI核心。系统同时出「价值/稳胜单」与「博冷/反打单」。\n"
        "考核原则：盈亏用 ROI 衡量，不用胜率。博冷单天生低胜率高赔率，胜率低是正常的，只要 ROI 为正就是赢；\n"
        "严禁因为博冷单胜率低就建议「少博冷/别反打」——只有当博冷账本 ROI 明显为负时才可收紧博冷。\n"
        f"以下是昨日双账本复盘：\n{summary}\n逐场详情："
    )
    prompt += json.dumps(review_log, ensure_ascii=False)
    prompt += (
        "\n请分别反思价值单与博冷单：价值单是否选错方向/高估主队；博冷单是否背离信号不足却硬反打。"
        "输出精炼的【今日调参建议】。\n"
    )
    prompt += '请严格返回纯JSON: {"reflection": "昨天的教训与今日策略(120字以内)", "risk_adjustment": "稳健/进取/中性"}'

    print("  🤖 正在请求 GPT 深度反思...")
    diary_data = {}
    try:
        h = {"Authorization": f"Bearer {GPT_API_KEY}", "Content-Type": "application/json"}
        payload = {"model": "gpt-5.4", "messages": [{"role": "user", "content": prompt}], "temperature": 0.5}
        r = requests.post(GPT_API_URL, headers=h, json=payload, timeout=20)

        t = r.json()["choices"][0]["message"]["content"].strip()
        start = t.find("{"); end = t.rfind("}") + 1
        diary_data = json.loads(t[start:end])
        print(f"  ✅ 复盘完成！今日AI策略: {diary_data.get('reflection', '')}")
    except Exception as e:
        print(f"  ❌ AI反思失败（仍落地真实账本）: {e}")
        diary_data = {"reflection": "AI反思失败，以下仅为客观账本数据", "risk_adjustment": "中性"}

    # 无论 AI 反思成败，都落地双账本真实结果
    b = agg["bettable"]
    diary_data["ledger"] = agg
    diary_data["yesterday_summary"] = summary
    # 保留旧字段兼容，但明确标注这是含观望的全样本方向命中率，不是盈亏指标
    diary_data["yesterday_direction_accuracy"] = f"{agg['direction_accuracy_pct']}%"
    diary_data["yesterday_bettable_roi"] = f"{b['roi_pct']}%"
    diary_data["yesterday_win_rate"] = (
        f"dir {agg['direction_accuracy_pct']}% | bet ROI {b['roi_pct']}% ({b['staked']}单)"
    )
    try:
        with open(DIARY_FILE, "w", encoding="utf-8") as f:
            json.dump(diary_data, f, ensure_ascii=False, indent=2)
        print("  📒 双账本已写入日记。")
    except Exception as e:
        print(f"  ❌ 日记写入失败: {e}")

if __name__ == "__main__":
    self_learn()

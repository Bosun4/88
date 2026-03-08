""" 主运行脚本：抓取 → 预测 → 生成JSON """
import json
import os
from datetime import datetime
from zoneinfo import ZoneInfo

from config import *
from fetch_data import collect_all
from predict import run_predictions

def get_today(offset=0):
    return (datetime.now(ZoneInfo(TIMEZONE)) + timedelta(days=offset)).strftime("%Y-%m-%d")

def main():
    date = get_today()
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    session = "morning" if now.hour < 15 else "evening"

    print("=" * 60)
    print("⚽ 竞彩足球AI预测系统")
    print(f"📅 日期: {date} 时段: {'上午' if session == 'morning' else '晚上'}")
    print("=" * 60)

    # 1. 数据抓取（全部在 fetch_data.py）
    raw_data = collect_all(date)

    os.makedirs("data", exist_ok=True)
    with open("data/raw_data.json", "w", encoding="utf-8") as f:
        json.dump(raw_data, f, ensure_ascii=False, indent=2)

    if not raw_data.get("matches"):
        print("⚠️ 今日无竞彩比赛数据，生成空预测页面")
        output = {
            "date": date, "session": session,
            "update_time": now.strftime("%Y-%m-%d %H:%M:%S"),
            "total_matches": 0, "results": [], "top4": []
        }
    else:
        # 2. AI预测（9模型 + 双AI）
        results, top4 = run_predictions(raw_data)
        output = {
            "date": date, "session": session,
            "update_time": now.strftime("%Y-%m-%d %H:%M:%S"),
            "total_matches": len(results),
            "results": results,
            "top4": [{"rank": i+1, **t} for i, t in enumerate(top4)]
        }

    # 3. 保存
    with open("data/predictions.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

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
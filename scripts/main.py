import json
import os
import sys
import traceback
from datetime import datetime, timedelta, timezone

from fetch_data import collect_all
from predict import run_predictions

def get_target_date(offset=0):
    beijing_tz = timezone(timedelta(hours=8))
    now = datetime.now(beijing_tz) - timedelta(hours=11)
    return (now + timedelta(days=offset)).strftime("%Y-%m-%d")

def main():
    beijing_tz = timezone(timedelta(hours=8))
    now_time = datetime.now(beijing_tz)
        
    session = "morning" if now_time.hour < 15 else "evening"

    print("=" * 70)
    print("⚽ 量化足球投研终端 (断点续存省钱版)")
    print(f"📅 运行时间: {now_time.strftime('%Y-%m-%d %H:%M:%S')} | 时段: {session}")
    print("=" * 70)

    os.makedirs("data", exist_ok=True)
    target_path = "data/predictions.json"
    history_path = f"data/history_{now_time.strftime('%Y%m%d')}_{session}.json"

    final_output = {
        "update_time": now_time.strftime("%Y-%m-%d %H:%M:%S"),
        "top4": [], 
        "matches": {
            "yesterday": [],
            "today": [],
            "tomorrow": []
        }
    }

    days_map = {"yesterday": -1, "today": 0, "tomorrow": 1}

    for day_key, offset in days_map.items():
        try:
            target_date = get_target_date(offset)
            print(f"\n{'='*20} 正在处理 {day_key} ({target_date}) 的赛事 {'='*20}")
            
            raw_data = collect_all(target_date)
            if not raw_data or not raw_data.get("matches"):
                print(f"  [WARN] {target_date} 暂无比赛数据。")
                continue

            # 🔥 核心省钱指令：坚决不让 AI 去分析昨天的死局！只分析今天和明天！
            use_ai = (day_key in ["today", "tomorrow"])
            if not use_ai:
                print("  [INFO] 历史赛事，已自动切断AI通道以节省API费用，仅运行本地算力...")

            results, top4 = run_predictions(raw_data, use_ai=use_ai)
            final_output["matches"][day_key] = results
            
            if day_key == "today":
                final_output["top4"] = [{"rank": i + 1, **t} for i, t in enumerate(top4)]

            # 🔥 核心救命指令：每跑完一天，立刻存盘！
            # 哪怕后续程序崩了，你已经烧掉 API 算出来的数据也绝不会丢！
            with open(target_path, "w", encoding="utf-8") as f:
                json.dump(final_output, f, ensure_ascii=False, indent=2)
            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(final_output, f, ensure_ascii=False, indent=2)
                
            print(f"  ✅ {day_key} 数据已安全落盘，进度已保存！")

        except Exception as e:
            print("\n" + "!" * 70)
            print(f"🚨 {day_key} 处理发生异常: {type(e).__name__}。已跳过该日，之前的数据已安全保存！")
            traceback.print_exc()
            continue

    print(f"\n{'='*70}")
    print("✅ 全链路执行完毕！")
    print(f"📁 数据最终保存至: {target_path}")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()

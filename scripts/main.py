import json
import os
import sys
import traceback
from datetime import datetime, timedelta

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None

from fetch_data import collect_all
from predict import run_predictions

def get_target_date(offset=0):
    if ZoneInfo:
        now = datetime.now(ZoneInfo("Asia/Shanghai"))
    else:
        now = datetime.now()
    return (now + timedelta(days=offset)).strftime("%Y-%m-%d")

def main():
    if ZoneInfo:
        now_time = datetime.now(ZoneInfo("Asia/Shanghai"))
    else:
        now_time = datetime.now()
        
    session = "morning" if now_time.hour < 15 else "evening"

    print("=" * 70)
    print("⚽ 量化足球投研终端 (极客解耦版)")
    print(f"📅 运行时间: {now_time.strftime('%Y-%m-%d %H:%M:%S')} | 时段: {session}")
    print("=" * 70)

    os.makedirs("data", exist_ok=True)

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

    try:
        for day_key, offset in days_map.items():
            target_date = get_target_date(offset)
            print(f"\n{'='*20} 处理 {day_key} ({target_date}) 赛事 {'='*20}")
            
            raw_data = collect_all(target_date)
            if not raw_data or not raw_data.get("matches"):
                print(f"  [WARN] {target_date} 暂无比赛数据。")
                continue

            results, top4 = run_predictions(raw_data)
            final_output["matches"][day_key] = results
            
            if day_key == "today":
                final_output["top4"] = [{"rank": i + 1, **t} for i, t in enumerate(top4)]

        target_path = "data/predictions.json"
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(final_output, f, ensure_ascii=False, indent=2)

        history_path = f"data/history_{now_time.strftime('%Y%m%d')}_{session}.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(final_output, f, ensure_ascii=False, indent=2)

        print(f"\n{'='*70}")
        print("✅ 全链路执行成功！")
        print(f"📁 数据已保存至: {target_path}")
        print(f"{'='*70}")

    except Exception as e:
        print("\n" + "!" * 70)
        print(f"🚨 致命崩溃: {type(e).__name__}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

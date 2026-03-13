import json
import os
import sys
import traceback
from datetime import datetime, timedelta, timezone

def get_target_date(offset=0):
    beijing_tz = timezone(timedelta(hours=8))
    now = datetime.now(beijing_tz) - timedelta(hours=11)
    return (now + timedelta(days=offset)).strftime("%Y-%m-%d")

def main():
    beijing_tz = timezone(timedelta(hours=8))
    now_time = datetime.now(beijing_tz)
        
    session = "morning" if now_time.hour < 15 else "evening"

    print("=" * 70)
    print("⚽ 量化足球投研终端 (绝对路径物理锁死版)")
    print(f"📅 运行时间: {now_time.strftime('%Y-%m-%d %H:%M:%S')} | 时段: {session}")
    print("=" * 70)

    # 🔥 核心防御装甲：获取当前脚本的绝对物理路径，彻底根除相对路径迷失陷阱！
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    
    # 强制在绝对路径下创建 data 文件夹
    os.makedirs(data_dir, exist_ok=True)
    
    # 将保存路径绑定到绝对物理坐标
    target_path = os.path.join(data_dir, "predictions.json")
    history_path = os.path.join(data_dir, f"history_{now_time.strftime('%Y%m%d')}_{session}.json")

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
        from fetch_data import collect_all
        from predict import run_predictions
        
        for day_key, offset in days_map.items():
            target_date = get_target_date(offset)
            print(f"\n{'='*20} 正在抓取并清洗 {day_key} ({target_date}) 的赛事 {'='*20}")
            
            raw_data = collect_all(target_date)
            if not raw_data or not raw_data.get("matches"):
                print(f"  [WARN] {target_date} 暂无比赛数据。")
                continue

            # 🔥 AI 资金节流阀
            use_ai = (day_key in ["today", "tomorrow"])
            if not use_ai:
                print("  [INFO] 历史赛事，已自动切断AI通道以节省API费用...")

            results, top4 = run_predictions(raw_data, use_ai=use_ai)
            final_output["matches"][day_key] = results
            
            if day_key == "today":
                final_output["top4"] = [{"rank": i + 1, **t} for i, t in enumerate(top4)]

            # 绝对路径安全落盘
            with open(target_path, "w", encoding="utf-8") as f:
                json.dump(final_output, f, ensure_ascii=False, indent=2)
            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(final_output, f, ensure_ascii=False, indent=2)
                
            print(f"  ✅ {day_key} 数据已安全落盘，进度已保存！")

        print(f"\n{'='*70}")
        print("✅ 全链路执行成功！")
        print(f"📁 数据最终保存至: {target_path}")
        print(f"{'='*70}")

    except Exception as e:
        print("\n" + "!" * 70)
        print(f"🚨 致命崩溃: {type(e).__name__}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

"""
主运行脚本：
1. 调度 fetch_data (拉取问彩情报)
2. 调度 predict (执行 14模型量化 + 双 AI)
3. 生成前端依赖的 JSON 格式
"""
import json
import os
import sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from fetch_data import collect_all, get_today
from predict import run_predictions

def main():
    date = get_today()
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    session = "morning" if now.hour < 15 else "evening"

    print("=" * 60)
    print("⚽ 竞彩足球AI量化引擎 (Wencai 进化版)")
    print(f"📅 日期: {date} 时段: {'上午' if session == 'morning' else '晚上'}")
    print("=" * 60)

    # 1. 启动全新的全维度数据抓取（自带伤停、情报、赔率）
    raw_data = collect_all(date)

    os.makedirs("data", exist_ok=True)
    with open("data/raw_data.json", "w", encoding="utf-8") as f:
        json.dump(raw_data, f, ensure_ascii=False, indent=2)

    # 如果抓取到了比赛
    if not raw_data.get("matches"):
        print("⚠️ 今日无竞彩比赛数据，生成空预测页面保护前端")
        output = {
            "date": date,
            "session": session,
            "update_time": now.strftime("%Y-%m-%d %H:%M:%S"),
            "total_matches": 0,
            "results": [],
            "top4": [],
        }
    else:
        # 2. 14大模型运算与AI深度研判
        results, top4 = run_predictions(raw_data)
        
        # 3. 对齐前端数据结构
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

    # 4. 保存最终预测结果
    with open("data/predictions.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # 同时保存历史存档
    history_file = f"data/history_{date}_{session}.json"
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"✅ 全部链路执行完成！")
    print(f" 📊 共计算 {output['total_matches']} 场比赛")
    print(f" 🎯 生成核心推荐 {len(output['top4'])} 场")
    print(f" 📁 最终文件已同步: data/predictions.json")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

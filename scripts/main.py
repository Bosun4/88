"""
主运行脚本：
1. 连续抓取：昨天、今天、明天 3个日期的竞彩赛事。
2. 调度 predict 执行 AI 预测。
3. 封装为单一 JSON，供前端 Tab 页面渲染。
"""
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
    """获取带偏移量的北京时间日期"""
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
    print("⚽ 量化足球投研终端 (三日全量版)")
    print(f"📅 运行时间: {now_time.strftime('%Y-%m-%d %H:%M:%S')} | 时段: {session}")
    print("=" * 70)

    os.makedirs("data", exist_ok=True)

    # 前端所需要的全局字典结构
    final_output = {
        "update_time": now_time.strftime("%Y-%m-%d %H:%M:%S"),
        "top4": [], # 仅存放"今日"的推荐4场
        "matches": {
            "yesterday": [],
            "today": [],
            "tomorrow": []
        }
    }

    # 定义要抓取的偏移量映射
    days_map = {
        "yesterday": -1,
        "today": 0,
        "tomorrow": 1
    }

    try:
        # 依次抓取和预测 3 天的比赛
        for day_key, offset in days_map.items():
            target_date = get_target_date(offset)
            print(f"\n{'='*20} 正在处理 {day_key} ({target_date}) 的赛事 {'='*20}")
            
            raw_data = collect_all(target_date)
            
            if not raw_data or not raw_data.get("matches"):
                print(f"  [WARN] {target_date} 暂无比赛数据。")
                continue

            # 运行核心预测引擎
            results, top4 = run_predictions(raw_data)
            
            # 存入对应的分类列表
            final_output["matches"][day_key] = results
            
            # 如果是"今天"，则提取全局 TOP4 供首页展示
            if day_key == "today":
                final_output["top4"] = [{"rank": i + 1, **t} for i, t in enumerate(top4)]

        # 写入前端所需的 latest.json (或 predictions.json)
        target_path = "data/predictions.json"
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(final_output, f, ensure_ascii=False, indent=2)

        # 归档历史
        history_path = f"data/history_{now_time.strftime('%Y%m%d')}_{session}.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(final_output, f, ensure_ascii=False, indent=2)

        print(f"\n{'='*70}")
        print("✅ 全链路执行成功！")
        print(f"📊 昨日: {len(final_output['matches']['yesterday'])}场 | 今日: {len(final_output['matches']['today'])}场 | 明日: {len(final_output['matches']['tomorrow'])}场")
        print(f"🎯 今日黄金推荐: {len(final_output['top4'])}场")
        print(f"📁 实时数据已保存至: {target_path}")
        print(f"{'='*70}")

    except Exception as e:
        print("\n" + "!" * 70)
        print(f"🚨 系统遭遇致命崩溃！错误类型: {type(e).__name__}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

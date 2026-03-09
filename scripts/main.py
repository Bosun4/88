import json
import os
import sys
import traceback
from datetime import datetime
from config import *
from fetch_data import collect_all
from predict import run_predictions

def main():
    print("\n" + "=" * 70)
    print(f"⚽ 量化足球投研终端 | 核心引擎启动 | 系统时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    os.makedirs("data", exist_ok=True)

    try:
        raw = collect_all()
        if not raw or not raw.get("matches"):
            print("  [WARN] 今日无有效赛事数据，系统进入空载运行。")
            return

        results, best_parlay = run_predictions(raw)

        print("\n[STEP 3/3] 正在封装最新展现层 JSON 并执行强类型安全校验...")
        
        safe_top_ids = []
        valid_results = []
        
        for match in results:
            if isinstance(match, dict) and "home_team" in match:
                valid_results.append(match)
                if match.get("is_recommended") == True:
                    m_id = match.get("id") or match.get("match_id")
                    if m_id: safe_top_ids.append(m_id)

        final_json = {
            "fetch_time": raw.get("fetch_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            "total_matches": len(valid_results),
            "best_parlay": best_parlay,
            "matches": valid_results,
            "top4_ids": safe_top_ids
        }

        target_path = "data/latest.json"
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(final_json, f, ensure_ascii=False, indent=4)
        
        dt_tag = datetime.now().strftime("%Y%m%d_%H")
        history_path = f"data/history_{dt_tag}.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(final_json, f, ensure_ascii=False, indent=4)

        print(f"\n{'='*70}")
        print("✅ 任务流成功闭环！")
        print(f"📊 监控场次: {len(valid_results)} 场 | 强推: {len(safe_top_ids)} 场")
        
        if best_parlay:
            print(f"🎯 黄金 2串1: {best_parlay.get('combo', '无')}")
            print(f"📈 组合赔率: {best_parlay.get('combined_odds', 1.0)}")
            
        print(f"📁 实时同步: {target_path}")
        print(f"{'='*70}")

    except Exception as e:
        print("\n" + "!" * 70)
        print(f"🚨 系统遭遇崩溃！错误类型: {type(e).__name__}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

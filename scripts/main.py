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

    # 1. 保障数据路径
    os.makedirs("data", exist_ok=True)

    try:
        # STEP 1: 全量深度情报采集
        print("\n[STEP 1/3] 正在拉取全量接口情报数据...")
        raw = collect_all()
        
        if not raw or not raw.get("matches"):
            print("  [WARN] 今日无有效赛事数据，系统进入空载运行。")
            return

        # STEP 2: 启动量化研判与 2串1 策略生成
        print("\n[STEP 2/3] 正在启动 11 核心模型与 AI 双向交叉验证...")
        results, best_parlay = run_predictions(raw)

        # STEP 3: 构建高密度输出数据 (安全防御修复区)
        print("\n[STEP 3/3] 正在封装最新展现层 JSON 并执行强类型安全校验...")
        
        safe_top_ids = []
        valid_results = []
        
        # 🔥 彻底解决 TypeError 的关键循环
        # 这里严密防止了任何来自于 AI 幻觉导致的非法数据结构
        for match in results:
            if isinstance(match, dict) and "home_team" in match:
                valid_results.append(match)
                
                # 检查是否为精选推荐场次
                if match.get("is_recommended") == True:
                    # 获取 ID（兼容不同数据源的键名差异）
                    m_id = match.get("id")
                    if not m_id:
                        m_id = match.get("match_id")
                    if m_id: 
                        safe_top_ids.append(m_id)

        # 构建给 index.html 渲染用的全量字典
        final_json = {
            "fetch_time": raw.get("fetch_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            "total_matches": len(valid_results),
            "best_parlay": best_parlay,
            "matches": valid_results,
            "top4_ids": safe_top_ids
        }

        # STEP 4: 实时同步与存档
        target_path = "data/latest.json"
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(final_json, f, ensure_ascii=False, indent=4)
        
        # 按小时建立历史存档，防止数据意外被覆盖
        dt_tag = datetime.now().strftime("%Y%m%d_%H")
        history_path = f"data/history_{dt_tag}.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(final_json, f, ensure_ascii=False, indent=4)

        # 华丽的终端报告
        print(f"\n{'='*70}")
        print("✅ 任务流成功闭环！")
        print(f"📊 监控场次: {len(valid_results)} 场 | AI 强推: {len(safe_top_ids)} 场")
        
        if best_parlay:
            print(f"🎯 黄金 2串1: {best_parlay.get('combo', '无')}")
            print(f"📈 组合联合赔率: {best_parlay.get('combined_odds', 1.0)}")
            
        print(f"📁 实时读数已同步: {target_path}")
        print(f"{'='*70}")

    except Exception as e:
        # 当发生致命错误时，绝不掩饰，抛出并退出，让 Actions 报红
        print("\n" + "!" * 70)
        print(f"🚨 系统遭遇致命崩溃！错误类型: {type(e).__name__}")
        print("以下为详细错误回溯日志:")
        traceback.print_exc()
        print("!" * 70)
        sys.exit(1)

if __name__ == "__main__":
    main()

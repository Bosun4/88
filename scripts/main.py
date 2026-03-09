"""
主运行脚本 v4.5 (终极修复版):
1. [Fetch] 抓取含伤停、异动、专家推介的全量数据
2. [Predict] 执行11模型运算与双AI交叉验证，生成2串1方案
3. [Storage] 安全持久化至 latest.json，防止 TypeError 崩溃
"""
import json
import os
import sys
from datetime import datetime
from config import *
from fetch_data import collect_all
from predict import run_predictions

def main():
    # 打印纯净版 Header
    print("=" * 65)
    print(f"⚽ 量化足球投研终端 | 运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    # 确保存储目录存在
    os.makedirs("data", exist_ok=True)

    try:
        # STEP 1: 全情报采集
        raw = collect_all()
        if not raw or not raw.get("matches"):
            print("  ⚠️ 今日暂无竞彩足球赛事，任务中止。")
            return

        # STEP 2: 核心预测与 2串1 策略生成
        results, best_parlay = run_predictions(raw)

        # STEP 3: 构建全维度输出字典 (核心修复区)
        print("\n[STEP 3/3] 正在对齐前端数据结构并执行安全检查...")
        
        final_output = {
            "fetch_time": raw.get("fetch_time", datetime.now().strftime("%H:%M:%S")),
            "total_matches": len(results),
            "best_parlay": best_parlay,
            "matches": results,
            # 安全提取精选场次 ID，加入 isinstance 检查防止 TypeError
            "top4_ids": [
                m["match_id"] for m in results 
                if isinstance(m, dict) and m.get("is_recommended")
            ]
        }

        # STEP 4: 写入最新数据 (latest.json 是前端渲染的唯一依赖)
        target_path = "data/latest.json"
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(final_output, f, ensure_ascii=False, indent=4)
        
        # STEP 5: 历史存档 (用于日后胜率复盘)
        history_tag = datetime.now().strftime("%Y%m%d_%H%M")
        with open(f"data/history_{history_tag}.json", "w", encoding="utf-8") as f:
            json.dump(final_output, f, ensure_ascii=False, indent=4)

        # 终端总结报告
        print(f"\n{'='*65}")
        print("✅ 全量预测任务成功闭环！")
        print(f"📊 总计场次: {len(results)} | 精选: {len(final_output['top4_ids'])} 场")
        
        if best_parlay:
            print(f"🎯 黄金 2串1 建议: {best_parlay['combo']}")
            print(f"📈 组合总赔率: {best_parlay['combined_odds']} | 平均信心: {best_parlay['confidence']}%")
        
        print(f"📁 实时文件已同步: {target_path}")
        print(f"{'='*65}")

    except Exception as e:
        print("\n" + "!" * 65)
        print(f"🚨 系统遭遇致命崩溃！")
        print(f"错误类型: {type(e).__name__}")
        print(f"详细描述: {str(e)}")
        # 打印详细错误位置
        import traceback
        traceback.print_exc()
        print("!" * 65)
        # 必须 exit(1) 让 GitHub Actions 报错提醒用户
        sys.exit(1)

if __name__ == "__main__":
    main()

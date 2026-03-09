"""
主运行脚本 (完全体 v5.0):
1. [Fetch] 调用高级 JSON 接口提取：伤停名单、专家点评、基本面深度研报及实时水位异动
2. [Predict] 11核心量化模型运算 + 双 AI (GPT/Gemini) 深度情报研判与交叉比分验证
3. [Strategy] 2串1 自动组合优化，利用联合EV计算当日最高期望回报方案
4. [Storage] 生成对齐最新前端的 latest.json 与 早/晚盘历史档案
"""
import json
import os
import sys
import time
import re
from datetime import datetime, timedelta

# 尝试导入时区库，若失败则降级处理
try:
    from zoneinfo import ZoneInfo
except ImportError:
    # 适配低版本 Python 环境或某些 Actions 环境
    ZoneInfo = None

from config import *
from fetch_data import collect_all
from predict import run_predictions

# ==================== 1. 系统时间与调度逻辑 ====================
def get_system_time():
    """获取精准时区时间、日期字符串及场次时段(早盘/晚盘)"""
    if ZoneInfo and TIMEZONE:
        try:
            tz = ZoneInfo(TIMEZONE)
            now = datetime.now(tz)
        except:
            now = datetime.now()
    else:
        now = datetime.now()
        
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    
    # 划分场次：下午15:00前运行视为“早盘”，15:00后视为“晚盘”
    # 对应竞彩大部分比赛的截售与开赛节奏
    session = "morning" if now.hour < 15 else "evening"
    session_zn = "早盘阶段" if session == "morning" else "晚盘阶段"
    
    return date_str, time_str, session, session_zn

# ==================== 2. 主逻辑执行枢纽 ====================
def main():
    date_str, time_str, session, session_zn = get_system_time()

    print("=" * 65)
    print(f"🚀 AI 量化足球投研终端 (全维度情报融合版) | {date_str}")
    print(f"🕒 系统启动时间: {time_str} | 运行场次: {session_zn}")
    print("=" * 65)

    # 目录初始化
    os.makedirs("data", exist_ok=True)
    print("  [Init] 数据持久化目录就绪")

    try:
        # --- 阶段 1: 深度情报采集 ---
        # 调用已升级的 fetch_data.py，捕获 wencai API 的专家推介、伤停利空、投票及实时水位
        print(f"\n[STEP 1/3] 正在接入深度情报网络...")
        raw_data = collect_all(date_str)
        
        # 兜底：如果 API 没给数据，生成空预测防止前端白屏
        if not raw_data or not raw_data.get("matches") or len(raw_data["matches"]) == 0:
            print("  ⚠️ 警告: 今日暂无有效竞彩足球赛事数据，生成空存档。")
            final_output = {
                "fetch_time": f"{date_str} {time_str}",
                "total_matches": 0,
                "best_parlay": None,
                "matches": [],
                "date": date_str,
                "session": session
            }
        else:
            # --- 阶段 2: 核心预测与 2串1 优化策略 ---
            # run_predictions 此时返回：
            # results: 包含预测、伤停、专家文本、AI比分、盘口动向的所有场次列表
            # best_parlay: 经过联合EV优化后的当日最佳 2串1 方案
            print(f"\n[STEP 2/3] 正在对 {len(raw_data['matches'])} 场赛事执行量化研判...")
            results, best_parlay = run_predictions(raw_data)
            
            # --- 阶段 3: 封装高密度展现层 JSON ---
            print("\n[STEP 3/3] 正在构建全维度输出数据...")
            final_output = {
                "fetch_time": f"{date_str} {time_str}",
                "date": date_str,
                "session": session,
                "total_matches": len(results),
                "best_parlay": best_parlay, # 置顶的黄金组合建议
                "matches": results,         # 携带 intelligence, handicap 等所有核心字段
                "top4_ids": [m["match_id"] for m in results if m.get("is_recommended")]
            }

        # --- 阶段 4: 写入持久化文件 ---
        # 1. latest.json: 前端 index.html 实时渲染的唯一金标文件
        target_file = "data/latest.json"
        with open(target_file, "w", encoding="utf-8") as f:
            json.dump(final_output, f, ensure_ascii=False, indent=4)
        
        # 2. 存档文件: 用于未来 AI 进行“错题本”自学习和胜率追溯
        history_file = f"data/history_{date_str}_{session}.json"
        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(final_output, f, ensure_ascii=False, indent=4)

        # 华丽的终端报告
        print(f"\n{'='*65}")
        print("✅ 全量预测任务成功闭环！")
        print(f"📊 总分析场次: {final_output['total_matches']} 场")
        
        if final_output.get("best_parlay"):
            bp = final_output["best_parlay"]
            print(f"🎯 今日最佳对冲方案 (2串1): {bp['combo']}")
            print(f"📈 组合赔率积: {bp['combined_odds']} | 平均模型信心: {bp['confidence']}%")
        
        print(f"📁 核心文件已更新: {target_file}")
        print(f"📚 历史数据已归档: {history_file}")
        print(f"{'='*65}")

    except Exception as e:
        print("\n" + "!" * 65)
        print(f"🚨 系统运行崩溃！")
        print(f"错误类型: {type(e).__name__}")
        print(f"详细描述: {str(e)}")
        # 详细堆栈回溯，方便在 GitHub Actions 日志里直接抓虫
        import traceback
        traceback.print_exc()
        print("!" * 65)
        # 强行抛出错误，让 GitHub Actions 标红，而不是假装运行成功
        sys.exit(1)

if __name__ == "__main__":
    main()

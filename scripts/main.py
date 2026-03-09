"""
主调度枢纽 v5.8 (量化投研完全体):
1. [Safety Architecture] 核心防御：引入 isinstance 严苛校验，彻底根治 'list indices must be integers' 的 TypeError 崩溃。
2. [Session Management] 业务分段：基于 Asia/Shanghai 时区自动划分 早盘/晚盘 时段，执行差异化存档。
3. [Intelligence Integration] 情报中转：确保 专家点评(intro)、利空名单、水位异动 等高密度字段完整进入最新 JSON 数据库。
4. [Strategy Strategy] 决策下发：获取当日黄金 2串1 方案，并实时同步至 index.html 前端渲染引擎。
"""
import json
import os
import sys
import time
import re
import traceback
from datetime import datetime, timedelta

# 导入时区处理逻辑，确保服务器运行时间与北京时间对齐
try:
    from zoneinfo import ZoneInfo
except ImportError:
    # 兼容低版本 Python 运行环境
    ZoneInfo = None

# 加载自定义业务功能模块
from config import *
from fetch_data import collect_all
from predict import run_predictions

# ==================== 1. 系统时间与时段管理逻辑 ====================
def get_run_session():
    """
    确定当前的运行环境时间及业务时段划分。
    15:00 前运行为早盘 (morning)，15:00 后运行为晚盘 (evening)。
    """
    if ZoneInfo and TIMEZONE:
        try:
            tz = ZoneInfo(TIMEZONE)
            now = datetime.now(tz)
        except Exception:
            now = datetime.now()
    else:
        now = datetime.now()
        
    date_tag = now.strftime("%Y-%m-%d")
    full_time = now.strftime("%Y-%m-%d %H:%M:%S")
    hour = now.hour
    
    # 时段划分逻辑 (对应竞彩截售时间节点)
    session_key = "morning" if hour < 15 else "evening"
    session_label = "【早盘扫描阶段】" if hour < 15 else "【晚盘核心阶段】"
    
    return date_tag, full_time, session_key, session_label

# ==================== 2. 主逻辑执行指挥中心 ====================
def main():
    date_str, time_str, session_id, session_name = get_run_session()

    # 输出高密度终端抬头
    print("\n" + "=" * 75)
    print(f"🚀 量化足球投研终端 (Intelligence Integrated Edition) v5.8")
    print(f"📅 运行日期: {date_str} | 系统时间: {time_str}")
    print(f"🏁 业务时段: {session_name}")
    print("=" * 75)

    # 目录健壮性检查
    if not os.path.exists("data"):
        os.makedirs("data")
        print("  [Init] 数据存储目录已初始化...")

    try:
        # --- PHASE 1: 深度情报采集 ---
        # 激活 fetch_data.py，捕获 wencai 接口的文字情报、伤停及异动
        print(f"\n[STEP 1/3] 正在建立情报链路，深度同步文本研报与伤停数据...")
        raw_data = collect_all(date_str)
        
        # 防御逻辑：API 异常或当日无赛
        if not raw_data or not raw_data.get("matches") or len(raw_data["matches"]) == 0:
            print(f"  [WARN] {date_str} {session_name} 暂无有效赛事，正在生成空存档保护前端。")
            empty_json = {
                "fetch_time": f"{date_str} {time_str}",
                "total_matches": 0,
                "best_parlay": None,
                "matches": [],
                "date": date_str,
                "session": session_id
            }
            with open("data/latest.json", "w", encoding="utf-8") as f:
                json.dump(empty_json, f, ensure_ascii=False, indent=4)
            return

        # --- PHASE 2: 模型运算与 AI 联合研判 ---
        # 此时的 run_predictions 返回 results(列表) 和 best_parlay(字典)
        # 它已内部集成了 11个模型 + 专家点评分析 + AI交叉比分
        print(f"\n[STEP 2/3] 正在对 {len(raw_data['matches'])} 场赛事执行量化建模与多路 AI 复核...")
        results, best_parlay = run_predictions(raw_data)
        
        # --- PHASE 3: 封装输出与 TypeError 防崩校验 ---
        print("\n[STEP 3/3] 正在执行全维度数据封装与类型安全校验...")
        
        # 修复核心：严密的类型过滤，防止单场预测因 AI 格式错误导致的索引崩溃
        # 彻底解决 "list indices must be integers" 的 Bug
        valid_results = []
        safe_top_ids = []
        
        for m in results:
            # 必须确保每一场都是字典结构，且包含核心键
            if isinstance(m, dict) and "home_team" in m:
                valid_results.append(m)
                # 提取被标记为强信心的推荐场次索引
                if m.get("is_recommended") == True:
                    m_id = m.get("match_id")
                    if m_id is not None:
                        safe_top_ids.append(m_id)

        # 构造对齐 index.html 的最终数据字典
        final_output = {
            "fetch_time": f"{date_str} {time_str}",
            "date": date_str,
            "session": session_id,
            "total_matches": len(valid_results),
            "best_parlay": best_parlay,       # 存入当日黄金组合建议
            "matches": valid_results,         # 携带 intelligence, analysis 等深度字段
            "top4_ids": safe_top_ids          # 网页顶部展示位所引用的 ID 列表
        }

        # --- PHASE 4: 持久化持久化与早/晚盘存档 ---
        # 1. latest.json: 前端实时渲染的唯一数据源，必须保持最新
        target_file = "data/latest.json"
        with open(target_file, "w", encoding="utf-8") as f:
            json.dump(final_output, f, ensure_ascii=False, indent=4)
        
        # 2. History Archive: 带时间戳的历史文件，用于 AI 自学习和回测
        dt_tag = datetime.now().strftime("%Y%m%d")
        history_path = f"data/history_{dt_tag}_{session_id}.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(final_output, f, ensure_ascii=False, indent=4)

        # 华丽的终端运行总结
        print(f"\n{'='*75}")
        print("✅ 投研流水线执行成功！")
        print(f"📊 总监控场次: {len(valid_results)} 场")
        print(f"🌟 强信心推荐: {len(safe_top_ids)} 场")
        
        if best_parlay:
            print(f"🎯 黄金 2串1 方案: {best_parlay.get('combo', '待生成')}")
            print(f"📈 联合赔率积: {best_parlay.get('combined_odds')} | 信心: {best_parlay.get('confidence')}%")
        else:
            print(f"🎯 黄金 2串1 方案: (当日无合意对冲组合)")
        
        print(f"📁 实时读数已同步: {target_file}")
        print(f"📚 历史档案已存入: {history_path}")
        print(f"{'='*75}")

    except Exception as e:
        print("\n" + "!" * 75)
        print(f"🚨 系统中心调度异常！崩溃类型: {type(e).__name__}")
        print(f"详细描述: {str(e)}")
        # 详细堆栈输出，方便直接抓虫
        traceback.print_exc()
        print("!" * 75)
        # 抛出 exit code 1，让 GitHub Actions 标红提醒用户
        sys.exit(1)

if __name__ == "__main__":
    main()

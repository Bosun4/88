"""
主调度枢纽 (完美对接版):
1. [时区对齐] 兼容标准环境，精准生成北京时间戳与业务时段。
2. [安全调度] 调度 fetch_data 与 predict，全程捕获异常，阻断级联崩溃。
3. [数据组装] 提取推荐池，完全对齐前端“小鱼儿足球分析”的 JSON 结构。
4. [双向归档] 实时写入 latest.json，并按时段自动备份 history.json。
"""
import json
import os
import sys
import traceback
from datetime import datetime, timedelta

# 兼容不同 Python 版本的时区库
try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None

from config import *
from fetch_data import collect_all
from predict import run_predictions

# ==================== 1. 系统时间与时段管理 ====================
def get_run_session():
    """获取北京时间，并划分早盘/晚盘业务时段"""
    if ZoneInfo and TIMEZONE:
        try:
            tz = ZoneInfo(TIMEZONE)
            now = datetime.now(tz)
        except Exception:
            now = datetime.now()
    else:
        now = datetime.now()
        
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%Y-%m-%d %H:%M:%S")
    hour = now.hour
    
    session_id = "morning" if hour < 15 else "evening"
    session_name = "【早盘扫描阶段】" if hour < 15 else "【晚盘核心阶段】"
    
    return date_str, time_str, session_id, session_name

# ==================== 2. 主调度逻辑 ====================
def main():
    date_str, time_str, session_id, session_name = get_run_session()

    print("\n" + "=" * 75)
    print(f"🚀 量化足球投研终端 (最终闭环版) | 系统启动")
    print(f"📅 运行日期: {date_str} | 时间: {time_str}")
    print(f"🏁 当前时段: {session_name}")
    print("=" * 75)

    # 确保持久化目录存在，绝不报错
    if not os.path.exists("data"):
        os.makedirs("data")
        print("  [Init] 数据存储目录 data/ 已初始化...")

    try:
        # --- 阶段 1: 全维度数据抓取 ---
        print(f"\n[STEP 1/3] 正在拉取 Wencai 情报与 Odds 基础数据...")
        raw_data = collect_all(date_str)
        
        # 兜底逻辑：如果今天没比赛，生成安全的空文件保护前端不报错
        if not raw_data or not raw_data.get("matches"):
            print(f"  [WARN] {date_str} 暂无有效赛事数据，系统进入空载运行。")
            empty_output = {
                "fetch_time": time_str,
                "total_matches": 0,
                "best_parlay": None,
                "matches": [],
                "top4_ids": []
            }
            with open("data/latest.json", "w", encoding="utf-8") as f:
                json.dump(empty_output, f, ensure_ascii=False, indent=4)
            return

        # --- 阶段 2: 核心预测与 AI 融合 ---
        print(f"\n[STEP 2/3] 引擎就绪，开始处理 {len(raw_data['matches'])} 场赛事...")
        results, best_parlay = run_predictions(raw_data)

        # --- 阶段 3: 数据封装与安全校验 ---
        print("\n[STEP 3/3] 正在封装前端展现层 JSON...")
        
        safe_top_ids = []
        valid_results = []
        
        # 遍历每一场比赛，提取被推荐（EV高/评分高）的场次 ID
        for match in results:
            if isinstance(match, dict) and "home_team" in match:
                valid_results.append(match)
                
                # 提取在 predict.py 中被打上 is_recommended 标签的场次
                if match.get("is_recommended") == True:
                    m_id = match.get("match_id")
                    if m_id is not None:
                        safe_top_ids.append(m_id)

        # 构建给 index.html 渲染用的全量字典，严密对齐所有字段
        final_json = {
            "fetch_time": time_str,
            "total_matches": len(valid_results),
            "best_parlay": best_parlay,
            "matches": valid_results,
            "top4_ids": safe_top_ids  # 前端顶部的“核心4场”依赖于此
        }

        # --- 阶段 4: 双重持久化存储 ---
        # 1. 实时文件 (前端读取的唯一凭证)
        target_path = "data/latest.json"
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(final_json, f, ensure_ascii=False, indent=4)
        
        # 2. 历史留档 (用于复盘 AI 准确率)
        dt_tag = datetime.now().strftime(f"%Y%m%d_{session_id}")
        history_path = f"data/history_{dt_tag}.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(final_json, f, ensure_ascii=False, indent=4)

        # --- 阶段 5: 运行报告摘要 ---
        print(f"\n{'='*75}")
        print("✅ 量化投研全链路成功闭环！")
        print(f"📊 监控总场次: {len(valid_results)} 场")
        print(f"🌟 入选精选池: {len(safe_top_ids)} 场")
        
        if best_parlay:
            print(f"🎯 黄金 2串1 推荐: {best_parlay.get('combo', '无')}")
            print(f"📈 联合期望赔率: {best_parlay.get('combined_odds', 1.0)}")
        else:
            print(f"🎯 黄金 2串1 推荐: (暂无符合高 EV 要求的稳健组合)")
            
        print(f"📁 实时数据已推送至: {target_path}")
        print(f"📚 归档档案已保存至: {history_path}")
        print(f"{'='*75}")

    except Exception as e:
        # 天地大冲撞拦截：打印完整错误栈并退出，触发 Actions 红灯
        print("\n" + "!" * 75)
        print(f"🚨 系统中枢遭遇致命异常！错误类型: {type(e).__name__}")
        print("以下为核心报错回溯 (Traceback):")
        traceback.print_exc()
        print("!" * 75)
        sys.exit(1)

if __name__ == "__main__":
    main()

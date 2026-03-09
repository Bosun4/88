"""
主运行脚本 v5.2 (量化投研完全体):
1. [Environment] 初始化系统环境，自动创建缺失的数据目录
2. [Time Management] 精准识别 Asia/Shanghai 时区，自动划分早/晚盘业务时段
3. [Intelligence Fetch] 启动全维度抓取引擎：捕获伤停名单、专家推介语、基本面研报及水位异动
4. [Engine Analysis] 运行底层 11 核心量化模型矩阵 + 双 AI 深度对冲逻辑
5. [Strategy Optimization] 计算 2串1 笛卡尔积组合，筛选当日最高 EV 联合对冲方案
6. [Data Persistence] 执行强类型 isinstance 校验，生成对齐前端的 latest.json 与 高密度历史存档
"""
import json
import os
import sys
import time
import re
import traceback
from datetime import datetime, timedelta

# 导入时区处理逻辑
try:
    from zoneinfo import ZoneInfo
except ImportError:
    # 兼容低版本 Python 环境或特殊 Linux 容器环境
    ZoneInfo = None

# 加载自定义业务模块
from config import *
from fetch_data import collect_all
from predict import run_predictions

# ==================== 1. 系统时间与时段划分函数 ====================
def get_session_info():
    """
    根据北京时间自动划分业务时段。
    早盘 (Morning): 00:00 - 14:59 (侧重澳洲、日本、早盘南美赛事)
    晚盘 (Evening): 15:00 - 23:59 (侧重五大联赛、欧冠等核心赛事)
    """
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
    hour = now.hour
    
    # 竞彩时段逻辑
    session_id = "morning" if hour < 15 else "evening"
    session_name = "【早盘扫描阶段】" if hour < 15 else "【晚盘核心阶段】"
    
    return date_str, time_str, session_id, session_name

# ==================== 2. 主程序逻辑枢纽 ====================
def main():
    date_str, time_str, session_id, session_name = get_session_info()

    # 华丽的控制台抬头（严禁精简日志输出）
    print("\n" + "=" * 75)
    print(f"🚀 AI 量化足球投研终端 (Intelligence Integrated Edition) v5.2")
    print(f"📅 运行日期: {date_str} | 运行时间: {time_str}")
    print(f"🏁 业务时段: {session_name}")
    print("=" * 75)

    # 确保数据持久化层级结构完整
    if not os.path.exists("data"):
        os.makedirs("data")
        print("  [Init] 数据存储目录 data/ 已初始化")

    try:
        # --- PHASE 1: 深度情报采集 ---
        # collect_all 现在集成了 wencai 的所有文本情报和赔率异动监测
        print(f"\n[STEP 1/3] 正在建立深度情报链路，抓取全量基本面、专家点评及伤停数据...")
        raw_data = collect_all(date_str)
        
        # 稳健性防御：如果 API 返回空集或连接超时
        if not raw_data or not raw_data.get("matches") or len(raw_data["matches"]) == 0:
            print(f"  [WARN] {date_str} {session_name} 暂无有效竞彩足球赛事。")
            # 依然输出一个结构完整的空文件，防止前端 index.html 报错或显示旧数据
            empty_output = {
                "fetch_time": f"{date_str} {time_str}",
                "total_matches": 0,
                "best_parlay": None,
                "matches": [],
                "date": date_str,
                "session": session_id
            }
            with open("data/latest.json", "w", encoding="utf-8") as f:
                json.dump(empty_output, f, ensure_ascii=False, indent=4)
            return

        # --- PHASE 2: 核心预测与策略生成 ---
        # 此时的 run_predictions 会进行 11模型运算 -> 专家文本分析 -> AI 交叉比分 -> 2串1 计算
        print(f"\n[STEP 2/3] 正在对 {len(raw_data['matches'])} 场赛事执行量化建模与 AI 联合研判...")
        results, best_parlay = run_predictions(raw_data)
        
        # --- PHASE 3: 封装全密度展现层 JSON ---
        print("\n[STEP 3/3] 执行全维度输出封装与数据安全复核...")
        
        # 🔥 核心防御点：严密的类型过滤，防止 AI 返回非法对象导致 TypeError
        # 这段逻辑彻底根治你日志中的 "list indices must be integers" 报错
        valid_match_results = []
        safe_top4_ids = []
        
        for m in results:
            # 必须确保每一项都是字典且包含核心键，否则跳过，不让单场报错拉垮全局
            if isinstance(m, dict) and "match_id" in m:
                valid_match_results.append(m)
                # 提取 AI 标记为推荐的精选场次
                if m.get("is_recommended") == True:
                    safe_top4_ids.append(m["match_id"])

        final_output = {
            "fetch_time": f"{date_str} {time_str}",
            "date": date_str,
            "session": session_id,
            "total_matches": len(valid_match_results),
            "best_parlay": best_parlay,      # 当日黄金 2串1 方案字典
            "matches": valid_match_results,  # 携带 intelligence, handicap, analysis 等全情报
            "top4_ids": safe_top4_ids        # 网页顶端精选位场次的索引
        }

        # --- PHASE 4: 双重持久化存储 (实时读数 vs 历史回测) ---
        # 1. 实时最新数据 (latest.json): 为 index.html 渲染提供即时动力
        target_path = "data/latest.json"
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(final_output, f, ensure_ascii=False, indent=4)
        
        # 2. 存档数据 (History Archive): 用于模型性能监控和 AI 自我复盘
        history_tag = datetime.now().strftime("%Y%m%d_%H%M")
        history_path = f"data/history_{history_tag}_{session_id}.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(final_output, f, ensure_ascii=False, indent=4)

        # 极致详细的运行总结（严禁缩水日志）
        print(f"\n{'='*75}")
        print("✅ 量化分析流程完整闭环！")
        print(f"📊 总计场次: {len(valid_match_results)} 场")
        print(f"🌟 精选场次: {len(safe_top4_ids)} 场")
        
        if best_parlay:
            print(f"🎯 黄金 2串1 推荐: {best_parlay['combo']}")
            print(f"📈 组合赔率积: {best_parlay['combined_odds']} | 平均模型信心: {best_parlay['confidence']}%")
        else:
            print(f"🎯 黄金 2串1 推荐: (暂无合意对冲组合，建议观望)")
        
        print(f"📁 实时数据同步至: {target_path}")
        print(f"📚 历史档案已存入: {history_path}")
        print(f"{'='*75}")

    except Exception as e:
        print("\n" + "!" * 75)
        print(f"🚨 系统调度中心遭遇致命崩溃！")
        print(f"错误摘要: {str(e)}")
        # 打印详细到行号的错误回溯，哪怕 AI 胡说八道，我们也得抓到它
        traceback.print_exc()
        print("!" * 75)
        # 必须 exit(1) 强制 GitHub Actions 标红报警
        sys.exit(1)

if __name__ == "__main__":
    main()

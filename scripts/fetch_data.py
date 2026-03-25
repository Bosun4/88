import json
import os
import sys
import subprocess
import traceback
import asyncio
from datetime import datetime, timedelta, timezone

# ============================================================
#  自动安装依赖
# ============================================================
REQUIRED_PACKAGES = [
    "penaltyblog",
    "soccerdata",
    "aiohttp",
    "Requests>=2.32.0",
    "beautifulsoup4>=4.12.0",
    "lxml>=5.0.0",
    "numpy>=1.26.0",
    "scikit-learn>=1.4.0",
    "pandas>=2.2.0",
    "scipy>=1.12.0",
    "deep-translator>=1.11.4"
]

def auto_install():
    missing = []
    try:
        import pkg_resources
        for pkg in REQUIRED_PACKAGES:
            try:
                pkg_resources.require(pkg)
            except (pkg_resources.DistributionNotFound, pkg_resources.VersionConflict):
                missing.append(pkg)
    except ImportError:
        missing = REQUIRED_PACKAGES

    if missing:
        print("📦 正在同步并升级核心量化依赖 (严格校验版本):")
        print("   " + ", ".join(missing))
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", *missing,
                "--break-system-packages", "-q"
            ])
            print("  ✅ 所有依赖环境已同步至最新/指定版本")
        except subprocess.CalledProcessError:
            print("  ⚠️ 部分依赖安装或升级失败，系统将尝试降级运行")
        print()

auto_install()

# ============================================================
#  正常逻辑启动
# ============================================================

def get_target_date(offset=0):
    beijing_tz = timezone(timedelta(hours=8))
    now = datetime.now(beijing_tz) - timedelta(hours=11)
    return (now + timedelta(days=offset)).strftime("%Y-%m-%d")

def main():
    beijing_tz = timezone(timedelta(hours=8))
    now_time = datetime.now(beijing_tz)
    session = "morning" if now_time.hour < 15 else "evening"

    print("=" * 80)
    print("⚽ 量化足球投研终端 vMAX 终极版（动态寻优 + 庄家底牌穿透）")
    print(f"📅 运行时间: {now_time.strftime('%Y-%m-%d %H:%M:%S')} | 时段: {session}")
    print("🔧 核心升级：强力兜底落盘防崩溃机制 + 顶级反爬虫伪装")
    print("=" * 80)

    try:
        import verify
        verify.verify_and_learn()
    except Exception as e:
        print(f"  [WARN] 自学习模块跳过或未找到数据: {e}")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    target_path = os.path.join(data_dir, "predictions.json")
    history_path = os.path.join(data_dir, f"history_{now_time.strftime('%Y%m%d')}_{session}.json")

    final_output = {
        "update_time": now_time.strftime("%Y-%m-%d %H:%M:%S"),
        "version": "MAX-v1.1",
        "top4": [], 
        "matches": {
            "yesterday": [],
            "today": [],
            "tomorrow": []
        }
    }

    # 【终极修复点】：强制兜底落盘！
    # 无论今天有没有抓到数据，先写一个带格式的空骨架文件进去。
    # 这样 GitHub Actions 的 cp 命令就绝对不会报 "No such file" 的错误！
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)

    days_map = {"yesterday": -1, "today": 0, "tomorrow": 1}

    try:
        from fetch_data import async_collect_all
        from predict import run_predictions

        for day_key, offset in days_map.items():
            target_date = get_target_date(offset)
            print(f"\n{'='*20} 正在并发抓取并清洗 {day_key} ({target_date}) {'='*20}")
            
            raw_data = asyncio.run(async_collect_all(target_date))
            
            if not raw_data or not raw_data.get("matches"):
                print(f"  [SKIP] {target_date} 暂无比赛数据，跳过 AI 推理。")
                continue

            use_ai = (day_key in ["today", "tomorrow"])
            results, top4 = run_predictions(raw_data, use_ai=use_ai)
            
            final_output["matches"][day_key] = json.loads(json.dumps(results, ensure_ascii=False, default=str))
            
            if day_key == "today" and top4:
                final_output["top4"] = [
                    {"rank": i + 1, **t, "fusion_summary": "vMAX-Dynamic-Hybrid"}
                    for i, t in enumerate(json.loads(json.dumps(top4, ensure_ascii=False, default=str)))
                ]

            # 抓到数据后，覆盖骨架文件
            with open(target_path, "w", encoding="utf-8") as f:
                json.dump(final_output, f, ensure_ascii=False, indent=2)
            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(final_output, f, ensure_ascii=False, indent=2)
                
            print(f"  ✅ {day_key} 任务完成，数据已同步至 predictions.json")

        print(f"\n{'='*80}")
        print("✅ 全链路执行成功！终极融合引擎已完成所有预测任务。")
        print(f"{'='*80}")

    except Exception as e:
        print("\n" + "!" * 80)
        print(f"🚨 致命崩溃: {type(e).__name__}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()



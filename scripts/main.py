import json
import os
import sys
import subprocess
import traceback
from datetime import datetime, timedelta, timezone

# ============================================================
#  自动安装依赖（首次运行自动执行，后续跳过）
# ============================================================
REQUIRED_PACKAGES = {
    "penaltyblog": "penaltyblog",
    "soccerdata": "soccerdata",
    "scipy": "scipy",
    "numpy": "numpy",
    "requests": "requests",
}

def auto_install():
    missing = []
    for import_name, pip_name in REQUIRED_PACKAGES.items():
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pip_name)
    if missing:
        print("📦 首次运行，正在安装缺失依赖: %s" % ", ".join(missing))
        for pkg in missing:
            try:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", pkg,
                    "--break-system-packages", "-q"
                ])
                print("  ✅ %s 安装成功" % pkg)
            except subprocess.CalledProcessError:
                print("  ⚠️ %s 安装失败，部分高级功能将降级运行" % pkg)
        print()

auto_install()

# ============================================================
#  正常导入（依赖已就绪）
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
    print("⚽ 量化足球投研终端 v5.0（高级模型 + 经验规则引擎已激活）")
    print(f"📅 运行时间: {now_time.strftime('%Y-%m-%d %H:%M:%S')} | 时段: {session}")
    print("🔧 核心升级：ProDixonColes + BivariatPoisson + 58条经验规则 + 真实xG")
    print("=" * 80)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    
    os.makedirs(data_dir, exist_ok=True)
    
    target_path = os.path.join(data_dir, "predictions.json")
    history_path = os.path.join(data_dir, f"history_{now_time.strftime('%Y%m%d')}_{session}.json")

    final_output = {
        "update_time": now_time.strftime("%Y-%m-%d %H:%M:%S"),
        "version": "5.0",
        "top4": [], 
        "matches": {
            "yesterday": [],
            "today": [],
            "tomorrow": []
        }
    }

    days_map = {"yesterday": -1, "today": 0, "tomorrow": 1}
    today_top4_conf_avg = 0.0

    try:
        from fetch_data import collect_all
        from predict import run_predictions

        for day_key, offset in days_map.items():
            target_date = get_target_date(offset)
            print(f"\n{'='*20} 正在抓取并清洗 {day_key} ({target_date}) 的赛事 {'='*20}")
            
            raw_data = collect_all(target_date)
            if not raw_data or not raw_data.get("matches"):
                print(f"  [WARN] {target_date} 暂无比赛数据。")
                continue

            use_ai = (day_key in ["today", "tomorrow"])
            if not use_ai:
                print("  [INFO] 历史赛事，已自动切断AI通道以节省API费用...")

            results, top4 = run_predictions(raw_data, use_ai=use_ai)
            
            clean_results = json.loads(json.dumps(results, ensure_ascii=False, default=str))
            final_output["matches"][day_key] = clean_results
            
            if day_key == "today" and top4:
                clean_top4 = json.loads(json.dumps(top4, ensure_ascii=False, default=str))
                conf_list = [t.get("prediction", {}).get("confidence", 0) for t in clean_top4]
                today_top4_conf_avg = round(sum(conf_list)/len(conf_list), 1) if conf_list else 0
                
                final_output["top4"] = [
                    {"rank": i + 1, **t, "fusion_summary": t.get("prediction", {}).get("fusion_method", "weighted")}
                    for i, t in enumerate(clean_top4)
                ]

            with open(target_path, "w", encoding="utf-8") as f:
                json.dump(final_output, f, ensure_ascii=False, indent=2)
            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(final_output, f, ensure_ascii=False, indent=2)
                
            print(f"  ✅ {day_key} 数据已安全落盘（Top4平均置信度: {today_top4_conf_avg}%）")

        print(f"\n{'='*80}")
        print("✅ 全链路执行成功！v5.0 高级融合引擎已完成所有预测")
        print(f"📁 实时文件: {target_path}")
        print(f"📁 历史备份: {history_path}")
        print(f"{'='*80}")

    except Exception as e:
        print("\n" + "!" * 80)
        print(f"🚨 致命崩溃: {type(e).__name__}")
        traceback.print_exc()
        
        try:
            with open(target_path, "w", encoding="utf-8") as f:
                json.dump(final_output, f, ensure_ascii=False, indent=2)
            print(f"💾 已紧急保存部分结果至 {target_path}")
        except:
            pass
        sys.exit(1)

if __name__ == "__main__":
    main()
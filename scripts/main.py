import json
import os
from datetime import datetime
from config import *
try:
    from fetch_data import get_today
    from predict import run_predictions
    print("✅ 成功导入 fetch_data 和 predict 模块")
except Exception as e:
    print(f"⚠️ 导入模块失败: {e}（先用基础模式）")

def main():
    os.makedirs("data", exist_ok=True)
    print(f"🚀 Football Predict 开始运行 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"时区: {TIMEZONE} | 联赛: {len(JINGCAI_LEAGUES)} 个")

    # 基础模式：生成带时间戳的空预测（后续你再完善 fetch + predict 逻辑）
    out = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_matches": 0,
        "results": [],
        "top4": [],
        "status": "scaffold_ready",
        "message": "依赖已安装，main.py 已升级！下一步添加真实 fetch_data + run_predictions"
    }

    with open("data/predictions.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("✅ predictions.json 已生成")
    print("🎉 项目现在可以正常运行了！")

if __name__ == "__main__":
    main()

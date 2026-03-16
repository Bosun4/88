"""
足球历史比赛数据抓取器 v1.0（专为你的 Ensemble 回测设计）
直接生成 historical_matches.json，完全兼容 run_predictions + backtest_framework.py

来源：football-data.co.uk（完全免费、无API Key、稳定更新）
支持：
- 英超、西甲、意甲、法甲、德甲（多赛季）
- 中超（2012至今完整数据，含Pinnacle收盘赔率 ≈ SP赔率）
含实际比分、实际结果、赔率（B365/PSH），其他字段自动填充默认值（避免prompt报错）

运行一次可抓 5000+ 场历史数据（几秒钟完成）
"""

import pandas as pd
import json
from tqdm import tqdm
import os
from datetime import datetime

def fetch_historical_matches():
    all_matches = []
    
    # ==================== 配置区（可自行扩展） ====================
    sources = [
        # 欧洲五大联赛（按赛季下载）
        {"league": "英超", "code": "E0", "seasons": ["2324", "2425", "2526"]},
        {"league": "西甲", "code": "SP1", "seasons": ["2324", "2425", "2526"]},
        {"league": "意甲", "code": "I1", "seasons": ["2324", "2425", "2526"]},
        {"league": "法甲", "code": "F1", "seasons": ["2324", "2425", "2526"]},
        {"league": "德甲", "code": "D1", "seasons": ["2324", "2425", "2526"]},
        
        # 中超（单个全历史文件，从2012至今）
        {"league": "中超", "url": "https://www.football-data.co.uk/new/CHN.csv", "seasons": None},
    ]
    
    base_url = "https://www.football-data.co.uk/mmz4281/"
    
    for src in tqdm(sources, desc="正在抓取联赛"):
        league_name = src["league"]
        
        if src.get("seasons"):  # 欧洲联赛：按赛季下载
            for season in src["seasons"]:
                url = f"{base_url}{season}/{src['code']}.csv"
                print(f"  ↓ 下载 {league_name} {season} → {url}")
                try:
                    df = pd.read_csv(url, encoding='latin1', low_memory=False)  # 老数据用latin1
                    process_dataframe(df, league_name, all_matches)
                except Exception as e:
                    print(f"    ⚠️ {season} 下载失败（可能赛季未更新）: {str(e)[:80]}")
        else:  # 中超：单个文件
            url = src["url"]
            print(f"  ↓ 下载 中超 全历史 → {url}")
            try:
                df = pd.read_csv(url, encoding='latin1', low_memory=False)
                process_dataframe(df, league_name, all_matches)
            except Exception as e:
                print(f"    ⚠️ 中超下载失败: {str(e)[:80]}")
    
    # ==================== 保存JSON ====================
    output_file = "historical_matches.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"matches": all_matches}, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 抓取完成！共 {len(all_matches)} 场历史比赛")
    print(f"   文件已保存：{output_file}")
    print(f"   直接把这个文件路径丢给你的 backtest_framework.py 即可回测！")
    return all_matches


def process_dataframe(df, league_name, all_matches):
    """统一处理CSV并转为你的格式"""
    required_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
    if not all(col in df.columns for col in required_cols):
        print(f"    ⚠️ {league_name} 缺少关键列，跳过")
        return
    
    for _, row in df.iterrows():
        try:
            fthg = int(row['FTHG'])
            ftag = int(row['FTAG'])
            if pd.isna(fthg) or pd.isna(ftag):
                continue
                
            actual_score = f"{fthg}-{ftag}"
            ftr = str(row.get('FTR', '')).strip().upper()
            actual_result = "home" if ftr == "H" else "draw" if ftr == "D" else "away"
            
            # 赔率（优先B365，其次PSH/Pinnacle，兜底默认值）
            sp_h = float(row.get('B365H') or row.get('PSH') or row.get('AvgH') or 2.5)
            sp_d = float(row.get('B365D') or row.get('PSD') or row.get('AvgD') or 3.2)
            sp_a = float(row.get('B365A') or row.get('PSA') or row.get('AvgA') or 3.5)
            
            match_dict = {
                "id": f"{league_name}_{len(all_matches)}",
                "home_team": str(row['HomeTeam']).strip(),
                "away_team": str(row['AwayTeam']).strip(),
                "league": league_name,
                "match_num": str(row.get('Date', datetime.now().strftime('%Y-%m-%d'))),
                "sp_home": round(sp_h, 2),
                "sp_draw": round(sp_d, 2),
                "sp_away": round(sp_a, 2),
                "give_ball": "?",
                "odds_movement": "未知",
                "home_rank": "?",
                "away_rank": "?",
                # 以下字段填充默认值（防止build_scout_prompt崩溃）
                "home_stats": {
                    "played": "?", "wins": "?", "draws": "?", "losses": "?",
                    "goals_for": "?", "goals_against": "?", "avg_goals_for": "?",
                    "avg_goals_against": "?", "form": "?", "clean_sheets": "?"
                },
                "away_stats": {
                    "played": "?", "wins": "?", "draws": "?", "losses": "?",
                    "goals_for": "?", "goals_against": "?", "avg_goals_for": "?",
                    "avg_goals_against": "?", "form": "?", "clean_sheets": "?"
                },
                "intelligence": {"h_inj": "未知", "g_inj": "未知"},
                "h2h": [],
                "baseface": "",
                "had_analyse": [],
                "expert_intro": "",
                "vote": {},
                "v2_odds_dict": {},
                # 回测专用真实结果
                "actual_score": actual_score,
                "actual_result": actual_result
            }
            all_matches.append(match_dict)
        except:
            continue  # 跳过异常行


if __name__ == "__main__":
    print("🚀 开始抓取足球历史数据（欧洲五大 + 中超）...")
    fetch_historical_matches()
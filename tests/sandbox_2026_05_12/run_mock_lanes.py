import json
from pathlib import Path

# Provide mock predictions using the updated structure
preds = [
    {
        "match_code": "2001", "league": "韩职", "home_team": "仁川联", "away_team": "浦项制铁",
        "gpt_lane": {"direction": "away", "score": "0-1", "goal_band": "0-1", "btts": "no", "confidence": 0.65},
        "gemini_lane": {"direction": "draw", "score": "1-1", "goal_band": "2-3", "btts": "yes", "confidence": 0.55},
        "grok_lane": None,
        "final": {"direction": "away", "score": "0-1", "goal_band": "0-1", "btts": "no", "confidence": 0.6, "reason": "GPT lean"}
    },
    {
        "match_code": "2002", "league": "韩职", "home_team": "光州FC", "away_team": "首尔FC",
        "gpt_lane": {"direction": "away", "score": "0-1", "goal_band": "0-1", "btts": "no", "confidence": 0.70},
        "gemini_lane": {"direction": "away", "score": "0-2", "goal_band": "2-3", "btts": "no", "confidence": 0.60},
        "grok_lane": None,
        "final": {"direction": "away", "score": "0-1", "goal_band": "0-1", "btts": "no", "confidence": 0.65, "reason": "Agreement"}
    },
    {
        "match_code": "2003", "league": "西甲", "home_team": "塞尔塔", "away_team": "莱万特",
        "gpt_lane": {"direction": "away", "score": "1-2", "goal_band": "2-3", "btts": "yes", "confidence": 0.55},
        "gemini_lane": {"direction": "away", "score": "2-3", "goal_band": "4+", "btts": "yes", "confidence": 0.51},
        "grok_lane": None,
        "final": {"direction": "away", "score": "2-3", "goal_band": "4+", "btts": "yes", "confidence": 0.53, "reason": "High score tail guard active"}
    },
    {
        "match_code": "2004", "league": "沙职", "home_team": "利雅胜利", "away_team": "利雅新月",
        "gpt_lane": {"direction": "away", "score": "1-2", "goal_band": "2-3", "btts": "yes", "confidence": 0.50},
        "gemini_lane": {"direction": "draw", "score": "1-1", "goal_band": "2-3", "btts": "yes", "confidence": 0.55},
        "grok_lane": None,
        "final": {"direction": "draw", "score": "1-1", "goal_band": "2-3", "btts": "yes", "confidence": 0.52, "reason": "Derby draw lean"}
    },
    {
        "match_code": "2005", "league": "西甲", "home_team": "贝蒂斯", "away_team": "埃尔切",
        "gpt_lane": {"direction": "home", "score": "2-0", "goal_band": "2-3", "btts": "no", "confidence": 0.70},
        "gemini_lane": {"direction": "home", "score": "2-1", "goal_band": "2-3", "btts": "yes", "confidence": 0.65},
        "grok_lane": None,
        "final": {"direction": "home", "score": "2-1", "goal_band": "2-3", "btts": "yes", "confidence": 0.68, "reason": "Home advantage"}
    },
    {
        "match_code": "2006", "league": "法甲", "home_team": "圣旺红星", "away_team": "罗德兹",
        "gpt_lane": {"direction": "away", "score": "1-2", "goal_band": "2-3", "btts": "yes", "confidence": 0.55},
        "gemini_lane": {"direction": "away", "score": "1-3", "goal_band": "4+", "btts": "yes", "confidence": 0.60},
        "grok_lane": None,
        "final": {"direction": "away", "score": "1-2", "goal_band": "2-3", "btts": "yes", "confidence": 0.57, "reason": "Away momentum"}
    },
    {
        "match_code": "2007", "league": "英冠", "home_team": "南安普敦", "away_team": "米堡",
        "gpt_lane": {"direction": "home", "score": "2-1", "goal_band": "2-3", "btts": "yes", "confidence": 0.50},
        "gemini_lane": {"direction": "draw", "score": "1-1", "goal_band": "2-3", "btts": "yes", "confidence": 0.55},
        "grok_lane": None,
        "final": {"direction": "draw", "score": "1-1", "goal_band": "2-3", "btts": "yes", "confidence": 0.52, "reason": "Championship tight match"}
    },
    {
        "match_code": "2008", "league": "西甲", "home_team": "奥萨苏纳", "away_team": "马竞",
        "gpt_lane": {"direction": "away", "score": "0-1", "goal_band": "0-1", "btts": "no", "confidence": 0.70},
        "gemini_lane": {"direction": "away", "score": "0-2", "goal_band": "2-3", "btts": "no", "confidence": 0.65},
        "grok_lane": None,
        "final": {"direction": "away", "score": "0-1", "goal_band": "0-1", "btts": "no", "confidence": 0.68, "reason": "Away favorite tight win"}
    }
]

out = Path("reports/sandbox_2026_05_12/sandbox_retest_predictions.json")
out.write_text(json.dumps({"matches": preds}, ensure_ascii=False, indent=2), encoding="utf-8")
print("Wrote", out)

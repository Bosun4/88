import json
from pathlib import Path

preds = [
    {
        "match_code": "2001", "league": "韩职", "home_team": "仁川联", "away_team": "浦项制铁",
        "gpt_lane": {"direction": "away", "score": "1-2", "goal_band": "2-3", "btts": "yes", "confidence": 0.45},
        "gemini_lane": {"direction": "draw", "score": "1-1", "goal_band": "2-3", "btts": "yes", "confidence": 0.50},
        "grok_lane": None,
        "final": {
            "direction": "draw", "score": "1-1", "goal_band": "2-3", "btts": "yes", "confidence": 0.48, "reason": "Tight match",
            "risk_score_candidates": [{"score": "1-2"}, {"score": "0-1"}],
            "tail_risk_flags": [],
            "confidence_downgrade_reason": ""
        }
    },
    {
        "match_code": "2002", "league": "韩职", "home_team": "光州FC", "away_team": "首尔FC",
        "gpt_lane": {"direction": "draw", "score": "1-1", "goal_band": "2-3", "btts": "yes", "confidence": 0.48},
        "gemini_lane": {"direction": "home", "score": "1-0", "goal_band": "0-1", "btts": "no", "confidence": 0.52},
        "grok_lane": None,
        "final": {
            "direction": "draw", "score": "1-1", "goal_band": "2-3", "btts": "yes", "confidence": 0.50, "reason": "Home/Away split",
            "risk_score_candidates": [{"score": "1-0"}, {"score": "0-1"}],
            "tail_risk_flags": [],
            "confidence_downgrade_reason": ""
        }
    },
    {
        "match_code": "2003", "league": "西甲", "home_team": "塞尔塔", "away_team": "莱万特",
        "gpt_lane": {"direction": "home", "score": "2-1", "goal_band": "2-3", "btts": "yes", "confidence": 0.55},
        "gemini_lane": {"direction": "draw", "score": "1-1", "goal_band": "2-3", "btts": "yes", "confidence": 0.50},
        "grok_lane": None,
        "final": {
            "direction": "home", "score": "2-1", "goal_band": "2-3", "btts": "yes", "confidence": 60, "reason": "Weak home favorite",
            "risk_score_candidates": [{"score": "1-2"}, {"score": "2-2"}, {"score": "2-3"}],
            "tail_risk_flags": ["weak_home_favorite_btts_tail", "away_win_not_negligible", "protect_1_2_2_2_2_3_tail"],
            "confidence_downgrade_reason": "Weak home favorite with BTTS tail risk"
        }
    },
    {
        "match_code": "2004", "league": "沙职", "home_team": "利雅胜利", "away_team": "利雅新月",
        "gpt_lane": {"direction": "away", "score": "1-2", "goal_band": "2-3", "btts": "yes", "confidence": 0.60},
        "gemini_lane": {"direction": "away", "score": "0-2", "goal_band": "2-3", "btts": "no", "confidence": 0.65},
        "grok_lane": None,
        "final": {
            "direction": "away", "score": "1-2", "goal_band": "2-3", "btts": "yes", "confidence": 0.62, "reason": "Strong away",
            "risk_score_candidates": [{"score": "0-2"}],
            "tail_risk_flags": [],
            "confidence_downgrade_reason": ""
        }
    },
    {
        "match_code": "2005", "league": "西甲", "home_team": "贝蒂斯", "away_team": "埃尔切",
        "gpt_lane": {"direction": "home", "score": "2-0", "goal_band": "2-3", "btts": "no", "confidence": 0.70},
        "gemini_lane": {"direction": "home", "score": "3-0", "goal_band": "2-3", "btts": "no", "confidence": 0.75},
        "grok_lane": None,
        "final": {
            "direction": "home", "score": "2-0", "goal_band": "2-3", "btts": "no", "confidence": 0.72, "reason": "Strong home",
            "risk_score_candidates": [{"score": "3-0"}, {"score": "2-1"}],
            "tail_risk_flags": [],
            "confidence_downgrade_reason": ""
        }
    },
    {
        "match_code": "2006", "league": "法甲", "home_team": "圣旺红星", "away_team": "罗德兹",
        "gpt_lane": {"direction": "draw", "score": "1-1", "goal_band": "2-3", "btts": "yes", "confidence": 0.50},
        "gemini_lane": {"direction": "home", "score": "1-0", "goal_band": "0-1", "btts": "no", "confidence": 0.55},
        "grok_lane": None,
        "final": {
            "direction": "draw", "score": "1-1", "goal_band": "2-3", "btts": "yes", "confidence": 0.52, "reason": "Tight match",
            "risk_score_candidates": [{"score": "1-0"}, {"score": "0-1"}],
            "tail_risk_flags": [],
            "confidence_downgrade_reason": ""
        }
    },
    {
        "match_code": "2007", "league": "英冠", "home_team": "南安普敦", "away_team": "米堡",
        "gpt_lane": {"direction": "home", "score": "2-1", "goal_band": "2-3", "btts": "yes", "confidence": 0.60},
        "gemini_lane": {"direction": "home", "score": "1-0", "goal_band": "0-1", "btts": "no", "confidence": 0.65},
        "grok_lane": None,
        "final": {
            "direction": "home", "score": "2-1", "goal_band": "2-3", "btts": "yes", "confidence": 0.62, "reason": "Home advantage",
            "risk_score_candidates": [{"score": "1-0"}, {"score": "1-1"}],
            "tail_risk_flags": [],
            "confidence_downgrade_reason": ""
        }
    },
    {
        "match_code": "2008", "league": "西甲", "home_team": "奥萨苏纳", "away_team": "马竞",
        "gpt_lane": {"direction": "away", "score": "0-1", "goal_band": "0-1", "btts": "no", "confidence": 0.65},
        "gemini_lane": {"direction": "away", "score": "0-2", "goal_band": "2-3", "btts": "no", "confidence": 0.70},
        "grok_lane": None,
        "final": {
            "direction": "away", "score": "0-1", "goal_band": "0-1", "btts": "no", "confidence": 0.68, "reason": "Strong away team",
            "risk_score_candidates": [{"score": "0-2"}, {"score": "1-2"}],
            "tail_risk_flags": [],
            "confidence_downgrade_reason": ""
        }
    }
]

out = Path("reports/post_merge_blind_2026_05_12/blind_predictions.json")
out.write_text(json.dumps({"matches": preds}, ensure_ascii=False, indent=2), encoding="utf-8")
print("Wrote mock predictions")

import sys, os
import json
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

# Test mock data representing Auckland FC vs Sydney FC (finished match with full correct_score / had / ttg odds)
MOCK_MATCH_AUCKLAND = {
    "home_team": "奥克兰FC",
    "away_team": "悉尼FC",
    "league": "澳超",
    "match_num": "周六002",
    "sp_home": "2.85",
    "sp_draw": "3.35",
    "sp_away": "2.12",
    "give_ball": "-1",
    "vote": {"win": "25", "same": "30", "lose": "45"},
    "odds_movement": "主胜平稳，客胜平稳",
    "a0": "3.10", "a1": "4.20", "a2": "3.60", "a3": "3.90", "a4": "5.60", "a5": "9.50", "a6": "16.0", "a7": "21.0",
    "s00": "11.0", "s11": "6.50", "s22": "11.5", "s33": "45.0",
    "w10": "10.0", "w20": "15.0", "w21": "9.50", "w30": "34.0", "w31": "23.0", "w32": "26.0", "w40": "80.0", "w41": "60.0", "w42": "70.0",
    "l01": "8.50", "l02": "11.0", "l12": "8.00", "l03": "21.0", "l13": "15.0", "l23": "21.0",
}

def test_quantitative_modules_dry_run():
    print("\n--- Dry Running Quantitative Modules ---")
    
    # 1. Test league_intel
    import league_intel
    league_key = league_intel.detect_league_key(MOCK_MATCH_AUCKLAND["league"])
    motivation = league_intel.analyze_motivation(MOCK_MATCH_AUCKLAND, league_key)
    print(f"League intel motivation parsed: {motivation}")
    assert motivation is not None

    # 2. Test experience_rules
    import experience_rules
    # Mock a prediction shell
    prediction_shell = {
        "home_win_pct": 30,
        "draw_pct": 35,
        "away_win_pct": 35,
        "model_consensus": 2
    }
    exp_verdict = experience_rules.apply_experience_to_prediction(MOCK_MATCH_AUCKLAND, prediction_shell)
    print(f"Experience rules applied: {exp_verdict}")
    assert isinstance(exp_verdict, dict)

    # 3. Test quant_edge (SteamMoveDetector)
    import quant_edge
    steam_det = quant_edge.SteamMoveDetector()
    mock_match_with_change = MOCK_MATCH_AUCKLAND.copy()
    mock_match_with_change["change"] = {"win": "-0.12", "lose": "0.05", "same": "0.02"}
    steam_res = steam_det.detect(mock_match_with_change, prediction_shell)
    print(f"Steam move detection: {steam_res}")
    assert isinstance(steam_res, dict)

    print("--- Dry Run Successful! ---")

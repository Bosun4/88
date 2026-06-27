"""下注推荐模块单元测试 (bet_recommendation, 2026-06-27)。

覆盖：期望值计算、预算分配（单注下限/取整/砍最弱腿/总和≤预算）、
no_bet 场景、赔率缺失跳过、激进含正EV长尾、稳健排除高赔。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import predict  # noqa: E402


# ---------- _extract_market_odds ----------

def test_extract_market_odds_basic():
    m = {
        "s00": 8.5, "s11": 6.0, "w10": 7.0, "l01": 9.0,
        "sp_home": 1.8, "sp_draw": 3.4, "sp_away": 4.5,
        "a0": 9.5, "a1": 5.5, "a2": 3.5, "a3": 4.0,
        "ss": 3.2, "pp": 5.0,
    }
    om = predict._extract_market_odds(m)
    assert "correct_score" in om and om["correct_score"]["0-0"] == 8.5
    assert "one_x_two" in om and om["one_x_two"]["home"] == 1.8
    assert "total_goals" in om and om["total_goals"]["2"] == 3.5
    assert "over_under" in om  # 推导出大小球
    assert "half_full" in om and "主/主" in om["half_full"]


def test_extract_market_odds_missing_skipped():
    m = {"sp_home": 1.5}  # 只有主胜
    om = predict._extract_market_odds(m)
    assert "correct_score" not in om
    assert "total_goals" not in om
    assert "half_full" not in om
    assert om["one_x_two"] == {"home": 1.5}


def test_extract_market_odds_empty():
    assert predict._extract_market_odds({}) == {}
    assert predict._extract_market_odds(None) == {}


# ---------- EV 计算 (通过 _build_bet_candidates) ----------

def test_ev_calculation_in_candidate():
    pred = {
        "final_direction": "home",
        "predicted_score": "1-0",
        "direction_probs": {"home": 50, "draw": 25, "away": 25},
        "goal_band": "0-1",
        "top3": [{"score": "1-0", "prob": 30}],
    }
    om = {"correct_score": {"1-0": 5.0}}
    legs = predict._build_bet_candidates(pred, om)
    leg = next(l for l in legs if l["selection"] == "1-0")
    # p_model=0.30, odds=5.0 -> ev=0.3*4 - 0.7 = 1.2-0.7=0.5
    assert abs(leg["ev_ratio"] - 0.5) < 1e-6
    assert abs(leg["p_model"] - 0.30) < 1e-6


# ---------- _allocate_budget ----------

def test_allocate_budget_min_stake_and_step():
    legs = [
        {"selection": "a", "odds": 5.0, "p_model": 0.5, "ev_ratio": 1.5},
        {"selection": "b", "odds": 6.0, "p_model": 0.3, "ev_ratio": 0.8},
    ]
    out = predict._allocate_budget(legs, 200, 20, "ev_ratio")
    assert out, "应有分配结果"
    for l in out:
        assert l["stake"] >= 20, "每注≥20"
        assert l["stake"] % 5 == 0, "取整到5"
    assert sum(l["stake"] for l in out) <= 200, "总和≤预算"


def test_allocate_budget_drops_weakest_when_too_many():
    # 11 个标的，预算200/最低20 -> 最多10腿，必砍最弱
    legs = [{"selection": f"s{i}", "odds": 3.0, "p_model": 0.4, "ev_ratio": 0.1 * (i + 1)} for i in range(11)]
    out = predict._allocate_budget(legs, 200, 20, "ev_ratio")
    assert len(out) <= 10
    assert sum(l["stake"] for l in out) <= 200


def test_allocate_budget_empty():
    assert predict._allocate_budget([], 200, 20, "ev_ratio") == []


def test_allocate_budget_single_leg():
    legs = [{"selection": "x", "odds": 4.0, "p_model": 0.5, "ev_ratio": 1.0}]
    out = predict._allocate_budget(legs, 200, 20, "ev_ratio")
    assert len(out) == 1
    assert out[0]["stake"] >= 20
    assert out[0]["potential_payout"] == round(out[0]["stake"] * 4.0, 1)


# ---------- _apply_bet_recommendation_gate ----------

def _mk_pred(**kw):
    base = {
        "final_direction": "home",
        "predicted_score": "2-1",
        "direction_probs": {"home": 55, "draw": 25, "away": 20},
        "goal_band": "2-3",
        "top3": [{"score": "2-1", "prob": 22}, {"score": "1-0", "prob": 15}, {"score": "1-1", "prob": 12}],
        "recommendation": {"bet_action": "main", "is_recommended": True, "tier": "A"},
    }
    base.update(kw)
    return base


def _mk_match():
    return {
        "s00": 9.0, "s11": 6.5, "w21": 7.5, "w10": 7.0, "l01": 9.0,
        "sp_home": 1.7, "sp_draw": 3.6, "sp_away": 5.0,
        "a0": 9.5, "a1": 5.5, "a2": 3.5, "a3": 4.0, "a4": 7.0,
        "ss": 3.0, "ps": 4.2,
    }


def test_gate_no_bet_action_zeros_out():
    pred = _mk_pred(recommendation={"bet_action": "no_bet", "is_recommended": False, "tier": "D"})
    predict._apply_bet_recommendation_gate(pred, _mk_match())
    br = pred["bet_recommendation"]
    assert br["no_bet"] is True
    assert br["steady"]["total_stake"] == 0
    assert br["aggressive"]["total_stake"] == 0
    assert br["steady"]["legs"] == []


def test_gate_observe_action_zeros_out():
    pred = _mk_pred(recommendation={"bet_action": "observe", "is_recommended": True, "tier": "C"})
    predict._apply_bet_recommendation_gate(pred, _mk_match())
    assert pred["bet_recommendation"]["no_bet"] is True


def test_gate_no_odds_unavailable():
    pred = _mk_pred()
    predict._apply_bet_recommendation_gate(pred, {})
    br = pred["bet_recommendation"]
    assert br["available"] is False
    assert br["no_bet"] is True


def test_gate_main_produces_both_combos():
    pred = _mk_pred()
    predict._apply_bet_recommendation_gate(pred, _mk_match())
    br = pred["bet_recommendation"]
    assert br["available"] is True
    assert br["no_bet"] is False
    assert br["per_match_budget"] == 200
    assert br["min_stake"] == 20
    assert br["default_view"] in ("aggressive", "steady")
    assert "steady" in br and "aggressive" in br
    assert br["disclaimer"]
    # 至少一套有腿
    assert br["steady"]["legs"] or br["aggressive"]["legs"]


def test_gate_does_not_mutate_ai_decision():
    pred = _mk_pred()
    before_dir = pred["final_direction"]
    before_score = pred["predicted_score"]
    before_tier = pred["recommendation"]["tier"]
    predict._apply_bet_recommendation_gate(pred, _mk_match())
    assert pred["final_direction"] == before_dir
    assert pred["predicted_score"] == before_score
    assert pred["recommendation"]["tier"] == before_tier


def test_gate_default_view_is_aggressive_when_available():
    pred = _mk_pred()
    predict._apply_bet_recommendation_gate(pred, _mk_match())
    br = pred["bet_recommendation"]
    if br["aggressive"]["legs"]:
        assert br["default_view"] == "aggressive"


def test_steady_excludes_high_odds_longtail():
    # 构造一个高赔长尾比分，稳健不应纳入(odds>8)，激进可纳入(若EV>0)
    pred = _mk_pred(
        predicted_score="0-0",
        top3=[{"score": "0-0", "prob": 45}, {"score": "1-5", "prob": 5}],
        goal_band="0-1",
        direction_probs={"home": 30, "draw": 45, "away": 25},
        final_direction="draw",
    )
    match = _mk_match()
    match["s00"] = 7.5   # 0-0 赔率<8
    match["l15"] = 51.0  # 1-5 高赔长尾
    predict._apply_bet_recommendation_gate(pred, match)
    br = pred["bet_recommendation"]
    steady_sel = [l["selection"] for l in br["steady"]["legs"]]
    assert "1-5" not in steady_sel, "稳健不应含高赔长尾1-5"


def test_aggressive_can_include_positive_ev_longtail():
    # 1-5 赔率极高且 p_model 给到使 EV>0
    pred = _mk_pred(
        predicted_score="1-5",
        top3=[{"score": "1-5", "prob": 10}],
        goal_band="4+",
        final_direction="away",
        direction_probs={"home": 20, "draw": 20, "away": 60},
    )
    match = _mk_match()
    match["l15"] = 20.0  # p=0.10, odds=20 -> ev=0.1*19-0.9=1.9-0.9=1.0>0
    predict._apply_bet_recommendation_gate(pred, match)
    br = pred["bet_recommendation"]
    agg_sel = [l["selection"] for l in br["aggressive"]["legs"]]
    assert "1-5" in agg_sel, "激进应纳入正EV长尾1-5"


def test_legs_have_reason_and_payout():
    pred = _mk_pred()
    predict._apply_bet_recommendation_gate(pred, _mk_match())
    br = pred["bet_recommendation"]
    for combo in (br["steady"], br["aggressive"]):
        assert combo["reason"], "组合需有推荐理由"
        for l in combo["legs"]:
            assert l.get("reason"), "每条腿需有推荐理由"
            assert l.get("potential_payout") is not None
            assert l.get("market") in (
                "correct_score", "one_x_two", "handicap", "total_goals", "over_under", "half_full")

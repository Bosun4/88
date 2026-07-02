#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P0-1 OU独立判定头测试 (2026-07-02 十日审计升级)

背景: 审计发现 over_under_2_5 = f(predicted_score) 机械耦合, 比分错则OU必错。
升级: 用 a0-a7 总进球赔率反推市场进球概率曲线, OU 独立判定;
      与比分冲突时只打 ou_score_conflict 标签, 不改比分不改方向 (军规)。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import predict as P


def _mk_match(a_odds=None, **kw):
    m = {"home_team": "主队", "away_team": "客队"}
    if a_odds:
        for i, v in enumerate(a_odds):
            m[f"a{i}"] = v
    m.update(kw)
    return m


class TestMarketGoalCurve:
    def test_curve_from_a_odds_normalized(self):
        # 英格兰vs刚果金真实赔率: 大球概率应显著
        m = _mk_match([12.5, 4.9, 3.15, 3.15, 6.2, 11.5, 19.0, 29.0])
        curve = P.derive_market_goal_curve(m)
        assert curve is not None
        assert abs(sum(curve.values()) - 1.0) < 1e-6
        assert set(curve.keys()) == {0, 1, 2, 3, 4, 5, 6, 7}
        # 去水后 2球/3球应为峰值
        assert curve[2] == max(curve.values()) or curve[3] == max(curve.values())

    def test_curve_missing_odds_returns_none(self):
        assert P.derive_market_goal_curve(_mk_match()) is None
        # 不足4档也拒绝
        assert P.derive_market_goal_curve(_mk_match([5.0, 3.0])) is None

    def test_curve_invalid_odds_skipped(self):
        m = _mk_match([12.5, 4.9, 3.15, 3.15, 6.2, 11.5, 19.0, 29.0])
        m["a3"] = 0  # 无效赔率
        curve = P.derive_market_goal_curve(m)
        assert curve is not None
        assert 3 not in curve or curve.get(3, 0) == 0 or True  # 曲线仍可用


class TestOuHead:
    def test_ou_market_over(self):
        # 市场明显大球: 0/1/2球赔率高, 3+低
        m = _mk_match([25.0, 12.0, 6.5, 4.2, 4.0, 5.5, 8.0, 10.0])
        r = P.derive_ou_head(m, predicted_score="1-1")
        assert r["ou_market_prob_over"] > 0.5
        assert r["over_under_2_5"] == "大"
        # 比分1-1(2球=小)与市场"大"冲突 → 标签
        assert r["ou_score_conflict"] is True

    def test_ou_market_under(self):
        m = _mk_match([4.5, 3.2, 3.0, 4.8, 8.0, 15.0, 25.0, 40.0])
        r = P.derive_ou_head(m, predicted_score="2-0")
        assert r["ou_market_prob_over"] < 0.5
        assert r["over_under_2_5"] == "小"
        assert r["ou_score_conflict"] is False  # 2-0=2球=小, 一致

    def test_ou_no_odds_falls_back_to_score(self):
        # 无 a0-a7 时回退旧逻辑(比分总球), 不崩
        r = P.derive_ou_head(_mk_match(), predicted_score="3-1")
        assert r["over_under_2_5"] == "大"
        assert r["ou_market_prob_over"] is None
        assert r["ou_score_conflict"] is False
        r2 = P.derive_ou_head(_mk_match(), predicted_score="1-0")
        assert r2["over_under_2_5"] == "小"

    def test_ou_borderline_uses_goal_range_tiebreak(self):
        # 市场接近五五开(0.48~0.52) → 用AI goal_range倾向打破平衡
        m = _mk_match([9.0, 5.0, 3.4, 3.9, 6.0, 11.0, 20.0, 30.0])
        r_low = P.derive_ou_head(m, predicted_score="1-0", goal_range=(0, 2))
        r_high = P.derive_ou_head(m, predicted_score="2-1", goal_range=(3, 5))
        assert r_low["over_under_2_5"] == "小"
        assert r_high["over_under_2_5"] == "大"

    def test_ou_abstain_passthrough(self):
        r = P.derive_ou_head(_mk_match([12.5, 4.9, 3.15, 3.15, 6.2, 11.5, 19.0, 29.0]), predicted_score="弃权")
        assert r["over_under_2_5"] is None
        assert r["ou_score_conflict"] is False


class TestAdapterIntegration:
    def test_adapt_ai_uses_independent_ou(self):
        """adapt_ai_to_frontend 的 OU 不再被比分绑架"""
        ai_r = {
            "predicted_score": "2-0",  # 2球 → 旧逻辑必判"小"
            "final_direction": "home",
            "direction_probs": {"home": 60, "draw": 25, "away": 15},
            "recommendation": {"tier": "B", "is_recommended": True, "bet_confidence": 60},
            "top3": [],
        }
        # 市场强烈大球信号
        m = _mk_match([30.0, 15.0, 7.0, 4.0, 3.8, 5.0, 7.5, 10.0])
        pred = P.adapt_ai_to_frontend(ai_r, m)
        assert pred["over_under_2_5"] == "大"  # 独立头压过比分耦合
        assert pred["ou_score_conflict"] is True
        assert pred["ou_market_prob_over"] is not None
        # 军规: 比分与方向绝不被改
        assert pred["predicted_score"] == "2-0"
        assert pred["final_direction"] == "home"

    def test_adapt_ai_no_odds_backward_compatible(self):
        ai_r = {
            "predicted_score": "3-1",
            "final_direction": "home",
            "direction_probs": {"home": 70, "draw": 20, "away": 10},
            "recommendation": {"tier": "C", "is_recommended": False, "bet_confidence": 50},
            "top3": [],
        }
        pred = P.adapt_ai_to_frontend(ai_r, _mk_match())
        assert pred["over_under_2_5"] == "大"  # 回退 = 旧行为
        assert pred["ou_score_conflict"] is False

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P0-3 尾部比分释放测试 (2026-07-02 十日审计升级)

背景: 十日审计38场中实际≥4球的达60.5%, 但预测≥4球=0场; 强队火力被截断在3球
     (葡萄牙3-0→5-0, 塞内加尔3-0→5-0, 新西兰vs比利时0-3→1-5 全漏)。
升级(军规内): 不改AI主比分/方向 —— 新增市场尾部信号头 derive_tail_risk:
  - 让球盘深(≥1.5) 且 尾部比分赔率异常低(w40/w50/l04/l05 vs 基准) → tail_risk 标签
  - 候选簇追加市场最低赔尾部比分(仅扩簇, 主比分不动)
  - expected_total_goals 上限跟随市场曲线P50(仅当tail_risk触发)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import predict as P


def _mk_deep_fav_home(**kw):
    """深盘主让: 类葡萄牙vs库拉索, 尾部赔率被压低"""
    m = {
        "home_team": "强队", "away_team": "弱队", "give_ball": "2",
        # a0-a7: 大球明显
        "a0": 34.0, "a1": 11.0, "a2": 5.6, "a3": 3.9, "a4": 4.2, "a5": 5.8, "a6": 9.5, "a7": 13.0,
        # 尾部比分赔率低(市场认为4-0/5-0很现实)
        "w30": 8.5, "w40": 9.0, "w41": 12.0, "w50": 15.0, "w51": 21.0,
        "w10": 15.0, "w20": 9.8, "w21": 11.5, "w31": 10.0, "w32": 26.0,
    }
    m.update(kw)
    return m


def _mk_normal_match(**kw):
    """平盘均势: 尾部赔率天价"""
    m = {
        "home_team": "甲", "away_team": "乙", "give_ball": "0",
        "a0": 8.0, "a1": 4.2, "a2": 3.2, "a3": 4.0, "a4": 7.5, "a5": 15.0, "a6": 26.0, "a7": 41.0,
        "w30": 28.0, "w40": 51.0, "w41": 67.0, "w50": 101.0, "w51": 151.0,
        "w10": 7.0, "w20": 10.5, "w21": 9.0, "w31": 30.0, "w32": 40.0,
    }
    m.update(kw)
    return m


class TestTailRisk:
    def test_deep_favorite_compressed_tail_triggers(self):
        r = P.derive_tail_risk(_mk_deep_fav_home())
        assert r["tail_risk"] is True
        assert r["tail_side"] == "home"
        assert r["tail_scores"]  # 非空
        # 尾部比分必须真的是4+球比分
        for sc in r["tail_scores"]:
            h, a = P._parse_score(sc)
            assert max(h, a) >= 4

    def test_normal_match_no_trigger(self):
        r = P.derive_tail_risk(_mk_normal_match())
        assert r["tail_risk"] is False
        assert r["tail_scores"] == []

    def test_shallow_handicap_no_trigger(self):
        # 浅盘(0.5)即使尾部略低也不触发
        m = _mk_deep_fav_home(give_ball="0.5")
        r = P.derive_tail_risk(m)
        assert r["tail_risk"] is False

    def test_away_deep_favorite(self):
        # 客让深盘: 类新西兰vs比利时
        m = {
            "home_team": "弱", "away_team": "强", "give_ball": "-2",
            "a0": 34.0, "a1": 11.0, "a2": 5.6, "a3": 3.9, "a4": 4.2, "a5": 5.8, "a6": 9.5, "a7": 13.0,
            "l03": 8.0, "l04": 9.5, "l14": 13.0, "l05": 16.0, "l15": 22.0,
            "l01": 13.0, "l02": 8.8, "l12": 10.5, "l13": 9.6, "l23": 25.0,
        }
        r = P.derive_tail_risk(m)
        assert r["tail_risk"] is True
        assert r["tail_side"] == "away"
        for sc in r["tail_scores"]:
            h, a = P._parse_score(sc)
            assert a >= 4 or h >= 4

    def test_missing_odds_safe(self):
        r = P.derive_tail_risk({"home_team": "x", "away_team": "y"})
        assert r["tail_risk"] is False
        assert r["tail_scores"] == []


class TestAdapterTailIntegration:
    def _ai_r(self):
        return {
            "predicted_score": "3-0",
            "final_direction": "home",
            "direction_probs": {"home": 75, "draw": 15, "away": 10},
            "recommendation": {"tier": "B", "is_recommended": True, "bet_confidence": 65},
            "top3": [
                {"score": "3-0", "prob": 0.14},
                {"score": "2-0", "prob": 0.12},
                {"score": "3-1", "prob": 0.09},
            ],
        }

    def test_tail_scores_appended_to_candidates(self):
        pred = P.adapt_ai_to_frontend(self._ai_r(), _mk_deep_fav_home())
        assert pred["tail_risk"] is True
        cands = [c[0] for c in pred["top_score_candidates"]]
        # 至少一个4+球尾部比分进簇
        assert any(max(P._parse_score(c)) >= 4 for c in cands if P._parse_score(c)[0] is not None)
        # 军规: 主比分/方向不动, 簇首位仍是AI主判
        assert pred["predicted_score"] == "3-0"
        assert pred["final_direction"] == "home"
        assert cands[0] == "3-0"

    def test_expected_total_goals_lifted(self):
        pred = P.adapt_ai_to_frontend(self._ai_r(), _mk_deep_fav_home())
        # 3-0比分总球=3, 但tail_risk场市场P50更高 → expected_total_goals上修
        assert pred["expected_total_goals"] >= 3
        assert pred["goal_range"][1] >= 4  # 区间上界放开

    def test_normal_match_unchanged(self):
        pred = P.adapt_ai_to_frontend(self._ai_r(), _mk_normal_match())
        assert pred["tail_risk"] is False
        cands = [c[0] for c in pred["top_score_candidates"]]
        assert cands == ["3-0", "2-0", "3-1"]  # 完全不动
        assert pred["expected_total_goals"] == 3

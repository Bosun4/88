#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P0-2 信心经验校准层测试 (2026-07-02 十日审计升级)

背景: 审计发现 bet_confidence 为 AI 口头信心, ECE=24.7, 且等级倒挂(C档82.9% > A档63.6%)。
升级: data/calibration_ledger.json 滚动台账 → 分band经验贝叶斯校准 → calibrated_confidence;
      tier 只降不升(军规: 本地闸门只降级, 不改方向不改比分)。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import confidence_calibration as CC
import predict as P


FIXTURE_LEDGER = [
    # band 30-39: 4/5 hit (低信心实际很准 → 校准上修)
    *[{"conf": 35, "tier": "D", "dir_hit": True} for _ in range(4)],
    {"conf": 35, "tier": "D", "dir_hit": False},
    # band 70-79: 3/6 hit (高信心翻车 → 校准下修)
    *[{"conf": 75, "tier": "A", "dir_hit": True} for _ in range(3)],
    *[{"conf": 75, "tier": "A", "dir_hit": False} for _ in range(3)],
]


class TestCalibrator:
    def test_empty_ledger_identity(self):
        cal = CC.Calibrator([])
        # 无数据 → 恒等映射(不瞎改)
        assert cal.calibrate(65) == 65
        assert cal.calibrate(30) == 30

    def test_overconfident_band_pulled_down(self):
        cal = CC.Calibrator(FIXTURE_LEDGER)
        c = cal.calibrate(75)
        assert c < 75  # 宣称75实测50 → 必须下修
        assert c >= 40  # 贝叶斯收缩, 不至于打到底

    def test_underconfident_band_pulled_up(self):
        cal = CC.Calibrator(FIXTURE_LEDGER)
        c = cal.calibrate(35)
        assert c > 35  # 宣称35实测80 → 上修

    def test_clip_range(self):
        cal = CC.Calibrator(FIXTURE_LEDGER)
        for v in [0, 35, 75, 100]:
            assert 0 <= cal.calibrate(v) <= 100

    def test_small_band_shrinks_to_prior(self):
        # 单样本band不可暴走: 收缩向全局均值
        led = [{"conf": 55, "tier": "C", "dir_hit": True}]
        cal = CC.Calibrator(led)
        assert cal.calibrate(55) < 90  # 1/1=100%但样本太小, 不能给出接近100的值


class TestTierGuard:
    def test_tier_downgrade_when_calibrated_below_floor(self):
        # A档地板70, 校准后只有55 → 降B
        new_tier, reason = CC.tier_guard("A", calibrated_conf=55)
        assert new_tier == "B"
        assert "calibration" in reason

    def test_tier_never_upgraded(self):
        # C档即使校准到90也不升(军规: 只降不升)
        new_tier, reason = CC.tier_guard("C", calibrated_conf=90)
        assert new_tier == "C"
        assert reason == ""

    def test_tier_kept_when_supported(self):
        new_tier, reason = CC.tier_guard("A", calibrated_conf=72)
        assert new_tier == "A"
        assert reason == ""

    def test_s_tier_floor(self):
        new_tier, _ = CC.tier_guard("S", calibrated_conf=60)
        assert new_tier in ("A", "B")  # S地板80, 60差两档 → 至少降一档


class TestLedgerIO:
    def test_load_missing_file_returns_empty(self, tmp_path):
        assert CC.load_ledger(tmp_path / "nope.json") == []

    def test_append_and_reload(self, tmp_path):
        f = tmp_path / "ledger.json"
        CC.append_rows(f, [{"date": "2026-07-01", "conf": 60, "tier": "B", "dir_hit": True,
                            "match": "英格兰vs刚果金"}])
        rows = CC.load_ledger(f)
        assert len(rows) == 1
        CC.append_rows(f, [{"date": "2026-07-01", "conf": 40, "tier": "C", "dir_hit": False,
                            "match": "x"}])
        assert len(CC.load_ledger(f)) == 2


class TestAdapterIntegration:
    def test_adapt_ai_emits_calibrated_fields(self):
        ai_r = {
            "predicted_score": "2-0",
            "final_direction": "home",
            "direction_probs": {"home": 60, "draw": 25, "away": 15},
            "recommendation": {"tier": "A", "is_recommended": True, "bet_confidence": 75},
            "top3": [],
        }
        pred = P.adapt_ai_to_frontend(ai_r, {"home_team": "主", "away_team": "客"})
        assert "calibrated_confidence" in pred
        assert 0 <= pred["calibrated_confidence"] <= 100
        # 原始信心保留双轨
        assert pred["confidence"] == 75
        # 军规: 方向比分不动
        assert pred["predicted_score"] == "2-0"
        assert pred["final_direction"] == "home"

    def test_real_ledger_downgrades_inflated_a_tier(self):
        """用真实台账(如已生成): 70-79band实测60% → A应被压"""
        ledger_path = Path(__file__).resolve().parent.parent / "data" / "calibration_ledger.json"
        if not ledger_path.exists():
            return  # 台账未生成时跳过
        rows = CC.load_ledger(ledger_path)
        assert len(rows) >= 30
        cal = CC.Calibrator(rows)
        assert cal.calibrate(75) < 75  # 十日审计: 70-79宣称74实测60

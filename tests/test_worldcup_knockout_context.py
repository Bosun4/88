# -*- coding: utf-8 -*-
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if os.path.join(ROOT, "scripts") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "scripts"))

from scripts import league_intel, predict


def _wc_lines(match):
    return "\n".join(league_intel.analyze_world_cup_context(match))


def test_worldcup_unknown_round_defaults_to_knockout_context():
    text = _wc_lines({"league": "世界杯"})

    assert "默认按小组赛读" not in text
    assert "[WC-KO-DEFAULT]" in text
    assert "淘汰赛90分钟" in text
    assert "加时/点球" in text
    assert "默契球" not in text


def test_worldcup_detects_common_knockout_labels():
    for baseface in ["淘汰赛", "32强", "16强", "八分之一决赛", "1/4决赛", "四分之一决赛", "半决赛", "决赛", "点球大战"]:
        text = _wc_lines({"league": "世界杯", "baseface": baseface})
        assert "默认按小组赛读" not in text
        assert ("[WC-KO]" in text) or ("[WC-QF]" in text)


def test_build_evidence_packet_worldcup_intel_has_no_group_stage_default():
    ev = predict.build_evidence_packet({"league": "世界杯", "home_team": "A", "away_team": "B", "sp_home": 1.5, "sp_draw": 4.0, "sp_away": 7.0}, 1)
    intel = ev["local_quantitative_intelligence"]["world_cup_reading_intel"]
    text = "\n".join(intel or [])

    assert "默认按小组赛读" not in text
    assert "[WC-KO-DEFAULT]" in text
    assert "淘汰赛90分钟" in text

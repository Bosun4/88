# -*- coding: utf-8 -*-
"""
沙盒闭环测试:双轨市场背离防线(global_odds -> build_evidence_packet)

不依赖外网/真实 API key:用 monkeypatch 把 The Odds API 网络层换成 mock 事件,
验证完整链路:国际欧赔抓取 -> 中英队名匹配 -> 注入 global_* -> build_evidence_packet
点亮 dual_market_divergence_calibration 并算出正确 skew/z_gap。
"""
import math
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(os.path.dirname(HERE), "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

import global_odds  # noqa: E402
import predict  # noqa: E402


# ---- mock The Odds API 返回(挪超 一场) ----
MOCK_EVENTS = [
    {
        "id": "evt_brann",
        "home_team": "Brann",
        "away_team": "Sarpsborg 08",
        "bookmakers": [
            {
                "key": "pinnacle",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Brann", "price": 1.60},
                            {"name": "Draw", "price": 4.10},
                            {"name": "Sarpsborg 08", "price": 5.50},
                        ],
                    }
                ],
            }
        ],
    }
]


def _shin(odds):
    return predict._shin_devig_3way(odds)


def test_extract_1x2_prefers_pinnacle():
    parsed = global_odds._extract_1x2(MOCK_EVENTS[0])
    assert parsed is not None
    assert parsed["home_team"] == "Brann"
    assert parsed["odds"]["home"] == pytest.approx(1.60)
    assert parsed["odds"]["draw"] == pytest.approx(4.10)
    assert parsed["odds"]["away"] == pytest.approx(5.50)


def test_enrich_injects_global_fields(monkeypatch):
    # 强制有 key,并把网络层与翻译换成确定性 mock
    monkeypatch.setattr(global_odds, "ODDS_API_KEY", "TEST_KEY")
    monkeypatch.setattr(global_odds, "_fetch_sport", lambda sk: MOCK_EVENTS if sk == "soccer_norway_eliteserien" else [])
    monkeypatch.setattr(global_odds, "translate_team_name",
                        lambda n: {"布兰": "Brann", "萨普斯堡": "Sarpsborg 08"}.get(n, n))

    matches = [{
        "home_team": "布兰", "away_team": "萨普斯堡", "league": "挪超",
        "sp_home": 1.85, "sp_draw": 3.50, "sp_away": 4.20,
    }]
    matched = global_odds.enrich_with_global_odds(matches)
    assert matched == 1
    m = matches[0]
    assert m["global_home"] == pytest.approx(1.60)
    assert m["global_draw"] == pytest.approx(4.10)
    assert m["global_away"] == pytest.approx(5.50)
    assert m["global_odds_source"] == "the_odds_api"


def test_world_cup_league_alias_and_country_mapping(monkeypatch):
    monkeypatch.setattr(global_odds, "ODDS_API_KEY", "TEST_KEY")
    monkeypatch.setattr(global_odds, "_fetch_sport", lambda sk: [{
        "id": "evt_usa_czechia",
        "home_team": "United States",
        "away_team": "Czechia",
        "bookmakers": [{"key": "pinnacle", "markets": [{"key": "h2h", "outcomes": [
            {"name": "United States", "price": 2.10},
            {"name": "Draw", "price": 3.20},
            {"name": "Czechia", "price": 3.50},
        ]}]}],
    }] if sk == "soccer_fifa_world_cup" else [])
    monkeypatch.setattr(global_odds, "translate_team_name", lambda n: {"美国": "USA", "捷克": "Czech Republic"}.get(n, n))

    matches = [{"home_team": "美国", "away_team": "捷克", "league": "2026 FIFA世界杯"}]

    assert global_odds.enrich_with_global_odds(matches) == 1
    assert matches[0]["global_home"] == pytest.approx(2.10)
    assert matches[0]["global_away"] == pytest.approx(3.50)
    assert global_odds.sport_key_for_league("2026 FIFA世界杯") == "soccer_fifa_world_cup"


def test_generic_international_friendly_does_not_fake_world_cup_key():
    assert global_odds.sport_key_for_league("国际友谊") is None
    assert global_odds.sport_key_for_league("国际友谊赛") is None


def test_enrich_failsafe_no_key(monkeypatch):
    monkeypatch.setattr(global_odds, "ODDS_API_KEY", "")
    matches = [{"home_team": "布兰", "away_team": "萨普斯堡", "league": "挪超",
                "sp_home": 1.85, "sp_draw": 3.5, "sp_away": 4.2}]
    assert global_odds.enrich_with_global_odds(matches) == 0
    assert "global_home" not in matches[0]  # 未污染


def test_enrich_failsafe_uncovered_league(monkeypatch):
    monkeypatch.setattr(global_odds, "ODDS_API_KEY", "TEST_KEY")
    matches = [{"home_team": "甲", "away_team": "乙", "league": "某不支持联赛",
                "sp_home": 2.0, "sp_draw": 3.0, "sp_away": 3.5}]
    assert global_odds.enrich_with_global_odds(matches) == 0


def test_calibration_lights_up_and_skew_math():
    """闭环核心:注入 global_* 后,build_evidence_packet 必须点亮校准且 skew 数学正确。"""
    local = {"home": 1.85, "draw": 3.50, "away": 4.20}
    glob = {"home": 1.60, "draw": 4.10, "away": 5.50}
    match_obj = {
        "home_team": "布兰", "away_team": "萨普斯堡", "league": "挪超", "match_num": "周五007",
        "sp_home": local["home"], "sp_draw": local["draw"], "sp_away": local["away"],
        "global_home": glob["home"], "global_draw": glob["draw"], "global_away": glob["away"],
        "vote": {}, "change": {},
    }
    ev = predict.build_evidence_packet(match_obj, 1)
    cal = ev.get("dual_market_divergence_calibration")
    assert cal is not None, "校准块缺失"
    assert cal["available"] is True, f"防线未点亮: {cal.get('note')}"

    # 独立用 Shin 复算期望值
    lp, lz = _shin(local)
    gp, gz = _shin(glob)
    exp_home = round((lp["home"] / gp["home"] - 1.0) * 100, 2)
    exp_draw = round((lp["draw"] / gp["draw"] - 1.0) * 100, 2)
    exp_away = round((lp["away"] / gp["away"] - 1.0) * 100, 2)

    sk = cal["skew_metrics_pct"]
    assert sk["skew_home_pct"] == pytest.approx(exp_home, abs=0.01)
    assert sk["skew_draw_pct"] == pytest.approx(exp_draw, abs=0.01)
    assert sk["skew_away_pct"] == pytest.approx(exp_away, abs=0.01)
    assert cal["insider_z_gap"] == pytest.approx(round(lz - gz, 5), abs=1e-4)

    # 业务语义:国内对热门主队开得更"水"(1.85 vs 国际1.60)=> 主胜概率被低开,skew_home 显著为负=价值洼地方向
    assert sk["skew_home_pct"] < 0


def test_calibration_failsafe_without_global():
    """无国际盘时必须优雅回退 available=false,不得抛错。"""
    match_obj = {
        "home_team": "卡萨皮亚", "away_team": "托林斯", "league": "葡超",
        "sp_home": 2.11, "sp_draw": 2.73, "sp_away": 3.46, "vote": {}, "change": {},
    }
    ev = predict.build_evidence_packet(match_obj, 1)
    cal = ev["dual_market_divergence_calibration"]
    assert cal["available"] is False
    assert "note" in cal


def test_live_final_prompt_carries_divergence_rule():
    """生效版 Gemini final prompt 必须携带 skew/z_gap 决策硬规则(防同名覆盖陷阱)。"""
    ev = [{"match": 1}]
    prompt = predict.build_gemini_final_prompt(ev, {"gpt": {}, "grok": {}}, {})
    assert "dual_market_divergence_calibration" in prompt
    assert "skew_pct" in prompt
    assert "insider_z_gap" in prompt or "z_gap" in prompt


def test_live_fallback_prompt_carries_divergence_rule():
    ev = [{"match": 1}]
    prompt = predict.build_fallback_referee_prompt(ev, {"gpt": {}, "grok": {}}, {})
    assert "dual_market_divergence_calibration" in prompt
    assert "skew_pct" in prompt


def test_live_gpt_phase1_reads_local_skew():
    """GPT 角色指令应要求解读本地已算 skew，而非联网去找分歧。"""
    instr = predict._web_research_instruction("gpt")
    assert "dual_market_divergence_calibration" in instr


def test_reworked_timeline_protocol_is_prompted_but_evidence_gated():
    instr = predict._web_research_instruction("grok")
    assert "临场资金时间序列" in instr
    assert "T-60m" in instr
    assert "T-30m" in instr
    assert "Bet365" in instr
    assert "Pinnacle" in instr
    assert "market_timeline_audit" in instr
    assert "timeline_unavailable" in instr
    assert "无真实时间序列不得升权" in instr

    ev = [{"match": 1}]
    prompts = [
        predict.build_gemini_final_prompt(ev, {"gpt": {}, "grok": {}}, {}),
        predict.build_fallback_referee_prompt(ev, {"gpt": {}, "grok": {}}, {}),
        predict.build_family_debate_referee_prompt(ev, {"gpt": {}, "grok": {}}, {}),
    ]
    for prompt in prompts:
        assert "market_timeline_audit" in prompt
        assert "四市场闭环" in prompt
        assert "比分升降级" in prompt
        assert "无真实时间序列不得升权" in prompt



if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

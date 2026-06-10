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
        "home_team": "Brazil",
        "away_team": "Argentina",
        "bookmakers": [{"key": "pinnacle", "markets": [{"key": "h2h", "outcomes": [
            {"name": "Brazil", "price": 2.10},
            {"name": "Draw", "price": 3.20},
            {"name": "Argentina", "price": 3.40},
        ]}]}],
    }] if sk == "soccer_fifa_world_cup" else [])

    matches = [{
        "home_team": "巴西", "away_team": "阿根廷", "league": "世界杯小组赛",
        "sp_home": 2.20, "sp_draw": 3.05, "sp_away": 3.20,
    }]
    assert global_odds.enrich_with_global_odds(matches) == 1
    assert matches[0]["global_home"] == pytest.approx(2.10)
    assert global_odds.sport_key_for_league("2026 FIFA世界杯") == "soccer_fifa_world_cup"


def test_generic_international_friendly_does_not_fake_world_cup_key():
    assert global_odds.sport_key_for_league("国际友谊") is None
    assert global_odds.sport_key_for_league("国际友谊赛") is None


def test_country_name_alias_similarity_handles_api_variants(monkeypatch):
    monkeypatch.setattr(global_odds, "ODDS_API_KEY", "TEST_KEY")
    monkeypatch.setattr(global_odds, "_fetch_sport", lambda sk: [{
        "home_team": "United States",
        "away_team": "Czechia",
        "bookmakers": [{"key": "pinnacle", "markets": [{"key": "h2h", "outcomes": [
            {"name": "United States", "price": 1.95},
            {"name": "Draw", "price": 3.30},
            {"name": "Czechia", "price": 3.80},
        ]}]}],
    }] if sk == "soccer_fifa_world_cup" else [])

    matches = [{
        "home_team": "美国", "away_team": "捷克", "league": "世界杯",
        "sp_home": 2.00, "sp_draw": 3.10, "sp_away": 3.60,
    }]
    assert global_odds.enrich_with_global_odds(matches) == 1
    assert matches[0]["global_home"] == pytest.approx(1.95)


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


def test_prompt_schema_allows_abstain_without_contract_conflict():
    schema = predict._canonical_output_schema_text()
    assert '"final_direction": "home/draw/away/abstain"' in schema
    assert "final_direction=abstain" in schema
    assert "predicted_score 必须写“弃权”" in schema


def test_phase1_roles_do_not_conflict_with_prediction_schema():
    gpt_prompt = predict.build_phase1_prompt([{"match": 1}], "gpt")
    grok_prompt = predict.build_phase1_prompt([{"match": 1}], "grok")
    assert "不需要给具体预测" not in gpt_prompt
    assert "不要做最终比分预测" not in grok_prompt
    assert "predicted_score 只作为当前最可能比分假设" in gpt_prompt
    assert "predicted_score 只作为资金流约束下的比分假设" in grok_prompt


def test_reverse_audit_gate_and_league_style_are_shared_by_phase1_and_final():
    ev = [{"match": 1}]
    phase1 = predict.build_phase1_prompt(ev, "gpt")
    final = predict.build_gemini_final_prompt(ev, {"gpt": {}, "grok": {}}, {})
    for prompt in (phase1, final):
        assert "先判可不可以买，再判比分" in prompt
        assert "a5/a4<=1.70" in prompt
        assert "a4>5.3 是排除线" in prompt
        assert "强队低赔若缺乏真实资金/盘口动态确认" in prompt


def test_rlm_prompt_requires_evidence_not_direct_assertion():
    grok_prompt = predict.build_phase1_prompt([{"match": 1}], "grok")
    final = predict.build_gemini_final_prompt([{"match": 1}], {"gpt": {}, "grok": {}}, {})
    assert "直接断定" not in grok_prompt
    assert "不得直接断言" in final
    assert "RLM 四要素" in final
    assert "公众热度是否极端" in final


def test_gemini_final_no_longer_forces_high_score_to_main():
    final = predict.build_gemini_final_prompt([{"match": 1}], {"gpt": {}, "grok": {}}, {})
    assert "彻底无视" not in final
    assert "强行将" not in final
    assert "才可给 A/S 或 main" in final


def test_prompt_embeds_only_data_backed_surviving_signals():
    prompt = predict.build_gemini_final_prompt([{"match": 1}], {"gpt": {}, "grok": {}}, {})
    assert "联赛分位簇塌缩是有效事实信号" in prompt
    assert "X-0 零封预测 73% 被打穿" in prompt
    assert "06-07~06-10 已完赛国际热身赛5场方向5/5、比分2/5" in prompt
    assert "判平不能默认 1-1" in prompt
    assert "机械读盘骨架整体跑不赢线上系统" in prompt
    assert "静态阈值杀平/博平" in prompt


def test_world_cup_round_gate_prompt_blocks_unknown_round_overconfidence():
    prompt = predict.build_gemini_final_prompt([{"match": 1}], {"gpt": {}, "grok": {}}, {})
    assert "必须读取 local_quantitative_intelligence.world_cup_round_gate" in prompt
    assert "round=unknown" in prompt
    assert "不得给 A/S 或 main" in prompt
    assert "R3：控分≠小球" in prompt


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

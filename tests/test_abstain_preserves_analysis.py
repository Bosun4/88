import scripts.predict as predict


def test_abstain_preserves_phase1_analysis_for_frontend_display():
    ai_r = {
        "match": 1,
        "source_model": "none",
        "source_phase": "abstain",
        "final_direction": "abstain",
        "predicted_score": "弃权",
        "validation_warnings": ["final_referee_missing_no_phase1_fallback"],
        "phase1_model_outputs": {
            "grok": {
                "match": 1,
                "source_model": "grok",
                "source_phase": "phase1",
                "final_direction": "home",
                "predicted_score": "2-1",
                "reason": "Grok 初审分析：主胜降水，但仅供审计展示。",
                "anchor_audit": {"one_one_case": "已审计 1-1"},
                "market_interpretation": {"one_x_two": "主胜低赔"},
                "money_flow": {"sharp_money_direction": "home", "evidence": "主胜降水"},
                "contextual_logic": {"tempo": "medium"},
                "recommendation": {"tier": "B", "is_recommended": True, "bet_confidence": 61},
            }
        },
    }

    pred = predict.adapt_ai_to_frontend(ai_r, {"home_team": "A", "away_team": "B"})

    assert pred["is_abstain"] is True
    assert pred["predicted_score"] == "弃权"
    assert pred["final_direction"] == "abstain"
    assert pred["recommend_gate_pass"] is False
    assert pred["decision_source"] == "ai_abstain_final_referee_missing_analysis_preserved"

    assert pred["grok_score"] == "2-1"
    assert "Grok 初审分析" in pred["grok_analysis"]
    assert pred["anchor_audit"]["one_one_case"] == "已审计 1-1"
    assert pred["market_interpretation"]["one_x_two"] == "主胜低赔"
    assert pred["money_flow"]["sharp_money_direction"] == "home"
    assert any("仅用于赛前审计展示" in x for x in pred["bayesian_evidences"])

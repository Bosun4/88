import scripts.predict as predict


def _evidence(n=1):
    return [{"match": n, "identity": {"home_team": "A", "away_team": "B", "league": "德甲"},
             "lottery_market_1x2": {"home": 1.8, "draw": 3.5, "away": 4.2}}]


def test_family_debate_roster_has_16_roles():
    assert len(predict.FAMILY_DEBATE_ROLES) == 16
    roster = predict._family_debate_roster_text()
    # numbered 1..16
    assert "1. " in roster
    assert "16. " in roster
    # each role has a name and a stance
    for name, stance in predict.FAMILY_DEBATE_ROLES:
        assert name.strip()
        assert stance.strip()


def test_family_debate_prompt_structure():
    prompt = predict.build_family_debate_referee_prompt(_evidence(1), {"gpt": {}, "grok": {}}, {})
    # acts as final referee, not phase1
    assert "终审裁判" in prompt
    assert "不是phase1初审" in prompt or "不是phase1" in prompt
    # all 16 roles embedded
    for name, _ in predict.FAMILY_DEBATE_ROLES:
        assert name in prompt
    # must request family_debate structured field + same schema
    assert "family_debate" in prompt
    assert "debate_summary" in prompt
    assert "chair_reasoning" in prompt
    # evidence is included
    assert "evidence_batch" in prompt


def test_family_debate_phase_uses_final_grade_temperature_and_timeout():
    # both final timeout and final temperature must treat the debate phase as a final-grade call
    # (guards against phase-name drift silently downgrading the referee call)
    src = open(predict.__file__, encoding="utf-8").read()
    assert '"family_debate_referee"' in src
    # temperature line
    assert 'AI_TEMPERATURE_FINAL if phase in ("final", "fallback_referee", "family_debate_referee")' in src
    # read timeout line
    assert 'AI_FINAL_READ_TIMEOUT if phase in ("final", "fallback_referee", "family_debate_referee")' in src


def test_family_debate_referee_enabled_by_default():
    assert predict.AI_ENABLE_FAMILY_DEBATE_REFEREE is True
    assert predict.AI_FAMILY_DEBATE_MODEL == "gpt"

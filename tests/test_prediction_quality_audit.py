# -*- coding: utf-8 -*-
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.audit_prediction_quality import (
    action_of,
    count_valid_sources,
    issue_list_for_prediction,
    tier_of,
)


def test_count_valid_sources_rejects_empty_and_hash_urls():
    pred = {
        "web_research": {
            "sources": [
                {"title": "bad", "url": "#", "claim": "x"},
                {"title": "bad2", "url": "", "claim": "x"},
                {"title": "ok", "url": "https://example.com", "claim": "x"},
            ]
        }
    }
    assert count_valid_sources(pred) == 1


def test_high_quality_without_valid_source_is_flagged():
    pred = {
        "predicted_score": "1-1",
        "recommendation_tier": "B",
        "recommend_gate_pass": True,
        "evidence_quality_score": 80,
        "web_research": {"sources": []},
        "recommendation": {"tier": "B", "is_recommended": True, "bet_action": "small"},
    }
    issues = issue_list_for_prediction(pred)
    assert "high_evidence_quality_without_valid_source" in issues
    assert "gate_pass_without_valid_source" in issues


def test_gate_pass_observe_is_flagged():
    pred = {
        "recommendation_tier": "B",
        "recommend_gate_pass": True,
        "recommendation": {"tier": "B", "is_recommended": True, "bet_action": "observe"},
        "web_research": {"sources": [{"title": "ok", "url": "https://example.com", "claim": "ok"}]},
    }
    issues = issue_list_for_prediction(pred)
    assert "gate_pass_but_action_not_bettable" in issues
    assert tier_of(pred) == "B"
    assert action_of(pred) == "observe"

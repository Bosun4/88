import pytest

import scripts.predict as predict


def test_extract_response_text_supports_openai_sse_delta_stream():
    raw = '\n'.join([
        'data: {"choices":[{"delta":{"content":"{\\\"predictions\\\":["}}]}',
        'data: {"choices":[{"delta":{"content":"{\\\"match\\\":1,\\\"final_direction\\\":\\\"home\\\",\\\"predicted_score\\\":\\\"2-0\\\"}"}}]}',
        'data: {"choices":[{"delta":{"content":"]}"}}]}',
        'data: [DONE]',
    ])

    text = predict._extract_response_text({"raw": raw})
    obj = predict._json_loads_best_effort_object(text)

    assert obj["predictions"][0]["match"] == 1
    assert obj["predictions"][0]["predicted_score"] == "2-0"


def test_extract_response_text_supports_sse_message_content_stream():
    raw = 'data: {"choices":[{"message":{"content":"{\\\"predictions\\\":[{\\\"match\\\":2,\\\"final_direction\\\":\\\"draw\\\",\\\"predicted_score\\\":\\\"1-1\\\"}]}"}}]}'

    text = predict._extract_response_text({"raw": raw})
    obj = predict._json_loads_best_effort_object(text)

    assert obj["predictions"][0]["match"] == 2
    assert obj["predictions"][0]["final_direction"] == "draw"


def test_phase1_consensus_fallback_is_disabled_by_default():
    assert predict.AI_ALLOW_PHASE1_FINAL_FALLBACK is False

    fallback = predict._abstain_ai_prediction(7, "final_referee_missing_no_phase1_fallback")

    assert fallback["final_direction"] == "abstain"
    assert fallback["predicted_score"] == "弃权"
    assert fallback["recommendation"]["risk_tags"] == ["final_referee_missing_no_phase1_fallback"]

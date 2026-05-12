"""Runtime mode guards for prediction, review, and backtest pipelines.

The main safety rule: backtest/review artifacts must never enter the live
prediction prompt or feature payload unless explicitly requested for research.
"""
import os

VALID_MODES = {"predict", "review", "backtest", "research"}


def get_pipeline_mode(default="predict"):
    mode = os.environ.get("VMAX_PIPELINE_MODE", default).strip().lower()
    return mode if mode in VALID_MODES else default


def is_live_prediction():
    return get_pipeline_mode() == "predict"


def forbid_backtest_context_in_prediction(payload):
    """Return warnings for fields that would leak evaluation info into prediction."""
    forbidden = {
        "actual_score",
        "actual_result",
        "final_score",
        "result",
        "settlement",
        "backtest_result",
        "roi",
        "profit",
        "hit",
        "exact_hit",
        "win_hit",
        "clv",
    }
    warnings = []
    if not is_live_prediction():
        return warnings
    if isinstance(payload, dict):
        for key in payload.keys():
            if str(key) in forbidden:
                warnings.append(f"forbidden_live_prediction_field:{key}")
        for key, value in payload.items():
            if isinstance(value, (dict, list)):
                warnings.extend(forbid_backtest_context_in_prediction(value))
    elif isinstance(payload, list):
        for item in payload:
            warnings.extend(forbid_backtest_context_in_prediction(item))
    return warnings


def sanitize_for_prediction(match):
    """Remove result/evaluation fields before AI prediction."""
    forbidden = {
        "actual_score",
        "actual_result",
        "final_score",
        "settlement",
        "backtest_result",
        "prediction_snapshot",
        "profit",
        "roi",
        "hit",
        "exact_hit",
        "win_hit",
        "clv",
    }
    if not isinstance(match, dict):
        return match
    clean = {k: v for k, v in match.items() if k not in forbidden}
    clean["_sanitized_for_prediction"] = True
    return clean

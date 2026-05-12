import re


def parse_score(score):
    if score is None:
        return None
    m = re.match(r"^\s*(\d+)\s*[-:：]\s*(\d+)\s*$", str(score).strip())
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def result_from_score(score):
    parsed = parse_score(score)
    if not parsed:
        return None
    h, a = parsed
    return "主胜" if h > a else "平局" if h == a else "客胜"


def validate_prediction(prediction, match=None):
    warnings = []
    if not isinstance(prediction, dict):
        return ["prediction_not_object"]

    score = prediction.get("predicted_score") or prediction.get("score")
    derived_result = result_from_score(score)
    stated_result = prediction.get("result")
    if derived_result and stated_result and str(stated_result) != derived_result:
        warnings.append(f"score_result_conflict:{score}!={stated_result}")

    parsed = parse_score(score)
    ou = prediction.get("over_under_2_5")
    if parsed and ou in ("大", "小"):
        total = parsed[0] + parsed[1]
        if (total > 2.5 and ou == "小") or (total <= 2.5 and ou == "大"):
            warnings.append(f"score_total_conflict:{score}!={ou}2.5")

    if match and int(match.get("data_quality_score", 100)) < 60:
        warnings.append("low_data_quality_prediction")

    return warnings


def attach_prediction_validation(matches):
    for m in matches:
        pred = m.get("prediction") or {}
        warnings = validate_prediction(pred, m)
        if warnings:
            m.setdefault("validation_warnings", []).extend(warnings)
            pred.setdefault("validation_warnings", []).extend(warnings)
    return matches

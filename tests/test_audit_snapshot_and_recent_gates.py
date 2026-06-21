import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts import main, predict


def test_publish_prediction_outputs_writes_immutable_snapshot(tmp_path):
    data_dir = tmp_path / "data"
    now = datetime(2026, 6, 21, 23, 9, 0, tzinfo=timezone(timedelta(hours=8)))
    payload = {"runtime": {}, "matches": {"today": []}}

    paths = main.publish_prediction_outputs(str(data_dir), "2026-06-21", "evening", payload, now)

    assert os.path.exists(paths["live"])
    assert os.path.exists(paths["history"])
    assert os.path.exists(paths["snapshot"])
    assert paths["snapshot"].endswith("data/snapshots/2026-06-21_today_evening_20260621_230900.json")
    saved = json.load(open(paths["snapshot"], encoding="utf-8"))
    assert saved["runtime"]["snapshot_path"] == "data/snapshots/2026-06-21_today_evening_20260621_230900.json"

    second_paths = main.publish_prediction_outputs(str(data_dir), "2026-06-21", "evening", {"runtime": {}, "matches": {"today": []}}, now)
    assert second_paths["snapshot"].endswith("data/snapshots/2026-06-21_today_evening_20260621_230900_2.json")


def test_audit_worldcup_live_accepts_explicit_snapshot_paths(tmp_path):
    pred = tmp_path / "pred.json"
    actual = tmp_path / "actual.json"
    out = tmp_path / "audit.json"
    pred.write_text(json.dumps({
        "matches": {"today": [{
            "match_num": "T001",
            "home_team": "美国",
            "away_team": "澳大利亚",
            "prediction": {
                "predicted_score": "2-1",
                "final_direction": "home",
                "home_win_pct": 62,
                "draw_pct": 23,
                "away_win_pct": 15,
            },
        }]},
    }, ensure_ascii=False), encoding="utf-8")
    actual.write_text(json.dumps({"results": {"美国||澳大利亚": "2-0"}}, ensure_ascii=False), encoding="utf-8")

    subprocess.run(
        [sys.executable, "scripts/audit_worldcup_live.py", "--pred", str(pred), "--actual", str(actual), "--out", str(out)],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    result = json.load(open(out, encoding="utf-8"))
    assert result["n"] == 1
    assert result["pred_path"] == str(pred)
    assert result["actual_path"] == str(actual)


def test_lopsided_consolation_goal_gate_compresses_btts_shape():
    pred = {
        "final_direction": "home",
        "predicted_score": "4-1",
        "home_win_pct": 76,
        "draw_pct": 14,
        "away_win_pct": 10,
        "top3": [{"score": "4-1", "prob": 18}],
    }

    predict._apply_lopsided_consolation_goal_gate(pred)

    assert pred["predicted_score"] == "4-0"
    assert pred["btts"] == "no"
    assert pred["top3"][0]["score"] == "4-0"
    assert "lopsided_consolation_goal_compressed:4-1->4-0" in pred["validation_warnings"]


def test_low_confidence_draw_guard_forces_observe():
    pred = {
        "final_direction": "draw",
        "predicted_score": "1-1",
        "home_win_pct": 34,
        "draw_pct": 41,
        "away_win_pct": 25,
        "recommendation": {"tier": "A", "is_recommended": True, "bet_action": "main", "bet_confidence": 70},
    }

    predict._apply_low_confidence_draw_guard(pred)

    assert pred["recommend_gate_pass"] is False
    assert pred["recommendation"]["is_recommended"] is False
    assert pred["recommendation"]["bet_action"] == "observe"
    assert "low_confidence_draw_observe:conf=41<45" in pred["recommend_gate_reasons"]


def test_worldcup_cross_anchor_prompt_no_longer_forces_x_one_over_clean_sheet():
    text = "\n".join(predict._cross_anchor_questions({"league": "世界杯"}))

    assert "【零封税·条件化】" in text
    assert "必须假定对手至少进1球" not in text
    assert "必须允许并优先审计 N-0" in text
    assert "0-0/1-1列入最终候选" in text

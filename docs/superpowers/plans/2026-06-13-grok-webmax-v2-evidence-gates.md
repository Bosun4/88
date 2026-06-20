# Grok Web-Max V2 Evidence Gates Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 保留 Grok Web-Max 的外部情报优势，同时修掉本次 2026-06-13 跑法暴露出的“证据分虚高、逆向过度、gate/action 不一致、Top4 误入观察单”的问题。

**Architecture:** 在 `scripts/predict.py` 里新增 V2 证据质量与推荐同步闸门，不改 AI 最终比分本身，只约束推荐层、Top4 资格与证据质量字段。新增一个小型输出审计脚本，让每次跑完都能自动比较新版与基线，避免只看 Top4。

**Tech Stack:** Python 3, pytest, existing `.venv`, existing `scripts/predict.py` AI-native prediction pipeline.

---

## Current Evidence And Problem Statement

已用真实仓库 `/root/.openclaw/workspace/repos/88/` 核验：

- `origin/main = 2525ee7d57414aee125c4b848d507180d4ebb90b`
- `origin/main` 已包含 PR `#37` 的 Grok Web-Max 代码和 2026-06-13 新预测数据。
- 新版 `data/predictions.json` 的 `update_time=2026-06-13 20:14:32`。
- 旧基线 `fab0990ecfcfd98183d3a16f1214e40839e5aef3` 的 `update_time=2026-06-10 21:04:22`。
- 新旧共同比赛 20 场；新增 6 场芬兰赛事；移除 6 场世界杯赛事。
- 新版有效 web source 覆盖从旧版 `6` 场提升到 `16` 场，方向是对的。
- 新版暴露的问题：
  - `evidence_quality_score >= 70` 但没有有效 URL，例如法国-塞内加尔、沙特-乌拉圭。
  - `recommend_gate_pass=True` 但 `action=observe`，例如德国-库拉索、荷兰-日本。
  - 逆向 Steam / 大热必死被过度放大，例如奥地利-约旦 `3-1 -> 2-2`，且 warning 已提示方向概率和 top3 排序不一致。
  - Top4 可进入 `observe` 或无有效来源样本。

目标不是退回旧版，而是取长补短：

- 保留新版：Grok 外部事实层、web source 覆盖、无来源降级、强队热门风险意识。
- 吸收旧版：更稳的盘口结构、避免无证据极端反打、Top4 更像可用推荐。

---

## File Structure

**Modify:**
- `scripts/predict.py`
  - Add V2 evidence quality normalization helpers near current `_valid_external_source_count()` and `_apply_external_fact_source_gate()`.
  - Add direction/top3 consistency gate after `apply_pre_match_factor_v2_gate()` and before `assign_selection_layer()`.
  - Add recommendation/action sync helper after `assign_selection_layer()`.
  - Tighten `select_top4()` eligibility.

**Create:**
- `tests/test_grok_webmax_v2_gates.py`
  - Focused unit tests for V2 hard gates.
- `scripts/audit_prediction_quality.py`
  - Read one or two `data/predictions.json` files/refs and emit objective quality issues.
- `tests/test_prediction_quality_audit.py`
  - Unit tests for the audit script pure helpers.

**Do not modify:**
- `data/predictions.json` during implementation tasks.
- `origin/main` directly.

**Branching:**
- Start from `origin/main` after fetching.
- Create branch `feat/grok-webmax-v2-evidence-gates-20260613`.
- Create backup tag `backup/pre-grok-webmax-v2-20260613` pointing to `origin/main`.

---

### Task 1: Create Branch And Baseline Guard

**Files:**
- No file changes.

- [ ] **Step 1: Fetch and verify remote main**

Run:

```bash
cd /root/.openclaw/workspace/repos/88
git fetch origin main --quiet
git rev-parse origin/main
git ls-remote origin refs/heads/main | awk '{print $1}'
```

Expected: both commands return the same SHA, currently `2525ee7d57414aee125c4b848d507180d4ebb90b` unless remote moved.

- [ ] **Step 2: Create implementation branch**

Run:

```bash
cd /root/.openclaw/workspace/repos/88
git switch -c feat/grok-webmax-v2-evidence-gates-20260613 origin/main
```

Expected: branch created from `origin/main`.

- [ ] **Step 3: Create backup tag**

Run:

```bash
cd /root/.openclaw/workspace/repos/88
git tag backup/pre-grok-webmax-v2-20260613 origin/main
git rev-parse backup/pre-grok-webmax-v2-20260613
```

Expected: tag SHA equals `origin/main` SHA.

- [ ] **Step 4: Commit**

No commit in this task.

---

### Task 2: Add Prediction Quality Audit Helpers

**Files:**
- Create: `scripts/audit_prediction_quality.py`
- Create: `tests/test_prediction_quality_audit.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_prediction_quality_audit.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd /root/.openclaw/workspace/repos/88
.venv/bin/python -m pytest tests/test_prediction_quality_audit.py -q
```

Expected: FAIL because `scripts/audit_prediction_quality.py` does not exist.

- [ ] **Step 3: Implement audit helper script**

Create `scripts/audit_prediction_quality.py`:

```python
# -*- coding: utf-8 -*-
"""Quality audit helpers for 88 prediction JSON outputs.

This script is intentionally read-only. It flags recommendation-quality issues
without judging actual match outcomes.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


BETTABLE_ACTIONS = {"main", "small", "hedge"}


def load_json_ref(ref: str) -> Dict[str, Any]:
    if ":" in ref and not Path(ref).exists():
        raw = subprocess.check_output(["git", "show", ref])
        return json.loads(raw)
    return json.loads(Path(ref).read_text(encoding="utf-8"))


def prediction_rows(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = data.get("matches", {}).get("today", [])
    return rows if isinstance(rows, list) else []


def prediction_of(row: Dict[str, Any]) -> Dict[str, Any]:
    pred = row.get("prediction", {})
    return pred if isinstance(pred, dict) else {}


def recommendation_of(pred: Dict[str, Any]) -> Dict[str, Any]:
    rec = pred.get("recommendation", {})
    return rec if isinstance(rec, dict) else {}


def tier_of(pred: Dict[str, Any]) -> str:
    return str(pred.get("recommendation_tier") or recommendation_of(pred).get("tier") or "D").upper()


def action_of(pred: Dict[str, Any]) -> str:
    rec = recommendation_of(pred)
    return str(rec.get("bet_action") or pred.get("final_action") or pred.get("selection_layer") or "").lower()


def count_valid_sources(pred: Dict[str, Any]) -> int:
    count = 0
    web = pred.get("web_research", {}) if isinstance(pred.get("web_research"), dict) else {}
    sources = web.get("sources", []) if isinstance(web.get("sources", []), list) else []
    for src in sources:
        if not isinstance(src, dict):
            continue
        url = str(src.get("url", "")).strip()
        claim = str(src.get("claim", "")).strip()
        title = str(src.get("title", src.get("source_title", ""))).strip()
        if url and url != "#" and (claim or title):
            count += 1
    facts = pred.get("external_fact_table", [])
    if isinstance(facts, list):
        for fact in facts:
            if not isinstance(fact, dict):
                continue
            url = str(fact.get("source_url", fact.get("url", ""))).strip()
            claim = str(fact.get("claim", "")).strip()
            if url and url != "#" and claim:
                count += 1
    return count


def evidence_quality(pred: Dict[str, Any]) -> int:
    try:
        return int(float(pred.get("evidence_quality_score", 0) or 0))
    except (TypeError, ValueError):
        return 0


def issue_list_for_prediction(pred: Dict[str, Any]) -> List[str]:
    issues: List[str] = []
    valid_sources = count_valid_sources(pred)
    eq = evidence_quality(pred)
    gate = bool(pred.get("recommend_gate_pass"))
    tier = tier_of(pred)
    action = action_of(pred)
    if eq >= 70 and valid_sources == 0:
        issues.append("high_evidence_quality_without_valid_source")
    if gate and valid_sources == 0:
        issues.append("gate_pass_without_valid_source")
    if gate and action and action not in BETTABLE_ACTIONS:
        issues.append("gate_pass_but_action_not_bettable")
    if action == "main" and tier not in {"S", "A", "B"}:
        issues.append("main_action_below_b_tier")
    if action == "main" and not gate:
        issues.append("main_action_without_gate_pass")
    return issues


def row_key(row: Dict[str, Any]) -> Tuple[str, str]:
    return (str(row.get("home_team", "")), str(row.get("away_team", "")))


def audit_data(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for row in prediction_rows(data):
        pred = prediction_of(row)
        issues = issue_list_for_prediction(pred)
        if issues:
            out.append({
                "home_team": row_key(row)[0],
                "away_team": row_key(row)[1],
                "score": pred.get("predicted_score"),
                "tier": tier_of(pred),
                "action": action_of(pred),
                "gate": bool(pred.get("recommend_gate_pass")),
                "valid_sources": count_valid_sources(pred),
                "evidence_quality_score": evidence_quality(pred),
                "issues": issues,
            })
    return out


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_ref")
    args = parser.parse_args(list(argv) if argv is not None else None)
    data = load_json_ref(args.json_ref)
    print(json.dumps({"issues": audit_data(data)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
cd /root/.openclaw/workspace/repos/88
.venv/bin/python -m pytest tests/test_prediction_quality_audit.py -q
```

Expected: PASS.

- [ ] **Step 5: Run audit on current output**

Run:

```bash
cd /root/.openclaw/workspace/repos/88
.venv/bin/python scripts/audit_prediction_quality.py origin/main:data/predictions.json
```

Expected: JSON list includes current known issues such as high evidence quality without valid source.

- [ ] **Step 6: Commit**

Run:

```bash
cd /root/.openclaw/workspace/repos/88
git add scripts/audit_prediction_quality.py tests/test_prediction_quality_audit.py
git commit -m "test: add prediction quality audit helper"
```

---

### Task 3: Normalize External Evidence Quality

**Files:**
- Modify: `scripts/predict.py:1720-1810`
- Create/Modify: `tests/test_grok_webmax_v2_gates.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_grok_webmax_v2_gates.py`:

```python
# -*- coding: utf-8 -*-
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts import predict


def _row(**overrides):
    row = {
        "final_direction": "home",
        "predicted_score": "2-0",
        "direction_probs": {"home": 62, "draw": 22, "away": 16},
        "top3": [{"score": "2-0", "prob": 18}, {"score": "1-0", "prob": 14}],
        "anchor_audit": {},
        "score_cluster_audit": {},
        "sharp_money_audit": {},
        "recommendation_components": {"direction_edge": 80, "score_cluster_strength": 75, "goal_band_strength": 72, "btts_alignment": 70, "sharp_alignment": 60, "web_source_quality": 80, "market_conflict_penalty": 0},
        "web_research": {"used": True, "sources": [{"title": "bad", "url": "#", "claim": "阵容完整"}]},
        "external_fact_table": [],
        "evidence_quality_score": 85,
        "recommendation": {"tier": "A", "is_recommended": True, "bet_action": "main", "bet_confidence": 78},
        "reason": "官方首发与战意明确，适合主推。",
    }
    row.update(overrides)
    return row


def test_evidence_quality_is_capped_when_url_is_invalid():
    front = predict.adapt_ai_to_frontend(_row(), {"league": "世界杯", "sp_home": 1.45, "s11": 7.2})
    assert front["evidence_quality_score"] <= 40
    assert front["recommend_gate_pass"] is False
    assert "external_source_url_missing_or_invalid" in front["validation_warnings"]


def test_no_valid_source_caps_evidence_quality_even_if_model_claims_high_score():
    row = _row(web_research={"used": False, "sources": []}, evidence_quality_score=92)
    front = predict.adapt_ai_to_frontend(row, {"league": "世界杯", "sp_home": 1.45, "s11": 7.2})
    assert front["evidence_quality_score"] <= 30
    assert front["recommendation"]["tier"] == "C"
    assert front["recommend_gate_pass"] is False


def test_valid_sources_preserve_high_evidence_quality():
    row = _row(
        web_research={"used": True, "sources": [{"title": "official", "url": "https://example.com/team-news", "claim": "官方首发确认"}]},
        external_fact_table=[{"claim": "官方首发确认", "source_url": "https://example.com/team-news"}],
        evidence_quality_score=88,
    )
    front = predict.adapt_ai_to_frontend(row, {"league": "世界杯", "sp_home": 1.45, "s11": 7.2})
    assert front["evidence_quality_score"] == 88
    assert "external_fact_without_source" not in front.get("validation_warnings", [])
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd /root/.openclaw/workspace/repos/88
.venv/bin/python -m pytest tests/test_grok_webmax_v2_gates.py::test_evidence_quality_is_capped_when_url_is_invalid -q
```

Expected: FAIL because current code only warns invalid source but does not cap `evidence_quality_score`.

- [ ] **Step 3: Implement evidence quality normalization**

Modify `scripts/predict.py` near `_valid_external_source_count()`:

```python
def _source_quality_floor_and_cap(web_research: Dict[str, Any], external_fact_table: Any = None) -> Tuple[int, List[str]]:
    warnings: List[str] = []
    sources = web_research.get("sources", []) if isinstance(web_research, dict) else []
    total_sources = len(sources) if isinstance(sources, list) else 0
    valid_sources = _valid_external_source_count(web_research, external_fact_table)
    if valid_sources <= 0:
        if total_sources:
            warnings.append("external_source_url_missing_or_invalid")
            return 40, warnings
        warnings.append("missing_external_confirmation")
        return 30, warnings
    if valid_sources == 1:
        return 75, warnings
    return 100, warnings


def _normalize_external_evidence_quality(pred: Dict[str, Any]) -> int:
    web = pred.get("web_research", {}) if isinstance(pred.get("web_research"), dict) else {}
    raw_score = int(_clip(_f(pred.get("evidence_quality_score", 0), 0), 0, 100))
    cap, warnings = _source_quality_floor_and_cap(web, pred.get("external_fact_table"))
    normalized = min(raw_score, cap) if raw_score else 0
    existing = pred.setdefault("validation_warnings", [])
    if not isinstance(existing, list):
        existing = [str(existing)] if existing else []
    existing.extend(warnings)
    pred["validation_warnings"] = list(dict.fromkeys(existing))
    pred["evidence_quality_score"] = normalized
    return normalized
```

Then replace this line in `_apply_external_fact_source_gate()`:

```python
evidence_quality = int(_clip(_f(pred.get("evidence_quality_score", 0), 0), 0, 100))
```

with:

```python
evidence_quality = _normalize_external_evidence_quality(pred)
```

- [ ] **Step 4: Run V2 tests**

Run:

```bash
cd /root/.openclaw/workspace/repos/88
.venv/bin/python -m pytest tests/test_grok_webmax_v2_gates.py -q
```

Expected: tests from Step 1 pass.

- [ ] **Step 5: Run current Grok tests**

Run:

```bash
cd /root/.openclaw/workspace/repos/88
.venv/bin/python -m pytest tests/test_grok_webmax_protocol.py tests/test_prematch_context_gate.py tests/test_prematch_factor_gate.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```bash
cd /root/.openclaw/workspace/repos/88
git add scripts/predict.py tests/test_grok_webmax_v2_gates.py
git commit -m "fix: cap Grok evidence quality by valid sources"
```

---

### Task 4: Add Direction And Candidate Consistency Gate

**Files:**
- Modify: `scripts/predict.py:1772-1810`, `scripts/predict.py:5288-5296`
- Modify: `tests/test_grok_webmax_v2_gates.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_grok_webmax_v2_gates.py`:

```python

def test_final_direction_not_top_probability_caps_recommendation():
    row = _row(
        final_direction="draw",
        predicted_score="2-2",
        direction_probs={"home": 40, "draw": 35, "away": 25},
        top3=[{"score": "2-2", "prob": 15}, {"score": "1-1", "prob": 20}],
        web_research={"used": True, "sources": [{"title": "heat", "url": "https://example.com/heat", "claim": "高温风险"}]},
        evidence_quality_score=75,
        recommendation={"tier": "B", "is_recommended": True, "bet_action": "small", "bet_confidence": 62},
        reason="主胜造热，但高温和大球曲线支持2-2。",
    )
    front = predict.adapt_ai_to_frontend(row, {"league": "世界杯", "sp_home": 1.23, "s11": 8.0})
    assert front["predicted_score"] == "2-2"
    assert front["final_direction"] == "draw"
    assert front["recommend_gate_pass"] is False
    assert front["recommendation"]["tier"] == "C"
    assert "direction_probability_not_supporting_final_direction" in front["validation_warnings"]
    assert "top3_probability_order_conflict" in front["validation_warnings"]
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd /root/.openclaw/workspace/repos/88
.venv/bin/python -m pytest tests/test_grok_webmax_v2_gates.py::test_final_direction_not_top_probability_caps_recommendation -q
```

Expected: FAIL because current code preserves warning text but does not consistently cap recommendation.

- [ ] **Step 3: Implement consistency helper**

Add to `scripts/predict.py` near `_apply_external_fact_source_gate()`:

```python
def _direction_probability_for(pred: Dict[str, Any], direction: str) -> float:
    probs = pred.get("direction_probs", {}) if isinstance(pred.get("direction_probs"), dict) else {}
    return _f(probs.get(direction), 0.0)


def _max_direction_probability(pred: Dict[str, Any]) -> Tuple[str, float]:
    probs = pred.get("direction_probs", {}) if isinstance(pred.get("direction_probs"), dict) else {}
    rows = [(d, _f(probs.get(d), 0.0)) for d in VALID_DIRS]
    return max(rows, key=lambda x: x[1]) if rows else ("", 0.0)


def _cap_to_observe(pred: Dict[str, Any], reason: str, max_tier: str = "C") -> None:
    rec = pred.setdefault("recommendation", {}) if isinstance(pred.get("recommendation"), dict) else {}
    if not isinstance(pred.get("recommendation"), dict):
        pred["recommendation"] = rec
    current_tier = str(rec.get("tier", pred.get("recommendation_tier", "D"))).upper()
    rec["tier"] = max_tier if _tier_cap_value(current_tier) > _tier_cap_value(max_tier) else current_tier
    rec["is_recommended"] = False
    rec["bet_action"] = "observe"
    pred["recommendation_tier"] = rec["tier"]
    pred["recommend_gate_pass"] = False
    pred.setdefault("recommend_gate_reasons", []).append(reason)
    warnings = pred.setdefault("validation_warnings", [])
    if not isinstance(warnings, list):
        warnings = [str(warnings)] if warnings else []
        pred["validation_warnings"] = warnings
    warnings.append(reason)
    pred["validation_warnings"] = list(dict.fromkeys(warnings))


def _apply_direction_candidate_consistency_gate(pred: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(pred, dict) or pred.get("is_abstain"):
        return pred
    final_dir = str(pred.get("final_direction", ""))
    if final_dir in VALID_DIRS:
        max_dir, max_prob = _max_direction_probability(pred)
        final_prob = _direction_probability_for(pred, final_dir)
        if max_dir in VALID_DIRS and max_dir != final_dir and max_prob - final_prob >= 3.0:
            _cap_to_observe(pred, "direction_probability_not_supporting_final_direction", "C")
    top3 = pred.get("top3", []) if isinstance(pred.get("top3"), list) else []
    if len(top3) >= 2 and isinstance(top3[0], dict):
        first_score = _score_from_candidate(top3[0].get("score"))
        predicted_score = _score_from_candidate(pred.get("predicted_score"))
        first_prob = _f(top3[0].get("prob"), 0.0)
        higher_later = any(isinstance(x, dict) and _f(x.get("prob"), 0.0) > first_prob + 0.01 for x in top3[1:])
        if first_score == predicted_score and higher_later:
            _cap_to_observe(pred, "top3_probability_order_conflict", "C")
    return pred
```

Call it in `adapt_ai_to_frontend()` after `_apply_external_fact_source_gate(pred)` and before `assign_selection_layer(pred, match_obj)`:

```python
try:
    _apply_direction_candidate_consistency_gate(pred)
except Exception as e:
    pred.setdefault("validation_warnings", []).append(f"direction_candidate_consistency_gate_error:{str(e)[:120]}")
```

- [ ] **Step 4: Run V2 tests**

Run:

```bash
cd /root/.openclaw/workspace/repos/88
.venv/bin/python -m pytest tests/test_grok_webmax_v2_gates.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
cd /root/.openclaw/workspace/repos/88
git add scripts/predict.py tests/test_grok_webmax_v2_gates.py
git commit -m "fix: gate inconsistent direction and score candidates"
```

---

### Task 5: Add Contrarian Steam Throttle

**Files:**
- Modify: `scripts/predict.py`
- Modify: `tests/test_grok_webmax_v2_gates.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_grok_webmax_v2_gates.py`:

```python

def test_contrarian_steam_claim_without_valid_market_source_is_observe_only():
    row = _row(
        final_direction="draw",
        predicted_score="1-1",
        direction_probs={"home": 38, "draw": 36, "away": 26},
        web_research={"used": False, "sources": []},
        evidence_quality_score=80,
        recommendation={"tier": "B", "is_recommended": True, "bet_action": "small", "bet_confidence": 70},
        reason="主胜反向Steam造热，大热必死，必须反打下盘。",
    )
    front = predict.adapt_ai_to_frontend(row, {"league": "世界杯", "sp_home": 1.30, "s11": 6.8})
    assert front["recommend_gate_pass"] is False
    assert front["recommendation"]["tier"] == "C"
    assert "contrarian_market_claim_without_valid_market_source" in front["validation_warnings"]


def test_contrarian_steam_claim_with_valid_market_source_can_remain_small():
    row = _row(
        final_direction="draw",
        predicted_score="1-1",
        direction_probs={"home": 36, "draw": 37, "away": 27},
        web_research={"used": True, "sources": [{"title": "odds", "url": "https://example.com/odds", "claim": "market snapshot shows home drift", "source_type": "market_snapshot"}]},
        external_fact_table=[{"claim": "home drift", "source_url": "https://example.com/odds", "category": "market_snapshot"}],
        evidence_quality_score=82,
        recommendation={"tier": "B", "is_recommended": True, "bet_action": "small", "bet_confidence": 58},
        reason="主胜反向Steam造热，但外部market_snapshot确认主胜漂移。",
    )
    front = predict.adapt_ai_to_frontend(row, {"league": "世界杯", "sp_home": 1.80, "s11": 6.8})
    assert "contrarian_market_claim_without_valid_market_source" not in front.get("validation_warnings", [])
```

- [ ] **Step 2: Run test to verify first one fails**

Run:

```bash
cd /root/.openclaw/workspace/repos/88
.venv/bin/python -m pytest tests/test_grok_webmax_v2_gates.py::test_contrarian_steam_claim_without_valid_market_source_is_observe_only -q
```

Expected: FAIL because current code only handles external context terms, not market-contrarian claims.

- [ ] **Step 3: Implement contrarian helper**

Add to `scripts/predict.py` near consistency helpers:

```python
CONTRARIAN_MARKET_TERMS = ["反向steam", "steam", "造热", "大热必死", "诱盘", "反打", "聪明钱", "sharp", "rlm", "reverse line"]
MARKET_SOURCE_TERMS = ["market", "odds", "盘口", "赔率", "亚盘", "大小球", "market_snapshot", "pinnacle", "bet365", "william"]


def _has_valid_market_source(pred: Dict[str, Any]) -> bool:
    web = pred.get("web_research", {}) if isinstance(pred.get("web_research"), dict) else {}
    sources = web.get("sources", []) if isinstance(web.get("sources", []), list) else []
    facts = pred.get("external_fact_table", []) if isinstance(pred.get("external_fact_table", []), list) else []
    blobs = sources + facts
    for row in blobs:
        if not isinstance(row, dict):
            continue
        url = str(row.get("url", row.get("source_url", ""))).strip()
        if not url or url == "#":
            continue
        text = _json_compact(row, 1200).lower()
        if any(term.lower() in text for term in MARKET_SOURCE_TERMS):
            return True
    return False


def _apply_contrarian_market_claim_gate(pred: Dict[str, Any]) -> Dict[str, Any]:
    text = _external_context_text(pred)
    if any(term.lower() in text for term in CONTRARIAN_MARKET_TERMS) and not _has_valid_market_source(pred):
        _cap_to_observe(pred, "contrarian_market_claim_without_valid_market_source", "C")
    return pred
```

Call it after `_apply_direction_candidate_consistency_gate(pred)`:

```python
try:
    _apply_contrarian_market_claim_gate(pred)
except Exception as e:
    pred.setdefault("validation_warnings", []).append(f"contrarian_market_claim_gate_error:{str(e)[:120]}")
```

- [ ] **Step 4: Run V2 tests**

Run:

```bash
cd /root/.openclaw/workspace/repos/88
.venv/bin/python -m pytest tests/test_grok_webmax_v2_gates.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
cd /root/.openclaw/workspace/repos/88
git add scripts/predict.py tests/test_grok_webmax_v2_gates.py
git commit -m "fix: require market evidence for contrarian steam claims"
```

---

### Task 6: Sync Gate, Action, And Top4 Eligibility

**Files:**
- Modify: `scripts/predict.py:2707-2721`, `scripts/predict.py:5292-5296`
- Modify: `tests/test_grok_webmax_v2_gates.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_grok_webmax_v2_gates.py`:

```python

def test_gate_pass_observe_action_is_synchronized_to_not_recommended():
    row = _row(
        web_research={"used": True, "sources": [{"title": "ok", "url": "https://example.com", "claim": "ok"}]},
        evidence_quality_score=80,
        recommendation={"tier": "B", "is_recommended": True, "bet_action": "observe", "bet_confidence": 65},
        reason="盘口结构可看，但只观察。",
    )
    front = predict.adapt_ai_to_frontend(row, {"league": "世界杯", "sp_home": 1.7, "s11": 7.0})
    assert front["recommend_gate_pass"] is False
    assert front["recommendation"]["is_recommended"] is False
    assert "gate_action_not_bettable" in front["recommend_gate_reasons"]


def test_select_top4_excludes_observe_even_when_ai_marks_recommended():
    rows = []
    for i, action in enumerate(["main", "small", "observe", "main", "small"], 1):
        rows.append({
            "home_team": f"H{i}",
            "away_team": f"A{i}",
            "prediction": {
                "predicted_score": "1-0",
                "recommend_gate_pass": action != "observe",
                "recommendation": {"tier": "B", "is_recommended": True, "bet_action": action, "bet_confidence": 70 - i},
            },
        })
    top = predict.select_top4(rows)
    assert all(x["prediction"]["recommendation"]["bet_action"] != "observe" for x in top)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd /root/.openclaw/workspace/repos/88
.venv/bin/python -m pytest tests/test_grok_webmax_v2_gates.py::test_gate_pass_observe_action_is_synchronized_to_not_recommended tests/test_grok_webmax_v2_gates.py::test_select_top4_excludes_observe_even_when_ai_marks_recommended -q
```

Expected: FAIL because current `select_top4()` only checks `is_recommended` and tier.

- [ ] **Step 3: Implement sync helper**

Add to `scripts/predict.py` near `assign_selection_layer()` or V2 gates:

```python
BETTABLE_ACTIONS = {"main", "small", "hedge"}


def _sync_gate_with_bet_action(pred: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(pred, dict) or pred.get("is_abstain"):
        return pred
    rec = pred.get("recommendation", {}) if isinstance(pred.get("recommendation"), dict) else {}
    action = str(rec.get("bet_action") or pred.get("final_action") or pred.get("selection_layer") or "").lower()
    if action and action not in BETTABLE_ACTIONS:
        rec["is_recommended"] = False
        pred["recommend_gate_pass"] = False
        reasons = pred.setdefault("recommend_gate_reasons", [])
        if not isinstance(reasons, list):
            reasons = [str(reasons)] if reasons else []
            pred["recommend_gate_reasons"] = reasons
        reasons.append("gate_action_not_bettable")
        pred["recommend_gate_reasons"] = list(dict.fromkeys(reasons))
    return pred
```

Call it after `assign_selection_layer(pred, match_obj)`:

```python
try:
    _sync_gate_with_bet_action(pred)
except Exception as e:
    pred.setdefault("validation_warnings", []).append(f"gate_action_sync_error:{str(e)[:120]}")
```

Modify `select_top4()` eligibility:

```python
if bool(rec.get("is_recommended", False)) and bool(pr.get("recommend_gate_pass")) and _min_tier_ok(rec.get("tier", "D")) and str(rec.get("bet_action", "")).lower() in BETTABLE_ACTIONS:
    eligible.append(p)
```

- [ ] **Step 4: Run V2 tests**

Run:

```bash
cd /root/.openclaw/workspace/repos/88
.venv/bin/python -m pytest tests/test_grok_webmax_v2_gates.py -q
```

Expected: PASS.

- [ ] **Step 5: Run selection/prematch tests**

Run:

```bash
cd /root/.openclaw/workspace/repos/88
.venv/bin/python -m pytest tests/test_selection_layer.py tests/test_prematch_factor_gate.py tests/test_prematch_context_gate.py -q
```

Expected: PASS or only failures that reflect intentional tightening; update assertions only when they conflict with the new rule and the test name/docstring agrees.

- [ ] **Step 6: Commit**

Run:

```bash
cd /root/.openclaw/workspace/repos/88
git add scripts/predict.py tests/test_grok_webmax_v2_gates.py tests/test_selection_layer.py tests/test_prematch_factor_gate.py tests/test_prematch_context_gate.py
git commit -m "fix: sync recommendation gate with bet action"
```

---

### Task 7: Run Full Regression And Quality Audit

**Files:**
- No source changes unless tests expose defects.

- [ ] **Step 1: Compile**

Run:

```bash
cd /root/.openclaw/workspace/repos/88
.venv/bin/python -m compileall -q scripts tests
```

Expected: no output and exit code 0.

- [ ] **Step 2: Full pytest**

Run:

```bash
cd /root/.openclaw/workspace/repos/88
.venv/bin/python -m pytest tests/ -q
```

Expected: all tests pass. Baseline before this plan was `102 passed`; after adding tests, expected count is higher.

- [ ] **Step 3: Audit current main predictions with new audit script**

Run:

```bash
cd /root/.openclaw/workspace/repos/88
.venv/bin/python scripts/audit_prediction_quality.py origin/main:data/predictions.json
```

Expected: issues are printed for the current `2525ee7` output. This is expected because the output was generated before V2 gates.

- [ ] **Step 4: Dry-run a local prediction only if user approves runtime/API cost**

Do not run API-consuming prediction automatically. Ask user before any command that triggers real AI/web calls.

- [ ] **Step 5: Commit if no changes from this task**

No commit if no files changed.

---

### Task 8: Push Branch And Stop Before Main

**Files:**
- No additional source changes.

- [ ] **Step 1: Show diff summary before push**

Run:

```bash
cd /root/.openclaw/workspace/repos/88
git status --short
git log --oneline origin/main..HEAD
git diff --stat origin/main..HEAD
```

Expected: only V2 code/tests/audit script changes are listed; `data/predictions.json` is not modified.

- [ ] **Step 2: Push new branch**

Run:

```bash
cd /root/.openclaw/workspace/repos/88
git push -u origin feat/grok-webmax-v2-evidence-gates-20260613
```

Expected: GitHub prints a PR creation URL.

- [ ] **Step 3: Verify remote SHA**

Run:

```bash
cd /root/.openclaw/workspace/repos/88
git rev-parse HEAD
git ls-remote origin refs/heads/feat/grok-webmax-v2-evidence-gates-20260613 | awk '{print $1}'
git ls-remote origin refs/heads/main | awk '{print $1}'
```

Expected: branch remote SHA equals local HEAD; `origin/main` remains unchanged.

- [ ] **Step 4: Stop**

Do not merge PR. User runs the branch and provides output. Then compare against `origin/main` and `fab0990` with the audit script.

---

## Self-Review

### Spec Coverage

- “全方位升级”：covered by source-quality cap, consistency gate, contrarian throttle, gate/action sync, Top4 filtering, and audit tooling.
- “取长补短”：keeps Grok Web-Max, restores old-version stability by blocking unsupported contrarian upgrades.
- “Superpowers”：plan is saved as a Superpowers implementation plan and uses test-first tasks.
- “不直接改 main”：branch/tag/push workflow is explicit.
- “全 slate 审计”：audit script checks full `matches.today`, not just Top4.

### Placeholder Scan

No `TBD`, no “implement later”, no unspecified tests.

### Type Consistency

- `evidence_quality_score` remains integer 0-100.
- `external_fact_table` remains list of dicts.
- `web_research.sources` remains list of dicts.
- `recommend_gate_pass` remains boolean.
- `recommendation.bet_action` remains string.

---

## Execution Choice

Plan complete and saved to `docs/superpowers/plans/2026-06-13-grok-webmax-v2-evidence-gates.md`.

Two execution options:

1. **Subagent-Driven (recommended)** - dispatch a fresh subagent per task, review between tasks, safer for this multi-gate patch.
2. **Inline Execution** - execute tasks in this session using executing-plans, faster but higher context-load risk.

Recommended: **Subagent-Driven**, because we need separate review of source-quality gating, contrarian gating, and Top4 eligibility.

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

# -*- coding: utf-8 -*-
"""metrics_ledger.py — 混合型预测系统的双账本绩效度量。

设计动机（grill 结论）：
  本系统是「混合型」：大多数场观望(observe)，少数高信心出 main/small，
  其中含「反打/背离博冷」单。用单一胜率 (correct/finished) 度量会同时
  害死两类单：
    - 稳胜单：被博冷单的低胜率拖低；
    - 博冷单：天然低胜率高赔率，普通胜率看着烂，实际 ROI 可能为正。
  因此必须分两本账：
    1) 可下注单 → 按 ROI / 期望值 度量（这是真正决定盈亏的指标）；
    2) 全样本方向命中率 → 仅作分析能力参考，绝不当盈亏指标。

本模块纯函数、无网络、无副作用，便于单测。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


# 视为「可下注」的动作 / 档位
BETTABLE_ACTIONS = {"main", "small", "hedge"}
BETTABLE_TIERS = {"S", "A", "B"}
# 博冷单的标记线索（动作或风险标签里出现这些 → 归入博冷账本）
UPSET_HINT_TERMS = (
    "反打", "博冷", "冷门", "upset", "contrarian", "reverse",
    "背离", "sharp", "rlm", "fade",
)


def _f(v: Any, default: float = 0.0) -> float:
    try:
        if v is None or v == "":
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def _result_cn(gh: int, ga: int) -> str:
    return "主胜" if gh > ga else ("平局" if gh == ga else "客胜")


def is_bettable(action: str, tier: str) -> bool:
    """是否算「下了注」的单。observe / no_bet / C / D 不算。"""
    a = (action or "").lower()
    t = (tier or "").upper()
    if a in ("observe", "no_bet", "skip", "", "pass"):
        return False
    return a in BETTABLE_ACTIONS or t in BETTABLE_TIERS


def is_upset_bet(action: str, risk_tags: Optional[List[Any]], reason: str = "") -> bool:
    """是否属于「博冷/反打」单。供分账。"""
    hay = (action or "").lower() + " " + (reason or "").lower()
    if isinstance(risk_tags, list):
        hay += " " + " ".join(str(x).lower() for x in risk_tags)
    return any(term in hay for term in UPSET_HINT_TERMS)


def settle_one(pred: Dict[str, Any], gh: int, ga: int) -> Dict[str, Any]:
    """结算单场。返回该场的方向命中、是否下注、博冷与否、收益(profit)。

    收益模型（单位注金=1）：
      - 命中：profit = (odds - 1)，odds 取该方向赔率，缺失则按 2.0 估
      - 未中：profit = -1
      - 未下注：profit = 0（不计入 ROI 分母）
    """
    rec = pred.get("recommendation", {}) if isinstance(pred.get("recommendation"), dict) else {}
    action = str(rec.get("bet_action") or pred.get("final_action") or pred.get("selection_layer") or "").lower()
    tier = str(rec.get("tier") or pred.get("recommendation_tier") or "D").upper()
    reason = str(pred.get("reason") or rec.get("why_recommended") or "")
    risk_tags = rec.get("risk_tags") if isinstance(rec.get("risk_tags"), list) else pred.get("tail_risk_flags")

    pred_dir = str(pred.get("result") or pred.get("final_direction") or "")
    # 中文/英文方向归一
    dir_map = {"home": "主胜", "draw": "平局", "away": "客胜"}
    pred_dir = dir_map.get(pred_dir.lower(), pred_dir)
    actual_dir = _result_cn(gh, ga)
    hit = (pred_dir == actual_dir) if pred_dir else False

    bettable = is_bettable(action, tier)
    upset = is_upset_bet(action, risk_tags, reason)

    # 赔率：优先用该方向赔率，退化到 main_odds，再退化到 2.0
    odds = _f(rec.get("odds") or rec.get("bet_odds") or pred.get("odds"), 0.0)
    if odds <= 1.0:
        odds = 2.0  # 缺赔率时的保守估计

    if not bettable:
        profit = 0.0
    elif hit:
        profit = odds - 1.0
    else:
        profit = -1.0

    return {
        "pred_dir": pred_dir,
        "actual_dir": actual_dir,
        "actual_score": f"{gh}-{ga}",
        "hit": hit,
        "bettable": bettable,
        "upset": upset,
        "tier": tier,
        "action": action,
        "odds": odds,
        "profit": profit,
    }


def aggregate(settled: List[Dict[str, Any]]) -> Dict[str, Any]:
    """把多场结算汇总成双账本。"""
    n = len(settled)
    # 全样本方向命中率（含观望）——仅作分析参考
    dir_hits = sum(1 for s in settled if s["hit"])
    dir_acc = (dir_hits / n * 100.0) if n else 0.0

    def _book(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        staked = len(rows)
        wins = sum(1 for r in rows if r["hit"])
        pnl = sum(r["profit"] for r in rows)
        roi = (pnl / staked * 100.0) if staked else 0.0
        win_rate = (wins / staked * 100.0) if staked else 0.0
        return {
            "staked": staked,
            "wins": wins,
            "win_rate": round(win_rate, 1),
            "pnl": round(pnl, 3),
            "roi_pct": round(roi, 1),
        }

    bettable_rows = [s for s in settled if s["bettable"]]
    upset_rows = [s for s in bettable_rows if s["upset"]]
    value_rows = [s for s in bettable_rows if not s["upset"]]

    return {
        "samples": n,
        # 分析能力参考（不是盈亏指标）
        "direction_accuracy_pct": round(dir_acc, 1),
        # 盈亏账本（真正的指标）
        "bettable": _book(bettable_rows),   # 所有下注单合账
        "value_bets": _book(value_rows),    # 稳胜/价值单
        "upset_bets": _book(upset_rows),    # 博冷/反打单
    }


def coaching_summary(agg: Dict[str, Any]) -> str:
    """给 AI 复盘的中文摘要——强调 ROI，避免把博冷单按胜率惩罚。"""
    b = agg["bettable"]; v = agg["value_bets"]; u = agg["upset_bets"]
    lines = [
        f"全样本方向命中率(含观望,仅参考): {agg['direction_accuracy_pct']}% (n={agg['samples']})",
        f"下注单合账: {b['staked']}单 ROI={b['roi_pct']}% PnL={b['pnl']} 胜率={b['win_rate']}%",
        f"  ├ 价值/稳胜单: {v['staked']}单 ROI={v['roi_pct']}% 胜率={v['win_rate']}%",
        f"  └ 博冷/反打单: {u['staked']}单 ROI={u['roi_pct']}% 胜率={u['win_rate']}%（低胜率高赔率属正常，只看ROI）",
    ]
    return "\n".join(lines)

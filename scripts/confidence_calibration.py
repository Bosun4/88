#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P0-2 信心经验校准层 (2026-07-02 十日审计升级)

审计发现:
  - bet_confidence 是 AI 口头信心, 从未用真实命中率回校; ECE=24.7
  - 等级倒挂: C档方向82.9% > A档63.6% (最该信的高档反而最差)

设计 (军规: 本地只降推荐等级, 不改方向不改比分):
  1. data/calibration_ledger.json 滚动收集 (date, conf, tier, dir_hit, match)
  2. Calibrator: 按信心band经验命中率映射, 经验贝叶斯收缩(小样本向全局均值靠)
  3. tier_guard: 校准信心低于该档地板 → 只降不升
  4. adapt_ai_to_frontend 输出 calibrated_confidence 双轨字段, 原始信心保留存档
"""
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# 信心分档 (与十日审计口径一致)
_BANDS = [(0, 39), (40, 49), (50, 59), (60, 69), (70, 79), (80, 100)]
# 各推荐档位的"校准信心地板": 低于此值则降级
_TIER_FLOOR = {"S": 80, "A": 70, "B": 55, "C": 0, "D": 0}
_TIER_ORDER = ["D", "C", "B", "A", "S"]
# 贝叶斯收缩强度(等效先验样本数): band样本越少越向全局均值收缩
_SHRINK_K = 6.0
# 台账总样本低于此值不激活校准(避免小样本暴走), 恒等返回
_MIN_TOTAL = 10


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _band_of(conf: float) -> Tuple[int, int]:
    for lo, hi in _BANDS:
        if lo <= conf <= hi:
            return (lo, hi)
    return _BANDS[-1] if conf > 100 else _BANDS[0]


class Calibrator:
    """经验贝叶斯信心校准器。"""

    def __init__(self, ledger_rows: List[Dict[str, Any]]):
        self.rows = [r for r in (ledger_rows or []) if isinstance(r, dict) and "conf" in r]
        self._global = self._global_rate()
        self._band_rate = self._compute_band_rates()

    def _global_rate(self) -> Optional[float]:
        hits = [1 if r.get("dir_hit") else 0 for r in self.rows]
        return (sum(hits) / len(hits)) if hits else None

    def _compute_band_rates(self) -> Dict[Tuple[int, int], float]:
        out: Dict[Tuple[int, int], float] = {}
        if self._global is None:
            return out
        for lo, hi in _BANDS:
            sub = [r for r in self.rows if lo <= float(r.get("conf", -1)) <= hi]
            if not sub:
                continue
            n = len(sub)
            hit = sum(1 for r in sub if r.get("dir_hit"))
            # 经验贝叶斯: (hit + k*prior) / (n + k)
            rate = (hit + _SHRINK_K * self._global) / (n + _SHRINK_K)
            out[(lo, hi)] = rate
        return out

    def calibrate(self, conf: Union[int, float]) -> int:
        """AI原始信心 → 校准信心(0-100整数)。无台账时恒等返回。"""
        try:
            c = float(conf)
        except (TypeError, ValueError):
            return int(_clip(float(conf or 0), 0, 100)) if conf else 0
        if self._global is None or not self._band_rate or len(self.rows) < _MIN_TOTAL:
            return int(_clip(c, 0, 100))
        band = _band_of(c)
        rate = self._band_rate.get(band)
        if rate is None:
            # 该band无样本 → 用全局率
            rate = self._global
        return int(_clip(round(rate * 100), 0, 100))


def tier_guard(tier: str, calibrated_conf: Union[int, float]) -> Tuple[str, str]:
    """校准信心撑不住当前档位 → 降级(只降不升)。返回(新档, 原因)。"""
    t = str(tier or "D").upper()
    if t not in _TIER_FLOOR:
        return t, ""
    c = float(calibrated_conf or 0)
    idx = _TIER_ORDER.index(t) if t in _TIER_ORDER else 0
    new_idx = idx
    # 逐级下探: 只要校准信心低于当前档地板就降一档
    while new_idx > 0 and c < _TIER_FLOOR[_TIER_ORDER[new_idx]]:
        new_idx -= 1
    if new_idx < idx:
        return _TIER_ORDER[new_idx], f"calibration_downgrade(conf={int(c)}<floor{_TIER_FLOOR[t]})"
    return t, ""


def load_ledger(path: Union[str, Path]) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    if isinstance(data, dict):
        data = data.get("rows", [])
    return [r for r in data if isinstance(r, dict)] if isinstance(data, list) else []


def append_rows(path: Union[str, Path], rows: List[Dict[str, Any]]) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    existing = load_ledger(p)
    existing.extend([r for r in rows if isinstance(r, dict)])
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(existing, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, p)
    return len(existing)


_DEFAULT_LEDGER = Path(__file__).resolve().parent.parent / "data" / "calibration_ledger.json"
_CACHE: Dict[str, Any] = {"mtime": None, "cal": None}


def get_calibrator(path: Union[str, Path, None] = None) -> Calibrator:
    """带 mtime 缓存的全局校准器工厂。"""
    p = Path(path or _DEFAULT_LEDGER)
    try:
        mtime = p.stat().st_mtime if p.exists() else None
    except OSError:
        mtime = None
    if _CACHE["cal"] is None or _CACHE["mtime"] != mtime:
        _CACHE["cal"] = Calibrator(load_ledger(p))
        _CACHE["mtime"] = mtime
    return _CACHE["cal"]

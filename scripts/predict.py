# -*- coding: utf-8 -*-
"""
vMAX 18.4 — PURE RAW-AI 主审版
============================================================
设计边界：
1. 纯 AI 主审：GPT/Grok/Gemini 初审，Claude 终审；Claude 失败时使用 Phase1 AI 共识。
2. 完全移除本地比分主裁：不跑 CRS 矩阵、不跑贝叶斯后验、不跑本地比分矩阵、不跑 T1-T16/D17-D19 风控裁决。
3. 不给 AI 喂本地判断：不喂 fair_1x2、本地理论盘口、强方深浅差、本地陷阱、本地候选比分排序、本地校准 lambda。
4. AI 成功时，本地只做 JSON 解析、字段闭环、前端兼容字段填充。
5. AI 全失败时直接弃权，不使用本地兜底比分。
6. 四个模型默认全部走统一 OpenAI-compatible 中转：API_KEY / API_URL。兼容旧变量作为兜底。
7. singleflight + cache：同批次重复触发时不重复扣费；但 0/4 AI 失败不写缓存。

推荐环境变量：
    API_KEY=你的统一中转key
    API_URL=https://xxx/v1
    GPT_MODEL=gpt-5.4
    GROK_MODEL=grok-4.3
    GEMINI_MODEL=gemini-3.1-pro-preview-thinking-high
    CLAUDE_MODEL=claude-opus-4-7
    AI_DECISION_CACHE_TTL=900
    AI_SINGLEFLIGHT_ENABLED=true
    AI_MAX_REQUESTS_PER_AI=1
    ENABLE_EXTERNAL_CONTEXT=false/true

入口：
    run_predictions(raw, use_ai=True) -> (res, top4)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

try:
    import aiohttp
except Exception as e:  # pragma: no cover
    aiohttp = None

try:
    import structlog
    logger = structlog.get_logger()
except Exception:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)

# ============================================================
# 版本常量
# ============================================================

ENGINE_VERSION = "vMAX 18.4"
ENGINE_ARCHITECTURE = "PURE RAW-AI: GPT/Grok/Gemini 初审 + Claude 终审；无 CRS/无本地矩阵/无本地风控兜底/AI失败即弃权"

VALID_DIRS = {"home", "draw", "away"}
AI_NAMES = ["gpt", "grok", "gemini", "claude"]
PHASE1_NAMES = ["gpt", "grok", "gemini"]

DEFAULT_MODELS = {
    "gpt": "gpt-5.4",
    "grok": "grok-4.3",
    "gemini": "gemini-3.1-pro-preview-thinking-high",
    "claude": "claude-opus-4-7",
}

# 只保留投注字段映射，用于显示/赔率读取；不做 CRS 分析。
CRS_FULL_MAP = {
    "1-0": "w10", "2-0": "w20", "2-1": "w21", "3-0": "w30", "3-1": "w31",
    "3-2": "w32", "4-0": "w40", "4-1": "w41", "4-2": "w42", "5-0": "w50",
    "5-1": "w51", "5-2": "w52",
    "0-0": "s00", "1-1": "s11", "2-2": "s22", "3-3": "s33",
    "0-1": "l01", "0-2": "l02", "1-2": "l12", "0-3": "l03", "1-3": "l13",
    "2-3": "l23", "0-4": "l04", "1-4": "l14", "2-4": "l24", "0-5": "l05",
    "1-5": "l15", "2-5": "l25",
}

SCORE_OTHERS_HOME = [
    "4-3", "5-3", "5-4", "6-0", "6-1", "6-2", "6-3", "6-4",
    "7-0", "7-1", "7-2", "胜其他", "9-0",
]
SCORE_OTHERS_DRAW = ["4-4", "5-5", "6-6", "平其他", "9-9"]
SCORE_OTHERS_AWAY = [
    "3-4", "0-6", "1-6", "2-6", "3-6", "0-7", "1-7", "2-7",
    "负其他", "0-9",
]
ALL_SCORE_OTHERS = SCORE_OTHERS_HOME + SCORE_OTHERS_DRAW + SCORE_OTHERS_AWAY

HFTF_MAP = {
    "ss": "主/主", "sp": "主/平", "sf": "主/负",
    "ps": "平/主", "pp": "平/平", "pf": "平/负",
    "fs": "负/主", "fp": "负/平", "ff": "负/负",
}

# ============================================================
# 配置
# ============================================================

def _env_bool(name: str, default: bool = False) -> bool:
    v = str(os.environ.get(name, str(default))).strip().lower()
    return v in ("1", "true", "yes", "y", "on")


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(str(os.environ.get(name, default)).strip()))
    except Exception:
        return default


AI_DECISION_CACHE_TTL = _env_int("AI_DECISION_CACHE_TTL", 900)
AI_DISABLE_CACHE = _env_bool("AI_DISABLE_CACHE", False)
AI_SINGLEFLIGHT_ENABLED = _env_bool("AI_SINGLEFLIGHT_ENABLED", True)
AI_MAX_REQUESTS_PER_AI = max(1, _env_int("AI_MAX_REQUESTS_PER_AI", 1))
AI_FORCE_COMMON_GATEWAY = _env_bool("FORCE_COMMON_GATEWAY_URL", True)
AI_READ_TIMEOUT = _env_int("AI_READ_TIMEOUT", 520)
AI_CONNECT_TIMEOUT = _env_int("AI_CONNECT_TIMEOUT", 25)
ENABLE_EXTERNAL_CONTEXT = _env_bool("ENABLE_EXTERNAL_CONTEXT", False)
AI_PARSE_DEBUG = _env_bool("AI_PARSE_DEBUG", False)

_AI_RESULT_CACHE: Dict[str, Tuple[float, Dict[str, Dict[int, Dict[str, Any]]], Dict[str, Any]]] = {}
_AI_INFLIGHT_TASKS: Dict[str, asyncio.Task] = {}
AI_CALL_STATUS: Dict[str, Dict[str, Any]] = {n: {} for n in AI_NAMES}

# ============================================================
# 基础工具
# ============================================================

def _f(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        s = str(v).strip()
        if s == "" or s.lower() in ("none", "nan", "null", "-", "n/a"):
            return default
        return float(s.replace("%", ""))
    except Exception:
        return default


def _i(v: Any, default: int = 0) -> int:
    try:
        return int(float(v))
    except Exception:
        return default


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, _f(v, 0.0)))


def _safe_str(v: Any, default: str = "") -> str:
    if v is None:
        return default
    return str(v)


def _mask_key(k: str) -> str:
    if not k:
        return ""
    if len(k) <= 8:
        return "***"
    return f"{k[:4]}...{k[-4:]}"


def _json_compact(obj: Any, max_len: int = 4000) -> str:
    try:
        s = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        s = str(obj)
    return s[:max_len]


def _normalize_prob_dict(d: Dict[Any, float], floor: float = 0.0) -> Dict[Any, float]:
    out = {k: max(floor, _f(v, 0.0)) for k, v in (d or {}).items()}
    s = sum(out.values())
    if s <= 0:
        if not out:
            return {}
        return {k: 1.0 / len(out) for k in out}
    return {k: v / s for k, v in out.items()}


def _deep_find_value(obj: Any, aliases: List[str], skip_keys: Optional[set] = None) -> Any:
    skip_keys = set(skip_keys or [])
    aliases_low = {str(a).lower() for a in aliases}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).lower() in aliases_low:
                return v
        for k, v in obj.items():
            if str(k).lower() in skip_keys:
                continue
            found = _deep_find_value(v, aliases, skip_keys)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for it in obj:
            found = _deep_find_value(it, aliases, skip_keys)
            if found is not None:
                return found
    return None

# ============================================================
# 比分与方向
# ============================================================

def _normalize_score_text(s: Any) -> str:
    return str(s).strip().replace(" ", "").replace("：", "-").replace(":", "-")


def _parse_score(s: Any) -> Tuple[Optional[int], Optional[int]]:
    try:
        ss = _normalize_score_text(s)
        if not ss:
            return None, None
        if "胜" in ss and "其他" in ss:
            return 9, 0
        if "平" in ss and "其他" in ss:
            return 9, 9
        if "负" in ss and "其他" in ss:
            return 0, 9
        if ss in ("主胜", "客胜", "平局", "胜", "平", "负", "home", "draw", "away", "弃权"):
            return None, None
        m = re.search(r"(\d{1,2})\s*[-:：]\s*(\d{1,2})", ss)
        if not m:
            return None, None
        return int(m.group(1)), int(m.group(2))
    except Exception:
        return None, None


def _score_direction(score_str: Any) -> Optional[str]:
    ss = _normalize_score_text(score_str)
    if ss in SCORE_OTHERS_HOME or ss == "9-0":
        return "home"
    if ss in SCORE_OTHERS_DRAW or ss == "9-9":
        return "draw"
    if ss in SCORE_OTHERS_AWAY or ss == "0-9":
        return "away"
    h, a = _parse_score(ss)
    if h is None:
        return None
    if h > a:
        return "home"
    if h < a:
        return "away"
    return "draw"


def _score_total(score_str: Any) -> Optional[int]:
    h, a = _parse_score(score_str)
    if h is None:
        return None
    return h + a


def _direction_cn(direction: str) -> str:
    return {"home": "主胜", "draw": "平局", "away": "客胜", "abstain": "弃权"}.get(direction, "弃权")


def _dir_from_cn(v: Any) -> Optional[str]:
    s = str(v).strip().lower()
    if s in ("home", "主胜", "胜", "主", "主队", "home_win"):
        return "home"
    if s in ("draw", "平局", "平", "和", "tie"):
        return "draw"
    if s in ("away", "客胜", "负", "客", "客队", "away_win"):
        return "away"
    return None


def _score_display_label(score_str: Any, direction_code: Optional[str] = None) -> str:
    ss = _normalize_score_text(score_str)
    if ss in SCORE_OTHERS_HOME or ss in ("胜其他", "9-0"):
        return "胜其他"
    if ss in SCORE_OTHERS_DRAW or ss in ("平其他", "9-9"):
        return "平其他"
    if ss in SCORE_OTHERS_AWAY or ss in ("负其他", "0-9"):
        return "负其他"
    return ss

# ============================================================
# match 规范化：只做字段兼容，不做本地判断
# ============================================================

def normalize_match(raw_m: Dict[str, Any]) -> Dict[str, Any]:
    m = dict(raw_m or {})
    nested_keys = ["v2_odds_dict", "odds_dict", "odds", "v2", "odds_v2", "packet", "raw_odds", "data", "detail"]
    for nk in nested_keys:
        if isinstance(m.get(nk), dict):
            # 不覆盖已存在关键字段，尽量保留外层原始值
            for k, v in m[nk].items():
                if k not in m:
                    m[k] = v

    home = m.get("home_team") or m.get("home") or m.get("host") or m.get("team_home") or m.get("homeName") or "Home"
    away = m.get("away_team") or m.get("guest") or m.get("away") or m.get("team_away") or m.get("awayName") or "Away"
    m["home_team"] = home
    m["away_team"] = away
    m["home"] = home
    m["guest"] = away

    skip = {"vote", "change", "points", "information", "prediction", "stats", "smart_signals"}
    if m.get("sp_home") is None:
        v = _deep_find_value(m, ["win", "odds_win", "spf_sp3", "sp3", "胜"], skip)
        if v is not None:
            m["sp_home"] = v
            m["win"] = v
    if m.get("sp_draw") is None:
        v = _deep_find_value(m, ["draw", "same", "odds_draw", "spf_sp1", "sp1", "平"], skip)
        if v is not None:
            m["sp_draw"] = v
            m["same"] = v
    if m.get("sp_away") is None:
        v = _deep_find_value(m, ["lose", "away_win", "odds_lose", "spf_sp0", "sp0", "负"], skip)
        if v is not None:
            m["sp_away"] = v
            m["lose"] = v

    if "give_ball" not in m:
        m["give_ball"] = m.get("handicap") or m.get("rq") or m.get("let_ball") or "0"

    if not isinstance(m.get("change"), dict):
        ch = {}
        for src_key, dst_key in [
            ("change_win", "win"), ("cw", "win"),
            ("change_same", "same"), ("cs", "same"), ("change_draw", "same"),
            ("change_lose", "lose"), ("cl", "lose"), ("change_away", "lose"),
        ]:
            if src_key in m:
                ch[dst_key] = m.get(src_key)
        m["change"] = ch
    return m


def get_market_odds_for_score(match_obj: Dict[str, Any], score: str) -> float:
    # 仅用于展示赔率，不做价值主裁。
    score = _normalize_score_text(score)
    key = CRS_FULL_MAP.get(score)
    if key:
        return _f(match_obj.get(key, 0))
    if score in SCORE_OTHERS_HOME or score in ("胜其他", "9-0"):
        return _f(match_obj.get("crs_win", 0))
    if score in SCORE_OTHERS_DRAW or score in ("平其他", "9-9"):
        return _f(match_obj.get("crs_same", 0))
    if score in SCORE_OTHERS_AWAY or score in ("负其他", "0-9"):
        return _f(match_obj.get("crs_lose", 0))
    return 0.0

# ============================================================
# 外部联网情报入口：只作为原始情报，不做本地结论
# ============================================================

async def _fetch_json_or_text(session: aiohttp.ClientSession, method: str, url: str, **kwargs) -> Tuple[bool, Any, str]:
    try:
        async with session.request(method, url, timeout=aiohttp.ClientTimeout(total=30), **kwargs) as r:
            text = await r.text()
            if r.status < 200 or r.status >= 300:
                return False, {"status": r.status, "text": text[:1000]}, "http_error"
            try:
                return True, json.loads(text), "json"
            except Exception:
                return True, text, "text"
    except Exception as e:
        return False, {"error": str(e)[:300]}, "exception"


def _parse_external_endpoints() -> List[str]:
    raw = os.environ.get("EXTERNAL_CONTEXT_ENDPOINTS", "") or os.environ.get("EXTERNAL_CONTEXT_URLS", "")
    if not raw:
        one = os.environ.get("EXTERNAL_CONTEXT_URL", "").strip()
        return [one] if one else []
    try:
        arr = json.loads(raw)
        if isinstance(arr, list):
            return [str(x).strip() for x in arr if str(x).strip()]
    except Exception:
        pass
    return [x.strip() for x in raw.split(",") if x.strip()]


async def build_external_context_for_match(session: aiohttp.ClientSession, match_obj: Dict[str, Any]) -> Dict[str, Any]:
    ctx = {
        "enabled": bool(ENABLE_EXTERNAL_CONTEXT),
        "source_quality": "disabled" if not ENABLE_EXTERNAL_CONTEXT else "missing",
        "items": [],
        "errors": [],
    }
    if not ENABLE_EXTERNAL_CONTEXT:
        return ctx

    h = match_obj.get("home_team", "")
    a = match_obj.get("away_team", "")
    league = match_obj.get("league", match_obj.get("cup", ""))
    payload = {"home_team": h, "away_team": a, "league": league, "match": match_obj}

    endpoints = _parse_external_endpoints()
    token = os.environ.get("EXTERNAL_CONTEXT_TOKEN", "").strip()
    if endpoints:
        for url in endpoints[:3]:
            headers = {"Content-Type": "application/json"}
            if token:
                headers["Authorization"] = f"Bearer {token}"
            ok, data, kind = await _fetch_json_or_text(session, os.environ.get("EXTERNAL_CONTEXT_METHOD", "POST").upper(), url, headers=headers, json=payload)
            if ok:
                ctx["items"].append({"source": url, "kind": kind, "data": data})
                ctx["source_quality"] = "provider"
            else:
                ctx["errors"].append({"source": url, "data": data})

    # Bing 可选。只抓摘要，不做判断。
    bing_key = os.environ.get("BING_SEARCH_API_KEY", "").strip()
    if bing_key:
        endpoint = os.environ.get("BING_SEARCH_API_ENDPOINT", "https://api.bing.microsoft.com/v7.0/search").strip()
        q = f"{h} {a} {league} injury lineup weather football"
        headers = {"Ocp-Apim-Subscription-Key": bing_key}
        ok, data, kind = await _fetch_json_or_text(session, "GET", endpoint, headers=headers, params={"q": q, "count": 5, "mkt": os.environ.get("BING_SEARCH_MARKET", "en-US")})
        if ok:
            ctx["items"].append({"source": "bing", "kind": kind, "query": q, "data": data})
            ctx["source_quality"] = "search"
        else:
            ctx["errors"].append({"source": "bing", "data": data})

    if ctx["items"] and ctx["source_quality"] == "missing":
        ctx["source_quality"] = "external"
    return ctx


async def enrich_external_context(match_analyses: List[Dict[str, Any]]) -> None:
    if not ENABLE_EXTERNAL_CONTEXT or aiohttp is None:
        for ma in match_analyses:
            ma["external_context"] = {"enabled": False, "source_quality": "disabled", "items": [], "errors": []}
        return
    async with aiohttp.ClientSession() as session:
        tasks = [build_external_context_for_match(session, ma["match"]) for ma in match_analyses]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for ma, r in zip(match_analyses, results):
            if isinstance(r, Exception):
                ma["external_context"] = {"enabled": True, "source_quality": "error", "items": [], "errors": [str(r)[:300]]}
            else:
                ma["external_context"] = r


def _format_external_context_for_prompt(ctx: Dict[str, Any]) -> str:
    if not isinstance(ctx, dict) or not ctx.get("enabled"):
        return "external_context: disabled\n"
    lines = [f"source_quality:{ctx.get('source_quality','missing')}"]
    items = ctx.get("items", [])
    if not items:
        lines.append("items: []")
    else:
        for idx, item in enumerate(items[:5], 1):
            lines.append(f"item{idx}:{_json_compact(item, 1800)}")
    if ctx.get("errors"):
        lines.append(f"errors:{_json_compact(ctx.get('errors'), 800)}")
    return "\n".join(lines) + "\n"

# ============================================================
# RAW prompt
# ============================================================

def _raw_field_line(label: str, value: Any, limit: int = 1200) -> str:
    if value is None or value == "" or value == {} or value == []:
        return ""
    return f"{label}:{str(value)[:limit].replace(chr(10), ' ')}\n"


def build_phase1_prompt(match_analyses: List[Dict[str, Any]]) -> str:
    p = "<context>\n"
    p += "你是竞彩足球 RAW-AI 比分预测模型。你只能基于本段给出的原始字段和 external_context 做判断。\n"
    p += "禁止引用或假设任何未给出的本地模型结论。禁止使用 CRS 框架、贝叶斯、本地矩阵、陷阱编号、固定常见比分模板。\n"
    p += "如果外部情报为空，必须写 data_missing，不得编造伤停、天气、首发或新闻。\n"
    p += "</context>\n\n"

    p += "<task_rules>\n"
    p += "1. 直接从原始 1X2、让球、总进球、半全场、赔率变动、散户热度、原始情报、external_context 推导方向与比分。\n"
    p += "2. 不要默认输出 1-1、2-1、1-0、0-1、1-2；若选择这些比分，必须是你从原始字段独立推导出的结果。\n"
    p += "3. top3 必须是比分分布，不要只给方向。top3[0].score 必须与 final_direction 一致。\n"
    p += "4. 必须说明为什么不选另外两个方向。\n"
    p += "5. 0-1、0-2、0-3 是合法客胜比分；其他比分可输出 4-3、3-4 等精确比分。\n"
    p += "</task_rules>\n\n"

    p += "<output_format>\n"
    p += "严格输出 JSON 数组。每场一个对象：\n"
    p += '{"match":1,"final_direction":"home/draw/away","top3":[{"score":"2-0","prob":0.18,"market_logic":"..."}],"reason":"...","ai_confidence":0-100,"risk_level":"low/medium/high","data_missing":[],"audit":{"direction_rejection":"...","total_goals":"...","money_flow":"...","external_context":"..."}}\n'
    p += "禁止 markdown，禁止 JSON 外文本。match 字段优先输出数字序号。\n"
    p += "</output_format>\n\n"

    p += "<match_data>\n"
    for i, ma in enumerate(match_analyses, 1):
        m = ma["match"]
        h = m.get("home_team", m.get("home", "Home"))
        a = m.get("away_team", m.get("guest", "Away"))
        league = m.get("league", m.get("cup", ""))
        p += f'<match index="{i}">\n'
        p += f"[{i}] {h} vs {a} | {league}\n"
        p += f"1X2欧赔原始值: 主胜={m.get('sp_home', m.get('win',''))} 平局={m.get('sp_draw', m.get('same',''))} 客胜={m.get('sp_away', m.get('lose',''))}\n"
        p += f"让球原始值: {m.get('give_ball', m.get('handicap', m.get('rq','')))}\n"
        p += "总进球a0-a7原始赔率:" + " | ".join([f"{g}={m.get(f'a{g}','')}" for g in range(8)]) + "\n"

        hf_l = []
        for k, lb in HFTF_MAP.items():
            v = m.get(k, None)
            if v not in (None, "", 0, "0"):
                hf_l.append(f"{lb}={v}")
        if hf_l:
            p += "半全场原始赔率:" + " | ".join(hf_l) + "\n"

        p += _raw_field_line("赔率变动原始字段", m.get("change"), 2000)
        p += _raw_field_line("散户/热度原始字段", m.get("vote"), 1600)
        p += _raw_field_line("information原始字段", m.get("information"), 2500)
        p += _raw_field_line("points原始字段", m.get("points"), 3000)
        p += _raw_field_line("raw_extra原始字段", {k: v for k, v in m.items() if k in ("weather", "injury", "lineup", "news", "motivation", "schedule", "home_form", "away_form")}, 2500)
        p += "<external_context>\n" + _format_external_context_for_prompt(ma.get("external_context", {})) + "</external_context>\n"
        p += "</match>\n\n"
    p += "</match_data>\n"
    return p


def build_claude_final_audit_prompt(match_analyses: List[Dict[str, Any]], phase1_results: Dict[str, Dict[int, Dict[str, Any]]]) -> str:
    p = "<final_audit_context>\n"
    p += "你是 Claude 最终 RAW-AI 主裁。你看到的是原始比赛字段和 GPT/Grok/Gemini 初审。\n"
    p += "你不是反指模型，不需要为了体现审计而反对初审。选择证据最完整、最符合原始字段的一组。\n"
    p += "禁止引用 CRS、贝叶斯、本地矩阵、本地风控编号、固定常见比分模板。\n"
    p += "如果你改动初审比分，必须指出原始字段中的硬依据；不能只是把 2-0 改成 2-1 或把 3-1 改成 2-1。\n"
    p += "</final_audit_context>\n\n"
    p += build_phase1_prompt(match_analyses)
    p += "\n<phase1_ai_results>\n"
    for ai in PHASE1_NAMES:
        p += f"<{ai}>\n"
        rs = phase1_results.get(ai, {}) or {}
        for idx in range(1, len(match_analyses) + 1):
            r = rs.get(idx)
            if r:
                p += json.dumps({
                    "match": idx,
                    "ai_score": r.get("ai_score"),
                    "final_direction": r.get("final_direction"),
                    "top3": r.get("top3", []),
                    "ai_confidence": r.get("ai_confidence"),
                    "risk_level": r.get("risk_level"),
                    "reason": r.get("reason", ""),
                    "audit": r.get("audit", {}),
                    "data_missing": r.get("data_missing", []),
                }, ensure_ascii=False) + "\n"
            else:
                p += f"[{idx}] 弃权\n"
        p += f"</{ai}>\n"
    p += "</phase1_ai_results>\n\n"
    p += "<final_output_rule>严格输出 JSON 数组，字段同 phase1。禁止 JSON 外文本。</final_output_rule>\n"
    return p

# ============================================================
# API gateway
# ============================================================

def _clean_env_key(*names: str) -> str:
    for name in names:
        v = str(os.environ.get(name, "")).strip(" \t\n\r\"'")
        if v:
            return v
    return ""


def _clean_env_url(*names: str) -> str:
    for name in names:
        v = str(os.environ.get(name, "")).strip(" \t\n\r\"'")
        if not v:
            continue
        m = re.search(r"(https?://[^\s'\"]+)", v)
        return m.group(1) if m else v
    return ""


def get_common_key() -> str:
    # API_KEY 是唯一推荐；旧变量只作为兼容兜底。
    return _clean_env_key("API_KEY", "GPT_API_KEY", "OPENAI_API_KEY", "GROK_API_KEY", "GEMINI_API_KEY", "CLAUDE_API_KEY")


def get_common_url() -> str:
    return _clean_env_url("API_URL", "GPT_API_URL", "OPENAI_API_URL", "BASE_URL", "GROK_API_URL", "GEMINI_API_URL", "CLAUDE_API_URL")


def _model_for(ai_name: str) -> str:
    env_name = f"{ai_name.upper()}_MODEL"
    return str(os.environ.get(env_name, DEFAULT_MODELS[ai_name])).strip() or DEFAULT_MODELS[ai_name]


def _chat_url(base_url: str) -> str:
    u = (base_url or "").rstrip("/")
    if not u:
        return ""
    if u.endswith("/chat/completions"):
        return u
    if "/chat/completions" in u:
        return u
    return u + "/chat/completions"


def debug_ai_config() -> None:
    key = get_common_key()
    url = get_common_url()
    print(f"[COMMON GATEWAY] API_URL={url or '<missing>'} API_KEY={_mask_key(key)}")
    for n in AI_NAMES:
        print(f"[AI CONFIG] {n.upper()} model={_model_for(n)}")
    print(f"[AI MODE] pure_raw=True singleflight={AI_SINGLEFLIGHT_ENABLED} cache_ttl={AI_DECISION_CACHE_TTL} max_req_per_ai={AI_MAX_REQUESTS_PER_AI} external={ENABLE_EXTERNAL_CONTEXT}")

# ============================================================
# AI response extraction / parsing
# ============================================================

def _save_debug_dump(ai_name: str, data: Any, tag: str, raw_text: Optional[str] = None) -> None:
    try:
        os.makedirs("data/debug", exist_ok=True)
        ts = int(time.time())
        f1 = f"data/debug/{ai_name}_{tag}_{ts}.json"
        with open(f1, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"    失败响应已保存: {f1}")
        if raw_text is not None:
            f2 = f"data/debug/{ai_name}_{tag}_raw_{ts}.txt"
            with open(f2, "w", encoding="utf-8") as f:
                f.write(raw_text)
            print(f"    原始文本已保存: {f2}")
    except Exception:
        pass


def _extract_response_text(data: Any, ai_name: str = "") -> str:
    candidates: List[Tuple[str, str]] = []

    def add(v: Any, path: str) -> None:
        if isinstance(v, str):
            t = v.strip()
            if t and len(t) >= 2:
                candidates.append((path, t))

    try:
        if isinstance(data, dict):
            # OpenAI chat compatible
            for ch in data.get("choices", []) or []:
                if not isinstance(ch, dict):
                    continue
                msg = ch.get("message", {}) or {}
                if isinstance(msg, dict):
                    content = msg.get("content")
                    if isinstance(content, str):
                        add(content, "choices.message.content")
                    elif isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict):
                                add(item.get("text"), "choices.message.content[].text")
                                add(item.get("content"), "choices.message.content[].content")
                    for k in ["text", "answer", "response", "output_text", "final_answer", "output", "result", "model_response"]:
                        add(msg.get(k), f"choices.message.{k}")
                add(ch.get("text"), "choices.text")

            # Responses API style
            add(data.get("output_text"), "output_text")
            add(data.get("text"), "text")
            add(data.get("answer"), "answer")
            add(data.get("result"), "result")
            if isinstance(data.get("output"), list):
                for oi in data["output"]:
                    if isinstance(oi, dict):
                        add(oi.get("text"), "output[].text")
                        if isinstance(oi.get("content"), list):
                            for ct in oi["content"]:
                                if isinstance(ct, dict):
                                    add(ct.get("text"), "output[].content[].text")

            # Gemini style if gateway returns original format
            if isinstance(data.get("candidates"), list):
                for cand in data["candidates"]:
                    if isinstance(cand, dict):
                        cont = cand.get("content", {})
                        if isinstance(cont, dict):
                            for part in cont.get("parts", []) or []:
                                if isinstance(part, dict):
                                    add(part.get("text"), "candidates.content.parts.text")

            # deep fallback
            def walk(obj: Any, path: str = "") -> None:
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        if k in {"reasoning", "reasoning_content", "thinking", "chain_of_thought", "thoughts"}:
                            continue
                        walk(v, path + "." + str(k))
                elif isinstance(obj, list):
                    for i, v in enumerate(obj[:20]):
                        walk(v, f"{path}[{i}]")
                elif isinstance(obj, str):
                    s = obj.strip()
                    if len(s) >= 20 and ("top3" in s or "final_direction" in s or '"match"' in s or "'match'" in s):
                        add(s, path)

            walk(data, "root")
    except Exception as e:
        print(f"    响应文本提取异常:{str(e)[:120]}")

    if not candidates:
        if isinstance(data, str):
            return data.strip()
        return ""

    def score_item(item: Tuple[str, str]) -> Tuple[int, int]:
        _, t = item
        score = 0
        if "top3" in t:
            score += 5
        if "final_direction" in t:
            score += 3
        if re.search(r"\[\s*\{", t):
            score += 3
        return score, len(t)

    return max(candidates, key=score_item)[1].strip()


def _strip_json_fences(text: str) -> str:
    clean = text or ""
    clean = re.sub(r"<think(?:ing)?>.*?</think(?:ing)?>", "", clean, flags=re.DOTALL | re.IGNORECASE)
    clean = re.sub(r"```(?:json|JSON|javascript|js|python)?", "", clean)
    clean = clean.replace("```", "")
    return clean.strip()


def _json_loads_loose(s: str) -> Any:
    s = s.strip()
    # remove trailing commas
    s = re.sub(r",\s*([}\]])", r"\1", s)
    try:
        return json.loads(s)
    except Exception:
        pass
    # Python-like single quote fallback
    try:
        import ast
        return ast.literal_eval(s)
    except Exception:
        pass
    raise ValueError("json_parse_failed")


def _extract_json_array(text: str) -> List[Any]:
    clean = _strip_json_fences(text)
    if not clean:
        return []

    # Direct object wrappers
    for candidate in [clean]:
        try:
            obj = _json_loads_loose(candidate)
            if isinstance(obj, list):
                return obj
            if isinstance(obj, dict):
                for k in ["predictions", "results", "matches", "data", "items", "output"]:
                    if isinstance(obj.get(k), list):
                        return obj[k]
                if "match" in obj or "top3" in obj or "score" in obj:
                    return [obj]
        except Exception:
            pass

    arrays = []
    start_positions = [m.start() for m in re.finditer(r"\[", clean)]
    for start in start_positions:
        depth = 0
        in_str = False
        esc = False
        end = -1
        quote = ""
        for i in range(start, len(clean)):
            ch = clean[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == quote:
                    in_str = False
                continue
            if ch in ('"', "'"):
                in_str = True
                quote = ch
            elif ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end > start:
            frag = clean[start:end]
            if "top3" not in frag and "match" not in frag and "score" not in frag:
                continue
            try:
                arr = _json_loads_loose(frag)
                if isinstance(arr, list):
                    arrays.append(arr)
            except Exception:
                continue
    if arrays:
        return max(arrays, key=lambda a: sum(1 for x in a if isinstance(x, dict)))
    return []


def _prob_to_float(v: Any) -> float:
    if v is None:
        return 0.0
    if isinstance(v, str):
        s = v.strip().replace("％", "%")
        pct = "%" in s
        fv = _f(s, 0.0)
        if pct:
            return fv
        if 0 < fv <= 1:
            return fv * 100
        return fv
    fv = _f(v, 0.0)
    if 0 < fv <= 1:
        return fv * 100
    return fv


def _score_from_candidate(obj: Any) -> str:
    if isinstance(obj, str):
        return _normalize_score_text(obj)
    if not isinstance(obj, dict):
        return ""
    for k in ["score", "predicted_score", "ai_score", "final_score", "比分", "预测比分", "top_score", "result_score"]:
        if obj.get(k) not in (None, ""):
            return _normalize_score_text(obj.get(k))
    if obj.get("home_goals") is not None and obj.get("away_goals") is not None:
        return f"{_i(obj.get('home_goals'))}-{_i(obj.get('away_goals'))}"
    return ""


def _normalize_top3(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_top3 = None
    for k in ["top3", "top_3", "top_scores", "scores", "score_candidates", "candidates", "比分候选"]:
        if isinstance(item.get(k), list):
            raw_top3 = item[k]
            break
    if raw_top3 is None:
        raw_top3 = []

    top3: List[Dict[str, Any]] = []
    for cand in raw_top3[:8]:
        sc = _score_from_candidate(cand)
        if _parse_score(sc)[0] is None:
            continue
        if isinstance(cand, dict):
            prob = cand.get("prob", cand.get("probability", cand.get("pct", cand.get("chance", 0))))
            logic = cand.get("market_logic", cand.get("reason", cand.get("logic", "")))
            top3.append({"score": sc, "prob": round(_prob_to_float(prob), 3), "market_logic": str(logic)[:500]})
        else:
            top3.append({"score": sc, "prob": 0.0, "market_logic": ""})
        if len(top3) >= 3:
            break
    if not top3:
        sc = _score_from_candidate(item)
        if _parse_score(sc)[0] is not None:
            top3 = [{"score": sc, "prob": round(_prob_to_float(item.get("prob", item.get("probability", 0))), 3), "market_logic": ""}]
    return top3


def _normalize_direction(v: Any, top_score: str = "") -> str:
    d = _dir_from_cn(v)
    if d:
        return d
    sd = _score_direction(top_score)
    return sd or "draw"


def _match_index_from_item(item: Dict[str, Any], fallback_idx: int, num_matches: int) -> Optional[int]:
    raw = item.get("match", item.get("index", item.get("match_index", item.get("id"))))
    if isinstance(raw, int):
        idx = raw
        return idx if 1 <= idx <= num_matches else None
    if raw is not None:
        s = str(raw).strip()
        # Prefer pure numeric or bracketed [1]
        m = re.match(r"^\s*(\d+)\s*$", s) or re.match(r"^\s*\[(\d+)\]", s)
        if m:
            idx = int(m.group(1))
            return idx if 1 <= idx <= num_matches else None
        # Name string like 阿森纳 vs 马竞: use array order fallback
        if fallback_idx <= num_matches:
            return fallback_idx
        return None
    return fallback_idx if fallback_idx <= num_matches else None


def _parse_ai_json(raw_text: str, num_matches: int, ai_name: str = "") -> Dict[int, Dict[str, Any]]:
    arr = _extract_json_array(raw_text)
    results: Dict[int, Dict[str, Any]] = {}
    if not isinstance(arr, list):
        arr = []
    for pos, item in enumerate(arr, 1):
        if not isinstance(item, dict):
            continue
        top3 = _normalize_top3(item)
        if not top3:
            continue
        top_score = top3[0]["score"]
        mid = _match_index_from_item(item, pos, num_matches)
        if not mid:
            continue
        final_direction = _normalize_direction(item.get("final_direction", item.get("direction", item.get("result", ""))), top_score)
        conf = item.get("ai_confidence", item.get("confidence", item.get("conf", 60)))
        traps = item.get("detected_traps", item.get("traps", item.get("risk_flags", [])))
        if not isinstance(traps, list):
            traps = [str(traps)] if traps else []
        data_missing = item.get("data_missing", [])
        if not isinstance(data_missing, list):
            data_missing = [str(data_missing)] if data_missing else []
        results[mid] = {
            "top3": top3,
            "ai_score": top_score,
            "reason": str(item.get("reason", item.get("analysis", item.get("explanation", ""))))[:4000],
            "ai_confidence": int(_clip(_f(conf, 60), 0, 100)),
            "risk_level": str(item.get("risk_level", item.get("risk", "medium"))),
            "is_score_others": _score_display_label(top_score) in ("胜其他", "平其他", "负其他"),
            "detected_traps": traps,
            "data_missing": data_missing,
            "audit": item.get("audit", {}) if isinstance(item.get("audit", {}), dict) else {},
            "final_direction": final_direction,
            "raw_item": item,
        }
    if not results and AI_PARSE_DEBUG:
        print(f"    [{ai_name}] parse empty. raw={raw_text[:500]}")
    return results

# ============================================================
# AI calls
# ============================================================

async def async_call_one_ai_batch(
    session: aiohttp.ClientSession,
    prompt: str,
    num_matches: int,
    ai_name: str,
    system_text: str,
) -> Tuple[str, Dict[int, Dict[str, Any]], str]:
    key = get_common_key()
    base_url = get_common_url()
    model = _model_for(ai_name)
    AI_CALL_STATUS[ai_name] = {"ok": False, "status": "init", "model": model, "count": 0, "requests": 0}

    if not key:
        print(f"  [{ai_name.upper()}] no_key: 检查 API_KEY / GPT_API_KEY 等环境变量")
        AI_CALL_STATUS[ai_name].update({"status": "no_key"})
        return ai_name, {}, "no_key"
    if not base_url:
        print(f"  [{ai_name.upper()}] no_url: 检查 API_URL / GPT_API_URL 等环境变量")
        AI_CALL_STATUS[ai_name].update({"status": "no_url"})
        return ai_name, {}, "no_url"

    url = _chat_url(base_url)
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_text},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.12 if ai_name == "claude" else 0.18 if ai_name in ("gpt", "gemini") else 0.22,
    }

    for req_no in range(1, AI_MAX_REQUESTS_PER_AI + 1):
        AI_CALL_STATUS[ai_name]["requests"] = req_no
        gateway = url.split("/v1")[0][:60]
        print(f"  [连接中] {ai_name.upper()} | {model} @ {gateway} | request#{req_no}")
        t0 = time.time()
        try:
            timeout = aiohttp.ClientTimeout(total=None, connect=AI_CONNECT_TIMEOUT, sock_connect=AI_CONNECT_TIMEOUT, sock_read=AI_READ_TIMEOUT)
            async with session.post(url, headers=headers, json=payload, timeout=timeout) as r:
                elapsed = round(time.time() - t0, 1)
                text_for_error = ""
                if r.status != 200:
                    try:
                        text_for_error = await r.text()
                    except Exception:
                        pass
                    print(f"    HTTP {r.status} | {elapsed}s | {text_for_error[:180]}")
                    AI_CALL_STATUS[ai_name].update({"status": f"http_{r.status}"})
                    continue
                try:
                    data = await r.json(content_type=None)
                except Exception:
                    text = await r.text()
                    raw_text = text.strip()
                    data = {"raw": raw_text}
                raw_text = _extract_response_text(data, ai_name)
                if not raw_text:
                    print("    空文本响应")
                    _save_debug_dump(ai_name, data, "empty", "")
                    AI_CALL_STATUS[ai_name].update({"status": "empty"})
                    continue
                parsed = _parse_ai_json(raw_text, num_matches, ai_name)
                if parsed:
                    print(f"    {ai_name.upper()} 完成: {len(parsed)}/{num_matches} | {round(time.time()-t0,1)}s")
                    AI_CALL_STATUS[ai_name].update({"ok": True, "status": "ok", "count": len(parsed), "model": model})
                    return ai_name, parsed, model
                print(f"    解析0条，raw前200字: {raw_text[:200].replace(chr(10),' ')}")
                _save_debug_dump(ai_name, data, "parse0", raw_text)
                AI_CALL_STATUS[ai_name].update({"status": "parse0"})
        except asyncio.TimeoutError:
            print(f"    {ai_name.upper()} 读取超时")
            AI_CALL_STATUS[ai_name].update({"status": "timeout"})
        except Exception as e:
            print(f"    {ai_name.upper()} 调用异常: {str(e)[:160]}")
            AI_CALL_STATUS[ai_name].update({"status": "error", "error": str(e)[:300]})
    return ai_name, {}, "all_failed"


def _phase_system(ai_name: str) -> str:
    base = "只输出 JSON 数组。禁止 markdown、禁止 JSON 外说明。你不得引用 CRS、本地矩阵、贝叶斯、陷阱编号或固定常见比分模板。"
    if ai_name == "gpt":
        return "你是 RAW 赔率结构和比分分布分析师。" + base
    if ai_name == "grok":
        return "你是 RAW 资金流、散户热度和变盘分析师。" + base
    if ai_name == "gemini":
        return "你是 RAW 多市场一致性和异常结构分析师。" + base
    if ai_name == "claude":
        return "你是最终 RAW-AI 主裁，不是反指模型。默认尊重证据最完整的初审，只有原始字段硬反证才改票。" + base
    return base


def _stable_ai_cache_key(match_analyses: List[Dict[str, Any]], phase: str = "pure") -> str:
    compact = []
    for ma in match_analyses:
        m = ma.get("match", {})
        compact.append({
            "home": m.get("home_team"),
            "away": m.get("away_team"),
            "league": m.get("league", m.get("cup", "")),
            "match_num": m.get("match_num"),
            "id": m.get("id"),
            "sp_home": m.get("sp_home", m.get("win")),
            "sp_draw": m.get("sp_draw", m.get("same")),
            "sp_away": m.get("sp_away", m.get("lose")),
            "give_ball": m.get("give_ball"),
            "a": [m.get(f"a{i}") for i in range(8)],
            "change": m.get("change"),
            "vote": m.get("vote"),
        })
    raw = json.dumps({"version": ENGINE_VERSION, "phase": phase, "matches": compact}, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


async def _run_ai_matrix_two_phase_inner(match_analyses: List[Dict[str, Any]], cache_key: str) -> Dict[str, Dict[int, Dict[str, Any]]]:
    if aiohttp is None:
        print("  [AI ERROR] aiohttp 未安装，AI弃权")
        return {n: {} for n in AI_NAMES}

    debug_ai_config()
    num = len(match_analyses)
    prompt = build_phase1_prompt(match_analyses)
    print(f"  [v18.4 Phase1 Prompt] {len(prompt):,}字符 → GPT/Grok/Gemini | common_gateway={bool(get_common_key() and get_common_url())}")

    all_results: Dict[str, Dict[int, Dict[str, Any]]] = {n: {} for n in AI_NAMES}
    connector = aiohttp.TCPConnector(limit=8, use_dns_cache=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            async_call_one_ai_batch(session, prompt, num, name, _phase_system(name))
            for name in PHASE1_NAMES
        ]
        phase1 = await asyncio.gather(*tasks, return_exceptions=True)
        for res in phase1:
            if isinstance(res, tuple):
                all_results[res[0]] = res[1]
            else:
                print(f"  [Phase1 ERROR] {res}")

        audit_prompt = build_claude_final_audit_prompt(match_analyses, all_results)
        print(f"  [v18.4 Phase2 Claude Audit] {len(audit_prompt):,}字符")
        cl_name, cl_res, _ = await async_call_one_ai_batch(session, audit_prompt, num, "claude", _phase_system("claude"))
        all_results["claude"] = cl_res or {}

    ok = sum(1 for n in AI_NAMES if all_results.get(n))
    status = {k: AI_CALL_STATUS.get(k, {}) for k in AI_NAMES}
    print(f"  [完成] {ok}/4 AI有数据 | status={status}")

    if not AI_DISABLE_CACHE and ok > 0:
        _AI_RESULT_CACHE[cache_key] = (time.time(), all_results, status)
    elif ok == 0:
        print("  [AI CACHE] 本轮 0/4 AI 有效，不写入缓存；PURE 模式将弃权")
    return all_results


async def run_ai_matrix_two_phase(match_analyses: List[Dict[str, Any]]) -> Dict[str, Dict[int, Dict[str, Any]]]:
    cache_key = _stable_ai_cache_key(match_analyses)
    now = time.time()
    if not AI_DISABLE_CACHE and cache_key in _AI_RESULT_CACHE:
        ts, results, _ = _AI_RESULT_CACHE[cache_key]
        if now - ts <= AI_DECISION_CACHE_TTL:
            print(f"  [AI CACHE] 命中同一批次结果，避免重复请求/重复扣费 ttl={AI_DECISION_CACHE_TTL}s")
            return results
        _AI_RESULT_CACHE.pop(cache_key, None)

    if AI_SINGLEFLIGHT_ENABLED and cache_key in _AI_INFLIGHT_TASKS:
        print("  [AI SINGLEFLIGHT] 同批次AI正在运行，等待首个结果，避免重复扣费")
        return await _AI_INFLIGHT_TASKS[cache_key]

    task = asyncio.create_task(_run_ai_matrix_two_phase_inner(match_analyses, cache_key))
    if AI_SINGLEFLIGHT_ENABLED:
        _AI_INFLIGHT_TASKS[cache_key] = task
    try:
        return await task
    finally:
        if AI_SINGLEFLIGHT_ENABLED:
            _AI_INFLIGHT_TASKS.pop(cache_key, None)


def _run_async(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    if loop and loop.is_running():
        import concurrent.futures
        def runner():
            nl = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(nl)
                return nl.run_until_complete(coro)
            finally:
                nl.close()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(runner).result()
    return asyncio.run(coro)

# ============================================================
# AI result selection and frontend compatibility
# ============================================================

def _valid_ai_score_from_response(r: Dict[str, Any]) -> str:
    if not isinstance(r, dict):
        return ""
    sc = _score_from_candidate(r.get("ai_score", ""))
    if _parse_score(sc)[0] is not None:
        return sc
    top3 = r.get("top3", [])
    if isinstance(top3, list) and top3:
        sc = _score_from_candidate(top3[0])
        if _parse_score(sc)[0] is not None:
            return sc
    return ""


def _phase1_exact_consensus(ai_responses: Dict[str, Dict[str, Any]]) -> Tuple[str, int, List[str]]:
    counts: Dict[str, List[str]] = {}
    for name in PHASE1_NAMES:
        sc = _valid_ai_score_from_response(ai_responses.get(name, {}))
        if sc:
            counts.setdefault(sc, []).append(name)
    if not counts:
        return "", 0, []
    sc, names = sorted(counts.items(), key=lambda kv: (len(kv[1]), kv[0]), reverse=True)[0]
    return sc, len(names), names


def _choose_final_ai(ai_responses: Dict[str, Dict[str, Any]]) -> Tuple[str, Dict[str, Any], str]:
    # Claude 成功，直接主裁。纯模式不再本地 guarded 改票。
    if _valid_ai_score_from_response(ai_responses.get("claude", {})):
        return "claude", ai_responses["claude"], "claude_pure_authority"
    # Claude 失败：Phase1 同比分共识 >=2。
    sc, n, names = _phase1_exact_consensus(ai_responses)
    if sc and n >= 2:
        # 取共识里置信度最高那家作为 reason/top3 来源。
        best_name = max(names, key=lambda nm: _f(ai_responses.get(nm, {}).get("ai_confidence", 60), 60))
        return best_name, ai_responses[best_name], f"phase1_exact_consensus:{','.join(names)}"
    # 否则用置信度最高的有效 phase1。
    valid = [(name, ai_responses.get(name, {})) for name in PHASE1_NAMES if _valid_ai_score_from_response(ai_responses.get(name, {}))]
    if valid:
        name, r = max(valid, key=lambda nr: _f(nr[1].get("ai_confidence", 60), 60))
        return name, r, f"phase1_best_confidence:{name}"
    return "", {}, "ai_abstain_no_valid_model"


def _direction_pct_from_top3(top3: List[Dict[str, Any]], final_direction: str) -> Dict[str, float]:
    mass = {"home": 0.0, "draw": 0.0, "away": 0.0}
    for cand in top3 or []:
        sc = _score_from_candidate(cand)
        d = _score_direction(sc)
        if d in mass:
            mass[d] += max(0.0, _prob_to_float(cand.get("prob", 0) if isinstance(cand, dict) else 0))
    if sum(mass.values()) <= 0:
        # 这不是本地模型，只是 UI 兼容。方向来自 AI，概率用保守显示。
        if final_direction in mass:
            mass[final_direction] = 50.0
            for d in mass:
                if d != final_direction:
                    mass[d] = 25.0
        else:
            mass = {"home": 33.3, "draw": 33.3, "away": 33.4}
    s = sum(mass.values())
    return {k: round(v / s * 100, 1) for k, v in mass.items()}


def _goal_range_from_score(score: str) -> Tuple[int, int, str]:
    total = _score_total(score)
    if total is None:
        return 0, 7, "ai_raw_unknown"
    if total <= 1:
        return 0, 1, "ai_raw_0_1_goals"
    if total == 2:
        return 1, 2, "ai_raw_2_goals"
    if total == 3:
        return 2, 3, "ai_raw_3_goals"
    if total == 4:
        return 3, 4, "ai_raw_4_goals"
    if total == 5:
        return 4, 5, "ai_raw_5_goals"
    return 5, 8, "ai_raw_6plus_goals"


def _abstain_prediction(reason: str = "AI全失败，PURE模式不使用本地兜底") -> Dict[str, Any]:
    return {
        "predicted_score": "弃权",
        "predicted_label": "弃权",
        "result": "弃权",
        "display_direction": "弃权",
        "final_direction": "abstain",
        "is_abstain": True,
        "is_score_others": False,
        "home_win_pct": 0.0,
        "draw_pct": 0.0,
        "away_win_pct": 0.0,
        "confidence": 0,
        "confidence_meaning": "PURE RAW-AI 模式：AI全失败即弃权，不使用本地兜底",
        "risk_level": "高",
        "dir_confidence": 0,
        "dir_gap": 0,
        "scenario": "ai_abstain",
        "goal_range": (0, 0),
        "bayesian_evidences": [reason],
        "bayesian_prior": {},
        "override_triggered": False,
        "traps_detected": [],
        "trap_count": 0,
        "trap_severity": 0,
        "trap_details": [],
        "trap_flags": {},
        "fair_1x2": {},
        "fair_1x2_method": "disabled_pure_raw_ai",
        "market_overround": 0.0,
        "raw_implied_1x2": {},
        "crs_shape": "disabled_pure_raw_ai",
        "crs_moments": {},
        "crs_margin": 0.0,
        "crs_coverage": 0.0,
        "crs_implied_probs": {},
        "crs_low_rank_info": {},
        "top_score_candidates": [],
        "unified_matrix_top_scores": [],
        "unified_goal_probs": {},
        "fair_1x2_pack": {},
        "mixed_target_dir": {},
        "unified_source": "disabled_pure_raw_ai",
        "decision_source": "ai_abstain_no_local_fallback",
        "ai_authority_mode": "pure_raw_ai",
        "suggested_kelly": 0.0,
        "edge_vs_market": 0.0,
        "is_value": False,
        "ev_note": "disabled_pure_raw_ai",
        "score_model_prob": 0.0,
        "score_market_odds": 0.0,
        "score_market_implied_pct": None,
        "smart_money_signal": "",
        "smart_signals": [],
        "cold_door": {"is_cold_door": False, "strength": 0, "level": "普通", "signals": [], "sharp_confirmed": False, "dark_verdict": ""},
        "xG_home": "?",
        "xG_away": "?",
        "expected_total_goals": 0,
        "over_under_2_5": "弃权",
        "both_score": "弃权",
        "ai_avg_confidence": 0,
        "ai_abstained": ["GPT", "GROK", "GEMINI", "CLAUDE"],
        "gpt_score": "弃权", "gpt_analysis": "弃权",
        "grok_score": "弃权", "grok_analysis": "弃权",
        "gemini_score": "弃权", "gemini_analysis": "弃权",
        "claude_score": "弃权", "claude_analysis": "弃权",
        "engine_version": ENGINE_VERSION,
        "engine_architecture": ENGINE_ARCHITECTURE,
    }


def _make_ai_prediction(
    final_name: str,
    final_r: Dict[str, Any],
    decision_source: str,
    all_ai: Dict[str, Dict[str, Any]],
    match_obj: Dict[str, Any],
) -> Dict[str, Any]:
    score = _valid_ai_score_from_response(final_r)
    direction = _score_direction(score) or _normalize_direction(final_r.get("final_direction", ""), score)
    top3 = final_r.get("top3", []) if isinstance(final_r.get("top3", []), list) else []
    top_candidates = []
    for cand in top3[:8]:
        sc = _score_from_candidate(cand)
        if _parse_score(sc)[0] is not None:
            prob = _prob_to_float(cand.get("prob", 0) if isinstance(cand, dict) else 0)
            top_candidates.append((sc, round(prob, 3)))
    if not top_candidates and score != "弃权":
        top_candidates = [(score, 0.0)]

    pct = _direction_pct_from_top3(top3, direction)
    conf = int(_clip(_f(final_r.get("ai_confidence", 60), 60), 0, 100))
    gmin, gmax, scenario = _goal_range_from_score(score)
    total_goals = _score_total(score) or 0
    final_odds = get_market_odds_for_score(match_obj, score)
    market_implied = round(100.0 / final_odds, 3) if final_odds > 1.05 else None

    ai_abstained = [n.upper() for n in AI_NAMES if not _valid_ai_score_from_response(all_ai.get(n, {}))]
    confs = [_f(r.get("ai_confidence", 60), 60) for r in all_ai.values() if isinstance(r, dict) and _valid_ai_score_from_response(r)]
    avg_conf = round(sum(confs) / len(confs), 1) if confs else conf

    def sc_of(name: str) -> str:
        return _valid_ai_score_from_response(all_ai.get(name, {})) or "弃权"

    def reason_of(name: str) -> str:
        r = all_ai.get(name, {})
        if not isinstance(r, dict) or not _valid_ai_score_from_response(r):
            return "弃权"
        return str(r.get("reason", ""))[:3000]

    evidences = [
        "PURE RAW-AI：AI成功时不使用本地比分矩阵、不使用CRS、不使用本地风控兜底。",
        f"最终来源:{decision_source}; final_model={final_name}; score={score}; direction={direction}",
        f"AI top3:{top_candidates[:5]}",
    ]
    if final_r.get("audit"):
        evidences.append("AI audit:" + _json_compact(final_r.get("audit"), 1500))
    if final_r.get("data_missing"):
        evidences.append("data_missing:" + _json_compact(final_r.get("data_missing"), 800))

    return {
        "predicted_score": score,
        "predicted_label": _score_display_label(score, direction),
        "result": _direction_cn(direction),
        "display_direction": _direction_cn(direction),
        "final_direction": direction,
        "is_abstain": False,
        "is_score_others": _score_display_label(score, direction) in ("胜其他", "平其他", "负其他"),
        "home_win_pct": pct.get("home", 0.0),
        "draw_pct": pct.get("draw", 0.0),
        "away_win_pct": pct.get("away", 0.0),
        "confidence": conf,
        "confidence_meaning": "AI自报置信度，非历史命中率；PURE模式不做本地概率改写",
        "risk_level": str(final_r.get("risk_level", "medium")),
        "dir_confidence": pct.get(direction, 0.0),
        "dir_gap": round(max(pct.values()) - sorted(pct.values(), reverse=True)[1], 1) if len(pct) >= 2 else 0.0,
        "scenario": scenario,
        "goal_range": (gmin, gmax),
        "bayesian_evidences": evidences,
        "bayesian_prior": {},
        "override_triggered": False,
        "traps_detected": [],
        "trap_count": 0,
        "trap_severity": 0,
        "trap_details": [],
        "trap_flags": {},
        "fair_1x2": {},
        "fair_1x2_method": "disabled_pure_raw_ai",
        "market_overround": 0.0,
        "raw_implied_1x2": {},
        "crs_shape": "disabled_pure_raw_ai",
        "crs_moments": {},
        "crs_margin": 0.0,
        "crs_coverage": 0.0,
        "crs_implied_probs": {},
        "crs_low_rank_info": {},
        "top_score_candidates": top_candidates,
        "unified_matrix_top_scores": top_candidates,
        "unified_goal_probs": {},
        "fair_1x2_pack": {},
        "mixed_target_dir": {},
        "unified_source": "disabled_pure_raw_ai",
        "decision_source": decision_source,
        "ai_authority_mode": "pure_raw_ai",
        "gpt_score": sc_of("gpt"), "gpt_analysis": reason_of("gpt"),
        "grok_score": sc_of("grok"), "grok_analysis": reason_of("grok"),
        "gemini_score": sc_of("gemini"), "gemini_analysis": reason_of("gemini"),
        "claude_score": sc_of("claude"), "claude_analysis": reason_of("claude"),
        "ai_abstained": ai_abstained,
        "ai_avg_confidence": avg_conf,
        "value_kill_count": 0,
        "suggested_kelly": 0.0,
        "edge_vs_market": 0.0,
        "is_value": False,
        "ev_note": "disabled_pure_raw_ai_no_local_probability",
        "score_model_prob": top_candidates[0][1] if top_candidates else 0.0,
        "score_market_odds": final_odds,
        "score_market_implied_pct": market_implied,
        "smart_money_signal": "",
        "smart_signals": [],
        "cold_door": {"is_cold_door": False, "strength": 0, "level": "普通", "signals": [], "sharp_confirmed": False, "dark_verdict": ""},
        "xG_home": "?",
        "xG_away": "?",
        "over_under_2_5": "大" if total_goals >= 3 else "小",
        "both_score": "是" if (_parse_score(score)[0] or 0) > 0 and (_parse_score(score)[1] or 0) > 0 else "否",
        "expected_total_goals": total_goals,
        "over_2_5": None,
        "btts": None,
        "bookmaker_implied_home_xg": "?",
        "bookmaker_implied_away_xg": "?",
        "sharp_detected": False,
        "sharp_dir": None,
        "fair_dir": None,
        "shin_dir": None,
        "actual_handicap_signed": None,
        "theoretical_handicap_signed": None,
        "model_consensus": len([n for n in AI_NAMES if _valid_ai_score_from_response(all_ai.get(n, {}))]),
        "total_models": 4,
        "extreme_warning": "",
        "refined_poisson": {}, "poisson": {}, "elo": {}, "random_forest": {}, "gradient_boost": {},
        "neural_net": {}, "logistic": {}, "svm": {}, "knn": {}, "dixon_coles": {}, "bradley_terry": {},
        "home_form": {}, "away_form": {}, "handicap_signal": "", "odds_movement": {}, "vote_analysis": {},
        "h2h_blood": {}, "crs_analysis": {}, "ttg_analysis": {}, "halftime": {}, "pace_rating": "",
        "kelly_home": {}, "kelly_away": {}, "odds": {}, "experience_analysis": {}, "pro_odds": {},
        "asian_handicap_probs": {}, "top_scores": [],
        "engine_version": ENGINE_VERSION,
        "engine_architecture": ENGINE_ARCHITECTURE,
    }


def merge_result_pure_ai(all_ai: Dict[str, Dict[int, Dict[str, Any]]], idx: int, match_obj: Dict[str, Any]) -> Dict[str, Any]:
    per_match = {name: (all_ai.get(name, {}) or {}).get(idx, {}) for name in AI_NAMES}
    final_name, final_r, source = _choose_final_ai(per_match)
    if not final_name:
        return _abstain_prediction()
    return _make_ai_prediction(final_name, final_r, source, per_match, match_obj)

# ============================================================
# 后处理 / 推荐
# ============================================================

def _enforce_consistency(mg: Dict[str, Any]) -> Dict[str, Any]:
    if mg.get("is_abstain") or mg.get("predicted_score") == "弃权":
        mg["predicted_score"] = "弃权"
        mg["predicted_label"] = "弃权"
        mg["result"] = "弃权"
        mg["display_direction"] = "弃权"
        mg["final_direction"] = "abstain"
        return mg
    score = _normalize_score_text(mg.get("predicted_score", ""))
    d = _score_direction(score) or _dir_from_cn(mg.get("result", "")) or "draw"
    mg["predicted_score"] = score
    mg["predicted_label"] = _score_display_label(score, d)
    mg["result"] = _direction_cn(d)
    mg["display_direction"] = _direction_cn(d)
    mg["final_direction"] = d
    mg["is_score_others"] = mg["predicted_label"] in ("胜其他", "平其他", "负其他")
    return mg


def _validate_prediction_consistency(mg: Dict[str, Any]) -> Dict[str, Any]:
    warnings = list(mg.get("validation_warnings", []))
    if mg.get("is_abstain"):
        mg["validation_warnings"] = warnings
        return _enforce_consistency(mg)
    score = _normalize_score_text(mg.get("predicted_score", ""))
    h, a = _parse_score(score)
    if h is None:
        warnings.append("score_unparseable_in_pure_ai")
    else:
        total = h + a
        gr = mg.get("goal_range")
        if isinstance(gr, list):
            gr = tuple(gr)
        if isinstance(gr, tuple) and len(gr) == 2:
            gmin, gmax = _i(gr[0]), _i(gr[1])
            if not (gmin <= total <= gmax):
                warnings.append(f"goal_range_adjusted_for_ai_score:{gr}->{total}")
                mg["goal_range"] = (min(gmin, total), max(gmax, total))
    mg["validation_warnings"] = warnings
    return _enforce_consistency(mg)


def select_top4(preds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for p in preds:
        pr = p.get("prediction", {})
        if pr.get("is_abstain"):
            p["recommend_score"] = -999
            continue
        s = _f(pr.get("confidence", 0), 0)
        s += _f(pr.get("dir_confidence", 0), 0) * 0.2
        if pr.get("risk_level") in ("low", "低"):
            s += 5
        if pr.get("risk_level") in ("high", "高"):
            s -= 8
        p["recommend_score"] = round(s, 2)
    return sorted(preds, key=lambda x: x.get("recommend_score", -999), reverse=True)[:4]


def extract_num(ms: Any) -> int:
    wm = {"一":1000,"二":2000,"三":3000,"四":4000,"五":5000,"六":6000,"日":7000,"天":7000}
    base = next((v for k, v in wm.items() if k in str(ms)), 0)
    nums = re.findall(r"\d+", str(ms))
    return base + int(nums[0]) if nums else 9999

# ============================================================
# 主入口
# ============================================================

def run_predictions(raw: Dict[str, Any], use_ai: bool = True):
    ms = raw.get("matches", []) if isinstance(raw, dict) else []
    ms = [normalize_match(m) for m in ms]

    print("\n" + "=" * 88)
    print(f"  [{ENGINE_VERSION}] PURE RAW-AI 主审 | {len(ms)} 场 | AI失败即弃权 | 无本地兜底")
    print("=" * 88)

    match_analyses: List[Dict[str, Any]] = []
    for i, m in enumerate(ms, 1):
        match_analyses.append({"match": m, "index": i, "external_context": {"enabled": False, "source_quality": "disabled", "items": [], "errors": []}})

    if ENABLE_EXTERNAL_CONTEXT and match_analyses:
        print(f"  [{ENGINE_VERSION} External] 联网情报入口 enabled=True ...")
        try:
            _run_async(enrich_external_context(match_analyses))
        except Exception as e:
            logger.warning(f"external_context 失败: {e}")
    else:
        print(f"  [{ENGINE_VERSION} External] 联网情报入口 enabled=False")

    all_ai: Dict[str, Dict[int, Dict[str, Any]]] = {n: {} for n in AI_NAMES}
    if use_ai and match_analyses:
        print(f"  [{ENGINE_VERSION} AI] 启动 GPT/Grok/Gemini 初审 + Claude 终审")
        try:
            all_ai = _run_async(run_ai_matrix_two_phase(match_analyses))
        except Exception as e:
            logger.error(f"AI矩阵执行失败: {e}")
            all_ai = {n: {} for n in AI_NAMES}
    elif not use_ai:
        print("  [PURE RAW-AI] use_ai=False → 全部弃权，不启用本地兜底")

    res = []
    for i, ma in enumerate(match_analyses, 1):
        m = ma["match"]
        mg = merge_result_pure_ai(all_ai, i, m)
        mg = _validate_prediction_consistency(mg)
        res.append({**m, "prediction": mg})

        if mg.get("is_abstain"):
            print(f"  [{i}] {m.get('home_team')} vs {m.get('away_team')} => 弃权 | PURE模式AI无有效结果")
        else:
            print(
                f"  [{i}] {m.get('home_team')} vs {m.get('away_team')} => "
                f"{mg['result']} ({mg['predicted_score']}={mg['predicted_label']}) | "
                f"AI_CF:{mg['confidence']} | 来源:{mg.get('decision_source')}"
            )

    t4 = select_top4(res)
    t4_ids = set(id(x) for x in t4)
    for r in res:
        r["is_recommended"] = id(r) in t4_ids
    res.sort(key=lambda x: extract_num(x.get("match_num", "")))
    return res, t4


if __name__ == "__main__":
    logger.info(f"{ENGINE_VERSION} 启动")
    print(f"✅ {ENGINE_VERSION} 加载完成")
    print(f"   架构: {ENGINE_ARCHITECTURE}")
    print("   模式: AI成功=AI直出；AI失败=弃权；无本地兜底")

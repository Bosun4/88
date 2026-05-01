# ====================================================================
# 🚀 vMAX 19.4 — RAW-ONLY 四AI审计版
# --------------------------------------------------------------------
# 这版原则:
#   ✅ 只把原始抓包数据喂给 AI
#   ✅ 不调用 predict_match / ensemble / experience / advanced_models
#   ✅ 不生成 OBS/T1-T16
#   ✅ 不跑贝叶斯后验
#   ✅ 不跑 CRS矩阵几何结论
#   ✅ 不跑 TTG主模态结论
#   ✅ 不跑候选比分排序
#   ✅ GPT / Grok / Gemini 三家独立分析
#   ✅ Claude 接收三家结论 + 原始抓包做最终审计
#   ✅ Claude 不按票数裁决，必须重新审计抓包
#   ✅ 程序只做 JSON 校验、字段兼容、方向一致
#   ✅ 默认不做 Phase1 Repair，避免模型调用膨胀
#   ✅ 默认 STRICT_FOUR_MODEL_CALLS=True，每家只尝试一次主通道
# ====================================================================

import json
import os
import re
import time
import asyncio
import aiohttp
import logging
from typing import Dict, List, Any, Tuple, Optional


# ====================================================================
# 日志
# ====================================================================

try:
    import structlog
    logger = structlog.get_logger()
except Exception:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)


# ====================================================================
# 基础配置
# ====================================================================

ENGINE_VERSION = "vMAX 19.4"
ENGINE_ARCHITECTURE = "Raw Packet Only + 3 Analysts + Claude Final Audit"

# 严格控制逻辑调用次数：
# True  = GPT/Grok/Gemini/Claude 每家只请求一次主通道，不做URL重试、不做Repair
# False = 允许备用URL兜底，但HTTP请求次数可能增加
STRICT_FOUR_MODEL_CALLS = True

# 禁止 Phase1 单场补跑，避免 3 + N + 1 的模型调用膨胀
ENABLE_PHASE1_REPAIR = False

# 不使用 LLM 主观比分概率计算正式 EV/Kelly
ENABLE_LLM_VALUE_BET = False

AI_CALL_STATUS = {
    "gpt": "",
    "grok": "",
    "gemini": "",
    "claude": "",
}

VALID_DIRS = {"home", "draw", "away"}

CRS_FULL_MAP = {
    "1-0": "w10",
    "2-0": "w20",
    "2-1": "w21",
    "3-0": "w30",
    "3-1": "w31",
    "3-2": "w32",
    "4-0": "w40",
    "4-1": "w41",
    "4-2": "w42",
    "5-0": "w50",
    "5-1": "w51",
    "5-2": "w52",

    "0-0": "s00",
    "1-1": "s11",
    "2-2": "s22",
    "3-3": "s33",

    "0-1": "l01",
    "0-2": "l02",
    "1-2": "l12",
    "0-3": "l03",
    "1-3": "l13",
    "2-3": "l23",
    "0-4": "l04",
    "1-4": "l14",
    "2-4": "l24",
    "0-5": "l05",
    "1-5": "l15",
    "2-5": "l25",
}

HFTF_MAP = {
    "ss": "主/主",
    "sp": "主/平",
    "sf": "主/负",
    "ps": "平/主",
    "pp": "平/平",
    "pf": "平/负",
    "fs": "负/主",
    "fp": "负/平",
    "ff": "负/负",
}

CUP_KEYWORDS = [
    "杯",
    "淘汰",
    "决赛",
    "半决赛",
    "四分之一",
    "欧冠",
    "欧联",
    "国王杯",
    "足总杯",
    "联赛杯",
    "解放者杯",
    "南球杯",
]

STANDARD_GOAL_ODDS = {
    0: 9.5,
    1: 5.5,
    2: 3.5,
    3: 4.0,
    4: 7.0,
    5: 14.0,
    6: 30.0,
    7: 70.0,
}


# ====================================================================
# 通用工具
# ====================================================================

def _f(v, default=0.0):
    try:
        if v is None:
            return default
        if isinstance(v, str):
            v = v.replace("%", "").replace(",", "").strip()
            if v == "":
                return default
        return float(v)
    except Exception:
        return default


def _i(v, default=0):
    try:
        return int(round(_f(v, default)))
    except Exception:
        return default


def _safe_str(v, max_len=None) -> str:
    s = "" if v is None else str(v)
    s = s.replace("\r", " ").replace("\n", " ").strip()
    if max_len and len(s) > max_len:
        return s[:max_len] + "..."
    return s


def _short_json(obj: Any, max_len: int = 6000) -> str:
    try:
        s = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        s = str(obj)
    if len(s) > max_len:
        return s[:max_len] + "...[TRUNCATED]"
    return s


def _ttg_label(g: int) -> str:
    return "7+球" if int(g) >= 7 else f"{int(g)}球"


def _parse_score(s: str) -> Tuple[Optional[int], Optional[int]]:
    try:
        s_str = str(s).strip().replace(" ", "").replace("：", "-").replace(":", "-")

        if "胜" in s_str and "其他" in s_str:
            return 9, 0
        if "平" in s_str and "其他" in s_str:
            return 9, 9
        if "负" in s_str and "其他" in s_str:
            return 0, 9

        if s_str in [
            "主胜",
            "客胜",
            "平局",
            "home",
            "away",
            "draw",
            "SCORE",
            "SCORE_STRING",
            "SCORE_STRING_FROM_DATA",
            "DATA_DECIDES",
            "数据决定",
        ]:
            return None, None

        p = s_str.split("-")
        if len(p) != 2:
            return None, None

        return int(p[0]), int(p[1])
    except Exception:
        return None, None


def _score_direction(score_str: str) -> Optional[str]:
    h, a = _parse_score(score_str)
    if h is None:
        return None

    s = str(score_str)

    if "胜其他" in s or score_str == "9-0":
        return "home"
    if "平其他" in s or score_str == "9-9":
        return "draw"
    if "负其他" in s or score_str == "0-9":
        return "away"

    if h > a:
        return "home"
    if h < a:
        return "away"
    return "draw"


def _direction_cn(direction: str) -> str:
    return {
        "home": "主胜",
        "draw": "平局",
        "away": "客胜",
    }.get(direction, "平局")


def _score_to_label(score: str) -> Tuple[str, bool]:
    s = str(score).strip().replace("：", "-").replace(":", "-").replace(" ", "")

    if "胜其他" in s or s == "9-0":
        return "胜其他", True
    if "平其他" in s or s == "9-9":
        return "平其他", True
    if "负其他" in s or s == "0-9":
        return "负其他", True

    return s, False


def _goal_range_from_score(score: str) -> str:
    h, a = _parse_score(score)
    if h is None:
        return "未知"

    total = h + a

    if total <= 1:
        return "0-1球"
    if total == 2:
        return "2球"
    if total == 3:
        return "3球"
    if total == 4:
        return "4球"
    if total == 5:
        return "5球"
    return "6+球"


def _normalize_goal_range_for_ui(goal_range: str, score: str = "") -> Tuple[str, str]:
    s = str(goal_range or "").strip().replace(" ", "")

    if not s or s in ["?", "None", "null", "其他", "未知"]:
        s = _goal_range_from_score(score)

    if s in ["0-1球", "0-1", "0~1球", "0到1球"]:
        return "0-1", "0-1球"

    if "6+" in s or "6＋" in s or "六+" in s:
        return "6+", "6+球"

    m = re.search(r"(\d+)", s)
    if m:
        n = int(m.group(1))
        if n >= 6:
            return "6+", "6+球"
        return str(n), f"{n}球"

    return "?", "未知"


def _pct_normalize(h, d, a) -> Tuple[float, float, float]:
    h = _f(h, 33.3)
    d = _f(d, 33.3)
    a = _f(a, 33.4)

    total = h + d + a

    if 0.8 <= total <= 1.2:
        h, d, a = h * 100, d * 100, a * 100
        total = h + d + a

    if total <= 0:
        return 33.3, 33.3, 33.4

    return (
        round(h / total * 100, 1),
        round(d / total * 100, 1),
        round(a / total * 100, 1),
    )


def _fallback_score_for_direction(direction: str) -> str:
    if direction == "home":
        return "1-0"
    if direction == "away":
        return "0-1"
    return "1-1"


def _norm_key(k: str) -> str:
    return str(k).lower().replace(" ", "").replace("_", "").replace("-", "")


def _deep_find_value(obj, aliases: List[str], positive_only=False, default=0, skip_keys: Optional[List[str]] = None):
    alias_set = {_norm_key(a) for a in aliases}
    skip_set = {_norm_key(k) for k in (skip_keys or [])}

    def _ok(v):
        if positive_only:
            return 1.01 < _f(v, 0) < 1000
        return v is not None and str(v).strip() != ""

    def _walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                nk = _norm_key(k)
                if nk in skip_set:
                    continue
                if nk in alias_set and _ok(v):
                    return v
                found = _walk(v)
                if found is not None:
                    return found
        elif isinstance(x, list):
            for item in x:
                found = _walk(item)
                if found is not None:
                    return found
        return None

    found = _walk(obj)
    return found if found is not None else default


def _get_score_odds(match: Dict, score: str, direction: str, is_others: bool) -> float:
    if is_others:
        if direction == "home":
            return _f(match.get("crs_win", 0))
        if direction == "away":
            return _f(match.get("crs_lose", 0))
        return _f(match.get("crs_same", 0))

    key = CRS_FULL_MAP.get(score, "")
    if not key:
        return 0.0

    return _f(match.get(key, 0))


def _extract_score_prob_from_ai(ai_result: Dict, score: str) -> float:
    candidates = []

    for key in ["top5_scores", "top3", "score_probs", "scores"]:
        arr = ai_result.get(key, [])
        if isinstance(arr, list):
            candidates.extend(arr)

    for item in candidates:
        if not isinstance(item, dict):
            continue

        if str(item.get("score", "")).strip() == str(score).strip():
            p = _f(item.get("prob", item.get("prob_pct", 0)))
            if p > 1:
                p /= 100.0
            return max(0.0, min(1.0, p))

    return 0.0


def _calculate_score_ev(score_prob: float, odds: float) -> Tuple[float, float, bool]:
    if odds <= 1.05 or score_prob <= 0:
        return 0.0, 0.0, False

    ev = score_prob * odds - 1.0
    b = odds - 1.0
    q = 1.0 - score_prob

    if b > 0:
        kelly = ((b * score_prob) - q) / b
        kelly_pct = round(max(0.0, kelly * 0.5) * 100, 2)
    else:
        kelly_pct = 0.0

    ev_pct = round(ev * 100, 2)
    is_value = ev_pct > 5

    return ev_pct, kelly_pct, is_value


# ====================================================================
# 环境变量 / API 通道
# ====================================================================

FALLBACK_URLS = [
    None,
    "https://www.api522.pro/v1",
    "https://api522.pro/v1",
    "https://api521.pro/v1",
    "http://69.63.213.33:666/v1",
]

GPT_DEFAULT_URL = "https://ai.newapi.life/v1"
GPT_DEFAULT_KEY = globals().get("GPT_DEFAULT_KEY", "")

GPT_KEY_ALIASES = [
    "GPT_API_KEY",
    "API_KEY",
]

GPT_URL_ALIASES = [
    "GPT_API_URL",
]


def get_clean_env_url(name, default=""):
    v = str(os.environ.get(name, globals().get(name, default))).strip(" \t\n\r\"'")
    match = re.search(r"(https?://[a-zA-Z0-9._:/?#=&%-]+)", v)
    return match.group(1) if match else v


def get_clean_env_key(name):
    return str(os.environ.get(name, globals().get(name, ""))).strip(" \t\n\r\"'")


def get_first_clean_env_key(names: List[str], default="") -> str:
    for name in names:
        v = str(os.environ.get(name, globals().get(name, ""))).strip(" \t\n\r\"'")
        if v:
            return v
    return str(default or "").strip(" \t\n\r\"'")


def get_first_clean_env_url(names: List[str], default="") -> str:
    for name in names:
        v = str(os.environ.get(name, globals().get(name, ""))).strip(" \t\n\r\"'")
        if v:
            match = re.search(r"(https?://[a-zA-Z0-9._:/?#=&%-]+)", v)
            return match.group(1) if match else v

    v = str(default or "").strip(" \t\n\r\"'")
    match = re.search(r"(https?://[a-zA-Z0-9._:/?#=&%-]+)", v)
    return match.group(1) if match else v


def _mask_key(v: str) -> str:
    if not v:
        return "EMPTY"
    s = str(v)
    if len(s) >= 8:
        return s[:4] + "****" + s[-4:]
    return "SET"


def debug_ai_config():
    print("\n[AI配置检查]")
    print(f"GPT key    = {_mask_key(get_first_clean_env_key(GPT_KEY_ALIASES, GPT_DEFAULT_KEY))}")
    print(f"GPT url    = {get_first_clean_env_url(GPT_URL_ALIASES, GPT_DEFAULT_URL)}")
    print(f"GROK key   = {_mask_key(get_clean_env_key('GROK_API_KEY'))}")
    print(f"GROK url   = {get_clean_env_url('GROK_API_URL')}")
    print(f"GEMINI key = {_mask_key(get_clean_env_key('GEMINI_API_KEY'))}")
    print(f"GEMINI url = {get_clean_env_url('GEMINI_API_URL')}")
    print(f"CLAUDE key = {_mask_key(get_clean_env_key('CLAUDE_API_KEY'))}")
    print(f"CLAUDE url = {get_clean_env_url('CLAUDE_API_URL')}")
    print(f"STRICT_FOUR_MODEL_CALLS = {STRICT_FOUR_MODEL_CALLS}")
    print(f"ENABLE_PHASE1_REPAIR    = {ENABLE_PHASE1_REPAIR}")


# ====================================================================
# Match 标准化：只做字段兼容，不做量化结论
# ====================================================================

def normalize_match(raw_m: Dict) -> Dict:
    raw_m = raw_m or {}
    m = dict(raw_m)

    for nested_key in [
        "v2_odds_dict",
        "odds_dict",
        "odds",
        "v2",
        "odds_v2",
        "packet",
        "raw_odds",
        "data",
        "detail",
    ]:
        nested = m.get(nested_key)
        if isinstance(nested, dict):
            m.update(nested)

    m["home_team"] = (
        m.get("home_team")
        or m.get("home")
        or m.get("host")
        or m.get("team_home")
        or m.get("homeName")
        or "Home"
    )
    m["away_team"] = (
        m.get("away_team")
        or m.get("guest")
        or m.get("away")
        or m.get("team_away")
        or m.get("awayName")
        or "Away"
    )

    m["home"] = m.get("home") or m["home_team"]
    m["guest"] = m.get("guest") or m["away_team"]

    odds_skip = [
        "vote",
        "change",
        "points",
        "information",
        "prediction",
        "stats",
        "smart_signals",
        "analysis",
    ]

    sp_home = _deep_find_value(m, [
        "sp_home",
        "win",
        "odds_win",
        "home_win",
        "home_win_odds",
        "odds_home",
        "h_odds",
        "had_h",
        "had_win",
        "spf_sp3",
        "spf_3",
        "sp3",
        "homeOdds",
        "winOdds",
        "胜",
    ], positive_only=True, default=0, skip_keys=odds_skip)

    sp_draw = _deep_find_value(m, [
        "sp_draw",
        "same",
        "draw",
        "odds_draw",
        "draw_odds",
        "had_d",
        "had_draw",
        "spf_sp1",
        "spf_1",
        "sp1",
        "drawOdds",
        "sameOdds",
        "平",
    ], positive_only=True, default=0, skip_keys=odds_skip)

    sp_away = _deep_find_value(m, [
        "sp_away",
        "lose",
        "away_win",
        "odds_away",
        "away_win_odds",
        "guest_win",
        "guest_odds",
        "had_a",
        "had_lose",
        "spf_sp0",
        "spf_0",
        "sp0",
        "awayOdds",
        "loseOdds",
        "负",
    ], positive_only=True, default=0, skip_keys=odds_skip)

    m["sp_home"] = sp_home
    m["sp_draw"] = sp_draw
    m["sp_away"] = sp_away
    m["win"] = sp_home
    m["same"] = sp_draw
    m["lose"] = sp_away

    m["give_ball"] = (
        m.get("give_ball")
        if m.get("give_ball") not in [None, ""]
        else m.get("handicap", m.get("rq", m.get("let_ball", "0")))
    )

    change = m.get("change", {})
    if not isinstance(change, dict):
        change = {}

    change["win"] = _deep_find_value(m, [
        "change_win",
        "cw",
        "home_change",
        "win_change",
        "odds_change_home",
        "change_sp3",
        "sp3_change",
    ], positive_only=False, default=change.get("win", 0), skip_keys=["vote", "points", "information", "prediction"])

    change["same"] = _deep_find_value(m, [
        "change_same",
        "cs",
        "draw_change",
        "same_change",
        "odds_change_draw",
        "change_sp1",
        "sp1_change",
    ], positive_only=False, default=change.get("same", 0), skip_keys=["vote", "points", "information", "prediction"])

    change["lose"] = _deep_find_value(m, [
        "change_lose",
        "cl",
        "away_change",
        "lose_change",
        "odds_change_away",
        "change_sp0",
        "sp0_change",
    ], positive_only=False, default=change.get("lose", 0), skip_keys=["vote", "points", "information", "prediction"])

    m["change"] = change

    vote = m.get("vote", {})
    if not isinstance(vote, dict):
        vote = {}

    vote["win"] = _deep_find_value(m, [
        "vote_win",
        "hot_home",
        "public_home",
        "win_vote",
        "home_vote",
        "vote_sp3",
        "support_home",
    ], positive_only=False, default=vote.get("win", 0), skip_keys=["change", "points", "information", "prediction"])

    vote["same"] = _deep_find_value(m, [
        "vote_same",
        "hot_draw",
        "public_draw",
        "draw_vote",
        "same_vote",
        "vote_sp1",
        "support_draw",
    ], positive_only=False, default=vote.get("same", 0), skip_keys=["change", "points", "information", "prediction"])

    vote["lose"] = _deep_find_value(m, [
        "vote_lose",
        "hot_away",
        "public_away",
        "away_vote",
        "lose_vote",
        "vote_sp0",
        "support_away",
    ], positive_only=False, default=vote.get("lose", 0), skip_keys=["change", "points", "information", "prediction"])

    m["vote"] = vote

    return m


def _raw_smart_signals_from_match(m: Dict) -> List[str]:
    """
    Raw-Only 版只读取抓包里自带的 smart_signals / signals。
    不调用 ensemble 生成任何新信号。
    """
    arr = (
        m.get("smart_signals")
        or m.get("signals")
        or m.get("raw_smart_signals")
        or []
    )

    if isinstance(arr, str):
        return [arr]
    if isinstance(arr, list):
        return [str(x) for x in arr]
    return []


# ====================================================================
# 抓包格式化：只展示原始数据，不输出系统结论
# ====================================================================

def format_raw_match_block(idx: int, match: Dict) -> str:
    home = match.get("home_team", match.get("home", "Home"))
    away = match.get("away_team", match.get("guest", "Away"))
    league = match.get("league", match.get("cup", ""))
    is_cup = any(kw in str(league) for kw in CUP_KEYWORDS)

    sp_h = _f(match.get("sp_home", match.get("win", 0)))
    sp_d = _f(match.get("sp_draw", match.get("same", 0)))
    sp_a = _f(match.get("sp_away", match.get("lose", 0)))

    block = "\n════════════════════════════════════\n"
    block += f"第 {idx} 场: {home} vs {away}\n"
    block += "════════════════════════════════════\n"
    block += f"联赛/赛事: {league}{' [杯赛/淘汰赛属性仅作原始标签]' if is_cup else ''}\n"
    block += f"比赛编号: {match.get('match_num', match.get('id', idx))}\n"
    block += f"开赛时间: {match.get('match_time', match.get('time', match.get('date', 'N/A')))}\n"

    block += "\n【1. 胜平负欧赔原始数据】\n"
    block += f"即时欧赔: 主胜 {sp_h:.2f} / 平局 {sp_d:.2f} / 客胜 {sp_a:.2f}\n"

    init_h = _f(match.get("init_home", match.get("initial_home", match.get("sp_home_init", 0))))
    init_d = _f(match.get("init_draw", match.get("initial_draw", match.get("sp_draw_init", 0))))
    init_a = _f(match.get("init_away", match.get("initial_away", match.get("sp_away_init", 0))))

    if init_h > 1 or init_d > 1 or init_a > 1:
        block += f"初始欧赔: 主胜 {init_h:.2f} / 平局 {init_d:.2f} / 客胜 {init_a:.2f}\n"

    change = match.get("change", {}) or {}
    cw = _f(change.get("win", 0))
    cs = _f(change.get("same", 0))
    cl = _f(change.get("lose", 0))

    if cw or cs or cl:
        block += f"赔率变化: 主胜 {cw:+.2f} / 平局 {cs:+.2f} / 客胜 {cl:+.2f}。负数=赔率下调。\n"

    block += "\n【2. 让球/盘口原始数据】\n"
    block += f"give_ball/handicap/rq: {match.get('give_ball', match.get('handicap', match.get('rq', '0')))}\n"

    rq_fields = [
        ("rq_win", "让胜"),
        ("rq_draw", "让平"),
        ("rq_lose", "让负"),
        ("let_win", "让胜"),
        ("let_draw", "让平"),
        ("let_lose", "让负"),
    ]
    rq_lines = []
    for k, label in rq_fields:
        v = _f(match.get(k, 0))
        if v > 1:
            rq_lines.append(f"{label}({k})={v:.2f}")
    if rq_lines:
        block += "让球胜平负: " + " | ".join(rq_lines) + "\n"

    block += "\n【3. 总进球数赔率 a0~a7 原始数据】\n"
    ttg_lines = []
    for g in range(8):
        v = _f(match.get(f"a{g}", 0))
        if v > 1:
            ttg_lines.append(f"{_ttg_label(g)}={v:.2f}")
        else:
            ttg_lines.append(f"{_ttg_label(g)}=N/A")
    block += " | ".join(ttg_lines) + "\n"

    block += "\n【4. 精确比分 CRS 原始数据】\n"

    crs_h = []
    crs_d = []
    crs_a = []

    for sc, key in CRS_FULL_MAP.items():
        v = _f(match.get(key, 0))
        if v <= 1:
            continue

        text = f"{sc}={v:.1f}"
        direction = _score_direction(sc)

        if direction == "home":
            crs_h.append(text)
        elif direction == "draw":
            crs_d.append(text)
        elif direction == "away":
            crs_a.append(text)

    block += "主胜系: " + (" | ".join(crs_h) if crs_h else "N/A") + "\n"
    block += "平局系: " + (" | ".join(crs_d) if crs_d else "N/A") + "\n"
    block += "客胜系: " + (" | ".join(crs_a) if crs_a else "N/A") + "\n"

    others = []
    for k, label in [
        ("crs_win", "胜其他"),
        ("crs_same", "平其他"),
        ("crs_lose", "负其他"),
    ]:
        v = _f(match.get(k, 0))
        if v > 1:
            others.append(f"{label}={v:.1f}")

    if others:
        block += "其他比分: " + " | ".join(others) + "\n"

    crs_items = []
    for sc, key in CRS_FULL_MAP.items():
        v = _f(match.get(key, 0))
        if v > 1:
            crs_items.append((sc, v))

    if crs_items:
        low = sorted(crs_items, key=lambda x: x[1])[:10]
        block += "CRS赔率从低到高Top10，仅为赔率原始排序，不是系统结论: "
        block += " | ".join([f"{sc}={v:.1f}" for sc, v in low]) + "\n"

    hf_lines = []
    for k, label in HFTF_MAP.items():
        v = _f(match.get(k, 0))
        if v > 1:
            hf_lines.append(f"{label}={v:.2f}")

    if hf_lines:
        block += "\n【5. 半全场赔率原始数据】\n"
        block += " | ".join(hf_lines) + "\n"

    vote = match.get("vote", {}) or {}
    if vote and any(str(vote.get(k, "")).strip() not in ["", "0", "0.0"] for k in ["win", "same", "lose"]):
        block += "\n【6. 散户/投注热度原始数据】\n"
        block += f"主胜 {vote.get('win', '?')}% / 平局 {vote.get('same', '?')}% / 客胜 {vote.get('lose', '?')}%\n"

    points = match.get("points", {}) or {}
    if isinstance(points, dict):
        block += "\n【7. 基本面/情报原文】\n"

        h_text = _safe_str(points.get("home_strength", ""), 800)
        a_text = _safe_str(points.get("guest_strength", ""), 800)
        m_text = _safe_str(points.get("match_points", ""), 800)
        h2h_text = _safe_str(points.get("history", points.get("h2h", "")), 600)

        if h_text:
            block += f"主队: {h_text}\n"
        if a_text:
            block += f"客队: {a_text}\n"
        if h2h_text:
            block += f"交锋: {h2h_text}\n"
        if m_text:
            block += f"赛事要点: {m_text}\n"

    info = match.get("information", {}) or {}
    if isinstance(info, dict):
        info_lines = []

        for k, label in [
            ("home_injury", "主伤停"),
            ("guest_injury", "客伤停"),
            ("home_bad_news", "主利空"),
            ("guest_bad_news", "客利空"),
            ("home_good_news", "主利好"),
            ("guest_good_news", "客利好"),
        ]:
            v = info.get(k)
            if v:
                info_lines.append(f"{label}: {_safe_str(v, 500)}")

        if info_lines:
            block += "\n【8. 伤停/异动消息原文】\n"
            block += "\n".join(info_lines) + "\n"

    raw_sigs = _raw_smart_signals_from_match(match)
    if raw_sigs:
        block += "\n【9. 抓包自带资金/智能信号原文】\n"
        block += "注意：以下只来自原始抓包字段，不是本程序生成。\n"
        for s in raw_sigs[:20]:
            block += f"- {_safe_str(s, 300)}\n"

    block += "\n【10. 原始抓包JSON摘要】\n"
    block += _short_json(match, max_len=8000) + "\n"

    return block


# ====================================================================
# Prompt 构建：Phase 1 三家 RAW 独立分析
# ====================================================================

PHASE1_ROLES = {
    "gpt": {
        "name": "赔率结构分析师",
        "focus": (
            "只看原始赔率结构：胜平负、让球、总进球、CRS、半全场。"
            "你需要自己从赔率中判断方向、进球数和比分路径，禁止依赖任何系统结论。"
        ),
        "temperature": 0.18,
    },
    "grok": {
        "name": "资金/变盘分析师",
        "focus": (
            "只看原始变盘、降水、投注热度、抓包自带Sharp/Steam/资金字样。"
            "你需要判断资金信号是真支持还是诱导，禁止把系统标签当结论。"
        ),
        "temperature": 0.26,
    },
    "gemini": {
        "name": "基本面/场景分析师",
        "focus": (
            "只看原始基本面、赛程、杯赛属性、主客场状态、伤停和交锋。"
            "你需要判断赔率结论是否被基本面支持或反对。"
        ),
        "temperature": 0.20,
    },
}


def build_phase1_prompt(match_blocks: List[str], role_key: str) -> str:
    role = PHASE1_ROLES[role_key]

    p = ""
    p += "<context>\n"
    p += "你是竞彩足球 RAW 抓包审计团队中的独立分析师。\n"
    p += "下面只给你原始抓包数据。没有系统陷阱结论，没有贝叶斯后验，没有候选比分排序。\n"
    p += "你必须基于原始赔率和原始情报自主分析。\n"
    p += "</context>\n\n"

    p += "<your_role>\n"
    p += f"角色: {role['name']}\n"
    p += f"专长: {role['focus']}\n"
    p += "</your_role>\n\n"

    p += "<raw_only_rules>\n"
    p += "1. 只能使用 match_data 中的原始抓包字段进行判断。\n"
    p += "2. 禁止声称系统已经判断某方向、某陷阱、某主模态；因为本版本没有系统判断。\n"
    p += "3. 可以自行计算赔率隐含概率、CRS低赔路径、总进球路径，但必须说明这是你的人工计算。\n"
    p += "4. 禁止因为总进球接近3球就默认输出某固定比分。\n"
    p += "5. 如果选择任何3球比分，必须比较同场的1球、2球、4球路径。\n"
    p += "6. 如果选择主胜，必须比较同场的1-0、2-0、2-1、3-0、3-1和1-1保护。\n"
    p += "7. 如果选择客胜，必须比较同场的0-1、0-2、1-2、0-3、1-3和1-1保护。\n"
    p += "8. 如果选择平局，必须比较0-0、1-1、2-2以及主/客小胜路径。\n"
    p += "9. 如果外部市场不可用，不得写外部市场支持或确认。\n"
    p += "10. 若你具备实时联网能力，可用于核对公开赔率/伤停；若不可用，external_check.available=false。\n"
    p += "11. 输出必须是严格 JSON 数组，不要 markdown，不要代码块，不要前缀后缀。\n"
    p += "</raw_only_rules>\n\n"

    p += "<anti_template_rules>\n"
    p += "禁止模板化输出。\n"
    p += "任何场次都不能因为常见比分模式而默认选择固定比分。\n"
    p += "如果你的 predicted_score 是某个热门模板比分，必须在 reason 中说明它相对于至少4个替代比分的优势。\n"
    p += "</anti_template_rules>\n\n"

    p += "<match_data>\n"
    for block in match_blocks:
        p += block
    p += "\n</match_data>\n\n"

    p += "<output_schema>\n"
    p += "严格输出 JSON 数组。字段说明如下，字段说明里的占位词禁止照抄：\n"
    p += "[\n"
    p += "  {\n"
    p += "    \"match\": 整数场次编号,\n"
    p += "    \"predicted_score\": \"根据本场数据得出的具体比分字符串\",\n"
    p += "    \"predicted_direction\": \"home/draw/away\",\n"
    p += "    \"goal_range\": \"0-1球/2球/3球/4球/5球/6+球\",\n"
    p += "    \"home_win_pct\": 数字,\n"
    p += "    \"draw_pct\": 数字,\n"
    p += "    \"away_win_pct\": 数字,\n"
    p += "    \"confidence\": 数字,\n"
    p += "    \"top5_scores\": [\n"
    p += "      {\"score\":\"具体比分\", \"prob\":数字},\n"
    p += "      {\"score\":\"具体比分\", \"prob\":数字}\n"
    p += "    ],\n"
    p += "    \"main_market_path\": \"本场最主要的赔率路径\",\n"
    p += "    \"sharp_or_steam_view\": \"若原始抓包有资金信号，判断其有效性；无则写无明确资金信号\",\n"
    p += "    \"crs_view\": \"你自己从CRS原始赔率看到的比分路径\",\n"
    p += "    \"ttg_view\": \"你自己从总进球赔率看到的进球路径\",\n"
    p += "    \"handicap_view\": \"你自己从让球胜平负/盘口看到的路径\",\n"
    p += "    \"why_not_1_0\": \"为什么不是或为什么防1-0\",\n"
    p += "    \"why_not_2_0\": \"为什么不是或为什么防2-0\",\n"
    p += "    \"why_not_1_1\": \"为什么不是或为什么防1-1\",\n"
    p += "    \"why_not_3_1\": \"为什么不是或为什么防3-1\",\n"
    p += "    \"why_not_1_2\": \"为什么不是或为什么防1-2\",\n"
    p += "    \"external_check\": {\"available\": 布尔值, \"summary\": \"联网核对情况或不可用说明\"},\n"
    p += "    \"doubts\": [\"反对自己结论的证据\"],\n"
    p += "    \"reason\": \"中文完整推理，必须合法JSON字符串\"\n"
    p += "  }\n"
    p += "]\n"
    p += "</output_schema>\n"

    return p


# ====================================================================
# Prompt 构建：Phase 2 Claude RAW 终审
# ====================================================================

def _phase1_summary_line(r: Dict) -> str:
    if not isinstance(r, dict) or not r:
        return "无数据"

    top5 = r.get("top5_scores", r.get("top3", []))
    top5_str = ""

    if isinstance(top5, list):
        tmp = []
        for t in top5[:5]:
            if isinstance(t, dict):
                tmp.append(f"{t.get('score', '?')}({t.get('prob', '?')}%)")
        top5_str = ", ".join(tmp)

    return (
        f"方向={r.get('predicted_direction', '?')} | "
        f"比分={r.get('predicted_score', '?')} | "
        f"进球区间={r.get('goal_range_label', r.get('goal_range', '?'))} | "
        f"信心={r.get('confidence', '?')} | "
        f"top5=[{top5_str}]"
    )


def build_phase2_prompt(match_blocks: List[str], phase1_results: Dict[str, Dict[int, Dict]]) -> str:
    num_matches = len(match_blocks)

    p = ""
    p += "<context>\n"
    p += "你是 Claude RAW 终审审计者。\n"
    p += "GPT / Grok / Gemini 已分别基于原始抓包做了独立分析。\n"
    p += "你现在拿到三家完整结论 + 原始抓包，需要重新审计并输出最终预测。\n"
    p += "</context>\n\n"

    p += "<critical_rules>\n"
    p += "1. 你不是投票裁判，不能简单按三家多数派裁决。\n"
    p += "2. 三家一致也不代表正确，必须重新审计原始抓包。\n"
    p += "3. 如果三家都给同一个热门模板比分，你必须重新比较CRS、总进球、让球和半全场路径。\n"
    p += "4. 你不能引用任何系统陷阱结论，因为本版本没有系统陷阱结论。\n"
    p += "5. 你可以自行计算和解释赔率结构，但必须基于原始数据。\n"
    p += "6. 选择最终比分前，必须比较至少5个替代比分路径。\n"
    p += "7. predicted_score 方向必须与 predicted_direction 一致。\n"
    p += "8. top5_scores[0] 必须等于 predicted_score。\n"
    p += "9. 如果 external_check.available=false，不得写外部市场确认。\n"
    p += "10. 输出必须是严格 JSON 数组，不要 markdown，不要代码块，不要前缀后缀。\n"
    p += "</critical_rules>\n\n"

    p += "<audit_priority>\n"
    p += "证据优先级建议：\n"
    p += "A. CRS精确比分赔率路径与总进球赔率是否共振。\n"
    p += "B. 让球胜平负是否支持穿盘、赢球不穿盘、或冷门保护。\n"
    p += "C. 胜平负欧赔和赔率变动是否支持方向。\n"
    p += "D. 原始抓包自带Sharp/Steam/资金信号是否可信。\n"
    p += "E. 半全场是否验证比赛脚本。\n"
    p += "F. 基本面、杯赛、赛程、伤停是否支持或反对赔率路径。\n"
    p += "</audit_priority>\n\n"

    p += "<three_analysts_results>\n"

    for i in range(1, num_matches + 1):
        p += f"\n════════ 第 {i} 场三家结论 ════════\n"

        for ai_name in ["gpt", "grok", "gemini"]:
            r = phase1_results.get(ai_name, {}).get(i, {})
            p += f"\n【{ai_name.upper()}】{_phase1_summary_line(r)}\n"

            if not r:
                continue

            for key in [
                "main_market_path",
                "sharp_or_steam_view",
                "crs_view",
                "ttg_view",
                "handicap_view",
                "why_not_1_0",
                "why_not_2_0",
                "why_not_1_1",
                "why_not_3_1",
                "why_not_1_2",
            ]:
                v = _safe_str(r.get(key, ""), 800)
                if v:
                    p += f"{key}: {v}\n"

            doubts = r.get("doubts", [])
            if doubts:
                p += "doubts: " + " | ".join(str(x) for x in doubts[:8]) + "\n"

            reason = _safe_str(r.get("reason", ""), 2000)
            if reason:
                p += f"reason: {reason}\n"

    p += "\n</three_analysts_results>\n\n"

    p += "<raw_match_data>\n"
    for block in match_blocks:
        p += block
    p += "\n</raw_match_data>\n\n"

    p += "<output_schema>\n"
    p += "严格输出 JSON 数组。字段说明如下，字段说明里的占位词禁止照抄：\n"
    p += "[\n"
    p += "  {\n"
    p += "    \"match\": 整数场次编号,\n"
    p += "    \"predicted_score\": \"最终具体比分字符串\",\n"
    p += "    \"predicted_direction\": \"home/draw/away\",\n"
    p += "    \"goal_range\": \"0-1球/2球/3球/4球/5球/6+球\",\n"
    p += "    \"home_win_pct\": 数字,\n"
    p += "    \"draw_pct\": 数字,\n"
    p += "    \"away_win_pct\": 数字,\n"
    p += "    \"confidence\": 数字,\n"
    p += "    \"agreement_pattern\": \"三家分歧或共识模式\",\n"
    p += "    \"analysis_coverage\": {\"gpt\": 布尔值, \"grok\": 布尔值, \"gemini\": 布尔值, \"valid_count\": 数字},\n"
    p += "    \"adopted_analysts\": [\"采纳的分析师\"],\n"
    p += "    \"rejected_analysts\": [\"否决的分析师\"],\n"
    p += "    \"top5_scores\": [\n"
    p += "      {\"score\":\"具体比分\", \"prob\":数字},\n"
    p += "      {\"score\":\"具体比分\", \"prob\":数字}\n"
    p += "    ],\n"
    p += "    \"main_market_path\": \"最终采用的主路径\",\n"
    p += "    \"crs_view\": \"终审CRS判断\",\n"
    p += "    \"ttg_view\": \"终审总进球判断\",\n"
    p += "    \"handicap_view\": \"终审盘口判断\",\n"
    p += "    \"why_not_1_0\": \"为什么不是或为什么防1-0\",\n"
    p += "    \"why_not_2_0\": \"为什么不是或为什么防2-0\",\n"
    p += "    \"why_not_1_1\": \"为什么不是或为什么防1-1\",\n"
    p += "    \"why_not_3_1\": \"为什么不是或为什么防3-1\",\n"
    p += "    \"why_not_1_2\": \"为什么不是或为什么防1-2\",\n"
    p += "    \"external_check\": {\"available\": 布尔值, \"summary\": \"联网核对情况或不可用说明\"},\n"
    p += "    \"doubts\": [\"反对最终结论的证据\"],\n"
    p += "    \"audit_result\": \"一句话终审结论\",\n"
    p += "    \"arbitration_reason\": \"中文完整终审理由，必须合法JSON字符串\"\n"
    p += "  }\n"
    p += "]\n"
    p += "</output_schema>\n"

    return p


# ====================================================================
# AI 调用配置
# ====================================================================

PHASE1_CONFIGS = [
    {
        "ai_name": "gpt",
        "url_env": "GPT_API_URL",
        "key_env": "GPT_API_KEY",
        "models": [
            "gpt-5.5",
        ],
        "role_key": "gpt",
    },
    {
        "ai_name": "grok",
        "url_env": "GROK_API_URL",
        "key_env": "GROK_API_KEY",
        "models": [
            "熊猫-A-5-grok-4.2-fast-200w上下文",
        ],
        "role_key": "grok",
    },
    {
        "ai_name": "gemini",
        "url_env": "GEMINI_API_URL",
        "key_env": "GEMINI_API_KEY",
        "models": [
            "熊猫特供-按量-SSS-gemini-3.1-pro-preview-thinking",
        ],
        "role_key": "gemini",
    },
]

CLAUDE_CONFIG = {
    "ai_name": "claude",
    "url_env": "CLAUDE_API_URL",
    "key_env": "CLAUDE_API_KEY",
    "models": [
        "熊猫-69-满血openrouter-claude-opus-4.7-上下文1000k",
    ],
    "temperature": 0.22,
}


def _build_urls_for_ai(ai_name: str, url_env: str) -> List[str]:
    if ai_name == "gpt":
        primary_url = get_first_clean_env_url(GPT_URL_ALIASES, GPT_DEFAULT_URL)
        urls = []
        if primary_url:
            urls.append(primary_url)
        if not STRICT_FOUR_MODEL_CALLS and GPT_DEFAULT_URL and GPT_DEFAULT_URL not in urls:
            urls.append(GPT_DEFAULT_URL)
        return urls[:1] if STRICT_FOUR_MODEL_CALLS else urls

    primary_url = get_clean_env_url(url_env)
    urls = [primary_url] if primary_url else []

    if not STRICT_FOUR_MODEL_CALLS:
        backup = [u for u in FALLBACK_URLS if u and u != primary_url][:1]
        urls += backup

    return urls[:1] if STRICT_FOUR_MODEL_CALLS else urls


# ====================================================================
# AI 调用层：不限制输出字数，不传 max_tokens
# ====================================================================

async def async_call_ai_batch(
    session,
    prompt: str,
    url_env: str,
    key_env: str,
    models_list: List[str],
    num_matches: int,
    ai_name: str,
    sys_prompt: str,
    temperature: float,
    phase: str
) -> Tuple[str, Dict[int, Dict], str]:
    if ai_name == "gpt":
        key = get_first_clean_env_key(GPT_KEY_ALIASES, GPT_DEFAULT_KEY)
    else:
        key = get_clean_env_key(key_env)

    if not key:
        status = f"no_key:{key_env}"
        print(f"  [跳过] {ai_name.upper()} 无可用 KEY: {key_env}")
        return ai_name, {}, status

    urls = _build_urls_for_ai(ai_name, url_env)

    if not urls:
        print(f"  [跳过] {ai_name.upper()} 无可用 URL: {url_env}")
        return ai_name, {}, "no_url"

    CONNECT_TIMEOUT = 20
    READ_TIMEOUT_MAP = {
        "claude": 1800,
        "grok": 900,
        "gpt": 1200,
        "gemini": 1800,
    }
    READ_TIMEOUT = READ_TIMEOUT_MAP.get(ai_name, 900)

    model_attempts = models_list[:1] if STRICT_FOUR_MODEL_CALLS else models_list

    for mn in model_attempts:
        connected = False

        for base_url in urls:
            if not base_url:
                continue

            is_gem = "generateContent" in base_url
            url = base_url.rstrip("/")

            if not is_gem and "chat/completions" not in url:
                url += "/chat/completions"

            headers = {"Content-Type": "application/json"}

            if is_gem:
                headers["x-goog-api-key"] = key
                payload = {
                    "contents": [
                        {
                            "role": "user",
                            "parts": [{"text": prompt}],
                        }
                    ],
                    "generationConfig": {
                        "temperature": temperature,
                    },
                    "systemInstruction": {
                        "parts": [{"text": sys_prompt}],
                    },
                }
            else:
                headers["Authorization"] = f"Bearer {key}"
                payload = {
                    "model": mn,
                    "messages": [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": temperature,
                }

            gw = url.split("/v1")[0][:60]
            print(f"  [🔌] {ai_name.upper()} | {mn[:42]} @ {gw}")

            t0 = time.time()

            try:
                timeout = aiohttp.ClientTimeout(
                    total=None,
                    connect=CONNECT_TIMEOUT,
                    sock_connect=CONNECT_TIMEOUT,
                    sock_read=READ_TIMEOUT,
                )

                async with session.post(url, headers=headers, json=payload, timeout=timeout) as r:
                    elapsed = round(time.time() - t0, 1)

                    if r.status != 200:
                        text = await r.text()
                        print(f"    ⚠️ HTTP {r.status} | {elapsed}s | {text[:300]}")
                        if STRICT_FOUR_MODEL_CALLS:
                            return ai_name, {}, f"http_{r.status}"
                        continue

                    connected = True
                    print(f"    ✅ 已连上 {elapsed}s | 等待完整数据...")

                    try:
                        data = await r.json(content_type=None)
                    except Exception as e:
                        print(f"    ⚠️ JSON响应读取失败: {str(e)[:120]}")
                        return ai_name, {}, "response_json_failed"

                    elapsed = round(time.time() - t0, 1)
                    raw_text = _extract_response_text(data, is_gem)

                    if not raw_text or len(raw_text) < 10:
                        print("    ⚠️ 空响应")
                        return ai_name, {}, "empty_response"

                    results = _parse_ai_json(raw_text, num_matches, phase=phase)

                    if results:
                        print(f"    ✅ {ai_name.upper()} 完成: {len(results)}/{num_matches} | {elapsed}s")
                        return ai_name, results, mn

                    print("    ⚠️ JSON解析0条")
                    return ai_name, {}, "parse_zero"

            except aiohttp.ClientConnectorError:
                if STRICT_FOUR_MODEL_CALLS:
                    return ai_name, {}, "connect_error"
                continue
            except asyncio.TimeoutError:
                if not connected:
                    if STRICT_FOUR_MODEL_CALLS:
                        return ai_name, {}, "connect_timeout"
                    continue
                else:
                    print(f"    ⏱️ {ai_name.upper()} 读取超时: {READ_TIMEOUT}s")
                    return ai_name, {}, "read_timeout"
            except Exception as e:
                print(f"    ⚠️ 调用异常: {str(e)[:160]}")
                return ai_name, {}, "error"

            await asyncio.sleep(0.2)

    return ai_name, {}, "all_failed"


def _extract_response_text(data, is_gem=False) -> str:
    raw_text = ""

    try:
        if is_gem:
            candidates = data.get("candidates", [])
            if candidates:
                content = candidates[0].get("content", {})
                parts = content.get("parts", [])
                if parts and isinstance(parts[0], dict):
                    raw_text = str(parts[0].get("text", "")).strip()
                    return raw_text

        if data.get("choices"):
            msg = data["choices"][0].get("message", {})

            if isinstance(msg, dict):
                content_val = msg.get("content", "")

                if isinstance(content_val, str) and content_val.strip():
                    raw_text = content_val.strip()

                elif isinstance(content_val, list):
                    best = ""
                    for item in content_val:
                        if isinstance(item, dict):
                            t = item.get("text", item.get("content", ""))
                            if isinstance(t, str) and len(t) > len(best):
                                best = t.strip()
                    raw_text = best

                if not raw_text:
                    for field in [
                        "text",
                        "answer",
                        "response",
                        "output_text",
                        "final_answer",
                        "output",
                        "result",
                        "completion",
                        "message_content",
                        "assistant_content",
                        "model_response",
                    ]:
                        v = msg.get(field, "")
                        if isinstance(v, str) and v.strip():
                            raw_text = v.strip()
                            break

                if not raw_text:
                    skip = (
                        "reasoning_content",
                        "thinking",
                        "reasoning",
                        "thoughts",
                        "chain_of_thought",
                        "cot",
                    )
                    best = ""
                    for k, v in msg.items():
                        if k in skip:
                            continue
                        if isinstance(v, str) and "[" in v and "match" in v:
                            if len(v) > len(best):
                                best = v.strip()
                    if best:
                        raw_text = best

        if not raw_text and isinstance(data.get("output"), list):
            best = ""
            for out_item in data["output"]:
                if not isinstance(out_item, dict):
                    continue
                for ct in out_item.get("content", []):
                    if isinstance(ct, dict):
                        t = ct.get("text", "")
                        if isinstance(t, str) and len(t) > len(best):
                            best = t.strip()
            raw_text = best

        if not raw_text:
            full_str = json.dumps(data, ensure_ascii=False)
            m_match = re.search(r'\[\s*\{\s*\\?"match\\?"', full_str)
            if m_match:
                start_pos = m_match.start()
                depth = 0
                in_str = False
                escape = False
                end_pos = start_pos

                for ci in range(start_pos, min(start_pos + 800000, len(full_str))):
                    ch = full_str[ci]

                    if escape:
                        escape = False
                        continue

                    if ch == "\\":
                        escape = True
                        continue

                    if ch == '"':
                        in_str = not in_str
                        continue

                    if in_str:
                        continue

                    if ch == "[":
                        depth += 1
                    elif ch == "]":
                        depth -= 1

                    if depth == 0:
                        end_pos = ci + 1
                        break

                if end_pos > start_pos:
                    extracted = full_str[start_pos:end_pos]
                    if '\\"' in extracted:
                        try:
                            extracted = json.loads('"' + extracted + '"')
                        except Exception:
                            extracted = extracted.replace('\\"', '"')
                    raw_text = extracted

    except Exception as ex:
        print(f"    ⚠️ 响应提取异常: {str(ex)[:100]}")

    return raw_text


# ====================================================================
# JSON 解析与校验
# ====================================================================

def _strip_thinking_blocks(text: str) -> str:
    clean = str(text or "")
    clean = re.sub(r"<think(?:ing)?>.*?</think(?:ing)?>", "", clean, flags=re.DOTALL | re.IGNORECASE)
    clean = re.sub(r"```(?:json|JSON|python|txt)?", "", clean)
    clean = clean.replace("```", "")
    return clean.strip()


def _extract_json_array(clean: str) -> str:
    m_re = re.search(r'\[\s*\{\s*"match"', clean)
    if not m_re:
        m_re = re.search(r"\[\s*\{\s*'match'", clean)

    if m_re:
        start_idx = m_re.start()
        depth = 0
        in_str = False
        escape = False
        end_idx = start_idx

        for i in range(start_idx, len(clean)):
            ch = clean[i]

            if escape:
                escape = False
                continue

            if ch == "\\":
                escape = True
                continue

            if ch == '"':
                in_str = not in_str
                continue

            if in_str:
                continue

            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    end_idx = i + 1
                    break

        if end_idx > start_idx:
            return clean[start_idx:end_idx]

    start = clean.find("[")
    end = clean.rfind("]") + 1
    if start != -1 and end > start:
        return clean[start:end]

    return ""


def _parse_ai_json(raw_text: str, num_matches: int, phase: str) -> Dict[int, Dict]:
    clean = _strip_thinking_blocks(raw_text)
    json_str = _extract_json_array(clean)

    results = {}

    if not json_str:
        return results

    arr = []

    try:
        arr = json.loads(json_str)
    except json.JSONDecodeError:
        try:
            fixed = json_str.replace("'", '"')
            arr = json.loads(fixed)
        except Exception:
            try:
                last_brace = json_str.rfind("}")
                if last_brace != -1:
                    arr = json.loads(json_str[:last_brace + 1] + "]")
            except Exception:
                arr = []

    if isinstance(arr, dict):
        arr = [arr]

    if not isinstance(arr, list):
        return results

    for item in arr:
        if not isinstance(item, dict):
            continue

        valid, fixed, errors = validate_ai_item(item, phase=phase)

        if not fixed.get("match"):
            continue

        try:
            mid = int(fixed["match"])
        except Exception:
            continue

        fixed["ai_validation_errors"] = errors

        if 1 <= mid <= max(num_matches, 1):
            results[mid] = fixed

    return results


def validate_ai_item(item: Dict, phase: str) -> Tuple[bool, Dict, List[str]]:
    errors = []
    out = dict(item or {})

    try:
        out["match"] = int(out.get("match"))
    except Exception:
        errors.append("match_id_invalid")
        out["match"] = None
        return False, out, errors

    score = str(out.get("predicted_score", "")).strip().replace("：", "-").replace(":", "-").replace(" ", "")

    pred_dir_raw = str(out.get("predicted_direction", "")).strip().lower()
    if pred_dir_raw not in VALID_DIRS:
        pred_dir_raw = ""

    score_label, is_others = _score_to_label(score)
    h, a = _parse_score(score_label)

    if h is None and score_label not in ["胜其他", "平其他", "负其他"]:
        errors.append("score_invalid")
        if pred_dir_raw in VALID_DIRS:
            score_label = _fallback_score_for_direction(pred_dir_raw)
        else:
            score_label = "1-1"

    expected_dir = _score_direction(score_label)

    if expected_dir:
        out["predicted_direction"] = expected_dir
    else:
        pred_dir = pred_dir_raw if pred_dir_raw in VALID_DIRS else "draw"
        out["predicted_direction"] = pred_dir

    out["predicted_score"] = score_label
    out["predicted_label"] = score_label
    out["is_score_others"] = is_others

    hp, dp, ap = _pct_normalize(
        out.get("home_win_pct", 33.3),
        out.get("draw_pct", 33.3),
        out.get("away_win_pct", 33.4),
    )

    out["home_win_pct"] = hp
    out["draw_pct"] = dp
    out["away_win_pct"] = ap

    conf = _i(out.get("confidence", out.get("ai_confidence", 55)), 55)
    conf = max(25, min(95, conf))
    out["confidence"] = conf

    raw_goal_range = out.get("goal_range") or _goal_range_from_score(score_label)
    bucket, label = _normalize_goal_range_for_ui(raw_goal_range, score_label)
    out["goal_range"] = bucket
    out["goal_range_label"] = label

    top5 = out.get("top5_scores", out.get("top3", []))
    if not isinstance(top5, list):
        top5 = []

    fixed_top5 = []
    has_main = False

    for t in top5:
        if not isinstance(t, dict):
            continue

        sc = str(t.get("score", "")).strip().replace("：", "-").replace(":", "-").replace(" ", "")
        if not sc:
            continue

        if _parse_score(sc)[0] is None and sc not in ["胜其他", "平其他", "负其他"]:
            continue

        prob = _f(t.get("prob", t.get("prob_pct", 0)))
        if prob <= 1 and prob > 0:
            prob *= 100

        fixed_top5.append({
            "score": sc,
            "prob": round(prob, 1),
        })

        if sc == score_label:
            has_main = True

    if not has_main:
        fixed_top5.insert(0, {"score": score_label, "prob": 0})

    fixed_top5 = sorted(
        fixed_top5,
        key=lambda x: 0 if x.get("score") == score_label else 1
    )[:5]

    out["top5_scores"] = fixed_top5
    out["top3"] = fixed_top5[:3]

    for k in [
        "doubts",
        "adopted_analysts",
        "rejected_analysts",
    ]:
        if not isinstance(out.get(k), list):
            out[k] = []

    for k in [
        "main_market_path",
        "sharp_or_steam_view",
        "crs_view",
        "ttg_view",
        "handicap_view",
        "why_not_1_0",
        "why_not_2_0",
        "why_not_1_1",
        "why_not_3_1",
        "why_not_1_2",
        "reason",
        "audit_result",
        "arbitration_reason",
        "agreement_pattern",
    ]:
        if out.get(k) is None:
            out[k] = ""
        else:
            out[k] = _safe_str(out.get(k), 3000)

    if not isinstance(out.get("external_check"), dict):
        out["external_check"] = {
            "available": False,
            "summary": "未返回可验证外部联网核对信息。",
        }

    if not isinstance(out.get("analysis_coverage"), dict):
        out["analysis_coverage"] = {}

    if phase == "claude":
        if not out.get("arbitration_reason"):
            out["arbitration_reason"] = out.get("reason", "")
        out["audit_result"] = _safe_str(out.get("audit_result", ""), 1200)
    else:
        out["reason"] = _safe_str(out.get("reason", ""), 5000)

    return len(errors) == 0, out, errors


# ====================================================================
# Phase 1 覆盖率
# ====================================================================

def _is_valid_phase1_result(r: Dict) -> bool:
    if not isinstance(r, dict) or not r:
        return False

    score = str(r.get("predicted_score", "")).strip()
    direction = str(r.get("predicted_direction", "")).strip()

    if not score or score in ["-", "None", "null"]:
        return False

    if direction not in VALID_DIRS:
        return False

    return True


def _phase1_coverage_for_match(phase1_results: Dict[str, Dict[int, Dict]], idx: int) -> Dict[str, Any]:
    gpt_ok = _is_valid_phase1_result(phase1_results.get("gpt", {}).get(idx, {}))
    grok_ok = _is_valid_phase1_result(phase1_results.get("grok", {}).get(idx, {}))
    gemini_ok = _is_valid_phase1_result(phase1_results.get("gemini", {}).get(idx, {}))

    valid_count = sum([gpt_ok, grok_ok, gemini_ok])

    return {
        "gpt": gpt_ok,
        "grok": grok_ok,
        "gemini": gemini_ok,
        "valid_count": valid_count,
        "coverage_ok": valid_count >= 2,
        "coverage_full": valid_count == 3,
    }


# ====================================================================
# Phase 1 / Phase 2 执行
# ====================================================================

async def run_phase1_three(match_blocks: List[str], num_matches: int) -> Dict[str, Dict[int, Dict]]:
    print(f"\n  [Phase 1 RAW] GPT/Grok/Gemini 三家并行分析 ({num_matches} 场)...")
    print("  [Phase 1 RAW] 不做Repair，不做单场补跑")

    sys_prompts = {
        "gpt": (
            "<role>你是RAW赔率结构分析师，只能基于原始抓包赔率自主分析。</role>"
            "<instruction>严格输出 JSON 数组，不要markdown。禁止模板比分。</instruction>"
        ),
        "grok": (
            "<role>你是RAW资金/变盘分析师，只能基于原始抓包的变盘、热度、资金字段判断。</role>"
            "<instruction>严格输出 JSON 数组，不要markdown。禁止模板比分。</instruction>"
        ),
        "gemini": (
            "<role>你是RAW基本面/场景分析师，只能基于原始抓包基本面、赛事情境、伤停字段判断。</role>"
            "<instruction>严格输出 JSON 数组，不要markdown。禁止模板比分。</instruction>"
        ),
    }

    connector = aiohttp.TCPConnector(limit=10, use_dns_cache=False)

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []

        for cfg in PHASE1_CONFIGS:
            role_key = cfg["role_key"]
            prompt = build_phase1_prompt(match_blocks, role_key)
            temp = PHASE1_ROLES[role_key]["temperature"]

            tasks.append(async_call_ai_batch(
                session=session,
                prompt=prompt,
                url_env=cfg["url_env"],
                key_env=cfg["key_env"],
                models_list=cfg["models"],
                num_matches=num_matches,
                ai_name=cfg["ai_name"],
                sys_prompt=sys_prompts[cfg["ai_name"]],
                temperature=temp,
                phase="phase1",
            ))

        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        output = {
            "gpt": {},
            "grok": {},
            "gemini": {},
        }

        for res in raw_results:
            if isinstance(res, tuple):
                ai_name, results, status = res
                output[ai_name] = results
                AI_CALL_STATUS[ai_name] = status
                print(f"  [状态] {ai_name.upper()} => {status} | 返回 {len(results)}/{num_matches} 场")
            else:
                print(f"  [Phase1异常] {res}")

    ok = sum(1 for v in output.values() if v)
    print(f"  [Phase 1 RAW 完成] {ok}/3 家有数据")

    return output


async def run_phase2_claude_audit(
    match_blocks: List[str],
    phase1_results: Dict[str, Dict[int, Dict]],
    num_matches: int
) -> Tuple[Dict[int, Dict], str]:
    print(f"\n  [Phase 2 RAW] Claude 终审复盘 ({num_matches} 场)...")

    prompt = build_phase2_prompt(match_blocks, phase1_results)
    print(f"  [Claude RAW Prompt] {len(prompt):,} 字符")

    sys_prompt = (
        "<role>你是 Claude RAW 终审复盘者。你必须重新审计原始抓包，不得简单按三家多数派裁决。</role>\n"
        "<instruction>严格输出 JSON 数组，禁止 markdown，禁止前缀后缀。禁止模板比分。</instruction>"
    )

    connector = aiohttp.TCPConnector(limit=5, use_dns_cache=False)

    async with aiohttp.ClientSession(connector=connector) as session:
        ai_name, results, model_name = await async_call_ai_batch(
            session=session,
            prompt=prompt,
            url_env=CLAUDE_CONFIG["url_env"],
            key_env=CLAUDE_CONFIG["key_env"],
            models_list=CLAUDE_CONFIG["models"],
            num_matches=num_matches,
            ai_name="claude",
            sys_prompt=sys_prompt,
            temperature=CLAUDE_CONFIG["temperature"],
            phase="claude",
        )

    AI_CALL_STATUS["claude"] = model_name

    print(f"  [Phase 2 RAW 完成] Claude 返回 {len(results)}/{num_matches} | 状态={model_name}")

    return results, f"claude:{model_name}"


# ====================================================================
# 输出包装 / 兼容字段
# ====================================================================

def _enforce_direction_consistency(result: Dict) -> Dict:
    out = dict(result or {})

    score = out.get("predicted_score", "1-1")
    score_label, is_others = _score_to_label(score)

    expected_dir = _score_direction(score_label)
    if not expected_dir:
        expected_dir = out.get("predicted_direction", "draw")
        if expected_dir not in VALID_DIRS:
            expected_dir = "draw"
        score_label = _fallback_score_for_direction(expected_dir)

    result_cn = _direction_cn(expected_dir)

    raw_goal_range = out.get("goal_range") or _goal_range_from_score(score_label)
    goal_bucket, goal_label = _normalize_goal_range_for_ui(raw_goal_range, score_label)

    conf = _i(out.get("confidence", 55), 55)
    conf = max(25, min(95, conf))

    hp, dp, ap = _pct_normalize(
        out.get("home_win_pct", 33.3),
        out.get("draw_pct", 33.3),
        out.get("away_win_pct", 33.4),
    )

    # 概率argmax不强制篡改为比分方向，只保证展示字段一致。
    # Raw-Only 版不做后验锁，不应人为改概率结构。
    out["predicted_score"] = score_label
    out["predicted_label"] = score_label
    out["predicted_direction"] = expected_dir
    out["final_direction"] = expected_dir
    out["result"] = result_cn
    out["display_direction"] = result_cn
    out["is_score_others"] = is_others

    out["goal_range"] = goal_bucket
    out["goal_interval"] = goal_bucket
    out["predicted_goal_range"] = goal_bucket
    out["goal_range_label"] = goal_label

    out["confidence"] = conf
    out["ai_confidence"] = conf
    out["ai_confidence_pct"] = conf
    out["ai_confidence_score"] = conf
    out["confidence_score"] = conf
    out["confidence_pct"] = conf
    out["analysis_confidence"] = conf
    out["dir_confidence"] = conf
    out["ai_confidence_value"] = conf
    out["aiConfidence"] = conf
    out["confidenceValue"] = conf
    out["final_confidence"] = conf
    out["prediction_confidence"] = conf
    out["finalConfidence"] = conf
    out["ai_conf"] = conf
    out["cf"] = conf
    out["ai_trust"] = conf
    out["ai_trust_pct"] = conf
    out["trust_score"] = conf
    out["model_confidence"] = conf
    out["model_confidence_pct"] = conf

    out["home_win_pct"] = hp
    out["draw_pct"] = dp
    out["away_win_pct"] = ap

    top5 = out.get("top5_scores", out.get("top3", []))
    if not isinstance(top5, list):
        top5 = []
    if not any(isinstance(x, dict) and x.get("score") == score_label for x in top5):
        top5.insert(0, {"score": score_label, "prob": 0})
    out["top5_scores"] = top5[:5]
    out["top3"] = top5[:3]

    return out


def _ai_score_summary(r: Dict) -> str:
    if not isinstance(r, dict) or not r:
        return "-"
    return str(r.get("predicted_score", "-"))


def _ai_reason_summary(r: Dict, empty_text: str) -> str:
    if not isinstance(r, dict) or not r:
        return empty_text
    return _safe_str(r.get("reason", ""), 2500)


def _fallback_claude_result_from_phase1(phase1_results: Dict[str, Dict[int, Dict]], idx: int) -> Dict:
    for name in ["gpt", "grok", "gemini"]:
        r = phase1_results.get(name, {}).get(idx, {})
        if r and r.get("predicted_score"):
            out = dict(r)
            out["agreement_pattern"] = f"Claude失败，采用{name.upper()} RAW兜底"
            out["adopted_analysts"] = [name]
            out["rejected_analysts"] = []
            out["audit_result"] = f"Claude 未返回有效终审，临时采用 {name.upper()} RAW 输出。"
            out["arbitration_reason"] = f"Claude 未返回有效终审，临时采用 {name.upper()} RAW 输出：{r.get('reason', '')}"
            out["confidence"] = max(25, int(_f(r.get("confidence", 45)) * 0.80))
            return out

    return {
        "match": idx,
        "predicted_score": "1-1",
        "predicted_direction": "draw",
        "goal_range": "2球",
        "home_win_pct": 33.3,
        "draw_pct": 34.0,
        "away_win_pct": 32.7,
        "confidence": 30,
        "top5_scores": [
            {"score": "1-1", "prob": 0},
            {"score": "1-0", "prob": 0},
            {"score": "0-1", "prob": 0},
            {"score": "2-1", "prob": 0},
            {"score": "1-2", "prob": 0},
        ],
        "agreement_pattern": "全部AI失败",
        "adopted_analysts": [],
        "rejected_analysts": [],
        "analysis_coverage": {"gpt": False, "grok": False, "gemini": False, "valid_count": 0},
        "audit_result": "全部AI失败，兜底输出1-1。",
        "arbitration_reason": "全部AI失败，兜底输出1-1。此结果不可作为强判断。",
        "doubts": ["AI未返回有效结果，使用保守兜底"],
    }


def assemble_final_prediction(
    match: Dict,
    phase1_results: Dict[str, Dict[int, Dict]],
    claude_result: Dict,
    idx: int,
    ai_provider: str
) -> Dict:
    cr = _enforce_direction_consistency(claude_result or {})

    predicted_score = cr.get("predicted_score", "1-1")
    predicted_label = cr.get("predicted_label", predicted_score)
    final_direction = cr.get("final_direction", "draw")
    result_cn = cr.get("result", "平局")
    is_others = bool(cr.get("is_score_others", False))

    h_score, a_score = _parse_score(predicted_score)
    goal_count = h_score + a_score if h_score is not None else None

    home_pct = _f(cr.get("home_win_pct", 33.3), 33.3)
    draw_pct = _f(cr.get("draw_pct", 33.3), 33.3)
    away_pct = _f(cr.get("away_win_pct", 33.4), 33.4)

    confidence = _i(cr.get("confidence", 55), 55)
    confidence = max(25, min(95, confidence))

    coverage = _phase1_coverage_for_match(phase1_results, idx)

    if coverage["valid_count"] <= 1:
        confidence = min(confidence, 62)
    elif coverage["valid_count"] == 2:
        confidence = min(confidence, 72)

    risk = "低" if confidence >= 75 else ("中" if confidence >= 55 else "高")

    final_odds = _get_score_odds(match, predicted_score, final_direction, is_others)
    raw_score_prob = _extract_score_prob_from_ai(cr, predicted_score)

    if ENABLE_LLM_VALUE_BET:
        score_prob = raw_score_prob
        ev_pct, kelly_pct, is_value = _calculate_score_ev(score_prob, final_odds)
        value_reason = "基于Claude RAW top5精确比分概率计算，未做历史校准"
    else:
        score_prob = raw_score_prob
        ev_pct, kelly_pct, is_value = 0.0, 0.0, False
        value_reason = "v19.4 RAW-ONLY 暂不使用 LLM 主观比分概率计算正式EV/Kelly"

    raw_goal_range = cr.get("goal_range") or _goal_range_from_score(predicted_score)
    goal_bucket, goal_label = _normalize_goal_range_for_ui(raw_goal_range, predicted_score)

    hp_list = [home_pct, draw_pct, away_pct]
    hp_sorted = sorted(hp_list)

    p1_gpt = phase1_results.get("gpt", {}).get(idx, {})
    p1_grok = phase1_results.get("grok", {}).get(idx, {})
    p1_gemini = phase1_results.get("gemini", {}).get(idx, {})

    claude_reason = (
        cr.get("arbitration_reason")
        or cr.get("reason")
        or cr.get("audit_result")
        or ""
    )

    raw_smart_signals = _raw_smart_signals_from_match(match)

    # UI显示用：不跑模型xG。只读原始字段；没有就用0。
    raw_home_xg = _f(
        match.get("bookmaker_implied_home_xg", match.get("xG_home", match.get("home_xg", 0))),
        0
    )
    raw_away_xg = _f(
        match.get("bookmaker_implied_away_xg", match.get("xG_away", match.get("away_xg", 0))),
        0
    )
    raw_exp_goals = _f(
        match.get("expected_total_goals", match.get("exp_goals", match.get("lambda_total", 0))),
        0
    )
    if raw_exp_goals <= 0 and goal_count is not None:
        raw_exp_goals = float(goal_count)

    return {
        "predicted_score": predicted_score,
        "predicted_label": predicted_label,
        "predicted_direction": final_direction,
        "result": result_cn,
        "display_direction": result_cn,
        "final_direction": final_direction,
        "is_score_others": is_others,

        "decision_title": "vMAX 19.4 RAW-ONLY 决策剖析",
        "decision_engine_version": ENGINE_VERSION,
        "decision_architecture": ENGINE_ARCHITECTURE,
        "engine_version": ENGINE_VERSION,
        "engine_architecture": ENGINE_ARCHITECTURE,

        "goal_range": goal_bucket,
        "goal_interval": goal_bucket,
        "predicted_goal_range": goal_bucket,
        "goal_range_label": goal_label,
        "goal_bucket": goal_bucket,
        "goals_range": goal_bucket,
        "goal_range_text": goal_label,
        "total_goals_range": goal_bucket,
        "predicted_goals_range": goal_bucket,
        "predicted_goals_label": goal_label,
        "expected_goal_range": goal_bucket,
        "expected_goals_range": goal_bucket,
        "goal_zone": goal_bucket,
        "goals_zone": goal_bucket,
        "score_goal_range": goal_bucket,
        "goal_count": goal_count,
        "total_goal_count": goal_count,
        "goals_count": goal_count,
        "goal_count_label": goal_label,

        "home_win_pct": home_pct,
        "draw_pct": draw_pct,
        "away_win_pct": away_pct,

        "confidence": confidence,
        "ai_confidence": confidence,
        "ai_confidence_pct": confidence,
        "ai_confidence_score": confidence,
        "ai_confidence_value": confidence,
        "aiConfidence": confidence,
        "confidence_score": confidence,
        "confidenceValue": confidence,
        "confidence_pct": confidence,
        "analysis_confidence": confidence,
        "final_confidence": confidence,
        "prediction_confidence": confidence,
        "finalConfidence": confidence,
        "ai_conf": confidence,
        "cf": confidence,
        "ai_trust": confidence,
        "ai_trust_pct": confidence,
        "trust_score": confidence,
        "model_confidence": confidence,
        "model_confidence_pct": confidence,
        "dir_confidence": confidence,
        "risk_level": risk,
        "dir_gap": round(hp_sorted[-1] - hp_sorted[-2], 1) if len(hp_sorted) >= 2 else 0,

        "ai_call_status": dict(AI_CALL_STATUS),
        "gpt_status": AI_CALL_STATUS.get("gpt", ""),
        "grok_status": AI_CALL_STATUS.get("grok", ""),
        "gemini_status": AI_CALL_STATUS.get("gemini", ""),
        "claude_status": AI_CALL_STATUS.get("claude", ""),

        "ai_provider": ai_provider,
        "claude_score": predicted_score,
        "claude_analysis": claude_reason[:3000],
        "arbitration_reason": claude_reason,
        "audit_result": cr.get("audit_result", ""),
        "agreement_pattern": cr.get("agreement_pattern", "Claude RAW终审复盘"),

        "phase1_coverage": coverage,
        "analysis_coverage": cr.get("analysis_coverage", coverage),
        "coverage_ok": coverage["coverage_ok"],
        "coverage_full": coverage["coverage_full"],

        "adopted_analysts": cr.get("adopted_analysts", []),
        "rejected_analysts": cr.get("rejected_analysts", []),

        "top5_scores": cr.get("top5_scores", []),
        "top3": cr.get("top3", cr.get("top5_scores", [])[:3]),

        "main_market_path": cr.get("main_market_path", ""),
        "crs_view": cr.get("crs_view", ""),
        "ttg_view": cr.get("ttg_view", ""),
        "handicap_view": cr.get("handicap_view", ""),
        "sharp_or_steam_view": cr.get("sharp_or_steam_view", ""),

        "why_not_1_0": cr.get("why_not_1_0", ""),
        "why_not_2_0": cr.get("why_not_2_0", ""),
        "why_not_1_1": cr.get("why_not_1_1", ""),
        "why_not_3_1": cr.get("why_not_3_1", ""),
        "why_not_1_2": cr.get("why_not_1_2", ""),

        "external_check": cr.get("external_check", {"available": False, "summary": ""}),
        "doubts": cr.get("doubts", []),
        "data_quality": cr.get("data_quality", {}),
        "ai_validation_errors": cr.get("ai_validation_errors", []),

        "gpt_score": _ai_score_summary(p1_gpt),
        "gpt_analysis": _ai_reason_summary(p1_gpt, "GPT 未返回有效RAW分析。"),
        "gpt_doubts": p1_gpt.get("doubts", []) if p1_gpt else [],
        "gpt_main_market_path": p1_gpt.get("main_market_path", "") if p1_gpt else "",

        "grok_score": _ai_score_summary(p1_grok),
        "grok_analysis": _ai_reason_summary(p1_grok, "GROK 未返回有效RAW分析。"),
        "grok_doubts": p1_grok.get("doubts", []) if p1_grok else [],
        "grok_main_market_path": p1_grok.get("main_market_path", "") if p1_grok else "",

        "gemini_score": _ai_score_summary(p1_gemini),
        "gemini_analysis": _ai_reason_summary(p1_gemini, "GEMINI 未返回有效RAW分析。"),
        "gemini_doubts": p1_gemini.get("doubts", []) if p1_gemini else [],
        "gemini_main_market_path": p1_gemini.get("main_market_path", "") if p1_gemini else "",

        "ai_abstained": [
            n.upper()
            for n in ["gpt", "grok", "gemini"]
            if not phase1_results.get(n, {}).get(idx)
        ],

        # Raw-Only 版不生成陷阱矩阵，仅保留字段兼容
        "traps_detected": [],
        "trap_codes": [],
        "trap_items": [],
        "observation_items": [],
        "trap_count": 0,
        "trap_facts": [],
        "observation_signals": [],
        "trap_matrix_title": "RAW-ONLY 无系统陷阱矩阵",
        "trap_matrix_subtitle": "本版本只展示AI基于原始抓包的审计结论",

        "score_odds": final_odds,
        "raw_llm_score_prob": round(raw_score_prob * 100, 2),
        "score_prob": round(score_prob * 100, 2),
        "suggested_kelly": kelly_pct,
        "edge_vs_market": ev_pct,
        "is_value": is_value,
        "value_reason": value_reason,

        "raw_smart_signals": raw_smart_signals,
        "smart_signals": raw_smart_signals,
        "smart_money_signal": " | ".join([str(s) for s in raw_smart_signals[:10]]),
        "sharp_detected": any("Sharp" in str(s) or "sharp" in str(s) for s in raw_smart_signals),
        "sharp_dir": None,

        "xG_home": round(raw_home_xg, 2),
        "xG_away": round(raw_away_xg, 2),
        "expected_total_goals": round(raw_exp_goals, 2),
        "over_2_5": match.get("over_25", match.get("over_2_5", 50)),
        "btts": match.get("btts", match.get("both_score", 45)),
        "bookmaker_implied_home_xg": raw_home_xg if raw_home_xg > 0 else "?",
        "bookmaker_implied_away_xg": raw_away_xg if raw_away_xg > 0 else "?",

        # Raw-Only 不调用 ensemble
        "model_consensus_dir": "",
        "model_consensus_count": 0,
        "total_models": 0,
        "ensemble_reference": {},

        "refined_poisson": {},
        "elo": {},
        "random_forest": {},
        "gradient_boost": {},
        "neural_net": {},
        "logistic": {},
        "svm": {},
        "knn": {},
        "dixon_coles": {},
        "bradley_terry": {},
        "experience_analysis": {},
        "top_scores": [],
    }


# ====================================================================
# Top4 推荐：Raw-Only 简化版
# ====================================================================

def select_top4(preds):
    def _score(x):
        p = x.get("prediction", {}) or {}

        confidence = _f(p.get("confidence", 0))
        risk_penalty = 0

        if p.get("risk_level") == "高":
            risk_penalty += 8
        if p.get("ai_validation_errors"):
            risk_penalty += 5
        if p.get("ai_abstained"):
            risk_penalty += len(p.get("ai_abstained", [])) * 2
        if not p.get("coverage_ok", True):
            risk_penalty += 8

        # 有明确CRS/TTG/盘口解释的，加一点稳定分
        explanation_bonus = 0
        for k in ["crs_view", "ttg_view", "handicap_view", "main_market_path"]:
            if len(str(p.get(k, ""))) >= 20:
                explanation_bonus += 1.5

        return confidence + explanation_bonus - risk_penalty

    preds_sorted = sorted(preds, key=_score, reverse=True)
    return preds_sorted[:4]


def extract_num(ms):
    wm = {
        "一": 1000,
        "二": 2000,
        "三": 3000,
        "四": 4000,
        "五": 5000,
        "六": 6000,
        "日": 7000,
        "天": 7000,
    }

    base = next((v for k, v in wm.items() if k in str(ms)), 0)
    nums = re.findall(r"\d+", str(ms))

    return base + int(nums[0]) if nums else 9999


# ====================================================================
# 主入口
# ====================================================================

def run_predictions(raw, use_ai=True):
    raw_matches = raw.get("matches", [])
    num = len(raw_matches)

    for k in AI_CALL_STATUS:
        AI_CALL_STATUS[k] = ""

    print("\n" + "=" * 80)
    print(f"  [{ENGINE_VERSION}] {ENGINE_ARCHITECTURE} | {num} 场")
    print("=" * 80)

    debug_ai_config()

    match_analyses = []
    match_blocks = []

    for i, raw_m in enumerate(raw_matches):
        m = normalize_match(raw_m)

        block = format_raw_match_block(
            idx=i + 1,
            match=m,
        )

        match_blocks.append(block)

        match_analyses.append({
            "raw_match": raw_m,
            "match": m,
            "index": i + 1,
        })

    phase1_results = {
        "gpt": {},
        "grok": {},
        "gemini": {},
    }
    claude_results = {}
    ai_provider = "no_ai"

    if use_ai and match_blocks:
        async def _run_full_ai():
            p1 = await run_phase1_three(match_blocks, num)
            p2, provider = await run_phase2_claude_audit(match_blocks, p1, num)
            return p1, p2, provider

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures

            def _run_in_thread(coro):
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(coro)
                finally:
                    new_loop.close()

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(_run_in_thread, _run_full_ai())
                try:
                    phase1_results, claude_results, ai_provider = future.result()
                except Exception as e:
                    logger.error(f"RAW 四家矩阵执行崩溃: {e}")
                    phase1_results = {"gpt": {}, "grok": {}, "gemini": {}}
                    claude_results = {}
                    ai_provider = "ai_crashed"
        else:
            try:
                phase1_results, claude_results, ai_provider = asyncio.run(_run_full_ai())
            except Exception as e:
                logger.error(f"RAW 四家矩阵执行崩溃: {e}")
                phase1_results = {"gpt": {}, "grok": {}, "gemini": {}}
                claude_results = {}
                ai_provider = "ai_crashed"

    res = []

    for i, ma in enumerate(match_analyses):
        idx = i + 1
        raw_m = ma["raw_match"]
        m = ma["match"]

        cr = claude_results.get(idx, {})
        if not cr:
            cr = _fallback_claude_result_from_phase1(phase1_results, idx)

        mg = assemble_final_prediction(
            match=m,
            phase1_results=phase1_results,
            claude_result=cr,
            idx=idx,
            ai_provider=ai_provider,
        )

        mg = _enforce_direction_consistency(mg)

        combined = {**raw_m, **m, "prediction": mg}

        for k in [
            "predicted_score",
            "predicted_label",
            "predicted_direction",
            "result",
            "display_direction",
            "final_direction",

            "confidence",
            "ai_confidence",
            "ai_confidence_pct",
            "ai_confidence_score",
            "ai_confidence_value",
            "aiConfidence",
            "confidence_score",
            "confidenceValue",
            "confidence_pct",
            "analysis_confidence",
            "final_confidence",
            "prediction_confidence",
            "finalConfidence",
            "ai_conf",
            "cf",
            "ai_trust",
            "ai_trust_pct",
            "trust_score",
            "model_confidence",
            "model_confidence_pct",

            "goal_range",
            "goal_interval",
            "predicted_goal_range",
            "goal_range_label",
            "goal_bucket",
            "goals_range",
            "goal_range_text",
            "total_goals_range",
            "predicted_goals_range",
            "predicted_goals_label",
            "expected_goal_range",
            "expected_goals_range",
            "goal_zone",
            "goals_zone",
            "score_goal_range",
            "goal_count",
            "total_goal_count",
            "goals_count",
            "goal_count_label",

            "decision_title",
            "decision_engine_version",
            "decision_architecture",
            "engine_version",
            "engine_architecture",

            "ai_call_status",
            "gpt_status",
            "grok_status",
            "gemini_status",
            "claude_status",

            "phase1_coverage",
            "analysis_coverage",
            "coverage_ok",
            "coverage_full",

            "claude_score",
            "claude_analysis",
            "gpt_score",
            "gpt_analysis",
            "grok_score",
            "grok_analysis",
            "gemini_score",
            "gemini_analysis",

            "main_market_path",
            "crs_view",
            "ttg_view",
            "handicap_view",
            "sharp_or_steam_view",
            "why_not_1_0",
            "why_not_2_0",
            "why_not_1_1",
            "why_not_3_1",
            "why_not_1_2",
            "external_check",
            "top5_scores",
            "top3",
        ]:
            combined[k] = mg.get(k)

        combined["engine_version"] = ENGINE_VERSION
        combined["decision_title"] = "vMAX 19.4 RAW-ONLY 决策剖析"
        combined["decision_engine_version"] = ENGINE_VERSION
        combined["decision_architecture"] = ENGINE_ARCHITECTURE

        for ck in [
            "confidence",
            "ai_confidence",
            "ai_confidence_pct",
            "ai_confidence_score",
            "ai_confidence_value",
            "aiConfidence",
            "confidence_score",
            "confidenceValue",
            "confidence_pct",
            "analysis_confidence",
            "final_confidence",
            "prediction_confidence",
            "finalConfidence",
            "ai_conf",
            "cf",
            "ai_trust",
            "ai_trust_pct",
            "trust_score",
            "model_confidence",
            "model_confidence_pct",
        ]:
            combined[ck] = mg.get("confidence", 0)

        for gk in [
            "goal_range",
            "goal_interval",
            "predicted_goal_range",
            "goal_bucket",
            "goals_range",
            "total_goals_range",
            "predicted_goals_range",
            "expected_goal_range",
            "expected_goals_range",
            "goal_zone",
            "goals_zone",
            "score_goal_range",
        ]:
            combined[gk] = mg.get("goal_range")

        for gk in [
            "goal_range_label",
            "goal_range_text",
            "predicted_goals_label",
            "goal_count_label",
        ]:
            combined[gk] = mg.get("goal_range_label")

        for gk in [
            "goal_count",
            "total_goal_count",
            "goals_count",
        ]:
            combined[gk] = mg.get("goal_count")

        res.append(combined)

        err_tag = f" [校验{len(mg.get('ai_validation_errors', []))}]" if mg.get("ai_validation_errors") else ""
        abstain_tag = f" [缺席{','.join(mg.get('ai_abstained', []))}]" if mg.get("ai_abstained") else ""
        cov = mg.get("phase1_coverage", {})
        cov_tag = f" [覆盖{cov.get('valid_count', 0)}/3]"

        print(
            f"  [{idx}] {m.get('home_team', m.get('home', '?'))} vs "
            f"{m.get('away_team', m.get('guest', '?'))} => "
            f"{mg['result']} ({mg['predicted_score']}) | "
            f"CF: {mg['confidence']}% | {mg.get('agreement_pattern', 'Claude RAW终审')}"
            f"{cov_tag}{err_tag}{abstain_tag}"
        )

    t4 = select_top4(res)
    t4ids = [t.get("id") for t in t4]

    for r in res:
        r["is_recommended"] = r.get("id") in t4ids

    res.sort(key=lambda x: extract_num(x.get("match_num", "")))

    return res, t4


# ====================================================================
# 本地启动
# ====================================================================

if __name__ == "__main__":
    logger.info(f"{ENGINE_VERSION} 启动")
    print(f"✅ {ENGINE_VERSION} 加载完成")
    print(f"   架构: {ENGINE_ARCHITECTURE}")
    print("   Phase 1: GPT / Grok / Gemini 三家 RAW 独立分析")
    print("   Phase 2: Claude 接收三家结论 + 原始抓包做最终审计")
    print("   规则: 不调用内部量化链，不生成OBS，不做陷阱矩阵，不跑贝叶斯")
    print("   调用: 默认 STRICT_FOUR_MODEL_CALLS=True，逻辑调用固定4次")
    print("   EV/Kelly: 默认不使用 LLM 主观比分概率")
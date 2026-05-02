# ====================================================================
# 🚀 vMAX 19.5 — Single-Shot RAW Packet 3AI + Claude Audit
# --------------------------------------------------------------------
# 这版原则:
#   ✅ 完整抓包一次性提交给 GPT / Grok / Gemini
#   ✅ 三家 AI 不分工，全部独立完整分析所有比赛
#   ✅ 不做 Phase1 Repair，不补跑，不分场次重复消耗 token
#   ✅ Claude 接收完整抓包 + 三家完整结论，做最终审计
#   ✅ Claude 不为独立而乱改；三家一致时默认沿用，除非有强反证
#   ✅ 抓包不截断，分析不截断
#   ✅ OpenAI-compatible 通道默认 stream=True，绕开中转 524 非流式超时
#   ✅ 本地不使用概率模型、不用本地算法限制 AI，只做 JSON 解析和 UI 字段兼容
#   ✅ 保存原始 AI 响应，便于定位 GPT / Grok / Gemini / Claude 失败原因
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
# 安全导入：保留兼容，但本版默认不使用本地预测引擎
# ====================================================================

try:
    from config import *
except Exception as e:
    logger.warning(f"config 导入异常: {e}")


# ====================================================================
# 基础配置
# ====================================================================

ENGINE_VERSION = "vMAX 19.5"
ENGINE_ARCHITECTURE = "Single-Shot RAW Packet + 3 Full Analysts + Claude Final Audit"

VALID_DIRS = {"home", "draw", "away"}

AI_CALL_STATUS = {
    "gpt": "",
    "grok": "",
    "gemini": "",
    "claude": "",
}

AI_RAW_STATUS = {
    "gpt": "",
    "grok": "",
    "gemini": "",
    "claude": "",
}

# 不限制抓包 / 分析长度
RAW_PACKET_MAX_CHARS = None
MATCH_BRIEF_FIELD_MAX_CHARS = None
AI_REASON_MAX_CHARS = None
AI_ANALYSIS_MAX_CHARS = None
CLAUDE_REASON_MAX_CHARS = None

# 强制 AI 输出足够长的分析
PHASE1_REASON_MIN_CHARS = 700
CLAUDE_REASON_MIN_CHARS = 900
KEY_EVIDENCE_MIN_ITEMS = 6
DOUBTS_MIN_ITEMS = 3

# OpenAI-compatible 接口默认流式，解决 524 网关超时
ENABLE_STREAMING_OPENAI_COMPAT = True
STREAM_SAVE_RAW_CHUNKS = True

# 保存原始响应
SAVE_RAW_AI_RESPONSES = True
RAW_AI_RESPONSE_DIR = "./ai_raw_responses"

# 单次调用，不 fallback，不补跑
STRICT_SINGLE_CALL_PER_AI = True

# 允许 Prompt 要求模型联网；如果通道不支持，要求模型不要编造
ENABLE_AI_WEB_RESEARCH = True


# ====================================================================
# 赔率 / 比分字段映射
# ====================================================================

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

    if max_len is not None and max_len > 0 and len(s) > max_len:
        return s[:max_len] + "..."

    return s


def _safe_json(obj: Any, max_len: Optional[int] = None) -> str:
    try:
        text = json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        text = str(obj)

    if max_len is not None and max_len > 0 and len(text) > max_len:
        return text[:max_len] + "\n...<TRUNCATED_BY_LOCAL_DISPLAY_ONLY>"

    return text


def _direction_cn(direction: str) -> str:
    return {
        "home": "主胜",
        "draw": "平局",
        "away": "客胜",
    }.get(str(direction), "未知")


def _normalize_direction(v: Any) -> str:
    s = str(v or "").strip().lower()

    if s in ["home", "h", "主胜", "胜", "主", "主队", "homewin", "win"]:
        return "home"
    if s in ["draw", "d", "平", "平局", "和", "same"]:
        return "draw"
    if s in ["away", "a", "客胜", "负", "客", "客队", "awaywin", "lose"]:
        return "away"

    return s


def _parse_score(s: str) -> Tuple[Optional[int], Optional[int]]:
    try:
        s_str = str(s).strip().replace(" ", "").replace("：", "-").replace(":", "-")

        if "胜" in s_str and "其他" in s_str:
            return 9, 0
        if "平" in s_str and "其他" in s_str:
            return 9, 9
        if "负" in s_str and "其他" in s_str:
            return 0, 9

        if s_str in ["主胜", "客胜", "平局", "home", "away", "draw"]:
            return None, None

        p = s_str.split("-")
        if len(p) != 2:
            return None, None

        h = int(p[0])
        a = int(p[1])

        if h < 0 or a < 0:
            return None, None

        return h, a
    except Exception:
        return None, None


def _is_valid_score(score: str) -> bool:
    s = str(score or "").strip()

    if s in ["胜其他", "平其他", "负其他"]:
        return True

    h, a = _parse_score(s)
    return h is not None and a is not None


def _score_direction(score_str: str) -> Optional[str]:
    s = str(score_str or "").strip()

    if "胜其他" in s or s == "9-0":
        return "home"
    if "平其他" in s or s == "9-9":
        return "draw"
    if "负其他" in s or s == "0-9":
        return "away"

    h, a = _parse_score(s)

    if h is None:
        return None

    if h > a:
        return "home"
    if h < a:
        return "away"
    return "draw"


def _goal_range_from_score(score: str) -> str:
    h, a = _parse_score(score)

    if h is None:
        return "?"

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


def _score_to_label(score: str) -> Tuple[str, bool]:
    s = str(score or "").strip()

    if "胜其他" in s or s == "9-0":
        return "胜其他", True
    if "平其他" in s or s == "9-9":
        return "平其他", True
    if "负其他" in s or s == "0-9":
        return "负其他", True

    return s, False


def _get_score_odds(match: Dict, score: str, direction: str, is_others: bool) -> float:
    if is_others:
        if direction == "home":
            return _f(match.get("crs_win", 0))
        if direction == "away":
            return _f(match.get("crs_lose", 0))
        return _f(match.get("crs_same", 0))

    key = CRS_FULL_MAP.get(str(score), "")
    if not key:
        return 0.0

    return _f(match.get(key, 0))


# ====================================================================
# 环境变量 / API 通道
# ====================================================================

GPT_DEFAULT_URL = "https://ai.newapi.life/v1"
GPT_DEFAULT_KEY = globals().get("GPT_DEFAULT_KEY", "")

GPT_KEY_ALIASES = [
    "GPT_API_KEY",
    "OPENAI_API_KEY",
    "OPENAI_KEY",
    "AI_API_KEY",
    "NEWAPI_KEY",
    "API_KEY",
]

GPT_URL_ALIASES = [
    "GPT_API_URL",
    "OPENAI_API_BASE",
    "OPENAI_BASE_URL",
    "AI_API_URL",
    "NEWAPI_URL",
    "BASE_URL",
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


def _get_key_for_ai(ai_name: str, key_env: str) -> str:
    if ai_name == "gpt":
        return get_first_clean_env_key(GPT_KEY_ALIASES, GPT_DEFAULT_KEY)

    return get_clean_env_key(key_env)


def _build_single_url_for_ai(ai_name: str, url_env: str) -> str:
    if ai_name == "gpt":
        return get_first_clean_env_url(GPT_URL_ALIASES, GPT_DEFAULT_URL)

    return get_clean_env_url(url_env)


# ====================================================================
# Match 标准化：仅用于 UI 和摘要，不参与本地裁决
# ====================================================================

def _first_non_empty(*vals):
    for v in vals:
        if v not in [None, ""]:
            return v
    return None


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
        nested = raw_m.get(nested_key)
        if isinstance(nested, dict):
            for k, v in nested.items():
                if k not in m or m.get(k) in [None, ""]:
                    m[k] = v

    m["home_team"] = _first_non_empty(
        m.get("home_team"),
        m.get("home"),
        m.get("host"),
        m.get("team_home"),
        m.get("homeName"),
        m.get("home_name"),
        "Home",
    )

    m["away_team"] = _first_non_empty(
        m.get("away_team"),
        m.get("guest"),
        m.get("away"),
        m.get("team_away"),
        m.get("awayName"),
        m.get("away_name"),
        "Away",
    )

    m["home"] = _first_non_empty(m.get("home"), m["home_team"])
    m["guest"] = _first_non_empty(m.get("guest"), m["away_team"])

    m["sp_home"] = _first_non_empty(
        m.get("sp_home"),
        m.get("win"),
        m.get("odds_win"),
        m.get("home_win_odds"),
        m.get("homeOdds"),
        m.get("spf_sp3"),
        m.get("spf_3"),
        m.get("sp3"),
        0,
    )

    m["sp_draw"] = _first_non_empty(
        m.get("sp_draw"),
        m.get("same"),
        m.get("draw"),
        m.get("draw_odds"),
        m.get("drawOdds"),
        m.get("spf_sp1"),
        m.get("spf_1"),
        m.get("sp1"),
        0,
    )

    m["sp_away"] = _first_non_empty(
        m.get("sp_away"),
        m.get("lose"),
        m.get("away_win_odds"),
        m.get("awayOdds"),
        m.get("guest_odds"),
        m.get("spf_sp0"),
        m.get("spf_0"),
        m.get("sp0"),
        0,
    )

    m["win"] = m.get("win", m["sp_home"])
    m["same"] = m.get("same", m["sp_draw"])
    m["lose"] = m.get("lose", m["sp_away"])

    m["give_ball"] = _first_non_empty(
        m.get("give_ball"),
        m.get("handicap"),
        m.get("rq"),
        m.get("let_ball"),
        "0",
    )

    return m


# ====================================================================
# 原始抓包格式化：完整、不截断
# ====================================================================

def build_raw_match_brief(idx: int, match: Dict) -> str:
    home = match.get("home_team", match.get("home", "Home"))
    away = match.get("away_team", match.get("guest", "Away"))
    league = match.get("league", match.get("cup", ""))
    match_num = match.get("match_num", match.get("id", idx))

    lines = []
    lines.append("════════════════════════════════════")
    lines.append(f"第 {idx} 场: {home} vs {away}")
    lines.append("════════════════════════════════════")
    lines.append(f"联赛/赛事: {league}")
    lines.append(f"比赛编号: {match_num}")

    important_keys = [
        "home_team", "away_team", "home", "guest", "league", "cup", "match_num", "id",

        "win", "same", "lose",
        "sp_home", "sp_draw", "sp_away",
        "give_ball", "handicap", "rq", "let_ball",

        "change", "vote", "points", "information", "smart_signals",

        "expected_total_goals",
        "bookmaker_implied_home_xg",
        "bookmaker_implied_away_xg",
        "xG_home",
        "xG_away",
        "over_25",
        "over_2_5",
        "btts",

        "a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7",

        "w10", "w20", "w21", "w30", "w31", "w32",
        "w40", "w41", "w42", "w50", "w51", "w52",

        "s00", "s11", "s22", "s33",

        "l01", "l02", "l12", "l03", "l13", "l23",
        "l04", "l14", "l24", "l05", "l15", "l25",

        "crs_win", "crs_same", "crs_lose",

        "ss", "sp", "sf", "ps", "pp", "pf", "fs", "fp", "ff",
    ]

    for k in important_keys:
        if k in match and match.get(k) not in [None, ""]:
            v = match.get(k)

            if isinstance(v, (dict, list)):
                lines.append(f"{k}: {_safe_json(v, MATCH_BRIEF_FIELD_MAX_CHARS)}")
            else:
                lines.append(f"{k}: {v}")

    return "\n".join(lines)


def build_full_raw_packet(raw: Dict, normalized_matches: List[Dict]) -> str:
    brief_blocks = []

    for i, m in enumerate(normalized_matches, 1):
        brief_blocks.append(build_raw_match_brief(i, m))

    raw_json = _safe_json(raw, RAW_PACKET_MAX_CHARS)

    p = ""
    p += "<match_brief_index>\n"
    p += "\n\n".join(brief_blocks)
    p += "\n</match_brief_index>\n\n"

    p += "<full_raw_packet_json>\n"
    p += raw_json
    p += "\n</full_raw_packet_json>\n"

    return p


# ====================================================================
# 联网搜索 Prompt
# ====================================================================

def build_ai_web_research_instruction() -> str:
    if not ENABLE_AI_WEB_RESEARCH:
        return ""

    p = ""
    p += "<web_research_required>\n"
    p += "如果当前模型通道具备联网搜索能力，你必须主动联网检索每场比赛的最新外部信息。\n"
    p += "如果当前通道不支持联网搜索，不要卡住等待，也不要编造来源；请在 web_research.searched=false 中说明通道未提供可用联网工具，然后仅基于完整抓包分析。\n"
    p += "联网搜索优先关注：\n"
    p += "1. 最新欧赔、亚盘、大小球赔率变化。\n"
    p += "2. Sharp money、Steam move、盘口异动、交易所资金倾向。\n"
    p += "3. Polymarket 或类似交易市场是否有相关比赛/球队/赛事市场；没有则写 not_found。\n"
    p += "4. 最新伤停、首发、轮换、赛程、战意、杯赛属性。\n"
    p += "5. 如果联网信息与抓包冲突，必须写 conflicts_with_raw_packet。\n"
    p += "6. 禁止虚构具体网页、具体赔率和具体新闻来源。\n"
    p += "</web_research_required>\n\n"

    return p


# ====================================================================
# Prompt：Phase 1 三家不分工完整分析
# ====================================================================

def build_phase1_unified_prompt(raw_packet_text: str, num_matches: int) -> str:
    p = ""
    p += "<context>\n"
    p += "你是竞彩足球独立分析 AI。下面是一次性提交的完整抓包数据，包含所有比赛、赔率、盘口、比分赔率、总进球、基本面、伤停、资金信号等原始字段。\n"
    p += "你不需要分工，也不要等待其他模型。你要独立阅读完整抓包，自行联网搜索、自行计算、自行判断每场最终方向和比分。\n"
    p += "抓包全文已经完整给出，不要只看摘要，不要忽略 full_raw_packet_json。\n"
    p += "</context>\n\n"

    p += build_ai_web_research_instruction()

    p += "<hard_rules>\n"
    p += "1. 必须分析全部比赛，不要漏场。\n"
    p += f"2. 本次共有 {num_matches} 场，输出 JSON 数组长度应为 {num_matches}。\n"
    p += "3. match 字段必须对应第几场，从 1 开始编号。\n"
    p += "4. 如果能联网，必须先联网搜索，再结合抓包分析；如果不能联网，必须说明不能联网，不得编造。\n"
    p += "5. 不要因为常见比分就机械输出 1-1；只有抓包和外部材料共同支持平局小球时才可输出 1-1。\n"
    p += "6. predicted_score 必须和 predicted_direction 一致。\n"
    p += "7. top3[0] 必须等于 predicted_score。\n"
    p += "8. 必须基于完整抓包自主计算，不要只复述赔率最低项。\n"
    p += f"9. key_evidence 至少 {KEY_EVIDENCE_MIN_ITEMS} 条，每条必须具体到赔率、盘口、CRS、总进球、伤停或联网信息。\n"
    p += f"10. doubts 至少 {DOUBTS_MIN_ITEMS} 条，必须写反证和风险。\n"
    p += f"11. reason 每场不少于 {PHASE1_REASON_MIN_CHARS} 个中文字符，不能只写十几个字。\n"
    p += "12. 必须输出 expected_total_goals、over_25_pct、btts_pct。如果抓包没有，就由你根据赔率、总进球和比分分布自行估算。\n"
    p += "13. 如果数据矛盾，可以降低 confidence，但不要漏场。\n"
    p += "14. 严格输出 JSON 数组，不要 markdown，不要代码块，不要前缀，不要后缀。\n"
    p += "</hard_rules>\n\n"

    p += "<analysis_requirements>\n"
    p += "你需要综合判断：欧赔主平客结构、让球/盘口方向、总进球 a0~a7、精确比分 CRS、半全场、赔率变化、Sharp/Steam/散户热度、联网搜索到的最新赔率、聪明钱、Polymarket、伤停、首发、战意、基本面、赛程、杯赛属性，以及各市场是否共振或互相矛盾。\n"
    p += "分析必须具体，不允许只写“主队实力强”“客队伤停严重”这种短句，必须指出对应抓包字段或联网发现。\n"
    p += "</analysis_requirements>\n\n"

    p += "<raw_data>\n"
    p += raw_packet_text
    p += "\n</raw_data>\n\n"

    p += "<output_schema>\n"
    p += "[\n"
    p += "  {\n"
    p += "    \"match\": 1,\n"
    p += "    \"predicted_score\": \"2-1\",\n"
    p += "    \"predicted_direction\": \"home\",\n"
    p += "    \"result\": \"主胜\",\n"
    p += "    \"goal_range\": \"3球\",\n"
    p += "    \"expected_total_goals\": 2.85,\n"
    p += "    \"over_25_pct\": 56,\n"
    p += "    \"btts_pct\": 52,\n"
    p += "    \"home_win_pct\": 52,\n"
    p += "    \"draw_pct\": 27,\n"
    p += "    \"away_win_pct\": 21,\n"
    p += "    \"confidence\": 68,\n"
    p += "    \"top3\": [\n"
    p += "      {\"score\":\"2-1\", \"prob\":17},\n"
    p += "      {\"score\":\"1-1\", \"prob\":13},\n"
    p += "      {\"score\":\"1-0\", \"prob\":12}\n"
    p += "    ],\n"
    p += "    \"key_evidence\": [\n"
    p += "      \"具体证据1，必须包含抓包字段或联网发现\",\n"
    p += "      \"具体证据2\",\n"
    p += "      \"具体证据3\",\n"
    p += "      \"具体证据4\",\n"
    p += "      \"具体证据5\",\n"
    p += "      \"具体证据6\"\n"
    p += "    ],\n"
    p += "    \"web_research\": {\n"
    p += "      \"searched\": true,\n"
    p += "      \"queries\": [\"Team A vs Team B odds\", \"Team A Team B injury lineup\"],\n"
    p += "      \"odds_market_findings\": [\"最新欧赔/亚盘/大小球变化\"],\n"
    p += "      \"smart_money_findings\": [\"Sharp/Steam/盘口异动/交易所发现\"],\n"
    p += "      \"polymarket_findings\": [\"Polymarket相关市场或not_found\"],\n"
    p += "      \"team_news_findings\": [\"伤停/首发/轮换/战意\"],\n"
    p += "      \"source_notes\": [\"网站或媒体来源说明\"],\n"
    p += "      \"conflicts_with_raw_packet\": []\n"
    p += "    },\n"
    p += "    \"doubts\": [\"反证1\", \"反证2\", \"反证3\"],\n"
    p += "    \"data_quality\": {\"raw_complete\": true, \"odds_complete\": true, \"crs_complete\": true, \"notes\": []},\n"
    p += "    \"reason\": \"这里必须写完整中文分析，逐项解释方向、比分、总进球、CRS、盘口、联网情报、反证和最终取舍，不少于要求字数。\"\n"
    p += "  }\n"
    p += "]\n"
    p += "</output_schema>\n"

    return p


# ====================================================================
# Prompt：Phase 2 Claude 终审
# ====================================================================

def _phase1_result_for_prompt(ai_name: str, results: Dict[int, Dict], num_matches: int) -> str:
    p = ""
    p += f"\n===== {ai_name.upper()} 完整结论 =====\n"

    for i in range(1, num_matches + 1):
        r = results.get(i, {})

        if not r:
            p += f"\n第 {i} 场: 无有效返回\n"
            continue

        p += f"\n第 {i} 场:\n"
        p += _safe_json(r, None)
        p += "\n"

    return p


def build_phase2_claude_prompt(
    raw_packet_text: str,
    phase1_results: Dict[str, Dict[int, Dict]],
    num_matches: int
) -> str:
    p = ""
    p += "<context>\n"
    p += "你是最终审计 AI。GPT / Grok / Gemini 已经分别拿到同一份完整抓包，并且不分工完成独立分析。\n"
    p += "你现在拿到：完整原始抓包 + 三家完整结论。你需要自行联网搜索后输出最终预测。\n"
    p += "抓包全文已经完整给出，不要只看摘要，不要忽略 full_raw_packet_json。\n"
    p += "</context>\n\n"

    p += build_ai_web_research_instruction()

    p += "<core_rules>\n"
    p += "1. 你必须重新审计完整抓包，不能只按票数机械裁决。\n"
    p += "2. 如果当前通道可以联网，你也必须自行联网搜索最新赔率、盘口、聪明钱、Polymarket、伤停、首发、战意信息。\n"
    p += "3. 如果 GPT/Grok/Gemini 三家的 predicted_score 和 predicted_direction 完全一致，你默认必须沿用一致结论。\n"
    p += "4. 三家完全一致时，只有在原始抓包或你联网搜索到的最新强反证非常明确时，你才允许推翻。\n"
    p += "5. 如果你推翻三家一致结论，必须写出具体反证字段、联网来源类型、盘口变化或伤停突发原因。\n"
    p += "6. 禁止出现三家都给 1-1，而你没有强反证就改成 2-1 的情况。\n"
    p += "7. 禁止为了显得独立而刻意改比分。\n"
    p += "8. 二对一时，按抓包证据 + 联网证据质量裁决，不按人数裁决。\n"
    p += "9. 三家都缺失时，你才基于完整抓包 + 联网搜索独立判断，并降低 confidence。\n"
    p += "10. predicted_score 必须和 predicted_direction 一致。\n"
    p += "11. top3[0] 必须等于 predicted_score。\n"
    p += "12. 必须输出 analyst_consensus，说明三家是否一致。\n"
    p += "13. 必须输出 override_unanimous，如果三家一致但你推翻，则为 true，否则 false。\n"
    p += "14. 必须输出 expected_total_goals、over_25_pct、btts_pct。\n"
    p += f"15. key_evidence 至少 {KEY_EVIDENCE_MIN_ITEMS} 条。\n"
    p += f"16. doubts 至少 {DOUBTS_MIN_ITEMS} 条。\n"
    p += f"17. arbitration_reason 每场不少于 {CLAUDE_REASON_MIN_CHARS} 个中文字符，不能只写十几个字。\n"
    p += "18. audit_result 至少 150 个中文字符，必须说明最终裁决依据。\n"
    p += "19. 严格输出 JSON 数组，不要 markdown，不要代码块，不要前缀，不要后缀。\n"
    p += "</core_rules>\n\n"

    p += "<decision_policy>\n"
    p += "你的任务是审计三家结论、原始抓包与联网材料是否匹配。\n"
    p += "当三家一致且抓包与联网材料没有强反证，最终比分应与三家一致。\n"
    p += "当三家分歧，优先选择与原始抓包 + 联网材料多市场共振更强的一方。\n"
    p += "当比分很接近，例如 1-1 与 2-1，只能在总进球、CRS、让球、欧赔、基本面、联网信息同时支持时才上调或下调。\n"
    p += "你必须逐场说明为什么采用或否决 GPT/Grok/Gemini，不允许只写一句话。\n"
    p += "</decision_policy>\n\n"

    p += "<three_ai_results>\n"
    p += _phase1_result_for_prompt("gpt", phase1_results.get("gpt", {}), num_matches)
    p += _phase1_result_for_prompt("grok", phase1_results.get("grok", {}), num_matches)
    p += _phase1_result_for_prompt("gemini", phase1_results.get("gemini", {}), num_matches)
    p += "\n</three_ai_results>\n\n"

    p += "<full_raw_packet_again>\n"
    p += raw_packet_text
    p += "\n</full_raw_packet_again>\n\n"

    p += "<output_schema>\n"
    p += "[\n"
    p += "  {\n"
    p += "    \"match\": 1,\n"
    p += "    \"predicted_score\": \"2-1\",\n"
    p += "    \"predicted_direction\": \"home\",\n"
    p += "    \"result\": \"主胜\",\n"
    p += "    \"goal_range\": \"3球\",\n"
    p += "    \"expected_total_goals\": 2.85,\n"
    p += "    \"over_25_pct\": 56,\n"
    p += "    \"btts_pct\": 52,\n"
    p += "    \"home_win_pct\": 55,\n"
    p += "    \"draw_pct\": 25,\n"
    p += "    \"away_win_pct\": 20,\n"
    p += "    \"confidence\": 72,\n"
    p += "    \"top3\": [\n"
    p += "      {\"score\":\"2-1\", \"prob\":17},\n"
    p += "      {\"score\":\"1-1\", \"prob\":13},\n"
    p += "      {\"score\":\"1-0\", \"prob\":12}\n"
    p += "    ],\n"
    p += "    \"analyst_consensus\": {\n"
    p += "      \"gpt_score\": \"2-1\",\n"
    p += "      \"grok_score\": \"2-1\",\n"
    p += "      \"gemini_score\": \"1-1\",\n"
    p += "      \"same_score_count\": 2,\n"
    p += "      \"same_direction_count\": 3,\n"
    p += "      \"unanimous_score\": false,\n"
    p += "      \"unanimous_direction\": true\n"
    p += "    },\n"
    p += "    \"override_unanimous\": false,\n"
    p += "    \"adopted_analysts\": [\"gpt\", \"grok\"],\n"
    p += "    \"rejected_analysts\": [\"gemini\"],\n"
    p += "    \"key_evidence\": [\n"
    p += "      \"原始抓包证据1\",\n"
    p += "      \"原始抓包证据2\",\n"
    p += "      \"赔率/盘口证据3\",\n"
    p += "      \"CRS/总进球证据4\",\n"
    p += "      \"联网证据5\",\n"
    p += "      \"反证取舍证据6\"\n"
    p += "    ],\n"
    p += "    \"web_research\": {\n"
    p += "      \"searched\": true,\n"
    p += "      \"queries\": [\"Team A vs Team B latest odds\", \"Team A Team B sharp money\", \"Team A Team B injuries\"],\n"
    p += "      \"odds_market_findings\": [\"终审阶段最新赔率/盘口/大小球发现\"],\n"
    p += "      \"smart_money_findings\": [\"Sharp/Steam/交易所/盘口异动发现\"],\n"
    p += "      \"polymarket_findings\": [\"Polymarket相关市场或not_found\"],\n"
    p += "      \"team_news_findings\": [\"伤停/首发/轮换/战意发现\"],\n"
    p += "      \"source_notes\": [\"来源说明\"],\n"
    p += "      \"conflicts_with_raw_packet\": [\"联网信息与抓包冲突点\"]\n"
    p += "    },\n"
    p += "    \"doubts\": [\"风险1\", \"反证2\", \"不确定性3\"],\n"
    p += "    \"audit_result\": \"终审结论摘要，至少150个中文字符，需要说明为什么最终选择该方向和比分。\",\n"
    p += "    \"arbitration_reason\": \"完整中文终审理由，不少于要求字数，逐项解释原始抓包、三家AI结论、联网赔率、聪明钱、Polymarket、伤停首发、CRS、总进球、反证与最终裁决。\"\n"
    p += "  }\n"
    p += "]\n"
    p += "</output_schema>\n"

    return p


# ====================================================================
# AI 配置
# ====================================================================

PHASE1_CONFIGS = [
    {
        "ai_name": "gpt",
        "url_env": "GPT_API_URL",
        "key_env": "GPT_API_KEY",
        "models": ["gpt-5.5"],
    },
    {
        "ai_name": "grok",
        "url_env": "GROK_API_URL",
        "key_env": "GROK_API_KEY",
        "models": ["熊猫-A-5-grok-4.2-fast-200w上下文"],
    },
    {
        "ai_name": "gemini",
        "url_env": "GEMINI_API_URL",
        "key_env": "GEMINI_API_KEY",
        "models": ["熊猫特供-按量-SSS-gemini-3.1-pro-preview-thinking"],
    },
]

CLAUDE_CONFIG = {
    "ai_name": "claude",
    "url_env": "CLAUDE_API_URL",
    "key_env": "CLAUDE_API_KEY",
    "models": ["熊猫-69-满血openrouter-claude-opus-4.7-上下文1000k"],
    "temperature": 0.18,
}

AI_TEMPERATURE = {
    "gpt": 0.20,
    "grok": 0.20,
    "gemini": 0.20,
    "claude": 0.18,
}


# ====================================================================
# AI 响应提取 / 流式读取
# ====================================================================

async def _read_openai_compatible_stream(response, ai_name: str, phase: str) -> str:
    pieces = []
    raw_lines = []

    async for raw in response.content:
        try:
            line = raw.decode("utf-8", errors="ignore")
        except Exception:
            continue

        if not line:
            continue

        raw_lines.append(line)

        for part in line.splitlines():
            part = part.strip()

            if not part:
                continue

            if not part.startswith("data:"):
                continue

            data_str = part[5:].strip()

            if not data_str:
                continue

            if data_str == "[DONE]":
                continue

            try:
                data = json.loads(data_str)
            except Exception:
                continue

            choices = data.get("choices")

            if isinstance(choices, list) and choices:
                delta = choices[0].get("delta", {})
                msg = choices[0].get("message", {})

                if isinstance(delta, dict):
                    content = delta.get("content")

                    if isinstance(content, str):
                        pieces.append(content)

                    elif isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict):
                                t = item.get("text") or item.get("content")
                                if isinstance(t, str):
                                    pieces.append(t)

                if isinstance(msg, dict):
                    content = msg.get("content")
                    if isinstance(content, str):
                        pieces.append(content)

            event_type = data.get("type", "")

            if event_type in [
                "response.output_text.delta",
                "output_text.delta",
                "message.delta",
            ]:
                delta_text = data.get("delta")
                if isinstance(delta_text, str):
                    pieces.append(delta_text)

            if isinstance(data.get("text"), str):
                pieces.append(data["text"])

            if isinstance(data.get("content"), str):
                pieces.append(data["content"])

    text = "".join(pieces).strip()

    if SAVE_RAW_AI_RESPONSES and STREAM_SAVE_RAW_CHUNKS:
        try:
            os.makedirs(RAW_AI_RESPONSE_DIR, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            fp = os.path.join(RAW_AI_RESPONSE_DIR, f"{ts}_{phase}_{ai_name}_stream_raw_sse.txt")
            with open(fp, "w", encoding="utf-8") as f:
                f.write("\n".join(raw_lines))
            print(f"    🧾 {ai_name.upper()} SSE原始流已保存: {fp}")
        except Exception as e:
            print(f"    ⚠️ 保存 {ai_name.upper()} SSE原始流失败: {str(e)[:120]}")

    return text


def _extract_response_text(data, is_gem=False) -> str:
    raw_text = ""

    try:
        if is_gem:
            candidates = data.get("candidates", [])
            if candidates:
                content = candidates[0].get("content", {})
                parts = content.get("parts", [])

                texts = []
                for part in parts:
                    if isinstance(part, dict):
                        t = part.get("text")
                        if isinstance(t, str):
                            texts.append(t)

                return "\n".join(texts).strip()

        if isinstance(data.get("output_text"), str):
            return data["output_text"].strip()

        if isinstance(data.get("text"), str):
            return data["text"].strip()

        if isinstance(data.get("content"), str):
            return data["content"].strip()

        output = data.get("output")
        if isinstance(output, list):
            texts = []

            for item in output:
                if isinstance(item, dict):
                    content = item.get("content")

                    if isinstance(content, list):
                        for c in content:
                            if isinstance(c, dict):
                                t = c.get("text") or c.get("content")
                                if isinstance(t, str):
                                    texts.append(t)

                    elif isinstance(content, str):
                        texts.append(content)

            if texts:
                return "\n".join(texts).strip()

        if data.get("choices"):
            msg = data["choices"][0].get("message", {})

            if isinstance(msg, dict):
                content_val = msg.get("content", "")

                if isinstance(content_val, str) and content_val.strip():
                    raw_text = content_val.strip()

                elif isinstance(content_val, list):
                    best_parts = []

                    for item in content_val:
                        if isinstance(item, dict):
                            t = item.get("text", item.get("content", ""))
                            if isinstance(t, str) and t.strip():
                                best_parts.append(t.strip())

                    raw_text = "\n".join(best_parts).strip()

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
                    ]:
                        v = msg.get(field, "")
                        if isinstance(v, str) and v.strip():
                            raw_text = v.strip()
                            break

        if not raw_text:
            full_str = json.dumps(data, ensure_ascii=False)
            arr = _extract_json_array(full_str)
            if arr:
                raw_text = arr

    except Exception as ex:
        print(f"    ⚠️ 响应提取异常: {str(ex)[:100]}")

    return raw_text


# ====================================================================
# JSON 解析
# ====================================================================

def _strip_thinking_blocks(text: str) -> str:
    clean = str(text or "")
    clean = re.sub(r"<think(?:ing)?>.*?</think(?:ing)?>", "", clean, flags=re.DOTALL | re.IGNORECASE)
    clean = re.sub(r"```(?:json|JSON|python|txt)?", "", clean)
    clean = clean.replace("```", "")
    return clean.strip()


def _extract_json_array(clean: str) -> str:
    patterns = [
        r'\[\s*\{\s*"match"',
        r"\[\s*\{\s*'match'",
        r'\[\s*\{\s*"比赛"',
    ]

    m_re = None

    for pat in patterns:
        m_re = re.search(pat, clean)
        if m_re:
            break

    if m_re:
        start_idx = m_re.start()
        depth = 0
        in_str = False
        quote = ""
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

            if ch in ['"', "'"]:
                if not in_str:
                    in_str = True
                    quote = ch
                    continue
                elif quote == ch:
                    in_str = False
                    quote = ""
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


def _extract_json_payload(clean: str) -> str:
    text = clean.strip()

    arr = _extract_json_array(text)
    if arr:
        return arr

    start = text.find("{")
    end = text.rfind("}") + 1

    if start != -1 and end > start:
        return text[start:end]

    return ""


def _coerce_json_to_list(obj: Any) -> List[Dict]:
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]

    if isinstance(obj, dict):
        for key in [
            "predictions",
            "matches",
            "results",
            "data",
            "items",
            "output",
            "analysis",
            "final_predictions",
        ]:
            v = obj.get(key)

            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]

        if "match" in obj:
            return [obj]

    return []


def _extract_match_index(item: Dict) -> Optional[int]:
    for k in ["match", "match_id", "match_index", "index", "场次", "比赛"]:
        if k in item:
            try:
                return int(_f(item.get(k), None))
            except Exception:
                pass

    return None


def _parse_ai_json(raw_text: str, num_matches: int, phase: str) -> Dict[int, Dict]:
    clean = _strip_thinking_blocks(raw_text)
    json_str = _extract_json_payload(clean)

    results = {}

    if not json_str:
        return results

    obj = None

    try:
        obj = json.loads(json_str)
    except json.JSONDecodeError:
        try:
            fixed = json_str.replace("'", '"')
            obj = json.loads(fixed)
        except Exception:
            try:
                last_brace = json_str.rfind("}")
                last_bracket = json_str.rfind("]")
                cut = max(last_brace, last_bracket)

                if cut != -1:
                    obj = json.loads(json_str[:cut + 1])
            except Exception:
                obj = None

    arr = _coerce_json_to_list(obj)

    if not arr:
        return results

    for item in arr:
        if not isinstance(item, dict):
            continue

        if "match" not in item:
            for k in ["比赛", "场次", "match_id", "match_index", "index"]:
                if k in item:
                    item["match"] = item[k]
                    break

        if "predicted_score" not in item:
            for k in ["score", "比分", "预测比分", "final_score"]:
                if k in item:
                    item["predicted_score"] = item[k]
                    break

        if "predicted_direction" not in item:
            for k in ["direction", "方向", "赛果", "final_direction"]:
                if k in item:
                    item["predicted_direction"] = item[k]
                    break

        fixed, errors = validate_ai_item_without_overwrite(item, phase=phase)
        mid = _extract_match_index(fixed)

        if mid is None:
            continue

        fixed["ai_validation_errors"] = errors

        if 1 <= mid <= max(num_matches, 1):
            results[mid] = fixed

    return results


def validate_ai_item_without_overwrite(item: Dict, phase: str) -> Tuple[Dict, List[str]]:
    errors = []
    out = dict(item or {})

    try:
        out["match"] = int(_f(out.get("match"), None))
    except Exception:
        errors.append("match_id_invalid")
        out["match"] = None
        return out, errors

    score = str(out.get("predicted_score", "")).strip()
    direction = _normalize_direction(out.get("predicted_direction", ""))

    if not score:
        errors.append("score_missing")
    elif not _is_valid_score(score):
        errors.append("score_invalid")

    if direction and direction not in VALID_DIRS:
        errors.append("direction_invalid")

    score_dir = _score_direction(score)

    if score_dir and direction and score_dir != direction:
        errors.append(f"score_direction_mismatch:{score_dir}!={direction}")

    if not direction and score_dir:
        out["predicted_direction"] = score_dir
        direction = score_dir
    else:
        out["predicted_direction"] = direction

    if not out.get("result") and direction in VALID_DIRS:
        out["result"] = _direction_cn(direction)

    hp, dp, ap = _pct_normalize(
        out.get("home_win_pct", 33.3),
        out.get("draw_pct", 33.3),
        out.get("away_win_pct", 33.4),
    )

    out["home_win_pct"] = hp
    out["draw_pct"] = dp
    out["away_win_pct"] = ap

    conf = _i(out.get("confidence", 50), 50)
    conf = max(1, min(99, conf))
    out["confidence"] = conf

    out["ai_confidence"] = conf
    out["ai_confidence_pct"] = conf
    out["ai_confidence_score"] = conf
    out["ai_confidence_percent"] = conf
    out["aiConfidence"] = conf
    out["aiConfidencePct"] = conf
    out["confidence_pct"] = conf
    out["confidence_percent"] = conf
    out["confidence_score"] = conf
    out["analysis_confidence"] = conf
    out["final_confidence"] = conf

    raw_goal_range = out.get("goal_range") or _goal_range_from_score(score)
    bucket, label = _normalize_goal_range_for_ui(raw_goal_range, score)
    out["goal_range"] = bucket
    out["goal_range_label"] = label

    out["expected_total_goals"] = _f(out.get("expected_total_goals", out.get("exp_goals", 0)), 0)
    out["over_25_pct"] = _f(out.get("over_25_pct", out.get("over_2_5", out.get("over25", 0))), 0)
    out["btts_pct"] = _f(out.get("btts_pct", out.get("btts", 0)), 0)

    top3 = out.get("top3", [])

    if not isinstance(top3, list):
        errors.append("top3_invalid")
        top3 = []

    fixed_top3 = []

    for t in top3:
        if not isinstance(t, dict):
            continue

        sc = str(t.get("score", "")).strip()
        prob = _f(t.get("prob", 0))

        if prob <= 1 and prob > 0:
            prob *= 100

        fixed_top3.append({
            "score": sc,
            "prob": round(prob, 1),
        })

    if fixed_top3:
        if score and fixed_top3[0].get("score") != score:
            errors.append("top3_first_not_predicted_score")
    else:
        errors.append("top3_missing")

    out["top3"] = fixed_top3

    for k in [
        "accepted_observations",
        "rejected_observations",
        "doubts",
        "key_signals",
        "key_evidence",
        "adopted_analysts",
        "rejected_analysts",
    ]:
        if not isinstance(out.get(k), list):
            out[k] = []

    if not isinstance(out.get("data_quality"), dict):
        out["data_quality"] = {}

    if not isinstance(out.get("web_research"), dict):
        out["web_research"] = {
            "searched": False,
            "queries": [],
            "odds_market_findings": [],
            "smart_money_findings": [],
            "polymarket_findings": [],
            "team_news_findings": [],
            "source_notes": [],
            "conflicts_with_raw_packet": [],
        }
        errors.append("web_research_missing")

    if not isinstance(out.get("analyst_consensus"), dict):
        out["analyst_consensus"] = {}

    if phase == "claude":
        if not out.get("arbitration_reason"):
            out["arbitration_reason"] = out.get("reason", "")
        if not out.get("audit_result"):
            out["audit_result"] = out.get("reason", "")

    return out, errors


# ====================================================================
# AI 调用：单次、流式、不补跑
# ====================================================================

async def async_call_ai_once(
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
    key = _get_key_for_ai(ai_name, key_env)

    if not key:
        status = f"no_key:{key_env}"
        print(f"  [跳过] {ai_name.upper()} 无可用 KEY: {key_env}")
        return ai_name, {}, status

    base_url = _build_single_url_for_ai(ai_name, url_env)

    if not base_url:
        status = f"no_url:{url_env}"
        print(f"  [跳过] {ai_name.upper()} 无可用 URL: {url_env}")
        return ai_name, {}, status

    model_name = models_list[0] if models_list else ""

    CONNECT_TIMEOUT = 30

    READ_TIMEOUT_MAP = {
        "claude": 1800,
        "grok": 1200,
        "gpt": 1800,
        "gemini": 1800,
    }

    READ_TIMEOUT = READ_TIMEOUT_MAP.get(ai_name, 1200)

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
        use_stream = False
    else:
        headers["Authorization"] = f"Bearer {key}"
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
        }

        use_stream = bool(ENABLE_STREAMING_OPENAI_COMPAT)

        if use_stream:
            payload["stream"] = True

    gw = url.split("/v1")[0][:80]
    print(f"  [🔌 单次调用] {ai_name.upper()} | {model_name[:48]} @ {gw}")
    print(f"  [Prompt] {ai_name.upper()} 字符数: {len(prompt):,}")
    print(f"  [模式] {ai_name.upper()} stream={use_stream}")

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
                status = f"http_{r.status}"

                print(f"    ⚠️ {ai_name.upper()} HTTP {r.status} | {elapsed}s | {text[:800]}")

                if SAVE_RAW_AI_RESPONSES:
                    try:
                        os.makedirs(RAW_AI_RESPONSE_DIR, exist_ok=True)
                        ts = time.strftime("%Y%m%d_%H%M%S")
                        fp = os.path.join(RAW_AI_RESPONSE_DIR, f"{ts}_{phase}_{ai_name}_HTTP_{r.status}.txt")
                        with open(fp, "w", encoding="utf-8") as f:
                            f.write(text)
                        print(f"    🧾 {ai_name.upper()} HTTP错误响应已保存: {fp}")
                    except Exception as e:
                        print(f"    ⚠️ 保存 {ai_name.upper()} HTTP错误失败: {str(e)[:120]}")

                return ai_name, {}, status

            if use_stream:
                raw_text = await _read_openai_compatible_stream(
                    response=r,
                    ai_name=ai_name,
                    phase=phase,
                )
            else:
                try:
                    data = await r.json(content_type=None)
                except Exception as e:
                    text = await r.text()
                    print(f"    ⚠️ {ai_name.upper()} JSON响应读取失败: {str(e)[:160]} | raw={text[:500]}")
                    return ai_name, {}, "response_json_error"

                raw_text = _extract_response_text(data, is_gem=is_gem)

            if SAVE_RAW_AI_RESPONSES:
                try:
                    os.makedirs(RAW_AI_RESPONSE_DIR, exist_ok=True)
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    fp = os.path.join(RAW_AI_RESPONSE_DIR, f"{ts}_{phase}_{ai_name}.txt")
                    with open(fp, "w", encoding="utf-8") as f:
                        f.write(raw_text or "")
                    print(f"    🧾 {ai_name.upper()} 原始响应已保存: {fp}")
                except Exception as e:
                    print(f"    ⚠️ 保存 {ai_name.upper()} 原始响应失败: {str(e)[:120]}")

            if not raw_text or len(raw_text.strip()) < 5:
                print(f"    ⚠️ {ai_name.upper()} 空响应")
                return ai_name, {}, "empty_response"

            results = _parse_ai_json(raw_text, num_matches=num_matches, phase=phase)

            elapsed = round(time.time() - t0, 1)

            if results:
                print(f"    ✅ {ai_name.upper()} 完成: {len(results)}/{num_matches} | {elapsed}s")
                return ai_name, results, model_name

            print(f"    ⚠️ {ai_name.upper()} JSON解析0条 | {elapsed}s")
            return ai_name, {}, "json_parse_failed"

    except asyncio.TimeoutError:
        print(f"    ⏱️ {ai_name.upper()} 单次读取超时: {READ_TIMEOUT}s")
        return ai_name, {}, "read_timeout"

    except aiohttp.ClientConnectorError as e:
        print(f"    ⚠️ {ai_name.upper()} 连接失败: {str(e)[:160]}")
        return ai_name, {}, "connect_error"

    except Exception as e:
        print(f"    ⚠️ {ai_name.upper()} 调用异常: {str(e)[:160]}")
        return ai_name, {}, "error"


# ====================================================================
# Phase 1 / Phase 2 执行
# ====================================================================

async def run_phase1_three(raw_packet_text: str, num_matches: int) -> Dict[str, Dict[int, Dict]]:
    print(f"\n  [Phase 1] GPT/Grok/Gemini 三家一次性完整分析 ({num_matches} 场)...")

    output = {
        "gpt": {},
        "grok": {},
        "gemini": {},
    }

    sys_prompt = (
        "<role>你是竞彩足球独立分析 AI。你必须完整阅读抓包，自行计算方向、比分、总进球与风险。</role>"
        "<instruction>严格输出 JSON 数组，不要 markdown，不要代码块，不要前缀后缀。</instruction>"
    )

    connector = aiohttp.TCPConnector(limit=3, use_dns_cache=False)

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []

        for cfg in PHASE1_CONFIGS:
            ai_name = cfg["ai_name"]
            prompt = build_phase1_unified_prompt(raw_packet_text, num_matches)
            temp = AI_TEMPERATURE.get(ai_name, 0.20)

            tasks.append(async_call_ai_once(
                session=session,
                prompt=prompt,
                url_env=cfg["url_env"],
                key_env=cfg["key_env"],
                models_list=cfg["models"],
                num_matches=num_matches,
                ai_name=ai_name,
                sys_prompt=sys_prompt,
                temperature=temp,
                phase="phase1",
            ))

        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    for res in raw_results:
        if isinstance(res, tuple):
            ai_name, results, status = res
            output[ai_name] = results
            AI_CALL_STATUS[ai_name] = status
            AI_RAW_STATUS[ai_name] = status
            print(f"  [状态] {ai_name.upper()} => {status} | 返回 {len(results)}/{num_matches} 场")
        else:
            print(f"  [Phase1异常] {res}")

    print("  [Phase 1 完成] 不做补跑，不做分场次 Repair")

    return output


async def run_phase2_claude_audit(
    raw_packet_text: str,
    phase1_results: Dict[str, Dict[int, Dict]],
    num_matches: int
) -> Tuple[Dict[int, Dict], str]:
    print(f"\n  [Phase 2] Claude 一次性终审 ({num_matches} 场)...")

    prompt = build_phase2_claude_prompt(raw_packet_text, phase1_results, num_matches)
    print(f"  [Claude Prompt] {len(prompt):,} 字符")

    sys_prompt = (
        "<role>你是竞彩足球最终审计 AI。你必须完整审计原始抓包和三家 AI 结论。</role>"
        "<instruction>严格输出 JSON 数组，禁止 markdown，禁止前缀后缀。</instruction>"
    )

    connector = aiohttp.TCPConnector(limit=1, use_dns_cache=False)

    async with aiohttp.ClientSession(connector=connector) as session:
        ai_name, results, model_name = await async_call_ai_once(
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
    AI_RAW_STATUS["claude"] = model_name

    print(f"  [Phase 2 完成] Claude 返回 {len(results)}/{num_matches} | 状态={model_name}")

    return results, f"claude:{model_name}"


# ====================================================================
# 覆盖率 / 兜底
# ====================================================================

def _is_valid_ai_result(r: Dict) -> bool:
    if not isinstance(r, dict) or not r:
        return False

    score = str(r.get("predicted_score", "")).strip()
    direction = _normalize_direction(r.get("predicted_direction", ""))

    if not score:
        return False

    if direction not in VALID_DIRS:
        return False

    return True


def _phase1_coverage_for_match(phase1_results: Dict[str, Dict[int, Dict]], idx: int) -> Dict[str, Any]:
    gpt_ok = _is_valid_ai_result(phase1_results.get("gpt", {}).get(idx, {}))
    grok_ok = _is_valid_ai_result(phase1_results.get("grok", {}).get(idx, {}))
    gemini_ok = _is_valid_ai_result(phase1_results.get("gemini", {}).get(idx, {}))

    valid_count = sum([gpt_ok, grok_ok, gemini_ok])

    return {
        "gpt": gpt_ok,
        "grok": grok_ok,
        "gemini": gemini_ok,
        "valid_count": valid_count,
        "coverage_ok": valid_count >= 2,
        "coverage_full": valid_count == 3,
    }


def _build_consensus_meta(phase1_results: Dict[str, Dict[int, Dict]], idx: int) -> Dict:
    rows = []

    for name in ["gpt", "grok", "gemini"]:
        r = phase1_results.get(name, {}).get(idx, {})

        if not _is_valid_ai_result(r):
            rows.append({
                "ai": name,
                "valid": False,
                "score": "-",
                "direction": "-",
                "confidence": 0,
            })
            continue

        rows.append({
            "ai": name,
            "valid": True,
            "score": str(r.get("predicted_score", "")),
            "direction": _normalize_direction(r.get("predicted_direction", "")),
            "confidence": _i(r.get("confidence", 0), 0),
        })

    valid_rows = [x for x in rows if x["valid"]]

    score_counts = {}
    dir_counts = {}

    for x in valid_rows:
        score_counts[x["score"]] = score_counts.get(x["score"], 0) + 1
        dir_counts[x["direction"]] = dir_counts.get(x["direction"], 0) + 1

    same_score_count = max(score_counts.values()) if score_counts else 0
    same_direction_count = max(dir_counts.values()) if dir_counts else 0

    return {
        "gpt_score": rows[0]["score"],
        "grok_score": rows[1]["score"],
        "gemini_score": rows[2]["score"],
        "gpt_direction": rows[0]["direction"],
        "grok_direction": rows[1]["direction"],
        "gemini_direction": rows[2]["direction"],
        "same_score_count": same_score_count,
        "same_direction_count": same_direction_count,
        "unanimous_score": same_score_count == 3,
        "unanimous_direction": same_direction_count == 3,
        "valid_count": len(valid_rows),
    }


def _fallback_result_from_phase1(phase1_results: Dict[str, Dict[int, Dict]], idx: int) -> Dict:
    consensus = _build_consensus_meta(phase1_results, idx)

    valid = []

    for name in ["gpt", "grok", "gemini"]:
        r = phase1_results.get(name, {}).get(idx, {})
        if _is_valid_ai_result(r):
            valid.append((name, r))

    if not valid:
        return {
            "match": idx,
            "predicted_score": "",
            "predicted_direction": "",
            "result": "未知",
            "goal_range": "?",
            "expected_total_goals": 0,
            "over_25_pct": 0,
            "btts_pct": 0,
            "home_win_pct": 33.3,
            "draw_pct": 33.3,
            "away_win_pct": 33.4,
            "confidence": 1,
            "top3": [],
            "analyst_consensus": consensus,
            "override_unanimous": False,
            "adopted_analysts": [],
            "rejected_analysts": [],
            "audit_result": "Claude 未返回有效终审，且 GPT/Grok/Gemini 均未返回有效分析。本场不应视为有效预测。",
            "arbitration_reason": "Claude 未返回有效终审，三家 Phase1 也无有效结果。本地没有使用算法替代 AI 判断，因此只输出失败状态。",
            "doubts": ["四家 AI 未形成有效输出"],
            "key_evidence": [],
            "web_research": {"searched": False, "source_notes": ["AI调用失败"]},
            "ai_validation_errors": ["all_ai_failed"],
        }

    score_groups = {}

    for name, r in valid:
        sc = str(r.get("predicted_score", "")).strip()
        score_groups.setdefault(sc, []).append((name, r))

    best_score = None
    best_group = []

    for sc, group in score_groups.items():
        if len(group) > len(best_group):
            best_score = sc
            best_group = group

    if len(best_group) >= 2:
        adopted = [x[0] for x in best_group]
        chosen = max(best_group, key=lambda x: _i(x[1].get("confidence", 0), 0))[1]
        reason = "Claude 未返回有效终审；Phase1 有至少两家比分一致，临时沿用二家一致结果。此为失败兜底，不是 Claude 终审。"
    else:
        adopted_name, chosen = max(valid, key=lambda x: _i(x[1].get("confidence", 0), 0))
        adopted = [adopted_name]
        reason = f"Claude 未返回有效终审；Phase1 无二家比分一致，临时沿用 {adopted_name.upper()} 最高信心结果。此为失败兜底，不是 Claude 终审。"

    out = dict(chosen)
    out["match"] = idx
    out["analyst_consensus"] = consensus
    out["override_unanimous"] = False
    out["adopted_analysts"] = adopted
    out["rejected_analysts"] = [n for n in ["gpt", "grok", "gemini"] if n not in adopted]
    out["audit_result"] = reason
    out["arbitration_reason"] = reason + " 原始 Claude 调用状态：" + str(AI_CALL_STATUS.get("claude", ""))
    out["confidence"] = min(_i(out.get("confidence", 40), 40), 45)
    out.setdefault("ai_validation_errors", [])
    out["ai_validation_errors"] = list(out.get("ai_validation_errors", [])) + ["claude_failed_phase1_fallback"]

    return out


# ====================================================================
# 输出包装
# ====================================================================

def _ai_score_summary(r: Dict) -> str:
    if not isinstance(r, dict) or not r:
        return "-"

    return str(r.get("predicted_score", "-"))


def _ai_reason_summary(r: Dict, empty_text: str) -> str:
    if not isinstance(r, dict) or not r:
        return empty_text

    return _safe_str(
        r.get("reason", r.get("arbitration_reason", r.get("audit_result", ""))),
        AI_ANALYSIS_MAX_CHARS
    )


def assemble_final_prediction(
    match: Dict,
    phase1_results: Dict[str, Dict[int, Dict]],
    claude_result: Dict,
    idx: int,
    ai_provider: str
) -> Dict:
    cr = dict(claude_result or {})

    score = str(cr.get("predicted_score", "")).strip()
    score_label, is_others = _score_to_label(score)

    direction = _normalize_direction(cr.get("predicted_direction", ""))
    score_dir = _score_direction(score_label)

    if not direction and score_dir:
        direction = score_dir

    result_cn = cr.get("result") or _direction_cn(direction)

    raw_goal_range = cr.get("goal_range") or _goal_range_from_score(score_label)
    goal_bucket, goal_label = _normalize_goal_range_for_ui(raw_goal_range, score_label)

    h_score, a_score = _parse_score(score_label)
    goal_count = h_score + a_score if h_score is not None else None

    hp, dp, ap = _pct_normalize(
        cr.get("home_win_pct", 33.3),
        cr.get("draw_pct", 33.3),
        cr.get("away_win_pct", 33.4),
    )

    confidence = _i(cr.get("confidence", 1), 1)
    confidence = max(1, min(99, confidence))

    coverage = _phase1_coverage_for_match(phase1_results, idx)

    p1_gpt = phase1_results.get("gpt", {}).get(idx, {})
    p1_grok = phase1_results.get("grok", {}).get(idx, {})
    p1_gemini = phase1_results.get("gemini", {}).get(idx, {})

    claude_reason = (
        cr.get("arbitration_reason")
        or cr.get("reason")
        or cr.get("audit_result")
        or ""
    )

    final_odds = _get_score_odds(match, score_label, direction, is_others)

    expected_total_goals = _f(cr.get("expected_total_goals", cr.get("exp_goals", 0)), 0)
    over_25_pct = _f(cr.get("over_25_pct", cr.get("over_2_5", cr.get("over25", 0))), 0)
    btts_pct = _f(cr.get("btts_pct", cr.get("btts", 0)), 0)

    risk = "低" if confidence >= 75 else ("中" if confidence >= 55 else "高")

    return {
        "predicted_score": score_label,
        "predicted_label": score_label,
        "predicted_direction": direction,
        "final_direction": direction,
        "result": result_cn,
        "display_direction": result_cn,
        "is_score_others": is_others,

        "decision_title": "vMAX 19.5 决策剖析",
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

        "expected_total_goals": expected_total_goals,
        "exp_goals": expected_total_goals,
        "over_25_pct": over_25_pct,
        "over_2_5": over_25_pct,
        "over25": over_25_pct,
        "btts_pct": btts_pct,
        "btts": btts_pct,

        "home_win_pct": hp,
        "draw_pct": dp,
        "away_win_pct": ap,

        "confidence": confidence,
        "ai_confidence": confidence,
        "ai_confidence_pct": confidence,
        "ai_confidence_score": confidence,
        "ai_confidence_percent": confidence,
        "ai_confidence_value": confidence,
        "aiConfidence": confidence,
        "aiConfidencePct": confidence,
        "confidence_score": confidence,
        "confidenceValue": confidence,
        "confidence_pct": confidence,
        "confidence_percent": confidence,
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

        "ai_call_status": dict(AI_CALL_STATUS),
        "ai_raw_status": dict(AI_RAW_STATUS),
        "gpt_status": AI_CALL_STATUS.get("gpt", ""),
        "grok_status": AI_CALL_STATUS.get("grok", ""),
        "gemini_status": AI_CALL_STATUS.get("gemini", ""),
        "claude_status": AI_CALL_STATUS.get("claude", ""),

        "ai_provider": ai_provider,
        "claude_score": score_label,
        "claude_analysis": _safe_str(claude_reason, CLAUDE_REASON_MAX_CHARS),
        "arbitration_reason": claude_reason,
        "audit_result": cr.get("audit_result", ""),
        "agreement_pattern": cr.get("agreement_pattern", "Claude终审复盘"),
        "analyst_consensus": cr.get("analyst_consensus", _build_consensus_meta(phase1_results, idx)),
        "override_unanimous": bool(cr.get("override_unanimous", False)),

        "phase1_coverage": coverage,
        "analysis_coverage": cr.get("analysis_coverage", coverage),
        "coverage_ok": coverage["coverage_ok"],
        "coverage_full": coverage["coverage_full"],

        "adopted_analysts": cr.get("adopted_analysts", []),
        "rejected_analysts": cr.get("rejected_analysts", []),

        "top3": cr.get("top3", []),
        "key_evidence": cr.get("key_evidence", cr.get("key_signals", [])),
        "doubts": cr.get("doubts", []),
        "data_quality": cr.get("data_quality", {}),
        "web_research": cr.get("web_research", {}),
        "ai_validation_errors": cr.get("ai_validation_errors", []),

        "gpt_score": _ai_score_summary(p1_gpt),
        "gpt_analysis": _ai_reason_summary(p1_gpt, "GPT 未返回有效分析。"),
        "gpt_doubts": p1_gpt.get("doubts", []) if p1_gpt else [],
        "gpt_key_evidence": p1_gpt.get("key_evidence", []) if p1_gpt else [],
        "gpt_web_research": p1_gpt.get("web_research", {}) if p1_gpt else {},

        "grok_score": _ai_score_summary(p1_grok),
        "grok_analysis": _ai_reason_summary(p1_grok, "GROK 未返回有效分析。"),
        "grok_doubts": p1_grok.get("doubts", []) if p1_grok else [],
        "grok_key_evidence": p1_grok.get("key_evidence", []) if p1_grok else [],
        "grok_web_research": p1_grok.get("web_research", {}) if p1_grok else {},

        "gemini_score": _ai_score_summary(p1_gemini),
        "gemini_analysis": _ai_reason_summary(p1_gemini, "GEMINI 未返回有效分析。"),
        "gemini_doubts": p1_gemini.get("doubts", []) if p1_gemini else [],
        "gemini_key_evidence": p1_gemini.get("key_evidence", []) if p1_gemini else [],
        "gemini_web_research": p1_gemini.get("web_research", {}) if p1_gemini else {},

        "ai_abstained": [
            n.upper()
            for n in ["gpt", "grok", "gemini"]
            if not phase1_results.get(n, {}).get(idx)
        ],

        "score_odds": final_odds,
        "raw_llm_score_prob": 0,
        "score_prob": 0,
        "suggested_kelly": 0,
        "edge_vs_market": 0,
        "is_value": False,
        "value_reason": "v19.5 不用本地算法计算 EV/Kelly；AI 可在 reason 中自行分析价值。",

        "trap_matrix_title": "RAW-ONLY 观察",
        "trap_matrix_subtitle": "本版不做本地陷阱判定，最终以 AI 抓包审计为准",
        "observation_signals": [],
        "trap_facts": [],
        "trap_count": 0,
    }


# ====================================================================
# Top4 推荐：修复 id=None 全部推荐 bug
# ====================================================================

def select_top4(preds):
    def _score(x):
        p = x.get("prediction", {}) or {}

        confidence = _f(p.get("confidence", 0))
        risk_penalty = 0

        if p.get("risk_level") == "高":
            risk_penalty += 8
        if p.get("ai_validation_errors"):
            risk_penalty += min(12, len(p.get("ai_validation_errors", [])) * 2)
        if p.get("ai_abstained"):
            risk_penalty += len(p.get("ai_abstained", [])) * 3
        if not p.get("coverage_ok", True):
            risk_penalty += 8
        if str(p.get("claude_status", "")).startswith(("http_", "read_timeout", "json_parse_failed", "empty")):
            risk_penalty += 10

        return confidence - risk_penalty

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
    raw = raw or {}
    raw_matches = raw.get("matches", [])
    num = len(raw_matches)

    for k in AI_CALL_STATUS:
        AI_CALL_STATUS[k] = ""

    for k in AI_RAW_STATUS:
        AI_RAW_STATUS[k] = ""

    print("\n" + "=" * 80)
    print(f"  [{ENGINE_VERSION}] {ENGINE_ARCHITECTURE} | {num} 场")
    print("=" * 80)

    debug_ai_config()

    normalized_matches = []
    match_analyses = []

    for i, raw_m in enumerate(raw_matches):
        m = normalize_match(raw_m)
        normalized_matches.append(m)

        match_analyses.append({
            "idx": i + 1,
            "raw_match": raw_m,
            "match": m,
        })

    raw_packet_text = build_full_raw_packet(raw, normalized_matches)

    print(f"\n[RAW Packet] 完整抓包字符数: {len(raw_packet_text):,}")

    phase1_results = {
        "gpt": {},
        "grok": {},
        "gemini": {},
    }

    claude_results = {}
    ai_provider = "no_ai"

    if use_ai and num > 0:
        async def _run_full_ai():
            p1 = await run_phase1_three(raw_packet_text, num)
            p2, provider = await run_phase2_claude_audit(raw_packet_text, p1, num)
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
                    logger.error(f"AI 四家矩阵执行崩溃: {e}")
                    phase1_results = {"gpt": {}, "grok": {}, "gemini": {}}
                    claude_results = {}
                    ai_provider = "ai_crashed"
        else:
            try:
                phase1_results, claude_results, ai_provider = asyncio.run(_run_full_ai())
            except Exception as e:
                logger.error(f"AI 四家矩阵执行崩溃: {e}")
                phase1_results = {"gpt": {}, "grok": {}, "gemini": {}}
                claude_results = {}
                ai_provider = "ai_crashed"

    res = []

    for ma in match_analyses:
        idx = ma["idx"]
        raw_m = ma["raw_match"]
        m = ma["match"]

        cr = claude_results.get(idx, {})

        if not cr:
            cr = _fallback_result_from_phase1(phase1_results, idx)

        mg = assemble_final_prediction(
            match=m,
            phase1_results=phase1_results,
            claude_result=cr,
            idx=idx,
            ai_provider=ai_provider,
        )

        combined = {**raw_m, **m, "prediction": mg}
        combined["_vm_idx"] = idx

        sync_keys = [
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
            "ai_confidence_percent",
            "ai_confidence_value",
            "aiConfidence",
            "aiConfidencePct",
            "confidence_score",
            "confidenceValue",
            "confidence_pct",
            "confidence_percent",
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

            "expected_total_goals",
            "exp_goals",
            "over_25_pct",
            "over_2_5",
            "over25",
            "btts_pct",
            "btts",

            "decision_title",
            "decision_engine_version",
            "decision_architecture",
            "engine_version",
            "engine_architecture",

            "ai_call_status",
            "ai_raw_status",
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
            "arbitration_reason",
            "audit_result",
            "analyst_consensus",
            "override_unanimous",

            "gpt_score",
            "gpt_analysis",
            "grok_score",
            "grok_analysis",
            "gemini_score",
            "gemini_analysis",

            "top3",
            "key_evidence",
            "doubts",
            "web_research",
            "data_quality",
            "ai_validation_errors",
        ]

        for k in sync_keys:
            combined[k] = mg.get(k)

        combined["engine_version"] = ENGINE_VERSION
        combined["decision_title"] = "vMAX 19.5 决策剖析"
        combined["decision_engine_version"] = ENGINE_VERSION
        combined["decision_architecture"] = ENGINE_ARCHITECTURE

        res.append(combined)

        err_tag = f" [校验{len(mg.get('ai_validation_errors', []))}]" if mg.get("ai_validation_errors") else ""
        abstain_tag = f" [缺席{','.join(mg.get('ai_abstained', []))}]" if mg.get("ai_abstained") else ""
        cov = mg.get("phase1_coverage", {})
        cov_tag = f" [覆盖{cov.get('valid_count', 0)}/3]"

        print(
            f"  [{idx}] {m.get('home_team', m.get('home', '?'))} vs "
            f"{m.get('away_team', m.get('guest', '?'))} => "
            f"{mg.get('result', '?')} ({mg.get('predicted_score', '-')}) | "
            f"CF: {mg.get('confidence', 0)}% | "
            f"Claude={AI_CALL_STATUS.get('claude', '')}"
            f"{cov_tag}{err_tag}{abstain_tag}"
        )

    t4 = select_top4(res)
    t4idx = {t.get("_vm_idx") for t in t4}

    for r in res:
        r["is_recommended"] = r.get("_vm_idx") in t4idx

    res.sort(key=lambda x: extract_num(x.get("match_num", "")))

    return res, t4


# ====================================================================
# 本地启动
# ====================================================================

if __name__ == "__main__":
    logger.info(f"{ENGINE_VERSION} 启动")
    print(f"✅ {ENGINE_VERSION} 加载完成")
    print(f"   架构: {ENGINE_ARCHITECTURE}")
    print("   Phase 1: GPT / Grok / Gemini 三家一次性完整分析")
    print("   Phase 1 Repair: 已关闭，不补跑，不重复消耗 token")
    print("   Phase 2: Claude 接收三家结论 + 完整原始抓包一次性终审")
    print("   规则: 三家一致时 Claude 默认沿用，除非有强反证")
    print("   调用层: OpenAI-compatible 默认 stream=True，规避中转 524")
    print("   本地算法: 不参与方向/比分裁决，只做 JSON 解析和 UI 字段兼容")
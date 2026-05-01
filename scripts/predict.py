# ====================================================================
# 🚀 vMAX 19.4 — RAW-ONLY 一次性四AI审计版
# --------------------------------------------------------------------
# 这版原则:
#   ✅ 抓包和原始数据一次性完整提交给 AI
#   ✅ GPT / Grok / Gemini 三家不分工，全部看同一份完整数据
#   ✅ 不使用本地市场概率核心限制 AI
#   ✅ 不使用本地比分矩阵覆盖 AI
#   ✅ 不做 Phase1 Repair，不逐场补跑，避免 Grok/Gemini 重复消耗 token
#   ✅ 每家 AI 只调用一次
#   ✅ Claude 接收完整抓包 + 三家完整结论做最终审计
#   ✅ Claude 不按票数机械裁决，但三家完全一致时不得无证据乱改比分
#   ✅ 本地只做 JSON 解析、字段兼容、展示包装，不篡改 Claude 最终比分
#   ✅ EV/Kelly 默认不使用 LLM 主观比分概率
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
# 安全导入：只保留兼容，不让本地模型决定最终结果
# ====================================================================

EnsemblePredictor = None
ExperienceEngine = None
predict_match = None
apply_experience_to_prediction = None
upgrade_ensemble_predict = None

try:
    from config import *
except Exception as e:
    logger.warning(f"config 导入异常: {e}")

try:
    from models import EnsemblePredictor
except Exception as e:
    logger.warning(f"models 导入异常: {e}")

try:
    from odds_engine import predict_match
except Exception as e:
    logger.warning(f"odds_engine 导入异常: {e}")

try:
    from league_intel import build_league_intelligence
except Exception as e:
    logger.warning(f"league_intel 导入异常: {e}")

try:
    from experience_rules import ExperienceEngine, apply_experience_to_prediction
except Exception as e:
    logger.warning(f"experience_rules 导入异常: {e}")

try:
    from advanced_models import upgrade_ensemble_predict
except Exception as e:
    logger.warning(f"advanced_models 导入异常: {e}")

try:
    from odds_history import apply_odds_history
except Exception:
    def apply_odds_history(m, mg):
        return mg

try:
    from quant_edge import apply_quant_edge
except Exception:
    def apply_quant_edge(m, mg):
        return mg

try:
    from wencai_intel import apply_wencai_intel
except Exception:
    def apply_wencai_intel(m, mg):
        return mg


try:
    ensemble = EnsemblePredictor() if EnsemblePredictor else None
except Exception as e:
    logger.warning(f"EnsemblePredictor 初始化失败: {e}")
    ensemble = None

try:
    exp_engine = ExperienceEngine() if ExperienceEngine else None
except Exception as e:
    logger.warning(f"ExperienceEngine 初始化失败: {e}")
    exp_engine = None


# ====================================================================
# 基础配置
# ====================================================================

ENGINE_VERSION = "vMAX 19.4"
ENGINE_ARCHITECTURE = "RAW-ONLY Single-Shot 3AI + Claude Audit"

# 关键开关：这版不让本地算法决定方向/比分
ENABLE_LOCAL_MARKET_CORE = False
ENABLE_PHASE1_REPAIR = False
ENABLE_GROK_REPAIR = False
ENABLE_LLM_VALUE_BET = False
APPLY_LEGACY_ENHANCERS = False

# 每家 AI 严格只尝试一次主通道
STRICT_SINGLE_CALL_PER_AI = True

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


def _safe_json(obj: Any, max_len: Optional[int] = None) -> str:
    try:
        text = json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        text = str(obj)
    if max_len and len(text) > max_len:
        return text[:max_len] + "\n...<TRUNCATED_BY_LOCAL_DISPLAY_ONLY>"
    return text


def _parse_score(s: str) -> Tuple[Optional[int], Optional[int]]:
    try:
        s_str = str(s).strip().replace(" ", "").replace("：", "-").replace(":", "-")

        if "胜" in s_str and "其他" in s_str:
            return 9, 0
        if "平" in s_str and "其他" in s_str:
            return 9, 9
        if "负" in s_str and "其他" in s_str:
            return 0, 9

        if s_str in ["主胜", "客胜", "平局", "home", "away", "draw", "", "待定"]:
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
    }.get(str(direction), "待定")


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


def _score_to_label(score: str) -> Tuple[str, bool]:
    s = str(score or "").strip()

    if "胜其他" in s or s == "9-0":
        return "胜其他", True
    if "平其他" in s or s == "9-9":
        return "平其他", True
    if "负其他" in s or s == "0-9":
        return "负其他", True

    return s, False


def _is_valid_score(score: str) -> bool:
    s = str(score or "").strip()

    if s in ["胜其他", "平其他", "负其他", "9-0", "9-9", "0-9"]:
        return True

    h, a = _parse_score(s)
    if h is None or a is None:
        return False

    return 0 <= h <= 20 and 0 <= a <= 20


def _extract_match_index(item: Dict) -> Optional[int]:
    try:
        return int(item.get("match"))
    except Exception:
        return None


def _stable_match_key(raw_m: Dict, idx: int) -> str:
    for k in ["id", "match_id", "matchId", "match_num", "match_no", "serial", "num"]:
        v = raw_m.get(k)
        if v not in [None, ""]:
            return str(v)

    home = raw_m.get("home_team") or raw_m.get("home") or raw_m.get("host") or ""
    away = raw_m.get("away_team") or raw_m.get("guest") or raw_m.get("away") or ""
    league = raw_m.get("league") or raw_m.get("cup") or ""
    return f"idx:{idx}|{league}|{home}|{away}"


# ====================================================================
# 环境变量 / API 配置
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


# ====================================================================
# Match 标准化：只做展示字段，不做赔率误吸附，不做本地决策
# ====================================================================

def normalize_match(raw_m: Dict) -> Dict:
    raw_m = raw_m or {}
    m = dict(raw_m)

    # 只做浅层兼容，不通过泛化 win/draw/lose 深搜吸赔率，避免字段误吸附
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
                if k not in m:
                    m[k] = v

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

    m["give_ball"] = (
        m.get("give_ball")
        if m.get("give_ball") not in [None, ""]
        else m.get("handicap", m.get("rq", m.get("let_ball", "0")))
    )

    m["_raw_packet_original"] = raw_m
    return m


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

    # 以下只做摘要展示，不做本地判断
    for k in [
        "win", "same", "lose",
        "sp_home", "sp_draw", "sp_away",
        "give_ball", "handicap",
        "a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7",
        "w10", "w20", "w21", "w30", "w31", "w32",
        "s00", "s11", "s22",
        "l01", "l02", "l12", "l03", "l13", "l23",
        "crs_win", "crs_same", "crs_lose",
    ]:
        if k in match and match.get(k) not in [None, ""]:
            lines.append(f"{k}: {match.get(k)}")

    points = match.get("points")
    if isinstance(points, dict):
        lines.append("【points】")
        lines.append(_safe_json(points))

    information = match.get("information")
    if isinstance(information, dict):
        lines.append("【information】")
        lines.append(_safe_json(information))

    return "\n".join(lines)


def build_full_raw_packet(raw: Dict, normalized_matches: List[Dict]) -> str:
    brief_blocks = []
    for i, m in enumerate(normalized_matches, 1):
        brief_blocks.append(build_raw_match_brief(i, m))

    raw_json = _safe_json(raw)

    p = ""
    p += "<match_brief_index>\n"
    p += "\n\n".join(brief_blocks)
    p += "\n</match_brief_index>\n\n"

    p += "<full_raw_packet_json>\n"
    p += raw_json
    p += "\n</full_raw_packet_json>\n"

    return p


# ====================================================================
# Prompt：Phase 1 三家不分工，一次性完整分析
# ====================================================================

def build_phase1_unified_prompt(raw_packet_text: str, num_matches: int) -> str:
    p = ""
    p += "<context>\n"
    p += "你是竞彩足球独立分析 AI。下面是一次性提交的完整抓包数据，包含所有比赛、赔率、盘口、比分赔率、总进球、基本面、伤停、资金信号等原始字段。\n"
    p += "你不需要分工，也不要等待其他模型。你要独立阅读完整抓包，自行计算、自行判断每场最终方向和比分。\n"
    p += "</context>\n\n"

    p += "<hard_rules>\n"
    p += "1. 必须分析全部比赛，不要漏场。\n"
    p += f"2. 本次共有 {num_matches} 场，输出 JSON 数组长度应为 {num_matches}。\n"
    p += "3. match 字段必须对应第几场，从 1 开始编号。\n"
    p += "4. 不要因为常见比分就机械输出 1-1；只有抓包证据支持平局小球时才可输出 1-1。\n"
    p += "5. predicted_score 必须和 predicted_direction 一致。\n"
    p += "6. top3[0] 必须等于 predicted_score。\n"
    p += "7. 必须基于完整抓包自主计算，不要只复述赔率最低项。\n"
    p += "8. 必须列出 key_evidence 和 doubts。\n"
    p += "9. 如果数据矛盾，可以降低 confidence，但不要漏场。\n"
    p += "10. 严格输出 JSON 数组，不要 markdown，不要代码块，不要前缀，不要后缀。\n"
    p += "</hard_rules>\n\n"

    p += "<analysis_requirements>\n"
    p += "你需要综合判断：\n"
    p += "- 欧赔主平客结构\n"
    p += "- 让球/盘口方向\n"
    p += "- 总进球 a0~a7\n"
    p += "- 精确比分 CRS\n"
    p += "- 半全场\n"
    p += "- 赔率变化\n"
    p += "- Sharp / Steam / 散户热度\n"
    p += "- 基本面、战意、伤停、赛程、杯赛属性\n"
    p += "- 各市场是否共振或互相矛盾\n"
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
    p += "    \"home_win_pct\": 52,\n"
    p += "    \"draw_pct\": 27,\n"
    p += "    \"away_win_pct\": 21,\n"
    p += "    \"confidence\": 68,\n"
    p += "    \"top3\": [\n"
    p += "      {\"score\":\"2-1\", \"prob\":17},\n"
    p += "      {\"score\":\"1-1\", \"prob\":13},\n"
    p += "      {\"score\":\"1-0\", \"prob\":12}\n"
    p += "    ],\n"
    p += "    \"key_evidence\": [\"证据1\", \"证据2\"],\n"
    p += "    \"doubts\": [\"反证1\", \"风险1\"],\n"
    p += "    \"data_quality\": {\"raw_complete\": true, \"odds_complete\": true, \"crs_complete\": true, \"notes\": []},\n"
    p += "    \"reason\": \"中文分析\"\n"
    p += "  }\n"
    p += "]\n"
    p += "</output_schema>\n"

    return p


# ====================================================================
# Prompt：Phase 2 Claude 一次性终审
# ====================================================================

def _phase1_result_for_prompt(ai_name: str, results: Dict[int, Dict], num_matches: int) -> str:
    p = ""
    p += f"\n<{ai_name}_full_result>\n"

    arr = []
    for i in range(1, num_matches + 1):
        r = results.get(i)
        if isinstance(r, dict):
            arr.append(r)
        else:
            arr.append({
                "match": i,
                "missing": True,
                "reason": f"{ai_name} 未返回本场有效 JSON",
            })

    p += _safe_json(arr)
    p += f"\n</{ai_name}_full_result>\n"
    return p


def build_phase2_claude_prompt(
    raw_packet_text: str,
    phase1_results: Dict[str, Dict[int, Dict]],
    num_matches: int
) -> str:
    p = ""
    p += "<context>\n"
    p += "你是最终审计 AI。GPT / Grok / Gemini 已经分别拿到同一份完整抓包，并且不分工完成独立分析。\n"
    p += "你现在拿到：完整原始抓包 + 三家完整结论。你需要输出最终预测。\n"
    p += "</context>\n\n"

    p += "<core_rules>\n"
    p += "1. 你必须重新审计完整抓包，不能只按票数机械裁决。\n"
    p += "2. 但如果 GPT/Grok/Gemini 三家的 predicted_score 和 predicted_direction 完全一致，你默认必须沿用一致结论。\n"
    p += "3. 三家完全一致时，只有在原始抓包存在非常明确的反证时，你才允许推翻；推翻必须写出具体反证字段和原因。\n"
    p += "4. 禁止出现三家都给 1-1，而你没有强反证就改成 2-1 的情况。\n"
    p += "5. 禁止为了显得独立而刻意改比分。\n"
    p += "6. 二对一时，按抓包证据质量裁决，不按人数裁决。\n"
    p += "7. 三家都缺失时，你才基于完整抓包独立判断，并降低 confidence。\n"
    p += "8. predicted_score 必须和 predicted_direction 一致。\n"
    p += "9. top3[0] 必须等于 predicted_score。\n"
    p += "10. 必须输出 analyst_consensus，说明三家是否一致。\n"
    p += "11. 必须输出 override_unanimous，如果三家一致但你推翻，则为 true，否则 false。\n"
    p += "12. 严格输出 JSON 数组，不要 markdown，不要代码块，不要前缀，不要后缀。\n"
    p += "</core_rules>\n\n"

    p += "<decision_policy>\n"
    p += "你的任务不是制造新答案，而是审计三家结论与原始抓包是否匹配。\n"
    p += "当三家一致且抓包没有强反证，最终比分应与三家一致。\n"
    p += "当三家分歧，优先选择与原始抓包多市场共振更强的一方。\n"
    p += "当比分很接近，例如 1-1 与 2-1，只能在总进球、CRS、让球、欧赔、基本面同时支持时才上调或下调。\n"
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
    p += "    \"key_evidence\": [\"原始抓包证据1\", \"原始抓包证据2\"],\n"
    p += "    \"doubts\": [\"风险1\", \"反证1\"],\n"
    p += "    \"audit_result\": \"终审结论摘要\",\n"
    p += "    \"arbitration_reason\": \"完整中文终审理由\"\n"
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
        "models": ["gpt-5.4"],
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


def _build_single_url_for_ai(ai_name: str, url_env: str) -> str:
    if ai_name == "gpt":
        return get_first_clean_env_url(GPT_URL_ALIASES, GPT_DEFAULT_URL)

    return get_clean_env_url(url_env)


def _get_key_for_ai(ai_name: str, key_env: str) -> str:
    if ai_name == "gpt":
        return get_first_clean_env_key(GPT_KEY_ALIASES, GPT_DEFAULT_KEY)

    return get_clean_env_key(key_env)


# ====================================================================
# AI 调用层：严格单次调用，不 fallback，不 repair
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
        "grok": 900,
        "gpt": 1200,
        "gemini": 1800,
    }
    READ_TIMEOUT = READ_TIMEOUT_MAP.get(ai_name, 900)

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
            "model": model_name,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
        }

    gw = url.split("/v1")[0][:80]
    print(f"  [🔌 单次调用] {ai_name.upper()} | {model_name[:48]} @ {gw}")
    print(f"  [Prompt] {ai_name.upper()} 字符数: {len(prompt):,}")

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
                print(f"    ⚠️ {ai_name.upper()} HTTP {r.status} | {elapsed}s | {text[:500]}")
                return ai_name, {}, status

            try:
                data = await r.json(content_type=None)
            except Exception as e:
                text = await r.text()
                print(f"    ⚠️ {ai_name.upper()} JSON响应读取失败: {str(e)[:160]} | raw={text[:300]}")
                return ai_name, {}, "response_json_error"

            raw_text = _extract_response_text(data, is_gem=is_gem)

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


def _extract_response_text(data, is_gem=False) -> str:
    pieces = []

    try:
        if is_gem:
            candidates = data.get("candidates", [])
            for cand in candidates:
                content = cand.get("content", {})
                parts = content.get("parts", [])
                for part in parts:
                    t = part.get("text")
                    if isinstance(t, str) and t.strip():
                        pieces.append(t.strip())

            return "\n".join(pieces).strip()

        # OpenAI-compatible Chat Completions
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            msg = choices[0].get("message", {})

            if isinstance(msg, dict):
                content_val = msg.get("content", "")

                if isinstance(content_val, str) and content_val.strip():
                    pieces.append(content_val.strip())

                elif isinstance(content_val, list):
                    for item in content_val:
                        if isinstance(item, dict):
                            for key in ["text", "content", "output_text"]:
                                t = item.get(key)
                                if isinstance(t, str) and t.strip():
                                    pieces.append(t.strip())

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
                        pieces.append(v.strip())

        # Responses API / xAI style / mixed proxy
        output = data.get("output")
        if isinstance(output, list):
            for block in output:
                if isinstance(block, dict):
                    content = block.get("content")
                    if isinstance(content, list):
                        for c in content:
                            if isinstance(c, dict):
                                for key in ["text", "content", "output_text"]:
                                    t = c.get(key)
                                    if isinstance(t, str) and t.strip():
                                        pieces.append(t.strip())
                    for key in ["text", "content", "output_text"]:
                        t = block.get(key)
                        if isinstance(t, str) and t.strip():
                            pieces.append(t.strip())
        elif isinstance(output, str) and output.strip():
            pieces.append(output.strip())

        # Claude native style
        content = data.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") in ["text", "output_text"] and isinstance(block.get("text"), str):
                        pieces.append(block["text"].strip())
                    elif isinstance(block.get("text"), str):
                        pieces.append(block["text"].strip())
        elif isinstance(content, str) and content.strip():
            pieces.append(content.strip())

        raw_text = "\n".join([p for p in pieces if p]).strip()

        if raw_text:
            return raw_text

        # 最后兜底：从完整 JSON 中找数组
        full_str = json.dumps(data, ensure_ascii=False)
        m_match = re.search(r'\[\s*\{\s*\\?"match\\?"', full_str)
        if m_match:
            start_pos = m_match.start()
            depth = 0
            in_str = False
            escape = False
            end_pos = start_pos

            for ci in range(start_pos, len(full_str)):
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
                return extracted.strip()

    except Exception as ex:
        print(f"    ⚠️ 响应提取异常: {str(ex)[:120]}")

    return ""


# ====================================================================
# JSON 解析与轻校验：不静默改写比分
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
                # 尽量修闭合，但不改写具体内容
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
        out["match"] = int(out.get("match"))
    except Exception:
        errors.append("match_id_invalid")
        out["match"] = None
        return out, errors

    score = str(out.get("predicted_score", "")).strip()
    direction = str(out.get("predicted_direction", "")).strip().lower()

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
        # 这里只做缺失填充，不做反向覆盖
        out["predicted_direction"] = score_dir
        direction = score_dir

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

    raw_goal_range = out.get("goal_range") or _goal_range_from_score(score)
    bucket, label = _normalize_goal_range_for_ui(raw_goal_range, score)
    out["goal_range"] = bucket
    out["goal_range_label"] = label

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

    if phase == "claude":
        if not out.get("arbitration_reason"):
            out["arbitration_reason"] = out.get("reason", "")
        if not out.get("audit_result"):
            out["audit_result"] = out.get("reason", "")

    return out, errors


# ====================================================================
# Phase 1 / Phase 2 执行
# ====================================================================

async def run_phase1_three_once(raw_packet_text: str, num_matches: int) -> Dict[str, Dict[int, Dict]]:
    print(f"\n  [Phase 1] GPT/Grok/Gemini 三家一次性完整抓包分析 ({num_matches} 场)...")
    print("  [规则] 不分工、不分段、不逐场、不 Repair，每家只调用一次")

    sys_prompt = (
        "<role>你是竞彩足球独立分析AI。</role>"
        "<instruction>你会收到完整抓包。请独立分析全部比赛，严格输出 JSON 数组，不要 markdown。</instruction>"
    )

    prompt = build_phase1_unified_prompt(raw_packet_text, num_matches)

    connector = aiohttp.TCPConnector(limit=10, use_dns_cache=False)

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []

        for cfg in PHASE1_CONFIGS:
            tasks.append(async_call_ai_once(
                session=session,
                prompt=prompt,
                url_env=cfg["url_env"],
                key_env=cfg["key_env"],
                models_list=cfg["models"],
                num_matches=num_matches,
                ai_name=cfg["ai_name"],
                sys_prompt=sys_prompt,
                temperature=0.20,
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

    print("  [Phase 1 完成] 无 Repair，无重复补跑")
    return output


async def run_phase2_claude_once(
    raw_packet_text: str,
    phase1_results: Dict[str, Dict[int, Dict]],
    num_matches: int
) -> Tuple[Dict[int, Dict], str]:
    print(f"\n  [Phase 2] Claude 一次性终审 ({num_matches} 场)...")

    prompt = build_phase2_claude_prompt(raw_packet_text, phase1_results, num_matches)
    print(f"  [Claude Prompt] {len(prompt):,} 字符")

    sys_prompt = (
        "<role>你是最终审计AI。</role>"
        "<instruction>你必须审计完整抓包和三家结论。三家完全一致时不得无证据改比分。严格输出 JSON 数组。</instruction>"
    )

    connector = aiohttp.TCPConnector(limit=5, use_dns_cache=False)

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

    print(f"  [Phase 2 完成] Claude 返回 {len(results)}/{num_matches} | 状态={model_name}")
    return results, f"claude:{model_name}"


# ====================================================================
# 三家一致性检查 / 兜底：不制造 1-1
# ====================================================================

def _get_ai_result(phase1_results: Dict[str, Dict[int, Dict]], name: str, idx: int) -> Dict:
    r = phase1_results.get(name, {}).get(idx, {})
    return r if isinstance(r, dict) else {}


def build_analyst_consensus(phase1_results: Dict[str, Dict[int, Dict]], idx: int) -> Dict[str, Any]:
    rows = {}

    for name in ["gpt", "grok", "gemini"]:
        r = _get_ai_result(phase1_results, name, idx)
        rows[name] = {
            "score": str(r.get("predicted_score", "")).strip() if r else "",
            "direction": str(r.get("predicted_direction", "")).strip() if r else "",
            "confidence": _i(r.get("confidence", 0), 0) if r else 0,
            "valid": bool(r and r.get("predicted_score")),
        }

    valid_scores = [v["score"] for v in rows.values() if v["valid"] and v["score"]]
    valid_dirs = [v["direction"] for v in rows.values() if v["valid"] and v["direction"]]

    score_counts = {}
    dir_counts = {}

    for s in valid_scores:
        score_counts[s] = score_counts.get(s, 0) + 1

    for d in valid_dirs:
        dir_counts[d] = dir_counts.get(d, 0) + 1

    best_score = max(score_counts, key=score_counts.get) if score_counts else ""
    best_dir = max(dir_counts, key=dir_counts.get) if dir_counts else ""

    return {
        "gpt_score": rows["gpt"]["score"],
        "grok_score": rows["grok"]["score"],
        "gemini_score": rows["gemini"]["score"],
        "gpt_direction": rows["gpt"]["direction"],
        "grok_direction": rows["grok"]["direction"],
        "gemini_direction": rows["gemini"]["direction"],
        "same_score_count": score_counts.get(best_score, 0) if best_score else 0,
        "same_direction_count": dir_counts.get(best_dir, 0) if best_dir else 0,
        "best_score": best_score,
        "best_direction": best_dir,
        "unanimous_score": len(valid_scores) == 3 and len(set(valid_scores)) == 1,
        "unanimous_direction": len(valid_dirs) == 3 and len(set(valid_dirs)) == 1,
        "valid_count": sum(1 for v in rows.values() if v["valid"]),
    }


def _fallback_result_from_phase1(phase1_results: Dict[str, Dict[int, Dict]], idx: int) -> Dict:
    consensus = build_analyst_consensus(phase1_results, idx)

    # 三家完全一致：兜底沿用一致结果
    if consensus["unanimous_score"]:
        for name in ["gpt", "grok", "gemini"]:
            r = _get_ai_result(phase1_results, name, idx)
            if r and r.get("predicted_score") == consensus["best_score"]:
                out = dict(r)
                out["agreement_pattern"] = "Claude失败，三家AI比分一致，沿用三家一致结果"
                out["analyst_consensus"] = consensus
                out["override_unanimous"] = False
                out["adopted_analysts"] = ["gpt", "grok", "gemini"]
                out["rejected_analysts"] = []
                out["audit_result"] = "Claude 未返回有效终审；三家 Phase1 比分和方向一致，临时沿用一致结果。"
                out["arbitration_reason"] = out["audit_result"]
                out["confidence"] = min(_i(out.get("confidence", 50), 50), 68)
                return out

    # 二家一致：沿用二家一致，不制造新比分
    if consensus["same_score_count"] >= 2 and consensus["best_score"]:
        best_score = consensus["best_score"]
        adopted = []
        base = {}

        for name in ["gpt", "grok", "gemini"]:
            r = _get_ai_result(phase1_results, name, idx)
            if r and r.get("predicted_score") == best_score:
                adopted.append(name)
                if not base:
                    base = dict(r)

        if base:
            base["agreement_pattern"] = f"Claude失败，{','.join(adopted).upper()} 比分一致，沿用二家一致结果"
            base["analyst_consensus"] = consensus
            base["override_unanimous"] = False
            base["adopted_analysts"] = adopted
            base["rejected_analysts"] = [x for x in ["gpt", "grok", "gemini"] if x not in adopted]
            base["audit_result"] = "Claude 未返回有效终审；Phase1 有二家比分一致，临时沿用二家一致结果。"
            base["arbitration_reason"] = base["audit_result"]
            base["confidence"] = min(_i(base.get("confidence", 45), 45), 60)
            return base

    # 都不一致：取最高 confidence 的 Phase1，但标记高风险
    candidates = []

    for name in ["gpt", "grok", "gemini"]:
        r = _get_ai_result(phase1_results, name, idx)
        if r and r.get("predicted_score"):
            candidates.append((name, _i(r.get("confidence", 0), 0), r))

    if candidates:
        candidates.sort(key=lambda x: x[1], reverse=True)
        name, _, r = candidates[0]
        out = dict(r)
        out["agreement_pattern"] = f"Claude失败，三家分歧，临时采用 {name.upper()} 最高信心输出"
        out["analyst_consensus"] = consensus
        out["override_unanimous"] = False
        out["adopted_analysts"] = [name]
        out["rejected_analysts"] = [x for x in ["gpt", "grok", "gemini"] if x != name]
        out["audit_result"] = "Claude 未返回有效终审；三家分歧，仅临时采用最高信心 Phase1，风险较高。"
        out["arbitration_reason"] = out["audit_result"]
        out["confidence"] = min(_i(out.get("confidence", 40), 40), 50)
        return out

    # 完全失败：不再假装 1-1
    return {
        "match": idx,
        "predicted_score": "",
        "predicted_direction": "",
        "result": "待审",
        "goal_range": "未知",
        "home_win_pct": 33.3,
        "draw_pct": 33.3,
        "away_win_pct": 33.4,
        "confidence": 1,
        "top3": [],
        "agreement_pattern": "全部AI失败",
        "analyst_consensus": consensus,
        "override_unanimous": False,
        "adopted_analysts": [],
        "rejected_analysts": [],
        "audit_result": "GPT/Grok/Gemini/Claude 均未返回有效结果。本场不输出伪比分。",
        "arbitration_reason": "全部AI失败，不再静默兜底为1-1，避免污染复盘。",
        "doubts": ["AI未返回有效结果"],
        "prediction_valid": False,
    }


# ====================================================================
# 输出包装：不覆盖 Claude 比分，只做兼容字段
# ====================================================================

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


def _extract_score_prob_from_result(ai_result: Dict, score: str) -> float:
    candidates = []

    for key in ["top3", "score_probs", "scores"]:
        arr = ai_result.get(key, [])
        if isinstance(arr, list):
            candidates.extend(arr)

    for item in candidates:
        if not isinstance(item, dict):
            continue

        if str(item.get("score", "")).strip() == str(score).strip():
            p = _f(item.get("prob", 0))
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


def _ai_score_summary(r: Dict) -> str:
    if not isinstance(r, dict) or not r:
        return "-"
    return str(r.get("predicted_score", "-"))


def _ai_reason_summary(r: Dict, empty_text: str) -> str:
    if not isinstance(r, dict) or not r:
        return empty_text
    return _safe_str(r.get("reason", r.get("arbitration_reason", "")), 3000)


def assemble_final_prediction(
    match: Dict,
    phase1_results: Dict[str, Dict[int, Dict]],
    claude_result: Dict,
    idx: int,
    ai_provider: str
) -> Dict:
    cr = dict(claude_result or {})

    predicted_score = str(cr.get("predicted_score", "")).strip()
    predicted_label, is_others = _score_to_label(predicted_score)
    final_direction = str(cr.get("predicted_direction", "")).strip().lower()

    # 不覆盖 Claude，只在 direction 缺失时用比分补字段
    score_dir = _score_direction(predicted_score)
    if not final_direction and score_dir:
        final_direction = score_dir

    result_cn = cr.get("result") or _direction_cn(final_direction)

    h_score, a_score = _parse_score(predicted_score)
    goal_count = h_score + a_score if h_score is not None else None

    raw_goal_range = cr.get("goal_range") or _goal_range_from_score(predicted_score)
    goal_bucket, goal_label = _normalize_goal_range_for_ui(raw_goal_range, predicted_score)

    home_pct, draw_pct, away_pct = _pct_normalize(
        cr.get("home_win_pct", 33.3),
        cr.get("draw_pct", 33.3),
        cr.get("away_win_pct", 33.4),
    )

    confidence = _i(cr.get("confidence", 1), 1)
    confidence = max(1, min(99, confidence))

    risk = "低" if confidence >= 75 else ("中" if confidence >= 55 else "高")

    consensus = cr.get("analyst_consensus")
    if not isinstance(consensus, dict):
        consensus = build_analyst_consensus(phase1_results, idx)

    p1_gpt = _get_ai_result(phase1_results, "gpt", idx)
    p1_grok = _get_ai_result(phase1_results, "grok", idx)
    p1_gemini = _get_ai_result(phase1_results, "gemini", idx)

    claude_reason = (
        cr.get("arbitration_reason")
        or cr.get("reason")
        or cr.get("audit_result")
        or ""
    )

    top3 = cr.get("top3", [])
    if not isinstance(top3, list):
        top3 = []

    final_odds = _get_score_odds(match, predicted_score, final_direction, is_others)
    raw_score_prob = _extract_score_prob_from_result(cr, predicted_score)

    if ENABLE_LLM_VALUE_BET:
        score_prob = raw_score_prob
        ev_pct, kelly_pct, is_value = _calculate_score_ev(score_prob, final_odds)
        value_reason = "基于 Claude top3 精确比分概率计算，未做历史校准"
    else:
        score_prob = raw_score_prob
        ev_pct, kelly_pct, is_value = 0.0, 0.0, False
        value_reason = "v19.4 默认不使用 LLM 主观比分概率计算正式EV/Kelly"

    validation_errors = cr.get("ai_validation_errors", [])
    if not isinstance(validation_errors, list):
        validation_errors = []

    prediction_valid = bool(predicted_score and final_direction in VALID_DIRS)

    out = {
        "predicted_score": predicted_label,
        "predicted_label": predicted_label,
        "predicted_direction": final_direction,
        "result": result_cn,
        "display_direction": result_cn,
        "final_direction": final_direction,
        "is_score_others": is_others,
        "prediction_valid": prediction_valid,

        "decision_title": "vMAX 19.4 RAW-ONLY 一次性四AI审计",
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

        "ai_call_status": dict(AI_CALL_STATUS),
        "gpt_status": AI_CALL_STATUS.get("gpt", ""),
        "grok_status": AI_CALL_STATUS.get("grok", ""),
        "gemini_status": AI_CALL_STATUS.get("gemini", ""),
        "claude_status": AI_CALL_STATUS.get("claude", ""),

        "ai_provider": ai_provider,
        "claude_score": predicted_label,
        "claude_analysis": claude_reason[:5000],
        "arbitration_reason": claude_reason,
        "audit_result": cr.get("audit_result", ""),
        "agreement_pattern": cr.get("agreement_pattern", cr.get("audit_result", "Claude终审")),
        "analyst_consensus": consensus,
        "override_unanimous": bool(cr.get("override_unanimous", False)),

        "adopted_analysts": cr.get("adopted_analysts", []),
        "rejected_analysts": cr.get("rejected_analysts", []),
        "top3": top3,

        "gpt_score": _ai_score_summary(p1_gpt),
        "gpt_analysis": _ai_reason_summary(p1_gpt, "GPT 未返回有效分析。"),
        "gpt_doubts": p1_gpt.get("doubts", []) if p1_gpt else [],
        "gpt_key_evidence": p1_gpt.get("key_evidence", p1_gpt.get("key_signals", [])) if p1_gpt else [],

        "grok_score": _ai_score_summary(p1_grok),
        "grok_analysis": _ai_reason_summary(p1_grok, "GROK 未返回有效分析。"),
        "grok_doubts": p1_grok.get("doubts", []) if p1_grok else [],
        "grok_key_evidence": p1_grok.get("key_evidence", p1_grok.get("key_signals", [])) if p1_grok else [],

        "gemini_score": _ai_score_summary(p1_gemini),
        "gemini_analysis": _ai_reason_summary(p1_gemini, "GEMINI 未返回有效分析。"),
        "gemini_doubts": p1_gemini.get("doubts", []) if p1_gemini else [],
        "gemini_key_evidence": p1_gemini.get("key_evidence", p1_gemini.get("key_signals", [])) if p1_gemini else [],

        "ai_abstained": [
            n.upper()
            for n in ["gpt", "grok", "gemini"]
            if not phase1_results.get(n, {}).get(idx)
        ],

        "accepted_observations": cr.get("accepted_observations", []),
        "rejected_observations": cr.get("rejected_observations", []),
        "key_evidence": cr.get("key_evidence", cr.get("key_signals", [])),
        "doubts": cr.get("doubts", []),
        "data_quality": cr.get("data_quality", {}),
        "ai_validation_errors": validation_errors,

        "score_odds": final_odds,
        "raw_llm_score_prob": round(raw_score_prob * 100, 2),
        "score_prob": round(score_prob * 100, 2),
        "suggested_kelly": kelly_pct,
        "edge_vs_market": ev_pct,
        "is_value": is_value,
        "value_reason": value_reason,
    }

    return out


# ====================================================================
# Top4 推荐：修复 id=None 全部推荐 bug
# ====================================================================

def select_top4(preds):
    def _score(x):
        p = x.get("prediction", {}) or {}

        confidence = _f(p.get("confidence", 0))
        risk_penalty = 0

        if not p.get("prediction_valid", True):
            risk_penalty += 20
        if p.get("risk_level") == "高":
            risk_penalty += 8
        if p.get("ai_validation_errors"):
            risk_penalty += 4
        if p.get("ai_abstained"):
            risk_penalty += len(p.get("ai_abstained", [])) * 2
        if p.get("override_unanimous"):
            risk_penalty += 6

        consensus = p.get("analyst_consensus", {})
        if isinstance(consensus, dict):
            if consensus.get("unanimous_score"):
                risk_penalty -= 5
            elif consensus.get("same_score_count", 0) >= 2:
                risk_penalty -= 2

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
    if isinstance(raw, list):
        raw = {"matches": raw}

    raw = raw or {}
    raw_matches = raw.get("matches", [])
    num = len(raw_matches)

    for k in AI_CALL_STATUS:
        AI_CALL_STATUS[k] = ""

    print("\n" + "=" * 88)
    print(f"  [{ENGINE_VERSION}] {ENGINE_ARCHITECTURE} | {num} 场")
    print("=" * 88)

    print("  规则:")
    print("   - GPT/Grok/Gemini 三家一次性读取完整抓包")
    print("   - 三家不分工，不分段，不逐场")
    print("   - 禁止 Phase1 Repair，避免重复跑 Grok / Gemini")
    print("   - Claude 一次性读取完整抓包 + 三家结论终审")
    print("   - 本地不使用概率核心覆盖 AI 比分")
    print("   - 本地不静默兜底 1-1")
    print("=" * 88)

    debug_ai_config()

    normalized_matches = []
    match_analyses = []

    for i, raw_m in enumerate(raw_matches, 1):
        m = normalize_match(raw_m)
        normalized_matches.append(m)

        # 兼容旧模块，但不作为最终决策
        eng = {}
        sp = {}
        exp_result = {}

        try:
            if predict_match:
                eng = predict_match(m)
        except Exception as e:
            logger.warning(f"predict_match 兼容调用失败，不影响RAW-ONLY主流程: {e}")
            eng = {}

        try:
            sp = ensemble.predict(m, {}) if ensemble else {}
        except Exception as e:
            logger.warning(f"ensemble.predict 兼容调用失败，不影响RAW-ONLY主流程: {e}")
            sp = {}

        try:
            exp_result = exp_engine.analyze(m) if exp_engine else {}
        except Exception as e:
            logger.warning(f"exp_engine.analyze 兼容调用失败，不影响RAW-ONLY主流程: {e}")
            exp_result = {}

        match_analyses.append({
            "raw_match": raw_m,
            "match": m,
            "engine": eng,
            "stats": sp,
            "experience": exp_result,
            "_stable_key": _stable_match_key(raw_m, i),
        })

    raw_packet_text = build_full_raw_packet(raw, normalized_matches)

    phase1_results = {
        "gpt": {},
        "grok": {},
        "gemini": {},
    }
    claude_results = {}
    ai_provider = "no_ai"

    if use_ai and num > 0:
        async def _run_full_ai_once():
            p1 = await run_phase1_three_once(raw_packet_text, num)
            p2, provider = await run_phase2_claude_once(raw_packet_text, p1, num)
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
                future = pool.submit(_run_in_thread, _run_full_ai_once())
                try:
                    phase1_results, claude_results, ai_provider = future.result()
                except Exception as e:
                    logger.error(f"AI 一次性四家矩阵执行崩溃: {e}")
                    phase1_results = {"gpt": {}, "grok": {}, "gemini": {}}
                    claude_results = {}
                    ai_provider = "ai_crashed"
        else:
            try:
                phase1_results, claude_results, ai_provider = asyncio.run(_run_full_ai_once())
            except Exception as e:
                logger.error(f"AI 一次性四家矩阵执行崩溃: {e}")
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
            cr = _fallback_result_from_phase1(phase1_results, idx)

        mg = assemble_final_prediction(
            match=m,
            phase1_results=phase1_results,
            claude_result=cr,
            idx=idx,
            ai_provider=ai_provider,
        )

        combined = {**raw_m, **m, "prediction": mg}
        combined["_stable_key"] = ma["_stable_key"]

        # 旧 UI 兼容字段同步
        sync_keys = [
            "predicted_score",
            "predicted_label",
            "predicted_direction",
            "result",
            "display_direction",
            "final_direction",
            "prediction_valid",

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

            "claude_score",
            "claude_analysis",
            "gpt_score",
            "gpt_analysis",
            "grok_score",
            "grok_analysis",
            "gemini_score",
            "gemini_analysis",

            "analyst_consensus",
            "override_unanimous",
            "agreement_pattern",
            "audit_result",
            "arbitration_reason",
        ]

        for k in sync_keys:
            combined[k] = mg.get(k)

        combined["engine_version"] = ENGINE_VERSION
        combined["decision_title"] = "vMAX 19.4 RAW-ONLY 一次性四AI审计"
        combined["decision_engine_version"] = ENGINE_VERSION
        combined["decision_architecture"] = ENGINE_ARCHITECTURE

        res.append(combined)

        consensus = mg.get("analyst_consensus", {}) or {}
        unanimous_tag = ""
        if consensus.get("unanimous_score"):
            unanimous_tag = " [三家比分一致]"
        elif consensus.get("same_score_count", 0) >= 2:
            unanimous_tag = " [两家比分一致]"

        override_tag = " [Claude推翻三家一致]" if mg.get("override_unanimous") else ""
        err_tag = f" [校验{len(mg.get('ai_validation_errors', []))}]" if mg.get("ai_validation_errors") else ""
        abstain_tag = f" [缺席{','.join(mg.get('ai_abstained', []))}]" if mg.get("ai_abstained") else ""

        print(
            f"  [{idx}] {m.get('home_team', m.get('home', '?'))} vs "
            f"{m.get('away_team', m.get('guest', '?'))} => "
            f"{mg.get('result', '待审')} ({mg.get('predicted_score', '')}) | "
            f"CF: {mg.get('confidence', 0)}% | "
            f"{mg.get('agreement_pattern', 'Claude终审')}"
            f"{unanimous_tag}{override_tag}{err_tag}{abstain_tag}"
        )

    t4 = select_top4(res)

    # 修复 id=None 全部推荐 bug：用稳定 key
    top_keys = {x.get("_stable_key") for x in t4 if x.get("_stable_key")}

    for r in res:
        r["is_recommended"] = bool(r.get("_stable_key") in top_keys)

    res.sort(key=lambda x: extract_num(x.get("match_num", "")))

    return res, t4


# ====================================================================
# 本地启动
# ====================================================================

if __name__ == "__main__":
    logger.info(f"{ENGINE_VERSION} 启动")
    print(f"✅ {ENGINE_VERSION} 加载完成")
    print(f"   架构: {ENGINE_ARCHITECTURE}")
    print("   Phase 1: GPT / Grok / Gemini 三家不分工，一次性读取完整抓包")
    print("   Phase 1 Repair: 已禁用，避免重复跑和 token 浪费")
    print("   Phase 2: Claude 一次性读取完整抓包 + 三家完整结论")
    print("   规则: 三家完全一致时，Claude 不得无抓包强反证乱改比分")
    print("   本地算法: 不使用本地概率核心限制 AI")
    print("   EV/Kelly: 默认不使用 LLM 主观比分概率")
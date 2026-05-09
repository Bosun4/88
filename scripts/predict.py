# -*- coding: utf-8 -*-
"""
vMAX 20.2.1-FULL 满血版后端
============================================================

定位：
- AI-NATIVE 三AI架构：GPT 初审 + Grok 初审 -> Gemini 终审裁判
- Claude 已移除
- 本地不做最终足球预测裁判，不篡改 AI 最终比分
- 本地只负责：
  1) 抓包字段标准化
  2) Prompt 构建
  3) API 调用编排
  4) JSON 解析/修复
  5) 字段闭环与前端兼容
  6) 推荐分层字段展示

重要约束：
- API URL / KEY / model / fallback / timeout / payload 结构保持原架构风格
- 如果你本地已经有稳定 API 调用层，只替换 Prompt 与 merge_result 即可
- 不要把本文件里的 API 配置理解成必须改你的密钥或网关

版本：
- engine_version: vMAX 20.2.1-FULL
"""

import json
import os
import re
import time
import math
import asyncio
import aiohttp
import logging
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional


# ====================================================================
# 日志与外部模块兼容
# ====================================================================

try:
    import structlog
    logger = structlog.get_logger()
except Exception:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)


try:
    from config import *  # noqa
except Exception as e:
    logger.warning(f"config 导入异常: {e}")


try:
    from models import EnsemblePredictor
except Exception as e:
    logger.warning(f"models.EnsemblePredictor 导入异常: {e}")

    class EnsemblePredictor:
        def predict(self, m, ctx=None):
            return {}


try:
    from odds_engine import predict_match
except Exception as e:
    logger.warning(f"odds_engine.predict_match 导入异常: {e}")

    def predict_match(m):
        return {}


try:
    from league_intel import build_league_intelligence
except Exception as e:
    logger.warning(f"league_intel.build_league_intelligence 导入异常: {e}")

    def build_league_intelligence(m):
        return {}, {}, {}, {}


try:
    from experience_rules import ExperienceEngine, apply_experience_to_prediction
except Exception as e:
    logger.warning(f"experience_rules 导入异常: {e}")

    class ExperienceEngine:
        def analyze(self, m):
            return {}

    def apply_experience_to_prediction(m, mg, exp_engine=None):
        return mg


try:
    from advanced_models import upgrade_ensemble_predict
except Exception as e:
    logger.warning(f"advanced_models.upgrade_ensemble_predict 导入异常: {e}")

    def upgrade_ensemble_predict(m, mg):
        return mg


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
    ensemble = EnsemblePredictor()
except Exception as e:
    logger.warning(f"EnsemblePredictor 初始化失败: {e}")
    ensemble = None


try:
    exp_engine = ExperienceEngine()
except Exception as e:
    logger.warning(f"ExperienceEngine 初始化失败: {e}")
    exp_engine = None


# ====================================================================
# 常量
# ====================================================================

ENGINE_VERSION = "vMAX 20.2.1-FULL"
VALID_DIRS = {"home", "draw", "away"}

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

CRS_FULL_MAP = {
    "1-0": "w10", "2-0": "w20", "2-1": "w21", "3-0": "w30", "3-1": "w31",
    "3-2": "w32", "4-0": "w40", "4-1": "w41", "4-2": "w42", "5-0": "w50",
    "5-1": "w51", "5-2": "w52",
    "0-0": "s00", "1-1": "s11", "2-2": "s22", "3-3": "s33",
    "0-1": "l01", "0-2": "l02", "1-2": "l12", "0-3": "l03", "1-3": "l13",
    "2-3": "l23", "0-4": "l04", "1-4": "l14", "2-4": "l24", "0-5": "l05",
    "1-5": "l15", "2-5": "l25",
}

HFTF_MAP = {
    "ss": "主/主", "sp": "主/平", "sf": "主/负",
    "ps": "平/主", "pp": "平/平", "pf": "平/负",
    "fs": "负/主", "fp": "负/平", "ff": "负/负",
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


# ====================================================================
# 通用工具
# ====================================================================

def _f(v, default=0.0):
    try:
        if v is None:
            return default
        s = str(v).strip()
        if s == "" or s.lower() in {"none", "null", "nan", "-", "?"}:
            return default
        return float(s)
    except Exception:
        return default


def _safe_str(v, default=""):
    if v is None:
        return default
    return str(v)


def _parse_score(s: str) -> Tuple[Optional[int], Optional[int]]:
    try:
        s = str(s).strip().replace(" ", "")
        s = s.replace("：", "-").replace(":", "-").replace("–", "-").replace("—", "-")
        if not s:
            return None, None
        if "胜" in s and "其他" in s:
            return 9, 0
        if "平" in s and "其他" in s:
            return 9, 9
        if "负" in s and "其他" in s:
            return 0, 9
        p = s.split("-")
        if len(p) != 2:
            return None, None
        return int(p[0]), int(p[1])
    except Exception:
        return None, None


def _score_direction(score: str) -> Optional[str]:
    h, a = _parse_score(score)
    if h is None:
        return None
    if h > a:
        return "home"
    if h < a:
        return "away"
    return "draw"


def _direction_cn(d: str) -> str:
    return {"home": "主胜", "draw": "平局", "away": "客胜"}.get(d, "平局")


def _normalize_pct(h: float, d: float, a: float) -> Tuple[float, float, float]:
    h, d, a = max(0, h), max(0, d), max(0, a)
    s = h + d + a
    if s <= 0:
        return 33.0, 34.0, 33.0
    h = h / s * 100
    d = d / s * 100
    a = 100 - h - d
    return round(h, 1), round(d, 1), round(a, 1)


def _extract_json_array(text: str) -> List[Dict[str, Any]]:
    if not text:
        return []

    clean = str(text).strip()
    clean = re.sub(r"<think(?:ing)?>.*?</think(?:ing)?>", "", clean, flags=re.DOTALL | re.IGNORECASE)
    clean = re.sub(r"```(?:json|JSON|python|PYTHON)?", "", clean).replace("```", "").strip()

    # 精准找 JSON array
    start = clean.find("[")
    end = clean.rfind("]") + 1
    if start < 0 or end <= start:
        return []

    raw = clean[start:end]

    try:
        arr = json.loads(raw)
        return arr if isinstance(arr, list) else []
    except Exception:
        pass

    # 断尾修复
    try:
        last_obj = raw.rfind("}")
        if last_obj > 0:
            arr = json.loads(raw[:last_obj + 1] + "]")
            return arr if isinstance(arr, list) else []
    except Exception:
        return []

    return []


def _deep_find_value(obj, aliases, skip_keys=None):
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


def normalize_match(raw_m: Dict[str, Any]) -> Dict[str, Any]:
    m = dict(raw_m or {})

    nested_keys = [
        "v2_odds_dict", "odds_dict", "odds", "v2", "odds_v2",
        "packet", "raw_odds", "data", "detail",
    ]
    for nk in nested_keys:
        if isinstance(m.get(nk), dict):
            m.update(m[nk])

    home = (
        m.get("home_team") or m.get("home") or m.get("host") or
        m.get("team_home") or m.get("homeName") or "Home"
    )
    away = (
        m.get("away_team") or m.get("guest") or m.get("away") or
        m.get("team_away") or m.get("awayName") or "Away"
    )
    m["home_team"] = home
    m["away_team"] = away
    m["home"] = home
    m["guest"] = away

    skip = {"vote", "change", "points", "information", "prediction", "stats", "smart_signals"}

    sp_home = m.get("sp_home")
    if sp_home is None:
        sp_home = _deep_find_value(m, ["win", "odds_win", "spf_sp3", "sp3", "胜"], skip)

    sp_draw = m.get("sp_draw")
    if sp_draw is None:
        sp_draw = _deep_find_value(m, ["draw", "same", "odds_draw", "spf_sp1", "sp1", "平"], skip)

    sp_away = m.get("sp_away")
    if sp_away is None:
        sp_away = _deep_find_value(m, ["lose", "away_win", "odds_lose", "spf_sp0", "sp0", "负"], skip)

    if sp_home is not None:
        m["sp_home"] = sp_home
        m["win"] = sp_home
    if sp_draw is not None:
        m["sp_draw"] = sp_draw
        m["same"] = sp_draw
    if sp_away is not None:
        m["sp_away"] = sp_away
        m["lose"] = sp_away

    if "give_ball" not in m:
        m["give_ball"] = m.get("handicap") or m.get("rq") or m.get("let_ball") or "0"

    if not isinstance(m.get("change"), dict):
        ch = {}
        for src_key, dst_key in [
            ("change_win", "win"), ("cw", "win"),
            ("change_same", "same"), ("cs", "same"),
            ("change_draw", "same"),
            ("change_lose", "lose"), ("cl", "lose"),
            ("change_away", "lose"),
        ]:
            if src_key in m:
                ch[dst_key] = m.get(src_key)
        m["change"] = ch

    return m


# ====================================================================
# AI 指令：满血版 Prompt
# ====================================================================

AI_NATIVE_COMMON_RULES = """
你正在分析中国竞彩/竞彩赔率/亚洲盘口/总进球/比分赔率构成的足球预测市场。
你的目标不是套模板比分，而是从市场定价结构中推断最合理的比分分布。

核心原则：
1. 禁止默认 1-1、2-1。选择 1-1 或 2-1 时，必须解释为什么不是 0-0、1-0、0-1、1-2、2-0、3-2。
2. 比分必须来自“方向 + 进球数 + 让球深度 + CRS比分赔率 + 联赛风格 + 球队状态”的联合判断。
3. 总进球赔率 a0-a7 是比分形态的强约束，不是附属信息。
4. CRS 正确比分赔率是比分落点的强约束，低赔比分、升降赔率、胜其他/负其他必须一起审计。
5. 让球盘决定比分差形态：平手/受让/半球/一球/球半/两球，对应完全不同的比分池。
6. 联赛风格只能作为先验，不得硬锁。意甲/法甲/西甲偏低节奏，德甲/荷甲偏开放，英超/英冠要结合赛程、体能、战意、对抗强度。
7. 如果具备联网能力，必须核验：伤停、首发/轮换、战意、保级/争冠、赛程密度、天气、盘口外部一致性。没有联网能力时必须明确输出 no_live_web_access，禁止伪造新闻和来源。
8. 只输出 JSON，不要 markdown，不要解释 JSON 外文本。
"""

ANCHOR_ANALYSIS_RULES = """
<anchor_rules>

一、总进球锚点 a0-a7：

1. 0球赔率低于常规基准时：
   - a0 ≤ 11：0-0 必须进入候选池。
   - 若同时 a1/a2 也偏低，说明小球压缩，优先考虑 0-0、1-0、0-1、1-1。
   - 如果联赛是意甲、法甲、西甲、英冠保级战、强强谨慎战，0-0/1-1权重上调。

2. 1球赔率：
   - a1 ≤ 5.95：1球小胜/小负很重要。
   - 常见落点：1-0、0-1。
   - 如果让球方热度过高但一球低赔，注意热门只赢一球或不胜。

3. 2球赔率：
   - a2 ≤ 3.6：2球是核心总进球区。
   - 常见落点：1-1、2-0、0-2。
   - 如果方向不清，1-1必须优先审计。
   - 如果强弱明显，2-0/0-2优先。

4. 3球赔率：
   - a3 ≤ 3.8：3球是核心总进球区。
   - 常见落点：2-1、1-2、3-0、0-3。
   - 半球盘、一球浅盘中，2-1/1-2常见。
   - 但如果 1-1 是最低正确比分，不能机械选 2-1。

5. 4球赔率：
   - a4 > 6.0 且 a5/a6/a7 也偏高：明显压制大球，4球以上降权，小比分优先。
   - a4 在 4.0-5.3 区间：4球有真实权重，2-2、3-1、1-3 必须进入候选池。
   - a4 低但 a5/a6 高：更像 3-1/2-2，而不是 3-2/4-2。
   - a4 高、a3低：优先 2-1/1-2/1-1，不要盲目 3-1。

6. 5球赔率：
   - a5 ≤ 8.0：高比分警报，3-2、2-3、4-1、1-4 必须审计。
   - 若 a5低但 a4也低：比赛可能开放。
   - 若 a5低但 1X2 强弱极端：胜其他/负其他需要审计。

7. 6球/7+赔率：
   - a6 ≤ 13 或 a7 ≤ 16：极端大比分/胜其他风险上升。
   - a7 ≤ 15：必须审计胜其他/负其他，而不是只选 3-1/4-1。
   - 但如果联赛低节奏、让球不深、CRS大比分赔率不支持，则不能强行大球。

二、CRS正确比分锚点：

1. 0-0 低于常规区间：
   - 如果 0-0 明显压低，说明机构不排除极低节奏。
   - 结合 a0/a1/a2、联赛风格、防守强度判断。
   - 不能忽略 0-0。

2. 1-1 是最低或接近最低比分：
   - 平局中枢强。
   - 若 1X2三项接近、让球浅、a2/a3低，1-1优先级高。
   - 若双方都有进球倾向但大球不强，1-1优先于 2-1。

3. 2-1 / 1-2 低赔：
   - 代表3球分胜负剧本。
   - 必须结合方向赔率、让球、半全场判断主客侧。
   - 如果 2-1 和 1-2赔率接近，说明方向不稳，不能给高置信。

4. 3-1 / 1-3 压低：
   - 代表强方胜出且总进球4球。
   - 若 a4不支持，3-1/1-3只能作为次选。
   - 若让球深且强方方向低赔，3-1成立度提高。

5. 3-2 / 2-3 压低：
   - 代表开放互爆。
   - 德甲、荷甲、杯赛强攻、落后追分局加权。
   - 意甲、法甲、保级战默认降权，除非 a5/CRS明确支持。

6. 胜其他/负其他：
   - 如果胜其他/负其他赔率显著低，说明尾部比分不可忽略。
   - 若 a6/a7也低，必须把胜其他/负其他作为风险标签。
   - 但不能直接作为最终比分，除非多市场共振。

三、让球盘比分形态：

1. 平手盘 / 平半：
   - 方向不稳。
   - 常见比分：0-0、1-1、1-0、0-1、2-1、1-2。
   - 若总进球低，0-0/1-1优先。
   - 若双方进攻开放，2-1/1-2。

2. 半球盘：
   - 强方小胜常见。
   - 常见比分：1-0、2-1。
   - 若 a3低且BTTS强，2-1。
   - 若 a1/a2低且BTTS弱，1-0。

3. 一球盘：
   - 常见比分：2-0、2-1、1-0。
   - 如果热门过热且平赔/1-1压低，防 1-1。
   - 如果弱方进球能力差，2-0优先。
   - 如果双方均有进球，2-1优先。

4. 球半盘：
   - 常见比分：2-0、3-1、3-0。
   - 若 a4低，3-1更合理。
   - 若小球强，2-0。
   - 若弱方完全低进攻，3-0。

5. 两球及以上：
   - 常见比分：2-0、3-0、3-1、4-1、4-2。
   - 若 a4/a5高，不要机械选 4-1/4-2。
   - 若 a3低，3-0/2-1/2-0更合理。
   - 若胜其他低且 a6/a7低，考虑尾部风险。

四、联赛/场景风格：

1. 意甲、法甲：
   - 默认低节奏，1-0、1-1、0-0、2-0、2-1优先。
   - 强队如尤文类风格，通常在3球内，除非市场强烈给出大球。
   - 低比分不是保守，而是市场结构常态。

2. 德甲、荷甲：
   - 开放度更高，BTTS和3球以上更常见。
   - 多特蒙德类比赛，2-1、3-1、3-2、2-2权重提高。
   - 但若 a4/a5很高，仍然必须降大球。

3. 英超：
   - 不能简单大球化。
   - 强强战、密集赛程、争冠压力可能转向 1-1、1-0、2-1。
   - 强弱悬殊时 2-0、3-0、3-1。
   - 中下游对抗要看BTTS、让球和CRS，不可模板化。

4. 英冠/英甲：
   - 身体对抗强，波动大。
   - 保级战、升级附加压力下，0-0、1-1、1-0权重上调。
   - 如果盘口显示开放，再考虑 2-1/1-2。

5. 杯赛/淘汰赛：
   - 强队轮换、弱队保守、首回合/次回合比分关系很重要。
   - 强队低赔不等于大胜。
   - 必须审计 1-0、2-0、1-1、0-0。

五、反模板规则：

如果最终选择 1-1：
必须说明：
- 1X2是否接近；
- 让球是否浅；
- a2/a3是否支持；
- CRS 1-1是否低赔；
- 为什么不是 0-0 或 2-1。

如果最终选择 2-1：
必须说明：
- a3是否支持；
- BTTS是否支持；
- 让球是否适合半球/一球；
- 为什么不是 1-1 或 1-0。

如果最终选择 0-0：
必须说明：
- a0是否被压低；
- a1/a2是否同向；
- 联赛/战意/风格是否支持；
- 为什么不是 1-1。

如果最终选择 0-2 / 1-2：
必须说明：
- 客胜赔率是否真实支持；
- 让球盘是否偏客；
- CRS客胜比分是否靠前；
- 是否存在主队虚热。

</anchor_rules>
"""

GPT_SYSTEM_PROMPT = AI_NATIVE_COMMON_RULES + """
<role>
你是足球市场结构量化分析师，负责从 1X2、让球、总进球、CRS正确比分、半全场中重建比分分布。
你不是最终裁判，你的任务是给 Gemini 终审提供严谨、可审计的初审判断。
</role>

<focus>
1. 识别 1X2 主平客的真实方向。
2. 识别让球盘对应的比分差。
3. 识别 a0-a7 总进球锚点。
4. 识别 CRS 正确比分低赔锚点。
5. 明确哪些比分只是模板，哪些比分有市场结构支持。
</focus>

<anti_template>
禁止无脑输出 1-1 或 2-1。
如果你输出 1-1/2-1，必须给出市场结构证据。
没有证据时，宁可输出方向不稳、比分低置信，也不能硬凑。
</anti_template>

<output_schema>
严格输出 JSON 数组，每场一个对象：
[
  {
    "match": 1,
    "score": "1-1",
    "final_direction": "draw",
    "top3": [
      {"score": "1-1", "prob": 16},
      {"score": "0-0", "prob": 12},
      {"score": "1-0", "prob": 10}
    ],
    "goal_band": 2,
    "btts": "yes",
    "direction_confidence": "weak|medium|strong",
    "score_confidence": "weak|medium|strong",
    "ai_confidence": 60,
    "market_interpretation": {
      "one_x_two": "解释1X2方向",
      "handicap": "解释让球深浅",
      "total_goals": "解释a0-a7锚点",
      "correct_score": "解释CRS比分赔率",
      "league_style": "解释联赛和球队风格"
    },
    "risk_flags": ["方向不稳", "1-1低赔", "大球不支持"],
    "reason": "150-300字，必须说明为什么选择该比分，以及为什么排除2-3个相邻比分"
  }
]
</output_schema>
"""

GROK_SYSTEM_PROMPT = AI_NATIVE_COMMON_RULES + """
<role>
你是足球情报与资金流审计模型，负责检查球队新闻、战意、伤停、赛程、轮换、盘口热度、公众投注方向与赔率变化是否一致。
你不是最终裁判，你的结论会交给 Gemini 终审。
</role>

<web_policy>
如果你具备联网能力：
1. 必须核验球队近期状态、伤停、首发可能性、赛程密度、战意、排名压力。
2. 必须核验是否有保级战、争冠战、德比、杯赛轮换、首回合/次回合关系。
3. 必须输出 sources，至少包含 source/title/published_at/summary。
4. 如果资料冲突，必须说明冲突。

如果你没有联网能力：
1. web_research.used 必须为 false。
2. failure_reason 写 no_live_web_access。
3. 禁止伪造新闻、来源、伤停。
</web_policy>

<focus>
1. 判断公众热度是否过度集中。
2. 判断赔率变化是否与热门方向一致或反向。
3. 判断 Sharp/Steam 是否真实存在。
4. 判断基本面是否被市场过度定价。
5. 判断比赛是否存在保级、争冠、轮换、杯赛保守、强强谨慎等场景。
</focus>

<output_schema>
严格输出 JSON 数组：
[
  {
    "match": 1,
    "score": "1-0",
    "final_direction": "home",
    "top3": [
      {"score": "1-0", "prob": 14},
      {"score": "1-1", "prob": 12},
      {"score": "2-0", "prob": 10}
    ],
    "goal_band": 1,
    "btts": "no",
    "public_money_direction": "home|draw|away|unclear",
    "sharp_money_direction": "home|draw|away|unclear",
    "market_heat": "low|medium|high|extreme",
    "direction_confidence": "weak|medium|strong",
    "score_confidence": "weak|medium|strong",
    "ai_confidence": 60,
    "web_research": {
      "used": false,
      "failure_reason": "no_live_web_access",
      "sources": [],
      "key_findings": [],
      "source_conflicts": []
    },
    "risk_flags": ["主队过热", "客队反向资金", "保级战谨慎"],
    "reason": "150-300字，必须说明情报/资金/战意如何影响比分"
  }
]
</output_schema>
"""

GEMINI_FINAL_JUDGE_SYSTEM_PROMPT = AI_NATIVE_COMMON_RULES + ANCHOR_ANALYSIS_RULES + """
<role>
你是 vMAX 的最终裁判模型。Claude 已被移除，现在你负责最终审计。
你会看到：
1. 原始抓包数据；
2. GPT 初审；
3. Grok 初审。

你不能按票数裁决，必须重新审计市场结构。
你的最终比分必须由 1X2 + 让球 + a0-a7 + CRS正确比分 + 联赛风格 + 球队情报共同支持。
</role>

<final_judge_rules>
1. 不许机械服从 GPT 或 Grok。
2. GPT/Grok一致时，也必须检查是否只是共同套模板。
3. GPT/Grok分歧时，优先选择市场结构证据更完整的一方。
4. 如果 0-0、1-1、1-0、0-1 被赔率锚点支持，不能因为“比分保守”而忽略。
5. 如果 4球赔率高于6且5/6/7+也高，必须压低 3-2、4-1、4-2、胜其他。
6. 如果 a5/a6/a7明显压低，必须审计 3-2、2-3、4-1、1-4、胜其他/负其他。
7. 让球盘必须参与比分差判断，不能只看胜平负。
8. 联赛风格必须参与进球区间判断。
9. 最终推荐等级不能只看信心，还要看结构一致性、风险、赔率是否过热。
10. 如果证据不足，必须降低 recommendation.tier，不得强推。
</final_judge_rules>

<score_selection_protocol>
你必须按以下顺序决策：

Step 1：判断方向
- 1X2谁是真实低赔方向？
- 是否存在热门过热？
- 让球是否支持这个方向？
- 赔率变化是否支持或反向？

Step 2：判断总进球区间
- a0-a7谁最低？
- 4球、5球、7+是否被压低或抬高？
- 总进球区间是 0-1、2、3、4、5+ 哪一档？

Step 3：判断比分池
- 根据方向 + goal_band 建立候选池。
- 平手/半球/一球/球半/两球对应不同候选池。
- 不能直接选最常见比分。

Step 4：审计 CRS
- 候选比分是否在 CRS 中有低赔支持？
- 0-0/1-1/2-1/1-2/3-1/3-2是否存在赔率异常？
- 胜其他/负其他是否提示尾部风险？

Step 5：联赛/球队风格校正
- 意甲/法甲/西甲低节奏倾向；
- 德甲/荷甲开放倾向；
- 英超/英冠结合强强/保级/赛程；
- 具体球队风格优先于联赛平均。

Step 6：最终排除法
最终比分必须说明为什么不是另外两个高相邻候选比分。
例如选 1-1，必须说明为什么不是 0-0 和 2-1。
例如选 2-1，必须说明为什么不是 1-1 和 1-0。
例如选 0-2，必须说明为什么不是 0-1 和 1-2。
</score_selection_protocol>

<recommendation_tier>
推荐层级定义：
S：方向强、比分形态强、总进球锚点强、CRS支持强、风险少。
A：方向强，比分形态中强，存在1个可控风险。
B：方向或比分有一项中等不稳，但仍有可解释市场结构。
C：方向/比分冲突较多，只能观察。
D：不推荐，市场结构冲突或证据不足。

注意：
- S级必须非常少。
- B级不是差，是“可看但不要重注”。
- 如果 1X2 接近、CRS多比分接近、总进球锚点不清，最高只能 B。
</recommendation_tier>

<output_schema>
严格输出 JSON 数组。字段必须完整：
[
  {
    "match": 1,
    "final_model": "gemini",
    "score": "1-1",
    "final_direction": "draw",
    "result": "平局",
    "top3": [
      {"score": "1-1", "prob": 16, "why": "1-1最低赔且a2/a3支持"},
      {"score": "0-0", "prob": 11, "why": "a0低于常规，低节奏备选"},
      {"score": "2-1", "prob": 10, "why": "3球备选但大球不足"}
    ],
    "home_win_pct": 30,
    "draw_pct": 38,
    "away_win_pct": 32,
    "goal_band": 2,
    "btts": "yes",
    "expected_total_goals": 2.1,
    "direction_stability": "weak|medium|strong",
    "score_stability": "weak|medium|strong",
    "ai_confidence": 60,
    "recommendation": {
      "tier": "S|A|B|C|D",
      "is_recommended": true,
      "top4_priority": ["方向稳", "比分锚点强"],
      "why_recommended": "推荐理由",
      "stake_style": "观察|小注|中低仓|禁止重仓"
    },
    "anchor_audit": {
      "one_x_two": "1X2解释",
      "handicap": "让球解释",
      "total_goals": "a0-a7解释",
      "correct_score": "CRS正确比分解释",
      "league_style": "联赛/球队风格解释",
      "exclusion": "为什么不是其他候选比分"
    },
    "market_interpretation": {
      "public_money_direction": "home|draw|away|unclear",
      "sharp_money_direction": "home|draw|away|unclear",
      "heat_risk": "low|medium|high",
      "external_market": "联网或外部核验摘要"
    },
    "web_research": {
      "used": false,
      "failure_reason": "no_live_web_access",
      "sources": [],
      "freshness_grade": "missing|low|medium|high",
      "key_findings": [],
      "source_conflicts": [],
      "validation_warnings": []
    },
    "risk_flags": ["1-1低赔", "方向不稳", "主队过热"],
    "reason": "200-400字。必须按方向、总进球、CRS、让球、联赛风格、排除法说明最终比分。"
  }
]
</output_schema>
"""


# ====================================================================
# Prompt 构建
# ====================================================================

def _format_match_packet(ma: Dict[str, Any], idx: int) -> str:
    m = normalize_match(ma.get("match", {}))
    eng = ma.get("engine", {}) or {}
    stats = ma.get("stats", {}) or {}

    home = m.get("home_team") or m.get("home") or "Home"
    away = m.get("away_team") or m.get("guest") or "Away"
    league = m.get("league") or m.get("cup") or ""
    match_num = m.get("match_num") or m.get("id") or ""

    out = []
    out.append(f'<match index="{idx}">')
    out.append(f"比赛: {league} {match_num} | {home} vs {away}")

    out.append("[1X2]")
    out.append(
        f"主胜={m.get('sp_home', m.get('win', ''))} | "
        f"平={m.get('sp_draw', m.get('same', ''))} | "
        f"客胜={m.get('sp_away', m.get('lose', ''))}"
    )

    out.append("[让球]")
    out.append(f"give_ball={m.get('give_ball', m.get('handicap', m.get('rq', '0')))}")

    change = m.get("change", {})
    if isinstance(change, dict) and change:
        out.append("[赔率变动]")
        out.append(
            f"主胜变化={change.get('win', '')} | "
            f"平局变化={change.get('same', '')} | "
            f"客胜变化={change.get('lose', '')}"
        )

    out.append("[总进球 a0-a7]")
    out.append(" | ".join([f"a{g}={m.get(f'a{g}', '')}" for g in range(8)]))

    out.append("[CRS 正确比分]")
    crs_lines = []
    for score, key in CRS_FULL_MAP.items():
        v = m.get(key)
        if v not in [None, "", 0, "0"]:
            crs_lines.append(f"{score}={v}")
    out.append(" | ".join(crs_lines) if crs_lines else "无CRS字段")

    out.append("[其他比分]")
    out.append(
        f"胜其他={m.get('crs_win', '')} | "
        f"平其他={m.get('crs_same', '')} | "
        f"负其他={m.get('crs_lose', '')}"
    )

    hftf = []
    for k, label in HFTF_MAP.items():
        if m.get(k) not in [None, "", 0, "0"]:
            hftf.append(f"{label}={m.get(k)}")
    if hftf:
        out.append("[半全场]")
        out.append(" | ".join(hftf))

    vote = m.get("vote", {})
    if isinstance(vote, dict) and vote:
        out.append("[公众热度]")
        out.append(f"主={vote.get('win', '')}% | 平={vote.get('same', '')}% | 客={vote.get('lose', '')}%")

    out.append("[原始辅助字段]")
    out.append(
        f"bookmaker_implied_home_xg={eng.get('bookmaker_implied_home_xg', '')} | "
        f"bookmaker_implied_away_xg={eng.get('bookmaker_implied_away_xg', '')}"
    )
    out.append(f"expected_total_goals={eng.get('expected_total_goals', eng.get('expected_goals', ''))}")

    smart_signals = stats.get("smart_signals", [])
    if smart_signals:
        out.append("[市场信号]")
        out.append(" | ".join([str(x) for x in smart_signals[:12]]))

    info = m.get("information", {})
    if isinstance(info, dict) and info:
        out.append("[情报字段]")
        for k, v in info.items():
            if v:
                out.append(f"{k}: {str(v)[:500].replace(chr(10), ' ')}")

    points = m.get("points", {})
    if isinstance(points, dict) and points:
        out.append("[球队/比赛要点]")
        for k, v in points.items():
            if v:
                out.append(f"{k}: {str(v)[:700].replace(chr(10), ' ')}")

    out.append("</match>")
    return "\n".join(out)


def build_phase1_prompt(match_analyses: List[Dict[str, Any]]) -> str:
    p = "<task>\n"
    p += "你是初审模型。请对每场比赛进行独立市场结构审计，输出比分预测 JSON。\n"
    p += "重点不是猜常见比分，而是从 1X2、让球、总进球、CRS正确比分、半全场、联赛风格、球队状态中推断比分分布。\n"
    p += "</task>\n\n"
    p += ANCHOR_ANALYSIS_RULES + "\n\n"
    p += "<matches>\n"
    for i, ma in enumerate(match_analyses, 1):
        p += _format_match_packet(ma, i) + "\n\n"
    p += "</matches>\n\n"
    p += "严格输出 JSON 数组。禁止 markdown。禁止 JSON 外文字。\n"
    return p


def build_gemini_final_prompt(match_analyses: List[Dict[str, Any]], phase1_results: Dict[str, Dict[int, Dict[str, Any]]]) -> str:
    p = "<task>\n"
    p += "你是 Gemini 终审裁判。请基于原始抓包 + GPT初审 + Grok初审，重新审计并给出最终比分。\n"
    p += "不能按票数裁决。必须使用市场结构排除法。\n"
    p += "</task>\n\n"
    p += ANCHOR_ANALYSIS_RULES + "\n\n"

    p += "<raw_match_packet>\n"
    p += "<matches>\n"
    for i, ma in enumerate(match_analyses, 1):
        p += _format_match_packet(ma, i) + "\n\n"
    p += "</matches>\n"
    p += "</raw_match_packet>\n\n"

    p += "<phase1_results>\n"
    for ai_name in ["gpt", "grok"]:
        p += f"<{ai_name}_audit>\n"
        results = phase1_results.get(ai_name, {}) or {}
        if not results:
            p += "无有效结果\n"
        else:
            for idx in range(1, len(match_analyses) + 1):
                r = results.get(idx, {})
                if not r:
                    p += f"[{idx}] 弃权或解析失败\n"
                    continue
                p += json.dumps({
                    "match": idx,
                    "score": r.get("ai_score") or r.get("score"),
                    "top3": r.get("top3", []),
                    "final_direction": r.get("final_direction", ""),
                    "goal_band": r.get("goal_band", ""),
                    "btts": r.get("btts", ""),
                    "confidence": r.get("ai_confidence", ""),
                    "reason": r.get("reason") or r.get("analysis", ""),
                    "risk_flags": r.get("risk_flags", []),
                    "market_interpretation": r.get("market_interpretation", {}),
                    "web_research": r.get("web_research", {}),
                }, ensure_ascii=False)
                p += "\n"
        p += f"</{ai_name}_audit>\n\n"
    p += "</phase1_results>\n\n"

    p += "<final_instruction>\n"
    p += "请重新审计每场比赛。最终输出必须是 JSON 数组。\n"
    p += "每场必须明确：最终比分、方向、top3、goal_band、BTTS、推荐层级、风险标签、为什么不是另外两个候选比分。\n"
    p += "如果选择 1-1 或 2-1，必须进行反模板解释。\n"
    p += "如果 a0低、a4高、联赛低节奏，必须认真审计 0-0/1-1。\n"
    p += "如果 a5/a7低，必须认真审计 3-2/2-3/胜其他/负其他。\n"
    p += "禁止输出 JSON 之外任何文字。\n"
    p += "</final_instruction>\n"
    return p


# ====================================================================
# API 调用层
# 说明：这层保持原 v20 思路。若你本地已有稳定 API 调用层，直接保留你的原函数。
# ====================================================================

FALLBACK_URLS = [
    None,
    "https://www.api522.pro/v1",
    "https://api522.pro/v1",
    "https://api521.pro/v1",
    "http://69.63.213.33:666/v1",
]

GPT_DEFAULT_URL = "https://ai.newapi.life/v1"
GPT_DEFAULT_KEY = ""


def get_clean_env_url(name, default=""):
    v = str(os.environ.get(name, globals().get(name, default))).strip(" \t\n\r\"'")
    match = re.search(r"(https?://[a-zA-Z0-9._:/?#=&%-]+)", v)
    return match.group(1) if match else v


def get_clean_env_key(name):
    return str(os.environ.get(name, globals().get(name, ""))).strip(" \t\n\r\"'")


def _mask_key(k: str) -> str:
    if not k:
        return ""
    if len(k) <= 8:
        return "***"
    return f"{k[:4]}...{k[-4:]}"


def debug_ai_config():
    for name, url_env, key_env in [
        ("GPT", "GPT_API_URL", "GPT_API_KEY"),
        ("GROK", "GROK_API_URL", "GROK_API_KEY"),
        ("GEMINI", "GEMINI_API_URL", "GEMINI_API_KEY"),
    ]:
        print(
            f"[AI CONFIG] {name}: url={get_clean_env_url(url_env)} key={_mask_key(get_clean_env_key(key_env))}"
        )


async def async_call_one_ai_batch(
    session,
    prompt,
    url_env,
    key_env,
    models_list,
    num_matches,
    ai_name,
    system_prompt,
):
    key = get_clean_env_key(key_env)
    if not key and ai_name == "gpt":
        key = GPT_DEFAULT_KEY

    if not key:
        return ai_name, {}, "no_key"

    primary_url = get_clean_env_url(url_env, GPT_DEFAULT_URL if ai_name == "gpt" else "")

    if ai_name == "gpt":
        urls = [primary_url or GPT_DEFAULT_URL]
    else:
        backup = [u for u in FALLBACK_URLS if u and u != primary_url][:1]
        urls = [primary_url] + backup

    CONNECT_TIMEOUT = 20
    READ_TIMEOUT_MAP = {
        "grok": 380,
        "gpt": 380,
        "gemini": 480,
    }
    READ_TIMEOUT = READ_TIMEOUT_MAP.get(ai_name, 360)

    temp_map = {"gpt": 0.16, "grok": 0.24, "gemini": 0.12}
    temperature = temp_map.get(ai_name, 0.16)

    for mn in models_list:
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
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": temperature},
                    "systemInstruction": {"parts": [{"text": system_prompt}]},
                }
            else:
                headers["Authorization"] = f"Bearer {key}"
                payload = {
                    "model": mn,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": temperature,
                }

            gw = url.split("/v1")[0][:45]
            print(f"  [连接中] {ai_name.upper()} | {mn[:32]} @ {gw}")

            t0 = time.time()

            try:
                timeout = aiohttp.ClientTimeout(
                    total=None,
                    connect=CONNECT_TIMEOUT,
                    sock_connect=CONNECT_TIMEOUT,
                    sock_read=READ_TIMEOUT,
                )

                async with session.post(url, headers=headers, json=payload, timeout=timeout) as r:
                    elapsed_connect = round(time.time() - t0, 1)

                    if r.status in (502, 504):
                        print(f"    HTTP {r.status} | {elapsed_connect}s → 换URL")
                        continue

                    if r.status == 400:
                        print(f"    HTTP 400 | {elapsed_connect}s → 换模型")
                        break

                    if r.status == 429:
                        print(f"    HTTP 429 | {elapsed_connect}s → 换URL")
                        await asyncio.sleep(1)
                        continue

                    if r.status != 200:
                        print(f"    HTTP {r.status} | {elapsed_connect}s → 换URL")
                        continue

                    connected = True
                    print(f"    已连上 {elapsed_connect}s | 等待数据...")

                    try:
                        data = await r.json(content_type=None)
                    except Exception:
                        text = await r.text()
                        _save_debug_dump(ai_name, {"raw": text}, "non_json")
                        print("    响应非JSON → 换模型")
                        break

                    raw_text = _extract_response_text(data, is_gem)
                    if not raw_text or len(raw_text) < 10:
                        _save_debug_dump(ai_name, data, "empty")
                        print("    空数据 → 换模型")
                        break

                    results = _parse_ai_results(raw_text, num_matches)

                    elapsed = round(time.time() - t0, 1)
                    if results:
                        print(f"    {ai_name.upper()} 完成: {len(results)}/{num_matches} | {elapsed}s")
                        return ai_name, results, mn

                    print("    解析0条 → 换模型")
                    _save_debug_dump(ai_name, data, "parse0")
                    break

            except aiohttp.ClientConnectorError:
                print("    连接失败 → 换URL")
                continue
            except asyncio.TimeoutError:
                if not connected:
                    print("    连接超时 → 换URL")
                    continue
                print("    读取超时")
                return ai_name, {}, "read_timeout"
            except Exception as e:
                if not connected:
                    print(f"    {str(e)[:80]} → 换URL")
                    continue
                print(f"    调用异常: {str(e)[:120]}")
                return ai_name, {}, "error"

            await asyncio.sleep(0.2)

    return ai_name, {}, "all_failed"


def _extract_response_text(data: Dict[str, Any], is_gem: bool) -> str:
    raw_text = ""
    try:
        if is_gem:
            raw_text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        else:
            if data.get("choices"):
                msg = data["choices"][0].get("message", {})
                content_val = msg.get("content", "")

                if isinstance(content_val, str) and content_val.strip():
                    raw_text = content_val.strip()
                elif isinstance(content_val, list):
                    for item in content_val:
                        if isinstance(item, dict):
                            t = item.get("text") or item.get("content") or ""
                            if isinstance(t, str) and len(t) > len(raw_text):
                                raw_text = t.strip()

                if not raw_text:
                    for field in [
                        "text", "answer", "response", "output_text", "final_answer",
                        "output", "result", "completion", "message_content",
                        "assistant_content", "model_response",
                    ]:
                        v = msg.get(field, "")
                        if isinstance(v, str) and v.strip():
                            raw_text = v.strip()
                            break

        if not raw_text:
            full_str = json.dumps(data, ensure_ascii=False)
            m = re.search(r'\[\s*\{\s*\\?"match\\?"', full_str)
            if m:
                start = m.start()
                depth = 0
                end = start
                for i in range(start, min(start + 250000, len(full_str))):
                    if full_str[i] == "[":
                        depth += 1
                    elif full_str[i] == "]":
                        depth -= 1
                        if depth == 0:
                            end = i + 1
                            break
                if end > start:
                    extracted = full_str[start:end]
                    if '\\"' in extracted:
                        try:
                            extracted = json.loads('"' + extracted + '"')
                        except Exception:
                            extracted = extracted.replace('\\"', '"')
                    raw_text = extracted
    except Exception as e:
        print(f"    响应文本抽取异常: {str(e)[:100]}")

    return raw_text or ""


def _parse_ai_results(raw_text: str, num_matches: int) -> Dict[int, Dict[str, Any]]:
    arr = _extract_json_array(raw_text)
    results: Dict[int, Dict[str, Any]] = {}

    for item in arr:
        if not isinstance(item, dict):
            continue

        try:
            mid = int(item.get("match"))
        except Exception:
            continue

        if mid < 1 or mid > max(num_matches, 9999):
            continue

        score = (
            item.get("score")
            or item.get("ai_score")
            or item.get("predicted_score")
            or ""
        )

        top3 = item.get("top3", [])
        if not score and isinstance(top3, list) and top3:
            first = top3[0]
            if isinstance(first, dict):
                score = first.get("score", "")
            else:
                score = str(first)

        score = str(score).replace(" ", "").strip()
        if _parse_score(score)[0] is None:
            continue

        final_direction = item.get("final_direction") or _score_direction(score) or "draw"

        results[mid] = {
            "ai_score": score,
            "score": score,
            "top3": top3 if isinstance(top3, list) else [],
            "final_direction": final_direction,
            "result": item.get("result") or _direction_cn(final_direction),
            "goal_band": item.get("goal_band", item.get("goal_range", "")),
            "btts": item.get("btts", item.get("btts_ai", "")),
            "expected_total_goals": item.get("expected_total_goals", ""),
            "direction_stability": item.get("direction_stability", item.get("direction_confidence", "")),
            "score_stability": item.get("score_stability", item.get("score_confidence", "")),
            "ai_confidence": int(_f(item.get("ai_confidence", item.get("confidence", 60)), 60)),
            "recommendation": item.get("recommendation", {}),
            "anchor_audit": item.get("anchor_audit", {}),
            "market_interpretation": item.get("market_interpretation", {}),
            "web_research": item.get("web_research", {}),
            "risk_flags": item.get("risk_flags", item.get("warnings", [])),
            "reason": str(item.get("reason", item.get("analysis", ""))).strip(),
        }

    return results


def _save_debug_dump(ai_name, data, tag):
    try:
        os.makedirs("data/debug", exist_ok=True)
        dump_file = f"data/debug/{ai_name}_{tag}_{int(time.time())}.json"
        with open(dump_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"    失败响应已保存: {dump_file}")
    except Exception:
        pass


async def run_ai_matrix_two_phase(match_analyses: List[Dict[str, Any]]) -> Dict[str, Dict[int, Dict[str, Any]]]:
    num = len(match_analyses)
    phase1_prompt = build_phase1_prompt(match_analyses)

    print(f"  [v20.2.1 Phase1] Prompt {len(phase1_prompt):,} 字符 → GPT/Grok 初审")

    phase1_configs = [
        ("grok", "GROK_API_URL", "GROK_API_KEY", ["熊猫-A-5-grok-4.2-fast-200w上下文"], GROK_SYSTEM_PROMPT),
        ("gpt", "GPT_API_URL", "GPT_API_KEY", ["gpt-5.5"], GPT_SYSTEM_PROMPT),
    ]

    all_results: Dict[str, Dict[int, Dict[str, Any]]] = {"gpt": {}, "grok": {}, "gemini": {}}

    connector = aiohttp.TCPConnector(limit=6, use_dns_cache=False)

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            async_call_one_ai_batch(session, phase1_prompt, u, k, models, num, name, sys_prompt)
            for name, u, k, models, sys_prompt in phase1_configs
        ]

        phase1_raw = await asyncio.gather(*tasks, return_exceptions=True)
        for res in phase1_raw:
            if isinstance(res, tuple):
                all_results[res[0]] = res[1] or {}
            else:
                print(f"  [Phase1 ERROR] {res}")

        ok1 = sum(1 for n in ["gpt", "grok"] if all_results.get(n))
        print(f"  [Phase1完成] {ok1}/2 初审有数据")

        final_prompt = build_gemini_final_prompt(match_analyses, all_results)
        print(f"  [v20.2.1 Final] Prompt {len(final_prompt):,} 字符 → Gemini 终审裁判")

        gemini_name, gemini_result, gemini_model = await async_call_one_ai_batch(
            session=session,
            prompt=final_prompt,
            url_env="GEMINI_API_URL",
            key_env="GEMINI_API_KEY",
            models_list=["熊猫特供-按量-SSS-gemini-3.1-pro-preview-thinking"],
            num_matches=num,
            ai_name="gemini",
            system_prompt=GEMINI_FINAL_JUDGE_SYSTEM_PROMPT,
        )

        all_results["gemini"] = gemini_result or {}

    ok = sum(1 for v in all_results.values() if v)
    print(f"  [AI完成] {ok}/3 AI有数据 | 架构=GPT/Grok初审 + Gemini终审")
    return all_results


# ====================================================================
# 合并与字段闭环
# ====================================================================

def _fallback_from_phase1(gpt_r: Dict[str, Any], grok_r: Dict[str, Any]) -> Dict[str, Any]:
    valid = []
    for name, r in [("gpt", gpt_r), ("grok", grok_r)]:
        if isinstance(r, dict) and _parse_score(r.get("ai_score", ""))[0] is not None:
            valid.append((name, r))

    if not valid:
        return {}

    # 两家一致优先；否则选 ai_confidence 高的一家
    if len(valid) == 2 and valid[0][1].get("ai_score") == valid[1][1].get("ai_score"):
        return valid[0][1]

    return sorted(valid, key=lambda x: _f(x[1].get("ai_confidence", 0)), reverse=True)[0][1]


def _tier_from_recommendation(rec: Any, ai_conf: float, direction_stability: str, score_stability: str) -> str:
    if isinstance(rec, dict):
        tier = str(rec.get("tier", "")).upper()
        if tier in {"S", "A", "B", "C", "D"}:
            return tier

    ds = str(direction_stability or "").lower()
    ss = str(score_stability or "").lower()

    if ai_conf >= 82 and ds == "strong" and ss == "strong":
        return "S"
    if ai_conf >= 72 and ds in {"strong", "medium"} and ss in {"strong", "medium"}:
        return "A"
    if ai_conf >= 58:
        return "B"
    if ai_conf >= 45:
        return "C"
    return "D"


def _estimate_probs_from_direction(final_direction: str, ai_conf: float) -> Tuple[float, float, float]:
    ai_conf = max(35, min(88, _f(ai_conf, 60)))
    if final_direction == "home":
        h = ai_conf
        d = max(12, (100 - h) * 0.55)
        a = 100 - h - d
    elif final_direction == "away":
        a = ai_conf
        d = max(12, (100 - a) * 0.55)
        h = 100 - a - d
    else:
        d = max(32, min(48, ai_conf * 0.58))
        h = (100 - d) / 2
        a = 100 - d - h
    return _normalize_pct(h, d, a)


def merge_result(
    engine_result: Dict[str, Any],
    gpt_r: Dict[str, Any],
    grok_r: Dict[str, Any],
    gemini_r: Dict[str, Any],
    stats: Dict[str, Any],
    match_obj: Dict[str, Any],
) -> Dict[str, Any]:
    match_obj = normalize_match(match_obj)

    final_r = gemini_r if isinstance(gemini_r, dict) and _parse_score(gemini_r.get("ai_score", ""))[0] is not None else {}
    final_source = "gemini"

    if not final_r:
        final_r = _fallback_from_phase1(gpt_r, grok_r)
        final_source = "phase1_fallback"

    if not final_r:
        final_r = {
            "ai_score": "1-1",
            "final_direction": "draw",
            "ai_confidence": 45,
            "reason": "AI全失败，本地仅返回安全兜底字段；本场不推荐。",
            "recommendation": {"tier": "D", "is_recommended": False, "stake_style": "观察"},
            "risk_flags": ["all_ai_failed"],
        }
        final_source = "all_ai_failed_fallback"

    predicted_score = str(final_r.get("ai_score") or final_r.get("score") or "1-1").replace(" ", "").strip()
    h, a = _parse_score(predicted_score)
    final_direction = final_r.get("final_direction") or _score_direction(predicted_score) or "draw"

    # 强制比分方向闭环
    if h is not None:
        final_direction = "home" if h > a else ("away" if h < a else "draw")

    result_cn = _direction_cn(final_direction)
    ai_conf = _f(final_r.get("ai_confidence", 60), 60)

    home_pct = _f(final_r.get("home_win_pct", 0), 0)
    draw_pct = _f(final_r.get("draw_pct", 0), 0)
    away_pct = _f(final_r.get("away_win_pct", 0), 0)
    if home_pct + draw_pct + away_pct <= 0:
        home_pct, draw_pct, away_pct = _estimate_probs_from_direction(final_direction, ai_conf)
    else:
        home_pct, draw_pct, away_pct = _normalize_pct(home_pct, draw_pct, away_pct)

    rec = final_r.get("recommendation", {})
    tier = _tier_from_recommendation(
        rec,
        ai_conf,
        final_r.get("direction_stability", final_r.get("direction_confidence", "")),
        final_r.get("score_stability", final_r.get("score_confidence", "")),
    )

    is_score_others = predicted_score in ALL_SCORE_OTHERS or "其他" in predicted_score
    predicted_label = predicted_score
    if is_score_others:
        predicted_label = {"home": "胜其他", "draw": "平其他", "away": "负其他"}.get(final_direction, predicted_score)

    top3 = final_r.get("top3", [])
    if not isinstance(top3, list):
        top3 = []

    risk_flags = final_r.get("risk_flags", [])
    if not isinstance(risk_flags, list):
        risk_flags = [str(risk_flags)]

    reason = str(final_r.get("reason", "")).strip()

    web_research = final_r.get("web_research", {})
    if not isinstance(web_research, dict):
        web_research = {"used": False, "failure_reason": "invalid_web_research_field"}

    anchor_audit = final_r.get("anchor_audit", {})
    if not isinstance(anchor_audit, dict):
        anchor_audit = {}

    market_interpretation = final_r.get("market_interpretation", {})
    if not isinstance(market_interpretation, dict):
        market_interpretation = {}

    goal_band = final_r.get("goal_band", "")
    if goal_band in [None, ""]:
        try:
            goal_band = h + a if h is not None else ""
        except Exception:
            goal_band = ""

    btts = final_r.get("btts", "")
    if not btts and h is not None:
        btts = "yes" if h > 0 and a > 0 else "no"

    return {
        "predicted_score": predicted_score,
        "predicted_label": predicted_label,
        "claude_score": "移除",
        "claude_analysis": "Claude 已从 v20.2.1-FULL 架构移除",
        "gemini_score": gemini_r.get("ai_score", "弃权") if isinstance(gemini_r, dict) else "弃权",
        "gemini_analysis": gemini_r.get("reason", "弃权") if isinstance(gemini_r, dict) else "弃权",
        "gpt_score": gpt_r.get("ai_score", "弃权") if isinstance(gpt_r, dict) else "弃权",
        "gpt_analysis": gpt_r.get("reason", "弃权") if isinstance(gpt_r, dict) else "弃权",
        "grok_score": grok_r.get("ai_score", "弃权") if isinstance(grok_r, dict) else "弃权",
        "grok_analysis": grok_r.get("reason", "弃权") if isinstance(grok_r, dict) else "弃权",

        "result": result_cn,
        "display_direction": result_cn,
        "final_direction": final_direction,

        "home_win_pct": home_pct,
        "draw_pct": draw_pct,
        "away_win_pct": away_pct,

        "confidence": round(ai_conf, 1),
        "ai_avg_confidence": round(ai_conf, 1),
        "confidence_meaning": "AI结构审计信心，不等于历史命中率",

        "recommendation_tier": tier,
        "direction_tier": tier,
        "score_tier": tier,
        "overall_selection_score": round(ai_conf, 1),
        "direction_selection_score": round(ai_conf, 1),
        "score_shape_score": round(ai_conf, 1),

        "direction_stability": final_r.get("direction_stability", final_r.get("direction_confidence", "")),
        "score_stability": final_r.get("score_stability", final_r.get("score_confidence", "")),

        "goal_band": goal_band,
        "goal_range": goal_band,
        "btts": btts,
        "btts_ai": btts,
        "expected_total_goals": final_r.get("expected_total_goals", ""),

        "top_score_candidates": top3,
        "unified_matrix_top_scores": top3,

        "recommendation": rec,
        "why_recommended": rec.get("why_recommended", "") if isinstance(rec, dict) else "",
        "stake_style": rec.get("stake_style", "") if isinstance(rec, dict) else "",

        "anchor_audit": anchor_audit,
        "market_interpretation": market_interpretation,
        "web_research": web_research,
        "risk_flags": risk_flags,
        "selection_warnings": risk_flags,
        "validation_warnings": final_r.get("validation_warnings", []),

        "sharp_money_direction": market_interpretation.get("sharp_money_direction", ""),
        "public_money_direction": market_interpretation.get("public_money_direction", ""),
        "market_heat": market_interpretation.get("heat_risk", ""),

        "smart_signals": list(stats.get("smart_signals", [])) + [f"FINAL_SOURCE:{final_source}"],
        "smart_money_signal": " | ".join([str(x) for x in stats.get("smart_signals", [])][:12]),

        "final_audit_reason": reason,
        "reason": reason,

        "is_score_others": is_score_others,
        "engine_version": ENGINE_VERSION,
        "engine_architecture": "AI-NATIVE 三AI：GPT/Grok 初审 + Gemini 终审裁判；本地不篡改比分",
        "final_model": final_source,

        # 兼容旧前端字段
        "risk_level": "低" if tier in {"S", "A"} else ("中" if tier == "B" else "高"),
        "over_under_2_5": "大" if _f(goal_band, 2) >= 3 else "小",
        "both_score": "是" if btts == "yes" else "否",
        "over_2_5": 50,
        "bookmaker_implied_home_xg": engine_result.get("bookmaker_implied_home_xg", "?") if isinstance(engine_result, dict) else "?",
        "bookmaker_implied_away_xg": engine_result.get("bookmaker_implied_away_xg", "?") if isinstance(engine_result, dict) else "?",
        "xG_home": engine_result.get("bookmaker_implied_home_xg", "?") if isinstance(engine_result, dict) else "?",
        "xG_away": engine_result.get("bookmaker_implied_away_xg", "?") if isinstance(engine_result, dict) else "?",
        "poisson": {},
        "refined_poisson": stats.get("refined_poisson", {}) if isinstance(stats, dict) else {},
        "elo": stats.get("elo", {}) if isinstance(stats, dict) else {},
        "random_forest": stats.get("random_forest", {}) if isinstance(stats, dict) else {},
        "gradient_boost": stats.get("gradient_boost", {}) if isinstance(stats, dict) else {},
        "neural_net": stats.get("neural_net", {}) if isinstance(stats, dict) else {},
        "logistic": stats.get("logistic", {}) if isinstance(stats, dict) else {},
        "svm": stats.get("svm", {}) if isinstance(stats, dict) else {},
        "knn": stats.get("knn", {}) if isinstance(stats, dict) else {},
        "dixon_coles": stats.get("dixon_coles", {}) if isinstance(stats, dict) else {},
        "bradley_terry": stats.get("bradley_terry", {}) if isinstance(stats, dict) else {},
        "home_form": stats.get("home_form", {}) if isinstance(stats, dict) else {},
        "away_form": stats.get("away_form", {}) if isinstance(stats, dict) else {},
        "handicap_signal": stats.get("handicap_signal", "") if isinstance(stats, dict) else "",
        "odds_movement": stats.get("odds_movement", {}) if isinstance(stats, dict) else {},
        "vote_analysis": stats.get("vote_analysis", {}) if isinstance(stats, dict) else {},
        "h2h_blood": stats.get("h2h_blood", {}) if isinstance(stats, dict) else {},
        "crs_analysis": stats.get("crs_analysis", {}) if isinstance(stats, dict) else {},
        "ttg_analysis": stats.get("ttg_analysis", {}) if isinstance(stats, dict) else {},
        "halftime": stats.get("halftime", {}) if isinstance(stats, dict) else {},
        "pace_rating": stats.get("pace_rating", "") if isinstance(stats, dict) else "",
        "kelly_home": stats.get("kelly_home", {}) if isinstance(stats, dict) else {},
        "kelly_away": stats.get("kelly_away", {}) if isinstance(stats, dict) else {},
        "odds": stats.get("odds", {}) if isinstance(stats, dict) else {},
        "experience_analysis": stats.get("experience_analysis", {}) if isinstance(stats, dict) else {},
        "pro_odds": stats.get("pro_odds", {}) if isinstance(stats, dict) else {},
        "asian_handicap_probs": stats.get("asian_handicap_probs", {}) if isinstance(stats, dict) else {},
    }


def _enforce_consistency(mg: Dict[str, Any]) -> Dict[str, Any]:
    score = mg.get("predicted_score", "1-1")
    h, a = _parse_score(score)

    if h is not None:
        d = "home" if h > a else ("away" if h < a else "draw")
        mg["final_direction"] = d
        mg["result"] = _direction_cn(d)
        mg["display_direction"] = _direction_cn(d)

    if score in ALL_SCORE_OTHERS or "其他" in str(score):
        mg["is_score_others"] = True
        mg["predicted_label"] = {"home": "胜其他", "draw": "平其他", "away": "负其他"}.get(mg.get("final_direction"), score)
    else:
        mg["predicted_label"] = score

    return mg


# ====================================================================
# Top4 精选
# ====================================================================

def select_top4(preds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for p in preds:
        pr = p.get("prediction", {})
        tier = str(pr.get("recommendation_tier", "D")).upper()
        score = 0.0

        score += {"S": 95, "A": 82, "B": 65, "C": 45, "D": 20}.get(tier, 20)
        score += min(8, max(0, _f(pr.get("ai_avg_confidence", 0)) - 60) * 0.2)

        ds = str(pr.get("direction_stability", "")).lower()
        ss = str(pr.get("score_stability", "")).lower()
        if ds == "strong":
            score += 4
        if ss == "strong":
            score += 4

        risks = pr.get("risk_flags", [])
        if isinstance(risks, list):
            score -= min(12, len(risks) * 2)

        rec = pr.get("recommendation", {})
        if isinstance(rec, dict) and rec.get("is_recommended") is False:
            score -= 18

        p["recommend_score"] = round(score, 2)

    ordered = sorted(preds, key=lambda x: x.get("recommend_score", 0), reverse=True)
    return ordered[:4]


def extract_num(ms):
    wm = {
        "一": 1000, "二": 2000, "三": 3000, "四": 4000,
        "五": 5000, "六": 6000, "日": 7000, "天": 7000,
    }
    base = next((v for k, v in wm.items() if k in str(ms)), 0)
    nums = re.findall(r"\d+", str(ms))
    return base + int(nums[0]) if nums else 9999


# ====================================================================
# 主入口
# ====================================================================

def run_predictions(raw: Dict[str, Any], use_ai: bool = True):
    ms = raw.get("matches", [])
    ms = [normalize_match(m) for m in ms]

    print("\n" + "=" * 80)
    print(f"  [{ENGINE_VERSION}] AI-NATIVE 三AI满血版 | {len(ms)} 场")
    print("  架构: GPT/Grok 初审 -> Gemini 终审裁判 | 本地不篡改比分")
    print("=" * 80)

    match_analyses: List[Dict[str, Any]] = []

    for i, m in enumerate(ms):
        try:
            eng = predict_match(m)
        except Exception as e:
            logger.warning(f"predict_match 失败: {e}")
            eng = {}

        try:
            league_info, _, _, _ = build_league_intelligence(m)
        except Exception:
            league_info = {}

        try:
            sp = ensemble.predict(m, {}) if ensemble else {}
        except Exception as e:
            logger.warning(f"ensemble.predict 失败: {e}")
            sp = {}

        try:
            exp_result = exp_engine.analyze(m) if exp_engine else {}
        except Exception:
            exp_result = {}

        match_analyses.append({
            "match": m,
            "engine": eng or {},
            "league_info": league_info,
            "stats": sp or {},
            "index": i + 1,
            "experience": exp_result,
        })

    all_ai: Dict[str, Dict[int, Dict[str, Any]]] = {"gpt": {}, "grok": {}, "gemini": {}}

    if use_ai and match_analyses:
        print("  [AI] 启动 GPT/Grok 初审 + Gemini 终审...")
        start_t = time.time()

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
                future = pool.submit(_run_in_thread, run_ai_matrix_two_phase(match_analyses))
                try:
                    all_ai = future.result()
                except Exception as e:
                    logger.error(f"AI 矩阵并发执行崩溃: {e}")
                    all_ai = {"gpt": {}, "grok": {}, "gemini": {}}
        else:
            try:
                all_ai = asyncio.run(run_ai_matrix_two_phase(match_analyses))
            except Exception as e:
                logger.error(f"AI 矩阵执行失败: {e}")
                all_ai = {"gpt": {}, "grok": {}, "gemini": {}}

        print(f"  [AI完成] 耗时 {time.time() - start_t:.1f}s")

    res = []

    for i, ma in enumerate(match_analyses):
        m = ma["match"]

        mg = merge_result(
            engine_result=ma.get("engine", {}),
            gpt_r=all_ai.get("gpt", {}).get(i + 1, {}),
            grok_r=all_ai.get("grok", {}).get(i + 1, {}),
            gemini_r=all_ai.get("gemini", {}).get(i + 1, {}),
            stats=ma.get("stats", {}),
            match_obj=m,
        )

        # 下游模块只做附加字段，原则上不允许改比分；若你的模块会改比分，建议在模块内部关闭比分覆盖。
        try:
            if exp_engine:
                mg = apply_experience_to_prediction(m, mg, exp_engine)
        except Exception as e:
            logger.warning(f"apply_experience_to_prediction 失败: {e}")

        try:
            mg = apply_odds_history(m, mg)
        except Exception as e:
            logger.warning(f"apply_odds_history 失败: {e}")

        try:
            mg = apply_quant_edge(m, mg)
        except Exception as e:
            logger.warning(f"apply_quant_edge 失败: {e}")

        try:
            mg = apply_wencai_intel(m, mg)
        except Exception as e:
            logger.warning(f"apply_wencai_intel 失败: {e}")

        try:
            mg = upgrade_ensemble_predict(m, mg)
        except Exception as e:
            logger.warning(f"upgrade_ensemble_predict 失败: {e}")

        mg = _enforce_consistency(mg)

        res.append({**m, "prediction": mg})

        print(
            f"  [{i+1}] {m.get('home_team')} vs {m.get('away_team')} => "
            f"{mg.get('result')} ({mg.get('predicted_score')}) | "
            f"Tier:{mg.get('recommendation_tier')} | AI:{mg.get('ai_avg_confidence')} | "
            f"Final:{mg.get('final_model')}"
        )

    t4 = select_top4(res)

    # 修复 id=None 批量误标推荐
    t4_keys = set()
    for t in t4:
        tid = t.get("id")
        if tid is not None and str(tid).strip() != "":
            t4_keys.add(("id", str(tid)))
        else:
            t4_keys.add(("idx", id(t)))

    for r in res:
        rid = r.get("id")
        if rid is not None and str(rid).strip() != "":
            r["is_recommended"] = ("id", str(rid)) in t4_keys
        else:
            r["is_recommended"] = ("idx", id(r)) in t4_keys

    res.sort(key=lambda x: extract_num(x.get("match_num", "")))
    return res, t4


if __name__ == "__main__":
    logger.info(f"{ENGINE_VERSION} 启动")
    print(f"✅ {ENGINE_VERSION} 加载完成")
    print("   架构: GPT/Grok 初审 -> Gemini 终审裁判")
    print("   本地: 只做抓包、调用、解析、字段闭环；不篡改 AI 最终比分")

import os

# 足球数据API
API_FOOTBALL_KEY = os.environ.get("API_FOOTBALL_KEY", "")
API_FOOTBALL_BASE = "https://v3.football.api-sports.io"
FOOTBALL_DATA_KEY = os.environ.get("FOOTBALL_DATA_KEY", "")
FOOTBALL_DATA_BASE = "https://api.football-data.org/v4"
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
ODDS_API_BASE = "https://api.the-odds-api.com/v4"

# 爬虫目标
C500_URL = "https://trade.500.com/jczq/?date={date}"
OKOOO_DETAIL = "https://m.okooo.com/jczq/"
TIMEZONE = "Asia/Shanghai"

# Each provider has one model and one unnumbered URL/key pair.
DEFAULT_MODELS = {
    "gpt": "gpt-5.6-sol",
    "grok": "熊猫-A-10-grok-4.6",
    "gemini": "熊猫-顶级特供-X-17-gemini-3.1-pro-preview-联网",
}


def get_model_for(ai_name):
    name = str(ai_name or "").strip().lower()
    if name not in DEFAULT_MODELS:
        return ""
    return os.environ.get(name.upper() + "_MODEL", "").strip() or DEFAULT_MODELS[name]


# GPT
GPT_API_URL = os.environ.get("GPT_API_URL", "")
GPT_API_KEY = os.environ.get("GPT_API_KEY", "")
GPT_MODEL = get_model_for("gpt")

# Gemini
GEMINI_API_URL = os.environ.get("GEMINI_API_URL", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = get_model_for("gemini")

# Grok
GROK_API_URL = os.environ.get("GROK_API_URL", "")
GROK_API_KEY = os.environ.get("GROK_API_KEY", "")
GROK_MODEL = get_model_for("grok")

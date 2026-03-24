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

# GPT（thinking优先）
GPT_API_URL = os.environ.get("GPT_API_URL", "")
GPT_API_KEY = os.environ.get("GPT_API_KEY", "")
GPT_MODEL = "熊猫-A-7-gpt-5.4"

# Gemini（3.1 pro thinking优先）
GEMINI_API_URL = os.environ.get("GEMINI_API_URL", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "熊猫特供-按量-SSS-gemini-3.1-pro-preview-thinking"

# Grok（thinking优先）
GROK_API_URL = os.environ.get("GROK_API_URL", "")
GROK_API_KEY = os.environ.get("GROK_API_KEY", "")
GROK_MODEL = "熊猫-A-6-grok-4.2-thinking"

# Claude 中控（opus thinking优先）
CLAUDE_API_URL = os.environ.get("CLAUDE_API_URL", "")
CLAUDE_API_KEY = os.environ.get("CLAUDE_API_KEY", "")
CLAUDE_MODEL = "熊猫-按量-顶级特供-官max-claude-opus-4.6-thinking"
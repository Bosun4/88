import os
API_FOOTBALL_KEY=os.environ.get("API_FOOTBALL_KEY","")
API_FOOTBALL_BASE="https://v3.football.api-sports.io"
FOOTBALL_DATA_KEY=os.environ.get("FOOTBALL_DATA_KEY","")
FOOTBALL_DATA_BASE="https://api.football-data.org/v4"
ODDS_API_KEY=os.environ.get("ODDS_API_KEY","")
ODDS_API_BASE="https://api.the-odds-api.com/v4"
GEMINI_API_URL="https://once.novai.su/v1/chat/completions"
GEMINI_API_KEY=os.environ.get("GEMINI_API_KEY","")
GEMINI_MODEL="gemini-3.1-pro-preview-thinking"
GPT_API_URL="https://nan.meta-api.vip/v1/chat/completions"
GPT_API_KEY=os.environ.get("GPT_API_KEY","")
GPT_MODEL="gpt-5.4"
C500_URL="https://trade.500.com/jczq/?date={date}"
OKOOO_DETAIL="https://m.okooo.com/jczq/"
TIMEZONE="Asia/Shanghai"
# Claude 中控
CLAUDE_API_URL = os.environ.get("CLAUDE_API_URL", "")
CLAUDE_API_KEY = os.environ.get("CLAUDE_API_KEY", "")
CLAUDE_MODEL = "claude-opus-4.6-thinking"

# Grok
GROK_API_URL = os.environ.get("GROK_API_URL", "")
GROK_API_KEY = os.environ.get("GROK_API_KEY", "")
GROK_MODEL = "grok-4.2-thinking"

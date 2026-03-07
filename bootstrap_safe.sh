#!/usr/bin/env bash

set -euo pipefail

BASE_DIR="${HOME}/football-predict"
mkdir -p "${BASE_DIR}/scripts" "${BASE_DIR}/data" "${BASE_DIR}/.github/workflows"
cd "${BASE_DIR}"

cat > .nojekyll <<'EOF'
EOF

cat > .gitignore <<'EOF'
__pycache__/
*.pyc
.env
data/raw_data.json
EOF

cat > requirements.txt <<'EOF'
requests>=2.31.0
beautifulsoup4>=4.12.0
lxml>=5.0.0
EOF

cat > README.md <<'EOF'
# AI Football Predict

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/main.py
```

## Required Environment Variables

- API_FOOTBALL_KEY
- FOOTBALL_DATA_KEY
- ODDS_API_KEY
- GEMINI_API_KEY
- GPT_API_KEY
EOF

cat > data/predictions.json <<'EOF'
{"date":"","update_time":"wait","total_matches":0,"results":[],"top4":[]}
EOF

cat > scripts/config.py <<'EOF'
import os

API_FOOTBALL_KEY = os.environ.get("API_FOOTBALL_KEY", "")
API_FOOTBALL_BASE = "https://v3.football.api-sports.io"

FOOTBALL_DATA_KEY = os.environ.get("FOOTBALL_DATA_KEY", "")
FOOTBALL_DATA_BASE = "https://api.football-data.org/v4"

ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
ODDS_API_BASE = "https://api.the-odds-api.com/v4"

GEMINI_API_URL = "https://once.novai.su/v1/chat/completions"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-3.1-pro-preview"

GPT_API_URL = "https://once.novai.su/v1/chat/completions"
GPT_API_KEY = os.environ.get("GPT_API_KEY", "")
GPT_MODEL = "gpt-5.2"

OKOOO_DETAIL = "https://m.okooo.com/jczq/"
OKOOO_SIMPLE = "https://m.okooo.com/kaijiang/sport.php?LotteryType=SportteryNWDL&LotteryNo={date}"
C500_URL = "https://trade.500.com/jczq/?date={date}"

JINGCAI_LEAGUES = [39, 140, 135, 78, 61, 2, 3, 848, 88, 94, 203, 144, 235, 169, 292, 98, 253, 71, 128]
TIMEZONE = "Asia/Shanghai"
EOF

cat > scripts/main.py <<'EOF'
import json
import os
from datetime import datetime


def main() -> None:
    os.makedirs("data", exist_ok=True)
    out = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_matches": 0,
        "results": [],
        "top4": [],
    }
    with open("data/predictions.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("Project scaffold ready. Add fetch/predict logic next.")


if __name__ == "__main__":
    main()
EOF

cat > .github/workflows/predict.yml <<'EOF'
name: Football AI Predict

on:
  schedule:
    - cron: "30 1 * * *"
    - cron: "30 13 * * *"
  workflow_dispatch:

permissions:
  contents: write

jobs:
  predict:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -r requirements.txt
      - name: Run
        env:
          API_FOOTBALL_KEY: ${{ secrets.API_FOOTBALL_KEY }}
          FOOTBALL_DATA_KEY: ${{ secrets.FOOTBALL_DATA_KEY }}
          ODDS_API_KEY: ${{ secrets.ODDS_API_KEY }}
          GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
          GPT_API_KEY: ${{ secrets.GPT_API_KEY }}
        run: cd scripts && python main.py
      - name: Push data updates
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
          git add data/
          git diff --cached --quiet || git commit -m "update predictions data"
          git push
EOF

echo "Scaffold created at ${BASE_DIR}"
echo "Next: cd ${BASE_DIR} && python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"

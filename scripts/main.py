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

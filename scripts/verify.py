"""Production review entrypoint; shares exact identity and metrics with self_learn."""
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

try:
    from .config import TIMEZONE
    from .self_learn import fetch_actual_results, self_learn
except ImportError:
    from config import TIMEZONE
    from self_learn import fetch_actual_results, self_learn


def get_yesterday():
    return (datetime.now(ZoneInfo(TIMEZONE)) - timedelta(days=1)).date().isoformat()


def verify_and_learn():
    yesterday = get_yesterday()
    candidates = [Path(f'data/history_{yesterday}_today_{session}.json')
                  for session in ('morning', 'evening')]
    candidates.append(Path('data/predictions.json'))
    # Earliest stored version is deterministic; this is still retrospective.
    source = next((p for p in candidates if p.exists()), None)
    if source is None:
        print('无预测记录，跳过对账。')
        return None
    return self_learn(str(source), 'data/ai_diary.json', fetch_actual_results)


if __name__ == '__main__':
    verify_and_learn()

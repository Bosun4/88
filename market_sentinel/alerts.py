from dataclasses import dataclass
from typing import Optional, List
import json

@dataclass
class MarketAlert:
    alert_type: str
    severity: str
    direction: str
    evidence: str
    source: str
    timestamp_utc: str

def generate_alert_json(alerts: List[MarketAlert]) -> str:
    return json.dumps([
        {
            "alert_type": a.alert_type,
            "severity": a.severity,
            "direction": a.direction,
            "evidence": a.evidence,
            "source": a.source,
            "timestamp_utc": a.timestamp_utc
        } for a in alerts
    ], ensure_ascii=False, indent=2)

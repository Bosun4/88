from dataclasses import dataclass
from typing import Optional

@dataclass
class MarketSnapshot:
    match_id: str
    source: str
    timestamp_utc: str
    minutes_to_kickoff: int
    market_type: str
    selection: str
    odds: float
    implied_prob_raw: float
    implied_prob_fair: float
    line: Optional[float] = None
    back_price: Optional[float] = None
    lay_price: Optional[float] = None
    back_size: Optional[float] = None
    lay_size: Optional[float] = None
    traded_volume: Optional[float] = None
    bookmaker: Optional[str] = None
    exchange_market_id: Optional[str] = None
    runner_id: Optional[str] = None

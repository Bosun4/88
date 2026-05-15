from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

@dataclass
class LedgerEntry:
    prediction_file: str
    prediction_sha256: str
    created_at_utc: str
    match_count: int
    engine_commit: str
    match_id: str
    home_team: str
    away_team: str
    predicted_score: str
    final_direction: str
    confidence: int
    home_win_pct: float
    draw_pct: float
    away_win_pct: float
    
    risk_score_candidates: List[str] = field(default_factory=list)
    tail_risk_flags: List[str] = field(default_factory=list)
    
    matrix_recommended_score: Optional[str] = None
    matrix_recommended_direction: Optional[str] = None
    matrix_top_scores: List[str] = field(default_factory=list)
    matrix_disagreement_flags: List[str] = field(default_factory=list)
    
    sub50_tiebreaker_warning: bool = False
    no_bet_reason: Optional[str] = None
    score_cluster: List[str] = field(default_factory=list)
    
    score_moderation_applied: bool = False
    original_predicted_score: Optional[str] = None
    
    actual_score: Optional[str] = None
    actual_direction: Optional[str] = None
    
    direction_hit: Optional[bool] = None
    exact_score_hit: Optional[bool] = None
    goal_band_hit: Optional[bool] = None
    btts_hit: Optional[bool] = None
    
    risk_candidate_covered: Optional[bool] = None
    matrix_top_scores_covered: Optional[bool] = None
    score_cluster_covered: Optional[bool] = None
    score_moderation_helped: Optional[bool] = None
    score_moderation_hurt: Optional[bool] = None

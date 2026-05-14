def euro_asian_divergence_index(fair_home_prob: float, handicap_line: float, handicap_odds_h: float, handicap_odds_a: float) -> tuple:
    """
    Returns (DI_home, DI_abs)
    """
    if handicap_odds_h <= 1 or handicap_odds_a <= 1:
        return 0.0, 0.0
        
    base_expected = 0.50 - (handicap_line * 0.15)
    margin = (1 / handicap_odds_h) + (1 / handicap_odds_a)
    ah_fair_h = (1 / handicap_odds_h) / margin
    
    proxy_prob = base_expected + (ah_fair_h - 0.50)
    di_home = fair_home_prob - proxy_prob
    
    return di_home, abs(di_home)

def classify_divergence(di_abs: float) -> str:
    if di_abs >= 0.12:
        return "critical"
    elif di_abs >= 0.08:
        return "warning"
    elif di_abs >= 0.05:
        return "watch"
    return "none"

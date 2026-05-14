import math

def compute_odds_velocity(odds_t: float, odds_t_minus: float, minutes: float) -> float:
    if odds_t <= 1 or odds_t_minus <= 1 or minutes <= 0: return 0.0
    return math.log(odds_t / odds_t_minus) / minutes

def compute_prob_velocity(prob_t: float, prob_t_minus: float) -> float:
    return prob_t - prob_t_minus

def compute_volume_velocity(vol_t: float, vol_t_minus: float) -> float:
    return vol_t - vol_t_minus

def detect_late_steam(fair_prob_t: float, fair_prob_t_minus: float, volume_t: float, volume_t_minus: float, window_mins: int = 5):
    prob_vel = compute_prob_velocity(fair_prob_t, fair_prob_t_minus)
    delta_pp = abs(prob_vel) * 100
    vol_vel = compute_volume_velocity(volume_t, volume_t_minus)
    
    flags = []
    
    if delta_pp >= 10:
        flags.append("steam_critical")
    elif delta_pp >= 7:
        flags.append("steam_warning")
    elif delta_pp >= 4:
        flags.append("steam_watch")
        
    if flags and vol_vel > 10000:
        flags.append("sharp_confirmed")
    elif flags and vol_vel < 1000:
        flags.append("fake_steam_suspected")
        
    direction = "home" if prob_vel > 0 else "away"
    
    return {
        "late_steam_flags": flags,
        "steam_direction": direction if flags else "none",
        "steam_strength_pp": round(delta_pp, 2),
        "steam_window": f"{window_mins}m",
        "steam_evidence": f"Fair prob changed {delta_pp:.2f}pp towards {direction}. Volume delta: {vol_vel}."
    }

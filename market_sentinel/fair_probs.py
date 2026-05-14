import math

def fair_probs_from_1x2(odds: dict, method: str = "power") -> dict:
    """
    Computes fair probabilities from 1X2 odds.
    odds format: {"h": float, "d": float, "a": float}
    Returns: {"h": float, "d": float, "a": float} roughly summing to 1.
    """
    h, d, a = odds.get("h", 0), odds.get("d", 0), odds.get("a", 0)
    if not (h > 1 and d > 1 and a > 1):
        return {"h": 0, "d": 0, "a": 0}
        
    p_h, p_d, p_a = 1/h, 1/d, 1/a
    margin = p_h + p_d + p_a
    
    if margin <= 1.0:
        return {"h": p_h, "d": p_d, "a": p_a}
        
    if method == "multiplicative":
        return {"h": p_h / margin, "d": p_d / margin, "a": p_a / margin}
        
    elif method == "additive":
        excess = margin - 1.0
        return {
            "h": p_h - (excess / 3),
            "d": p_d - (excess / 3),
            "a": p_a - (excess / 3)
        }
        
    elif method == "power":
        # Solve for exponent k such that p_h^k + p_d^k + p_a^k = 1
        # Simple iterative approximation
        k = 1.0
        step = 0.01
        for _ in range(100):
            sm = (p_h**k) + (p_d**k) + (p_a**k)
            if abs(sm - 1.0) < 0.001:
                break
            if sm > 1.0:
                k += step
            else:
                k -= step
        return {"h": p_h**k, "d": p_d**k, "a": p_a**k}
        
    else:
        # Default fallback to multiplicative
        return {"h": p_h / margin, "d": p_d / margin, "a": p_a / margin}

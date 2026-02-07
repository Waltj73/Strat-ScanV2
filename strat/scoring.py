from typing import Dict, Tuple

def score_regime(flags: Dict[str, bool]) -> Tuple[int, int]:
    bull = 0
    bear = 0

    # Inside bars are “potential energy”, not directional by themselves,
    # so they do NOT add bull/bear points.
    bull += 2 if flags.get("W_212Up") else 0
    bull += 1 if flags.get("D_212Up") else 0

    bear += 2 if flags.get("W_212Dn") else 0
    bear += 1 if flags.get("D_212Dn") else 0

    return bull, bear

def market_bias(bull_total: int, bear_total: int) -> str:
    diff = bull_total - bear_total
    if diff >= 2:
        return "LONG"
    if diff <= -2:
        return "SHORT"
    return "MIXED"


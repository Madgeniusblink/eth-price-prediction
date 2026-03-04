#!/usr/bin/env python3
"""
Uniswap v3 LP Range Optimizer
Recommends optimal tick ranges based on volatility regime and calculates
expected fee APY and impermanent loss estimates.
"""

import math
from datetime import datetime


# Default assumptions (overridden by DeFiLlama if available)
DEFAULT_DAILY_VOLUME_USD = 500_000_000   # $500M/day ETH/USDC 0.3%
DEFAULT_TVL_USD          = 200_000_000   # $200M TVL
FEE_TIER                 = 0.003         # 0.3% fee tier

REGIME_RANGE_PCT = {
    "LOW":     0.05,   # ±5%
    "MEDIUM":  0.10,   # ±10%
    "HIGH":    0.20,   # ±20%
    "EXTREME": 0.35,   # ±35%
    "UNKNOWN": 0.15,   # ±15% default
}

REGIME_RECOMMENDATION = {
    "LOW":     "ADD_LIQUIDITY",
    "MEDIUM":  "ADD_LIQUIDITY",
    "HIGH":    "HOLD",
    "EXTREME": "AVOID",
    "UNKNOWN": "HOLD",
}


def impermanent_loss(p_ratio: float) -> float:
    """
    Calculate IL at a given price ratio (boundary).
    IL = 2*sqrt(P_ratio)/(1+P_ratio) - 1
    Returns IL as a negative percentage (loss).
    """
    il = 2 * math.sqrt(p_ratio) / (1 + p_ratio) - 1
    return il * 100  # as %


def estimate_fee_apy(daily_volume: float, tvl: float, range_pct: float) -> float:
    """
    Estimate fee APY for a concentrated liquidity position.
    Assumes capital efficiency multiplier for concentrated range vs full-range.
    """
    if tvl <= 0:
        return 0.0
    # Baseline fee APY if full range
    annual_fees = daily_volume * FEE_TIER * 365
    base_apy = (annual_fees / tvl) * 100

    # Concentrated liquidity capital efficiency: ~1/(2*range_pct) vs full range
    # Simplified: a ±range_pct position captures roughly (1/(2*range_pct)) * base_apy
    # Capped at 300% to avoid absurd estimates
    efficiency = min(1 / (2 * range_pct), 10)
    concentrated_apy = base_apy * efficiency
    return round(concentrated_apy, 1)


def calculate_range(current_price: float, regime: str) -> dict:
    """
    Calculate optimal LP range for current price and volatility regime.
    """
    if regime == "EXTREME":
        return {
            "regime": regime,
            "range_pct": 35.0,
            "lower_price": round(current_price * (1 - 0.35), 2),
            "upper_price": round(current_price * (1 + 0.35), 2),
            "recommendation": "AVOID",
            "note": "EXTREME volatility — LP positions likely to suffer heavy IL. Avoid.",
            "fee_apy": 0.0,
            "il_at_boundary_pct": 0.0,
            "net_expected_return_pct": 0.0,
        }

    range_pct = REGIME_RANGE_PCT.get(regime, 0.15)
    lower = current_price * (1 - range_pct)
    upper = current_price * (1 + range_pct)

    # IL at boundary: price moves from current to upper (P_ratio = upper/current)
    p_ratio_up = upper / current_price
    il_pct = impermanent_loss(p_ratio_up)  # symmetric for lower boundary

    return {
        "regime": regime,
        "range_pct": range_pct * 100,
        "lower_price": round(lower, 2),
        "upper_price": round(upper, 2),
        "recommendation": REGIME_RECOMMENDATION.get(regime, "HOLD"),
        "il_at_boundary_pct": round(il_pct, 2),
    }


def get_lp_recommendation(
    current_price: float,
    regime: str,
    defillama_data: dict = None,
) -> dict:
    """
    Full LP optimization output.

    Args:
        current_price: Current ETH price in USD
        regime: Volatility regime label (LOW/MEDIUM/HIGH/EXTREME/UNKNOWN)
        defillama_data: Optional dict from DeFiLlama with 'tvl' key

    Returns:
        Full recommendation dict
    """
    try:
        # Get TVL from DeFiLlama if available
        daily_volume = DEFAULT_DAILY_VOLUME_USD
        tvl = DEFAULT_TVL_USD
        if defillama_data:
            # DeFiLlama returns tvl at top level
            llama_tvl = defillama_data.get("tvl")
            if llama_tvl and llama_tvl > 0:
                tvl = float(llama_tvl)

        range_info = calculate_range(current_price, regime)
        range_pct = REGIME_RANGE_PCT.get(regime, 0.15)

        if regime != "EXTREME":
            fee_apy = estimate_fee_apy(daily_volume, tvl, range_pct)
            il_pct = range_info["il_at_boundary_pct"]
            # Net return = fee APY + IL (IL is negative)
            net_return = round(fee_apy + il_pct, 1)
            range_info["fee_apy"] = fee_apy
            range_info["net_expected_return_pct"] = net_return
            range_info["tvl_used"] = round(tvl / 1e6, 1)  # in $M

        range_info["current_price"] = current_price
        range_info["timestamp"] = datetime.utcnow().isoformat()
        return range_info

    except Exception as e:
        return {
            "regime": regime,
            "current_price": current_price,
            "recommendation": "HOLD",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat(),
        }


if __name__ == "__main__":
    import json
    result = get_lp_recommendation(2000.0, "MEDIUM")
    print(json.dumps(result, indent=2))

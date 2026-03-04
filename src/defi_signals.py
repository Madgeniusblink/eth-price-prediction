"""
DeFi Signal Engine — Uniswap v3 LP optimization, SushiSwap comparison,
impermanent loss estimation, and on-chain momentum signals.
"""

import math
import logging
import requests

logger = logging.getLogger(__name__)

TIMEOUT = 10
SUSHI_GRAPH = "https://api.thegraph.com/subgraphs/name/sushi-v2/sushiswap"
SUSHI_PAIR = "0x397ff1542f962076d0bfe58ea045ffa2d347aca0"  # ETH/USDC on SushiSwap v2


def impermanent_loss(price_change_pct: float) -> float:
    """
    Compute impermanent loss for a given price change percentage.
    Formula: IL = 2*sqrt(r) / (1+r) - 1  where r = price_ratio
    Returns IL as a positive percentage (loss).
    """
    r = 1 + price_change_pct / 100
    if r <= 0:
        return 100.0
    il = 2 * math.sqrt(r) / (1 + r) - 1
    return abs(il) * 100


def compute_uniswap_lp_analysis(pool_data: dict, atr_30d: float, current_price: float) -> dict:
    """
    Compute Uniswap v3 LP analysis including optimal ranges and fee APY.

    Args:
        pool_data: dict from onchain_data.fetch_uniswap_v3_pool()
        atr_30d:   30-day Average True Range (in USD)
        current_price: current ETH price in USD
    """
    result = {
        "optimal_range_1sigma": [None, None],
        "optimal_range_2sigma": [None, None],
        "projected_fee_apy": None,
        "il_at_10pct": round(impermanent_loss(10), 3),
        "il_at_20pct": round(impermanent_loss(20), 3),
        "il_at_30pct": round(impermanent_loss(30), 3),
        "tvl_usd": None,
        "fees_24h_usd": None,
        "recommendation": "WAIT",
    }

    try:
        tvl = pool_data.get("tvl_usd") or 0
        fees_24h = pool_data.get("fees_24h_usd") or 0

        result["tvl_usd"] = tvl
        result["fees_24h_usd"] = fees_24h

        # Fee APY = (24h_fees / TVL) * 365 * 100
        if tvl > 0:
            result["projected_fee_apy"] = round((fees_24h / tvl) * 365 * 100, 2)

        # Optimal LP range using ATR as proxy for sigma
        if atr_30d and current_price:
            lower_1s = round(current_price - atr_30d, 2)
            upper_1s = round(current_price + atr_30d, 2)
            lower_2s = round(current_price - 2 * atr_30d, 2)
            upper_2s = round(current_price + 2 * atr_30d, 2)

            result["optimal_range_1sigma"] = [max(lower_1s, 1), upper_1s]
            result["optimal_range_2sigma"] = [max(lower_2s, 1), upper_2s]

        # Recommendation heuristic
        apy = result.get("projected_fee_apy") or 0
        if apy > 5:
            result["recommendation"] = "PROVIDE_LIQUIDITY"
        elif apy > 2:
            result["recommendation"] = "HOLD"
        else:
            result["recommendation"] = "WAIT"

    except Exception as e:
        logger.warning(f"Uniswap LP analysis error: {e}")

    return result


def fetch_sushiswap_pair() -> dict:
    """Fetch ETH/USDC SushiSwap v2 pair data from The Graph."""
    query = f"""
    {{
      pair(id: "{SUSHI_PAIR}") {{
        token0Price
        token1Price
        reserveUSD
        volumeUSD
        pairHourData(first: 24, orderBy: hourStartUnix, orderDirection: desc) {{
          hourStartUnix
          volumeUSD
          reserveUSD
        }}
      }}
    }}
    """
    try:
        r = requests.post(
            SUSHI_GRAPH,
            json={"query": query},
            timeout=TIMEOUT,
            headers={"Content-Type": "application/json"},
        )
        r.raise_for_status()
        data = r.json()
        pair = data.get("data", {}).get("pair")
        if not pair:
            raise ValueError("No SushiSwap pair data returned")

        hour_data = pair.get("pairHourData", [])
        volume_24h = sum(float(h.get("volumeUSD", 0)) for h in hour_data)
        tvl = float(hour_data[0]["reserveUSD"]) if hour_data else float(pair.get("reserveUSD", 0))

        # SushiSwap fee = 0.25% of volume to LPs
        fees_24h = volume_24h * 0.0025

        return {
            "tvl_usd": tvl,
            "volume_24h_usd": volume_24h,
            "fees_24h_usd": fees_24h,
        }
    except Exception as e:
        logger.warning(f"SushiSwap fetch failed: {e}")
    return {"tvl_usd": None, "volume_24h_usd": None, "fees_24h_usd": None}


def compare_protocols(uniswap_lp: dict, sushi_data: dict) -> dict:
    """Compare Uniswap v3 vs SushiSwap APY."""
    uni_apy = uniswap_lp.get("projected_fee_apy") or 0

    sushi_apy = None
    tvl = sushi_data.get("tvl_usd") or 0
    fees = sushi_data.get("fees_24h_usd") or 0
    if tvl > 0:
        sushi_apy = round((fees / tvl) * 365 * 100, 2)

    if sushi_apy is None:
        winner = "UNISWAP_V3"
        winner_apy = uni_apy
    elif uni_apy >= sushi_apy:
        winner = "UNISWAP_V3"
        winner_apy = uni_apy
    else:
        winner = "SUSHISWAP"
        winner_apy = sushi_apy

    return {
        "uniswap_v3_apy": uni_apy,
        "sushiswap_apy": sushi_apy,
        "better_protocol": winner,
        "better_apy": winner_apy,
    }


def get_defi_signals(pool_data: dict, onchain: dict, atr_30d: float, current_price: float) -> dict:
    """
    Main DeFi signal aggregator.

    Args:
        pool_data:     from onchain_data.fetch_uniswap_v3_pool()
        onchain:       from onchain_data.get_all_onchain_signals()
        atr_30d:       30-day ATR in USD (from OHLC data)
        current_price: live ETH/USD price
    """
    uniswap_lp = compute_uniswap_lp_analysis(pool_data, atr_30d, current_price)
    sushi_data = fetch_sushiswap_pair()
    protocol_comparison = compare_protocols(uniswap_lp, sushi_data)

    fg = onchain.get("fear_greed", {})
    gas = onchain.get("gas", {})

    return {
        "uniswap_v3": uniswap_lp,
        "sushiswap": sushi_data,
        "protocol_comparison": protocol_comparison,
        "fear_greed_value": fg.get("value"),
        "fear_greed_label": fg.get("label"),
        "gas_gwei": gas.get("fast_gas_gwei"),
        "gas_label": gas.get("label"),
        "onchain_momentum": onchain.get("onchain_momentum", "NEUTRAL"),
    }


if __name__ == "__main__":
    import json
    from onchain_data import get_all_onchain_signals

    onchain = get_all_onchain_signals()
    pool = onchain["uniswap_v3_pool"]
    signals = get_defi_signals(pool, onchain, atr_30d=150.0, current_price=2000.0)
    print(json.dumps(signals, indent=2))

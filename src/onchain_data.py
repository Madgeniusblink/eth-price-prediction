"""
On-chain and DeFi signal fetcher
Free endpoints only — no API keys required.
"""

import requests
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

TIMEOUT = 10

UNISWAP_GRAPH = "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3"
WETH_USDC_POOL = "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"


def fetch_gas_price() -> dict:
    """Fetch ETH gas price from Etherscan gas oracle (free, no key)."""
    try:
        r = requests.get(
            "https://api.etherscan.io/api",
            params={"module": "gastracker", "action": "gasoracle"},
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
        if data.get("status") == "1":
            result = data["result"]
            return {
                "safe_gas_gwei": float(result.get("SafeGasPrice", 0)),
                "propose_gas_gwei": float(result.get("ProposeGasPrice", 0)),
                "fast_gas_gwei": float(result.get("FastGasPrice", 0)),
            }
    except Exception as e:
        logger.warning(f"Gas price fetch failed: {e}")
    return {"safe_gas_gwei": None, "propose_gas_gwei": None, "fast_gas_gwei": None}


def fetch_uniswap_v3_pool() -> dict:
    """Fetch WETH/USDC Uniswap v3 0.05% pool data from The Graph."""
    query = """
    {
      pool(id: "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640") {
        token0Price
        token1Price
        liquidity
        sqrtPrice
        tick
        poolHourData(first: 24, orderBy: periodStartUnix, orderDirection: desc) {
          periodStartUnix
          volumeUSD
          feesUSD
          tvlUSD
          close
        }
      }
    }
    """
    try:
        r = requests.post(
            UNISWAP_GRAPH,
            json={"query": query},
            timeout=TIMEOUT,
            headers={"Content-Type": "application/json"},
        )
        r.raise_for_status()
        data = r.json()
        pool = data.get("data", {}).get("pool")
        if not pool:
            raise ValueError("No pool data returned")

        hour_data = pool.get("poolHourData", [])
        total_volume_24h = sum(float(h.get("volumeUSD", 0)) for h in hour_data)
        total_fees_24h = sum(float(h.get("feesUSD", 0)) for h in hour_data)
        tvl = float(hour_data[0]["tvlUSD"]) if hour_data else 0.0

        return {
            "token0_price": float(pool.get("token0Price", 0)),   # USDC per WETH
            "token1_price": float(pool.get("token1Price", 0)),   # WETH per USDC
            "liquidity": pool.get("liquidity"),
            "tick": int(pool.get("tick", 0)),
            "tvl_usd": tvl,
            "volume_24h_usd": total_volume_24h,
            "fees_24h_usd": total_fees_24h,
            "hour_data": hour_data,
        }
    except Exception as e:
        logger.warning(f"Uniswap v3 pool fetch failed: {e}")
    return {
        "token0_price": None, "token1_price": None, "liquidity": None,
        "tick": None, "tvl_usd": None, "volume_24h_usd": None,
        "fees_24h_usd": None, "hour_data": [],
    }


def fetch_fear_greed() -> dict:
    """Fetch Fear & Greed Index (last 7 days) from alternative.me."""
    try:
        r = requests.get(
            "https://api.alternative.me/fng/",
            params={"limit": 7},
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        data = r.json().get("data", [])
        if data:
            latest = data[0]
            return {
                "value": int(latest.get("value", 0)),
                "label": latest.get("value_classification", ""),
                "history": [
                    {"value": int(d["value"]), "label": d["value_classification"]}
                    for d in data
                ],
            }
    except Exception as e:
        logger.warning(f"Fear & Greed fetch failed: {e}")
    return {"value": None, "label": None, "history": []}


def fetch_eth_supply_metrics() -> dict:
    """Fetch ETH supply metrics from ultrasound.money (free)."""
    try:
        r = requests.get(
            "https://api.ultrasound.money/fees/all-time",
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
        return {
            "eth_burned_all_time": data.get("feeBurn", None),
            "supply_change": data.get("supplyChange", None),
        }
    except Exception as e:
        logger.warning(f"ETH supply metrics fetch failed: {e}")
    return {"eth_burned_all_time": None, "supply_change": None}


def get_all_onchain_signals() -> dict:
    """
    Fetch and aggregate all on-chain and DeFi signals.
    Returns a structured dict; any failed sub-fetch is logged but won't crash.
    """
    gas = fetch_gas_price()
    uniswap = fetch_uniswap_v3_pool()
    fg = fetch_fear_greed()
    supply = fetch_eth_supply_metrics()

    # Compute on-chain momentum signal
    fast_gas = gas.get("fast_gas_gwei")
    fg_value = fg.get("value")

    if fast_gas and fast_gas > 50:
        onchain_momentum = "BULLISH"   # HIGH_ACTIVITY
    elif fg_value is not None and fg_value < 25:
        onchain_momentum = "BULLISH"   # EXTREME_FEAR contrarian long
    elif fg_value is not None and fg_value > 75:
        onchain_momentum = "BEARISH"   # EXTREME_GREED fade
    else:
        onchain_momentum = "NEUTRAL"

    gas_label = "LOW"
    if fast_gas:
        if fast_gas > 80:
            gas_label = "HIGH"
        elif fast_gas > 30:
            gas_label = "MED"

    return {
        "gas": {**gas, "label": gas_label},
        "uniswap_v3_pool": uniswap,
        "fear_greed": fg,
        "eth_supply": supply,
        "onchain_momentum": onchain_momentum,
        "fetched_at": datetime.utcnow().isoformat() + "Z",
    }


if __name__ == "__main__":
    import json
    signals = get_all_onchain_signals()
    print(json.dumps(signals, indent=2))

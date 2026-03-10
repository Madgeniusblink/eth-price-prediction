#!/usr/bin/env python3
"""
Fetch Fear & Greed Index from Alternative.me API.
Caches to data/fear_greed_history.json for use in backtests.

Rate limit: AP-006 — sleep 10s between calls.
Single call with limit=365 covers one year of daily values.
"""

import requests
import json
import time
from datetime import datetime, timezone
from pathlib import Path

# Resolve data dir relative to project root (parent of src/)
_SRC_DIR = Path(__file__).parent
_DATA_DIR = _SRC_DIR.parent / "data"


def fetch_fear_greed(limit: int = 365, output_path: Path | None = None) -> dict:
    """
    Fetch historical daily Fear & Greed index values.

    Args:
        limit       : Number of days to fetch (max 365).
        output_path : Where to cache JSON (default: data/fear_greed_history.json).

    Returns:
        dict mapping 'YYYY-MM-DD' → int value (0–100).
    """
    if output_path is None:
        output_path = _DATA_DIR / "fear_greed_history.json"

    url = f"https://api.alternative.me/fng/?limit={limit}&format=json"
    print(f"Fetching Fear & Greed history ({limit} days) from Alternative.me…")

    # AP-006: single call — no loop needed; log timestamp
    call_ts = datetime.now(timezone.utc).isoformat()
    print(f"  API call at {call_ts}")

    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    raw = resp.json()

    # AP-006: 10s sleep after call (single call here, but enforce the rule)
    time.sleep(10)

    entries = raw.get("data", [])
    fg_map: dict[str, int] = {}
    for entry in entries:
        ts = int(entry["timestamp"])
        date_str = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
        fg_map[date_str] = int(entry["value"])

    print(f"  Fetched {len(fg_map)} days of F&G data ({min(fg_map.keys())} → {max(fg_map.keys())})")

    # Cache to disk
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache = {
        "fetched_at": call_ts,
        "source": "https://api.alternative.me/fng/",
        "data": fg_map,
    }
    with open(output_path, "w") as f:
        json.dump(cache, f, indent=2)
    print(f"  Saved → {output_path}")

    return fg_map


def load_fear_greed(output_path: Path | None = None) -> dict:
    """
    Load F&G map from cache, refreshing if stale (>23 hours old).

    Returns:
        dict mapping 'YYYY-MM-DD' → int value.
    """
    if output_path is None:
        output_path = _DATA_DIR / "fear_greed_history.json"

    if output_path.exists():
        with open(output_path) as f:
            cache = json.load(f)
        fetched_at = datetime.fromisoformat(cache["fetched_at"])
        age_hours = (datetime.now(timezone.utc) - fetched_at).total_seconds() / 3600
        if age_hours < 23:
            print(f"  Loaded F&G cache ({len(cache['data'])} days, age {age_hours:.1f}h)")
            return cache["data"]
        print(f"  F&G cache stale ({age_hours:.1f}h) — refreshing…")

    return fetch_fear_greed(limit=365, output_path=output_path)


if __name__ == "__main__":
    fg = fetch_fear_greed(limit=365)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    print(f"\nToday ({today}) F&G: {fg.get(today, 'N/A')}")

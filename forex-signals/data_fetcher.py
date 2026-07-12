"""
Data fetcher for Twelve Data API.
Fetches GBP/AUD OHLCV on 4 timeframes + pair data for currency strength.
"""

import time
import requests
import pandas as pd
import streamlit as st

BASE_URL = "https://api.twelvedata.com"

GBP_PAIRS = ["GBP/USD", "GBP/EUR", "GBP/JPY", "GBP/CHF", "GBP/CAD", "GBP/NZD"]
AUD_PAIRS = ["AUD/USD", "AUD/EUR", "AUD/JPY", "AUD/CHF", "AUD/CAD", "AUD/NZD"]
ALL_STRENGTH_PAIRS = GBP_PAIRS + AUD_PAIRS

TIMEFRAME_MAP = {
    "Daily": "1day",
    "H4": "4h",
    "H1": "1h",
    "M5": "5min",
}


def _parse_response(response_json, symbol):
    """Parse a Twelve Data time_series response into a DataFrame."""
    if "values" not in response_json:
        code = response_json.get("code", "?")
        msg = response_json.get("message", "Unknown error")
        st.warning(f"API error for {symbol}: [{code}] {msg}")
        return None

    df = pd.DataFrame(response_json["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)


def fetch_ohlcv(symbol: str, interval: str, outputsize: int = 150, api_key: str = "") -> pd.DataFrame | None:
    """Fetch OHLCV candles from Twelve Data."""
    if not api_key:
        return None
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": api_key,
        "format": "JSON",
    }
    try:
        r = requests.get(f"{BASE_URL}/time_series", params=params, timeout=15)
        r.raise_for_status()
        return _parse_response(r.json(), symbol)
    except requests.RequestException as exc:
        st.warning(f"Network error fetching {symbol} ({interval}): {exc}")
        return None


def fetch_all_timeframes(api_key: str) -> dict[str, pd.DataFrame | None]:
    """Fetch GBP/AUD on Daily, H4, H1, M5 with polite rate-limiting."""
    results = {}
    for name, code in TIMEFRAME_MAP.items():
        results[name] = fetch_ohlcv("GBP/AUD", code, outputsize=200, api_key=api_key)
        time.sleep(0.8)  # Stay within 8 req/min free-tier limit
    return results


def fetch_strength_data(api_key: str) -> dict[str, pd.DataFrame | None]:
    """Fetch H1 data for 12 pairs used in currency strength calculation."""
    pair_data = {}
    for pair in ALL_STRENGTH_PAIRS:
        pair_data[pair] = fetch_ohlcv(pair, "1h", outputsize=30, api_key=api_key)
        time.sleep(0.8)
    return pair_data


def calculate_currency_strength(pair_data: dict) -> dict[str, float]:
    """
    Compute GBP and AUD strength normalised to [-100, +100].
    Each pair's % move over 20 bars is attributed to base (+) and quote (-).
    """
    raw: dict[str, list] = {}
    for pair, df in pair_data.items():
        if df is None or len(df) < 5:
            continue
        base, quote = pair.split("/")
        lookback = min(20, len(df) - 1)
        pct = (df["close"].iloc[-1] - df["close"].iloc[-1 - lookback]) / df["close"].iloc[-1 - lookback] * 100
        raw.setdefault(base, []).append(pct)
        raw.setdefault(quote, []).append(-pct)

    strength = {cur: sum(v) / len(v) for cur, v in raw.items() if v}

    # Normalise to [-100, +100]
    max_abs = max((abs(v) for v in strength.values()), default=1) or 1
    return {cur: round(val / max_abs * 100, 2) for cur, val in strength.items()}

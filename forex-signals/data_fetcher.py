"""
Data fetcher for Twelve Data API.
Fetches GBP/AUD OHLCV on 4 timeframes + pair data for currency strength.

Rate limits (free tier): 8 credits/minute, 800 credits/day.
Each symbol per request = 1 credit.  We stay safe by:
  • Fetching the 4 GBP/AUD timeframes one-by-one with RATE_SLEEP between each.
  • Fetching all 12 strength pairs in ONE batch request (comma-separated
    symbols) — 12 credits, but only 1 HTTP request against the rate limit.
Total HTTP requests per analysis run: 5  (4 timeframes + 1 strength batch).
"""

import time
import requests
import pandas as pd

BASE_URL = "https://api.twelvedata.com"

# Twelve Data free tier: 8 requests/minute → need ≥7.5 s between requests.
# We use 8 s for safety.
RATE_SLEEP = 8.0

GBP_PAIRS = ["GBP/USD", "GBP/EUR", "GBP/JPY", "GBP/CHF", "GBP/CAD", "GBP/NZD"]
AUD_PAIRS = ["AUD/USD", "AUD/EUR", "AUD/JPY", "AUD/CHF", "AUD/CAD", "AUD/NZD"]
ALL_STRENGTH_PAIRS = GBP_PAIRS + AUD_PAIRS

TIMEFRAME_MAP = {
    "Daily": "1day",
    "H4":    "4h",
    "H1":    "1h",
    "M5":    "5min",
}


# ─── Parsers ─────────────────────────────────────────────────────────────────

def _parse_response(response_json, symbol, interval="?"):
    """
    Parse a Twelve Data time_series response dict into a clean DataFrame.
    Returns None on API-level or data errors — never raises.
    Does NOT call any st.* functions (safe to call from inside run_analysis).
    """
    if not isinstance(response_json, dict):
        print(f"[API ERROR] {symbol} {interval}: non-dict response: {type(response_json)}", flush=True)
        return None

    if "values" not in response_json:
        code = response_json.get("code", "?")
        msg  = response_json.get("message", "Unknown error")
        print(
            f"[API ERROR] {symbol} {interval}: code={code} msg={msg}",
            flush=True,
        )
        return None

    rows = response_json["values"]
    if not rows:
        print(f"[API EMPTY] {symbol} {interval}: values list is empty", flush=True)
        return None

    print(
        f"[API OK] {symbol} {interval}: {len(rows)} bars | "
        f"newest={rows[0].get('datetime','?')} | "
        f"sample close={rows[0].get('close','?')}",
        flush=True,
    )

    try:
        df = pd.DataFrame(rows)
        print(f"[PARSE 1] {symbol} {interval}: DataFrame created, shape={df.shape}", flush=True)

        df["datetime"] = pd.to_datetime(df["datetime"])
        print(f"[PARSE 2] {symbol} {interval}: datetime parsed", flush=True)

        df = df.sort_values("datetime").reset_index(drop=True)
        print(f"[PARSE 3] {symbol} {interval}: sorted", flush=True)

        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        print(f"[PARSE 4] {symbol} {interval}: numeric cast done", flush=True)

        df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
        print(f"[PARSE 5] {symbol} {interval}: {len(df)} clean rows after dropna", flush=True)

        return df if len(df) > 0 else None

    except Exception as exc:
        print(f"[PARSE ERR] {symbol} {interval}: {exc}", flush=True)
        return None


# ─── Single-symbol fetch ─────────────────────────────────────────────────────

def fetch_ohlcv(symbol: str, interval: str, outputsize: int = 150, api_key: str = "") -> pd.DataFrame | None:
    """Fetch OHLCV candles from Twelve Data. Returns None on any error."""
    if not api_key:
        return None
    params = {
        "symbol":     symbol,
        "interval":   interval,
        "outputsize": outputsize,
        "apikey":     api_key,
        "format":     "JSON",
    }
    print(f"[FETCH] Requesting {symbol} {interval} outputsize={outputsize}", flush=True)
    try:
        r = requests.get(f"{BASE_URL}/time_series", params=params, timeout=20)
        r.raise_for_status()
        return _parse_response(r.json(), symbol, interval)
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "?"
        print(f"[HTTP ERR] {symbol} {interval}: HTTP {status} — {exc}", flush=True)
        return None
    except requests.RequestException as exc:
        print(f"[NET ERR] {symbol} {interval}: {exc}", flush=True)
        return None


# ─── Batch fetch (multiple symbols, same interval/outputsize) ─────────────────

def fetch_batch(symbols: list[str], interval: str, outputsize: int, api_key: str) -> dict[str, pd.DataFrame | None]:
    """
    Fetch multiple symbols in one HTTP request using Twelve Data's batch endpoint.
    Costs 1 credit per symbol but only 1 HTTP request against the rate limit.
    Returns dict keyed by symbol (e.g. "GBP/USD").
    """
    if not api_key or not symbols:
        return {s: None for s in symbols}

    symbol_str = ",".join(symbols)
    params = {
        "symbol":     symbol_str,
        "interval":   interval,
        "outputsize": outputsize,
        "apikey":     api_key,
        "format":     "JSON",
    }
    print(f"[BATCH FETCH] {len(symbols)} pairs @ {interval}: {symbol_str}", flush=True)
    try:
        r = requests.get(f"{BASE_URL}/time_series", params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "?"
        print(f"[BATCH HTTP ERR] HTTP {status} — {exc}", flush=True)
        return {s: None for s in symbols}
    except requests.RequestException as exc:
        print(f"[BATCH NET ERR] {exc}", flush=True)
        return {s: None for s in symbols}

    # When only 1 symbol is requested, the API returns the object directly.
    # When multiple symbols, it returns {"GBP/USD": {...}, "EUR/USD": {...}}.
    results = {}
    if len(symbols) == 1:
        results[symbols[0]] = _parse_response(data, symbols[0], interval)
    else:
        for sym in symbols:
            sym_data = data.get(sym, {})
            results[sym] = _parse_response(sym_data, sym, interval)
    return results


# ─── High-level fetchers ──────────────────────────────────────────────────────

def fetch_all_timeframes(api_key: str) -> dict[str, pd.DataFrame | None]:
    """
    Fetch GBP/AUD on Daily, H4, H1, M5 with rate-limit spacing.
    4 HTTP requests, RATE_SLEEP seconds apart.
    """
    results = {}
    items = list(TIMEFRAME_MAP.items())
    for i, (name, code) in enumerate(items):
        results[name] = fetch_ohlcv("GBP/AUD", code, outputsize=200, api_key=api_key)
        if i < len(items) - 1:          # don't sleep after the last one
            print(f"[RATE] Sleeping {RATE_SLEEP}s before next request…", flush=True)
            time.sleep(RATE_SLEEP)
    return results


def fetch_strength_data(api_key: str) -> dict[str, pd.DataFrame | None]:
    """
    Fetch H1 data for all 12 GBP/AUD cross-pairs in ONE batch HTTP request.
    Uses 12 credits but counts as only 1 request against the rate limit.
    """
    print(f"[RATE] Sleeping {RATE_SLEEP}s before strength batch request…", flush=True)
    time.sleep(RATE_SLEEP)
    return fetch_batch(ALL_STRENGTH_PAIRS, interval="1h", outputsize=30, api_key=api_key)


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
        pct = (
            (df["close"].iloc[-1] - df["close"].iloc[-1 - lookback])
            / df["close"].iloc[-1 - lookback]
            * 100
        )
        raw.setdefault(base, []).append(pct)
        raw.setdefault(quote, []).append(-pct)

    strength = {cur: sum(v) / len(v) for cur, v in raw.items() if v}

    # Normalise to [-100, +100]
    max_abs = max((abs(v) for v in strength.values()), default=1) or 1
    return {cur: round(val / max_abs * 100, 2) for cur, val in strength.items()}

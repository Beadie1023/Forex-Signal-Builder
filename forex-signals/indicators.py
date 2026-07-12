"""
Technical indicators computed from raw OHLCV DataFrames.
All functions accept pandas Series/DataFrame columns and return Series.
"""

import numpy as np
import pandas as pd


# ─── Heiken Ashi ─────────────────────────────────────────────────────────────

def heiken_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """Return a new DataFrame with ha_open, ha_high, ha_low, ha_close columns."""
    ha = df.copy()
    ha["ha_close"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4

    ha_open = np.zeros(len(df))
    ha_open[0] = (df["open"].iloc[0] + df["close"].iloc[0]) / 2
    for i in range(1, len(df)):
        ha_open[i] = (ha_open[i - 1] + ha["ha_close"].iloc[i - 1]) / 2
    ha["ha_open"] = ha_open

    ha["ha_high"] = ha[["high", "ha_open", "ha_close"]].max(axis=1)
    ha["ha_low"] = ha[["low", "ha_open", "ha_close"]].min(axis=1)

    ha["ha_bullish"] = ha["ha_close"] > ha["ha_open"]
    return ha


# ─── RSI ─────────────────────────────────────────────────────────────────────

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ─── Williams %R ─────────────────────────────────────────────────────────────

def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    hh = high.rolling(period).max()
    ll = low.rolling(period).min()
    return -100 * (hh - close) / (hh - ll).replace(0, np.nan)


# ─── ADX ─────────────────────────────────────────────────────────────────────

def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return (adx, plus_di, minus_di) as a tuple of Series."""
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

    atr_s = tr.ewm(span=period, adjust=False).mean()
    plus_di_s = 100 * plus_dm.ewm(span=period, adjust=False).mean() / atr_s.replace(0, np.nan)
    minus_di_s = 100 * minus_dm.ewm(span=period, adjust=False).mean() / atr_s.replace(0, np.nan)

    dx = 100 * (plus_di_s - minus_di_s).abs() / (plus_di_s + minus_di_s).replace(0, np.nan)
    adx_s = dx.ewm(span=period, adjust=False).mean()
    return adx_s, plus_di_s, minus_di_s


# ─── CCI ─────────────────────────────────────────────────────────────────────

def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    tp = (high + low + close) / 3
    mean_tp = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - mean_tp) / (0.015 * mad.replace(0, np.nan))


# ─── ATR ─────────────────────────────────────────────────────────────────────

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


# ─── ROC ─────────────────────────────────────────────────────────────────────

def roc(close: pd.Series, period: int = 10) -> pd.Series:
    return (close - close.shift(period)) / close.shift(period).replace(0, np.nan) * 100


# ─── Support & Resistance ────────────────────────────────────────────────────

def pivot_points(high_val: float, low_val: float, close_val: float) -> dict:
    """Classic pivot points from the last completed daily candle."""
    pp = (high_val + low_val + close_val) / 3
    r1 = 2 * pp - low_val
    s1 = 2 * pp - high_val
    r2 = pp + (high_val - low_val)
    s2 = pp - (high_val - low_val)
    r3 = high_val + 2 * (pp - low_val)
    s3 = low_val - 2 * (high_val - pp)
    return {"PP": pp, "R1": r1, "R2": r2, "R3": r3, "S1": s1, "S2": s2, "S3": s3}


def swing_levels(df: pd.DataFrame, window: int = 10, n_levels: int = 3) -> dict:
    """Detect recent swing highs and lows."""
    highs, lows = [], []
    for i in range(window, len(df) - window):
        if df["high"].iloc[i] == df["high"].iloc[i - window: i + window + 1].max():
            highs.append(df["high"].iloc[i])
        if df["low"].iloc[i] == df["low"].iloc[i - window: i + window + 1].min():
            lows.append(df["low"].iloc[i])

    resistances = sorted(set(highs), reverse=True)[:n_levels]
    supports = sorted(set(lows))[:n_levels]
    return {"resistances": resistances, "supports": supports}


# ─── Composite enrichment ────────────────────────────────────────────────────

def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Add all indicator columns to df in-place and return it."""
    if df is None or len(df) < 30:
        return df

    df = heiken_ashi(df)
    df["rsi"] = rsi(df["close"])
    df["wr"] = williams_r(df["high"], df["low"], df["close"])
    df["adx"], df["plus_di"], df["minus_di"] = adx(df["high"], df["low"], df["close"])
    df["cci"] = cci(df["high"], df["low"], df["close"])
    df["atr"] = atr(df["high"], df["low"], df["close"])
    df["roc"] = roc(df["close"])
    return df


def last(df: pd.DataFrame, col: str, default=float("nan")):
    """Safe last-value accessor."""
    if df is None or col not in df.columns or df[col].empty:
        return default
    v = df[col].iloc[-1]
    return default if pd.isna(v) else v

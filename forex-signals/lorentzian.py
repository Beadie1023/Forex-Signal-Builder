"""
Lorentzian k-NN Classifier for GBP/AUD signal generation.

Uses Lorentzian distance: d(a,b) = Σ log(1 + |a_i - b_i|)
which is more robust to outliers than Euclidean distance and
performs well in non-stationary financial time series.
"""

import numpy as np
import pandas as pd


FEATURE_COLS = ["rsi", "wr", "adx", "cci", "atr_norm", "roc"]


def _lorentzian_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(np.log1p(np.abs(a - b))))


def _min_max_norm(series: pd.Series) -> pd.Series:
    lo, hi = series.min(), series.max()
    if hi == lo:
        return series * 0
    return (series - lo) / (hi - lo)


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Build a normalised feature matrix from an enriched OHLCV DataFrame.
    Requires columns produced by indicators.enrich().
    """
    required = ["rsi", "wr", "adx", "cci", "atr", "roc"]
    if df is None or not all(c in df.columns for c in required):
        return None

    feat = pd.DataFrame(index=df.index)
    # Normalise each feature to [0, 1]
    feat["rsi"]      = _min_max_norm(df["rsi"])       # 0–100
    feat["wr"]       = _min_max_norm(df["wr"])         # -100–0
    feat["adx"]      = _min_max_norm(df["adx"])        # 0–100
    feat["cci"]      = _min_max_norm(df["cci"].clip(-200, 200))
    feat["atr_norm"] = _min_max_norm(df["atr"])
    feat["roc"]      = _min_max_norm(df["roc"].clip(-5, 5))
    return feat.dropna()


def _make_label(df: pd.DataFrame, idx: int, forward: int = 4) -> int:
    """
    Label for bar at position idx:
      +1  bullish (close rises by at least threshold over next `forward` bars)
      -1  bearish (close falls)
       0  neutral
    """
    if idx + forward >= len(df):
        return 0
    current = df["close"].iloc[idx]
    future  = df["close"].iloc[idx + forward]
    if current == 0:
        return 0
    pct = (future - current) / current * 100
    if pct > 0.15:
        return 1
    if pct < -0.15:
        return -1
    return 0


def lorentzian_knn(
    df: pd.DataFrame,
    k: int = 5,
    lookback: int = 200,
    forward_bars: int = 4,
) -> tuple[int, float, list]:
    """
    Run the Lorentzian k-NN classifier on the most recent bar.

    Returns
    -------
    signal : int
        +1 = bullish, -1 = bearish, 0 = neutral
    confidence : float
        Weighted vote fraction in [0, 1]
    neighbour_signals : list[int]
        The k nearest neighbours' labels
    """
    feat_df = build_feature_matrix(df)
    if feat_df is None or len(feat_df) < k + forward_bars + 10:
        return 0, 0.0, []

    # Use at most `lookback` historical bars (exclude the last few unlabelled ones)
    usable_end = len(feat_df) - forward_bars - 1
    train_start = max(0, usable_end - lookback)
    train_idx = list(range(train_start, usable_end))

    if len(train_idx) < k:
        return 0, 0.0, []

    current_bar = feat_df.iloc[-1].values.astype(float)

    # ── find k nearest neighbours ──────────────────────────────────────────
    distances = []
    for i in train_idx:
        bar = feat_df.iloc[i].values.astype(float)
        if np.any(np.isnan(bar)):
            continue
        d = _lorentzian_dist(current_bar, bar)
        label = _make_label(df, feat_df.index[i], forward=forward_bars)
        distances.append((d, label))

    distances.sort(key=lambda x: x[0])
    neighbours = distances[:k]

    if not neighbours:
        return 0, 0.0, []

    # Inverse-distance weighted vote
    votes = {1: 0.0, -1: 0.0, 0: 0.0}
    for dist, label in neighbours:
        weight = 1.0 / (dist + 1e-6)
        votes[label] += weight

    total = sum(votes.values())
    best_label = max(votes, key=votes.__getitem__)
    confidence = votes[best_label] / total if total > 0 else 0.0

    # Require clear majority
    if confidence < 0.45:
        best_label = 0

    neighbour_signals = [lbl for _, lbl in neighbours]
    return best_label, round(confidence, 3), neighbour_signals

"""
7-Member Council Debate Engine.
Each council member analyses a different aspect of the market and casts a
weighted vote. The final signal is determined by the weighted tally.
"""

import math
from typing import Any
import pandas as pd
from indicators import last

# ── Council roster ────────────────────────────────────────────────────────────
MEMBERS = [
    {
        "name": "The Machine",
        "role": "Heiken Ashi Analyst",
        "style": "Trend Structure",
        "weight": 2.0,
    },
    {
        "name": "The Macro King",
        "role": "ADX Trend Expert",
        "style": "Trend Strength & Direction",
        "weight": 2.0,
    },
    {
        "name": "The Level Hunter",
        "role": "Momentum Specialist",
        "style": "RSI & Rate of Change",
        "weight": 1.5,
    },
    {
        "name": "The Architect",
        "role": "Oscillator Expert",
        "style": "CCI & Williams %R",
        "weight": 1.5,
    },
    {
        "name": "The Statistician",
        "role": "Volatility Guardian",
        "style": "ATR Risk Management",
        "weight": 1.0,
    },
    {
        "name": "The Options Whisperer",
        "role": "Currency Strength",
        "style": "Cross-pair Flow Analysis",
        "weight": 2.5,
    },
    {
        "name": "The Risk Officer",
        "role": "ML Oracle",
        "style": "Lorentzian k-NN",
        "weight": 2.5,
    },
]

TOTAL_WEIGHT = sum(m["weight"] for m in MEMBERS)   # 13.0
# Threshold: >55% of total weight needed for a directional call
BUY_THRESHOLD  = TOTAL_WEIGHT * 0.55
SELL_THRESHOLD = TOTAL_WEIGHT * 0.55


# ── Individual member vote functions ─────────────────────────────────────────

def _vote_elena(tf_data: dict) -> tuple[int, str]:
    """Heiken Ashi direction across all four timeframes."""
    bullish, bearish = 0, 0
    for tf, df in tf_data.items():
        if df is None or "ha_bullish" not in df.columns:
            continue
        bull = last(df, "ha_bullish", default=None)
        if bull is True:
            bullish += 1
        elif bull is False:
            bearish += 1

    total = bullish + bearish
    if total == 0:
        return 0, "Insufficient data"
    if bullish >= 3:
        return 1, f"HA bullish on {bullish}/{total} TFs — trend structure up"
    if bearish >= 3:
        return -1, f"HA bearish on {bearish}/{total} TFs — trend structure down"
    return 0, f"Mixed HA signals ({bullish}↑ {bearish}↓) — no clear structure"


def _vote_marcus(tf_data: dict) -> tuple[int, str]:
    """ADX trend strength & direction (H4 + H1 confluence)."""
    votes_up, votes_down = 0, 0
    trending = 0
    for tf in ["H4", "H1"]:
        df = tf_data.get(tf)
        if df is None:
            continue
        adx_v   = last(df, "adx")
        plus_di = last(df, "plus_di")
        minus_di = last(df, "minus_di")
        if math.isnan(adx_v):
            continue
        if adx_v > 20:
            trending += 1
            if plus_di > minus_di:
                votes_up += 1
            else:
                votes_down += 1

    if trending == 0:
        return 0, "ADX < 20 — market is ranging, no trend entry"
    if votes_up == 2:
        return 1, f"ADX trending ({trending}/2 TFs), +DI dominant — bullish momentum"
    if votes_down == 2:
        return -1, f"ADX trending ({trending}/2 TFs), -DI dominant — bearish momentum"
    return 0, "Conflicting ADX direction between H4 and H1"


def _vote_sofia(tf_data: dict) -> tuple[int, str]:
    """RSI-14 and ROC-10 confluence on H1 & M5."""
    rsi_h1 = last(tf_data.get("H1"), "rsi")
    rsi_m5 = last(tf_data.get("M5"), "rsi")
    roc_m5 = last(tf_data.get("M5"), "roc")

    if math.isnan(rsi_h1) or math.isnan(rsi_m5):
        return 0, "RSI data unavailable"
    # Overbought / oversold = caution
    if rsi_h1 > 70 or rsi_m5 > 70:
        return 0, f"Overbought — RSI H1:{rsi_h1:.1f} M5:{rsi_m5:.1f}, caution"
    if rsi_h1 < 30 or rsi_m5 < 30:
        return 0, f"Oversold — RSI H1:{rsi_h1:.1f} M5:{rsi_m5:.1f}, caution"
    # Bullish momentum zone
    if rsi_h1 > 50 and rsi_m5 > 50 and (math.isnan(roc_m5) or roc_m5 > 0):
        return 1, f"Bullish momentum — RSI H1:{rsi_h1:.1f} M5:{rsi_m5:.1f}, ROC:{roc_m5:.2f}%"
    if rsi_h1 < 50 and rsi_m5 < 50 and (math.isnan(roc_m5) or roc_m5 < 0):
        return -1, f"Bearish momentum — RSI H1:{rsi_h1:.1f} M5:{rsi_m5:.1f}, ROC:{roc_m5:.2f}%"
    return 0, f"Neutral momentum — RSI H1:{rsi_h1:.1f} M5:{rsi_m5:.1f}"


def _vote_james(tf_data: dict) -> tuple[int, str]:
    """CCI-20 and Williams %R-14 on H1 & M5."""
    cci_h1 = last(tf_data.get("H1"), "cci")
    cci_m5 = last(tf_data.get("M5"), "cci")
    wr_h1  = last(tf_data.get("H1"), "wr")
    wr_m5  = last(tf_data.get("M5"), "wr")

    if math.isnan(cci_h1) or math.isnan(cci_m5):
        return 0, "CCI data unavailable"

    cci_bull = cci_h1 > 0 and cci_m5 > 0
    cci_bear = cci_h1 < 0 and cci_m5 < 0
    wr_bull  = not math.isnan(wr_m5) and wr_m5 > -50
    wr_bear  = not math.isnan(wr_m5) and wr_m5 < -50

    if cci_bull and wr_bull:
        return 1, f"CCI bullish (H1:{cci_h1:.0f} M5:{cci_m5:.0f}), WR:{wr_m5:.0f} above midline"
    if cci_bear and wr_bear:
        return -1, f"CCI bearish (H1:{cci_h1:.0f} M5:{cci_m5:.0f}), WR:{wr_m5:.0f} below midline"
    return 0, f"Mixed oscillators — CCI H1:{cci_h1:.0f} M5:{cci_m5:.0f}, WR:{wr_m5:.0f}"


def _vote_yuki(tf_data: dict, majority_dir: int) -> tuple[int, str]:
    """ATR volatility gate — passes the majority direction only if volatility is healthy."""
    df_m5 = tf_data.get("M5")
    if df_m5 is None:
        return 0, "M5 data unavailable"

    atr_v = last(df_m5, "atr")
    if math.isnan(atr_v):
        return 0, "ATR data unavailable"

    atr_avg = df_m5["atr"].rolling(50).mean().iloc[-1] if len(df_m5) >= 50 else atr_v
    if math.isnan(atr_avg):
        atr_avg = atr_v

    ratio = atr_v / atr_avg if atr_avg > 0 else 1.0

    if ratio > 3.0:
        return 0, f"ATR spike ({atr_v:.5f} = {ratio:.1f}x avg) — volatility too high, stand aside"
    if ratio < 0.3:
        return 0, f"ATR very low ({atr_v:.5f} = {ratio:.1f}x avg) — dead market, skip"
    if majority_dir == 1:
        return 1, f"Volatility healthy ({ratio:.1f}x avg ATR) — OK to buy"
    if majority_dir == -1:
        return -1, f"Volatility healthy ({ratio:.1f}x avg ATR) — OK to sell"
    return 0, f"Neutral volatility assessment (ATR ratio {ratio:.1f}x)"


def _vote_amara(gbp_strength: float, aud_strength: float) -> tuple[int, str]:
    """GBP vs AUD strength differential."""
    diff = gbp_strength - aud_strength
    if math.isnan(diff):
        return 0, "Currency strength data unavailable"
    if diff > 12:
        return 1, f"GBP ({gbp_strength:+.1f}) outperforming AUD ({aud_strength:+.1f}) by {diff:.1f}pts — flow favours GBP/AUD long"
    if diff < -12:
        return -1, f"AUD ({aud_strength:+.1f}) outperforming GBP ({gbp_strength:+.1f}) by {abs(diff):.1f}pts — flow favours GBP/AUD short"
    return 0, f"GBP/AUD strength differential {diff:+.1f}pts — too close to call"


def _vote_alan(knn_signal: int, confidence: float) -> tuple[int, str]:
    """Lorentzian k-NN ML Oracle."""
    if knn_signal == 1:
        return 1, f"Lorentzian k-NN: BULLISH (confidence {confidence:.1%})"
    if knn_signal == -1:
        return -1, f"Lorentzian k-NN: BEARISH (confidence {confidence:.1%})"
    return 0, f"Lorentzian k-NN: NEUTRAL (confidence {confidence:.1%})"


# ── Top-down waterfall gate ───────────────────────────────────────────────────

def _waterfall_check(tf_data: dict) -> tuple[bool, str]:
    """
    Daily → H4 → H1 → M5 alignment check.
    Returns (passed, reason).
    """
    steps = []
    direction = None   # Set by Daily HA

    # Daily gate
    df_d = tf_data.get("Daily")
    if df_d is None or "ha_bullish" not in df_d.columns:
        return False, "Daily data missing"
    daily_bull = last(df_d, "ha_bullish", default=None)
    if daily_bull is None:
        return False, "Daily HA undetermined"
    direction = 1 if daily_bull else -1
    steps.append(f"Daily HA: {'▲ Bullish' if direction == 1 else '▼ Bearish'}")

    # H4 gate — must confirm Daily + ADX > 20
    df_h4 = tf_data.get("H4")
    if df_h4 is None:
        return False, "H4 data missing"
    h4_bull = last(df_h4, "ha_bullish", default=None)
    h4_adx  = last(df_h4, "adx", default=0)
    if h4_bull is None or (1 if h4_bull else -1) != direction:
        return False, f"H4 HA conflicts with Daily direction\n" + "\n".join(steps)
    if h4_adx < 18:
        return False, f"H4 ADX={h4_adx:.1f} < 18 — trend not established\n" + "\n".join(steps)
    steps.append(f"H4 HA confirms + ADX {h4_adx:.1f}")

    # H1 gate — RSI alignment
    df_h1 = tf_data.get("H1")
    if df_h1 is None:
        return False, "H1 data missing"
    h1_rsi = last(df_h1, "rsi", default=50)
    if direction == 1 and h1_rsi < 45:
        return False, f"H1 RSI={h1_rsi:.1f} too weak for bullish entry\n" + "\n".join(steps)
    if direction == -1 and h1_rsi > 55:
        return False, f"H1 RSI={h1_rsi:.1f} too strong for bearish entry\n" + "\n".join(steps)
    steps.append(f"H1 RSI={h1_rsi:.1f} aligned")

    # M5 gate — HA candle aligns
    df_m5 = tf_data.get("M5")
    if df_m5 is None:
        return False, "M5 data missing"
    m5_bull = last(df_m5, "ha_bullish", default=None)
    if m5_bull is None or (1 if m5_bull else -1) != direction:
        return False, f"M5 HA not yet aligned — wait for entry candle\n" + "\n".join(steps)
    steps.append("M5 HA entry candle aligned ✓")

    return True, "Waterfall passed:\n" + "\n".join(steps)


# ── Main debate function ──────────────────────────────────────────────────────

def run_council(
    tf_data: dict,
    gbp_strength: float,
    aud_strength: float,
    knn_signal: int,
    knn_confidence: float,
) -> dict[str, Any]:
    """
    Run the full 7-member council debate.

    Returns a dict with:
      signal        : "BUY" | "SELL" | "NO TRADE"
      direction     : 1 | -1 | 0
      buy_score     : float
      sell_score    : float
      waterfall_ok  : bool
      waterfall_msg : str
      member_votes  : list of dicts
    """
    # ── Waterfall prerequisite ────────────────────────────────────────────────
    wf_ok, wf_msg = _waterfall_check(tf_data)

    # ── Preliminary majority (for Yuki's volatility gate) ────────────────────
    prelim_buy = prelim_sell = 0.0
    prelim_votes = [
        _vote_elena(tf_data),
        _vote_marcus(tf_data),
        _vote_sofia(tf_data),
        _vote_james(tf_data),
        _vote_amara(gbp_strength, aud_strength),
        _vote_alan(knn_signal, knn_confidence),
    ]
    weights_prelim = [MEMBERS[i]["weight"] for i in range(6)]
    for (sig, _), w in zip(prelim_votes, weights_prelim):
        if sig == 1:
            prelim_buy += w
        elif sig == -1:
            prelim_sell += w
    majority_dir = 1 if prelim_buy > prelim_sell else (-1 if prelim_sell > prelim_buy else 0)

    # ── Full vote ─────────────────────────────────────────────────────────────
    all_votes = prelim_votes[:4] + [_vote_yuki(tf_data, majority_dir)] + prelim_votes[4:]

    member_votes = []
    buy_score = sell_score = 0.0
    for member, (sig, reason) in zip(MEMBERS, all_votes):
        w = member["weight"]
        label = "BUY" if sig == 1 else ("SELL" if sig == -1 else "HOLD")
        member_votes.append({
            "name":   member["name"],
            "role":   member["role"],
            "weight": w,
            "vote":   label,
            "reason": reason,
        })
        if sig == 1:
            buy_score += w
        elif sig == -1:
            sell_score += w

    # ── Final signal ──────────────────────────────────────────────────────────
    if not wf_ok:
        signal = "NO TRADE"
        direction = 0
    elif buy_score >= BUY_THRESHOLD and buy_score > sell_score:
        signal = "BUY"
        direction = 1
    elif sell_score >= SELL_THRESHOLD and sell_score > buy_score:
        signal = "SELL"
        direction = -1
    else:
        signal = "NO TRADE"
        direction = 0

    return {
        "signal":        signal,
        "direction":     direction,
        "buy_score":     round(buy_score, 2),
        "sell_score":    round(sell_score, 2),
        "total_weight":  TOTAL_WEIGHT,
        "waterfall_ok":  wf_ok,
        "waterfall_msg": wf_msg,
        "member_votes":  member_votes,
    }

"""
GBP/AUD Forex Trading Signal App
Powered by Twelve Data API · Streamlit · Lorentzian k-NN · 7-Member Council
"""

import math
import time
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import indicators as ind
import lorentzian as lor
import trade_journal as tj
from council import run_council
from data_fetcher import (
    fetch_all_timeframes,
    fetch_strength_data,
    calculate_currency_strength,
)

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GBP/AUD Signal",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

tj.init_db()

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📡 GBP/AUD Signal")
    st.caption("Powered by Twelve Data + Lorentzian k-NN")
    st.divider()

    api_key = st.text_input(
        "Twelve Data API Key",
        type="password",
        placeholder="Paste your free API key here",
        help="Get a free key at twelvedata.com — 800 credits/day included.",
    )

    st.divider()
    st.markdown("**Settings**")
    knn_k = st.slider("k-NN neighbours (k)", 3, 15, 5, step=1)
    knn_lookback = st.slider("k-NN training bars", 50, 300, 150, step=25)
    atr_sl_mult = st.slider("ATR × SL multiplier", 1.0, 4.0, 2.0, step=0.5)
    tp1_rr = st.slider("TP1 R:R", 1.0, 3.0, 1.5, step=0.5)
    tp2_rr = st.slider("TP2 R:R", 2.0, 6.0, 3.0, step=0.5)

    st.divider()
    run_btn = st.button("🔄 Analyse Now", use_container_width=True, type="primary")
    st.caption("Auto-refreshes every 5 min")

# ─── Session state ────────────────────────────────────────────────────────────
if "analysis" not in st.session_state:
    st.session_state.analysis = None
if "last_run" not in st.session_state:
    st.session_state.last_run = None
if "_autorefresh_pending" not in st.session_state:
    st.session_state._autorefresh_pending = False

# ─── Native auto-refresh (no custom component) ───────────────────────────────
# st.fragment with run_every re-runs only this fragment every 5 minutes.
# It sets a flag and calls st.rerun() so the main script picks it up.
@st.fragment(run_every=300)
def _autorefresh_heartbeat():
    # Silently set the pending flag; main script will consume it below.
    if st.session_state.analysis is not None and api_key:
        st.session_state._autorefresh_pending = True
        st.rerun()

_autorefresh_heartbeat()

# ─── Helper: run full analysis ────────────────────────────────────────────────

def run_analysis(api_key: str):
    results = {}

    with st.status("Fetching data from Twelve Data…", expanded=True) as status:
        st.write("📥 Downloading GBP/AUD on 4 timeframes…")
        tf_raw = fetch_all_timeframes(api_key)
        results["tf_raw"] = tf_raw

        st.write("📊 Computing technical indicators…")
        # Guard: enrich() requires a non-None DataFrame; skip missing timeframes.
        tf_data = {}
        missing_tfs = []
        for tf, df in tf_raw.items():
            if df is not None and len(df) > 0:
                tf_data[tf] = ind.enrich(df)
            else:
                tf_data[tf] = None
                missing_tfs.append(tf)
        if missing_tfs:
            print(f"[WARN] Missing timeframes (API error or rate limit): {missing_tfs}", flush=True)
        results["tf_data"] = tf_data

        st.write("🌐 Fetching cross-pair data for currency strength…")
        pair_data = fetch_strength_data(api_key)
        strength = calculate_currency_strength(pair_data)
        results["strength"] = strength
        results["pair_data"] = pair_data

        st.write("🤖 Running Lorentzian k-NN classifier…")
        df_h1 = tf_data.get("H1")
        knn_sig, knn_conf, knn_nbrs = (0, 0.0, []) if df_h1 is None else lor.lorentzian_knn(
            df_h1, k=knn_k, lookback=knn_lookback
        )
        results["knn"] = {"signal": knn_sig, "confidence": knn_conf, "neighbours": knn_nbrs}

        st.write("🏛️ Running 7-member council debate…")
        gbp_s = strength.get("GBP", 0.0)
        aud_s = strength.get("AUD", 0.0)
        council = run_council(tf_data, gbp_s, aud_s, knn_sig, knn_conf)
        results["council"] = council

        # ── Entry / SL / TP ──────────────────────────────────────────────────
        df_m5 = tf_data.get("M5")
        entry = ind.last(df_m5, "close", default=float("nan"))
        atr_v = ind.last(df_m5, "atr", default=float("nan"))
        direction = council["direction"]

        if not math.isnan(entry) and not math.isnan(atr_v) and direction != 0:
            sl_dist = atr_v * atr_sl_mult
            stop_loss = entry - sl_dist * direction
            tp1 = entry + sl_dist * tp1_rr * direction
            tp2 = entry + sl_dist * tp2_rr * direction
        else:
            stop_loss = tp1 = tp2 = float("nan")

        results["entry"]      = entry
        results["stop_loss"]  = stop_loss
        results["tp1"]        = tp1
        results["tp2"]        = tp2

        # ── Support & Resistance ─────────────────────────────────────────────
        df_daily = tf_data.get("Daily")
        pivots = {}
        swings = {"resistances": [], "supports": []}
        if df_daily is not None and len(df_daily) >= 2:
            prev = df_daily.iloc[-2]
            pivots = ind.pivot_points(prev["high"], prev["low"], prev["close"])
            swings = ind.swing_levels(df_daily, window=5, n_levels=3)
        results["pivots"] = pivots
        results["swings"] = swings

        status.update(label="✅ Analysis complete!", state="complete", expanded=False)

    results["timestamp"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    return results


# ─── Trigger analysis ─────────────────────────────────────────────────────────
# Consume the auto-refresh flag (set by _autorefresh_heartbeat fragment).
_autorefresh_fired = st.session_state._autorefresh_pending
if _autorefresh_fired:
    st.session_state._autorefresh_pending = False

if run_btn or _autorefresh_fired:
    if not api_key:
        st.warning("Please enter your Twelve Data API key in the sidebar to run analysis.")
    else:
        try:
            st.session_state.analysis = run_analysis(api_key)
            st.session_state.last_run = datetime.now(datetime.UTC)
        except Exception as _exc:
            import traceback as _tb
            print(f"[ANALYSIS CRASH] {_exc}", flush=True)
            _tb.print_exc()
            st.error(f"Analysis failed — see console for traceback. Error: {_exc}")

analysis = st.session_state.analysis

# ─── Landing state ───────────────────────────────────────────────────────────
if analysis is None:
    st.markdown("## 📡 GBP/AUD Forex Signal Dashboard")
    st.info(
        "Enter your **Twelve Data API key** in the sidebar, then click **Analyse Now** to generate a signal.\n\n"
        "Get a free key at [twelvedata.com](https://twelvedata.com) — 800 credits/day included.",
        icon="ℹ️",
    )
    col1, col2, col3 = st.columns(3)
    col1.metric("Timeframes", "Daily · H4 · H1 · M5")
    col2.metric("Council Members", "7")
    col3.metric("Auto-refresh", "Every 5 min")
    st.stop()

# ─── Main layout ─────────────────────────────────────────────────────────────
council   = analysis["council"]
tf_data   = analysis["tf_data"]
strength  = analysis["strength"]
entry     = analysis["entry"]
stop_loss = analysis["stop_loss"]
tp1       = analysis["tp1"]
tp2       = analysis["tp2"]
pivots    = analysis["pivots"]
swings    = analysis["swings"]
knn       = analysis["knn"]

signal    = council["signal"]
direction = council["direction"]

# ── Timestamp strip ───────────────────────────────────────────────────────────
ts_col, lbl_col = st.columns([3, 1])
ts_col.caption(f"Last updated: {analysis['timestamp']} — Next auto-refresh in ~5 min")
lbl_col.caption(f"k-NN confidence: {knn['confidence']:.1%}")

# ─── Tab layout ──────────────────────────────────────────────────────────────
tab_signal, tab_chart, tab_council, tab_strength, tab_journal = st.tabs(
    ["📊 Signal", "📈 Chart", "🏛️ Council", "💹 Strength", "📒 Journal"]
)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — SIGNAL CARD
# ═══════════════════════════════════════════════════════════════════════════
with tab_signal:
    # Signal header
    colour = {"BUY": "🟢", "SELL": "🔴", "NO TRADE": "⚪"}[signal]
    st.markdown(
        f"""
        <div style="
            border-radius:12px;
            padding:32px 24px;
            margin-bottom:24px;
            background:{'#0d2b12' if signal == 'BUY' else ('#2b0d0d' if signal == 'SELL' else '#1e1e1e')};
            border: 2px solid {'#00c853' if signal == 'BUY' else ('#ff1744' if signal == 'SELL' else '#555')};
            text-align:center;
        ">
            <div style="font-size:64px;margin-bottom:8px">{colour}</div>
            <div style="font-size:52px;font-weight:800;letter-spacing:4px;color:{'#00e676' if signal == 'BUY' else ('#ff5252' if signal == 'SELL' else '#aaa')}">{signal}</div>
            <div style="font-size:16px;color:#888;margin-top:8px">GBP/AUD</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Trade levels
    if direction != 0 and not math.isnan(entry):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Entry Price",  f"{entry:.5f}")
        c2.metric("Stop Loss",    f"{stop_loss:.5f}",
                  delta=f"{(stop_loss - entry)*10000:.1f} pips",
                  delta_color="off")
        c3.metric("Take Profit 1", f"{tp1:.5f}",
                  delta=f"{(tp1 - entry)*10000:.1f} pips",
                  delta_color="normal" if direction == 1 else "inverse")
        c4.metric("Take Profit 2", f"{tp2:.5f}",
                  delta=f"{(tp2 - entry)*10000:.1f} pips",
                  delta_color="normal" if direction == 1 else "inverse")

        # R:R info
        if not math.isnan(stop_loss):
            sl_pips = abs(entry - stop_loss) * 10000
            st.caption(
                f"SL distance: {sl_pips:.1f} pips · "
                f"TP1 R:R 1:{tp1_rr} · TP2 R:R 1:{tp2_rr} · "
                f"ATR-based SL ({atr_sl_mult}× ATR)"
            )
    else:
        st.info("No trade levels — wait for a BUY or SELL signal before setting orders.")

    # Waterfall status
    st.divider()
    st.markdown("**Top-Down Waterfall Status**")
    if council["waterfall_ok"]:
        st.success(council["waterfall_msg"])
    else:
        st.error(council["waterfall_msg"])

    # Risk Officer veto callout (Signal tab)
    if council.get("risk_officer_vetoed", False):
        st.markdown(
            "<div style='background:#3d0000;border:1px solid #ff1744;border-radius:8px;"
            "padding:10px 14px;margin-top:8px;color:#ff8a80;font-size:13px'>"
            "🛑 <strong style='color:#ff1744'>Risk Officer Veto active</strong> — "
            "The Risk Officer returned NEUTRAL. Signal overridden to NO TRADE. "
            "See the Council tab for the full transcript."
            "</div>",
            unsafe_allow_html=True,
        )

    # Council score bar
    st.divider()
    st.markdown("**Council Weighted Score**")
    total_w = council["total_weight"]
    buy_w   = council["buy_score"]
    sell_w  = council["sell_score"]
    neutral_w = total_w - buy_w - sell_w

    bar_col1, bar_col2, bar_col3 = st.columns(3)
    bar_col1.metric("BUY votes",  f"{buy_w:.1f} / {total_w:.0f}", f"{buy_w/total_w:.0%}")
    bar_col2.metric("SELL votes", f"{sell_w:.1f} / {total_w:.0f}", f"{sell_w/total_w:.0%}")
    bar_col3.metric("HOLD votes", f"{neutral_w:.1f} / {total_w:.0f}", f"{neutral_w/total_w:.0%}")

    # Heiken Ashi status per timeframe
    st.divider()
    st.markdown("**Heiken Ashi Status by Timeframe**")
    ha_cols = st.columns(4)
    for col, tf in zip(ha_cols, ["Daily", "H4", "H1", "M5"]):
        df = tf_data.get(tf)
        bull = None
        rows = 0
        if df is not None:
            rows = len(df)
            # Use pre-computed ha_bullish (from enrich); fall back to
            # computing HA on-the-fly so we always show a direction even
            # when the full indicator suite couldn't run.
            bull = ind.last(df, "ha_bullish", default=None)
            if bull is None and rows >= 2:
                ha_df = ind.heiken_ashi(df)
                bull = ind.last(ha_df, "ha_bullish", default=None)
        if bull is True:
            icon, delta = "🟢 Bullish", f"{rows} bars"
        elif bull is False:
            icon, delta = "🔴 Bearish", f"{rows} bars"
        else:
            icon, delta = "⚪ N/A", f"{'0' if df is None else rows} bars"
        col.metric(tf, icon, delta)

    # ── Data debug expander ──────────────────────────────────────────────────
    with st.expander("🔬 Data Debug — raw fetch summary", expanded=False):
        st.caption(
            "Check here if any timeframe shows N/A. "
            "Console logs (workflow stdout) have the full API response."
        )
        debug_rows = []
        for tf in ["Daily", "H4", "H1", "M5"]:
            df = tf_data.get(tf)
            if df is None:
                debug_rows.append({
                    "TF": tf, "Rows": 0, "ha_bullish col": "missing",
                    "Last ha_bullish": "—", "Last close": "—",
                    "First datetime": "—", "Last datetime": "—",
                })
            else:
                ha_col = "yes" if "ha_bullish" in df.columns else "no (HA fallback)"
                bull_v = ind.last(df, "ha_bullish", default=None)
                if bull_v is None and len(df) >= 2:
                    ha_df = ind.heiken_ashi(df)
                    bull_v = ind.last(ha_df, "ha_bullish", default=None)
                debug_rows.append({
                    "TF": tf,
                    "Rows": len(df),
                    "ha_bullish col": ha_col,
                    "Last ha_bullish": ("True" if bull_v is True else ("False" if bull_v is False else "None")),
                    "Last close": f"{ind.last(df, 'close'):.5f}",
                    "First datetime": str(df["datetime"].iloc[0])[:16],
                    "Last datetime":  str(df["datetime"].iloc[-1])[:16],
                })
        st.dataframe(debug_rows, use_container_width=True)

    # Save to journal button
    st.divider()
    if direction != 0 and not math.isnan(entry):
        with st.expander("📒 Save this signal to Journal"):
            notes = st.text_area("Notes (optional)", placeholder="e.g. London session setup, news pending…")
            if st.button("Save Signal", type="primary"):
                tj.save_signal(
                    signal=signal,
                    entry_price=entry,
                    stop_loss=stop_loss,
                    take_profit1=tp1,
                    take_profit2=tp2,
                    gbp_strength=strength.get("GBP", 0),
                    aud_strength=strength.get("AUD", 0),
                    waterfall="PASS" if council["waterfall_ok"] else "FAIL",
                    council_buy=buy_w,
                    council_sell=sell_w,
                    notes=notes,
                )
                st.success("Signal saved to journal ✓")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — CHART
# ═══════════════════════════════════════════════════════════════════════════
with tab_chart:
    tf_select = st.selectbox("Timeframe", ["M5", "H1", "H4", "Daily"], index=0)
    df_chart = tf_data.get(tf_select)

    if df_chart is None or len(df_chart) < 5:
        st.warning(f"No data available for {tf_select}. Check your API key and try again.")
    else:
        # Keep last 100 bars for readability
        dc = df_chart.tail(100).reset_index(drop=True)

        fig = go.Figure()

        # ── Candlestick (raw OHLC) ────────────────────────────────────────
        fig.add_trace(go.Candlestick(
            x=dc["datetime"],
            open=dc["open"], high=dc["high"], low=dc["low"], close=dc["close"],
            name="OHLC",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
            opacity=0.4,
        ))

        # ── Heiken Ashi overlay ───────────────────────────────────────────
        if "ha_open" in dc.columns:
            ha_colours = ["#00c853" if b else "#ff1744"
                          for b in dc["ha_bullish"].fillna(False)]
            fig.add_trace(go.Candlestick(
                x=dc["datetime"],
                open=dc["ha_open"], high=dc["ha_high"],
                low=dc["ha_low"],   close=dc["ha_close"],
                name="Heiken Ashi",
                increasing_line_color="#00c853",
                decreasing_line_color="#ff1744",
                increasing_fillcolor="#00c853",
                decreasing_fillcolor="#ff1744",
            ))

        # ── Pivot points ──────────────────────────────────────────────────
        pv_styles = {
            "PP": ("white", "dash"),
            "R1": ("#ef9a9a", "dot"),  "R2": ("#e53935", "dot"),  "R3": ("#b71c1c", "dot"),
            "S1": ("#a5d6a7", "dot"),  "S2": ("#43a047", "dot"),  "S3": ("#1b5e20", "dot"),
        }
        for key, val in pivots.items():
            colour_pv, dash_pv = pv_styles.get(key, ("grey", "dash"))
            fig.add_hline(
                y=val, line_dash=dash_pv, line_color=colour_pv, opacity=0.7,
                annotation_text=f" {key} {val:.5f}", annotation_position="right",
                annotation_font_size=10,
            )

        # ── Swing levels ──────────────────────────────────────────────────
        for r in swings.get("resistances", []):
            fig.add_hline(y=r, line_dash="dashdot", line_color="#ff8f00", opacity=0.5,
                          annotation_text=f" Res {r:.5f}", annotation_position="right",
                          annotation_font_size=10)
        for s in swings.get("supports", []):
            fig.add_hline(y=s, line_dash="dashdot", line_color="#4fc3f7", opacity=0.5,
                          annotation_text=f" Sup {s:.5f}", annotation_position="right",
                          annotation_font_size=10)

        # ── Trade levels ──────────────────────────────────────────────────
        if direction != 0 and not math.isnan(entry):
            fig.add_hline(y=entry,     line_color="#ffffff", line_width=2,
                          annotation_text=f" Entry {entry:.5f}", annotation_position="right")
            fig.add_hline(y=stop_loss, line_color="#ff1744", line_width=2, line_dash="dot",
                          annotation_text=f" SL {stop_loss:.5f}", annotation_position="right")
            fig.add_hline(y=tp1,       line_color="#69f0ae", line_width=1, line_dash="dot",
                          annotation_text=f" TP1 {tp1:.5f}", annotation_position="right")
            fig.add_hline(y=tp2,       line_color="#00c853", line_width=2, line_dash="dot",
                          annotation_text=f" TP2 {tp2:.5f}", annotation_position="right")

        fig.update_layout(
            title=f"GBP/AUD — {tf_select}  |  Heiken Ashi + Pivots + S/R",
            xaxis_title="Time",
            yaxis_title="Price",
            xaxis_rangeslider_visible=False,
            template="plotly_dark",
            height=650,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        st.plotly_chart(fig, use_container_width=True)

        # ── Indicator sub-plots ───────────────────────────────────────────
        st.markdown("**Technical Indicators**")
        ic1, ic2 = st.columns(2)

        if "rsi" in dc.columns:
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=dc["datetime"], y=dc["rsi"], name="RSI-14", line_color="#ffab40"))
            fig_rsi.add_hline(y=70, line_dash="dot", line_color="#ff5252", opacity=0.6)
            fig_rsi.add_hline(y=30, line_dash="dot", line_color="#69f0ae", opacity=0.6)
            fig_rsi.add_hline(y=50, line_dash="dash", line_color="#888", opacity=0.4)
            fig_rsi.update_layout(title="RSI-14", template="plotly_dark", height=200,
                                  margin=dict(t=30, b=20), showlegend=False)
            ic1.plotly_chart(fig_rsi, use_container_width=True)

        if "cci" in dc.columns:
            fig_cci = go.Figure()
            fig_cci.add_trace(go.Scatter(x=dc["datetime"], y=dc["cci"], name="CCI-20", line_color="#ce93d8"))
            fig_cci.add_hline(y=100, line_dash="dot", line_color="#ff5252", opacity=0.6)
            fig_cci.add_hline(y=-100, line_dash="dot", line_color="#69f0ae", opacity=0.6)
            fig_cci.add_hline(y=0, line_dash="dash", line_color="#888", opacity=0.4)
            fig_cci.update_layout(title="CCI-20", template="plotly_dark", height=200,
                                  margin=dict(t=30, b=20), showlegend=False)
            ic2.plotly_chart(fig_cci, use_container_width=True)

        ic3, ic4 = st.columns(2)
        if "adx" in dc.columns:
            fig_adx = go.Figure()
            fig_adx.add_trace(go.Scatter(x=dc["datetime"], y=dc["adx"], name="ADX", line_color="#ffffff"))
            fig_adx.add_trace(go.Scatter(x=dc["datetime"], y=dc["plus_di"], name="+DI", line_color="#69f0ae"))
            fig_adx.add_trace(go.Scatter(x=dc["datetime"], y=dc["minus_di"], name="-DI", line_color="#ff5252"))
            fig_adx.add_hline(y=20, line_dash="dot", line_color="#888", opacity=0.5)
            fig_adx.update_layout(title="ADX-14", template="plotly_dark", height=200,
                                  margin=dict(t=30, b=20))
            ic3.plotly_chart(fig_adx, use_container_width=True)

        if "wr" in dc.columns:
            fig_wr = go.Figure()
            fig_wr.add_trace(go.Scatter(x=dc["datetime"], y=dc["wr"], name="Williams %R", line_color="#80deea"))
            fig_wr.add_hline(y=-20, line_dash="dot", line_color="#ff5252", opacity=0.6)
            fig_wr.add_hline(y=-80, line_dash="dot", line_color="#69f0ae", opacity=0.6)
            fig_wr.update_layout(title="Williams %R-14", template="plotly_dark", height=200,
                                 margin=dict(t=30, b=20), showlegend=False)
            ic4.plotly_chart(fig_wr, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — COUNCIL DEBATE
# ═══════════════════════════════════════════════════════════════════════════
with tab_council:
    st.markdown("### 🏛️ Council Debate — Full Transcript")
    st.caption(
        "Seven independent analysts review the market from different angles. "
        "Each casts a weighted vote. A directional signal requires >55% weighted consensus."
    )

    # ── Risk Officer veto banner ───────────────────────────────────────────────
    risk_vetoed = council.get("risk_officer_vetoed", False)
    if risk_vetoed:
        st.markdown(
            """
            <div style="
                background: linear-gradient(135deg, #3d0000 0%, #1a0000 100%);
                border: 2px solid #ff1744;
                border-radius: 10px;
                padding: 16px 20px;
                margin-bottom: 16px;
                display: flex;
                align-items: center;
                gap: 14px;
            ">
                <span style="font-size: 36px;">🛑</span>
                <div>
                    <div style="color: #ff1744; font-size: 18px; font-weight: 700; letter-spacing: 1px;">
                        RISK OFFICER VETO — SIGNAL BLOCKED
                    </div>
                    <div style="color: #ff8a80; font-size: 13px; margin-top: 4px;">
                        The Risk Officer returned NEUTRAL. This unconditional veto overrides all
                        other votes and the waterfall gate. Output is forced to NO TRADE.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Weighted vote donut chart
    buy_w   = council["buy_score"]
    sell_w  = council["sell_score"]
    hold_w  = council["total_weight"] - buy_w - sell_w

    fig_pie = go.Figure(go.Pie(
        labels=["BUY", "SELL", "HOLD"],
        values=[buy_w, sell_w, hold_w],
        hole=0.6,
        marker_colors=["#00c853", "#ff1744", "#555555"],
        textinfo="label+percent",
        textfont_size=13,
    ))
    fig_pie.add_annotation(
        text=f"<b>{signal}</b>", x=0.5, y=0.5,
        font=dict(size=22, color={"BUY": "#00e676", "SELL": "#ff5252", "NO TRADE": "#aaa"}[signal]),
        showarrow=False,
    )
    fig_pie.update_layout(
        template="plotly_dark", height=300,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        margin=dict(t=20, b=40),
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    st.divider()

    # Member cards
    for mv in council["member_votes"]:
        vote_icon   = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⚪"}[mv["vote"]]
        is_risk_officer = mv["name"] == "The Risk Officer"
        is_veto = is_risk_officer and risk_vetoed

        with st.container():
            col_left, col_right = st.columns([3, 1])
            with col_left:
                name_label = f"**{mv['name']}** · *{mv['role']}*"
                if is_veto:
                    st.markdown(
                        f"<span style='color:#ff1744;font-weight:700'>{mv['name']}</span>"
                        f" · <em>{mv['role']}</em>"
                        f" &nbsp;<span style='background:#ff1744;color:#fff;"
                        f"font-size:10px;font-weight:700;padding:2px 7px;"
                        f"border-radius:4px;letter-spacing:0.5px'>VETO</span>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(name_label)
                st.caption(mv["reason"])
            with col_right:
                veto_line = (
                    "<div style='text-align:right;color:#ff1744;font-size:11px;"
                    "font-weight:700;letter-spacing:0.5px'>🛑 SIGNAL BLOCKED</div>"
                    if is_veto else ""
                )
                st.markdown(
                    f"<div style='text-align:right;font-size:22px'>{vote_icon} {mv['vote']}</div>"
                    f"<div style='text-align:right;color:#888;font-size:12px'>weight {mv['weight']:.1f}</div>"
                    f"{veto_line}",
                    unsafe_allow_html=True,
                )
        st.divider()

    # k-NN neighbours breakdown
    st.markdown("**k-NN Neighbour Detail**")
    nbrs = knn.get("neighbours", [])
    if nbrs:
        knn_df = pd.DataFrame({
            "Neighbour": [f"#{i+1}" for i in range(len(nbrs))],
            "Label": [
                "🟢 Bullish" if n == 1 else ("🔴 Bearish" if n == -1 else "⚪ Neutral")
                for n in nbrs
            ],
        })
        st.dataframe(knn_df, use_container_width=True, hide_index=True)
    else:
        st.info("k-NN data unavailable (need more H1 bars to train on).")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — CURRENCY STRENGTH
# ═══════════════════════════════════════════════════════════════════════════
with tab_strength:
    st.markdown("### 💹 Currency Strength (normalised -100 to +100)")
    st.caption("Based on 20-bar % change across 6 GBP pairs and 6 AUD pairs on H1")

    gbp_s = strength.get("GBP", 0.0)
    aud_s = strength.get("AUD", 0.0)

    sc1, sc2 = st.columns(2)
    sc1.metric("GBP Strength", f"{gbp_s:+.1f}",
               delta="Strong" if gbp_s > 20 else ("Weak" if gbp_s < -20 else "Neutral"))
    sc2.metric("AUD Strength", f"{aud_s:+.1f}",
               delta="Strong" if aud_s > 20 else ("Weak" if aud_s < -20 else "Neutral"))

    diff = gbp_s - aud_s
    st.metric("GBP vs AUD differential", f"{diff:+.1f} pts",
              delta="Favours LONG" if diff > 12 else ("Favours SHORT" if diff < -12 else "Neutral zone"))

    st.divider()

    # Strength bar chart for all currencies
    all_currencies = {k: v for k, v in strength.items()}
    if all_currencies:
        cur_labels = list(all_currencies.keys())
        cur_values = list(all_currencies.values())
        bar_colours = [
            "#00c853" if v > 10 else ("#ff1744" if v < -10 else "#888")
            for v in cur_values
        ]

        fig_str = go.Figure(go.Bar(
            x=cur_labels, y=cur_values,
            marker_color=bar_colours,
            text=[f"{v:+.1f}" for v in cur_values],
            textposition="outside",
        ))
        fig_str.add_hline(y=0, line_color="#888", line_width=1)
        fig_str.update_layout(
            title="Currency Strength Ranking (all 8 currencies)",
            template="plotly_dark", height=350,
            yaxis=dict(range=[-110, 110], title="Strength"),
            xaxis_title="Currency",
            margin=dict(t=40, b=20),
        )
        st.plotly_chart(fig_str, use_container_width=True)

    # GBP vs AUD gauge
    st.markdown("**GBP vs AUD Head-to-Head**")
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=gbp_s - aud_s,
        title={"text": "GBP strength minus AUD strength"},
        delta={"reference": 0, "increasing": {"color": "#00c853"}, "decreasing": {"color": "#ff1744"}},
        gauge={
            "axis": {"range": [-200, 200]},
            "bar": {"color": "#ffab40"},
            "steps": [
                {"range": [-200, -12], "color": "#2b0d0d"},
                {"range": [-12, 12],   "color": "#1e1e1e"},
                {"range": [12, 200],   "color": "#0d2b12"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 2},
                "thickness": 0.75,
                "value": 0,
            },
        },
    ))
    fig_gauge.update_layout(template="plotly_dark", height=280, margin=dict(t=30, b=10))
    st.plotly_chart(fig_gauge, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 5 — TRADE JOURNAL
# ═══════════════════════════════════════════════════════════════════════════
with tab_journal:
    st.markdown("### 📒 Trade Journal")

    stats = tj.get_stats()
    j1, j2, j3, j4, j5 = st.columns(5)
    j1.metric("Total Closed", stats["total_closed"])
    j2.metric("Wins", stats["wins"])
    j3.metric("Losses", stats["losses"])
    j4.metric("Win Rate", f"{stats['win_rate']:.1f}%")
    j5.metric("Total Pips", f"{stats['total_pips']:+.1f}")

    st.divider()

    df_journal = tj.get_all_trades()

    if df_journal.empty:
        st.info("No trades in the journal yet. Save a signal from the **Signal** tab to get started.")
    else:
        # Close trade form
        open_trades = df_journal[df_journal["result"] == "OPEN"]
        if not open_trades.empty:
            st.markdown("**Close an Open Trade**")
            trade_options = {
                f"#{row['id']} | {row['signal']} @ {row['entry_price']:.5f} | {row['timestamp']}": row["id"]
                for _, row in open_trades.iterrows()
            }
            selected_label = st.selectbox("Select open trade", list(trade_options.keys()))
            selected_id = trade_options[selected_label]
            exit_col, res_col, btn_col = st.columns([2, 2, 1])
            exit_price = exit_col.number_input("Exit Price", value=0.0, format="%.5f")
            result_choice = res_col.selectbox("Result", ["WIN", "LOSS", "BREAKEVEN"])
            if btn_col.button("Close Trade", type="primary"):
                tj.close_trade(selected_id, result_choice, exit_price)
                st.success(f"Trade #{selected_id} closed as {result_choice} ✓")
                st.rerun()

            st.divider()

        # Full journal table
        display_cols = [
            "id", "timestamp", "signal", "entry_price", "stop_loss",
            "take_profit1", "take_profit2", "gbp_strength", "aud_strength",
            "waterfall", "council_buy", "council_sell", "result", "exit_price", "pips", "notes",
        ]
        cols_exist = [c for c in display_cols if c in df_journal.columns]
        st.dataframe(
            df_journal[cols_exist],
            use_container_width=True,
            hide_index=True,
        )

        # Delete trade
        st.divider()
        st.markdown("**Delete a Trade**")
        del_id = st.number_input("Trade ID to delete", min_value=1, step=1, value=1)
        if st.button("Delete Trade", type="secondary"):
            tj.delete_trade(int(del_id))
            st.warning(f"Trade #{del_id} deleted.")
            st.rerun()

"""
SQLite trade journal for recording and reviewing manual trade entries.
"""

import sqlite3
import pandas as pd
from datetime import datetime

DB_PATH = "trade_journal.db"

DDL = """
CREATE TABLE IF NOT EXISTS trades (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT    NOT NULL,
    signal      TEXT    NOT NULL,
    entry_price REAL,
    stop_loss   REAL,
    take_profit1 REAL,
    take_profit2 REAL,
    gbp_strength REAL,
    aud_strength REAL,
    waterfall   TEXT,
    council_buy  REAL,
    council_sell REAL,
    notes       TEXT,
    result      TEXT DEFAULT 'OPEN',
    exit_price  REAL,
    pips        REAL,
    closed_at   TEXT
);
"""


def init_db() -> None:
    """Ensure the trades table exists."""
    with sqlite3.connect(DB_PATH) as con:
        con.execute(DDL)
        con.commit()


def save_signal(
    signal: str,
    entry_price: float,
    stop_loss: float,
    take_profit1: float,
    take_profit2: float,
    gbp_strength: float = 0.0,
    aud_strength: float = 0.0,
    waterfall: str = "",
    council_buy: float = 0.0,
    council_sell: float = 0.0,
    notes: str = "",
) -> int:
    """Insert a new signal into the journal. Returns the new row id."""
    init_db()
    sql = """
    INSERT INTO trades
        (timestamp, signal, entry_price, stop_loss, take_profit1, take_profit2,
         gbp_strength, aud_strength, waterfall, council_buy, council_sell, notes)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    with sqlite3.connect(DB_PATH) as con:
        cur = con.execute(sql, (
            ts, signal, entry_price, stop_loss, take_profit1, take_profit2,
            gbp_strength, aud_strength, waterfall, council_buy, council_sell, notes,
        ))
        con.commit()
        return cur.lastrowid


def close_trade(trade_id: int, result: str, exit_price: float) -> None:
    """Mark a trade as WIN / LOSS / BREAKEVEN and record exit price."""
    init_db()
    with sqlite3.connect(DB_PATH) as con:
        row = con.execute(
            "SELECT signal, entry_price FROM trades WHERE id = ?", (trade_id,)
        ).fetchone()
        pips = None
        if row:
            direction = 1 if row[0] == "BUY" else -1
            pips = round((exit_price - row[1]) * direction * 10_000, 1)  # FX pips
        closed_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        con.execute(
            "UPDATE trades SET result=?, exit_price=?, pips=?, closed_at=? WHERE id=?",
            (result, exit_price, pips, closed_at, trade_id),
        )
        con.commit()


def delete_trade(trade_id: int) -> None:
    init_db()
    with sqlite3.connect(DB_PATH) as con:
        con.execute("DELETE FROM trades WHERE id = ?", (trade_id,))
        con.commit()


def get_all_trades() -> pd.DataFrame:
    """Return all trades as a DataFrame."""
    init_db()
    with sqlite3.connect(DB_PATH) as con:
        df = pd.read_sql_query(
            "SELECT * FROM trades ORDER BY id DESC", con
        )
    return df


def get_stats() -> dict:
    """Return simple P&L stats."""
    df = get_all_trades()
    closed = df[df["result"].isin(["WIN", "LOSS", "BREAKEVEN"])]
    total   = len(closed)
    wins    = len(closed[closed["result"] == "WIN"])
    losses  = len(closed[closed["result"] == "LOSS"])
    total_pips = closed["pips"].sum() if not closed.empty else 0.0
    win_rate   = (wins / total * 100) if total > 0 else 0.0
    return {
        "total_closed": total,
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 1),
        "total_pips": round(total_pips, 1),
        "open_trades": len(df[df["result"] == "OPEN"]),
    }

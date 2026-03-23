from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime
import pandas as pd


def reconstruct_positions_fifo(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    buy_queues: dict[str, deque] = defaultdict(deque)
    rows: list[dict] = []
    open_positions: list[dict] = []

    sorted_df = df.sort_values("timestamp").reset_index(drop=True)
    for _, row in sorted_df.iterrows():
        symbol = row["symbol"]
        side = row["buy_sell"]
        qty = float(row["quantity"])
        price = float(row["price"])
        ts = row["timestamp"]

        if side == "BUY":
            buy_queues[symbol].append({"remaining": qty, "price": price, "timestamp": ts})
            rows.append(
                {
                    "timestamp": ts,
                    "symbol": symbol,
                    "buy_sell": side,
                    "quantity": qty,
                    "price": price,
                    "pnl": 0.0,
                    "holding_duration": 0.0,
                    "open_position": 1,
                }
            )
            continue

        if side != "SELL":
            continue

        sell_remaining = qty
        weighted_pnl = 0.0
        weighted_holding_days = 0.0
        matched = 0.0
        while sell_remaining > 0 and buy_queues[symbol]:
            buy_lot = buy_queues[symbol][0]
            matched_qty = min(sell_remaining, buy_lot["remaining"])
            buy_lot["remaining"] -= matched_qty
            sell_remaining -= matched_qty
            matched += matched_qty
            weighted_pnl += (price - buy_lot["price"]) * matched_qty
            holding_days = (ts - buy_lot["timestamp"]).total_seconds() / 86400.0
            weighted_holding_days += holding_days * matched_qty
            if buy_lot["remaining"] <= 1e-9:
                buy_queues[symbol].popleft()
        avg_holding = weighted_holding_days / matched if matched else 0.0
        rows.append(
            {
                "timestamp": ts,
                "symbol": symbol,
                "buy_sell": side,
                "quantity": qty,
                "price": price,
                "pnl": weighted_pnl,
                "holding_duration": avg_holding,
                "open_position": 0 if sell_remaining <= 1e-9 else 1,
            }
        )

    for symbol, lots in buy_queues.items():
        for lot in lots:
            open_positions.append(
                {
                    "symbol": symbol,
                    "remaining_quantity": lot["remaining"],
                    "entry_price": lot["price"],
                    "entry_timestamp": lot["timestamp"].isoformat()
                    if isinstance(lot["timestamp"], datetime)
                    else str(lot["timestamp"]),
                }
            )
    return pd.DataFrame(rows), open_positions


def test_position_reconstruction() -> dict:
    return {"ok": True, "message": "fifo matcher wired"}

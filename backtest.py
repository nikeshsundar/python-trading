"""
=============================================================================
  BACKTESTER — Test the strategy on historical data before risking real money
=============================================================================

HOW TO USE:
    python backtest.py

What it does:
    1. Downloads historical OHLCV data from Binance
    2. Runs the exact same entry/exit logic as the live bot
    3. Reports: win rate, total return, max drawdown, Sharpe ratio

KEY METRICS TO EVALUATE:
    - Win Rate:        Aim for > 45% (with 1:2 RR, 45% WR is profitable)
    - Profit Factor:   Total gains / Total losses > 1.5 is good
    - Max Drawdown:    < 20% is acceptable for this strategy type
    - Sharpe Ratio:    > 1.0 is acceptable, > 1.5 is good
"""

import ccxt
import pandas as pd
import numpy as np
from trading_bot import BotConfig, calculate_indicators, generate_signal

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG = BotConfig()
INITIAL_BALANCE = 10_000.0
SYMBOL = "BTC/USDT"
TIMEFRAME = "4h"
LIMIT = 1000  # ~167 days of 4h candles


# ── Fetch Historical Data ─────────────────────────────────────────────────────
def fetch_history(symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    print(f"📥 Fetching {limit} {timeframe} candles for {symbol}...")
    exchange = ccxt.binance({"enableRateLimit": True})
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


# ── Backtest Engine ───────────────────────────────────────────────────────────
def run_backtest(df: pd.DataFrame) -> dict:
    """
    Walk-forward simulation: processes each candle in sequence,
    checking for entry signals and then managing the trade.
    """
    df = calculate_indicators(df, CONFIG)
    df = df.dropna().reset_index(drop=True)

    balance   = INITIAL_BALANCE
    peak      = INITIAL_BALANCE
    position  = None
    trades    = []

    print(f"\n🔁 Running backtest on {len(df)} candles...\n")

    # Start from candle 60+ (need enough history for indicators)
    for i in range(60, len(df)):
        window = df.iloc[:i+1]
        latest = df.iloc[i]
        price  = latest["close"]
        atr    = latest["atr"]

        # ── Check exit conditions if in a trade ───────────────────────────────
        if position:
            exit_reason = None

            if position["side"] == "long":
                if price <= position["stop_loss"]:
                    exit_reason = "STOP_LOSS"
                elif price >= position["take_profit"]:
                    exit_reason = "TAKE_PROFIT"
            else:  # short
                if price >= position["stop_loss"]:
                    exit_reason = "STOP_LOSS"
                elif price <= position["take_profit"]:
                    exit_reason = "TAKE_PROFIT"

            if exit_reason:
                # Calculate P&L
                if position["side"] == "long":
                    pnl = (price - position["entry"]) * position["size"]
                else:
                    pnl = (position["entry"] - price) * position["size"]

                balance += pnl
                peak = max(peak, balance)
                drawdown = (peak - balance) / peak * 100

                trade_record = {
                    "entry_time":  position["entry_time"],
                    "exit_time":   latest["timestamp"],
                    "side":        position["side"],
                    "entry_price": position["entry"],
                    "exit_price":  price,
                    "size":        position["size"],
                    "pnl":         pnl,
                    "pnl_pct":     (pnl / (position["entry"] * position["size"])) * 100,
                    "exit_reason": exit_reason,
                    "balance":     balance,
                    "drawdown_pct":drawdown
                }
                trades.append(trade_record)

                status = "✅" if pnl > 0 else "❌"
                print(f"  {status} {position['side'].upper()} | "
                      f"Entry ${position['entry']:,.0f} → Exit ${price:,.0f} | "
                      f"P&L: ${pnl:+.2f} | Balance: ${balance:,.2f}")
                position = None
                continue

        # ── Check for new entry signal ────────────────────────────────────────
        if position is None:
            signal = generate_signal(window, CONFIG)

            if signal:
                entry = price

                if signal == "long":
                    sl = entry - CONFIG.atr_stop_multiplier * atr
                    tp = entry + CONFIG.atr_tp_multiplier * atr
                else:
                    sl = entry + CONFIG.atr_stop_multiplier * atr
                    tp = entry - CONFIG.atr_tp_multiplier * atr

                # Fixed fractional sizing
                risk_amount   = balance * (CONFIG.risk_per_trade_pct / 100)
                stop_distance = abs(entry - sl)
                if stop_distance == 0:
                    continue
                size = risk_amount / stop_distance
                max_size = (balance * CONFIG.max_position_pct / 100) / entry
                size = min(size, max_size)

                position = {
                    "side":        signal,
                    "entry":       entry,
                    "stop_loss":   sl,
                    "take_profit": tp,
                    "size":        size,
                    "entry_time":  latest["timestamp"]
                }

    return {"trades": trades, "final_balance": balance}


# ── Performance Metrics ───────────────────────────────────────────────────────
def print_performance(results: dict):
    trades = results["trades"]
    final  = results["final_balance"]

    if not trades:
        print("No trades executed.")
        return

    df_t = pd.DataFrame(trades)

    wins   = df_t[df_t["pnl"] > 0]
    losses = df_t[df_t["pnl"] <= 0]

    win_rate     = len(wins) / len(df_t) * 100
    total_return = (final / INITIAL_BALANCE - 1) * 100
    gross_profit = wins["pnl"].sum() if len(wins) else 0
    gross_loss   = abs(losses["pnl"].sum()) if len(losses) else 1
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    max_drawdown  = df_t["drawdown_pct"].max()
    avg_win  = wins["pnl"].mean() if len(wins) else 0
    avg_loss = losses["pnl"].mean() if len(losses) else 0
    expectancy = (win_rate/100 * avg_win) + ((1 - win_rate/100) * avg_loss)

    # Sharpe ratio (simplified daily return estimation)
    df_t["daily_ret"] = df_t["pnl"] / INITIAL_BALANCE * 100
    sharpe = (df_t["daily_ret"].mean() / df_t["daily_ret"].std()) * np.sqrt(252) if df_t["daily_ret"].std() > 0 else 0

    print("\n" + "=" * 55)
    print("  📊  BACKTEST PERFORMANCE REPORT")
    print("=" * 55)
    print(f"  Total Trades:      {len(df_t)}")
    print(f"  Win Rate:          {win_rate:.1f}%")
    print(f"  Profit Factor:     {profit_factor:.2f}")
    print(f"  Total Return:      {total_return:.2f}%")
    print(f"  Final Balance:     ${final:,.2f}")
    print(f"  Max Drawdown:      -{max_drawdown:.2f}%")
    print(f"  Sharpe Ratio:      {sharpe:.2f}")
    print(f"  Avg Win:           ${avg_win:+.2f}")
    print(f"  Avg Loss:          ${avg_loss:+.2f}")
    print(f"  Expectancy/Trade:  ${expectancy:+.2f}")
    print("=" * 55)

    if win_rate >= 45 and profit_factor >= 1.5 and max_drawdown <= 20:
        print("  ✅ Strategy passes basic quality checks")
    else:
        print("  ⚠️  Strategy needs improvement before live trading")
    print()

    # ── Trade log ─────────────────────────────────────────────────────────────
    print("  Last 10 trades:")
    print(df_t[["entry_time", "side", "pnl", "exit_reason", "balance"]].tail(10).to_string(index=False))


if __name__ == "__main__":
    df = fetch_history(SYMBOL, TIMEFRAME, LIMIT)
    results = run_backtest(df)
    print_performance(results)

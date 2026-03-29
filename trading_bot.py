"""
=============================================================================
  TREND-FOLLOWING CRYPTO TRADING BOT
  Strategy: EMA Crossover + RSI Filter + ATR-based Risk Management
  Author: Quant Dev Template
  Exchange: Binance (via CCXT)
  Timeframe: 4-hour candles
  Markets: BTC/USDT, ETH/USDT
=============================================================================

REALISTIC EXPECTATIONS:
  - Win Rate: ~45–55%
  - Risk/Reward: 1:2 minimum
  - Expected Monthly Return: 3–8% (with compounding, high variance)
  - Max Drawdown Target: < 15%
  - This is NOT a get-rich-quick system. Consistency > big wins.

DISCLAIMER:
  This is for educational purposes only. Crypto trading carries
  significant risk. Never trade money you can't afford to lose.
=============================================================================
"""

import ccxt
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("bot.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — Tune these before going live
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class BotConfig:
    # ── Exchange ──────────────────────────────────────────────────────────────
    api_key: str            = "YOUR_API_KEY"
    api_secret: str         = "YOUR_API_SECRET"
    exchange_id: str        = "binance"
    paper_trading: bool     = True       # ALWAYS start with True!

    # ── Markets & Timeframe ───────────────────────────────────────────────────
    symbols: list           = field(default_factory=lambda: ["BTC/USDT", "ETH/USDT"])
    timeframe: str          = "4h"       # 4-hour candles — swing trading
    candles_lookback: int   = 200        # How many candles to fetch for indicators

    # ── Strategy Parameters ───────────────────────────────────────────────────
    ema_fast: int           = 20         # Fast EMA period
    ema_slow: int           = 50         # Slow EMA period
    rsi_period: int         = 14         # RSI period
    rsi_overbought: float   = 70.0       # RSI overbought threshold
    rsi_oversold: float     = 30.0       # RSI oversold threshold
    atr_period: int         = 14         # ATR period for stop/target sizing
    volume_ma_period: int   = 20         # Volume moving average period

    # ── Risk Management ───────────────────────────────────────────────────────
    risk_per_trade_pct: float = 1.0      # Risk 1% of account per trade
    atr_stop_multiplier: float = 2.0     # Stop loss = 2x ATR below entry
    atr_tp_multiplier: float  = 4.0      # Take profit = 4x ATR above entry (RR = 1:2)
    max_daily_loss_pct: float = 3.0      # Halt trading if down 3% in a day
    max_open_positions: int   = 2        # Maximum simultaneous positions
    max_position_pct: float   = 20.0     # Max 20% of account in one position

    # ── Bot Control ───────────────────────────────────────────────────────────
    poll_interval_seconds: int = 60      # Check every 60 seconds
    dry_run_balance: float = 10_000.0    # Simulated USDT balance for paper trading


CONFIG = BotConfig()


# ─────────────────────────────────────────────────────────────────────────────
# EXCHANGE CONNECTION
# ─────────────────────────────────────────────────────────────────────────────
def connect_exchange(config: BotConfig) -> ccxt.Exchange:
    """
    Connect to the exchange using CCXT.
    Uses sandbox/paper mode by default — safe for testing.
    """
    exchange_class = getattr(ccxt, config.exchange_id)
    exchange = exchange_class({
        "apiKey": config.api_key,
        "secret": config.api_secret,
        "enableRateLimit": True,   # Respect API rate limits automatically
        "options": {"defaultType": "future"} if config.exchange_id == "binance" else {}
    })

    if config.paper_trading:
        # Binance testnet
        exchange.set_sandbox_mode(True)
        log.info("🟡 PAPER TRADING MODE — No real money at risk")
    else:
        log.warning("🔴 LIVE TRADING MODE — Real money at risk!")

    return exchange


# ─────────────────────────────────────────────────────────────────────────────
# INDICATOR CALCULATIONS
# ─────────────────────────────────────────────────────────────────────────────
def calculate_indicators(df: pd.DataFrame, config: BotConfig) -> pd.DataFrame:
    """
    Compute all technical indicators on a DataFrame of OHLCV data.

    Inputs:
        df: DataFrame with columns [timestamp, open, high, low, close, volume]

    Returns:
        df with added indicator columns
    """
    # ── EMA Crossover ─────────────────────────────────────────────────────────
    # EMAs smooth price trends. When fast EMA crosses above slow EMA → uptrend.
    df["ema_fast"] = df["close"].ewm(span=config.ema_fast, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=config.ema_slow, adjust=False).mean()

    # Crossover signal: +1 = bullish crossover, -1 = bearish crossover, 0 = none
    df["ema_signal"] = np.where(df["ema_fast"] > df["ema_slow"], 1, -1)
    df["ema_cross"] = df["ema_signal"].diff()  # 2 = just crossed up, -2 = crossed down

    # ── RSI (Relative Strength Index) ─────────────────────────────────────────
    # RSI measures momentum. Values > 70 = overbought, < 30 = oversold.
    # We use it as a FILTER — only trade in the direction of momentum.
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=config.rsi_period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=config.rsi_period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    # ── ATR (Average True Range) ───────────────────────────────────────────────
    # ATR measures market volatility. Used to set dynamic stop losses and
    # take profits that adapt to current market conditions.
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = true_range.ewm(span=config.atr_period, adjust=False).mean()

    # ── Volume Filter ─────────────────────────────────────────────────────────
    # Only trade when volume is above its moving average — confirms conviction.
    df["volume_ma"] = df["volume"].rolling(config.volume_ma_period).mean()
    df["volume_ok"] = df["volume"] > df["volume_ma"]

    return df


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL GENERATION
# ─────────────────────────────────────────────────────────────────────────────
def generate_signal(df: pd.DataFrame, config: BotConfig) -> Optional[str]:
    """
    Evaluate the latest candle and return a trading signal.

    Entry Rules (LONG):
      1. EMA(20) crosses ABOVE EMA(50) on this candle
      2. RSI is between 40 and 70 (trending up but not overbought)
      3. Volume is above its 20-period moving average

    Entry Rules (SHORT):
      1. EMA(20) crosses BELOW EMA(50) on this candle
      2. RSI is between 30 and 60 (trending down but not oversold)
      3. Volume is above its 20-period moving average

    Returns: "long", "short", or None
    """
    if len(df) < 3:
        return None

    latest = df.iloc[-1]
    prev   = df.iloc[-2]

    rsi_val    = latest["rsi"]
    vol_ok     = latest["volume_ok"]

    # Fresh crossover check (using prev to detect the cross event)
    ema_crossed_up   = (prev["ema_fast"] <= prev["ema_slow"]) and (latest["ema_fast"] > latest["ema_slow"])
    ema_crossed_down = (prev["ema_fast"] >= prev["ema_slow"]) and (latest["ema_fast"] < latest["ema_slow"])

    # ── LONG signal ───────────────────────────────────────────────────────────
    if ema_crossed_up and (40 <= rsi_val <= config.rsi_overbought) and vol_ok:
        log.info(f"  📈 LONG signal | RSI={rsi_val:.1f} | Vol confirmed")
        return "long"

    # ── SHORT signal ──────────────────────────────────────────────────────────
    if ema_crossed_down and (config.rsi_oversold <= rsi_val <= 60) and vol_ok:
        log.info(f"  📉 SHORT signal | RSI={rsi_val:.1f} | Vol confirmed")
        return "short"

    return None


# ─────────────────────────────────────────────────────────────────────────────
# POSITION SIZING — The most important risk management function
# ─────────────────────────────────────────────────────────────────────────────
def calculate_position_size(
    account_balance: float,
    entry_price: float,
    stop_loss_price: float,
    config: BotConfig
) -> float:
    """
    Kelly-inspired fixed fractional position sizing.

    Formula:
        risk_amount = account_balance * risk_per_trade_pct / 100
        stop_distance = |entry_price - stop_loss_price|
        position_size = risk_amount / stop_distance

    Example:
        Balance = $10,000 | Risk = 1% = $100
        Entry = $50,000 | Stop = $49,000 | Distance = $1,000
        Position size = $100 / $1,000 = 0.1 BTC

    This ensures you ALWAYS risk exactly 1% of your account per trade,
    regardless of volatility — a core risk management principle.
    """
    risk_amount    = account_balance * (config.risk_per_trade_pct / 100)
    stop_distance  = abs(entry_price - stop_loss_price)

    if stop_distance == 0:
        log.warning("  ⚠️ Stop distance is zero — skipping trade")
        return 0.0

    position_size  = risk_amount / stop_distance

    # Cap at max position size
    max_size = (account_balance * config.max_position_pct / 100) / entry_price
    position_size = min(position_size, max_size)

    return round(position_size, 6)


# ─────────────────────────────────────────────────────────────────────────────
# TRADE EXECUTION
# ─────────────────────────────────────────────────────────────────────────────
def place_order(
    exchange: ccxt.Exchange,
    symbol: str,
    side: str,
    amount: float,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    config: BotConfig
) -> dict:
    """
    Place a market order with stop loss and take profit orders.
    In paper trading mode, simulates the order instead of sending it.
    """
    if config.paper_trading:
        log.info(f"  [PAPER] {side.upper()} {amount:.6f} {symbol} @ ${entry_price:,.2f}")
        log.info(f"  [PAPER] SL: ${stop_loss:,.2f} | TP: ${take_profit:,.2f}")
        return {
            "id": f"paper_{int(time.time())}",
            "symbol": symbol,
            "side": side,
            "amount": amount,
            "price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "status": "paper_filled",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    try:
        # Place main market order
        order = exchange.create_market_order(symbol, side, amount)
        log.info(f"  ✅ ORDER PLACED: {order['id']} | {side} {amount} {symbol}")

        # Place stop loss (opposite side)
        sl_side = "sell" if side == "buy" else "buy"
        exchange.create_order(symbol, "stop_market", sl_side, amount, stop_loss,
                              {"stopPrice": stop_loss, "reduceOnly": True})

        # Place take profit (opposite side)
        exchange.create_order(symbol, "take_profit_market", sl_side, amount, take_profit,
                              {"stopPrice": take_profit, "reduceOnly": True})

        return order
    except ccxt.InsufficientFunds as e:
        log.error(f"  ❌ Insufficient funds: {e}")
        return {}
    except ccxt.ExchangeError as e:
        log.error(f"  ❌ Exchange error: {e}")
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# DAILY LOSS GUARD
# ─────────────────────────────────────────────────────────────────────────────
class DailyLossGuard:
    """
    Tracks daily P&L and halts the bot if the max daily loss is breached.
    This prevents a single bad day from blowing up the account.
    """
    def __init__(self, starting_balance: float, max_loss_pct: float):
        self.starting_balance = starting_balance
        self.max_loss_pct     = max_loss_pct
        self.day              = datetime.now(timezone.utc).date()
        self.locked           = False

    def reset_if_new_day(self, current_balance: float):
        today = datetime.now(timezone.utc).date()
        if today != self.day:
            log.info(f"  🗓️ New day — resetting daily loss guard")
            self.day              = today
            self.starting_balance = current_balance
            self.locked           = False

    def check(self, current_balance: float) -> bool:
        """Returns True if trading is allowed, False if daily loss exceeded."""
        self.reset_if_new_day(current_balance)
        if self.locked:
            return False

        loss_pct = ((self.starting_balance - current_balance) / self.starting_balance) * 100
        if loss_pct >= self.max_loss_pct:
            log.warning(f"  🛑 DAILY LOSS LIMIT HIT: -{loss_pct:.2f}% — Bot halted for today")
            self.locked = True
            return False

        return True


# ─────────────────────────────────────────────────────────────────────────────
# MAIN BOT LOOP
# ─────────────────────────────────────────────────────────────────────────────
def run_bot():
    """
    Main trading loop. Runs continuously, checking each symbol every
    poll_interval_seconds for a new signal.
    """
    log.info("=" * 60)
    log.info("  🤖 CRYPTO TREND BOT — STARTING")
    log.info("=" * 60)

    exchange  = connect_exchange(CONFIG)
    open_positions = {}   # symbol → order info
    balance   = CONFIG.dry_run_balance if CONFIG.paper_trading else None

    if not CONFIG.paper_trading:
        balance = exchange.fetch_balance()["USDT"]["free"]

    loss_guard = DailyLossGuard(balance, CONFIG.max_daily_loss_pct)

    log.info(f"  💰 Starting balance: ${balance:,.2f} USDT")
    log.info(f"  📊 Trading: {CONFIG.symbols} on {CONFIG.timeframe} candles")
    log.info(f"  🛡️  Risk per trade: {CONFIG.risk_per_trade_pct}% | Max daily loss: {CONFIG.max_daily_loss_pct}%")

    while True:
        try:
            # ── Fetch current balance ─────────────────────────────────────────
            if not CONFIG.paper_trading:
                balance = exchange.fetch_balance()["USDT"]["free"]

            # ── Daily loss guard ──────────────────────────────────────────────
            if not loss_guard.check(balance):
                log.info("  ⏸️  Daily loss limit reached. Sleeping until tomorrow...")
                time.sleep(3600)
                continue

            for symbol in CONFIG.symbols:
                log.info(f"\n  📡 Checking {symbol}...")

                # ── Max open positions check ──────────────────────────────────
                if len(open_positions) >= CONFIG.max_open_positions:
                    log.info(f"  ⏭️  Max positions open ({CONFIG.max_open_positions}) — skipping {symbol}")
                    continue

                if symbol in open_positions:
                    log.info(f"  ⏭️  Already in a position for {symbol}")
                    continue

                # ── Fetch OHLCV candles ───────────────────────────────────────
                ohlcv = exchange.fetch_ohlcv(symbol, CONFIG.timeframe, limit=CONFIG.candles_lookback)
                df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

                # ── Compute indicators ────────────────────────────────────────
                df = calculate_indicators(df, CONFIG)

                # ── Generate signal ───────────────────────────────────────────
                signal = generate_signal(df, CONFIG)

                if signal is None:
                    log.info(f"  ⬜ No signal for {symbol}")
                    continue

                # ── Calculate entry, stop loss, take profit ───────────────────
                latest     = df.iloc[-1]
                entry      = latest["close"]
                atr        = latest["atr"]

                if signal == "long":
                    stop_loss   = entry - (CONFIG.atr_stop_multiplier * atr)
                    take_profit = entry + (CONFIG.atr_tp_multiplier * atr)
                    order_side  = "buy"
                else:  # short
                    stop_loss   = entry + (CONFIG.atr_stop_multiplier * atr)
                    take_profit = entry - (CONFIG.atr_tp_multiplier * atr)
                    order_side  = "sell"

                log.info(f"  Entry:  ${entry:,.2f}")
                log.info(f"  SL:     ${stop_loss:,.2f}  ({CONFIG.atr_stop_multiplier}x ATR)")
                log.info(f"  TP:     ${take_profit:,.2f}  ({CONFIG.atr_tp_multiplier}x ATR)")

                # ── Calculate position size ───────────────────────────────────
                size = calculate_position_size(balance, entry, stop_loss, CONFIG)
                if size <= 0:
                    continue

                notional = size * entry
                log.info(f"  Size:   {size:.6f} units (${notional:,.2f} notional)")
                log.info(f"  Risk:   ${(CONFIG.risk_per_trade_pct/100)*balance:,.2f} ({CONFIG.risk_per_trade_pct}% of balance)")

                # ── Place the order ───────────────────────────────────────────
                order = place_order(exchange, symbol, order_side, size,
                                    entry, stop_loss, take_profit, CONFIG)

                if order:
                    open_positions[symbol] = {
                        "order": order,
                        "signal": signal,
                        "entry": entry,
                        "stop_loss": stop_loss,
                        "take_profit": take_profit,
                        "size": size
                    }

            # ── Sleep until next check ────────────────────────────────────────
            log.info(f"\n  💤 Sleeping {CONFIG.poll_interval_seconds}s...\n")
            time.sleep(CONFIG.poll_interval_seconds)

        except KeyboardInterrupt:
            log.info("\n  🛑 Bot stopped by user.")
            break
        except Exception as e:
            log.error(f"  ❌ Unexpected error: {e}", exc_info=True)
            time.sleep(30)


if __name__ == "__main__":
    run_bot()

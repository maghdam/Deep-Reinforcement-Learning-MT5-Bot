#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Live trading runner (MT5) with verbose logs (ASCII-safe) + market-hours check.
"""
import os, sys, time, signal, argparse, logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import pandas as pd

try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv(*args, **kwargs):
        return False

try:
    import MetaTrader5 as mt5
except Exception:
    mt5 = None

from stable_baselines3 import PPO
import features
from adapters import broker

STOP = False

def handle_exit(signum, frame):
    global STOP
    STOP = True
    logging.getLogger("live").info("Received signal %s, shutting down...", signum)

for sig in (signal.SIGINT, signal.SIGTERM):
    try:
        signal.signal(sig, handle_exit)
    except Exception:
        pass

def setup_logging(log_path: Optional[Path] = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("live")
    logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); logger.addHandler(sh)
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(log_path, maxBytes=2000000, backupCount=3, encoding="utf-8")
        fh.setFormatter(fmt); logger.addHandler(fh)
    return logger

def minutes_per_bar(tf: str) -> int:
    tf = str(tf).upper()
    if tf.startswith("M"): return int(tf[1:])
    if tf.startswith("H"): return int(tf[1:]) * 60
    if tf in ("D1", "1D"): return 1440
    return 15

def seconds_to_next_bar(tf: str) -> float:
    m = minutes_per_bar(tf)
    now = datetime.now(timezone.utc)
    base = now.replace(second=0, microsecond=0)
    mins = now.minute
    next_min = ((mins // m) + 1) * m
    next_bar = base + timedelta(minutes=(next_min - mins))
    delta = (next_bar - now).total_seconds()
    if delta < 5: delta += m * 60
    return float(delta)

def scalar_action(action) -> int:
    return int(np.asarray(action).reshape(-1)[0])

def action_to_signals(a: int) -> Dict[str, bool]:
    return {"buy": a == 2, "sell": a == 0, "hold": a == 1}

def print_header(logger: logging.Logger, symbol: str, a: int, sig: Dict[str, bool]) -> None:
    logger.info("Model action: %s", a)
    logger.info("-" * 66)
    logger.info("Date: %s, SYMBOL: %s, BUY SIGNAL: %s, SELL SIGNAL: %s, HOLD SIGNAL: %s",
        datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"), symbol, sig["buy"], sig["sell"], sig["hold"])

def positions_snapshot(logger: logging.Logger, symbol: str) -> None:
    try:
        pos = broker.current_positions(symbol)
        if pos is None or len(pos) == 0:
            logger.info("No open positions.")
            return
        n_long = int((pos["type"] == 0).sum())
        n_short = int((pos["type"] == 1).sum())
        vol_total = float(pos["volume"].sum())
        flt_pnl = float(pos["profit"].sum())
        logger.info("Positions snapshot | longs=%d shorts=%d total_vol=%.3f floatingPnL=%.2f",
                    n_long, n_short, vol_total, flt_pnl)
    except Exception as e:
        logger.warning("Snapshot failed: %s", e)

def has_same_side_position(symbol: str, want_buy: bool, want_sell: bool) -> bool:
    try:
        pos_now = broker.current_positions(symbol)
        if pos_now is None or len(pos_now) == 0: return False
        if want_buy and (pos_now["type"] == 0).any(): return True
        if want_sell and (pos_now["type"] == 1).any(): return True
    except Exception:
        return False
    return False

def close_opposite_positions_if_any(logger: logging.Logger, symbol: str, sig: Dict[str, bool], dry_run: bool) -> bool:
    try:
        pos = broker.current_positions(symbol)
    except Exception as e:
        logger.warning("Could not fetch positions: %s", e)
        return False
    if pos is None or len(pos) == 0:
        return False
    if sig["buy"]:
        to_close = pos[pos["type"] == 1]
        side_text = "sell"
    elif sig["sell"]:
        to_close = pos[pos["type"] == 0]
        side_text = "buy"
    else:
        return False
    if len(to_close) == 0:
        return False
    if dry_run:
        tickets = ", ".join(map(lambda x: str(int(x)), to_close["ticket"]))
        logger.info("DRY_RUN=True -> Would close %d %s position(s): %s", len(to_close), side_text, tickets)
        return False
    logger.info("Existing %s positions found. Attempting to close...", side_text)
    any_closed = False
    for _, row in to_close.iterrows():
        ticket = int(row["ticket"])
        res = broker.close_position(ticket)
        if res.get("ok"):
            logger.info("Successfully closed position %s for %s", ticket, symbol); any_closed = True
        else:
            logger.warning("Failed to close position %s: %s", ticket, res)
    return any_closed

def place_signal_order(logger: logging.Logger, symbol: str, sig: Dict[str, bool], vol: float, dry_run: bool, comment: str):
    if sig["hold"]:
        logger.info("Hold signal detected. No actions taken."); return None
    side = "buy" if sig["buy"] else "sell"
    if has_same_side_position(symbol, sig["buy"], sig["sell"]):
        logger.info("Same-side position already exists. Skipping new %s order.", side)
        return None
    if dry_run:
        logger.info("DRY_RUN=True -> Skipping order. Would place %s %.3f on %s", side.upper(), vol, symbol)
        return {"ok": True, "dry_run": True, "side": side, "volume": vol}
    logger.info("%s positions closed (if any). Placing new %s order.", "Buy" if side=="buy" else "Sell", side)
    res = broker.place_market_order(symbol, side=side, volume=vol, comment=comment)
    detail = res.get("result", {}) if isinstance(res, dict) else {}
    price = detail.get("price") if isinstance(detail, dict) else None
    order = detail.get("order") if isinstance(detail, dict) else None
    deal  = detail.get("deal")  if isinstance(detail, dict) else None
    logger.info("Order result: ok=%s, retcode=%s, order=%s, deal=%s, price=%s",
                res.get("ok"), res.get("retcode"), order, deal, price)
    return res

def print_bar_context(logger: logging.Logger, last_row: pd.Series):
    fields = []
    if "close" in last_row: fields.append("close=%.5f" % float(last_row["close"]))
    if "ma_fast" in last_row: fields.append("ma_fast=%.5f" % float(last_row["ma_fast"]))
    if "ma_slow" in last_row: fields.append("ma_slow=%.5f" % float(last_row["ma_slow"]))
    if "rsi" in last_row: fields.append("rsi=%.1f" % float(last_row["rsi"]))
    if fields:
        logger.info("Context: " + " | ".join(fields))

def is_market_open(symbol: str, tf: str, logger: logging.Logger, allow_weekend: bool) -> bool:
    try:
        now = datetime.now(timezone.utc)
        dow = now.weekday()
        if not allow_weekend and dow in (5, 6):
            logger.info("Market likely closed (weekend).")
            return False
        if mt5 is None:
            return True
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logger.info("No tick available for %s.", symbol)
            return False
        tick_ts = datetime.fromtimestamp(tick.time, tz=timezone.utc)
        age_sec = (now - tick_ts).total_seconds()
        bar_min = minutes_per_bar(tf)
        thresh = max(bar_min * 120, 1800)
        if age_sec > thresh:
            logger.info("Market likely closed (last tick %.1f min ago).", age_sec / 60.0)
            return False
        return True
    except Exception as e:
        logger.warning("Market check failed: %s", e)
        return True

def main():
    load_dotenv()
    p = argparse.ArgumentParser(description="Live DRL trading bot (MT5)")
    p.add_argument("--symbol", default=os.getenv("TRAINING_SYMBOL","EURUSD"))
    p.add_argument("--timeframe", default=os.getenv("TIMEFRAME","M15"))
    p.add_argument("--model", default=None)
    p.add_argument("--features", default="notebooks/models/selected_features.json")
    p.add_argument("--volume", type=float, default=float(os.getenv("VOLUME","0.01")))
    p.add_argument("--live", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--order-comment", default=os.getenv("ORDER_COMMENT","DRL-Live"))
    p.add_argument("--log-file", default="logs/live_bot.log")
    p.add_argument("--skip-market-check", action="store_true", help="Disable market closed guard")
    args = p.parse_args()

    symbol = args.symbol; tf = args.timeframe
    dry_run = not args.live or args.dry_run
    vol = max(args.volume, 0.0); comment = args.order_comment
    allow_weekend = os.getenv("WEEKEND_TRADING","false").lower() in ("1","true","yes")

    logger = setup_logging(Path(args.log_file))
    logger.info("Connecting to MT5...")
    ok = broker.open_session(
        login=int(os.getenv("MT5_LOGIN","0")) or None,
        password=os.getenv("MT5_PASSWORD"),
        server=os.getenv("MT5_SERVER"),
        path=os.getenv("MT5_PATH"),
    )
    logger.info("MT5 connected: %s", ok)

    model_path = args.model or f"notebooks/models/ppo_{symbol}_{tf}.zip"
    mp = Path(model_path)
    if not mp.exists():
        alt = Path("notebooks") / "notebooks/models" / Path(f"ppo_{symbol}_{tf}.zip")
        if alt.exists():
            mp = alt
    if not mp.exists():
        logger.error("Model not found: %s", model_path)
        logger.error("Also checked notebooks/models/ppo_%s_%s.zip", symbol, tf)
        sys.exit(2)

    try:
        import json
        with open(args.features, "r", encoding="utf-8") as f:
            feature_cols = json.load(f)
    except Exception as e:
        logger.error("Failed to load features from %s: %s", args.features, e); sys.exit(2)

    logger.info("Loading model: %s", mp)
    model = PPO.load(mp.as_posix())

    logger.info("Starting live loop | symbol=%s timeframe=%s dry_run=%s volume=%.3f", symbol, tf, dry_run, vol)
    try:
        while not STOP:
            try:
                open_ok = True if args.skip_market_check else is_market_open(symbol, tf, logger, allow_weekend)
                if not open_ok:
                    secs = min(900, seconds_to_next_bar(tf))
                    logger.info("Sleeping %.0f seconds until next check (market closed).", secs)
                    time.sleep(secs)
                    continue

                df = broker.fetch_last_n(symbol, tf, n=500)
                if df is None or len(df) < 50:
                    logger.warning("Insufficient bars fetched; sleeping 5s"); time.sleep(5); continue

                df_feat = features.add_indicators(df.copy())
                last_row = df_feat.iloc[-1]
                print_bar_context(logger, last_row)

                obs = last_row[list(feature_cols)].astype(np.float32).values
                if np.isnan(obs).any() or np.isinf(obs).any():
                    logger.warning("Obs has NaN/Inf; skipping this tick"); time.sleep(5); continue

                action, _ = model.predict(obs, deterministic=True)
                a = scalar_action(action); sig = action_to_signals(a)
                print_header(logger, symbol, a, sig)

                positions_snapshot(logger, symbol)
                _ = close_opposite_positions_if_any(logger, symbol, sig, dry_run)

                if sig["hold"]:
                    logger.info("Appropriate position already exists or HOLD signal. No new order.")
                else:
                    _ = place_signal_order(logger, symbol, sig, vol, dry_run, comment)

                secs = seconds_to_next_bar(tf)
                logger.info("Waiting for new signals...")
                logger.info("Calculated sleep time: %.3f seconds", secs)
                time.sleep(secs)

            except Exception as e:
                logger.exception("Loop error: %s", e); time.sleep(5)
    finally:
        try: broker.close_session()
        except Exception: pass
        logger.info("Shutdown complete.")

if __name__ == "__main__":
    main()

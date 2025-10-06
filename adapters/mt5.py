"""
Minimal MetaTrader 5 adapter for this repo.
Keep credentials in environment variables or .env (never hardcode).
"""
import os
from typing import Optional, Dict, Any
import pandas as pd

try:
    import MetaTrader5 as mt5
except Exception as e:
    raise RuntimeError("MetaTrader5 package is required. pip install MetaTrader5") from e

_TF_MAP = {
    "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1, "W1": mt5.TIMEFRAME_W1, "MN1": mt5.TIMEFRAME_MN1,
}

def timeframe_to_mt5(tf: str):
    return _TF_MAP.get(str(tf).upper(), mt5.TIMEFRAME_M15)

def _ensure_symbol(symbol: str) -> bool:
    info = mt5.symbol_info(symbol)
    if info is None:
        return False
    if not info.visible:
        return mt5.symbol_select(symbol, True)
    return True

def open_session(login: Optional[int]=None, password: Optional[str]=None, server: Optional[str]=None, path: Optional[str]=None) -> bool:
    ok = mt5.initialize(path) if path else mt5.initialize()
    if not ok:
        print("MT5 initialize failed:", mt5.last_error())
        return False
    if login and password and server:
        if not mt5.login(login=login, password=password, server=server):
            print("MT5 login failed:", mt5.last_error())
            return False
    return True

def close_session() -> None:
    mt5.shutdown()

def fetch_ohlc(symbol: str, timeframe: str, start: str, end: str) -> pd.DataFrame:
    if not _ensure_symbol(symbol):
        raise ValueError(f"Symbol not available: {symbol}")
    tf = timeframe_to_mt5(timeframe)
    s = pd.to_datetime(start, utc=True)
    e = pd.to_datetime(end, utc=True)
    rates = mt5.copy_rates_range(symbol, tf, s.to_pydatetime(), e.to_pydatetime())
    if rates is None:
        err = mt5.last_error()
        raise RuntimeError(f"copy_rates_range failed: {err}")
    df = pd.DataFrame(rates)
    if df.empty:
        return df
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.set_index("time")
    if "tick_volume" in df.columns and "volume" not in df.columns:
        df["volume"] = df["tick_volume"]
    return df[["open","high","low","close","volume"]].sort_index()

def fetch_last_n(symbol: str, timeframe: str, n: int=500) -> pd.DataFrame:
    if not _ensure_symbol(symbol):
        raise ValueError(f"Symbol not available: {symbol}")
    tf = timeframe_to_mt5(timeframe)
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, n)
    if rates is None:
        err = mt5.last_error()
        raise RuntimeError(f"copy_rates_from_pos failed: {err}")
    df = pd.DataFrame(rates)
    if df.empty:
        return df
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.set_index("time")
    if "tick_volume" in df.columns and "volume" not in df.columns:
        df["volume"] = df["tick_volume"]
    return df[["open","high","low","close","volume"]].sort_index()

def place_market_order(symbol: str, side: str, volume: float, sl: Optional[float]=None, tp: Optional[float]=None, comment: str="", deviation: int=20, magic: Optional[int]=None) -> Dict[str, Any]:
    if not _ensure_symbol(symbol):
        return {"ok": False, "msg": f"Symbol not available: {symbol}"}
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return {"ok": False, "msg": "No tick data"}
    order_type = mt5.ORDER_TYPE_BUY if side.lower()=="buy" else mt5.ORDER_TYPE_SELL
    price = tick.ask if order_type==mt5.ORDER_TYPE_BUY else tick.bid
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(volume),
        "type": order_type,
        "price": float(price),
        "sl": float(sl) if sl else 0.0,
        "tp": float(tp) if tp else 0.0,
        "deviation": int(deviation),
        "magic": int(magic) if magic is not None else int(os.getenv("MAGIC_NUMBER","234002")),
        "comment": comment[:31],
        "type_filling": mt5.ORDER_FILLING_IOC,
        "type_time": mt5.ORDER_TIME_GTC,
    }
    result = mt5.order_send(request)
    if result is None:
        return {"ok": False, "msg": "order_send returned None", "error": mt5.last_error()}
    return {"ok": result.retcode==mt5.TRADE_RETCODE_DONE, "retcode": result.retcode, "result": result._asdict()}

def close_position(position_id: int) -> Dict[str, Any]:
    pos_list = mt5.positions_get(ticket=position_id)
    if pos_list is None or len(pos_list)==0:
        return {"ok": False, "msg": f"Position not found: {position_id}"}
    pos = pos_list[0]
    symbol = pos.symbol
    volume = pos.volume
    side = "buy" if pos.type==mt5.POSITION_TYPE_BUY else "sell"

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return {"ok": False, "msg": "No tick data"}
    order_type = mt5.ORDER_TYPE_SELL if side=="buy" else mt5.ORDER_TYPE_BUY
    price = tick.bid if order_type==mt5.ORDER_TYPE_SELL else tick.ask

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(volume),
        "type": order_type,
        "position": position_id,
        "price": float(price),
        "deviation": int(os.getenv("DEVIATION","20")),
        "magic": int(os.getenv("MAGIC_NUMBER","234002")),
        "comment": "close_position",
        "type_filling": mt5.ORDER_FILLING_IOC,
        "type_time": mt5.ORDER_TIME_GTC,
    }
    result = mt5.order_send(request)
    if result is None:
        return {"ok": False, "msg": "order_send returned None", "error": mt5.last_error()}
    return {"ok": result.retcode==mt5.TRADE_RETCODE_DONE, "retcode": result.retcode, "result": result._asdict()}

def current_positions(symbol: str|None=None) -> pd.DataFrame:
    positions = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
    if positions is None:
        err = mt5.last_error()
        raise RuntimeError(f"positions_get failed: {err}")
    if len(positions)==0:
        return pd.DataFrame(columns=["ticket","symbol","type","volume","price_open","sl","tp","profit"])
    df = pd.DataFrame([p._asdict() for p in positions])
    df = df.rename(columns={
        "ticket":"ticket","symbol":"symbol","type":"type","volume":"volume",
        "price_open":"price_open","sl":"sl","tp":"tp","profit":"profit"
    })
    return df[["ticket","symbol","type","volume","price_open","sl","tp","profit"]]
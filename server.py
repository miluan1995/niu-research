#!/usr/bin/env python3
"""
A股投研 Agent 后端 — 真实数据 + 回测引擎 + API 服务器。
数据源: AkShare (免费, 无需 API key)
策略: ETF 动量轮动 (Top4 / 20日动量 / MA20过滤 / 风控状态机)
"""
from __future__ import annotations

import json
import math
import datetime as dt
import http.server
import socketserver
import threading
import time
import os
import sys
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import akshare as ak
import pandas as pd
import numpy as np

# ─── 配置 ──────────────────────────────────────────────
ETF_POOL = {
    "510300": "沪深300ETF",
    "510500": "中证500ETF",
    "510050": "上证50ETF",
    "159915": "创业板ETF",
    "512100": "中证1000ETF",
    "512480": "半导体ETF",
    "512760": "半导体50ETF",
    "515790": "光伏ETF",
    "515030": "新能源车ETF",
    "512690": "白酒ETF",
    "512010": "医药ETF",
    "512800": "银行ETF",
    "512070": "非银ETF",
    "159766": "旅游ETF",
    "515170": "食品饮料ETF",
    "512660": "军工ETF",
    "515880": "通信ETF",
    "512670": "国防军工ETF",
    "562500": "机器人ETF",
}
HS300_CODE = "sh000300"
MA_PERIOD = 20
MOMENTUM_PERIOD = 20
REBALANCE_DAYS = 10
TOP_N = 4
STOP_LOSS_PCT = -0.12
COOLDOWN_DAYS = 10
COMMISSION_RATE = 0.0003
SLIPPAGE_RATE = 0.001
CACHE_DIR = Path(__file__).parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

# ─── 数据缓存 ──────────────────────────────────────────
_cache: dict[str, pd.DataFrame] = {}
_cache_ts: dict[str, float] = {}
CACHE_TTL = 3600  # 1 hour


def _cache_key(name: str) -> str:
    return name


def get_etf_hist(symbol: str) -> pd.DataFrame:
    """获取ETF前复权日线数据"""
    now = time.time()
    key = f"etf_{symbol}"
    if key in _cache and now - _cache_ts.get(key, 0) < CACHE_TTL:
        return _cache[key]

    # 检查磁盘缓存
    cache_file = CACHE_DIR / f"{key}.parquet"
    if cache_file.exists() and now - cache_file.stat().st_mtime < CACHE_TTL:
        df = pd.read_parquet(cache_file)
        _cache[key] = df
        _cache_ts[key] = now
        return df

    try:
        df = ak.fund_etf_hist_em(symbol=symbol, period="daily", adjust="qfq")
        df.columns = ["date", "open", "close", "high", "low", "volume", "amount", "amplitude", "pct_change", "change", "turnover"]
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        df["close"] = df["close"].astype(float)
        df["open"] = df["open"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df.to_parquet(cache_file, index=False)
        _cache[key] = df
        _cache_ts[key] = now
        return df
    except Exception as e:
        print(f"[WARN] Failed to fetch ETF {symbol}: {e}", file=sys.stderr)
        return pd.DataFrame()


def get_hs300_hist() -> pd.DataFrame:
    """获取沪深300指数日线"""
    now = time.time()
    key = "hs300"
    if key in _cache and now - _cache_ts.get(key, 0) < CACHE_TTL:
        return _cache[key]

    cache_file = CACHE_DIR / f"{key}.parquet"
    if cache_file.exists() and now - cache_file.stat().st_mtime < CACHE_TTL:
        df = pd.read_parquet(cache_file)
        _cache[key] = df
        _cache_ts[key] = now
        return df

    try:
        df = ak.stock_zh_index_daily(symbol=HS300_CODE)
        df.columns = ["date", "open", "high", "low", "close", "volume"]
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        df.to_parquet(cache_file, index=False)
        _cache[key] = df
        _cache_ts[key] = now
        return df
    except Exception as e:
        print(f"[WARN] Failed to fetch HS300: {e}", file=sys.stderr)
        return pd.DataFrame()


# ─── 策略计算 ──────────────────────────────────────────

def calc_momentum(df: pd.DataFrame, period: int = MOMENTUM_PERIOD) -> pd.Series:
    """计算 N 日动量 = close / close_shift(period) - 1"""
    return df["close"].pct_change(period)


def calc_ma(df: pd.DataFrame, period: int = MA_PERIOD) -> pd.Series:
    return df["close"].rolling(period).mean()


def calc_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """计算 RSI 指标"""
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """计算 ATR"""
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def get_market_state(hs300: pd.DataFrame) -> dict:
    """判断市场状态: 进攻/弱市/确认熊市"""
    if len(hs300) < MA_PERIOD + 1:
        return {"state": "unknown", "above_ma20": None, "momentum_20d": None, "position_cap": 0.25}

    close = hs300["close"].values
    ma20 = hs300["close"].rolling(MA_PERIOD).mean().values[-1]
    last_close = close[-1]
    mom_20d = close[-1] / close[-MOMENTUM_PERIOD - 1] - 1 if len(close) > MOMENTUM_PERIOD else 0

    above_ma20 = last_close > ma20
    mom_positive = mom_20d > 0

    if above_ma20 and mom_positive:
        state, cap = "进攻", 1.0
    elif not above_ma20 and not mom_positive:
        state, cap = "确认熊市", 0.25
    else:
        state, cap = "弱市", 0.70

    return {
        "state": state,
        "above_ma20": bool(above_ma20),
        "momentum_20d": round(mom_20d * 100, 2),
        "position_cap": cap,
        "hs300_close": round(last_close, 2),
        "hs300_ma20": round(ma20, 2),
    }


def get_etf_ranking(date: pd.Timestamp | None = None) -> list[dict]:
    """获取ETF动量排名 + RSI + 多策略评分"""
    results = []
    for symbol, name in ETF_POOL.items():
        df = get_etf_hist(symbol)
        if df.empty or len(df) < MOMENTUM_PERIOD + 1:
            continue
        if date:
            df = df[df["date"] <= date]
            if df.empty:
                continue

        close = df["close"].values
        ma20 = df["close"].rolling(MA_PERIOD).mean().values[-1]
        last_close = close[-1]
        mom = close[-1] / close[-MOMENTUM_PERIOD - 1] - 1
        above_ma20 = last_close > ma20

        # RSI
        rsi_series = calc_rsi(df)
        rsi_val = float(rsi_series.iloc[-1]) if not rsi_series.empty and not pd.isna(rsi_series.iloc[-1]) else 50.0

        # 成交量比 (今日量 / 20日均量)
        vol_now = float(df["volume"].iloc[-1]) if "volume" in df else 0
        vol_avg = float(df["volume"].iloc[-MA_PERIOD:].mean()) if len(df) >= MA_PERIOD else vol_now
        vol_ratio = vol_now / vol_avg if vol_avg > 0 else 1.0

        # ATR 波动率
        atr_series = calc_atr(df)
        atr_val = float(atr_series.iloc[-1]) if not atr_series.empty and not pd.isna(atr_series.iloc[-1]) else 0
        atr_pct = atr_val / last_close if last_close > 0 else 0

        # 多策略评分 (0-100)
        # 策略1: 动量 (40分) — 正动量加分
        s_momentum = min(40, max(0, mom * 100 + 20))
        # 策略2: 趋势 (30分) — 站上MA20 + RSI适中
        s_trend = 30 if above_ma20 else 5
        if above_ma20:
            s_trend = 25 if rsi_val > 70 else 30  # RSI超买扣分
        # 策略3: 量价 (15分) — 放量加分
        s_volume = min(15, max(0, (vol_ratio - 1) * 10 + 7))
        # 策略4: 风险 (15分) — 波动率越低越好
        s_risk = max(0, 15 - atr_pct * 100)

        total_score = round(s_momentum + s_trend + s_volume + s_risk, 1)

        results.append({
            "code": symbol,
            "name": name,
            "close": round(last_close, 4),
            "ma20": round(ma20, 4),
            "momentum_20d": round(mom * 100, 2),
            "above_ma20": bool(above_ma20),
            "rsi": round(rsi_val, 1),
            "vol_ratio": round(vol_ratio, 2),
            "atr_pct": round(atr_pct * 100, 2),
            "score": total_score,
            "score_breakdown": {
                "momentum": round(s_momentum, 1),
                "trend": round(s_trend, 1),
                "volume": round(s_volume, 1),
                "risk": round(s_risk, 1),
            },
            "selected": False,
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    # 标记 Top4 (且站上 MA20)
    selected = 0
    for item in results:
        if item["above_ma20"] and selected < TOP_N:
            item["selected"] = True
            selected += 1
    return results


# ─── 回测引擎 ──────────────────────────────────────────

def run_backtest(start_date: str = "2024-01-01", end_date: str = "2026-06-26") -> dict:
    """运行 ETF 动量轮动回测"""
    hs300 = get_hs300_hist()
    if hs300.empty:
        return {"error": "HS300 data unavailable"}

    # 加载所有ETF数据
    etf_data: dict[str, pd.DataFrame] = {}
    for symbol in ETF_POOL:
        df = get_etf_hist(symbol)
        if not df.empty:
            etf_data[symbol] = df

    if not etf_data:
        return {"error": "No ETF data available"}

    # 合并交易日历
    all_dates = hs300["date"].copy()
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    trade_dates = all_dates[(all_dates >= start) & (all_dates <= end)].tolist()

    if len(trade_dates) < REBALANCE_DAYS + MA_PERIOD:
        return {"error": f"Not enough trading days: {len(trade_dates)}"}

    # 初始化
    portfolio = {s: 0.0 for s in ETF_POOL}
    cash = 1_000_000.0
    initial_capital = cash
    cooldown: dict[str, int] = {}  # symbol -> cooldown remaining days
    holdings_history = []
    nav_history = []
    trade_log = []

    last_rebalance_idx = -REBALANCE_DAYS

    for i, date in enumerate(trade_dates):
        # 获取当日价格
        prices = {}
        for sym, df in etf_data.items():
            row = df[df["date"] <= date]
            if not row.empty:
                prices[sym] = row["close"].values[-1]

        if not prices:
            continue

        # 计算当前净值
        nav = cash
        for sym, shares in portfolio.items():
            if sym in prices and shares > 0:
                nav += shares * prices[sym]
        nav_history.append({"date": date.strftime("%Y-%m-%d"), "nav": round(nav, 2)})

        # 止损检查
        for sym in list(portfolio.keys()):
            if portfolio[sym] <= 0 or sym not in prices:
                continue
            # 简化: 用持仓以来的跌幅 (这里用最近 MOMENTUM_PERIOD 天)
            df = etf_data[sym]
            hist = df[df["date"] <= date]
            if len(hist) < MOMENTUM_PERIOD + 1:
                continue
            recent_return = hist["close"].values[-1] / hist["close"].values[-MOMENTUM_PERIOD - 1] - 1
            if recent_return <= STOP_LOSS_PCT:
                # 止损卖出
                proceeds = portfolio[sym] * prices[sym] * (1 - COMMISSION_RATE - SLIPPAGE_RATE)
                cash += proceeds
                trade_log.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "action": "stop_loss",
                    "symbol": sym,
                    "price": round(prices[sym], 4),
                    "shares": portfolio[sym],
                    "cash": round(cash, 2),
                })
                portfolio[sym] = 0
                cooldown[sym] = COOLDOWN_DAYS

        # 调仓检查
        if i - last_rebalance_idx >= REBALANCE_DAYS:
            last_rebalance_idx = i

            # 市场状态
            hs_hist = hs300[hs300["date"] <= date]
            if len(hs_hist) < MA_PERIOD + 1:
                continue
            market = get_market_state(hs_hist)
            position_cap = market["position_cap"]

            # ETF 排名
            ranking = get_etf_ranking(date)
            candidates = [r for r in ranking if r["above_ma20"] and r["selected"]]

            # 过滤冷却中的
            available = [r for r in candidates if cooldown.get(r["code"], 0) <= 0]

            # 目标持仓
            target_value = nav * position_cap
            per_etf = target_value / TOP_N if available else 0

            # 先卖出不在目标中的
            target_codes = {r["code"] for r in available[:TOP_N]}
            for sym in list(portfolio.keys()):
                if sym not in target_codes and portfolio[sym] > 0 and sym in prices:
                    proceeds = portfolio[sym] * prices[sym] * (1 - COMMISSION_RATE - SLIPPAGE_RATE)
                    cash += proceeds
                    trade_log.append({
                        "date": date.strftime("%Y-%m-%d"),
                        "action": "sell",
                        "symbol": sym,
                        "price": round(prices[sym], 4),
                        "shares": portfolio[sym],
                        "cash": round(cash, 2),
                    })
                    portfolio[sym] = 0

            # 买入目标
            for r in available[:TOP_N]:
                sym = r["code"]
                price = prices.get(sym)
                if not price:
                    continue
                buy_shares = (per_etf / price) * (1 - SLIPPAGE_RATE)
                cost = buy_shares * price * (1 + COMMISSION_RATE)
                if cash >= cost and buy_shares > 0:
                    cash -= cost
                    portfolio[sym] = portfolio.get(sym, 0) + buy_shares
                    trade_log.append({
                        "date": date.strftime("%Y-%m-%d"),
                        "action": "buy",
                        "symbol": sym,
                        "name": r["name"],
                        "price": round(price, 4),
                        "shares": round(buy_shares, 2),
                        "cash": round(cash, 2),
                    })

            holdings_history.append({
                "date": date.strftime("%Y-%m-%d"),
                "market_state": market["state"],
                "position_cap": market["position_cap"],
                "holdings": {s: round(portfolio[s] * prices.get(s, 0), 2) for s in portfolio if portfolio[s] > 0 and s in prices},
                "cash": round(cash, 2),
                "nav": round(nav, 2),
            })

        # 冷却倒计时
        for sym in list(cooldown.keys()):
            cooldown[sym] -= 1
            if cooldown[sym] <= 0:
                del cooldown[sym]

    # 最终净值
    final_nav = nav_history[-1]["nav"] if nav_history else initial_capital
    total_return = (final_nav - initial_capital) / initial_capital
    max_nav = max(n["nav"] for n in nav_history) if nav_history else initial_capital
    min_nav = min(n["nav"] for n in nav_history) if nav_history else initial_capital
    max_drawdown = (min_nav - max_nav) / max_nav if max_nav > 0 else 0

    # 年化
    days = (trade_dates[-1] - trade_dates[0]).days if len(trade_dates) > 1 else 1
    annual_return = (1 + total_return) ** (365.0 / days) - 1 if days > 0 else 0

    # 滚动窗口胜率
    window_size = 20
    wins = 0
    total_windows = 0
    for j in range(0, len(nav_history) - window_size, window_size):
        window_navs = [n["nav"] for n in nav_history[j:j + window_size]]
        if window_navs[-1] > window_navs[0]:
            wins += 1
        total_windows += 1
    win_rate = wins / total_windows if total_windows > 0 else 0

    return {
        "start_date": trade_dates[0].strftime("%Y-%m-%d") if trade_dates else start_date,
        "end_date": trade_dates[-1].strftime("%Y-%m-%d") if trade_dates else end_date,
        "initial_capital": initial_capital,
        "final_nav": final_nav,
        "total_return_pct": round(total_return * 100, 2),
        "annual_return_pct": round(annual_return * 100, 2),
        "max_drawdown_pct": round(max_drawdown * 100, 2),
        "win_rate_pct": round(win_rate * 100, 1),
        "total_trades": len(trade_log),
        "trade_dates": len(trade_dates),
        "nav_history": nav_history[-60:],
        "holdings_history": holdings_history[-20:],
        "trade_log": trade_log[-30:],
    }


# ─── API 服务器 ────────────────────────────────────────

class APIHandler(http.server.BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self._send_cors()
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        if path == "/" or path == "/index.html":
            self._serve_static("index.html")
        elif path == "/api/market":
            self._handle_market()
        elif path == "/api/ranking":
            self._handle_ranking()
        elif path == "/api/backtest":
            self._handle_backtest(qs)
        elif path == "/api/etf_spot":
            self._handle_etf_spot()
        elif path == "/api/heatmap":
            self._handle_heatmap()
        elif path == "/api/brief":
            self._handle_brief()
        elif path == "/api/chain":
            self._handle_chain()
        elif path == "/api/radar":
            self._handle_radar()
        else:
            self._json({"error": "not found"}, 404)

    def _send_cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _json(self, data: dict, code: int = 200):
        body = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._send_cors()
        self.end_headers()
        self.wfile.write(body)

    def _serve_static(self, filename: str):
        filepath = Path(__file__).parent / filename
        if not filepath.exists():
            self._json({"error": "file not found"}, 404)
            return
        body = filepath.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._send_cors()
        self.end_headers()
        self.wfile.write(body)

    def _handle_market(self):
        hs300 = get_hs300_hist()
        if hs300.empty:
            self._json({"error": "HS300 data unavailable"}, 503)
            return
        state = get_market_state(hs300)
        recent = hs300.tail(30)
        sparkline = [
            {"date": d.strftime("%Y-%m-%d"), "close": round(c, 2)}
            for d, c in zip(recent["date"], recent["close"])
        ]
        self._json({"market": state, "sparkline": sparkline})

    def _handle_ranking(self):
        ranking = get_etf_ranking()
        self._json({"ranking": ranking, "pool_size": len(ETF_POOL), "updated": dt.datetime.now().isoformat()})

    def _handle_backtest(self, qs: dict):
        start = qs.get("start", ["2024-01-01"])[0]
        end = qs.get("end", ["2026-06-26"])[0]
        result = run_backtest(start, end)
        self._json(result)

    def _handle_etf_spot(self):
        try:
            df = ak.fund_etf_spot_em()
            pool_codes = set(ETF_POOL.keys())
            df["代码"] = df["代码"].astype(str)
            filtered = df[df["代码"].isin(pool_codes)]
            result = []
            for _, row in filtered.iterrows():
                code = str(row.get("代码", ""))
                result.append({
                    "code": code,
                    "name": ETF_POOL.get(code, str(row.get("名称", ""))),
                    "price": float(row.get("最新价", 0) or 0),
                    "pct_change": float(row.get("涨跌幅", 0) or 0),
                    "volume": float(row.get("成交量", 0) or 0),
                    "amount": float(row.get("成交额", 0) or 0),
                    "main_net_inflow": float(row.get("主力净流入-净额", 0) or 0),
                    "main_net_pct": float(row.get("主力净流入-净占比", 0) or 0),
                })
            self._json({"spots": result, "updated": dt.datetime.now().isoformat()})
        except Exception as e:
            self._json({"error": str(e)}, 503)

    def _handle_heatmap(self):
        """ETF 热力图数据：涨跌幅 + 主力资金 + 评分"""
        try:
            ranking = get_etf_ranking()
            spots_map = {}
            try:
                df_spot = ak.fund_etf_spot_em()
                df_spot["代码"] = df_spot["代码"].astype(str)
                for _, row in df_spot[df_spot["代码"].isin(set(ETF_POOL.keys()))].iterrows():
                    code = str(row.get("代码", ""))
                    spots_map[code] = {
                        "pct_change": float(row.get("涨跌幅", 0) or 0),
                        "main_inflow": float(row.get("主力净流入-净额", 0) or 0),
                    }
            except Exception:
                pass

            heatmap = []
            for r in ranking:
                spot = spots_map.get(r["code"], {})
                heatmap.append({
                    "code": r["code"],
                    "name": r["name"],
                    "pct_change": spot.get("pct_change", r.get("momentum_20d", 0)),
                    "score": r["score"],
                    "main_inflow": spot.get("main_inflow", 0),
                    "momentum": r["momentum_20d"],
                    "rsi": r["rsi"],
                })
            self._json({"heatmap": heatmap, "updated": dt.datetime.now().isoformat()})
        except Exception as e:
            self._json({"error": str(e)}, 503)

    def _handle_brief(self):
        """盘报：当日市场摘要"""
        try:
            hs300 = get_hs300_hist()
            market = get_market_state(hs300)
            ranking = get_etf_ranking()
            selected = [r for r in ranking if r["selected"]]
            top_momentum = ranking[:5] if ranking else []
            gainers = [r for r in ranking if r["momentum_20d"] > 5]
            losers = [r for r in ranking if r["momentum_20d"] < -5]

            recent = hs300.tail(5)
            hs5d = [{"date": row["date"].strftime("%Y-%m-%d"), "close": round(row["close"], 2)} for _, row in recent.iterrows()]

            self._json({
                "market": market,
                "hs300_5d": hs5d,
                "top_momentum": [{"code": r["code"], "name": r["name"], "momentum": r["momentum_20d"], "score": r["score"]} for r in top_momentum],
                "selected": [{"code": r["code"], "name": r["name"], "score": r["score"], "rsi": r["rsi"]} for r in selected],
                "gainers_count": len(gainers),
                "losers_count": len(losers),
                "total_pool": len(ranking),
                "updated": dt.datetime.now().isoformat(),
            })
        except Exception as e:
            self._json({"error": str(e)}, 503)

    def _handle_chain(self):
        """产业链视图：AI算力 + 机器人，ETF 映射到产业链环节"""
        ranking_map = {}
        try:
            ranking = get_etf_ranking()
            ranking_map = {r["code"]: r for r in ranking}
        except Exception:
            pass

        chain = {
            "title": "AI算力 + 机器人 产业链",
            "subtitle": "ETF 池映射到产业链环节，参考 ainiusq.com/niu 看板",
            "updated": dt.datetime.now().isoformat(),
            "layers": [
                {"layer": "上游", "nodes": [
                    {"segment": "AI芯片", "etfs": ["512480", "512760"], "desc": "GPU/ASIC/芯片设计", "cycle": "成长期"},
                    {"segment": "存储芯片", "etfs": [], "desc": "HBM/DRAM/NAND", "cycle": "复苏期"},
                    {"segment": "光模块", "etfs": ["515880"], "desc": "800G/1.6T光连接", "cycle": "爆发期"},
                    {"segment": "PCB/连接器", "etfs": [], "desc": "高速PCB、连接器", "cycle": "成长期"},
                    {"segment": "电源/散热", "etfs": [], "desc": "VRM、液冷、热管理", "cycle": "成长期"},
                ]},
                {"layer": "中游", "nodes": [
                    {"segment": "服务器/算力", "etfs": [], "desc": "AI服务器、算力租赁", "cycle": "扩张期"},
                    {"segment": "通信基础设施", "etfs": ["515880"], "desc": "交换机、光网络", "cycle": "成长期"},
                    {"segment": "机器人执行器", "etfs": ["562500"], "desc": "关节、电控、散热", "cycle": "导入期"},
                    {"segment": "机器人感知", "etfs": ["562500"], "desc": "视觉、力矩、定位", "cycle": "导入期"},
                ]},
                {"layer": "下游", "nodes": [
                    {"segment": "应用/软件", "etfs": [], "desc": "AI应用、Agent、软件", "cycle": "概念期"},
                    {"segment": "整机/集成", "etfs": ["512660"], "desc": "机器人整机、系统集成", "cycle": "导入期"},
                    {"segment": "消费/白酒", "etfs": ["512690", "515170"], "desc": "消费场景", "cycle": "成熟期"},
                    {"segment": "医药/生物", "etfs": ["512010"], "desc": "医疗机器人、AI制药", "cycle": "概念期"},
                ]},
            ],
            "theme_chains": [
                {"name": "AI算力", "key": "ai_compute", "etfs": ["512480", "512760", "515880"]},
                {"name": "机器人", "key": "robotics", "etfs": ["562500", "512660"]},
                {"name": "半导体", "key": "semiconductor", "etfs": ["512480", "512760"]},
                {"name": "新能源", "key": "new_energy", "etfs": ["515790", "515030"]},
                {"name": "消费", "key": "consumer", "etfs": ["512690", "515170"]},
                {"name": "金融", "key": "finance", "etfs": ["512800", "512070"]},
                {"name": "军工", "key": "defense", "etfs": ["512660", "512670"]},
                {"name": "宽基", "key": "broad", "etfs": ["510300", "510500", "510050", "159915", "512100"]},
            ],
            "nodes": [],
        }

        # 填充每个环节的 ETF 详细数据 + S/A/B 分级
        for layer in chain["layers"]:
            for node in layer["nodes"]:
                node_etfs = []
                for code in node["etfs"]:
                    r = ranking_map.get(code)
                    if r:
                        grade = "S" if r["score"] >= 80 else "A" if r["score"] >= 60 else "B" if r["score"] >= 40 else "C"
                        node_etfs.append({
                            "code": code, "name": r["name"], "score": r["score"], "grade": grade,
                            "momentum": r["momentum_20d"], "rsi": r["rsi"], "above_ma20": r["above_ma20"],
                        })
                node["etf_data"] = node_etfs
                chain["nodes"].append(node)

        self._json(chain)

    def _handle_radar(self):
        """变局雷达：动量变化最大的标的"""
        try:
            ranking = get_etf_ranking()
            # 按动量绝对值排序（变化最大的）
            radar = sorted(ranking, key=lambda x: abs(x["momentum_20d"]), reverse=True)[:10]
            result = []
            for r in radar:
                grade = "S" if r["score"] >= 80 else "A" if r["score"] >= 60 else "B" if r["score"] >= 40 else "C"
                direction = "加速" if r["momentum_20d"] > 0 else "减速"
                alert = ""
                if r["rsi"] > 70:
                    alert = "RSI超买"
                elif r["rsi"] < 30:
                    alert = "RSI超卖"
                if not r["above_ma20"]:
                    alert += ("趋势破位" if alert else "趋势破位")
                result.append({
                    "code": r["code"], "name": r["name"], "score": r["score"], "grade": grade,
                    "momentum": r["momentum_20d"], "rsi": r["rsi"], "direction": direction,
                    "alert": alert, "above_ma20": r["above_ma20"],
                })
            self._json({"radar": result, "updated": dt.datetime.now().isoformat()})
        except Exception as e:
            self._json({"error": str(e)}, 503)

    def log_message(self, format, *args):
        # 静默日志
        pass


def main():
    port = 8765
    if len(sys.argv) > 1:
        port = int(sys.argv[1])

    # 预热缓存
    print("Preheating data cache...", file=sys.stderr)
    get_hs300_hist()
    for sym in ETF_POOL:
        get_etf_hist(sym)
    print(f"Cache ready. {len(ETF_POOL)} ETFs + HS300 loaded.", file=sys.stderr)

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", port), APIHandler) as httpd:
        print(f"Server running at http://127.0.0.1:{port}/", file=sys.stderr)
        httpd.serve_forever()


if __name__ == "__main__":
    main()

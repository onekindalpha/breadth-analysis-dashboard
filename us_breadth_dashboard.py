#!/usr/bin/env python3
from __future__ import annotations
# US Market Breadth Dashboard — 스탠 와인스태인 방식
# 데이터: yfinance (검증된 심볼만 사용)
# AD Line: 다우30 / NASDAQ100 구성종목 일별 등락 집계로 계산

import io
import requests as _requests
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    import yfinance as yf
    YF_OK = True
except ImportError:
    YF_OK = False

# ──────────────────────────────────────────────────────────────
# S&P500 (NYSE 대표) / NASDAQ100 구성종목
# Wikipedia에서 최신 목록 가져옴. 실패 시 하드코딩 fallback 사용.
# ──────────────────────────────────────────────────────────────

# S&P500 fallback (NYSE 전체 대표)
SP500_FALLBACK = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","TSLA","BRK-B","AVGO","JPM",
    "LLY","UNH","V","XOM","MA","JNJ","PG","HD","COST","MRK","ABBV","BAC",
    "NFLX","KO","CRM","PEP","AMD","TMO","WMT","ORCL","MCD","LIN","CSCO",
    "GE","ABT","ACN","IBM","TXN","CAT","INTU","GS","AMGN","SPGI","DHR",
    "AXP","NOW","RTX","VZ","ISRG","NEE","HON","PFE","MS","BX","BKNG","LOW",
    "UBER","UNP","PM","TJX","AMAT","QCOM","ELV","ETN","PLD","SYK","C","BA",
    "BSX","DE","REGN","VRTX","MDT","CB","ADI","PANW","MU","GILD","ADP","CVS",
    "WM","SO","CME","MMC","PGR","ZTS","SCHW","AMT","CI","DUK","ITW","AON",
    "NOC","APD","FI","ICE","SHW","MCO","EOG","MCK","USB","TGT","EMR","HCA",
    "MMM","WFC","BDX","LRCX","MO","ECL","KLAC","F","SNPS","CDNS","FCX","NSC",
]

# NASDAQ100 fallback
NDX100_FALLBACK = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","TSLA","AVGO","COST",
    "NFLX","AMD","ADBE","QCOM","INTU","TXN","AMGN","ISRG","BKNG","VRTX",
    "MU","LRCX","PANW","KLAC","MRVL","AMAT","SNPS","CDNS","ABNB","CRWD",
    "MELI","ORLY","REGN","FTNT","CTAS","PCAR","MNST","CPRT","DXCM","TEAM",
    "KDP","ODFL","ROST","WDAY","PAYX","IDXX","EXC","FAST","GEHC","DLTR",
    "BIIB","VRSK","CTSH","ZS","ANSS","ALGN","ON","CEG","DDOG","TTWO","MRNA",
]

@st.cache_data(show_spinner=False, ttl=86400)
def get_sp500_tickers() -> list[str]:
    """Wikipedia에서 S&P500 티커 목록 가져오기. 실패 시 fallback."""
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        df = tables[0]
        syms = df["Symbol"].tolist()
        # BRK.B → BRK-B 형식 변환
        syms = [s.replace(".", "-") for s in syms if isinstance(s, str)]
        return [s for s in syms if len(s) <= 5]
    except Exception:
        return SP500_FALLBACK

@st.cache_data(show_spinner=False, ttl=86400)
def get_ndx100_tickers() -> list[str]:
    """Wikipedia에서 NASDAQ100 티커 목록 가져오기. 실패 시 fallback."""
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100")
        for t in tables:
            cols = [str(c).lower() for c in t.columns]
            if any("ticker" in c or "symbol" in c for c in cols):
                col = t.columns[[i for i, c in enumerate(cols) if "ticker" in c or "symbol" in c][0]]
                syms = t[col].tolist()
                syms = [s.replace(".", "-") for s in syms if isinstance(s, str)]
                return [s for s in syms if len(s) <= 5]
    except Exception:
        pass
    return NDX100_FALLBACK

MARKET_CFG = {
    "NYSE": {
        "get_tickers": get_sp500_tickers,   # S&P500 = NYSE 대표 Index
        "idx_sym":     "^GSPC",             # S&P500 Index
        "cmp_sym":     "SPY",               # 비교용 ETF
        "label":       "NYSE (S&P500 기준, 500종목)",
        "yf_pd_sym":   "SPY",
        "div_fallback": 0.015,
    },
    "NASDAQ": {
        "get_tickers": get_ndx100_tickers,  # NASDAQ100
        "idx_sym":     "^IXIC",             # NASDAQ Composite
        "cmp_sym":     "QQQ",               # 비교용 ETF
        "label":       "NASDAQ (NDX100 기준)",
        "yf_pd_sym":   "QQQ",
        "div_fallback": 0.006,
    },
}

STATUS_MAP = {
    "BULLISH_CONFIRMATION":          ("✅ Bullish Confirmation",       "가격·A/D선 모두 High 동행",      "#2e7d32"),
    "BULLISH_DIVERGENCE":            ("🔴 부정적 불일치",    "가격 High / A/D선 크게 뒤처짐",  "#c62828"),
    "BULLISH_DIVERGENCE_CANDIDATE":  ("🟠 초기 경고",        "가격이 A/D선보다 빠르게 회복",   "#ef6c00"),
    "RECOVERY_IN_PROGRESS":          ("🟡Recovery in Progress",      "High 재공략 중, A/D 미확인",     "#f9a825"),
    "DOWNSIDE_DIVERGENCE_CANDIDATE": ("🟢 긍정적 불일치",     "가격 저점 / A/D선은 더 올라옴",  "#00838f"),
    "NORMAL_WEAKNESS":               ("⚫ Broad Weakness",        "가격·A/D선 모두 저점",           "#455a64"),
    "NEUTRAL":                       ("⬜ Neutral",              "No clear signal",                "#757575"),
}

# ──────────────────────────────────────────────────────────────
# 데이터 수집
# ──────────────────────────────────────────────────────────────
def _yf_download(syms: list[str], start: str, end: str) -> pd.DataFrame:
    """yf.download()로 여러 심볼 Close 일괄 수집. YYYYMMDD → DataFrame(날짜×심볼)"""
    s = pd.to_datetime(start, format="%Y%m%d").strftime("%Y-%m-%d")
    e = (pd.to_datetime(end,   format="%Y%m%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    raw = yf.download(syms, start=s, end=e, auto_adjust=True, progress=False, threads=True)
    if raw is None or raw.empty:
        raise RuntimeError("yfinance download 결과 없음")
    # Close 추출
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = raw[["Close"]] if "Close" in raw.columns else raw
    if hasattr(close.index, "tz") and close.index.tz is not None:
        close.index = close.index.tz_localize(None)
    else:
        close.index = pd.to_datetime(close.index)
    return close.sort_index()

def _yf_ticker_history(sym: str, start: str, end: str) -> pd.Series:
    """단일 심볼 Close Series"""
    s = pd.to_datetime(start, format="%Y%m%d").strftime("%Y-%m-%d")
    e = (pd.to_datetime(end,   format="%Y%m%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    raw = yf.Ticker(sym).history(start=s, end=e, auto_adjust=True)
    if raw is None or raw.empty:
        raise RuntimeError(f"yfinance {sym} 데이터 없음")
    if hasattr(raw.index, "tz") and raw.index.tz is not None:
        raw.index = raw.index.tz_localize(None)
    else:
        raw.index = pd.to_datetime(raw.index)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    return pd.to_numeric(raw["Close"], errors="coerce").dropna().sort_index()

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_breadth(market: str, start: str, end: str, base: float = 50000.0) -> pd.DataFrame:
    """
    NYSE / NASDAQ 전체 등락 데이터 수집.
    Yahoo Finance 심볼:
      C:ISSU = NYSE Advance/Decline/Unchanged  (전체 NYSE 종목)
      C:ISSQ = NASDAQ Advance/Decline/Unchanged (전체 NASDAQ 종목)
    Close  = 상승 종목 수
    Open   = 하락 종목 수  (Yahoo Finance 구조)
    트레이딩뷰 $ADD / $NAAD 와 동일한 거래소 전체 기준.
    """
    if not YF_OK:
        raise RuntimeError("yfinance 미설치")

    AD_SYMS = {"NYSE": "C:ISSU", "NASDAQ": "C:ISSQ"}
    ad_sym = AD_SYMS[market]

    s = pd.to_datetime(start, format="%Y%m%d").strftime("%Y-%m-%d")
    e = (pd.to_datetime(end, format="%Y%m%d") + timedelta(days=1)).strftime("%Y-%m-%d")

    raw = yf.Ticker(ad_sym).history(start=s, end=e, auto_adjust=False)
    if raw is None or raw.empty:
        raise RuntimeError(
            f"'{ad_sym}' 데이터 없음 — Yahoo Finance에서 해당 심볼을 지원하지 않습니다.\n"
            f"브라우저에서 https://finance.yahoo.com/quote/{ad_sym} 접속해 확인하세요."
        )
    if hasattr(raw.index, "tz") and raw.index.tz is not None:
        raw.index = raw.index.tz_localize(None)
    else:
        raw.index = pd.to_datetime(raw.index)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    # Yahoo Finance C:ISSU 구조: Close=상승, Open=하락, Volume=보합
    advances = pd.to_numeric(raw.get("Close", raw.iloc[:, 0]), errors="coerce").fillna(0)
    declines = pd.to_numeric(raw.get("Open",  raw.iloc[:, 1]), errors="coerce").fillna(0)
    ad_diff  = (advances - declines).sort_index()
    advances = advances.reindex(ad_diff.index)
    declines = declines.reindex(ad_diff.index)

    df = pd.DataFrame({
        "advances": advances.values,
        "declines": declines.values,
        "ad_diff":  ad_diff.values,
    }, index=ad_diff.index)
    df["ad_line"] = base + df["ad_diff"].cumsum()
    df["date"]    = df.index.strftime("%Y%m%d")
    return df.reset_index(drop=True)

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_index(market: str, start: str, end: str) -> pd.DataFrame:
    """Index OHLC — ^DJI / ^IXIC (yfinance에서 확실히 작동)"""
    if not YF_OK:
        raise RuntimeError("yfinance 미설치")
    cfg = MARKET_CFG[market]
    s = pd.to_datetime(start, format="%Y%m%d").strftime("%Y-%m-%d")
    e = (pd.to_datetime(end,   format="%Y%m%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    raw = yf.Ticker(cfg["idx_sym"]).history(start=s, end=e, auto_adjust=True)
    if raw is None or raw.empty:
        raise RuntimeError(f"{cfg['idx_sym']} 데이터 없음")
    if hasattr(raw.index, "tz") and raw.index.tz is not None:
        raw.index = raw.index.tz_localize(None)
    else:
        raw.index = pd.to_datetime(raw.index)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    out = pd.DataFrame({
        "date":  raw.index.strftime("%Y%m%d"),
        "open":  pd.to_numeric(raw["Open"],  errors="coerce"),
        "high":  pd.to_numeric(raw["High"],  errors="coerce"),
        "low":   pd.to_numeric(raw["Low"],   errors="coerce"),
        "close": pd.to_numeric(raw["Close"], errors="coerce"),
    })
    return out.dropna(subset=["close"]).reset_index(drop=True)

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_nhnl(market: str, start: str, end: str) -> pd.DataFrame | None:
    """
    52주 신고가/신저가: Yahoo Finance 전체 거래소 심볼 사용.
    NYSE:   C:HISU (신고가) / C:LOSU (신저가)
    NASDAQ: C:HISQ (신고가) / C:LOSQ (신저가)
    트레이딩뷰 $NHNL / $NANAHNL 과 동일한 거래소 전체 기준.
    """
    if not YF_OK:
        return None

    NH_SYMS = {"NYSE": ("C:HISU", "C:LOSU"), "NASDAQ": ("C:HISQ", "C:LOSQ")}
    hi_sym, lo_sym = NH_SYMS[market]

    s = pd.to_datetime(start, format="%Y%m%d").strftime("%Y-%m-%d")
    e = (pd.to_datetime(end,   format="%Y%m%d") + timedelta(days=1)).strftime("%Y-%m-%d")

    errors = []
    try:
        raw_hi = yf.Ticker(hi_sym).history(start=s, end=e, auto_adjust=False)
        raw_lo = yf.Ticker(lo_sym).history(start=s, end=e, auto_adjust=False)
        if raw_hi is None or raw_hi.empty:
            errors.append(f"{hi_sym} 데이터 없음")
        if raw_lo is None or raw_lo.empty:
            errors.append(f"{lo_sym} 데이터 없음")
        if errors:
            raise RuntimeError(" / ".join(errors))

        for r in [raw_hi, raw_lo]:
            if hasattr(r.index, "tz") and r.index.tz is not None:
                r.index = r.index.tz_localize(None)
            else:
                r.index = pd.to_datetime(r.index)
            if isinstance(r.columns, pd.MultiIndex):
                r.columns = r.columns.get_level_values(0)

        new_highs = pd.to_numeric(raw_hi["Close"], errors="coerce").dropna().sort_index()
        new_lows  = pd.to_numeric(raw_lo["Close"], errors="coerce").dropna().sort_index()

        df = pd.DataFrame({"new_highs": new_highs, "new_lows": new_lows}).dropna()
        # 주봉으로 리샘플 (금요일 기준)
        weekly_hi = df["new_highs"].resample("W-FRI").last()
        weekly_lo = df["new_lows"].resample("W-FRI").last()
        weekly    = pd.DataFrame({"new_highs": weekly_hi, "new_lows": weekly_lo}).dropna()
        weekly["nhnl"] = weekly["new_highs"] - weekly["new_lows"]
        weekly["date"] = weekly.index.strftime("%Y%m%d")
        return weekly.reset_index(drop=True)
    except Exception as e_:
        # 세션 상태에 오류 저장해서 UI에 표시
        import streamlit as _st
        _st.session_state["nhnl_error"] = str(e_)
        return None

@st.cache_data(show_spinner=False, ttl=86400)
def fetch_pd(market: str, months: int):
    """P/D = Index Close ÷ 연간 실제 배당금 (rolling 365일 합산)"""
    if not YF_OK:
        return None, "yfinance 미설치"
    try:
        cfg     = MARKET_CFG[market]
        end_d   = datetime.today()
        start_d = end_d - timedelta(days=max(365 * 6, months * 35))
        ticker  = yf.Ticker(cfg["yf_pd_sym"])
        ph = ticker.history(start=start_d.strftime("%Y-%m-%d"),
                            end=end_d.strftime("%Y-%m-%d"), auto_adjust=True)
        if ph is None or ph.empty:
            return None, "가격 데이터 없음"
        if hasattr(ph.index, "tz") and ph.index.tz is not None:
            ph.index = ph.index.tz_localize(None)
        else:
            ph.index = pd.to_datetime(ph.index)
        if isinstance(ph.columns, pd.MultiIndex):
            ph.columns = ph.columns.get_level_values(0)
        close_s = pd.to_numeric(ph["Close"], errors="coerce").dropna().sort_index()
        divs = ticker.dividends
        if divs is not None and not divs.empty:
            if hasattr(divs.index, "tz") and divs.index.tz is not None:
                divs.index = divs.index.tz_localize(None)
            else:
                divs.index = pd.to_datetime(divs.index)
            divs_d  = divs.reindex(close_s.index, fill_value=0.0)
            ann_div = divs_d.rolling(365, min_periods=1).sum()
        else:
            ann_div = close_s * cfg["div_fallback"]
        wc  = close_s.resample("W-FRI").last().dropna()
        wd  = ann_div.resample("W-FRI").last().reindex(wc.index).ffill().replace(0, float("nan"))
        pdr = wc / wd
        out = pd.DataFrame({
            "date": wc.index.strftime("%Y%m%d"),
            "close": wc.values, "dividend_est": wd.values, "pd_ratio": pdr.values,
        }).dropna(subset=["pd_ratio"])
        out["div_yield"] = out["dividend_est"] / out["close"]
        out["dt"] = pd.to_datetime(out["date"], format="%Y%m%d")
        return out.reset_index(drop=True), None
    except Exception as ex:
        return None, str(ex)

# ──────────────────────────────────────────────────────────────
# 판정
# ──────────────────────────────────────────────────────────────
def classify(poh, aoh, gap, pol, aol, pt=2.0, at=3.0, gw=1.5, gd=2.5):
    if poh >= -pt and aoh >= -at and gap >= -1.0: return "BULLISH_CONFIRMATION"
    if poh >= -pt and gap <= -gd:                 return "BULLISH_DIVERGENCE"
    if gap <= -gw:                                return "BULLISH_DIVERGENCE_CANDIDATE"
    if gap < -1.0:                                return "RECOVERY_IN_PROGRESS"
    if pol <= pt and not (aol <= at):             return "DOWNSIDE_DIVERGENCE_CANDIDATE"
    if pol <= pt and aol <= at:                   return "NORMAL_WEAKNESS"
    return "NEUTRAL"

def compute_hlab(df: pd.DataFrame, high_bars: int = 60, low_bars: int = 130) -> dict:
    closes  = df["close"].values.astype(float)
    ad_line = df["ad_line"].values.astype(float)
    dts     = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
    n = len(closes)

    def _safe_slice(arr, end_idx, length):
        start = max(0, end_idx - length)
        return arr[start:end_idx], start

    hb_window, hb_start = _safe_slice(closes, n, high_bars)
    if len(hb_window) == 0:
        hb_window = closes
        hb_start  = 0
    hb_idx_local = int(np.argmax(hb_window))
    hb_idx = hb_start + hb_idx_local
    hb_val, hb_dt, hb_ad = closes[hb_idx], dts.iloc[hb_idx], ad_line[hb_idx]

    ha_window, ha_start = _safe_slice(closes, hb_start + hb_idx_local, high_bars)
    if len(ha_window) > 0:
        ha_idx_local = int(np.argmax(ha_window))
        ha_idx = ha_start + ha_idx_local
        ha_val, ha_dt, ha_ad = closes[ha_idx], dts.iloc[ha_idx], ad_line[ha_idx]
    else:
        ha_val, ha_dt, ha_ad = hb_val, hb_dt, hb_ad

    lb_window, lb_start = _safe_slice(closes, n, low_bars)
    if len(lb_window) == 0:
        lb_window = closes
        lb_start  = 0
    lb_idx_local = int(np.argmin(lb_window))
    lb_idx = lb_start + lb_idx_local
    lb_val, lb_dt, lb_ad = closes[lb_idx], dts.iloc[lb_idx], ad_line[lb_idx]

    la_window, la_start = _safe_slice(closes, lb_start + lb_idx_local, low_bars)
    if len(la_window) > 0:
        la_idx_local = int(np.argmin(la_window))
        la_idx = la_start + la_idx_local
        la_val, la_dt, la_ad = closes[la_idx], dts.iloc[la_idx], ad_line[la_idx]
    else:
        la_val, la_dt, la_ad = lb_val, lb_dt, lb_ad

    bear_div     = bool(hb_val > ha_val and hb_ad < ha_ad)
    bear_div_pct = abs((ha_ad - hb_ad) / ha_ad * 100) if (bear_div and ha_ad != 0) else 0.0
    bull_div     = bool(lb_val < la_val and lb_ad > la_ad)
    bull_div_pct = abs((lb_ad - la_ad) / la_ad * 100) if (bull_div and la_ad != 0) else 0.0

    return dict(
        hb_val=hb_val, hb_dt=hb_dt, hb_ad=hb_ad,
        ha_val=ha_val, ha_dt=ha_dt, ha_ad=ha_ad,
        lb_val=lb_val, lb_dt=lb_dt, lb_ad=lb_ad,
        la_val=la_val, la_dt=la_dt, la_ad=la_ad,
        bear_div=bear_div, bear_div_pct=bear_div_pct,
        bull_div=bull_div, bull_div_pct=bull_div_pct,
    )

def compute_signals(df, lookback, pt, at, gw, gd):
    closes = df["close"].values.astype(float)
    ads    = df["ad_line"].values.astype(float)
    w      = closes[-lookback:]
    pi     = w.argmax(); da = lookback - 1 - pi
    ph     = w[pi]; ap = ads[-(da + 1)]
    lc     = closes[-1]; la = ads[-1]
    pl     = closes[-lookback:].min(); al = ads[-lookback:].min()
    poff   = (lc - ph) / abs(ph) * 100 if ph else float("nan")
    aoff   = (la - ap) / abs(ap) * 100 if ap else float("nan")
    gap    = aoff - poff
    poll   = (lc - pl) / abs(pl) * 100 if pl else float("nan")
    aoll   = (la - al) / abs(al) * 100 if al else float("nan")
    peak_d = str(df["date"].iloc[-(da + 1)])
    plbl   = "오늘" if da == 0 else f"{da}일전 ({peak_d})"
    sk     = classify(poff, aoff, gap, poll, aoll, pt, at, gw, gd)
    v, n, c = STATUS_MAP[sk]
    return dict(peak_label=plbl, price_off=poff, ad_off=aoff, gap=gap,
                verdict=v, note=n, color=c, last_close=lc, last_ad=la,
                price_high=ph, ad_at_peak=ap)

# ──────────────────────────────────────────────────────────────
# 차트 — 2패널 (위: Index캔들 단독 / 아래: A/D Line 단독)
# ──────────────────────────────────────────────────────────────
def make_plotly_chart(df, market, sig, chart_months, hlab) -> go.Figure:
    end_dt   = pd.to_datetime(df["date"].astype(str), format="%Y%m%d").max()
    start_dt = end_dt - pd.DateOffset(months=chart_months)
    mask     = pd.to_datetime(df["date"].astype(str), format="%Y%m%d") >= start_dt
    pf       = df[mask].copy().reset_index(drop=True)
    pf["dt"] = pd.to_datetime(pf["date"].astype(str), format="%Y%m%d")

    hb_color = "rgba(255,80,80,0.95)"  if hlab["bear_div"] else "rgba(160,160,160,0.8)"
    ha_color = "rgba(255,140,140,0.6)" if hlab["bear_div"] else "rgba(120,120,120,0.5)"
    lb_color = "rgba(38,210,160,0.95)" if hlab["bull_div"] else "rgba(160,160,160,0.8)"
    la_color = "rgba(38,210,160,0.6)"  if hlab["bull_div"] else "rgba(120,120,120,0.5)"

    if hlab["bear_div"]:
        div_text, div_color = f"⚠ 부정적 불일치 {hlab['bear_div_pct']:.1f}%", "#ff5050"
    elif hlab["bull_div"]:
        div_text, div_color = f"✓ 긍정적 불일치 {hlab['bull_div_pct']:.1f}%", "#26d2a0"
    else:
        div_text, div_color = "불일치 없음", "#aaaaaa"

    # domain 수동 분할 — rangeslider 문제 완전 회피
    fig = go.Figure()

    # ── 위 패널 (y축: domain 0.52~1.0): 캔들스틱
    fig.add_trace(go.Candlestick(
        x=pf["dt"], open=pf["open"], high=pf["high"], low=pf["low"], close=pf["close"],
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        name=market, showlegend=False,
        xaxis="x", yaxis="y1",
    ))

    # ── 아래 패널 (y축: domain 0.0~0.48): A/D Line
    fig.add_trace(go.Scatter(
        x=pf["dt"], y=pf["ad_line"].astype(float),
        line=dict(color="#1e88e5", width=2.0), name="A/D Line",
        xaxis="x", yaxis="y2",
    ))

    # 가격 정규화선 (아래 패널 — A/D 스케일로 맞춤)
    ad_vals = pf["ad_line"].astype(float)
    ad_min, ad_max = ad_vals.min(), ad_vals.max()
    pr_min, pr_max = pf["close"].min(), pf["close"].max()
    if pr_max != pr_min:
        price_mapped = ad_min + (pf["close"] - pr_min) / (pr_max - pr_min) * (ad_max - ad_min)
    else:
        price_mapped = ad_vals
    fig.add_trace(go.Scatter(
        x=pf["dt"], y=price_mapped,
        line=dict(color="rgba(180,180,180,0.35)", width=1.0),
        name="가격(참조)", showlegend=False,
        xaxis="x", yaxis="y2",
    ))

    # 위 패널 수평선 (H_b/H_a/L_b/L_a)
    for val, color, dash, ann in [
        (hlab["hb_val"], hb_color, "dash", f"H_b {hlab['hb_val']:,.0f}"),
        (hlab["ha_val"], ha_color, "dot",  f"H_a {hlab['ha_val']:,.0f}"),
        (hlab["lb_val"], lb_color, "dash", f"L_b {hlab['lb_val']:,.0f}"),
        (hlab["la_val"], la_color, "dot",  f"L_a {hlab['la_val']:,.0f}"),
    ]:
        fig.add_shape(type="line", x0=pf["dt"].iloc[0], x1=pf["dt"].iloc[-1],
                      y0=val, y1=val, xref="x", yref="y1",
                      line=dict(color=color, dash=dash, width=1.2))
        fig.add_annotation(x=pf["dt"].iloc[-1], y=val, xref="x", yref="y1",
                           text=ann, font=dict(color=color, size=10),
                           xanchor="left", showarrow=False)

    # 아래 패널 수평선 (A/D H_b/H_a/L_b/L_a)
    for val, color, dash, ann in [
        (hlab["hb_ad"], hb_color, "dash", f"A/D H_b {hlab['hb_ad']:,.0f}"),
        (hlab["ha_ad"], ha_color, "dot",  f"A/D H_a {hlab['ha_ad']:,.0f}"),
        (hlab["lb_ad"], lb_color, "dash", f"A/D L_b {hlab['lb_ad']:,.0f}"),
        (hlab["la_ad"], la_color, "dot",  f"A/D L_a {hlab['la_ad']:,.0f}"),
    ]:
        fig.add_shape(type="line", x0=pf["dt"].iloc[0], x1=pf["dt"].iloc[-1],
                      y0=val, y1=val, xref="x", yref="y2",
                      line=dict(color=color, dash=dash, width=1.0))
        fig.add_annotation(x=pf["dt"].iloc[0], y=val, xref="x", yref="y2",
                           text=ann, font=dict(color=color, size=9),
                           xanchor="right", showarrow=False)

    # 불일치 연결선
    if hlab["bear_div"]:
        fig.add_shape(type="line",
            x0=hlab["ha_dt"], y0=hlab["ha_ad"], x1=hlab["hb_dt"], y1=hlab["hb_ad"],
            xref="x", yref="y2",
            line=dict(color="rgba(255,80,80,0.9)", width=2, dash="dash"))
        mid_dt = hlab["ha_dt"] + (hlab["hb_dt"] - hlab["ha_dt"]) / 2
        fig.add_annotation(x=mid_dt, y=(hlab["ha_ad"]+hlab["hb_ad"])/2,
                           xref="x", yref="y2",
                           text=f"⚠ {hlab['bear_div_pct']:.1f}%",
                           font=dict(color="#ff5050", size=12), showarrow=False)
    if hlab["bull_div"]:
        fig.add_shape(type="line",
            x0=hlab["la_dt"], y0=hlab["la_ad"], x1=hlab["lb_dt"], y1=hlab["lb_ad"],
            xref="x", yref="y2",
            line=dict(color="rgba(38,210,160,0.9)", width=2, dash="dash"))
        mid_dt = hlab["la_dt"] + (hlab["lb_dt"] - hlab["la_dt"]) / 2
        fig.add_annotation(x=mid_dt, y=(hlab["la_ad"]+hlab["lb_ad"])/2,
                           xref="x", yref="y2",
                           text=f"✓ {hlab['bull_div_pct']:.1f}%",
                           font=dict(color="#26d2a0", size=12), showarrow=False)

    fig.update_layout(
        template="plotly_dark", height=660,
        title=dict(text=f"{MARKET_CFG[market]['label']} — {div_text}",
                   font=dict(size=14, color=div_color)),
        hovermode="x",
        hoverlabel=dict(bgcolor="#1a1a2e", font_color="#ffffff", font_size=12, bordercolor="#888888", namelength=-1),
        legend=dict(orientation="h", y=1.01, x=0),
        margin=dict(l=10, r=90, t=45, b=10),
        xaxis=dict(
            rangeslider=dict(visible=False),
            showspikes=True, spikemode="across", spikesnap="cursor",
            spikethickness=1, spikecolor="rgba(200,200,200,0.7)", spikedash="solid",
            tickformat="%Y/%m/%d", dtick=7*24*60*60*1000,
            tickangle=-45, tickfont=dict(size=8),
            domain=[0, 1],
        ),
        yaxis=dict(
            title="Index", domain=[0.52, 1.0],
            showspikes=True, spikethickness=1, spikecolor="rgba(200,200,200,0.4)",
        ),
        yaxis2=dict(
            title="A/D Line", domain=[0.0, 0.48],
            showspikes=True, spikethickness=1, spikecolor="rgba(200,200,200,0.4)",
        ),
    )
    return fig

# ──────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="미장 브레드스 대시보드", page_icon="🇺🇸", layout="wide")
    st.title("🇺🇸 미국 시장 브레드스 대시보드")
    st.caption("NYSE / NASDAQ — 스탠 와인스태인 브레드스 분석")

    with st.sidebar:
        st.header("⚙️ Settings")
        market   = st.selectbox("마켓", ["NYSE", "NASDAQ"])
        today    = datetime.today()
        start_dt = st.date_input("시작일", value=today - timedelta(days=730))
        end_dt   = st.date_input("종료일", value=today)
        fetch_btn = st.button("🔄 데이터 불러오기", type="primary", use_container_width=True)
        st.divider()
        st.subheader("Analysis Parameters")
        lookback     = st.slider("Lookback (일)",      20, 252, 126)
        chart_months = st.slider("Chart Display Period (months)", 1,  24,   6)
        high_bars    = st.slider("High 탐색 H_b (일)",  10, 500, 60)
        low_bars     = st.slider("저점 탐색 L_b (일)",  10, 500, 130)
        with st.expander("Threshold Settings"):
            price_thr  = st.number_input("Price Near-High Threshold (%)", value=2.0, step=0.1)
            ad_thr     = st.number_input("A/D Near-High Threshold (%)",  value=3.0, step=0.1)
            gap_warn   = st.number_input("Warning Divergence Threshold (%)",       value=1.5, step=0.1)
            gap_danger = st.number_input("Severe Divergence Threshold (%)",       value=2.5, step=0.1)

    if not fetch_btn and "us_df_merged" not in st.session_state:
        st.info("👈 사이드바에서 마켓 선택 후 **데이터 불러오기** 버튼을 눌러주세요.")
        return

    if fetch_btn:
        if not YF_OK:
            st.error("yfinance 미설치"); return
        start_str = start_dt.strftime("%Y%m%d")
        end_str   = end_dt.strftime("%Y%m%d")
        try:
            with st.spinner("구성종목 Collecting... (30~60초 소요)"):
                breadth_df = fetch_breadth(market, start_str, end_str)
            with st.spinner("Index OHLC Collecting..."):
                index_df = fetch_index(market, start_str, end_str)
            df = breadth_df.merge(
                index_df[["date", "open", "high", "low", "close"]], on="date", how="inner"
            ).sort_values("date").reset_index(drop=True)
            st.success(f"✅ {market} 완료 — {len(df)}일치 / 최신: {df['date'].iloc[-1]}")
            st.session_state["us_df_merged"] = df
            st.session_state["us_df_market"] = market
            with st.spinner("NH-NL 계산 중…"):
                nhnl_df = fetch_nhnl(market, start_str, end_str)
            st.session_state["us_nhnl"] = nhnl_df
        except Exception as e:
            st.error(f"Data collection failed: {e}"); return

    if st.session_state.get("us_df_market") != market:
        st.session_state.pop("us_df_merged", None)
        st.info("마켓이 변경됐습니다. 데이터 불러오기를 다시 눌러주세요."); return

    df      = st.session_state["us_df_merged"]
    nhnl_df = st.session_state.get("us_nhnl")
    if len(df) < lookback:
        st.warning(f"Not enough data: {len(df)}행"); return

    sig  = compute_signals(df, lookback, price_thr, ad_thr, gap_warn, gap_danger)
    hlab = compute_hlab(df, high_bars=high_bars, low_bars=low_bars)
    last = df.iloc[-1]
    tab1, tab2, tab3 = st.tabs(["📈 A/D Line", "⚡ MI 탄력Index", "🏔 NH-NL"])

    with tab1:
        gc = "#00897b" if sig["gap"] >= 0 else "#c62828"
        ga = "▲" if sig["gap"] >= 0 else "▼"
        st.markdown(
            f'<div style="text-align:center;padding:6px 0 2px 0">'
            f'<span style="font-size:0.85em;color:#aaa">Divergence (A/D − Price)</span><br>'
            f'<span style="font-size:2.6em;font-weight:900;color:{gc}">{ga} {sig["gap"]:+.2f}%</span>'
            f'<span style="font-size:0.8em;color:#aaa;margin-left:8px">Reference: {sig["peak_label"]}</span></div>',
            unsafe_allow_html=True)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Latest Date", pd.to_datetime(str(last["date"]), format="%Y%m%d").strftime("%Y-%m-%d"))
        c2.metric(f"{market} Close", f"{float(last['close']):,.2f}")
        c3.metric("Daily A/D Diff",   f"{float(last['ad_diff']):+,.0f}")
        c4.metric("Price vs High", f"{sig['price_off']:.2f}%")
        c5.metric("A/D vs High",  f"{sig['ad_off']:.2f}%")
        st.markdown(
            f'<div style="background:{sig["color"]};padding:12px 18px;border-radius:8px;margin:8px 0">'
            f'<b style="font-size:1.2em;color:white">{sig["verdict"]}</b>'
            f'&nbsp;&nbsp;<span style="color:#ffffffcc">{sig["note"]}</span></div>',
            unsafe_allow_html=True)
        try:
            st.plotly_chart(make_plotly_chart(df, market, sig, chart_months, hlab), use_container_width=True)
        except Exception as e:
            st.error(f"차트 오류: {e}")
        with st.expander("📋 원시 데이터"):
            show = df.copy()
            show["date"] = pd.to_datetime(show["date"].astype(str), format="%Y%m%d").dt.strftime("%Y-%m-%d")
            st.dataframe(show[["date", "ad_diff", "ad_line", "close"]].sort_values("date", ascending=False).reset_index(drop=True), use_container_width=True)

    with tab2:
        st.subheader("⚡ MI 탄력Index (Momentum Index)")
        st.caption("스탠 와인스태인: 등락종목수 차이(AD)의 200일 롤링 평균. 0선 위=강세.")
        mi_w  = st.slider("MA 기간", 50, 300, 200, step=10, key="us_mi")
        end2  = pd.to_datetime(df["date"].astype(str), format="%Y%m%d").max()
        mask2 = pd.to_datetime(df["date"].astype(str), format="%Y%m%d") >= end2 - pd.DateOffset(months=chart_months)
        pf2   = df[mask2].copy(); pf2["dt"] = pd.to_datetime(pf2["date"].astype(str), format="%Y%m%d")
        ads   = pd.Series(df["ad_diff"].values.astype(float))
        mif   = ads.rolling(mi_w).mean()
        mip   = mif.iloc[mask2.values].reset_index(drop=True)
        lm    = mif.iloc[-1]; pm = mif.iloc[-2] if len(mif) >= 2 else lm
        if pd.isna(lm):              mv, mc = "⚪ Not enough data", "#757575"
        elif lm > 0 and lm > pm:    mv, mc = "🟢 강세 상승", "#2e7d32"
        elif lm > 0:                 mv, mc = "🟡 강세 둔화", "#f9a825"
        elif lm < 0 and lm < pm:    mv, mc = "🔴 약세 하락", "#c62828"
        else:                        mv, mc = "🟠 약세 회복 중", "#ef6c00"
        m1, m2, m3 = st.columns(3)
        m1.metric(f"MI ({mi_w}일)", f"{lm:+.1f}" if not pd.isna(lm) else "N/A")
        m2.metric("전일 대비", f"{lm - pm:+.1f}" if not pd.isna(lm) else "N/A")
        m3.metric("판정", mv)
        fig_mi = go.Figure()
        fig_mi.add_trace(go.Bar(x=pf2["dt"], y=mip,
            marker_color=[("#26a69a" if v >= 0 else "#ef5350") for v in mip.fillna(0)],
            name=f"MI ({mi_w}일)", opacity=0.85))
        fig_mi.add_hline(y=0, line_color="gray", line_dash="dot", annotation_text="기준선(0)")
        fig_mi.update_layout(title=f"{market} MI 탄력Index", template="plotly_dark", height=420, yaxis_title="MI",
            hovermode="x",
            hoverlabel=dict(bgcolor="#1a1a2e", font_color="#ffffff", font_size=12, bordercolor="#888888", namelength=-1))
        st.plotly_chart(fig_mi, use_container_width=True)

    with tab3:
        st.subheader("🏔 NH-NL (52주 신고가 - 신저가 종목 수)")
        st.caption("NYSE/NASDAQ 전체 거래소 기준 52주 신고가/신저가 종목 수. 트레이딩뷰 $NHNL/$NANAHNL 동일 소스.")
        if nhnl_df is None:
            err = st.session_state.pop("nhnl_error", None)
            if err:
                st.warning(
                    f"NH-NL Data collection failed: {err}\n\n"
                    "Yahoo Finance가 해당 심볼을 지원하지 않을 수 있습니다. "
                    "**데이터 불러오기** 버튼을 다시 눌러보세요."
                )
            else:
                st.info("데이터 불러오기 버튼을 눌러주세요.")
        elif nhnl_df.empty:
            st.warning("NH-NL 데이터가 비어 있습니다.")
        if nhnl_df is not None and not nhnl_df.empty:
            from plotly.subplots import make_subplots as _msp
            end3  = pd.to_datetime(nhnl_df["date"].astype(str), format="%Y%m%d").max()
            mask3 = pd.to_datetime(nhnl_df["date"].astype(str), format="%Y%m%d") >= end3 - pd.DateOffset(months=chart_months)
            pf3   = nhnl_df[mask3].copy()
            pf3["dt"] = pd.to_datetime(pf3["date"].astype(str), format="%Y%m%d")

            # NH-NL 10주 MA
            ns_all  = pd.Series(nhnl_df["nhnl"].values.astype(float))
            nma_all = ns_all.rolling(10).mean()
            nma     = nma_all.iloc[mask3.values].reset_index(drop=True)

            # 판정: 10주 MA 기울기
            lma = nma_all.iloc[-1]; pma = nma_all.iloc[-2] if len(nma_all) >= 2 else lma
            ln  = int(ns_all.iloc[-1])
            lh  = int(nhnl_df["new_highs"].iloc[-1])
            ll  = int(nhnl_df["new_lows"].iloc[-1])
            if pd.isna(lma):        nv, nc = "⚪ Not enough data",   "#757575"
            elif lma > 0 and lma > pma: nv, nc = "🟢 강세 상승", "#2e7d32"
            elif lma > 0:           nv, nc = "🟡 강세 둔화",     "#f9a825"
            elif lma < 0 and lma < pma: nv, nc = "🔴 약세 하락", "#c62828"
            else:                   nv, nc = "🟠 약세 회복 중",   "#ef6c00"

            n1, n2, n3, n4 = st.columns(4)
            n1.metric("신고가 종목", f"{lh}")
            n2.metric("신저가 종목", f"{ll}")
            n3.metric("NH-NL", f"{ln:+}")
            n4.metric("판정", nv)

            # Index 같은 기간
            idx_mask = pd.to_datetime(df["date"].astype(str), format="%Y%m%d") >= end3 - pd.DateOffset(months=chart_months)
            pf_idx = df[idx_mask].copy()
            pf_idx["dt"] = pd.to_datetime(pf_idx["date"].astype(str), format="%Y%m%d")

            # 2패널 — 위: Index 곡선 / 아래: NH-NL 곡선 + MA
            fig_n = _msp(rows=2, cols=1, shared_xaxes=True,
                         row_heights=[0.5, 0.5], vertical_spacing=0.02)

            # 위 패널: Index 곡선 단독
            fig_n.add_trace(go.Scatter(
                x=pf_idx["dt"], y=pf_idx["close"],
                line=dict(color="rgba(200,200,200,0.9)", width=1.8),
                name=f"{market} Index",
            ), row=1, col=1)

            # 아래 패널: NH-NL 곡선
            fig_n.add_trace(go.Scatter(
                x=pf3["dt"], y=pf3["nhnl"].astype(float),
                line=dict(color="#26a69a", width=1.8),
                name="NH-NL",
            ), row=2, col=1)

            # 아래 패널: 10주 MA 곡선
            fig_n.add_trace(go.Scatter(
                x=pf3["dt"], y=nma,
                line=dict(color="orange", width=2),
                name="10주 MA",
            ), row=2, col=1)

            # 0선 (아래 패널)
            fig_n.add_hline(y=0, line_color="rgba(150,150,150,0.5)", line_dash="dot", row=2, col=1)

            fig_n.update_layout(
                template="plotly_dark", height=560,
                title=dict(text=f"{market} NH-NL — {nv}", font_size=13,
                           font=dict(color=nc)),
                hovermode="x",
                hoverlabel=dict(bgcolor="#1a1a2e", font_color="#ffffff",
                                font_size=12, bordercolor="#888888", namelength=-1),
                margin=dict(l=10, r=60, t=45, b=10),
                xaxis_rangeslider_visible=False,
                legend=dict(orientation="h", y=1.01),
                yaxis =dict(title="Index"),
                yaxis2=dict(title="NH-NL", zeroline=True,
                            zerolinecolor="rgba(150,150,150,0.4)"),
            )
            # 세로선: 두 패널 동시 관통
            fig_n.update_traces(xaxis="x")
            fig_n.update_xaxes(
                showspikes=True, spikemode="across", spikesnap="cursor",
                spikethickness=1, spikecolor="rgba(200,200,200,0.7)", spikedash="solid",
                tickformat="%Y/%m/%d", dtick=7*24*60*60*1000,
                tickangle=-45, tickfont=dict(size=8),
            )
            fig_n.update_yaxes(showspikes=True, spikethickness=1,
                               spikecolor="rgba(200,200,200,0.4)")
            st.plotly_chart(fig_n, use_container_width=True)
        else:
            st.warning("NH-NL 데이터를 가져오지 못했습니다.")


if __name__ == "__main__":
    main()

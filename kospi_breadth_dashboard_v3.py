#!/usr/bin/env python3
from __future__ import annotations
# Korea Market Breadth Dashboard (Streamlit)
# Run: streamlit run kospi_breadth_dashboard_v3.py
# GitHub raw CSV URL, loaded by Streamlit Cloud after data files are pushed
GITHUB_RAW = "https://raw.githubusercontent.com/onekindalpha/breadth-analysis-dashboard/main/data"
GITHUB_BREADTH = {
    "KOSPI":  f"{GITHUB_RAW}/kospi_breadth.csv",
    "KOSDAQ": f"{GITHUB_RAW}/kosdaq_breadth.csv",
}
GITHUB_INDEX = {
    "KOSPI":  f"{GITHUB_RAW}/kospi_index.csv",
    "KOSDAQ": f"{GITHUB_RAW}/kosdaq_index.csv",
}
GITHUB_NHNL = {
    "KOSPI":  f"{GITHUB_RAW}/kospi_nhnl.csv",
    "KOSDAQ": f"{GITHUB_RAW}/kosdaq_nhnl.csv",
}
GITHUB_NHNL_DAILY = {
    "KOSPI":  f"{GITHUB_RAW}/kospi_nhnl_daily.csv",
    "KOSDAQ": f"{GITHUB_RAW}/kosdaq_nhnl_daily.csv",
}

import hashlib
import io
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import platform
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

# ── Font settings ──
def _setup_korean_font():
    import matplotlib.font_manager as fm
    import subprocess
    sys_name = platform.system()
    if sys_name == "Darwin":
        plt.rcParams["font.family"] = "AppleGothic"
    elif sys_name == "Windows":
        plt.rcParams["font.family"] = "Malgun Gothic"
    else:
        # Linux / Streamlit Cloud: try NanumGothic when needed
        nanum = [f.name for f in fm.fontManager.ttflist if "Nanum" in f.name]
        if nanum:
            plt.rcParams["font.family"] = nanum[0]
        else:
            try:
                subprocess.run(
                    ["apt-get", "install", "-y", "-q", "fonts-nanum"],
                    check=True, capture_output=True
                )
                fm._load_fontmanager(try_read_cache=False)
                nanum2 = [f.name for f in fm.fontManager.ttflist if "Nanum" in f.name]
                if nanum2:
                    plt.rcParams["font.family"] = nanum2[0]
            except Exception:
                # Fallback to English labels if font setup fails
                pass
    plt.rcParams["axes.unicode_minus"] = False

_setup_korean_font()
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import traceback
import streamlit as st

try:
    from mplfinance.original_flavor import candlestick_ohlc
    MPL_OK = True
except ImportError:
    MPL_OK = False

try:
    import FinanceDataReader as fdr
    FDR_OK = True
except ImportError:
    FDR_OK = False

# ──────────────────────────────────────────────────────────────
# Settings
# ──────────────────────────────────────────────────────────────
API_BASE = "https://data-dbg.krx.co.kr/svc/apis/sto"
KRX_ENDPOINTS  = {"KOSPI": "/stk_bydd_trd", "KOSDAQ": "/ksq_bydd_trd"}
FDR_SYMBOLS    = {"KOSPI": "KS11",          "KOSDAQ": "KQ11"}
CACHE_DIR      = Path("./breadth_cache")

STATUS_MAP = {
    "BULLISH_CONFIRMATION":         ("✅ Bullish Confirmation",           "Price and A/D line are both near the recent high",                   "#2e7d32"),
    "BULLISH_DIVERGENCE":           ("🔴⚠️ Severe A/D Divergence",   "Price is near the high while A/D line lags significantly",                  "#c62828"),
    "BULLISH_DIVERGENCE_CANDIDATE": ("🟠⚠️ Early A/D Warning",       "Price is recovering faster than the A/D line",                    "#ef6c00"),
    "RECOVERY_IN_PROGRESS":         ("🟡Recovery in Progress",         "Price is retesting the high without breadth confirmation",                "#f9a825"),
    "DOWNSIDE_DIVERGENCE_CANDIDATE":("🟢Downside Divergence",      "Price is near lows while A/D line does not confirm lows",                 "#00838f"),
    "NORMAL_WEAKNESS":              ("⚫ Broad Weakness",           "Price and A/D line are both near recent lows",                          "#455a64"),
    "NEUTRAL":                      ("⬜ Neutral",                 "No clear signal",                                   "#757575"),
}

# ──────────────────────────────────────────────────────────────
# NH-NL cache path
# ──────────────────────────────────────────────────────────────
NHNL_CACHE_DIR = Path("./nhnl_cache_v2")

def _nhnl_cache_path(market: str, date_str: str) -> Path:
    NHNL_CACHE_DIR.mkdir(exist_ok=True)
    return NHNL_CACHE_DIR / f"nhnl_v2_{market}_{date_str}.csv"

def load_nhnl_cache(market: str, date_str: str) -> pd.DataFrame | None:
    p = _nhnl_cache_path(market, date_str)
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p, dtype={"date": str})
    except Exception:
        return None
    # Internal implementation note.
    if df.empty or len(df) < 20:
        return None
    return df

def save_nhnl_cache(df: pd.DataFrame, market: str, date_str: str):
    p = _nhnl_cache_path(market, date_str)
    df.to_csv(p, index=False)


def _is_common_stock_krx(df: pd.DataFrame) -> pd.Series:
    """
    Common-share-centered breadth calculation.
    Exclude preferred shares using name/code patterns where possible.
    Exclude ETFs, ETNs, ELWs, SPACs, REITs, funds, inverse, and leveraged products.
    """
    if df.empty:
        return pd.Series(dtype=bool)

    name_col = next((c for c in ["ISU_ABBRV", "ISU_NM", "Name", "name"] if c in df.columns), None)
    code_col = next((c for c in ["ISU_SRT_CD", "Code", "Symbol", "code"] if c in df.columns), None)

    name = df[name_col].astype(str).fillna("") if name_col else pd.Series([""] * len(df), index=df.index)
    code = df[code_col].astype(str).fillna("") if code_col else pd.Series([""] * len(df), index=df.index)

    exclude_pat = (
        r"(?:\uC6B0$|\uC6B0B$|\uC6B0C$|[0-9]\uC6B0$|\uC2A4\uD329|\uB9AC\uCE20|REIT|ETF|ETN|ELW|KODEX|TIGER|KOSEF|KBSTAR|ARIRANG|HANARO|"
        r"SOL|ACE|TIMEFOLIO|TREX|SMART|FOCUS|\uB9C8\uC774\uD2F0|TRUE|QV|RISE|\uB808\uBC84\uB9AC\uC9C0|\uC778\uBC84\uC2A4|\uC120\uBB3C|\uCC44\uAD8C|"
        r"\uD380\uB4DC|\uC561\uD2F0\uBE0C|TDF|TRF|BLN|\uD68C\uC0AC\uCC44|\uAD6D\uACE0\uCC44)"
    )
    bad_name = name.str.contains(exclude_pat, case=False, regex=True, na=False)

    # Internal implementation note.
    bad_code = code.str.endswith(("K", "L", "M", "N"))  # defensive code filter
    return ~(bad_name | bad_code)


def compute_nhnl_pykrx(market: str, end_date: str, prog=None, auth_key: str = "", chart_start_date: str | None = None) -> pd.DataFrame:
    """
    NH-NL implementation:
    Common-share-centered breadth calculation.
    - based on close price
    - number of 52-week high/low breakouts
    Common-share-centered breadth calculation.
    Common-share-centered breadth calculation.
    """
    if not auth_key or not str(auth_key).strip():
        raise RuntimeError("NH-NL currently requires a KRX API AUTH_KEY. Enter the key in the sidebar.")

    end_dt = pd.to_datetime(end_date, format="%Y%m%d")
    if chart_start_date:
        chart_start_dt = pd.to_datetime(chart_start_date, format="%Y%m%d")
        start_dt = chart_start_dt - timedelta(days=420)
    else:
        start_dt = end_dt - timedelta(days=800)
    dates = pd.bdate_range(start_dt, end_dt)
    session = requests.Session()

    daily_frames = []
    total = len(dates)
    for i, dt in enumerate(dates, 1):
        bas_dd = dt.strftime("%Y%m%d")
        try:
            raw = _fetch_daily(session, auth_key, bas_dd, market)
        except Exception:
            continue
        if raw is None or raw.empty:
            continue

        code_col = next((c for c in ["ISU_SRT_CD", "ISU_CD", "Code", "Symbol"] if c in raw.columns), None)
        name_col = next((c for c in ["ISU_ABBRV", "ISU_NM", "Name"] if c in raw.columns), None)
        close_col = next((c for c in ["TDD_CLSPRC", "Close", "close"] if c in raw.columns), None)

        if code_col is None or close_col is None:
            continue

        df = raw.copy()
        df["date"] = bas_dd
        df["code"] = df[code_col].astype(str).str.extract(r"(\d+)")[0].str.zfill(6)
        df["name"] = df[name_col].astype(str) if name_col else ""
        df["close"] = pd.to_numeric(df[close_col], errors="coerce")
        df = df.dropna(subset=["code", "close"])
        df = df[_is_common_stock_krx(df)].copy()
        if not df.empty:
            daily_frames.append(df[["date", "code", "name", "close"]])

        if prog:
            prog.progress(i / total, text=f"Collecting KRX data for NH-NL... {bas_dd} ({i}/{total})")

    if not daily_frames:
        raise RuntimeError("No KRX daily component data available for NH-NL calculation.")

    panel = pd.concat(daily_frames, ignore_index=True)
    panel["dt"] = pd.to_datetime(panel["date"], format="%Y%m%d")
    panel = panel.sort_values(["code", "dt"]).drop_duplicates(["code", "dt"], keep="last")

    # Exclude components with insufficient trading history
    valid_counts = panel.groupby("code")["dt"].size()
    valid_codes = valid_counts[valid_counts >= 260].index
    panel = panel[panel["code"].isin(valid_codes)].copy()
    if panel.empty:
        raise RuntimeError("No components have enough history for 52-week high/low detection.")

    def _mark_breakouts(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("dt").copy()
        prev_high = g["close"].rolling(252, min_periods=252).max().shift(1)
        prev_low = g["close"].rolling(252, min_periods=252).min().shift(1)
        g["new_high"] = ((g["close"] > prev_high) & prev_high.notna()).astype(int)
        g["new_low"] = ((g["close"] < prev_low) & prev_low.notna()).astype(int)
        return g[["dt", "new_high", "new_low"]]

    marked = panel.groupby("code", group_keys=False).apply(_mark_breakouts).reset_index(drop=True)
    daily = marked.groupby("dt", as_index=False)[["new_high", "new_low"]].sum()
    daily["nhnl"] = daily["new_high"] - daily["new_low"]

    weekly = daily.set_index("dt").resample("W-FRI").sum().reset_index()
    weekly = weekly.rename(columns={"new_high": "new_highs", "new_low": "new_lows"})
    weekly["date"] = weekly["dt"].dt.strftime("%Y%m%d")
    weekly = weekly[["date", "dt", "new_highs", "new_lows", "nhnl"]]
    weekly = weekly.sort_values("dt").reset_index(drop=True)
    cutoff_dt = start_dt + pd.Timedelta(days=365)
    weekly = weekly[weekly["dt"] >= cutoff_dt].reset_index(drop=True)
    if chart_start_date:
        chart_start_dt = pd.to_datetime(chart_start_date, format="%Y%m%d")
        weekly = weekly[weekly["dt"] >= chart_start_dt].reset_index(drop=True)

    # Remove early warm-up period
    cutoff = pd.to_datetime(start_dt) + pd.Timedelta(days=365)
    weekly = weekly[weekly["dt"] >= cutoff].reset_index(drop=True)
    return weekly


def compute_nhnl_fdr(market: str, end_date: str, prog=None, auth_key: str = "") -> pd.DataFrame:
    return compute_nhnl_pykrx(market=market, end_date=end_date, prog=prog, auth_key=auth_key)
# ──────────────────────────────────────────────────────────────
# File cache utilities
# ──────────────────────────────────────────────────────────────
def _cache_path(market: str, start: str, end: str, base: float) -> Path:
    CACHE_DIR.mkdir(exist_ok=True)
    key = f"{market}_{start}_{end}_{int(base)}"
    return CACHE_DIR / f"{key}.csv"

def load_cache(market: str, start: str, end: str, base: float) -> pd.DataFrame | None:
    p = _cache_path(market, start, end, base)
    if p.exists():
        df = pd.read_csv(p, dtype={"date": str})
        return df
    return None

def save_cache(df: pd.DataFrame, market: str, start: str, end: str, base: float) -> None:
    p = _cache_path(market, start, end, base)
    df.to_csv(p, index=False)

def list_caches() -> list[Path]:
    CACHE_DIR.mkdir(exist_ok=True)
    return sorted(CACHE_DIR.glob("*.csv"))

# ──────────────────────────────────────────────────────────────
# KRX API
# ──────────────────────────────────────────────────────────────
def _krx_post(session, auth_key, endpoint, payload):
    url = API_BASE + endpoint
    headers = {"AUTH_KEY": auth_key.strip(), "Content-Type": "application/json",
                "Accept": "application/json", "User-Agent": "Mozilla/5.0"}
    r = session.post(url, headers=headers, json=payload, timeout=15)
    if r.status_code != 200:
        raise RuntimeError(f"KRX {r.status_code}: {r.text[:200]}")
    data = r.json()
    if isinstance(data, dict) and data.get("respCode") not in (None, "000", 0, "0"):
        raise RuntimeError(f"KRX respCode {data.get('respCode')}: {data.get('respMsg')}")
    return data

def _fetch_daily(session, auth_key, bas_dd, market):
    data = _krx_post(session, auth_key, KRX_ENDPOINTS[market], {"basDd": bas_dd})
    rows = data.get("OutBlock_1", [])
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for c in ["TDD_CLSPRC", "CMPPREVDD_PRC", "FLUC_RT",
              "TDD_OPNPRC", "TDD_HGPRC", "TDD_LWPRC"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "", regex=False), errors="coerce")
    return df.rename(columns={"BAS_DD": "Date", "CMPPREVDD_PRC": "PrevDiff", "FLUC_RT": "FlucRate"})

def _classify_breadth(df):
    if df.empty:
        return 0, 0, 0
    col = "PrevDiff" if "PrevDiff" in df.columns else "FlucRate"
    v = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return int((v > 0).sum()), int((v < 0).sum()), int((v == 0).sum())

def build_breadth(auth_key, start, end, market, base_value=50000.0):
    dates = pd.bdate_range(pd.to_datetime(start), pd.to_datetime(end))
    rows, ad_line = [], base_value
    session = requests.Session()
    prog = st.progress(0, text="Collecting KRX breadth data...")
    for i, dt in enumerate(dates, 1):
        bas_dd = dt.strftime("%Y%m%d")
        try:
            df = _fetch_daily(session, auth_key, bas_dd, market)
            if not df.empty:
                adv, decl, unch = _classify_breadth(df)
                ad_line += adv - decl
                rows.append({"date": bas_dd, "advances": adv, "declines": decl,
                             "unchanged": unch, "ad_diff": adv - decl, "ad_line": ad_line})
        except Exception as e:
            st.warning(f"{bas_dd} skipped: {e}")
        prog.progress(i / len(dates), text=f"Collecting... {bas_dd} ({i}/{len(dates)})")
    prog.empty()
    if not rows:
        raise RuntimeError("No data collected")
    out = pd.DataFrame(rows)
    br = (out["advances"] / (out["advances"] + out["declines"]).replace(0, pd.NA)).astype(float)
    out["breadth_thrust_ema10"] = br.ewm(span=10, adjust=False).mean()
    return out

# ──────────────────────────────────────────────────────────────
# Load GitHub raw CSV
# ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=300)
def load_from_github(market: str) -> pd.DataFrame:
    """Load the pushed GitHub CSV and merge breadth/index data"""
    import requests as _req
    b_url = GITHUB_BREADTH[market]
    i_url = GITHUB_INDEX[market]

    resp_b = _req.get(b_url, timeout=15)
    if resp_b.status_code != 200:
        raise RuntimeError(f"GitHub breadth CSV not found ({resp_b.status_code})\n{b_url}\n→ Run update_and_push.sh locally and push the generated data files.")
    breadth = pd.read_csv(io.StringIO(resp_b.text), dtype={"date": str})

    resp_i = _req.get(i_url, timeout=15)
    if resp_i.status_code == 200:
        idx = pd.read_csv(io.StringIO(resp_i.text), dtype={"date": str})
        avail_cols = [c for c in ["date","open","high","low","close"] if c in idx.columns]
        df = breadth.merge(idx[avail_cols], on="date", how="left")
    else:
        # NH-NL calculation and visualization logic.
        df = breadth.copy()

    df = df.sort_values("date").reset_index(drop=True)
    return df

@st.cache_data(show_spinner=False, ttl=300)
def load_nhnl_daily_from_github(market: str):
    """Load pushed daily NH-NL CSV from GitHub, or return None"""
    import requests as _req
    if market not in GITHUB_NHNL_DAILY:
        return None
    url = GITHUB_NHNL_DAILY[market]
    try:
        resp = _req.get(url, timeout=15)
        if resp.status_code != 200:
            return None
        df = pd.read_csv(io.StringIO(resp.text))
        df["date"] = df["date"].astype(int)
        df["dt"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
        return df.sort_values("date").reset_index(drop=True)
    except Exception:
        return None

@st.cache_data(show_spinner=False, ttl=300)
def load_nhnl_from_github(market: str):
    """Load pushed NH-NL CSV from GitHub, or return None"""
    import requests as _req
    if market not in GITHUB_NHNL:
        return None
    url = GITHUB_NHNL[market]
    try:
        resp = _req.get(url, timeout=15)
        if resp.status_code != 200:
            return None
        df = pd.read_csv(io.StringIO(resp.text), dtype={"date": str})
        if df.empty:
            return None
        return df
    except Exception:
        return None

# ──────────────────────────────────────────────────────────────
# Index OHLC
# ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=300)
def fetch_index_ohlc(market, start, end):
    if not FDR_OK:
        raise RuntimeError("finance-datareader is not installed")
    symbol = FDR_SYMBOLS[market]
    end_dt = datetime.strptime(end, "%Y%m%d") + timedelta(days=1)
    raw = fdr.DataReader(symbol, start, end_dt.strftime("%Y-%m-%d"))
    if raw.empty:
        raise RuntimeError(f"{symbol} No data")
    raw.columns = [str(c).strip().title() for c in raw.columns]
    df = raw.reset_index()
    df.columns = [str(c).strip().title() for c in df.columns]
    date_col = next((c for c in df.columns if c.lower() in ("date", "datetime")), None)
    if not date_col:
        raise RuntimeError(f"Date column not found: {list(df.columns)}")
    def _find(*candidates):
        for c in candidates:
            if c in df.columns:
                return c
        raise RuntimeError(f"{candidates} column not found: {list(df.columns)}")
    out = pd.DataFrame({
        "date":  pd.to_datetime(df[date_col]).dt.strftime("%Y%m%d"),
        "open":  pd.to_numeric(df[_find("Open")],  errors="coerce"),
        "high":  pd.to_numeric(df[_find("High")],  errors="coerce"),
        "low":   pd.to_numeric(df[_find("Low")],   errors="coerce"),
        "close": pd.to_numeric(df[_find("Close", "Adj Close")], errors="coerce"),
    })
    return out[out["date"] <= end].dropna().reset_index(drop=True)

# ──────────────────────────────────────────────────────────────
# Signal classification logic
# ──────────────────────────────────────────────────────────────
def classify(price_off_high, ad_off_high, gap,
             price_off_low, ad_off_low,
             price_thr=2.0, ad_thr=3.0, gap_warn=1.5, gap_danger=2.5):
    # Internal implementation note.
    # Internal implementation note.
    ph = price_off_high >= -price_thr
    ah = ad_off_high    >= -ad_thr
    pl = price_off_low  <= price_thr
    al = ad_off_low     <= ad_thr
    if ph and ah and gap >= -1.0:            return "BULLISH_CONFIRMATION"
    if ph and gap <= -gap_danger:            return "BULLISH_DIVERGENCE"
    if gap <= -gap_warn:                     return "BULLISH_DIVERGENCE_CANDIDATE"
    if gap < -1.0:                           return "RECOVERY_IN_PROGRESS"
    if pl and not al:                        return "DOWNSIDE_DIVERGENCE_CANDIDATE"
    if pl and al:                            return "NORMAL_WEAKNESS"
    return "NEUTRAL"

def compute_signals(df, lookback, price_thr, ad_thr, gap_warn, gap_danger):
    closes   = df["close"].values.astype(float)
    ad_lines = df["ad_line"].values.astype(float)
    window   = closes[-lookback:]
    peak_idx      = window.argmax()
    days_ago      = lookback - 1 - peak_idx
    price_high    = window[peak_idx]
    ad_at_peak    = ad_lines[-(days_ago + 1)]
    price_low     = closes[-lookback:].min()
    ad_low        = ad_lines[-lookback:].min()
    last_close    = closes[-1]
    last_ad       = ad_lines[-1]

    # Internal implementation note.
    price_off = (last_close - price_high)  / abs(price_high)  * 100 if price_high  else float("nan")
    ad_off    = (last_ad    - ad_at_peak)  / abs(ad_at_peak)  * 100 if ad_at_peak  else float("nan")
    gap       = ad_off - price_off
    price_off_low = (last_close - price_low) / abs(price_low) * 100 if price_low else float("nan")
    ad_off_low    = (last_ad    - ad_low)    / abs(ad_low)    * 100 if ad_low    else float("nan")

    peak_date  = str(df["date"].iloc[-(days_ago + 1)])
    peak_label = "Today" if days_ago == 0 else f"{days_ago} days ago ({peak_date})"
    status_key = classify(price_off, ad_off, gap, price_off_low, ad_off_low,
                          price_thr, ad_thr, gap_warn, gap_danger)
    verdict, note, color = STATUS_MAP[status_key]
    return dict(peak_label=peak_label, price_off=price_off, ad_off=ad_off, gap=gap,
                verdict=verdict, note=note, color=color,
                last_close=last_close, last_ad=last_ad,
                price_high=price_high, ad_at_peak=ad_at_peak)

# ──────────────────────────────────────────────────────────────
# Internal implementation note.
# ──────────────────────────────────────────────────────────────
def compute_hlab(df: pd.DataFrame, high_bars: int = 60, low_bars: int = 130) -> dict:
    """
    Same logic as the Pine Script v16 reference:
    H_b = high over the recent high_bars window
    H_a = high over the previous high_bars window
    L_b = low over the recent low_bars window
    L_a = low over the previous low_bars window
    """
    # Internal implementation note.
    df = df[df["close"].notna() & (df["close"].astype(float) > 0)].copy().reset_index(drop=True)
    closes  = df["close"].values.astype(float)
    ad_line = df["ad_line"].values.astype(float)
    dts     = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
    n = len(closes)

    def _safe_slice(arr, end_idx, length):
        start = max(0, end_idx - length)
        return arr[start:end_idx], start

    # Pine-style reference logic.
    # H_b = ta.highest(close, highBars)         → high over the recent high_bars window
    # Internal implementation note.
    # Internal implementation note.

    # Internal implementation note.
    hb_window, hb_start = _safe_slice(closes, n, high_bars)
    if len(hb_window) == 0:
        hb_window = closes; hb_start = 0
    hb_idx_local = int(np.argmax(hb_window))
    hb_idx = hb_start + hb_idx_local
    hb_val = closes[hb_idx]
    hb_dt  = dts.iloc[hb_idx]
    hb_ad  = ad_line[hb_idx]

    # Internal implementation note.
    # Pine-style reference logic.
    ha_end   = max(0, n - high_bars)          # previous high_bars boundary (exclusive end)
    ha_window, ha_start = _safe_slice(closes, ha_end, high_bars)
    if len(ha_window) > 0:
        ha_idx_local = int(np.argmax(ha_window))
        ha_idx = ha_start + ha_idx_local
        ha_val = closes[ha_idx]
        ha_dt  = dts.iloc[ha_idx]
        ha_ad  = ad_line[ha_idx]
    else:
        ha_val, ha_dt, ha_ad, ha_idx = hb_val, hb_dt, hb_ad, hb_idx

    # Internal implementation note.
    lb_window, lb_start = _safe_slice(closes, n, low_bars)
    if len(lb_window) == 0:
        lb_window = closes; lb_start = 0
    lb_idx_local = int(np.argmin(lb_window))
    lb_idx = lb_start + lb_idx_local
    lb_val = closes[lb_idx]
    lb_dt  = dts.iloc[lb_idx]
    lb_ad  = ad_line[lb_idx]

    # Internal implementation note.
    # Pine-style reference logic.
    la_end   = max(0, n - low_bars)
    la_window, la_start = _safe_slice(closes, la_end, low_bars)
    if len(la_window) > 0:
        la_idx_local = int(np.argmin(la_window))
        la_idx = la_start + la_idx_local
        la_val = closes[la_idx]
        la_dt  = dts.iloc[la_idx]
        la_ad  = ad_line[la_idx]
    else:
        la_val, la_dt, la_ad, la_idx = lb_val, lb_dt, lb_ad, lb_idx

    # Internal implementation note.
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

# ──────────────────────────────────────────────────────────────
# Chart layout configuration.
# All traces share one x-axis → vertical hover line crosses all panels
# yaxis(upper candlestick panel) domain=[0.42,1.0], yaxis2(lower A/D panel) domain=[0.0,0.38]
# Internal implementation note.
# ──────────────────────────────────────────────────────────────
def make_plotly_chart(df: pd.DataFrame, market: str, sig: dict,
                      chart_months: int, hlab: dict) -> tuple[go.Figure, dict]:

    end_dt   = pd.to_datetime(df["date"].astype(str), format="%Y%m%d").max()
    start_dt = end_dt - pd.DateOffset(months=chart_months)
    mask     = pd.to_datetime(df["date"].astype(str), format="%Y%m%d") >= start_dt
    pf       = df[mask].copy().reset_index(drop=True)
    pf["dt"] = pd.to_datetime(pf["date"].astype(str), format="%Y%m%d")

    hb_color = "rgba(255,80,80,0.95)"  if hlab["bear_div"] else "rgba(160,160,160,0.8)"
    ha_color = "rgba(255,140,140,0.6)" if hlab["bear_div"] else "rgba(120,120,120,0.5)"
    lb_color = "rgba(38,210,160,0.95)" if hlab["bull_div"] else "rgba(160,160,160,0.8)"
    la_color = "rgba(38,210,160,0.6)"  if hlab["bull_div"] else "rgba(120,120,120,0.5)"

    price_low  = float(pf["low"].min())
    price_high = float(pf["high"].max())
    price_span = max(price_high - price_low, abs(price_high) * 0.02, 1.0)
    y1_range = [price_low - price_span * 0.08, price_high + price_span * 0.15]

    ad_vals = pf["ad_line"].astype(float)
    ad_min = float(ad_vals.min())
    ad_max = float(ad_vals.max())
    ad_span = max(ad_max - ad_min, max(abs(ad_max), 1.0) * 0.02, 1.0)
    y2_range = [ad_min - ad_span * 0.10, ad_max + ad_span * 0.10]

    # Internal implementation note.
    _warn_pct   = 0.5
    _danger_pct = 2.0
    if hlab["bear_div"]:
        _p = hlab["bear_div_pct"]
        if _p >= _danger_pct:
            div_text  = f"🔴 Negative Divergence (Risk) {_p:.1f}%"
            div_color = "#c62828"
        elif _p >= _warn_pct:
            div_text  = f"🟠 Negative Divergence (Caution) {_p:.1f}%"
            div_color = "#ef6c00"
        else:
            div_text  = f"🟡 Early Negative Divergence {_p:.1f}%"
            div_color = "#f9a825"
    elif hlab["bull_div"]:
        _p = hlab["bull_div_pct"]
        if _p >= _warn_pct:
            div_text  = f"🟢 Positive Divergence (Bottoming Signal) {_p:.1f}%"
            div_color = "#26d2a0"
        else:
            div_text  = f"🔵 Early Positive Divergence {_p:.1f}%"
            div_color = "#1e88e5"
    else:
        div_text, div_color = "No Divergence", "#aaaaaa"

    fig = go.Figure()

    # Chart layout configuration.
    fig.add_trace(go.Candlestick(
        x=pf["dt"], open=pf["open"], high=pf["high"], low=pf["low"], close=pf["close"],
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        name=market, showlegend=False,
        xaxis="x", yaxis="y1",
    ))

    # ── lower panel A/D Line (yaxis="y2", domain 0.0~0.49)
    fig.add_trace(go.Scatter(
        x=pf["dt"], y=ad_vals,
        line=dict(color="#1e88e5", width=2.0), name="A/D Line",
        hoverinfo="y",
        xaxis="x", yaxis="y2",
    ))

    # ── lower panel: map price to A/D scale (Pine: priceMapped)
    _close = pf["close"].astype(float)
    _price_min = float(_close.min())
    _price_max = float(_close.max())
    _ad_min_pf = float(ad_vals.min())
    _ad_max_pf = float(ad_vals.max())
    if _price_max != _price_min:
        _price_mapped = _ad_min_pf + (_close - _price_min) / (_price_max - _price_min) * (_ad_max_pf - _ad_min_pf)
    else:
        _price_mapped = ad_vals
    fig.add_trace(go.Scatter(
        x=pf["dt"], y=_price_mapped,
        line=dict(color="rgba(180,180,180,0.5)", width=1.0), name="Price (scaled)",
        hoverinfo="skip",
        xaxis="x", yaxis="y2",
    ))

    # Chart layout configuration.
    for val, color, dash, ann in [
        (hlab["hb_val"], hb_color, "dash", f"H_b {hlab['hb_val']:,.0f}"),
        (hlab["ha_val"], ha_color, "dot",  f"H_a {hlab['ha_val']:,.0f}"),
        (hlab["lb_val"], lb_color, "dash", f"L_b {hlab['lb_val']:,.0f}"),
        (hlab["la_val"], la_color, "dot",  f"L_a {hlab['la_val']:,.0f}"),
    ]:
        fig.add_shape(type="line", x0=pf["dt"].iloc[0], x1=pf["dt"].iloc[-1],
                      y0=val, y1=val, xref="x", yref="y1",
                      line=dict(color=color, dash=dash, width=1.2))
        fig.add_annotation(x=pf["dt"].iloc[0], y=val, xref="x", yref="y1",
                           text=ann, font=dict(color=color, size=10),
                           xanchor="right", showarrow=False)

    # lower panel horizontal line (yref="y2")
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

    # Pine-style reference logic.
    # Chart layout configuration.
    fig.add_annotation(
        x=hlab["ha_dt"], y=hlab["ha_ad"], xref="x", yref="y2",
        text=f"H_a<br>{hlab['ha_val']:,.0f}",
        showarrow=True, arrowhead=2, ax=0, ay=-25,
        font=dict(color=ha_color, size=10),
        bgcolor="rgba(60,60,60,0.8)", bordercolor=ha_color, borderwidth=1,
    )
    fig.add_annotation(
        x=hlab["hb_dt"], y=hlab["hb_ad"], xref="x", yref="y2",
        text=f"H_b<br>{hlab['hb_val']:,.0f}",
        showarrow=True, arrowhead=2, ax=0, ay=-25,
        font=dict(color=hb_color, size=10),
        bgcolor="rgba(60,60,60,0.8)", bordercolor=hb_color, borderwidth=1,
    )
    fig.add_annotation(
        x=hlab["la_dt"], y=hlab["la_ad"], xref="x", yref="y2",
        text=f"L_a<br>{hlab['la_val']:,.0f}",
        showarrow=True, arrowhead=2, ax=0, ay=25,
        font=dict(color=la_color, size=10),
        bgcolor="rgba(60,60,60,0.8)", bordercolor=la_color, borderwidth=1,
    )
    fig.add_annotation(
        x=hlab["lb_dt"], y=hlab["lb_ad"], xref="x", yref="y2",
        text=f"L_b<br>{hlab['lb_val']:,.0f}",
        showarrow=True, arrowhead=2, ax=0, ay=25,
        font=dict(color=lb_color, size=10),
        bgcolor="rgba(60,60,60,0.8)", bordercolor=lb_color, borderwidth=1,
    )
    # Internal implementation note.
    fig.add_shape(type="line",
        x0=hlab["ha_dt"], y0=hlab["ha_ad"], x1=hlab["hb_dt"], y1=hlab["hb_ad"],
        xref="x", yref="y2",
        line=dict(color=hb_color, width=2, dash="dash"))
    fig.add_shape(type="line",
        x0=hlab["la_dt"], y0=hlab["la_ad"], x1=hlab["lb_dt"], y1=hlab["lb_ad"],
        xref="x", yref="y2",
        line=dict(color=lb_color, width=2, dash="dash"))

    # Pine-style reference logic.
    _last_dt = pf["dt"].iloc[-1]
    fig.add_annotation(
        x=_last_dt, y=y2_range[0], xref="x", yref="y2",
        text=f"{div_text}",
        showarrow=False, xanchor="right",
        yanchor="bottom",
        font=dict(color="white", size=11),
        bgcolor=div_color, bordercolor=div_color, borderwidth=1,
    )

    # Data loading and preprocessing logic.
    ad_lookup = {
        dt.strftime("%Y-%m-%d"): float(v)
        for dt, v in zip(pf["dt"], ad_vals)
    }

    fig.update_layout(
        template="plotly_dark", height=660,
        paper_bgcolor="rgba(14,17,23,1)",
        plot_bgcolor="rgba(14,17,23,1)",
        title=dict(text=f"{market} — {div_text}", font=dict(size=14, color=div_color)),
        hovermode="x",
        hoverlabel=dict(bgcolor="rgba(0,0,0,0.9)", font_color="#ffffff", font_size=12, bordercolor="#555"),
        legend=dict(orientation="h", y=1.01, x=0,
                    bgcolor="rgba(0,0,0,0.85)",
                    bordercolor="#333", borderwidth=1,
                    font=dict(color="white", size=11)),
        margin=dict(l=10, r=90, t=55, b=35),
        xaxis=dict(
            domain=[0, 1],
            rangeslider=dict(visible=False),
            showspikes=True, spikemode="across", spikesnap="cursor",
            spikethickness=1, spikecolor="rgba(200,200,200,0.7)", spikedash="solid",
            tickformat="%Y/%m/%d", tickangle=-45, tickfont=dict(size=11),
            showline=True, mirror=True,
            rangebreaks=[
                dict(bounds=["sat", "mon"]),  # remove weekends
            ],
        ),
        yaxis=dict(
            title="Index", domain=[0.50, 1.0], range=y1_range,
            showspikes=True, spikemode="across", spikesnap="cursor",
            spikethickness=1, spikecolor="rgba(200,200,200,0.4)", spikedash="solid",
            showline=True, mirror=True,
        ),
        yaxis2=dict(
            title="A/D Line", domain=[0.0, 0.49], range=y2_range,
            showspikes=True, spikemode="across", spikesnap="data",
            spikethickness=2, spikecolor="rgba(255,255,255,1.0)", spikedash="solid",
            anchor="x",
        ),
    )
    return fig, ad_lookup

# ──────────────────────────────────────────────────────────────
# Main app
# ──────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="Korea Market Breadth Analysis Dashboard",
                       page_icon="📊", layout="wide")
    # Plotly rendering and interaction logic.
    st.markdown("""
    <style>
    .hoverlayer .hovertext rect { fill: rgba(0,0,0,0.88) !important; stroke: #555 !important; }
    .hoverlayer .hovertext text { fill: #ffffff !important; }
    .hoverlayer .hovertext path { stroke: #555 !important; }
    </style>
    """, unsafe_allow_html=True)
    st.title("Korea Market Breadth Analysis Dashboard")
    st.caption("A/D Line · Breadth Thrust · Price-Breadth Divergence")

    # Internal implementation note.
    with st.sidebar:
        st.header("⚙️ Settings")
        market = st.selectbox("Market", ["KOSPI", "KOSDAQ"])

        mode = st.radio("Data Source", ["☁️ GitHub (Fast)", "🔑 KRX API (Direct Collection)"],
                        index=0,
                        help="GitHub: use CSV files updated by GitHub Actions\nKRX API: direct collection with AUTH_KEY")

        if mode == "🔑 KRX API (Direct Collection)":
            auth_key = st.text_input("KRX AUTH_KEY",
                                     value=os.environ.get("KRX_AUTH_KEY", ""),
                                     type="password")
            c1, c2 = st.columns(2)
            today = datetime.today()
            start_dt = c1.date_input("Start Date", value=today - timedelta(days=730))
            end_dt   = c2.date_input("End Date", value=today)
            base_value = st.number_input("A/D Line Base Value", value=50000.0, step=1000.0)
        else:
            auth_key = ""
            today = datetime.today()
            start_dt = today - timedelta(days=730)
            end_dt   = today

        fetch_btn = st.button("🔄 Load Data", type="primary", width='stretch')
        if mode == "🔑 KRX API (Direct Collection)":
            st.caption("💡 Clear the cache below before reloading fresh data.")

        st.divider()
        st.subheader("Analysis Parameters")
        lookback     = st.slider("Lookback (days)",      20, 252, 126)
        chart_months = st.slider("Chart Display Period (months)", 1,  24,  6)
        high_bars    = st.slider("High Lookback H_b (days)", 10, 500, 30)
        low_bars     = st.slider("Low Lookback L_b (days)", 10, 500, 30)
        with st.expander("Threshold Settings"):
            price_thr  = st.number_input("Price Near-High Threshold (%)", value=2.0,  step=0.1)
            ad_thr     = st.number_input("A/D Near-High Threshold (%)",  value=3.0,  step=0.1)
            gap_warn   = st.number_input("Warning Divergence Threshold (%)",       value=1.5,  step=0.1)
            gap_danger = st.number_input("Severe Divergence Threshold (%)",       value=2.5,  step=0.1)

        st.divider()
        st.subheader("💾 Cached Files")
        caches = list_caches()
        if caches:
            for p in caches:
                col_a, col_b = st.columns([3, 1])
                col_a.caption(p.name)
                if col_b.button("🗑", key=str(p)):
                    p.unlink()
                    st.rerun()
        else:
            st.caption("No cached files")

    # ── Load Data ──────────────────────────────
    if not fetch_btn and "df_merged" not in st.session_state:
        st.info("👈 Select a market in the sidebar and click **Load Data**.")
        return

    if fetch_btn:
        st.session_state.pop(f"nhnl_{market}", None)
        if mode == "☁️ GitHub (Fast)":
            try:
                with st.spinner("Loading CSV files from GitHub..."):
                    load_from_github.clear()
                    load_nhnl_from_github.clear()
                    load_nhnl_daily_from_github.clear()
                    df = load_from_github(market)
                    nhnl_df = load_nhnl_from_github(market)
                    nhnl_daily_df = load_nhnl_daily_from_github(market)
                st.success(f"✅ GitHub load completed — {len(df)}rows / latest: {df['date'].iloc[-1]}")
                st.session_state[f"nhnl_{market}"] = nhnl_df if nhnl_df is not None and not nhnl_df.empty else None
                st.session_state[f"nhnl_daily_{market}"] = nhnl_daily_df if nhnl_daily_df is not None and not nhnl_daily_df.empty else None
                if nhnl_df is None or nhnl_df.empty:
                    st.info("In GitHub fast mode, the NH-NL tab is shown only when a saved NH-NL CSV exists.")
            except Exception as e:
                st.error(f"GitHub load failed: {e}")
                return
        else:
            if not auth_key:
                st.error("Please enter your KRX AUTH_KEY.")
                return
            start_str = start_dt.strftime("%Y%m%d")
            end_str   = end_dt.strftime("%Y%m%d")
            cached = load_cache(market, start_str, end_str, 50000.0)
            nhnl_cached = load_nhnl_cache(market, end_str)
            try:
                if cached is not None:
                    st.success(f"✅ Loaded from cache ({market} {start_str}~{end_str})")
                    df = cached
                else:
                    with st.spinner("Index OHLC Collecting..."):
                        index_df = fetch_index_ohlc(market, start_str, end_str)
                    breadth_df = build_breadth(auth_key, start_str, end_str, market, 50000.0)
                    df = breadth_df.merge(
                        index_df[["date","open","high","low","close"]],
                        on="date", how="inner"
                    ).sort_values("date").reset_index(drop=True)
                    save_cache(df, market, start_str, end_str, 50000.0)
                    st.success(f"✅ A/D data collection completed — {len(df)}rows")

                if nhnl_cached is not None and not nhnl_cached.empty:
                    nhnl_df = nhnl_cached
                    st.success(f"✅ Loaded NH-NL cache — {len(nhnl_df)}weeks")
                else:
                    prog3 = st.progress(0, text="Collecting KRX data for NH-NL...")
                    nhnl_df = compute_nhnl_pykrx(
                        market,
                        end_str,
                        prog=prog3,
                        auth_key=auth_key,
                        chart_start_date=start_str,
                    )
                    prog3.empty()
                    if nhnl_df is not None and not nhnl_df.empty:
                        save_nhnl_cache(nhnl_df, market, end_str)
                        st.success(f"✅ NH-NL calculation completed — {len(nhnl_df)}weeks")
                st.session_state[f"nhnl_{market}"] = nhnl_df if nhnl_df is not None and not nhnl_df.empty else None
            except Exception as e:
                st.error(f"Data collection failed: {type(e).__name__}: {e}")
                return

        st.session_state["df_merged"] = df
        st.session_state["df_market"] = market

    # Internal implementation note.
    if st.session_state.get("df_market") != market:
        st.session_state.pop("df_merged", None)
        st.info("Market changed. Click Load Data again.")
        return

    # Internal implementation note.
    df = st.session_state["df_merged"]

    if len(df) < lookback:
        st.warning(f"Not enough data: {len(df)} rows (lookback={lookback})")
        return

    sig  = compute_signals(df, lookback, price_thr, ad_thr, gap_warn, gap_danger)
    hlab = compute_hlab(df, high_bars=high_bars, low_bars=low_bars)
    last = df.iloc[-1]

    # ── tab setup ──
    # Internal implementation note.
    # reruns may reset the selected tab.
    TAB_LABELS = ["📈 A/D Line", "⚡ Momentum", "🏔 NH-NL"]
    if "active_tab" not in st.session_state:
        st.session_state["active_tab"] = TAB_LABELS[0]

    _default_idx = TAB_LABELS.index(st.session_state.get("active_tab", TAB_LABELS[0]))
    if hasattr(st, "segmented_control"):
        active_tab = st.segmented_control(
            "Analysis Tab",
            TAB_LABELS,
            selection_mode="single",
            default=TAB_LABELS[_default_idx],
            key="active_tab_selector",
        )
    else:
        active_tab = st.radio(
            "Analysis Tab",
            TAB_LABELS,
            index=_default_idx,
            horizontal=True,
            key="active_tab_selector",
        )
    st.session_state["active_tab"] = active_tab

    # ══════════════════════════════════════════════
    # TAB 1: A/D Line analysis
    # ══════════════════════════════════════════════
    if active_tab == "📈 A/D Line":
        # ── replicate Pine-style reference table ──────────────────────────────
        _ha_date = hlab["ha_dt"].strftime("%-m/%-d") if hasattr(hlab["ha_dt"], "strftime") else str(hlab["ha_dt"])
        _hb_date = hlab["hb_dt"].strftime("%-m/%-d") if hasattr(hlab["hb_dt"], "strftime") else str(hlab["hb_dt"])
        _la_date = hlab["la_dt"].strftime("%-m/%-d") if hasattr(hlab["la_dt"], "strftime") else str(hlab["la_dt"])
        _lb_date = hlab["lb_dt"].strftime("%-m/%-d") if hasattr(hlab["lb_dt"], "strftime") else str(hlab["lb_dt"])

        # Pine-style reference logic.
        _bear = hlab["bear_div"]
        _bull = hlab["bull_div"]
        _bdp  = hlab["bear_div_pct"]
        _bup  = hlab["bull_div_pct"]
        if _bear and _bdp >= 2.0:
            _status = "🔴 Negative Divergence (Risk)"
            _note   = f"New H_b high / A/D {_bdp:.2f}% lag"
            _scolor = "#c62828"
        elif _bear and _bdp >= 0.5:
            _status = "🟠 Negative Divergence (Caution)"
            _note   = f"New H_b high / A/D {_bdp:.2f}% lag"
            _scolor = "#ef6c00"
        elif _bear:
            _status = "🟡 Early Negative Divergence"
            _note   = "New H_b high / A/D slightly lagging"
            _scolor = "#f9a825"
        elif _bull and _bup >= 0.5:
            _status = "🟢 Positive Divergence (Bottoming Signal)"
            _note   = f"New L_b low / A/D {_bup:.2f}% higher"
            _scolor = "#26d2a0"
        elif _bull:
            _status = "🔵 Early Positive Divergence"
            _note   = "New L_b low / A/D slightly improving"
            _scolor = "#1565c0"
        else:
            _status = "Neutral"
            _note   = "No divergence"
            _scolor = "#757575"

        # Internal implementation note.
        st.markdown(
            f'<div style="background:{_scolor};padding:12px 18px;border-radius:8px;margin:4px 0 8px 0">'
            f'<b style="font-size:1.2em;color:white">{_status}</b>'
            f'&nbsp;&nbsp;<span style="color:#ffffffcc">{_note}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Internal implementation note.
        c1, c2, c3 = st.columns(3)
        c1.metric("Latest Date", pd.to_datetime(str(last["date"]), format="%Y%m%d").strftime("%Y-%m-%d"))
        c2.metric(f"{market} Close", f"{float(last['close']):,.2f}")
        c3.metric("Daily A/D Diff", f"{int(last['ad_diff']):+,}")

        try:
            fig_main, ad_lookup = make_plotly_chart(df, market, sig, chart_months, hlab)

            # Plotly rendering and interaction logic.
            # Internal implementation note.
            # Plotly rendering and interaction logic.
            import plotly.io as _pio
            _ad_json = json.dumps(ad_lookup)
            _fig_html = _pio.to_html(
                fig_main,
                full_html=False,
                include_plotlyjs="cdn",
                div_id="ad_main_chart",
                config={"responsive": True, "displayModeBar": False},
            )
            _magnet_js = f"""
<script>
(function() {{
  const adData = {_ad_json};

  function toDateKey(xVal) {{
    if (typeof xVal === 'number') {{
      const d = new Date(xVal);
      return d.getFullYear() + '-'
        + String(d.getMonth()+1).padStart(2,'0') + '-'
        + String(d.getDate()).padStart(2,'0');
    }}
    return String(xVal).substring(0, 10);
  }}

  function init() {{
    const gd = document.getElementById('ad_main_chart');
    if (!gd || !gd._fullLayout) {{ setTimeout(init, 300); return; }}

    gd.on('plotly_hover', function(data) {{
      if (!data || !data.points || !data.points.length) return;
      const dateKey = toDateKey(data.points[0].x);
      const adVal = adData[dateKey];
      if (adVal === undefined) return;
      const shapes = (gd.layout.shapes || []).filter(s => s.name !== '_ad_magnet');
      shapes.push({{
        name: '_ad_magnet',
        type: 'line',
        xref: 'paper', x0: 0, x1: 1,
        yref: 'y2', y0: adVal, y1: adVal,
        line: {{ color: 'rgba(255,255,255,0.95)', width: 2, dash: 'solid' }},
      }});
      Plotly.relayout(gd, {{ shapes: shapes }});
    }});

    gd.on('plotly_unhover', function() {{
      const shapes = (gd.layout.shapes || []).filter(s => s.name !== '_ad_magnet');
      Plotly.relayout(gd, {{ shapes: shapes }});
    }});
  }}

  // Run after Plotly CDN is loaded
  if (typeof Plotly !== 'undefined') {{
    setTimeout(init, 200);
  }} else {{
    document.addEventListener('plotly_loaded', function() {{ setTimeout(init, 200); }});
    setTimeout(init, 1500);  // fallback
  }}
}})();
</script>
"""
            _full_html = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    body {{ margin: 0; padding: 0; background: transparent; }}
    #ad_main_chart {{ width: 100%; }}
  </style>
</head>
<body>
{_fig_html}
{_magnet_js}
</body>
</html>
"""
            import streamlit.components.v1 as _stc
            _stc.html(_full_html, height=690, scrolling=False)

        except Exception as e:
            st.error(f"Chart rendering failed: {e}")

        # Pine-style reference logic.
        # Internal implementation note.
        st.markdown("---")
        if _bear:
            # Internal implementation note.
            st.markdown(f"""
| Item | Value |
|---|---|
| H_a Previous High ({_ha_date}) | {hlab['ha_val']:,.2f} |
| A/D @ H_a | {hlab['ha_ad']:,.0f} |
| H_b Recent High ({_hb_date}) | {hlab['hb_val']:,.2f} |
| A/D @ H_b | {hlab['hb_ad']:,.0f}  ⚠ |
| A/D Divergence % | {_bdp:.2f}% |
| Signal | {_status} |
""")
        elif _bull:
            # Internal implementation note.
            st.markdown(f"""
| Item | Value |
|---|---|
| L_a Previous Low ({_la_date}) | {hlab['la_val']:,.2f} |
| A/D @ L_a | {hlab['la_ad']:,.0f} |
| L_b Recent Low ({_lb_date}) | {hlab['lb_val']:,.2f} |
| A/D @ L_b | {hlab['lb_ad']:,.0f}  △ |
| A/D Divergence % | {_bup:.2f}% |
| Signal | {_status} |
""")
        else:
            # Neutral: show both comparisons
            col_h, col_l = st.columns(2)
            with col_h:
                st.markdown(f"""
| Item | Value |
|---|---|
| H_a Previous High ({_ha_date}) | {hlab['ha_val']:,.2f} |
| A/D @ H_a | {hlab['ha_ad']:,.0f} |
| H_b Recent High ({_hb_date}) | {hlab['hb_val']:,.2f} |
| A/D @ H_b | {hlab['hb_ad']:,.0f} |
| A/D Divergence % | {_bdp:.2f}% |
| Signal | {_status} |
""")
            with col_l:
                st.markdown(f"""
| Item | Value |
|---|---|
| L_a Previous Low ({_la_date}) | {hlab['la_val']:,.2f} |
| A/D @ L_a | {hlab['la_ad']:,.0f} |
| L_b Recent Low ({_lb_date}) | {hlab['lb_val']:,.2f} |
| A/D @ L_b | {hlab['lb_ad']:,.0f} |
| A/D Divergence % | {_bup:.2f}% |
| Signal | {_status} |
""")
        st.markdown("---")

        with st.expander("📋 View Raw Data"):
            show = df.copy()
            show["date"] = pd.to_datetime(show["date"].astype(str), format="%Y%m%d").dt.strftime("%Y-%m-%d")
            cols = [c for c in ["date","advances","declines","unchanged",
                      "ad_diff","ad_line","close","breadth_thrust_ema10"] if c in show.columns]
            st.dataframe(
                show[cols].sort_values("date", ascending=False).reset_index(drop=True),
                width='stretch',
            )
            csv = show.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
            st.download_button("📥 Download CSV", csv,
                               f"{market}_breadth.csv", "text/csv")

    # ══════════════════════════════════════════════
    # Internal implementation note.
    # ══════════════════════════════════════════════
    elif active_tab == "⚡ Momentum":
        st.subheader("⚡ Momentum Index (Momentum Index)")
        st.caption(
            "Definition: 200-day rolling average of the advance-decline difference (A/D). "
            "Above zero indicates broad strength; below zero indicates broad weakness."
        )

        mi_window = st.slider("MA Window (default 200 days)", 50, 300, 200, step=10, key="mi_win")

        end_dt2   = pd.to_datetime(df["date"].astype(str), format="%Y%m%d").max()
        start_dt2 = end_dt2 - pd.DateOffset(months=chart_months)
        mask2 = pd.to_datetime(df["date"].astype(str), format="%Y%m%d") >= start_dt2
        pf2   = df[mask2].copy().reset_index(drop=True)
        pf2["dt"] = pd.to_datetime(pf2["date"].astype(str), format="%Y%m%d")

        ad_diff_s  = pd.Series(df["ad_diff"].values.astype(float))
        mi_full    = ad_diff_s.rolling(mi_window).mean()   # Definition: simple rolling average over N days

        mi_plot    = mi_full.iloc[mask2.values].reset_index(drop=True)

        last_mi    = mi_full.iloc[-1]
        prev_mi    = mi_full.iloc[-2] if len(mi_full) >= 2 else last_mi
        if pd.isna(last_mi):
            mi_verdict = "⚪ Not enough data"
            mi_color   = "#757575"
        elif last_mi > 0 and last_mi > prev_mi:
            mi_verdict = "🟢 Strengthening"
            mi_color   = "#2e7d32"
        elif last_mi > 0:
            mi_verdict = "🟡 Strength Slowing"
            mi_color   = "#f9a825"
        elif last_mi < 0 and last_mi < prev_mi:
            mi_verdict = "🔴 Weakening"
            mi_color   = "#c62828"
        else:
            mi_verdict = "🟠 Weakness Recovering"
            mi_color   = "#ef6c00"

        m1, m2, m3 = st.columns(3)
        m1.metric(f"MI ({mi_window}-day average)", f"{last_mi:+.1f}" if not pd.isna(last_mi) else "N/A")
        m2.metric("Day-over-Day Change", f"{(last_mi - prev_mi):+.1f}" if not pd.isna(last_mi) else "N/A")
        m3.metric("Signal", mi_verdict)

        fig_mi = go.Figure()
        fig_mi.add_trace(go.Bar(
            x=pf2["dt"], y=mi_plot,
            marker_color=[("#26a69a" if v >= 0 else "#ef5350") for v in mi_plot.fillna(0)],
            name=f"MI ({mi_window}-day average)", opacity=0.85
        ))
        fig_mi.add_hline(y=0, line_color="gray", line_dash="dot",
                         annotation_text="Zero Line")
        fig_mi.update_layout(
            title=f"{market} Momentum Index — {mi_window}-day rolling average of A/D Difference",
            template="plotly_dark", height=420,
            paper_bgcolor="rgba(14,17,23,1)",
            plot_bgcolor="rgba(14,17,23,1)",
            hoverlabel=dict(bgcolor="rgba(0,0,0,0.9)", font_color="#ffffff", font_size=12, bordercolor="#555"),
            legend=dict(orientation="h", y=1.05,
                        bgcolor="rgba(0,0,0,0.85)",
                        bordercolor="#333", borderwidth=1,
                        font=dict(color="white", size=11)),
            yaxis_title="MI Value (A/D average)"
        )
        import plotly.io as _pio
        import streamlit.components.v1 as _stc
        _mi_fig_html = _pio.to_html(fig_mi, full_html=False, include_plotlyjs="cdn",
                                    div_id="mi_chart",
                                    config={"responsive": True, "displayModeBar": False})
        _mi_full = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
  body {{ margin:0; padding:0; background:transparent; }}
  #mi_chart {{ width:100%; }}
  .hoverlayer .hovertext rect {{ fill: rgba(0,0,0,0.88) !important; stroke:#555 !important; }}
  .hoverlayer .hovertext text {{ fill: #ffffff !important; }}
  .hoverlayer .hovertext path {{ stroke: #555 !important; }}
</style>
</head><body>{_mi_fig_html}</body></html>"""
        _stc.html(_mi_full, height=450, scrolling=False)

        if len(df) < mi_window:
            st.warning(f"⚠️ Data {len(df)} days — not enough observations for the {mi_window}-day MA. "
                       f"Increase the collection period or reduce the MA window.")

    # ══════════════════════════════════════════════
    # TAB 3: NH-NL
    # ══════════════════════════════════════════════
    elif active_tab == "🏔 NH-NL":
        st.subheader("🏔 NH-NL: New Highs minus New Lows")
        st.caption(
            "Weekly new highs minus new lows. "
            "KRX daily component snapshots are used to identify 52-week highs/lows and aggregate them weekly."
        )

        nhnl_df = st.session_state.get(f"nhnl_{market}")
        if nhnl_df is None or nhnl_df.empty:
            if mode == "☁️ GitHub (Fast)":
                st.info("In GitHub fast mode, NH-NL is displayed only when a saved NH-NL CSV is available. It is loaded together with Load Data.")
            else:
                st.info("In KRX direct collection mode, NH-NL is calculated when Load Data is clicked.")
        if nhnl_df is not None and not nhnl_df.empty:
            from plotly.subplots import make_subplots as _msp2
            nhnl_df["dt"] = pd.to_datetime(nhnl_df["date"].astype(str), format="%Y%m%d")
            _today_ts = pd.Timestamp(datetime.today().date())
            # Internal implementation note.
            end_dt3   = nhnl_df["dt"].max()
            start_dt3 = end_dt3 - pd.DateOffset(months=chart_months)
            pf3       = nhnl_df[(nhnl_df["dt"] >= start_dt3) & (nhnl_df["dt"] <= end_dt3)].copy().reset_index(drop=True)

            # Internal implementation note.
            ns_all   = pd.Series(nhnl_df["nhnl"].values.astype(float))
            nma_all  = ns_all.rolling(4).mean()
            nma_plot = nma_all.iloc[(nhnl_df["dt"] >= start_dt3).values].reset_index(drop=True)

            # NH-NL calculation and visualization logic.
            _nhnl_daily_local = st.session_state.get(f"nhnl_daily_{market}")
            _today_date_int   = int(datetime.today().strftime("%Y%m%d"))
            _nh_label = ""  # source label

            if _nhnl_daily_local is not None and not _nhnl_daily_local.empty:
                # Internal implementation note.
                _daily_sorted = _nhnl_daily_local.sort_values("date")
                _last_daily   = _daily_sorted.iloc[-1]
                last_nhnl = int(_last_daily["nhnl"])
                last_nh   = int(_last_daily["new_highs"])
                last_nl   = int(_last_daily["new_lows"])
                _nh_label = f"Daily ({str(int(_last_daily['date']))[4:6]}/{str(int(_last_daily['date']))[6:8]})"
            else:
                # Data loading and preprocessing logic.
                last_nhnl = int(ns_all.iloc[-1])
                last_nh   = int(nhnl_df["new_highs"].iloc[-1])
                last_nl   = int(nhnl_df["new_lows"].iloc[-1])
                _last_wk_date = nhnl_df["dt"].iloc[-1]
                _nh_label = f"Weekly({_last_wk_date.strftime('%m/%d')} aggregate)"

            # Internal implementation note.
            lma = nma_all.iloc[-1]; pma = nma_all.iloc[-2] if len(nma_all) >= 2 else lma
            nhnl_ma_vals = nma_all.dropna()
            slope = np.polyfit(np.arange(len(nhnl_ma_vals)), nhnl_ma_vals.values, 1)[0] if len(nhnl_ma_vals) >= 2 else 0.0
            if pd.isna(lma):            nhnl_verdict, trend_color = "⚪ Not enough data",   "#757575"
            elif lma > 0 and lma > pma: nhnl_verdict, trend_color = "🟢 Strength",   "#2e7d32"
            elif lma > 0:               nhnl_verdict, trend_color = "🟡 Slowing",   "#f9a825"
            elif lma < 0 and lma < pma: nhnl_verdict, trend_color = "🔴 Weakness",   "#c62828"
            else:                       nhnl_verdict, trend_color = "🟠 Recovering", "#ef6c00"

            # Internal implementation note.
            # Internal implementation note.
            _last_data_dt = nhnl_df["dt"].max()
            _today_ts2 = pd.Timestamp(datetime.today().date())
            # Internal implementation note.
            _actual_last = min(_last_data_dt, _today_ts2)
            # Internal implementation note.
            _actual_mon = _actual_last - pd.Timedelta(days=_actual_last.weekday())
            _last_data_str = f"{_actual_mon.strftime('%Y/%m/%d')} ~ {_actual_last.strftime('%Y/%m/%d')}"

            # Internal implementation note.
            _weekly_last = nhnl_df.sort_values("dt").iloc[-1]
            _weekly_nh = int(_weekly_last["new_highs"])
            _weekly_nl = int(_weekly_last["new_lows"])
            _weekly_nhnl = int(_weekly_last["nhnl"])
            _weekly_date = pd.to_datetime(_weekly_last["dt"]).strftime("%Y/%m/%d")

            if _nhnl_daily_local is not None and not _nhnl_daily_local.empty:
                _daily_sorted2 = _nhnl_daily_local.sort_values("date")
                _daily_last = _daily_sorted2.iloc[-1]
                _daily_nh = int(_daily_last["new_highs"])
                _daily_nl = int(_daily_last["new_lows"])
                _daily_nhnl = int(_daily_last["nhnl"])
                _daily_date = pd.to_datetime(str(int(_daily_last["date"])), format="%Y%m%d").strftime("%Y/%m/%d")
            else:
                _daily_nh, _daily_nl, _daily_nhnl, _daily_date = last_nh, last_nl, last_nhnl, _weekly_date

            st.markdown(f"**📅 Weekly Aggregate: {_last_data_str} / Reference Date {_weekly_date}**")
            w1, w2, w3, w4 = st.columns(4)
            w1.metric("Weekly New Highs", f"{_weekly_nh:,}")
            w2.metric("Weekly New Lows", f"{_weekly_nl:,}")
            w3.metric("Weekly NH-NL", f"{_weekly_nhnl:+,}")
            w4.metric("Weekly Signal", nhnl_verdict)

            st.markdown(f"**📆 Daily Aggregate: {_daily_date}**")
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Daily New Highs", f"{_daily_nh:,}")
            d2.metric("Daily New Lows", f"{_daily_nl:,}")
            d3.metric("Daily NH-NL", f"{_daily_nhnl:+,}")

            if _daily_nhnl > 0 and (_nhnl_daily_local is None or len(_nhnl_daily_local) < 2 or _daily_nhnl >= int(_nhnl_daily_local.sort_values("date").iloc[-2]["nhnl"])):
                _daily_verdict = "🟢 Positive"
            elif _daily_nhnl > 0:
                _daily_verdict = "⚠️ Breadth Weakening"
            elif _daily_nhnl < 0:
                _daily_verdict = "🔴 Weakness"
            else:
                _daily_verdict = "🟡 Neutral"
            d4.metric("Daily Signal", _daily_verdict)

            # Internal implementation note.
            _has_index = all(c in df.columns for c in ["close", "high", "low"])
            pf_idx3 = df[pd.to_datetime(df["date"].astype(str), format="%Y%m%d") >= start_dt3].copy()
            pf_idx3["dt"] = pd.to_datetime(pf_idx3["date"].astype(str), format="%Y%m%d")
            if _has_index:
                pf_idx3 = pf_idx3.dropna(subset=["close"])

            # Pine-style reference logic.
            # Pine-style reference logic.
            _STRONG = 200  # Pine-style threshold
            _idx_recent = pf_idx3.tail(20)
            _idx_up = (_has_index and len(_idx_recent) >= 2 and
                       float(_idx_recent["close"].iloc[-1]) > float(_idx_recent["close"].iloc[0]))
            _nhnl_up = (len(nhnl_df) >= 2 and
                        float(nhnl_df["nhnl"].iloc[-1]) >= float(nhnl_df["nhnl"].iloc[-2]))
            _strong_bull = last_nhnl > _STRONG   # Pine: nhnl > 200
            _strong_bear = last_nhnl < -_STRONG  # Pine: nhnl < -200
            if not pd.isna(lma):
                if _idx_up and lma > 0 and _strong_bull and _nhnl_up:
                    nhnl_verdict, trend_color = "🟢 Strong Uptrend",  "#2e7d32"   # Pine: strong bullish breadth
                elif _idx_up and lma > 0 and _nhnl_up:
                    nhnl_verdict, trend_color = "🟢 Positive",      "#43a047"   # Pine: Positive
                elif _idx_up and lma > 0 and not _nhnl_up:
                    nhnl_verdict, trend_color = "⚠️ Breadth Weakening", "#ef6c00"   # Index rising, but NH-NL weakened
                elif _idx_up and lma > 0:
                    nhnl_verdict, trend_color = "🟡 Slowingin progress",    "#f9a825"
                elif not _idx_up and lma > 0 and _nhnl_up:
                    nhnl_verdict, trend_color = "🔵 Leading Recovery",  "#1e88e5"   # NH-NL recovers first
                elif _strong_bear and lma < 0 and lma < pma:
                    nhnl_verdict, trend_color = "🔴 Strong Downtrend",  "#b71c1c"   # Pine: strong bearish breadth
                elif lma < 0 and lma < pma:
                    nhnl_verdict, trend_color = "🔴 Weakness",      "#c62828"   # Pine: Caution
                elif lma < 0:
                    nhnl_verdict, trend_color = "🟠 Recovering",    "#ef6c00"
                else:
                    nhnl_verdict, trend_color = "🟡 Mixed",      "#f9a825"   # Pine: Mixed

            # Pine-style reference logic.
            _verdict_desc = {
                "🟢 Strong Uptrend":  "NH-NL > 200, MA+, Index rising (strong breadth)",
                "🟢 Positive":      "NH-NL+, MA+, Index↑ (Positive)",
                "⚠️ Breadth Weakening": "Index rising, but NH-NL decreased WoW (weakening warning)",
                "🟡 Slowingin progress":    "Index rising, but MA strength is weakening",
                "🔵 Leading Recovery":  "NH-NL recovering while index still declines",
                "🔴 Strong Downtrend":  "NH-NL < -200, MA-, Index falling (weak breadth)",
                "🔴 Weakness":      "MA-, MA downtrend in progress (Caution)",
                "🟠 Recovering":    "MA negative, but decline is slowing",
                "🟡 Mixed":      "MA direction unclear (mixed)",
                "⚪ Not enough data":      "Not enough data",
            }
            _desc = _verdict_desc.get(nhnl_verdict, "")
            if _desc:
                st.caption(f"ℹ️ {_desc} | Pine-style ±200 threshold applied")

            # Chart layout configuration.
            # Plotly rendering and interaction logic.
            fig_hl = go.Figure()

            # Chart layout configuration.
            if _has_index and not pf_idx3.empty:
                fig_hl.add_trace(go.Scatter(
                    x=pf_idx3["dt"], y=pf_idx3["close"],
                    line=dict(color="rgba(200,200,200,0.9)", width=1.8),
                    name=f"{market} Index",
                    xaxis="x", yaxis="y1",
                ))
            else:
                fig_hl.add_trace(go.Scatter(
                    x=[], y=[],
                    name=f"{market} Index (No data)",
                    xaxis="x", yaxis="y1",
                ))

            # Plotly rendering and interaction logic.
            # Internal implementation note.
            _nhnl_mon = pf3["dt"] - pd.Timedelta(days=4)
            _nhnl_fri = pf3["dt"]
            _week_labels = [
                f"{m.strftime('%-m/%-d')}(Mon)~{f.strftime('%-m/%-d')}(Fri)"
                for m, f in zip(_nhnl_mon, _nhnl_fri)
            ]
            fig_hl.add_trace(go.Scatter(
                x=pf3["dt"], y=pf3["nhnl"].astype(float),
                mode="lines+markers",
                line=dict(color="#26a69a", width=1.8),
                marker=dict(size=6, color="#26a69a", symbol="circle"),
                name="NH-NL",
                customdata=_week_labels,
                hovertemplate="Aggregation Period: %{customdata}<br>NH-NL: %{y:+,}<extra></extra>",
                xaxis="x", yaxis="y2",
            ))

            # ── This week's NH-NL forecast ──────────────────────────────
            # Data loading and preprocessing logic.
            # Internal implementation note.
            _forecast_error = None
            # Internal implementation note.
            _nhnl_base = list(pf3["nhnl"].astype(float))
            _y_min = min(_nhnl_base) * 1.15 if min(_nhnl_base) < 0 else min(_nhnl_base) * 0.85
            _y_max = max(_nhnl_base) * 1.15
            try:
                _today = pd.Timestamp(datetime.today().date())
                _this_mon = _today - pd.Timedelta(days=_today.weekday())
                _this_fri = _this_mon + pd.Timedelta(days=4)
                # Internal implementation note.
                _forecast_end = _this_fri + pd.Timedelta(days=1) if _today >= _this_fri else _this_fri

                # Internal implementation note.
                _prev_weekly = nhnl_df[nhnl_df["dt"] < _this_mon].copy().reset_index(drop=True)
                if _prev_weekly.empty:
                    _prev_weekly = nhnl_df.head(1).copy()
                _last_wk_dt   = pd.Timestamp(_prev_weekly["dt"].iloc[-1])
                _last_wk_nhnl = float(_prev_weekly["nhnl"].iloc[-1])

                # Internal implementation note.
                _this_week_row = nhnl_df[nhnl_df["dt"] == _this_fri]

                # Data loading and preprocessing logic.
                nhnl_daily_df = st.session_state.get(f"nhnl_daily_{market}")
                if nhnl_daily_df is not None and not nhnl_daily_df.empty:
                    _this_week_daily = nhnl_daily_df[
                        (nhnl_daily_df["dt"] >= _this_mon) & (nhnl_daily_df["dt"] <= _today)
                    ].copy()
                else:
                    _this_week_daily = pd.DataFrame()

                if not _this_week_daily.empty:
                    # Data loading and preprocessing logic.
                    _days_done   = len(_this_week_daily)
                    _current_sum = int(_this_week_daily["nhnl"].sum())
                    _daily_avg   = _current_sum / _days_done
                    _est_nhnl    = int(_daily_avg * 5)
                    _today_x     = pd.Timestamp(_this_week_daily["dt"].iloc[-1])
                    _est_label   = (f"This Week Forecast (actual {_days_done}days)<br>"
                                    f"current cumulative: {_current_sum:+,} → Friday forecast: {_est_nhnl:+,}")
                    _x_pts = [_last_wk_dt, _today_x, _this_fri]
                    _y_pts = [_last_wk_nhnl, _current_sum, _est_nhnl]

                elif not _this_week_row.empty:
                    # Data loading and preprocessing logic.
                    _current_sum = int(_this_week_row["nhnl"].iloc[-1])
                    _df_dt = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
                    _days_done = max(int((_df_dt >= _this_mon).sum()), 1)
                    _daily_avg = _current_sum / _days_done
                    _est_nhnl  = int(_daily_avg * 5)
                    _est_label = (f"This Week Forecast ({_days_done}days aggregated)<br>"
                                  f"current cumulative: {_current_sum:+,} → Friday forecast: {_est_nhnl:+,}")
                    _x_pts = [_last_wk_dt, _today, _this_fri]
                    _y_pts = [_last_wk_nhnl, _current_sum, _est_nhnl]

                else:
                    # Internal implementation note.
                    _daily_avg   = _last_wk_nhnl / 5.0
                    _days_done   = min(int(_today.weekday()) + 1, 5)
                    _current_sum = int(_daily_avg * _days_done)
                    _est_nhnl    = int(_daily_avg * 5)
                    _est_label   = (f"This Week Forecast (previous-week estimate)<br>"
                                    f"{_days_done}elapsed-days estimate: {_current_sum:+,} → Friday: {_est_nhnl:+,}")
                    _x_pts = [_last_wk_dt, _today, _this_fri]
                    _y_pts = [_last_wk_nhnl, _current_sum, _est_nhnl]

                # Data loading and preprocessing logic.
                # Data loading and preprocessing logic.
                # Data loading and preprocessing logic.
                if not _this_week_daily.empty:
                    _latest_daily_dt = pd.Timestamp(_this_week_daily["dt"].iloc[-1])
                    if _latest_daily_dt < _this_mon:
                        _this_week_daily = pd.DataFrame()  # invalidate if not current-week data

                _this_fri_confirmed = _today > _this_fri or _this_week_daily.empty
                if not _this_fri_confirmed:
                    # Internal implementation note.
                    # Internal implementation note.
                    _recent4 = _prev_weekly.tail(4)["nhnl"].values / 5.0  # weekly value ÷ 5 = daily average
                    _avg_opt  = int(float(max(_recent4)) * 5)   # Optimistic: highest over the previous 4 weeks
                    _avg_base = _est_nhnl                        # Neutral: current pace
                    _avg_pes  = int(float(min(_recent4)) * 5)   # Pessimistic: lowest over the previous 4 weeks

                    # Internal implementation note.
                    _s_vals = sorted([_avg_opt, _avg_base, _avg_pes], reverse=True)
                    _scenarios = [
                        # (label, legend_sym, est, color, marker_symbol, marker_size)
                        ("Optimistic", "▲", _s_vals[0], "rgba(100,220,130,0.95)", "triangle-up",   13),
                        ("Neutral", "◆", _s_vals[1], "rgba(255,210,60,0.95)",  "diamond",        11),
                        ("Pessimistic", "▼", _s_vals[2], "rgba(255,80,80,0.95)",   "triangle-down",  13),
                    ]

                    # Internal implementation note.
                    _scenario_start_x = _today
                    _scenario_start_y = _current_sum  # current-week cumulative value

                    # NH-NL calculation and visualization logic.
                    _nhnl_vals = list(pf3["nhnl"].astype(float))
                    # Internal implementation note.
                    _all_y = _nhnl_vals + _s_vals + [_scenario_start_y]
                    _y_raw_min = min(_all_y)
                    _y_raw_max = max(_all_y)
                    _y_span = max(_y_raw_max - _y_raw_min, 100)
                    _y_pad = _y_span * 0.25  # 25% padding for marker size
                    _y_min = _y_raw_min - _y_pad
                    _y_max = _y_raw_max + _y_pad

                    for _slabel, _ssymtxt, _sest, _scol, _ssym, _ssz in _scenarios:
                        # Internal implementation note.
                        fig_hl.add_trace(go.Scatter(
                            x=[_scenario_start_x, _forecast_end],
                            y=[_scenario_start_y, _sest],
                            mode="lines+markers",
                            line=dict(color=_scol, width=1.5, dash="longdashdot"),
                            marker=dict(size=[0, _ssz], color=_scol, symbol=_ssym,
                                        line=dict(color="white", width=1.2)),
                            showlegend=False,
                            hovertemplate=(f"{_slabel}<br>Friday forecast: {_sest:+,}<extra></extra>"),
                            xaxis="x", yaxis="y2",
                        ))
                        # Internal implementation note.
                        fig_hl.add_trace(go.Scatter(
                            x=[None], y=[None],
                            mode="lines",
                            line=dict(color=_scol, width=2, dash="longdashdot"),
                            name=f"{_ssymtxt} {_slabel} {_sest:+,}",
                            showlegend=True,
                            xaxis="x", yaxis="y2",
                        ))

                    # Internal implementation note.
            except Exception as _fe:
                _forecast_error = str(_fe)

            if _forecast_error:
                st.caption(f"⚠ Forecast calculation error: {_forecast_error}")
            # Internal implementation note.
            with st.expander("🔍 Scenario Debug", expanded=False):
                try:
                    st.write(f"Today: {_today} | This Friday: {_this_fri} | "
                             f"this_week_daily rows: {len(_this_week_daily)} | "
                             f"_this_fri_confirmed: {_this_fri_confirmed}")
                    if not _this_fri_confirmed:
                        st.write(f"Scenario start x: {_scenario_start_x} | start y: {_scenario_start_y}")
                        st.write(f"Optimistic: {_s_vals[0]} | Neutral: {_s_vals[1]} | Pessimistic: {_s_vals[2]}")
                        st.write(f"y_min: {_y_min:.0f} | y_max: {_y_max:.0f}")
                except Exception as _dbg_e:
                    st.write(f"Debug error: {_dbg_e}")

            # Internal implementation note.
            def _extend_line(dt1, y1, dt2, y2, ext_days=7):
                if dt1 == dt2:
                    return [dt1, dt2], [y1, y2]
                _slope = (y2 - y1) / max((dt2 - dt1).days, 1)
                dt_ext = dt2 + pd.Timedelta(days=ext_days)
                y_ext = y2 + _slope * ext_days
                return [dt1, dt2, dt_ext], [y1, y2, y_ext]

            def _nearest_row(frame: pd.DataFrame, target_dt: str, max_days: int = 5):
                if frame is None or frame.empty:
                    return None
                center = pd.Timestamp(target_dt)
                diffs = (frame["dt"] - center).abs()
                i = diffs.argmin()
                if diffs.iloc[i] > pd.Timedelta(days=max_days):
                    return None
                return frame.loc[i]

            def _idx_y_at_dt(target_dt: str, basis: str = "close"):
                if not _has_index or pf_idx3.empty:
                    return None
                r = _nearest_row(pf_idx3.reset_index(drop=True), target_dt, max_days=5)
                if r is None:
                    return None
                col = basis if basis in r.index else "close"
                return pd.Timestamp(r["dt"]), float(r[col])

            def _nhnl_y_at_dt(target_dt: str):
                r = _nearest_row(pf3.reset_index(drop=True), target_dt, max_days=5)
                if r is None:
                    return None
                return pd.Timestamp(r["dt"]), float(r["nhnl"])

            def _add_panel_line(panel: str, d1: str, d2: str, color: str,
                                label: str, basis: str = "close", ext_days: int = 7,
                                dash: str = "dot", width: float = 2.0):
                if panel == "index":
                    p1 = _idx_y_at_dt(d1, basis=basis)
                    p2 = _idx_y_at_dt(d2, basis=basis)
                    yaxis = yref = "y1"
                    yshift = -16 if basis == "low" else 16
                else:
                    p1 = _nhnl_y_at_dt(d1)
                    p2 = _nhnl_y_at_dt(d2)
                    yaxis = yref = "y2"
                    yshift = -16

                if p1 is None or p2 is None:
                    return
                dt1, y1 = p1
                dt2, y2 = p2
                xs, ys = _extend_line(dt1, y1, dt2, y2, ext_days=ext_days)
                fig_hl.add_trace(go.Scatter(
                    x=xs, y=ys, mode="lines",
                    line=dict(color=color, width=width, dash=dash),
                    name=label, showlegend=False,
                    hovertemplate=f"{label}<br>%{{x|%Y/%m/%d}}: %{{y:,.2f}}<extra></extra>",
                    xaxis="x", yaxis=yaxis,
                ))
                for _dt, _y in [(dt1, y1), (dt2, y2)]:
                    fig_hl.add_annotation(
                        x=_dt, y=_y, xref="x", yref=yref,
                        text=_dt.strftime("%m/%d"), showarrow=False,
                        font=dict(size=10, color=color),
                        bgcolor="rgba(0,0,0,0.35)",
                        yshift=yshift,
                    )

            _green = "rgba(80,220,140,0.95)"
            _red   = "rgba(255,95,95,0.95)"
            _blue  = "rgba(120,180,255,0.95)"
            _gold  = "rgba(255,200,50,0.95)"

            if market == "KOSPI":
                _add_panel_line("index", "2026-02-13", "2026-02-24", _green, "Index higher lows 02/13→02/24", basis="low",  ext_days=7)
                _add_panel_line("index", "2026-02-26", "2026-03-18", _red,   "Index downtrend 02/26→03/18",      basis="high", ext_days=10)
                _add_panel_line("index", "2026-03-04", "2026-03-31", _blue,  "Index support 03/04→03/31",      basis="low",  ext_days=14)
                _add_panel_line("index", "2026-03-31", "2026-04-13", _green, "Index higher lows 03/31→04/13", basis="low",  ext_days=14)
                _add_panel_line("nhnl", "2026-02-13", "2026-02-27", _gold,  "NH-NL downtrend 02/13→02/27", ext_days=7)
                _add_panel_line("nhnl", "2026-02-27", "2026-03-20", _red,   "NH-NL downtrend 02/27→03/20", ext_days=7)
                _add_panel_line("nhnl", "2026-03-06", "2026-04-03", _blue,  "NH-NL uptrend 03/06→04/03", ext_days=7)
                _add_panel_line("nhnl", "2026-03-20", "2026-04-03", _blue,  "NH-NL secondary 03/20→04/03", ext_days=0)
                _add_panel_line("nhnl", "2026-04-03", "2026-04-24", _green, "NH-NL uptrend 04/03→04/24", ext_days=10)

            elif market == "KOSDAQ":
                _add_panel_line("index", "2026-03-04", "2026-04-07", _green, "Index higher lows 03/04→04/07", basis="low", ext_days=7)
                _add_panel_line("index", "2026-04-07", "2026-04-24", _green, "Index higher lows 04/07→04/24", basis="low", ext_days=10)
                _add_panel_line("nhnl", "2026-01-30", "2026-02-27", _red,   "NH-NL downtrend 01/30→02/27", ext_days=7)
                _add_panel_line("nhnl", "2026-03-06", "2026-04-03", _green, "NH-NL uptrend 03/06→04/03", ext_days=7)
                _add_panel_line("nhnl", "2026-04-10", "2026-04-24", _green, "NH-NL uptrend 04/10→04/24", ext_days=10)

            # Internal implementation note.
            if market == "KOSDAQ" and _has_index and not pf_idx3.empty:
                def _one_idx_point(date_str: str):
                    target = pd.Timestamp(date_str)
                    diffs = (pf_idx3["dt"] - target).abs()
                    i = diffs.idxmin()
                    return pd.Timestamp(pf_idx3.loc[i, "dt"]), float(pf_idx3.loc[i, "close"])

                def _one_idx_line(date_a: str, date_b: str):
                    xa, ya = _one_idx_point(date_a)
                    xb, yb = _one_idx_point(date_b)
                    col = "rgba(80,255,150,0.72)"
                    fig_hl.add_trace(go.Scatter(
                        x=[xa, xb], y=[ya, yb],
                        mode="lines+markers",
                        line=dict(color=col, width=1.8, dash="dot"),
                        marker=dict(size=5, color=col, symbol="circle"),
                        showlegend=False, xaxis="x", yaxis="y1",
                        hovertemplate="%{x|%Y/%m/%d}<br>KOSDAQ close: %{y:,.2f}<extra></extra>",
                    ))
                    for x, y in [(xa, ya), (xb, yb)]:
                        fig_hl.add_annotation(
                            x=x, y=y, text=x.strftime("%m/%d"),
                            font=dict(size=9, color=col), showarrow=False,
                            yshift=-16, xref="x", yref="y1",
                        )

                _one_idx_line("2026-01-21", "2026-02-25")
                _one_idx_line("2026-02-06", "2026-02-25")

            # Internal implementation note.
            for _y, _color, _dash, _width in [
                (0,    "rgba(255,255,255,0.3)", "solid", 0.8),
                (500,  "rgba(100,220,100,0.5)", "dash",  1.0),
                (-500, "rgba(255,100,100,0.5)", "dash",  1.0),
            ]:
                fig_hl.add_shape(type="line",
                    xref="paper", x0=0, x1=1,
                    yref="y2", y0=_y, y1=_y,
                    line=dict(color=_color, dash=_dash, width=_width),
                    layer="below",
                )

            fig_hl.update_layout(
                template="plotly_dark", height=560,
                paper_bgcolor="rgba(14,17,23,1)",
                plot_bgcolor="rgba(14,17,23,1)",
                title=dict(text=f"{market} NH-NL — {nhnl_verdict}",
                           font=dict(size=13, color=trend_color)),
                hovermode="x",
                hoverlabel=dict(bgcolor="rgba(0,0,0,0.9)", font_color="#ffffff",
                               font_size=12, bordercolor="#555"),
                margin=dict(l=10, r=60, t=45, b=35),
                legend=dict(orientation="h", y=1.01,
                            bgcolor="rgba(0,0,0,0.85)",
                            bordercolor="#333", borderwidth=1,
                            font=dict(color="white", size=11)),
                # Chart layout configuration.
                xaxis=dict(
                    domain=[0, 1],
                    range=[start_dt3, _today_ts + pd.Timedelta(days=9)],
                    showspikes=True, spikemode="across", spikesnap="cursor",
                    spikethickness=1, spikecolor="rgba(200,200,200,0.8)", spikedash="solid",
                    tickformat="%Y/%m/%d", dtick=7*24*60*60*1000,
                    tickangle=-45, tickfont=dict(size=10),
                ),
                yaxis=dict(title="Index", domain=[0.58, 1.0],
                           showspikes=True, spikemode="across", spikesnap="cursor",
                           spikethickness=1, spikecolor="rgba(200,200,200,0.4)"),
                yaxis2=dict(title="NH-NL", domain=[0.0, 0.42], zeroline=False, anchor="x",
                            range=[_y_min, _y_max],
                            showspikes=True, spikemode="across", spikesnap="cursor",
                            spikethickness=1, spikecolor="rgba(200,200,200,0.4)"),
            )
            import plotly.io as _pio
            import streamlit.components.v1 as _stc
            _hl_fig_html = _pio.to_html(fig_hl, full_html=False, include_plotlyjs="cdn",
                                        div_id="hl_chart",
                                        config={"responsive": True, "displayModeBar": False})
            _hl_full = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
  body {{ margin:0; padding:0; background:transparent; }}
  #hl_chart {{ width:100%; }}
  .hoverlayer .hovertext rect {{ fill: rgba(0,0,0,0.88) !important; stroke:#555 !important; }}
  .hoverlayer .hovertext text {{ fill: #ffffff !important; }}
  .hoverlayer .hovertext path {{ stroke: #555 !important; }}
</style>
</head><body>{_hl_fig_html}</body></html>"""
            _stc.html(_hl_full, height=590, scrolling=False)
            st.caption(
                "📌 Friday NH-NL forecast (long dashed lines) — "
                "▲ **Optimistic**: highest weekly daily average over the last 4 weeks × 5 | "
                "◆ **Neutral**: current weekly pace × 5 | "
                "▼ **Pessimistic**: lowest weekly daily average over the last 4 weeks × 5  |  "
                "● **Actual NH-NL**: confirmed weekly value (solid line)"
            )

            # Data loading and preprocessing logic.
            with st.expander("📋 View Raw Data", expanded=False):
                _nhnl_daily_raw = st.session_state.get(f"nhnl_daily_{market}")
                if _nhnl_daily_raw is not None and not _nhnl_daily_raw.empty:
                    _daily_disp = _nhnl_daily_raw.copy()
                    _daily_disp["Date"] = pd.to_datetime(_daily_disp["date"].astype(str), format="%Y%m%d").dt.strftime("%Y/%m/%d")
                    _daily_disp = _daily_disp.rename(columns={"new_highs":"New Highs","new_lows":"New Lows","nhnl":"NH-NL"})
                    _daily_disp = _daily_disp[["Date","New Highs","New Lows","NH-NL"]].sort_values("Date", ascending=False).reset_index(drop=True)
                    st.caption("📅 Daily Data")
                    st.dataframe(_daily_disp, use_container_width=True, height=400)
                else:
                    display_df = pf3[["dt","new_highs","new_lows","nhnl"]].copy()
                    display_df = display_df.rename(columns={"dt":"Date","new_highs":"New Highs","new_lows":"New Lows","nhnl":"NH-NL"})
                    display_df["Date"] = display_df["Date"].dt.strftime("%Y/%m/%d")
                    display_df = display_df.sort_values("Date", ascending=False).reset_index(drop=True)
                    st.caption("📅 Weekly Data (Daily No data)")
                    st.dataframe(display_df, use_container_width=True, height=300)

if __name__ == "__main__":
    main()

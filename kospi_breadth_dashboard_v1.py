#!/usr/bin/env python3
from __future__ import annotations
# KOSPI / KOSDAQ Breadth Dashboard (Streamlit)
# 실행: streamlit run kospi_breadth_dashboard_v1.py
# GitHub raw CSV URL (로컬에서 data/ 폴더 push 후 Cloud에서 읽음)
GITHUB_RAW = "https://raw.githubusercontent.com/onekindalpha/Kospi/main/data"
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

import hashlib
import io
import os
from datetime import datetime, timedelta
from pathlib import Path

import platform
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

# ── 한글 폰트 Settings ──
def _setup_korean_font():
    import matplotlib.font_manager as fm
    import subprocess
    sys_name = platform.system()
    if sys_name == "Darwin":
        plt.rcParams["font.family"] = "AppleGothic"
    elif sys_name == "Windows":
        plt.rcParams["font.family"] = "Malgun Gothic"
    else:
        # Linux (Streamlit Cloud): NanumGothic 설치 시도
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
                # 폰트 설치 실패 시 차트 레이블을 영어로 대체 (아래 make_chart_img 참조)
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
    "BULLISH_CONFIRMATION":         ("✅ Bullish Confirmation",           "가격·A/D선 모두 High 근접 (동행)",                   "#2e7d32"),
    "BULLISH_DIVERGENCE":           ("🔴⚠️ Severe A/D Divergence",   "가격 High인데 A/D선이 크게 뒤처짐",                  "#c62828"),
    "BULLISH_DIVERGENCE_CANDIDATE": ("🟠⚠️ Early A/D Warning",       "Price is recovering faster than the A/D line",                    "#ef6c00"),
    "RECOVERY_IN_PROGRESS":         ("🟡Recovery in Progress",         "가격 High 재공략 중, 브레드스 미확인",                "#f9a825"),
    "DOWNSIDE_DIVERGENCE_CANDIDATE":("🟢Downside Divergence",      "Price is near lows while A/D line does not confirm lows",                 "#00838f"),
    "NORMAL_WEAKNESS":              ("⚫ Broad Weakness",           "Price and A/D line are both near recent lows",                          "#455a64"),
    "NEUTRAL":                      ("⬜ Neutral",                 "No clear signal",                                   "#757575"),
}

# ──────────────────────────────────────────────────────────────
# NH-NL 캐시 경로
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
    # 예전 잘못 생성된 짧은 캐시(예: 5주치)는 자동 무시
    if df.empty or len(df) < 20:
        return None
    return df

def save_nhnl_cache(df: pd.DataFrame, market: str, date_str: str):
    p = _nhnl_cache_path(market, date_str)
    df.to_csv(p, index=False)


def _is_common_stock_krx(df: pd.DataFrame) -> pd.Series:
    """
    책 취지에 맞게 보통주 중심으로 필터링한다.
    우선주는 이름/단축코드 패턴으로 최대한 제거한다.
    ETF/ETN/ELW/스팩/리츠/펀드/인버스/레버리지도 제외한다.
    """
    if df.empty:
        return pd.Series(dtype=bool)

    name_col = next((c for c in ["ISU_ABBRV", "ISU_NM", "Name", "name"] if c in df.columns), None)
    code_col = next((c for c in ["ISU_SRT_CD", "Code", "Symbol", "code"] if c in df.columns), None)

    name = df[name_col].astype(str).fillna("") if name_col else pd.Series([""] * len(df), index=df.index)
    code = df[code_col].astype(str).fillna("") if code_col else pd.Series([""] * len(df), index=df.index)

    exclude_pat = (
        r"(?:우$|우B$|우C$|[0-9]우$|스팩|리츠|REIT|ETF|ETN|ELW|KODEX|TIGER|KOSEF|KBSTAR|ARIRANG|HANARO|"
        r"SOL|ACE|TIMEFOLIO|TREX|SMART|FOCUS|마이티|TRUE|QV|RISE|레버리지|인버스|선물|채권|"
        r"펀드|액티브|TDF|TRF|BLN|회사채|국고채)"
    )
    bad_name = name.str.contains(exclude_pat, case=False, regex=True, na=False)

    # KRX 보통주 외의 특수코드/우선주/기타 증권 일부 제거 보조
    bad_code = code.str.endswith(("K", "L", "M", "N"))  # 예외적 코드 방어
    return ~(bad_name | bad_code)


def compute_nhnl_pykrx(market: str, end_date: str, prog=None, auth_key: str = "", chart_start_date: str | None = None) -> pd.DataFrame:
    """
    책 기준 NH-NL 구현:
    - 보통주 중심
    - Close 기준
    - 52주(252거래일) 신고가/신저가 돌파 종목 수
    - 주간 합계(W-FRI)
    Data Source는 pykrx/FDR 대신 KRX 일별 전체종목 스냅샷 사용.
    """
    if not auth_key or not str(auth_key).strip():
        raise RuntimeError("NH-NL은 현재 KRX API AUTH_KEY 기반으로 계산합니다. 사이드바의 KRX AUTH_KEY를 입력하세요.")

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
            prog.progress(i / total, text=f"NH-NL 계산용 KRX Collecting... {bas_dd} ({i}/{total})")

    if not daily_frames:
        raise RuntimeError("NH-NL 계산용 KRX 일별 종목 데이터가 없습니다.")

    panel = pd.concat(daily_frames, ignore_index=True)
    panel["dt"] = pd.to_datetime(panel["date"], format="%Y%m%d")
    panel = panel.sort_values(["code", "dt"]).drop_duplicates(["code", "dt"], keep="last")

    # 종목별 거래일 수 기준으로 너무 짧은 히스토리는 제외
    valid_counts = panel.groupby("code")["dt"].size()
    valid_codes = valid_counts[valid_counts >= 260].index
    panel = panel[panel["code"].isin(valid_codes)].copy()
    if panel.empty:
        raise RuntimeError("52주 판정에 필요한 히스토리를 가진 종목이 없습니다.")

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

    # 너무 앞쪽 워밍업 구간 제거
    cutoff = pd.to_datetime(start_dt) + pd.Timedelta(days=365)
    weekly = weekly[weekly["dt"] >= cutoff].reset_index(drop=True)
    return weekly


def compute_nhnl_fdr(market: str, end_date: str, prog=None, auth_key: str = "") -> pd.DataFrame:
    return compute_nhnl_pykrx(market=market, end_date=end_date, prog=prog, auth_key=auth_key)
# ──────────────────────────────────────────────────────────────
# 파일 캐시 유틸
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
            st.warning(f"{bas_dd} 스킵: {e}")
        prog.progress(i / len(dates), text=f"Collecting... {bas_dd} ({i}/{len(dates)})")
    prog.empty()
    if not rows:
        raise RuntimeError("수집된 데이터 없음")
    out = pd.DataFrame(rows)
    br = (out["advances"] / (out["advances"] + out["declines"]).replace(0, pd.NA)).astype(float)
    out["breadth_thrust_ema10"] = br.ewm(span=10, adjust=False).mean()
    return out

# ──────────────────────────────────────────────────────────────
# GitHub raw CSV 로드
# ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=1800)
def load_from_github(market: str) -> pd.DataFrame:
    """GitHub에 push된 CSV(breadth + index 머지)를 읽어 반환"""
    import requests as _req
    b_url = GITHUB_BREADTH[market]
    i_url = GITHUB_INDEX[market]

    resp_b = _req.get(b_url, timeout=15)
    if resp_b.status_code != 200:
        raise RuntimeError(f"GitHub breadth CSV 없음 ({resp_b.status_code})\n{b_url}\n→ 로컬에서 update_and_push.sh 실행 후 push 해주세요.")
    breadth = pd.read_csv(io.StringIO(resp_b.text), dtype={"date": str})

    resp_i = _req.get(i_url, timeout=15)
    if resp_i.status_code != 200:
        raise RuntimeError(f"GitHub index CSV 없음 ({resp_i.status_code})\n{i_url}\n→ 로컬에서 update_and_push.sh 실행 후 push 해주세요.")
    idx = pd.read_csv(io.StringIO(resp_i.text), dtype={"date": str})

    df = breadth.merge(idx[["date","open","high","low","close"]], on="date", how="inner")
    df = df.sort_values("date").reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False, ttl=1800)
def load_nhnl_from_github(market: str) -> pd.DataFrame | None:
    import requests as _req
    url = GITHUB_NHNL[market]
    resp = _req.get(url, timeout=15)
    if resp.status_code != 200:
        return None
    df = pd.read_csv(io.StringIO(resp.text), dtype={"date": str})
    if df.empty:
        return None

    # 컬럼 이름 정규화
    rename_map = {}
    if "new_high" in df.columns and "new_highs" not in df.columns:
        rename_map["new_high"] = "new_highs"
    if "new_low" in df.columns and "new_lows" not in df.columns:
        rename_map["new_low"] = "new_lows"
    if rename_map:
        df = df.rename(columns=rename_map)

    required = {"date", "nhnl"}
    if not required.issubset(df.columns):
        raise RuntimeError(f"GitHub NH-NL CSV 컬럼 부족: {list(df.columns)}")

    if "new_highs" not in df.columns:
        df["new_highs"] = 0
    if "new_lows" not in df.columns:
        df["new_lows"] = 0

    return df.sort_values("date").reset_index(drop=True)

# ──────────────────────────────────────────────────────────────
# Index OHLC
# ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_index_ohlc(market, start, end):
    if not FDR_OK:
        raise RuntimeError("finance-datareader 미설치")
    symbol = FDR_SYMBOLS[market]
    end_dt = datetime.strptime(end, "%Y%m%d") + timedelta(days=1)
    raw = fdr.DataReader(symbol, start, end_dt.strftime("%Y-%m-%d"))
    if raw.empty:
        raise RuntimeError(f"{symbol} 데이터 없음")
    raw.columns = [str(c).strip().title() for c in raw.columns]
    df = raw.reset_index()
    df.columns = [str(c).strip().title() for c in df.columns]
    date_col = next((c for c in df.columns if c.lower() in ("date", "datetime")), None)
    if not date_col:
        raise RuntimeError(f"날짜 컬럼 없음: {list(df.columns)}")
    def _find(*candidates):
        for c in candidates:
            if c in df.columns:
                return c
        raise RuntimeError(f"{candidates} 컬럼 없음: {list(df.columns)}")
    out = pd.DataFrame({
        "date":  pd.to_datetime(df[date_col]).dt.strftime("%Y%m%d"),
        "open":  pd.to_numeric(df[_find("Open")],  errors="coerce"),
        "high":  pd.to_numeric(df[_find("High")],  errors="coerce"),
        "low":   pd.to_numeric(df[_find("Low")],   errors="coerce"),
        "close": pd.to_numeric(df[_find("Close", "Adj Close")], errors="coerce"),
    })
    return out[out["date"] <= end].dropna().reset_index(drop=True)

# ──────────────────────────────────────────────────────────────
# 판정 로직
# ──────────────────────────────────────────────────────────────
def classify(price_off_high, ad_off_high, gap,
             price_off_low, ad_off_low,
             price_thr=2.0, ad_thr=3.0, gap_warn=1.5, gap_danger=2.5):
    # 직관적 부호: - = High 아래, + = High 위
    # gap = adOff - priceOff: + = A/D 선행(좋음), - = A/D 지연(나쁨)
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

    # 직관적 부호: - = 아래, + = 위
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
# H_a / H_b / L_a / L_b 계산 (파인스크립트 로직 그대로)
# ──────────────────────────────────────────────────────────────
def compute_hlab(df: pd.DataFrame, high_bars: int = 60, low_bars: int = 130) -> dict:
    """
    파인스크립트 v16과 동일한 로직:
    H_b = 최근 high_bars 구간 High
    H_a = 그 이전 high_bars 구간 High
    L_b = 최근 low_bars 구간 저점
    L_a = 그 이전 low_bars 구간 저점
    """
    closes  = df["close"].values.astype(float)
    ad_line = df["ad_line"].values.astype(float)
    dts     = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
    n = len(closes)

    def _safe_slice(arr, end_idx, length):
        start = max(0, end_idx - length)
        return arr[start:end_idx], start

    # H_b: 최근 high_bars 구간
    hb_window, hb_start = _safe_slice(closes, n, high_bars)
    if len(hb_window) == 0:
        hb_window = closes
        hb_start  = 0
    hb_idx_local = int(np.argmax(hb_window))
    hb_idx  = hb_start + hb_idx_local
    hb_val  = closes[hb_idx]
    hb_dt   = dts.iloc[hb_idx]
    hb_ad   = ad_line[hb_idx]

    # H_a: 이전 high_bars 구간 (H_b 구간 앞)
    ha_window, ha_start = _safe_slice(closes, hb_start + hb_idx_local, high_bars)
    if len(ha_window) > 0:
        ha_idx_local = int(np.argmax(ha_window))
        ha_idx  = ha_start + ha_idx_local
        ha_val  = closes[ha_idx]
        ha_dt   = dts.iloc[ha_idx]
        ha_ad   = ad_line[ha_idx]
    else:
        ha_val, ha_dt, ha_ad, ha_idx = hb_val, hb_dt, hb_ad, hb_idx

    # L_b: 최근 low_bars 구간
    lb_window, lb_start = _safe_slice(closes, n, low_bars)
    if len(lb_window) == 0:
        lb_window = closes
        lb_start  = 0
    lb_idx_local = int(np.argmin(lb_window))
    lb_idx  = lb_start + lb_idx_local
    lb_val  = closes[lb_idx]
    lb_dt   = dts.iloc[lb_idx]
    lb_ad   = ad_line[lb_idx]

    # L_a: 이전 low_bars 구간
    la_window, la_start = _safe_slice(closes, lb_start + lb_idx_local, low_bars)
    if len(la_window) > 0:
        la_idx_local = int(np.argmin(la_window))
        la_idx  = la_start + la_idx_local
        la_val  = closes[la_idx]
        la_dt   = dts.iloc[la_idx]
        la_ad   = ad_line[la_idx]
    else:
        la_val, la_dt, la_ad, la_idx = lb_val, lb_dt, lb_ad, lb_idx

    # 불일치 판정
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
# 차트 — domain 수동 분할 (rangeslider 문제 완전 회피)
# ──────────────────────────────────────────────────────────────
def make_plotly_chart(df: pd.DataFrame, market: str, sig: dict,
                      chart_months: int, hlab: dict) -> go.Figure:
    end_dt   = pd.to_datetime(df["date"].astype(str), format="%Y%m%d").max()
    start_dt = end_dt - pd.DateOffset(months=chart_months)
    mask     = pd.to_datetime(df["date"].astype(str), format="%Y%m%d") >= start_dt
    pf       = df[mask].copy().reset_index(drop=True)
    pf["dt"] = pd.to_datetime(pf["date"].astype(str), format="%Y%m%d")

    hb_color = "rgba(255,80,80,0.95)"  if hlab["bear_div"] else "rgba(160,160,160,0.8)"
    ha_color = "rgba(255,140,140,0.6)" if hlab["bear_div"] else "rgba(120,120,120,0.5)"
    lb_color = "rgba(38,210,160,0.95)" if hlab["bull_div"] else "rgba(160,160,160,0.8)"
    la_color = "rgba(38,210,160,0.6)"  if hlab["bull_div"] else "rgba(120,120,120,0.5)"

    # y축 여백 강제 확보: 신고가 근처에서도 캔들이 천장에 붙지 않게 함
    price_low  = float(pf["low"].min())
    price_high = float(pf["high"].max())
    price_span = max(price_high - price_low, abs(price_high) * 0.02, 1.0)
    price_pad_top = price_span * 0.45
    price_pad_bot = price_span * 0.08
    y1_range = [price_low - price_pad_bot, price_high + price_pad_top]

    ad_min = float(pf["ad_line"].min())
    ad_max = float(pf["ad_line"].max())
    ad_span = max(ad_max - ad_min, max(abs(ad_max), 1.0) * 0.02, 1.0)
    ad_pad = ad_span * 0.10
    y2_range = [ad_min - ad_pad, ad_max + ad_pad]

    if hlab["bear_div"]:
        div_text, div_color = f"⚠ 부정적 불일치 {hlab['bear_div_pct']:.1f}%", "#ff5050"
    elif hlab["bull_div"]:
        div_text, div_color = f"✓ 긍정적 불일치 {hlab['bull_div_pct']:.1f}%", "#26d2a0"
    else:
        div_text, div_color = "불일치 없음", "#aaaaaa"

    fig = go.Figure()

    # ── 위 패널 (y1: domain 0.52~1.0): 캔들스틱
    fig.add_trace(go.Candlestick(
        x=pf["dt"], open=pf["open"], high=pf["high"], low=pf["low"], close=pf["close"],
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        name=market, showlegend=False,
        xaxis="x", yaxis="y1",
    ))

    # ── 아래 패널 (y2: domain 0.0~0.48): A/D Line
    fig.add_trace(go.Scatter(
        x=pf["dt"], y=pf["ad_line"].astype(float),
        line=dict(color="#1e88e5", width=2.0), name="A/D Line",
        xaxis="x", yaxis="y2",
    ))

    # 가격 정규화선 (아래 패널 참조용)
    ad_vals = pf["ad_line"].astype(float)
    ad_min, ad_max = ad_vals.min(), ad_vals.max()
    pr_min, pr_max = pf["close"].min(), pf["close"].max()
    price_mapped = (ad_min + (pf["close"] - pr_min) / (pr_max - pr_min) * (ad_max - ad_min)
                    if pr_max != pr_min else ad_vals)
    fig.add_trace(go.Scatter(
        x=pf["dt"], y=price_mapped,
        line=dict(color="rgba(180,180,180,0.35)", width=1.0),
        name="가격(참조)", showlegend=False,
        xaxis="x", yaxis="y2",
    ))

    # 위 패널 수평선
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

    # 아래 패널 수평선
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
        title=dict(text=f"{market} — {div_text}", font=dict(size=14, color=div_color)),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#1e1e2e", font_color="#ffffff", font_size=12, bordercolor="#555"),
        legend=dict(orientation="h", y=1.01, x=0),
        margin=dict(l=10, r=90, t=55, b=10),
        xaxis=dict(
            rangeslider=dict(visible=False),
            showspikes=True, spikemode="across", spikesnap="cursor",
            spikethickness=1, spikecolor="rgba(200,200,200,0.7)", spikedash="solid",
            tickformat="%Y/%m/%d", dtick=7*24*60*60*1000,
            tickangle=-45, tickfont=dict(size=8),
            domain=[0, 1],
        ),
        yaxis=dict(
            title="Index", domain=[0.52, 1.0], range=y1_range,
            showspikes=True, spikemode="across", spikesnap="hovered data",
            spikethickness=1, spikecolor="rgba(200,200,200,0.35)", spikedash="dot",
        ),
        yaxis2=dict(
            title="A/D Line", domain=[0.0, 0.48], range=y2_range,
            showspikes=True, spikemode="across", spikesnap="hovered data",
            spikethickness=1, spikecolor="rgba(200,200,200,0.55)", spikedash="dot",
        ),
    )
        # 참고: Plotly 기본 기능상 위 패널 hover만으로 아래 A/D의 대응 y값 가로선까지 자동 표시는 어렵다.
    # 대신 아래 A/D 패널 hover 시 가로 점선 crosshair가 보이도록 Settings했다.
    return fig

# ──────────────────────────────────────────────────────────────
# 메인 앱
# ──────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="KOSPI Breadth Analysis Dashboard",
                       page_icon="📊", layout="wide")
    st.title("KOSPI Breadth Analysis Dashboard")
    st.caption("A/D Line · Breadth Thrust · Price-Breadth Divergence")

    # ── 사이드바 ──────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")
        market = st.selectbox("Market", ["KOSPI", "KOSDAQ"])

        mode = st.radio("Data Source", ["☁️ GitHub (Fast)", "🔑 KRX API (Direct Collection)"],
                        index=0,
                        help="GitHub: Actions가 매일 자동 업데이트한 CSV 사용\nKRX API: 직접 수집 (AUTH_KEY 필요)")

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
            st.caption("💡 새로 불러오고 싶으면 아래 캐시를 지우고 불러오세요.")

        st.divider()
        st.subheader("Analysis Parameters")
        lookback     = st.slider("Lookback (days)",      20, 252, 126)
        chart_months = st.slider("Chart Display Period (months)", 1,  24,  6)
        high_bars    = st.slider("High 탐색 구간 H_b (일)", 10, 500, 60)
        low_bars     = st.slider("Low Lookback L_b (days)", 10, 500, 130)
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
        st.info("👈 사이드바에서 Market 선택 후 **Load Data** 버튼을 눌러주세요.")
        return

    if fetch_btn:
        st.session_state.pop(f"nhnl_{market}", None)
        if mode == "☁️ GitHub (Fast)":
            try:
                with st.spinner("GitHub에서 CSV 읽는 중…"):
                    df = load_from_github(market)
                    nhnl_df = load_nhnl_from_github(market)
                st.success(f"✅ GitHub 로드 완료 — {len(df)}일치 / 최신: {df['date'].iloc[-1]}")
                st.session_state[f"nhnl_{market}"] = nhnl_df if nhnl_df is not None and not nhnl_df.empty else None
                if nhnl_df is None or nhnl_df.empty:
                    st.info("NH-NL은 왼쪽에서 '🔑 KRX API (Direct Collection)'를 선택한 뒤, 'Load Data'를 누르면 함께 로드됩니다.")
            except Exception as e:
                st.error(f"GitHub 로드 실패: {e}")
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
                    st.success(f"✅ 캐시에서 로드 ({market} {start_str}~{end_str})")
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
                    st.success(f"✅ A/D 데이터 Collection completed — {len(df)}일치")

                if nhnl_cached is not None and not nhnl_cached.empty:
                    nhnl_df = nhnl_cached
                    st.success(f"✅ NH-NL 캐시 로드 — {len(nhnl_df)}주치")
                else:
                    prog3 = st.progress(0, text="NH-NL 계산용 KRX Collecting...")
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
                        st.success(f"✅ NH-NL 계산 완료 — {len(nhnl_df)}주치")
                st.session_state[f"nhnl_{market}"] = nhnl_df if nhnl_df is not None and not nhnl_df.empty else None
            except Exception as e:
                st.error(f"Data collection failed: {type(e).__name__}: {e}")
                return

        st.session_state["df_merged"] = df
        st.session_state["df_market"] = market

    # Market이 바뀌면 세션 초기화
    if st.session_state.get("df_market") != market:
        st.session_state.pop("df_merged", None)
        st.info("Market이 변경됐어요. Load Data를 다시 눌러주세요.")
        return

    # ── 차트 및 판정 출력 ───────────────────────────
    df = st.session_state["df_merged"]

    if len(df) < lookback:
        st.warning(f"Not enough data: {len(df)}행 (lookback={lookback})")
        return

    sig  = compute_signals(df, lookback, price_thr, ad_thr, gap_warn, gap_danger)
    hlab = compute_hlab(df, high_bars=high_bars, low_bars=low_bars)
    last = df.iloc[-1]

    # ── 탭 구성 ──
    # st.tabs 는 서버측에서 active tab을 제어/유지할 수 없어서
    # 버튼 클릭 시 rerun 되면 첫 탭으로 돌아가 보일 수 있음.
    TAB_LABELS = ["📈 A/D Line", "⚡ Momentum", "🏔 NH-NL"]
    current_tab = st.session_state.get("active_tab", TAB_LABELS[0])
    if current_tab not in TAB_LABELS:
        current_tab = TAB_LABELS[0]
        st.session_state["active_tab"] = current_tab

    _default_idx = TAB_LABELS.index(current_tab)
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
    # TAB 1: 기존 A/D Line 분석
    # ══════════════════════════════════════════════
    if active_tab == "📈 A/D Line":
        gap_color = "#00897b" if sig["gap"] >= 0 else "#c62828"
        gap_arrow = "▲" if sig["gap"] >= 0 else "▼"
        st.markdown(
            f'<div style="text-align:center;padding:6px 0 2px 0">'
            f'<span style="font-size:0.85em;color:#aaaaaa">Divergence (A/D − Price)</span><br>'
            f'<span style="font-size:2.6em;font-weight:900;color:{gap_color}">'
            f'{gap_arrow} {sig["gap"]:+.2f}%</span>'
            f'<span style="font-size:0.8em;color:#aaaaaa;margin-left:8px">'
            f'Reference: {sig["peak_label"]}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Latest Date",
                  pd.to_datetime(str(last["date"]), format="%Y%m%d").strftime("%Y-%m-%d"))
        c2.metric(f"{market} Close", f"{float(last['close']):,.2f}")
        c3.metric("Daily A/D Diff",   f"{int(last['ad_diff']):+,}")
        c4.metric("Price vs High", f"{sig['price_off']:.2f}%")
        c5.metric("A/D vs High",  f"{sig['ad_off']:.2f}%")

        st.markdown(
            f'<div style="background:{sig["color"]};padding:12px 18px;border-radius:8px;margin:8px 0">'
            f'<b style="font-size:1.2em;color:white">{sig["verdict"]}</b>'
            f'&nbsp;&nbsp;<span style="color:#ffffffcc">{sig["note"]}</span>'
            f'&nbsp;&nbsp;<span style="color:#ffffffaa;font-size:0.9em">Reference: {sig["peak_label"]}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        try:
            fig_main = make_plotly_chart(df, market, sig, chart_months, hlab)
            st.plotly_chart(fig_main, width='stretch')
        except Exception as e:
            st.error(f"Chart rendering failed: {e}")

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
    # TAB 2: MI 탄력Index (스탠 와인스태인 책 정의)
    # ══════════════════════════════════════════════
    elif active_tab == "⚡ Momentum":
        st.subheader("⚡ MI 탄력Index (Momentum Index)")
        st.caption(
            "스탠 와인스태인 책 정의: 등락종목수 차이(AD)의 200일 롤링 평균. "
            "0선 위 = 시장 강세, 0선 아래 = 시장 약세."
        )

        mi_window = st.slider("MA 기간 (기본 200일)", 50, 300, 200, step=10, key="mi_win")

        end_dt2   = pd.to_datetime(df["date"].astype(str), format="%Y%m%d").max()
        start_dt2 = end_dt2 - pd.DateOffset(months=chart_months)
        mask2 = pd.to_datetime(df["date"].astype(str), format="%Y%m%d") >= start_dt2
        pf2   = df[mask2].copy().reset_index(drop=True)
        pf2["dt"] = pd.to_datetime(pf2["date"].astype(str), format="%Y%m%d")

        ad_diff_s  = pd.Series(df["ad_diff"].values.astype(float))
        mi_full    = ad_diff_s.rolling(mi_window).mean()   # 책 정의: N일 단순 롤링 평균

        mi_plot    = mi_full.iloc[mask2.values].reset_index(drop=True)

        last_mi    = mi_full.iloc[-1]
        prev_mi    = mi_full.iloc[-2] if len(mi_full) >= 2 else last_mi
        if pd.isna(last_mi):
            mi_verdict = "⚪ Not enough data"
            mi_color   = "#757575"
        elif last_mi > 0 and last_mi > prev_mi:
            mi_verdict = "🟢 강세 상승"
            mi_color   = "#2e7d32"
        elif last_mi > 0:
            mi_verdict = "🟡 강세 둔화"
            mi_color   = "#f9a825"
        elif last_mi < 0 and last_mi < prev_mi:
            mi_verdict = "🔴 약세 하락"
            mi_color   = "#c62828"
        else:
            mi_verdict = "🟠 약세 회복 중"
            mi_color   = "#ef6c00"

        m1, m2, m3 = st.columns(3)
        m1.metric(f"MI ({mi_window}일 평균)", f"{last_mi:+.1f}" if not pd.isna(last_mi) else "N/A")
        m2.metric("전일 대비", f"{(last_mi - prev_mi):+.1f}" if not pd.isna(last_mi) else "N/A")
        m3.metric("판정", mi_verdict)

        fig_mi = go.Figure()
        fig_mi.add_trace(go.Bar(
            x=pf2["dt"], y=mi_plot,
            marker_color=[("#26a69a" if v >= 0 else "#ef5350") for v in mi_plot.fillna(0)],
            name=f"MI ({mi_window}일 평균)", opacity=0.85
        ))
        fig_mi.add_hline(y=0, line_color="gray", line_dash="dot",
                         annotation_text="기준선(0)")
        fig_mi.update_layout(
            title=f"{market} MI 탄력Index — AD차이 {mi_window}일 롤링 평균 (스탠 와인스태인)",
            template="plotly_dark", height=420,
            legend=dict(orientation="h", y=1.05),
            yaxis_title="MI 값 (AD 평균)"
        )
        st.plotly_chart(fig_mi, width='stretch')

        if len(df) < mi_window:
            st.warning(f"⚠️ 데이터 {len(df)}일 — {mi_window}일 MA 계산에 데이터가 부족합니다. "
                       f"수집 기간을 늘리거나 MA 기간을 줄여주세요.")

    # ══════════════════════════════════════════════
    # TAB 3: NH-NL
    # ══════════════════════════════════════════════
    elif active_tab == "🏔 NH-NL":
        st.subheader("🏔 High-저점 수치 (신고가 - 신저가 종목 수)")
        st.caption(
            "스탠 와인스태인 책 정의: 매주 신고가 기록 종목 수 - 신저가 기록 종목 수. "
            "KRX API 일별 전체 종목 스냅샷으로 52주 신고가/신저가를 판별해 주간 집계합니다."
        )

        nhnl_df = st.session_state.get(f"nhnl_{market}")
        if nhnl_df is None or nhnl_df.empty:
            if mode == "☁️ GitHub (Fast)":
                st.info("NH-NL은 왼쪽에서 '🔑 KRX API (Direct Collection)'를 선택한 뒤, 'Load Data'를 누르면 함께 로드됩니다.")
            else:
                st.info("KRX 직접 수집 모드에서는 'Load Data'를 누를 때 NH-NL도 함께 계산합니다.")
        if nhnl_df is not None and not nhnl_df.empty:
            from plotly.subplots import make_subplots as _msp2
            nhnl_df["dt"] = pd.to_datetime(nhnl_df["date"].astype(str), format="%Y%m%d")
            end_dt3   = nhnl_df["dt"].max()
            start_dt3 = end_dt3 - pd.DateOffset(months=chart_months)
            pf3       = nhnl_df[nhnl_df["dt"] >= start_dt3].copy().reset_index(drop=True)

            # 4주 MA 전체 기준 계산
            ns_all   = pd.Series(nhnl_df["nhnl"].values.astype(float))
            nma_all  = ns_all.rolling(4).mean()
            nma_plot = nma_all.iloc[(nhnl_df["dt"] >= start_dt3).values].reset_index(drop=True)

            last_nhnl = int(ns_all.iloc[-1])
            last_nh   = int(nhnl_df["new_highs"].iloc[-1])
            last_nl   = int(nhnl_df["new_lows"].iloc[-1])

            # 판정: 4주 MA 기울기
            lma = nma_all.iloc[-1]; pma = nma_all.iloc[-2] if len(nma_all) >= 2 else lma
            nhnl_ma_vals = nma_all.dropna()
            slope = np.polyfit(np.arange(len(nhnl_ma_vals)), nhnl_ma_vals.values, 1)[0] if len(nhnl_ma_vals) >= 2 else 0.0
            if pd.isna(lma):            nhnl_verdict, trend_color = "⚪ Not enough data",   "#757575"
            elif lma > 0 and lma > pma: nhnl_verdict, trend_color = "🟢 강세 상승",     "#2e7d32"
            elif lma > 0:               nhnl_verdict, trend_color = "🟡 강세 둔화",     "#f9a825"
            elif lma < 0 and lma < pma: nhnl_verdict, trend_color = "🔴 약세 하락",     "#c62828"
            else:                       nhnl_verdict, trend_color = "🟠 약세 회복 중",   "#ef6c00"

            h1, h2, h3, h4, h5 = st.columns(5)
            h1.metric("신고가 종목 수", f"{last_nh:,}")
            h2.metric("신저가 종목 수", f"{last_nl:,}")
            h3.metric("NH-NL",          f"{last_nhnl:+,}")
            h4.metric("4주 MA 기울기",  f"{slope:+.1f}/주")
            h5.metric("판정",            nhnl_verdict)

            # Index도 주간(FRI)으로 맞춰서 NH-NL과 같은 시간축으로 그림
            pf_idx3 = df.copy()
            pf_idx3["dt"] = pd.to_datetime(pf_idx3["date"].astype(str), format="%Y%m%d")
            pf_idx3 = pf_idx3[["dt", "close"]].sort_values("dt").set_index("dt")
            pf_idx3 = pf_idx3.resample("W-FRI").last().dropna().reset_index()
            pf_idx3 = pf_idx3[pf_idx3["dt"] >= start_dt3].copy().reset_index(drop=True)

            pf3 = pf3.sort_values("dt").reset_index(drop=True)

            # 2패널 — 위: 주간 Index / 아래: 주간 NH-NL
            fig_hl = _msp2(rows=2, cols=1, shared_xaxes=True,
                           row_heights=[0.5, 0.5], vertical_spacing=0.03)

            # 위 패널: Index(주간)
            fig_hl.add_trace(go.Scatter(
                x=pf_idx3["dt"], y=pf_idx3["close"].astype(float),
                line=dict(color="rgba(220,220,220,0.95)", width=1.8),
                name=f"{market} Index",
            ), row=1, col=1)

            # 아래 패널: NH-NL
            fig_hl.add_trace(go.Scatter(
                x=pf3["dt"], y=pf3["nhnl"].astype(float),
                line=dict(color="#26a69a", width=2.0),
                name="NH-NL",
            ), row=2, col=1)

            # 아래 패널: 기준선 0
            fig_hl.add_hline(
                y=0, line_color="rgba(180,180,180,0.7)", line_dash="dot",
                row=2, col=1
            )

            # 아래 패널: 4주 MA (임시 유지)
            fig_hl.add_trace(go.Scatter(
                x=pf3["dt"], y=nma_plot,
                line=dict(color="orange", width=1.8),
                name="4주 MA",
            ), row=2, col=1)

            # 0선 (아래 패널)
            fig_hl.add_hline(y=0, line_color="rgba(150,150,150,0.5)", line_dash="dot", row=2, col=1)

            fig_hl.update_layout(
                xaxis=dict(type="date"),
                xaxis2=dict(type="date"),
                template="plotly_dark", height=560,
                title=dict(text=f"{market} NH-NL — {nhnl_verdict}  (4주MA 기울기 {slope:+.1f}/주)",
                           font=dict(size=13, color=trend_color)),
                hovermode="x unified",
                hoverlabel=dict(bgcolor="#1e1e2e", font_color="white",
                               font_size=12, bordercolor="#444"),
                margin=dict(l=10, r=60, t=45, b=10),
                xaxis_rangeslider_visible=False,
                legend=dict(orientation="h", y=1.01),
                yaxis =dict(title="Index"),
                yaxis2=dict(title="NH-NL", zeroline=True,
                            zerolinecolor="rgba(150,150,150,0.4)"),
            )
            # 세로선: 두 패널 동시 관통
            fig_hl.update_traces(xaxis="x")
            fig_hl.update_xaxes(
                showspikes=True, spikemode="across", spikesnap="cursor",
                spikethickness=1, spikecolor="rgba(200,200,200,0.7)", spikedash="solid",
                tickformat="%Y/%m/%d", dtick=7*24*60*60*1000,
                tickangle=-45, tickfont=dict(size=8),
            )
            fig_hl.update_yaxes(showspikes=True, spikethickness=1,
                                spikecolor="rgba(200,200,200,0.4)")
            st.plotly_chart(fig_hl, width='stretch')

if __name__ == "__main__":
    main()
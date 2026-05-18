#!/usr/bin/env python3
"""
KOSPI / KOSDAQ Breadth Dashboard (Streamlit)
실행:
  pip install streamlit plotly pandas requests finance-datareader mplfinance matplotlib
  KRX_AUTH_KEY=your_key streamlit run kospi_breadth_dashboard.py
"""
from __future__ import annotations

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

# ── 한글 폰트 설정 ──
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
import pandas as pd
import plotly.graph_objects as go
import requests
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
# 설정
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
# 차트 — matplotlib (검증된 방식)
# ──────────────────────────────────────────────────────────────
def make_chart_img(df: pd.DataFrame, market: str, sig: dict,
                   chart_months: int) -> bytes:
    end_dt    = pd.to_datetime(df["date"].astype(str), format="%Y%m%d").max()
    start_dt  = end_dt - pd.DateOffset(months=chart_months)
    mask      = pd.to_datetime(df["date"].astype(str), format="%Y%m%d") >= start_dt
    pf        = df[mask].copy().reset_index(drop=True)
    pf["dt"]  = pd.to_datetime(pf["date"].astype(str), format="%Y%m%d")

    # mplfinance용 OHLC (date_num, open, high, low, close)
    ohlc = pf[["dt", "open", "high", "low", "close"]].copy()
    ohlc["dn"] = ohlc["dt"].map(mdates.date2num)
    ohlc_vals  = ohlc[["dn", "open", "high", "low", "close"]].values

    # High 기준일
    days_ago = int(sig["peak_label"].split(" days ago")[0]) if " days ago" in sig["peak_label"] else 0
    peak_dt  = pd.to_datetime(
        str(df["date"].iloc[-(days_ago + 1)]), format="%Y%m%d"
    )
    peak_dn  = mdates.date2num(peak_dt)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 9), sharex=True,
        gridspec_kw={"height_ratios": [1.4, 1]},   # 비율 균형 (비교하기 쉽도록)
        facecolor="#0e1117",
    )
    for ax in (ax1, ax2):
        ax.set_facecolor("#0e1117")
        ax.tick_params(colors="#aaaaaa")
        ax.spines[:].set_color("#333333")
        ax.yaxis.label.set_color("#aaaaaa")

    if MPL_OK:
        candlestick_ohlc(ax1, ohlc_vals, width=0.6,
                         colorup="#26a69a", colordown="#ef5350", alpha=0.9)
    else:
        ax1.plot(pf["dt"], pf["close"].astype(float), color="#26a69a", linewidth=1.5)

    ax1.set_title(f"{market} Index", color="#e0e0e0", fontsize=13)
    ax1.set_ylabel("Index", color="#aaaaaa")
    ax1.grid(True, color="#1e2530", linewidth=0.5)
    # 수직선: High 날짜
    ax1.axvline(peak_dn, color="orange", linestyle=":", linewidth=1.2, alpha=0.6)
    # 수평선: High 가격 — 캔들과 닿는 수준 확인용
    ax1.axhline(y=sig["price_high"], color="orange", linestyle="--",
                linewidth=1.2, alpha=0.8,
                label=f"Peak {sig['price_high']:,.2f}")
    ax1.legend(loc="upper left", fontsize=9,
               facecolor="#1a1a2e", labelcolor="#e0e0e0", framealpha=0.8)

    ax2.plot(pf["dt"], pf["ad_line"].astype(float),
             color="#1565c0", linewidth=1.8)
    ax2.set_ylabel("A/D Line", color="#aaaaaa")
    ax2.set_title("A/D Line", color="#e0e0e0", fontsize=11)
    ax2.grid(True, color="#1e2530", linewidth=0.5)
    # 수직선: High 날짜
    ax2.axvline(peak_dn, color="orange", linestyle=":", linewidth=1.2, alpha=0.6)
    # 수평선: High일 당시 A/D 값 — A/D선과 닿는 수준 확인용
    ax2.axhline(y=sig["ad_at_peak"], color="orange", linestyle="--",
                linewidth=1.2, alpha=0.8,
                label=f"A/D at Peak {sig['ad_at_peak']:,.0f}")
    ax2.legend(loc="upper left", fontsize=9,
               facecolor="#1a1a2e", labelcolor="#e0e0e0", framealpha=0.8)

    # x축 포맷
    locator   = mdates.AutoDateLocator()
    formatter = mdates.DateFormatter("%Y-%m")
    ax2.xaxis.set_major_locator(locator)
    ax2.xaxis.set_major_formatter(formatter)
    fig.autofmt_xdate(rotation=30, ha="right")

    # 판정 박스 — 영어로만 표시 (한글 폰트 없는 환경 대비)
    box_txt = (f"Peak: {sig['peak_label']}\n"
               f"Price vs Peak: {sig['price_off']:.2f}%\n"
               f"A/D vs Peak:   {sig['ad_off']:.2f}%\n"
               f"Gap:           {sig['gap']:.2f}%")
    ax1.text(0.01, 0.97, box_txt, transform=ax1.transAxes,
             va="top", ha="left", fontsize=10,
             color="white", family="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor=sig["color"], alpha=0.9))

    plt.tight_layout(pad=1.5)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()

# ──────────────────────────────────────────────────────────────
# 메인 앱
# ──────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="KOSPI Breadth Analysis Dashboard",
                       page_icon="📊", layout="wide")
    st.title("📊 국장 A/D Line 브레드스 대시보드")
    st.caption("A/D Line · Breadth Thrust · Price-Breadth Divergence")

    # ── 사이드바 ──────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")
        market = st.selectbox("Market", ["KOSPI", "KOSDAQ"])

        # 데이터 소스 선택
        mode = st.radio("데이터 소스", ["☁️ GitHub (빠름)", "🔑 KRX API (직접 수집)"],
                        index=0, help="GitHub: 미리 push된 CSV를 읽음 (빠름)\nKRX API: 직접 수집 (느림, AUTH_KEY 필요)")

        if mode == "🔑 KRX API (직접 수집)":
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

        fetch_btn = st.button("🔄 데이터 불러오기", type="primary", use_container_width=True)

        if mode == "🔑 KRX API (직접 수집)":
            st.caption("💡 새로 불러오고 싶으면 아래 캐시를 지우고 불러오세요.")

        st.divider()
        st.subheader("Analysis Parameters")
        lookback     = st.slider("Lookback (일)",      20, 252, 126)
        chart_months = st.slider("Chart Display Period (months)", 1,  24,  6)
        with st.expander("Threshold Settings"):
            price_thr  = st.number_input("Price Near-High Threshold (%)", value=2.0,  step=0.1)
            ad_thr     = st.number_input("A/D Near-High Threshold (%)",  value=3.0,  step=0.1)
            gap_warn   = st.number_input("Warning Divergence Threshold (%)",       value=1.5,  step=0.1)
            gap_danger = st.number_input("Severe Divergence Threshold (%)",       value=2.5,  step=0.1)

        if mode == "🔑 KRX API (직접 수집)":
            st.divider()
            st.subheader("💾 저장된 캐시")
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

    # ── 데이터 불러오기 ──────────────────────────────
    if not fetch_btn and "df_merged" not in st.session_state:
        st.info("👈 사이드바에서 마켓 선택 후 **데이터 불러오기** 버튼을 눌러주세요.")
        return

    if fetch_btn:
        if mode == "☁️ GitHub (빠름)":
            try:
                with st.spinner("GitHub에서 CSV 읽는 중…"):
                    df = load_from_github(market)
                st.success(f"✅ GitHub에서 로드 완료 — {len(df)}일치 / 최신: {df['date'].iloc[-1]}")
            except Exception as e:
                st.error(f"GitHub 로드 실패: {e}")
                return
        else:
            # KRX API 모드
            if not auth_key:
                st.error("Please enter your KRX AUTH_KEY.")
                return
            start_str = start_dt.strftime("%Y%m%d")
            end_str   = end_dt.strftime("%Y%m%d")
            cached = load_cache(market, start_str, end_str, 50000.0)
            if cached is not None:
                st.success(f"✅ 캐시에서 로드 ({market} {start_str}~{end_str})")
                df = cached
            else:
                try:
                    with st.spinner("Index OHLC Collecting..."):
                        index_df = fetch_index_ohlc(market, start_str, end_str)
                    breadth_df = build_breadth(auth_key, start_str, end_str, market, 50000.0)
                    df = breadth_df.merge(
                        index_df[["date","open","high","low","close"]],
                        on="date", how="inner"
                    ).sort_values("date").reset_index(drop=True)
                    save_cache(df, market, start_str, end_str, 50000.0)
                    st.success(f"✅ Collection completed — {len(df)}일치")
                except Exception as e:
                    st.error(f"Data collection failed: {e}")
                    return

        st.session_state["df_merged"] = df
        st.session_state["df_market"] = market

    # 마켓이 바뀌면 세션 초기화
    if st.session_state.get("df_market") != market:
        st.session_state.pop("df_merged", None)
        st.info("마켓이 변경됐어요. 데이터 불러오기를 다시 눌러주세요.")
        return

    # ── 차트 및 판정 출력 ───────────────────────────
    df = st.session_state["df_merged"]

    if len(df) < lookback:
        st.warning(f"Not enough data: {len(df)}행 (lookback={lookback})")
        return

    sig  = compute_signals(df, lookback, price_thr, ad_thr, gap_warn, gap_danger)
    last = df.iloc[-1]

    # ── 탭 구성 ──
    tab1, tab2, tab3, tab4 = st.tabs(["📈 A/D Line", "⚡ 모멘텀", "🏔 High-저점(NH-NL)", "📊 P/D 비율"])

    # ══════════════════════════════════════════════
    # TAB 1: 기존 A/D Line 분석
    # ══════════════════════════════════════════════
    with tab1:
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
            img = make_chart_img(df, market, sig, chart_months)
            st.image(img, use_container_width=True)
        except Exception as e:
            st.error(f"Chart rendering failed: {e}")

        with st.expander("📋 원시 데이터 보기"):
            show = df.copy()
            show["date"] = pd.to_datetime(show["date"].astype(str), format="%Y%m%d").dt.strftime("%Y-%m-%d")
            st.dataframe(
                show[["date","advances","declines","unchanged",
                      "ad_diff","ad_line","close","breadth_thrust_ema10"]]
                .sort_values("date", ascending=False).reset_index(drop=True),
                use_container_width=True,
            )
            csv = show.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
            st.download_button("📥 CSV 다운로드", csv,
                               f"{market}_breadth.csv", "text/csv")

    # ══════════════════════════════════════════════
    # TAB 2: 모멘텀 Index
    # ══════════════════════════════════════════════
    with tab2:
        st.subheader("⚡ 브레드스 모멘텀 Index")
        st.caption("등락종목수 단기MA - 장기MA 오실레이터. 0선 위 = 강세, 아래 = 약세")

        ma_fast = st.slider("단기 MA", 5, 30, 10, key="mom_fast")
        ma_slow = st.slider("장기 MA", 10, 60, 30, key="mom_slow")

        end_dt2   = pd.to_datetime(df["date"].astype(str), format="%Y%m%d").max()
        start_dt2 = end_dt2 - pd.DateOffset(months=chart_months)
        mask2 = pd.to_datetime(df["date"].astype(str), format="%Y%m%d") >= start_dt2
        pf2   = df[mask2].copy().reset_index(drop=True)
        pf2["dt"] = pd.to_datetime(pf2["date"].astype(str), format="%Y%m%d")

        ad_diff_s = pd.Series(df["ad_diff"].values.astype(float))
        maF = ad_diff_s.rolling(ma_fast).mean()
        maS = ad_diff_s.rolling(ma_slow).mean()
        momentum_full = maF - maS

        # 차트 구간
        mom_plot = momentum_full.iloc[mask2.values].reset_index(drop=True)
        signal_line = mom_plot.rolling(10).mean()

        last_mom = momentum_full.iloc[-1]
        last_sig = momentum_full.rolling(10).mean().iloc[-1]
        mom_verdict = ("🟢 강세" if last_mom > 0 and last_mom > last_sig else
                       "🟢 강세 유지" if last_mom > 0 else
                       "🔴 약세 전환" if last_mom < 0 and last_mom < last_sig else
                       "🔴 약세 유지")
        mom_color = "#00897b" if last_mom >= 0 else "#c62828"

        m1, m2, m3 = st.columns(3)
        m1.metric("모멘텀 (현재)", f"{last_mom:+.1f}")
        m2.metric("시그널 (10MA)", f"{last_sig:+.1f}")
        m3.metric("판정", mom_verdict)

        fig_mom = go.Figure()
        fig_mom.add_trace(go.Bar(
            x=pf2["dt"], y=mom_plot,
            marker_color=[("#26a69a" if v >= 0 else "#ef5350") for v in mom_plot],
            name="모멘텀", opacity=0.7
        ))
        fig_mom.add_trace(go.Scatter(
            x=pf2["dt"], y=signal_line,
            line=dict(color="orange", width=1.5),
            name="시그널(10MA)"
        ))
        fig_mom.add_hline(y=0, line_color="gray", line_dash="dot")
        fig_mom.update_layout(
            title=f"{market} 브레드스 모멘텀 ({ma_fast}MA - {ma_slow}MA)",
            template="plotly_dark", height=400,
            legend=dict(orientation="h", y=1.05)
        )
        st.plotly_chart(fig_mom, use_container_width=True)

    # ══════════════════════════════════════════════
    # TAB 3: High-저점 수치 (NH-NL)
    # ══════════════════════════════════════════════
    with tab3:
        st.subheader("🏔 High-저점 수치 (신고가 - 신저가)")
        st.caption("당일 52주 신고가 종목수 - 신저가 종목수. 플러스 유지 = 시장 건강")

        # 국장은 별도로 신고가/신저가 데이터가 없으므로
        # 대안: lookback일 신고가/신저가 종목 수를 A/D 데이터에서 추정
        # → 실제로는 closes 기준으로 rolling window 내 신고/신저 여부 계산 불가 (종목별 데이터 없음)
        # → 대신 High-저점 개념을 A/D 관점으로 재해석: 상승우세일 - 하락우세일 누적
        st.info("💡 국장 신고가/신저가 개별 종목 데이터는 KRX API에서 직접 제공되지 않습니다.\n"
                "대신 **등락종목수 기반 High-저점 대용 지표**를 표시합니다.")

        hl_window = st.slider("집계 기간 (일)", 10, 60, 20, key="hl_win")
        adv_s  = pd.Series(df["advances"].values.astype(float))
        decl_s = pd.Series(df["declines"].values.astype(float))

        # 상승우세일: advances > declines 인 날 수 - 하락우세일 수 (rolling)
        bull_days = (adv_s > decl_s).astype(int)
        bear_days = (adv_s < decl_s).astype(int)
        hl_proxy  = bull_days.rolling(hl_window).sum() - bear_days.rolling(hl_window).sum()
        hl_ma     = hl_proxy.rolling(5).mean()

        end_dt3   = pd.to_datetime(df["date"].astype(str), format="%Y%m%d").max()
        start_dt3 = end_dt3 - pd.DateOffset(months=chart_months)
        mask3 = pd.to_datetime(df["date"].astype(str), format="%Y%m%d") >= start_dt3
        pf3   = df[mask3].copy().reset_index(drop=True)
        pf3["dt"] = pd.to_datetime(pf3["date"].astype(str), format="%Y%m%d")
        hl_plot   = hl_proxy.iloc[mask3.values].reset_index(drop=True)
        hl_ma_plot = hl_ma.iloc[mask3.values].reset_index(drop=True)

        last_hl = hl_proxy.iloc[-1]
        hl_verdict = ("🟢 상승 우세 유지" if last_hl > hl_window * 0.3 else
                      "🟢 소폭 우세"       if last_hl > 0 else
                      "🔴 하락 우세"       if last_hl < -hl_window * 0.3 else
                      "🟠 소폭 약세")

        h1, h2 = st.columns(2)
        h1.metric(f"상승우세일 비율 ({hl_window}일)", f"{last_hl:+.0f}일")
        h2.metric("판정", hl_verdict)

        fig_hl = go.Figure()
        fig_hl.add_trace(go.Bar(
            x=pf3["dt"], y=hl_plot,
            marker_color=[("#26a69a" if v >= 0 else "#ef5350") for v in hl_plot],
            name=f"상승-하락우세일 ({hl_window}일)", opacity=0.7
        ))
        fig_hl.add_trace(go.Scatter(
            x=pf3["dt"], y=hl_ma_plot,
            line=dict(color="orange", width=1.5),
            name="5일 평균"
        ))
        fig_hl.add_hline(y=0, line_color="gray", line_dash="dot")
        fig_hl.update_layout(
            title=f"{market} High-저점 대용 지표 ({hl_window}일 집계)",
            template="plotly_dark", height=400,
        )
        st.plotly_chart(fig_hl, use_container_width=True)

    # ══════════════════════════════════════════════
    # TAB 4: P/D 비율
    # ══════════════════════════════════════════════
    with tab4:
        st.subheader("📊 P/D 비율 (상승 / 하락 종목수)")
        st.caption("2.0 이상 = 단기 과열 주의 / 0.5 이하 = 과매도 반등 가능 / 1.0 = 기준선")

        pd_ob = st.slider("과열 기준", 1.5, 3.0, 2.0, step=0.1, key="pd_ob")
        pd_os = st.slider("과매도 기준", 0.2, 1.0, 0.5, step=0.1, key="pd_os")
        pd_ma = st.slider("MA 기간", 3, 20, 10, key="pd_ma")

        adv_s2  = pd.Series(df["advances"].values.astype(float))
        decl_s2 = pd.Series(df["declines"].values.astype(float))
        pd_ratio_full = adv_s2 / decl_s2.replace(0, float("nan"))
        pd_ma_full    = pd_ratio_full.rolling(pd_ma).mean()

        end_dt4   = pd.to_datetime(df["date"].astype(str), format="%Y%m%d").max()
        start_dt4 = end_dt4 - pd.DateOffset(months=chart_months)
        mask4 = pd.to_datetime(df["date"].astype(str), format="%Y%m%d") >= start_dt4
        pf4   = df[mask4].copy().reset_index(drop=True)
        pf4["dt"] = pd.to_datetime(pf4["date"].astype(str), format="%Y%m%d")
        pd_plot    = pd_ratio_full.iloc[mask4.values].reset_index(drop=True)
        pd_ma_plot = pd_ma_full.iloc[mask4.values].reset_index(drop=True)

        last_pd = pd_ratio_full.iloc[-1]
        last_pd_ma = pd_ma_full.iloc[-1]
        pd_verdict = ("🔴 과열 — 단기 조정 주의" if last_pd >= pd_ob else
                      "🟢 과매도 — 반등 가능"     if last_pd <= pd_os else
                      "🟢 강세"                   if last_pd > 1.5 else
                      "🟢 양호"                   if last_pd > 1.0 else
                      "🟠 약세"                   if last_pd < 0.7 else "⚪ 중립")
        pd_color = ("#c62828" if last_pd >= pd_ob else
                    "#00897b" if last_pd <= pd_os else
                    "#2e7d32" if last_pd > 1.0 else "#ef6c00")

        p1, p2, p3, p4 = st.columns(4)
        p1.metric("상승 종목수", f"{int(last['advances']):,}")
        p2.metric("하락 종목수", f"{int(last['declines']):,}")
        p3.metric("P/D 비율",   f"{last_pd:.2f}")
        p4.metric("판정", pd_verdict)

        # 라인 차트
        pd_colors = []
        for v in pd_plot:
            if v >= pd_ob:     pd_colors.append("#ef5350")
            elif v <= pd_os:   pd_colors.append("#26a69a")
            elif v >= 1.0:     pd_colors.append("#66bb6a")
            else:              pd_colors.append("#ffa726")

        fig_pd = go.Figure()
        fig_pd.add_hrect(y0=pd_ob, y1=pd_plot.max(skipna=True) + 0.5,
                         fillcolor="red", opacity=0.05, line_width=0,
                         annotation_text="과열 구간", annotation_position="top left")
        fig_pd.add_hrect(y0=0, y1=pd_os,
                         fillcolor="teal", opacity=0.05, line_width=0,
                         annotation_text="과매도 구간", annotation_position="bottom left")
        fig_pd.add_trace(go.Scatter(
            x=pf4["dt"], y=pd_plot,
            line=dict(color="#42a5f5", width=2),
            name="P/D 비율"
        ))
        fig_pd.add_trace(go.Scatter(
            x=pf4["dt"], y=pd_ma_plot,
            line=dict(color="orange", width=1.5, dash="dash"),
            name=f"P/D {pd_ma}MA"
        ))
        fig_pd.add_hline(y=1.0,   line_color="gray",  line_dash="dot",
                         annotation_text="기준(1.0)")
        fig_pd.add_hline(y=pd_ob, line_color="red",   line_dash="dash",
                         annotation_text=f"과열({pd_ob})")
        fig_pd.add_hline(y=pd_os, line_color="teal",  line_dash="dash",
                         annotation_text=f"과매도({pd_os})")
        fig_pd.update_layout(
            title=f"{market} P/D 비율",
            template="plotly_dark", height=400,
            legend=dict(orientation="h", y=1.05),
            yaxis=dict(rangemode="tozero")
        )
        st.plotly_chart(fig_pd, use_container_width=True)


if __name__ == "__main__":
    main()
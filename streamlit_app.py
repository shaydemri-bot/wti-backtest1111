
# streamlit_app.py — WTI Strategy Lab v2 (Optimizer + Pro Metrics)
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="WTI Strategy Lab v2", layout="wide")

# ---------- UI helpers ----------
def help_popover(text: str):
    if not text:
        return
    with st.popover("ℹ️", use_container_width=False):
        st.write(text)

def sb_number(label, value, min_value=None, max_value=None, step=None, help_text=""):
    c1, c2 = st.sidebar.columns([6,1])
    with c1:
        v = st.number_input(label, value=value, min_value=min_value, max_value=max_value, step=step, key=f"num_{label}")
    with c2:
        help_popover(help_text)
    return v

def sb_selectbox(label, options, index=0, help_text=""):
    c1, c2 = st.sidebar.columns([6,1])
    with c1:
        v = st.selectbox(label, options, index=index, key=f"sel_{label}")
    with c2:
        help_popover(help_text)
    return v

def sb_checkbox(label, value=False, help_text=""):
    c1, c2 = st.sidebar.columns([6,1])
    with c1:
        v = st.checkbox(label, value=value, key=f"chk_{label}")
    with c2:
        help_popover(help_text)
    return v

def sb_slider(label, min_value, max_value, value, step=None, help_text=""):
    c1, c2 = st.sidebar.columns([6,1])
    with c1:
        v = st.slider(label, min_value=min_value, max_value=max_value, value=value, step=step, key=f"sld_{label}")
    with c2:
        help_popover(help_text)
    return v

def ensure_series(x):
    if isinstance(x, pd.DataFrame):
        return x.iloc[:, 0] if x.shape[1] >= 1 else pd.Series([], dtype=float)
    if not isinstance(x, pd.Series):
        return pd.Series(x)
    return x

# ---------- Indicators ----------
def ema(s, n):
    s = ensure_series(s).astype(float)
    return s.ewm(span=n, adjust=False).mean()

def rsi(close, n=14):
    close = ensure_series(close).astype(float)
    diff = close.diff()
    up = np.where(diff > 0, diff, 0.0)
    dn = np.where(diff < 0, -diff, 0.0)
    roll_up = pd.Series(up, index=close.index).rolling(n).mean()
    roll_dn = pd.Series(dn, index=close.index).rolling(n).mean()
    rs = roll_up / (roll_dn.replace(0, np.nan))
    return (100 - (100 / (1 + rs))).fillna(50)

def bollinger(close, n=20, k=2):
    close = ensure_series(close).astype(float)
    ma = close.rolling(n).mean()
    sd = close.rolling(n).std(ddof=0)
    return ma, ma + k*sd, ma - k*sd

def atr(df, n=14):
    high = ensure_series(df["High"]).astype(float)
    low  = ensure_series(df["Low"]).astype(float)
    close = ensure_series(df["Close"]).astype(float)
    prev_c = close.shift(1)
    tr = pd.concat([(high-low), (high-prev_c).abs(), (low-prev_c).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def volume_zscore(vol, n=50):
    m = vol.rolling(n).mean()
    s = vol.rolling(n).std(ddof=0)
    return (vol - m) / s

# ---------- Strategy & Backtest ----------
def run_backtest(df, params):
    rsi_len = params["rsi_len"]
    ema_fast = params["ema_fast"]
    ema_slow = params["ema_slow"]
    atr_len = params["atr_len"]
    atr_sl = params["atr_sl"]
    atr_tp = params["atr_tp"]
    trend_filter = params["trend_filter"]
    use_vol_spike = params["use_vol_spike"]
    vol_z_thr = params["vol_z_thr"]
    capital = params["capital"]
    risk_pct = params["risk_pct"]
    contract_size = params["contract_size"]
    leverage = params["leverage"]
    commission = params["commission"]
    spread = params["spread"]
    bars_timeout = params["bars_timeout"]
    usd_ils = params["usd_ils"]

    df = df.copy()
    df["EMA_FAST"] = ema(df["Close"], ema_fast)
    df["EMA_SLOW"] = ema(df["Close"], ema_slow)
    df["RSI"]      = rsi(df["Close"], rsi_len)
    df["ATR"]      = atr(df, atr_len)
    if "Volume" in df.columns:
        df["VOL_Z"] = volume_zscore(df["Volume"], 50)
    else:
        df["Volume"]=np.nan; df["VOL_Z"]=np.nan

    r = df["RSI"]
    cross_up30 = (r.shift(1) <= 30) & (r > 30)
    cross_dn70 = (r.shift(1) >= 70) & (r < 70)

    pos=None; entry_px=None; entry_ts=None; entry_bar=None; sl=None; tp=None; qty=0
    trades=[]; df["marker_px"]=np.nan; df["marker_text"]=None; df["marker_side"]=None

    for i,(ts,row) in enumerate(df.iterrows()):
        px=float(row["Close"]); high=float(row["High"]); low=float(row["Low"])
        if pos is None:
            ok_vol=True
            if use_vol_spike and not np.isnan(row["VOL_Z"]):
                ok_vol = row["VOL_Z"] >= vol_z_thr
            if cross_up30.loc[ts] and px>row["EMA_FAST"] and (not trend_filter or px>=row["EMA_SLOW"]) and ok_vol:
                stop_dist=atr_sl*float(row["ATR"]); risk_per_unit=stop_dist*contract_size; cash_risk=capital*(risk_pct/100.0)
                qty=max(1, int(cash_risk / max(risk_per_unit,1e-8))); pos,entry_px,entry_ts,entry_bar="long",px,ts,i
                sl=entry_px-stop_dist; tp=entry_px+atr_tp*float(row["ATR"])
                df.at[ts,"marker_px"]=px; df.at[ts,"marker_text"]=f"Long | qty={qty}"; df.at[ts,"marker_side"]="long"
            elif cross_dn70.loc[ts] and px<row["EMA_FAST"] and (not trend_filter or px<=row["EMA_SLOW"]) and ok_vol:
                stop_dist=atr_sl*float(row["ATR"]); risk_per_unit=stop_dist*contract_size; cash_risk=capital*(risk_pct/100.0)
                qty=max(1, int(cash_risk / max(risk_per_unit,1e-8))); pos,entry_px,entry_ts,entry_bar="short",px,ts,i
                sl=entry_px+stop_dist; tp=entry_px-atr_tp*float(row["ATR"])
                df.at[ts,"marker_px"]=px; df.at[ts,"marker_text"]=f"Short | qty={qty}"; df.at[ts,"marker_side"]="short"
        else:
            exit_reason=None; exit_px=None
            if pos=="long":
                if low<=sl: exit_px,exit_reason=sl,"SL"
                elif high>=tp: exit_px,exit_reason=tp,"TP"
                elif cross_dn70.loc[ts] or px<row["EMA_FAST"]: exit_px,exit_reason=px,"Rule"
            else:
                if high>=sl: exit_px,exit_reason=sl,"SL"
                elif low<=tp: exit_px,exit_reason=tp,"TP"
                elif cross_up30.loc[ts] or px>row["EMA_FAST"]: exit_px,exit_reason=px,"Rule"

            if exit_px is None and bars_timeout and (i-entry_bar)>=bars_timeout:
                exit_px,exit_reason=px,"Time"

            if exit_px is not None:
                per_unit_cost = spread + (commission/max(qty,1)/contract_size)
                unit_pnl = (exit_px-entry_px) if pos=="long" else (entry_px-exit_px)
                unit_pnl -= per_unit_cost
                usd = unit_pnl*qty*contract_size*leverage
                ils = usd*usd_ils
                trades.append({
                    "entry_time":entry_ts,"side":pos,"entry":entry_px,"sl":sl,"tp":tp,
                    "exit_time":ts,"exit":exit_px,"reason":exit_reason,"qty":qty,
                    "unit_pnl":unit_pnl,"pnl_usd":usd,"pnl_ils":ils
                })
                pos=entry_px=entry_ts=sl=tp=None; entry_bar=None; qty=0

    return df, pd.DataFrame(trades)

def max_drawdown(equity: pd.Series):
    if equity.empty: return 0.0, pd.NaT, pd.NaT
    roll_max = equity.cummax()
    dd = (equity - roll_max)
    min_dd = dd.min()
    end = dd.idxmin() if not dd.empty else pd.NaT
    start = equity.loc[:end].idxmax() if end is not pd.NaT else pd.NaT
    return float(min_dd), start, end

def perf_metrics(trades: pd.DataFrame):
    if trades.empty:
        return {
            "trades": 0, "win_rate": 0.0, "profit_factor": 0.0, "max_dd": 0.0,
            "sharpe": 0.0, "expectancy": 0.0, "avg_trade": 0.0, "total_usd": 0.0, "total_ils": 0.0,
            "exposure": 0.0
        }, pd.Series(dtype=float)

    eq = trades[["exit_time","pnl_usd"]].copy().set_index("exit_time").sort_index()
    eq["equity"] = eq["pnl_usd"].cumsum()

    wins = trades.loc[trades["pnl_usd"]>0, "pnl_usd"].sum()
    losses = -trades.loc[trades["pnl_usd"]<0, "pnl_usd"].sum()
    profit_factor = float(wins / losses) if losses > 0 else float("inf")

    win_rate = float((trades["pnl_usd"]>0).mean())
    expectancy = float(trades["pnl_usd"].mean())
    avg_trade = expectancy
    total_usd = float(trades["pnl_usd"].sum())
    total_ils = float(trades["pnl_ils"].sum())

    std = float(trades["pnl_usd"].std(ddof=0) or 1e-12)
    sharpe = float(expectancy / std)

    exposure = min(1.0, len(trades) / max(1, len(eq)))  # הערכה גסה

    mdd, _, _ = max_drawdown(eq["equity"])

    return {
        "trades": len(trades),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "max_dd": mdd,
        "sharpe": sharpe,
        "expectancy": expectancy,
        "avg_trade": avg_trade,
        "total_usd": total_usd,
        "total_ils": total_ils,
        "exposure": exposure
    }, eq["equity"]

# ---------- Sidebar controls ----------
st.title("📈 WTI (CL=F) — Strategy Lab v2")

st.sidebar.header("Data")
period = sb_selectbox("Period", ["7d","1mo","3mo","6mo","1y"], index=2, help_text="כמה היסטוריה להוריד.")
interval = sb_selectbox("Interval", ["15m","30m","1h","4h","1d"], index=2, help_text="גודל הנרות.")
ticker = st.sidebar.text_input("Ticker", "CL=F")

st.sidebar.header("Indicators")
ema_fast = sb_number("EMA fast", 20, 5, 200, 1, "ממוצע נע מהיר. ערך קטן = מגיב מהר (יותר אותות).")
ema_slow = sb_number("EMA slow", 50, 10, 300, 1, "ממוצע נע איטי. ערך גדול = טרנד 'נקי' יותר.")
rsi_len  = sb_number("RSI length", 14, 5, 50, 1, "אורך RSI. קטן יותר = רגיש יותר.")
atr_len  = sb_number("ATR length", 14, 5, 50, 1, "חישוב תנודתיות.")

st.sidebar.header("Entries")
trend_filter = sb_checkbox("Filter by EMA slow (avoid counter-trend)", True, "לא נכנסים נגד הטרנד האיטי")
use_vol_spike = sb_checkbox("Require Volume Spike", False, "דורש חריגת נפח")
vol_z_thr = sb_slider("Volume Z-score threshold", -1.0, 5.0, 1.5, 0.1, "סף חריגה")

st.sidebar.header("Exits (Risk)")
atr_sl = sb_number("SL = ATR ×", 1.5, 0.1, 10.0, 0.1, "סטופ ATR (גדול = רחוק יותר)")
atr_tp = sb_number("TP = ATR ×", 2.5, 0.1, 20.0, 0.1, "טייק ATR (גדול = יעד רחוק יותר)")
bars_timeout = sb_number("Time Exit (bars)", 60, 0, 1000, 5, "יציאה כפויה אחרי X נרות.")

st.sidebar.header("Position & Costs")
capital = sb_number("Account capital (USD)", 10_000, 100, 10_000_000, 100, "הון חשבון.")
risk_pct = sb_slider("Risk per trade (%)", 0.1, 5.0, 0.5, 0.1, "אחוז סיכון לטרייד.")
contract_size = sb_number("Units per contract (barrels)", 100, 1, 10_000, 1, "חביות בחוזה.")
leverage = sb_number("Leverage (simulation)", 20, 1, 100, 1, "מינוף סימולציה.")
commission = sb_number("Commission per trade (USD)", 1.0, 0.0, 50.0, 0.5, "עמלה קבועה.")
spread = sb_number("Spread/Slippage (USD per unit)", 0.02, 0.0, 2.0, 0.01, "ספרד/סליפג' ליחידה.")
usd_ils = sb_number("USD→ILS rate (approx.)", 3.7, 2.5, 6.0, 0.01, "שער המרה לש\"ח.")

# Optimizer
st.sidebar.header("Optimizer (Grid)")
opt_toggle = sb_checkbox("Run parameter grid", False, "מריץ כמה קומבינציות ומשווה ביצועים.")
opt_ema_fast = sb_slider("EMA fast range", 10, 60, (20,30), 1, "טווח לסריקה.")
opt_atr_tp = sb_slider("TP×ATR range", 1, 6, (2,3), 1, "טווח יעדי ATR.")

# ---------- Data ----------
raw = yf.download(ticker, period=period, interval=interval, auto_adjust=False)
if raw.empty:
    st.error("No data from yfinance. נסה Period/Interval אחרים.")
    st.stop()
if isinstance(raw.columns, pd.MultiIndex):
    raw.columns = raw.columns.get_level_values(0)
required = ["Open","High","Low","Close"]
missing = [c for c in required if c not in raw.columns]
if missing:
    st.error(f"Missing columns: {missing}")
    st.write("Columns:", list(raw.columns))
    st.stop()
df = raw.copy()

# ---------- Single-run (current params) ----------
params = dict(
    rsi_len=int(rsi_len), ema_fast=int(ema_fast), ema_slow=int(ema_slow), atr_len=int(atr_len),
    atr_sl=float(atr_sl), atr_tp=float(atr_tp), trend_filter=trend_filter, use_vol_spike=use_vol_spike, vol_z_thr=float(vol_z_thr),
    capital=float(capital), risk_pct=float(risk_pct), contract_size=int(contract_size), leverage=int(leverage),
    commission=float(commission), spread=float(spread), bars_timeout=int(bars_timeout), usd_ils=float(usd_ils)
)

df_ind, trades = run_backtest(df, params)

# ---------- Metrics ----------
def build_metrics(trades: pd.DataFrame):
    m, equity = perf_metrics(trades)
    return m, equity

metrics, equity = build_metrics(trades)

# ---------- Charts (price + overlays) ----------
x = df_ind.index
fig = make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.03)
fig.add_trace(go.Candlestick(x=x, open=df_ind["Open"], high=df_ind["High"], low=df_ind["Low"], close=df_ind["Close"],
                             name=ticker, opacity=0.9, increasing_line_width=1.2, decreasing_line_width=1.2), 1, 1)

show_overlays = st.sidebar.multiselect("Overlays visibility",
    ["EMA fast","EMA slow","Bollinger Bands","Entries"],
    default=["EMA fast","EMA slow","Bollinger Bands","Entries"])

if "EMA fast" in show_overlays:
    fig.add_trace(go.Scatter(x=x, y=df_ind["EMA_FAST"], name=f"EMA{int(ema_fast)}", mode="lines"), 1, 1)
if "EMA slow" in show_overlays:
    fig.add_trace(go.Scatter(x=x, y=df_ind["EMA_SLOW"], name=f"EMA{int(ema_slow)}", mode="lines"), 1, 1)
ma, up, dn = bollinger(df_ind["Close"], 20, 2)
if "Bollinger Bands" in show_overlays:
    fig.add_trace(go.Scatter(x=x, y=up, name="BB Upper", mode="lines"), 1, 1)
    fig.add_trace(go.Scatter(x=x, y=ma, name="BB MA", mode="lines"), 1, 1)
    fig.add_trace(go.Scatter(x=x, y=dn, name="BB Lower", mode="lines"), 1, 1)

if "Entries" in show_overlays:
    mk = df_ind.dropna(subset=["marker_px"])
    if not mk.empty:
        colors = np.where(mk["marker_side"]=="long", "green", "red")
        fig.add_trace(go.Scatter(x=mk.index, y=mk["marker_px"], mode="markers+text", name="Signals",
                                 text=mk["marker_text"], textposition="top center",
                                 marker_symbol="diamond", marker_size=10, marker_color=colors), 1, 1)

fig.update_layout(template="simple_white", height=680, xaxis_rangeslider_visible=False,
                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                  margin=dict(l=20,r=20,t=30,b=10))

# ---------- Tabs ----------
tab_chart, tab_rsi, tab_equity, tab_breakdown, tab_optimizer, tab_help = st.tabs(
    ["Chart","RSI","Equity","Breakdown","Optimizer","Help"]
)

with tab_chart:
    st.plotly_chart(fig, use_container_width=True)

with tab_rsi:
    rfig = go.Figure()
    rfig.add_trace(go.Scatter(x=x, y=df_ind["RSI"], name="RSI", mode="lines"))
    rfig.add_hline(y=70, line_dash="dash"); rfig.add_hline(y=30, line_dash="dash")
    rfig.update_layout(template="simple_white", height=220, xaxis_rangeslider_visible=False, margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(rfig, use_container_width=True)

with tab_equity:
    if not equity.empty:
        efig = go.Figure()
        efig.add_trace(go.Scatter(x=equity.index, y=equity, name="Equity", mode="lines"))
        efig.update_layout(template="simple_white", height=240, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(efig, use_container_width=True)
    else:
        st.info("No closed trades yet for equity curve.")

with tab_breakdown:
    left, right = st.columns([3,2])
    with left:
        st.subheader("Performance")
        m = metrics
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Trades", f"{m['trades']}")
        c2.metric("Win-Rate", f"{m['win_rate']:.1%}")
        c3.metric("Profit Factor", f"{m['profit_factor']:.2f}" if np.isfinite(m['profit_factor']) else "∞")
        c4.metric("Sharpe (per trade)", f"{m['sharpe']:.2f}")
        c5,c6,c7,c8 = st.columns(4)
        c5.metric("Total PnL (USD)", f"{m['total_usd']:,.2f}")
        c6.metric("Total PnL (ILS)", f"{m['total_ils']:,.2f}")
        c7.metric("Max Drawdown (USD)", f"{m['max_dd']:,.2f}")
        c8.metric("Expectancy / trade", f"{m['expectancy']:,.2f}")

    with right:
        st.subheader("Trades")
        if not trades.empty:
            st.dataframe(trades, use_container_width=True, hide_index=True)
            st.download_button("⬇️ Download trades CSV", trades.to_csv(index=False).encode("utf-8"), "trades.csv", "text/csv")
        else:
            st.info("No trades for the chosen parameters.")

    if not trades.empty:
        tdf = trades.copy()
        tdf["hour"] = pd.to_datetime(tdf["entry_time"]).dt.hour
        tdf["weekday"] = pd.to_datetime(tdf["entry_time"]).dt.day_name()
        g_hour = tdf.groupby("hour")["pnl_usd"].agg(["count","sum","mean"]).rename(columns={"count":"trades","sum":"total","mean":"avg"})
        g_wday = tdf.groupby("weekday")["pnl_usd"].agg(["count","sum","mean"]).rename(columns={"count":"trades","sum":"total","mean":"avg"})
        st.markdown("**By Hour (entry)**")
        st.dataframe(g_hour, use_container_width=True)
        st.markdown("**By Weekday (entry)**")
        st.dataframe(g_wday, use_container_width=True)

with tab_optimizer:
    st.write("Run a small grid over **EMA fast** and **TP×ATR** and compare results.")
    if opt_toggle:
        lo_f, hi_f = opt_ema_fast
        lo_tp, hi_tp = opt_atr_tp
        rows = []
        for f in range(int(lo_f), int(hi_f)+1):
            for tp in range(int(lo_tp), int(hi_tp)+1):
                p = dict(
                    rsi_len=int(rsi_len), ema_fast=int(f), ema_slow=int(ema_slow), atr_len=int(atr_len),
                    atr_sl=float(atr_sl), atr_tp=float(tp), trend_filter=trend_filter, use_vol_spike=use_vol_spike, vol_z_thr=float(vol_z_thr),
                    capital=float(capital), risk_pct=float(risk_pct), contract_size=int(contract_size), leverage=int(leverage),
                    commission=float(commission), spread=float(spread), bars_timeout=int(bars_timeout), usd_ils=float(usd_ils)
                )
                _, tr = run_backtest(df, p)
                met, _ = perf_metrics(tr)
                rows.append({
                    "ema_fast": f, "atr_tp": tp,
                    "trades": met["trades"],
                    "win_rate": met["win_rate"],
                    "profit_factor": met["profit_factor"],
                    "total_usd": met["total_usd"],
                    "sharpe": met["sharpe"]
                })
        opt_df = pd.DataFrame(rows).sort_values(["total_usd","profit_factor","sharpe"], ascending=[False, False, False])
        st.dataframe(opt_df, use_container_width=True, hide_index=True)
        if not opt_df.empty:
            best = opt_df.iloc[0].to_dict()
            st.success(f"Best: EMA fast={best['ema_fast']}, TP×ATR={best['atr_tp']} | PnL ${best['total_usd']:,.0f}, PF {best['profit_factor']:.2f}, Sharpe {best['sharpe']:.2f}")

with tab_help:
    st.markdown(
        """
        ### מה יש כאן?
        - מנוע Backtest עם RSI-cross + EMA + ATR, כולל עלויות וספייק נפח (אופציונלי).
        - מדדי ביצועים: Win-Rate, Profit Factor, Max Drawdown, Sharpe (לפי טרייד), Expectancy.
        - פילוח לפי שעה/יום, טבלת טריידים והורדת CSV.
        - Optimizer בסיסי לסריקת פרמטרים.

        **טיפ:** אם אין כמעט טריידים, הקטן TP×ATR/SL×ATR, קיצור RSI length או העלה את התקופה.
        """
    )

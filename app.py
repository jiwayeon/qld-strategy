"""QLD moving-average strategy dashboard. Educational/research only."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="QLD 전략 대시보드", page_icon="📈", layout="wide")
st.markdown("""<style>
.block-container {padding-top: 1.7rem; max-width: 1250px}
div[data-testid="stMetric"] {background: #162238; border: 1px solid #2b3951; padding: 18px; border-radius: 14px}
.sub {color: #94a3b8; font-size: 0.91rem}
</style>""", unsafe_allow_html=True)

STATES = [
    ("SMA5", "EMA20", "SMA180"),
    ("EMA20", "SMA5", "SMA180"),
    ("SMA5", "SMA180", "EMA20"),
    ("EMA20", "SMA180", "SMA5"),
    ("SMA180", "SMA5", "EMA20"),
    ("SMA180", "EMA20", "SMA5"),
]
DEFAULTS = [0.7, 1, 0.4, 0, 0.4, 0.1]

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_prices(ticker: str, start: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, auto_adjust=True, progress=False, threads=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.empty or "Close" not in df:
        raise ValueError("주가 데이터가 비어 있습니다.")
    df.index = pd.to_datetime(df.index).tz_localize(None) if df.index.tz is not None else pd.to_datetime(df.index)
    return df[["Close"]].dropna().sort_index()


def indicators(df, ema_type, long_days):
    d = df.copy()
    d["SMA5"] = d.Close.rolling(5).mean()
    # Keep both options explicit: source code's com=20 is mathematically EMA41.
    d["EMA20"] = d.Close.ewm(com=20, adjust=False).mean() if ema_type == "원본 com=20 (EMA41)" else d.Close.ewm(span=20, adjust=False).mean()
    d["SMA180"] = d.Close.rolling(long_days).mean()
    d = d.dropna().copy()
    d["state"] = [tuple(sorted(("SMA5", "EMA20", "SMA180"), key=lambda k: (-row[k], k))) for _, row in d.iterrows()]
    return d


def performance(equity):
    if len(equity) < 2: return np.nan, np.nan
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = equity.iloc[-1] ** (1 / years) - 1 if years > 0 and equity.iloc[-1] > 0 else np.nan
    mdd = (equity / equity.cummax() - 1).min()
    return cagr, mdd


def backtest(daily, weights, cost_rate, cash_rate):
    """Signals at Friday close; next week's trading approximated by previous week's Friday close.
    Exclude incomplete week. Applies desired exposure at weekly boundary, accounts for
    portfolio drift and one-way proportional turnover cost.
    """
    wk = daily.resample("W-FRI").last().dropna(subset=["Close", "state"])
    # Drop any week whose nominal Friday is beyond latest actual observation.
    wk = wk.loc[wk.index <= daily.index[-1]]
    if len(wk) < 3: raise ValueError("백테스트에 필요한 데이터가 부족합니다.")
    r = wk.Close.pct_change()
    # Baseline begins at first completed weekly close (100% QLD).
    initial_date = wk.index[0]
    equity_s = [1.0]
    equity_b = [1.0]
    prev_risky_w = 0.0  # before initial investment
    change_count, turnover_sum = 0, 0.0
    rows = []
    # At each week start use the immediately previous completed Friday's signal.
    for i in range(1, len(wk)):
        prior_state = wk.iloc[i - 1]["state"]
        target = weights[prior_state]
        turnover = abs(target - prev_risky_w)
        if turnover > 1e-8: change_count += 1
        turnover_sum += turnover
        fee = cost_rate * turnover
        weekly_r = float(r.iloc[i])
        cash_week = (1 + cash_rate) ** (7 / 365.25) - 1
        before = equity_s[-1]
        after_fee = before * (1 - fee)
        growth = target * (1 + weekly_r) + (1 - target) * (1 + cash_week)
        new_val = after_fee * growth
        equity_s.append(new_val)
        equity_b.append(equity_b[-1] * (1 + weekly_r))
        prev_risky_w = target * (1 + weekly_r) / growth if growth > 0 else 0.0
        rows.append({"날짜": wk.index[i], "전주 신호": " > ".join(prior_state), "목표비중": target, "QLD 주간수익률": weekly_r, "매매회전율": turnover, "전략자산": new_val, "단순보유자산": equity_b[-1]})
    curve = pd.DataFrame({"전략": equity_s, "QLD 장기보유": equity_b}, index=wk.index)
    trades = pd.DataFrame(rows).set_index("날짜")
    # A daily CLOSE mark-to-market curve avoids obscuring intrawEEK drawdowns.
    # Weekly simulation equity remains authoritative; derive daily proxy, recompute
    # actual equity with each week's preceding signal, week-start closing anchor.
    daily_equity = [1.0]
    dates = [initial_date]
    cash_day = (1 + cash_rate) ** (1 / 365.25) - 1
    equity = 1.0
    risky_w = 0.0
    for i in range(1, len(wk)):
        start, end = wk.index[i-1], wk.index[i]
        target = weights[wk.iloc[i-1]["state"]]
        equity *= 1 - cost_rate * abs(target - risky_w)
        prev_price = float(wk.iloc[i-1].Close)
        held = daily.loc[(daily.index > start) & (daily.index <= end), "Close"]
        q = equity
        for date, price in held.items():
            count = max((date - start).days, 0)
            val = q * (target * (float(price)/prev_price) + (1-target) * ((1+cash_rate)**(count/365.25)))
            dates.append(date)
            daily_equity.append(val)
        growth = target * (float(wk.iloc[i].Close)/prev_price) + (1-target)*((1+cash_rate)**((end-start).days/365.25))
        equity = q * growth
        risky_w = target * (float(wk.iloc[i].Close)/prev_price) / growth if growth > 0 else 0.0
    daily_curve = pd.Series(daily_equity, index=pd.to_datetime(dates)).sort_index()
    return curve, trades, change_count, turnover_sum, daily_curve

st.title("📈 QLD 이동평균선 전략")
st.caption("현재 신호 · 목표 투자 비중 · 백테스트 · 주간 리밸런싱 | 원본 코드 기반")

with st.sidebar:
    st.header("전략 설정")
    ticker = st.text_input("종목 코드", value="QLD").strip().upper()
    start = st.date_input("데이터 시작일", value=pd.Timestamp("2006-01-01").date(), max_value=pd.Timestamp.today().date())
    ema_type = st.radio("중기 이동평균", ["원본 com=20 (EMA41)", "수정 span=20 (EMA20)"], index=0, help="원본 코드의 com=20은 EMA41과 같은 평활계수입니다.")
    long_days = st.selectbox("장기 이동평균", [180, 200], index=0)
    cost_bps = st.number_input("편도 매매비용 (%)", 0.0, 2.0, 0.1, 0.05)
    cash_apr = st.number_input("현금 연수익률 (%)", 0.0, 15.0, 0.0, 0.25)
    investment = st.number_input("투자금 (원)", 100_000, 100_000_000_000, 100_000_000, step=10_000_000)
    st.divider()
    st.markdown("**상태별 목표 투자 비중**")
    st.caption("키 이름은 표시용으로 고정. SMA180은 선택한 180/200일선에 해당합니다.")
    allocs = []
    for j, state in enumerate(STATES):
        allocs.append(st.slider(" > ".join(state), min_value=0, max_value=100, value=int(DEFAULTS[j]*100), step=5, key=f"alloc{j}"))
    st.caption("수정하면 즉시 재계산됩니다.")
    if st.button("🔄 주가 데이터 새로고침", use_container_width=True):
        fetch_prices.clear()
        st.rerun()

weights = dict(zip(STATES, [v/100 for v in allocs]))
try:
    with st.spinner("QLD 가격 데이터를 가져오고 있습니다..."):
        raw = fetch_prices(ticker, str(start))
        d = indicators(raw, ema_type, long_days)
    if len(d) < 20: st.error("충분한 데이터가 없습니다. 시작일을 앞당겨 주세요."); st.stop()
    state = d.iloc[-1]["state"]
    target = weights[state]
    latest = d.index[-1]
    st.info(f"최신 확보 일봉: **{latest:%Y-%m-%d}** (미국 시장 거래일 기준). 현재 신호는 잠정 신호이며 확정 리밸런싱 신호는 마지막 완료 주간 종가를 기준으로 합니다.")
    a,b,c,dcol = st.columns(4)
    a.metric("현재 신호상 목표 비중", f"{target:.0%}")
    b.metric("현금 비중", f"{1-target:.0%}")
    c.metric("QLD 목표 금액", f"{investment*target:,.0f} 원")
    dcol.metric("현금 목표 금액", f"{investment*(1-target):,.0f} 원")
    st.markdown(f"**최신 배열:** `{' > '.join(state)}`")
    last_friday = d.resample("W-FRI").last().dropna(subset=["state"])
    last_friday = last_friday.loc[last_friday.index <= latest]
    if len(last_friday):
        confirmed_state = last_friday.iloc[-1]["state"]
        st.success(f"마지막 완료 주 기준 신호 ({last_friday.index[-1]:%Y-%m-%d}): {' > '.join(confirmed_state)} → 목표 QLD {weights[confirmed_state]:.0%}")
    st.caption("현재 보유 수량/금액을 입력받지 않으므로 실제 주문수량은 계산하지 않습니다. 환율, 세금, 주문 체결가격과 실제 현금 이자는 별도 고려하세요.")

    st.divider()
    curve, trades, trade_events, turnover, daily_eq = backtest(d, weights, cost_bps/100, cash_apr/100)
    strategy_cagr, weekly_mdd = performance(curve["전략"])
    bnh_cagr, bnh_weekly_mdd = performance(curve["QLD 장기보유"])
    daily_mdd = float((daily_eq / daily_eq.cummax() - 1).min())
    original_daily = raw.Close.loc[(raw.index >= curve.index[0]) & (raw.index <= curve.index[-1])]
    bh_daily_curve = original_daily / original_daily.iloc[0]
    bnh_daily_mdd = float((bh_daily_curve/bh_daily_curve.cummax()-1).min())
    st.subheader("백테스트 요약")
    st.caption("완료된 주간 종가 신호를 다음 주에 적용하는 모델. 주간 리밸런싱을 금요일 종가에 비용 차감 후 체결했다고 가정하므로 실제 월요일 주문 성과와 다릅니다.")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("전략 CAGR", f"{strategy_cagr:.2%}")
    c2.metric("전략 MDD (일별 종가)", f"{daily_mdd:.2%}")
    c3.metric("장기보유 CAGR", f"{bnh_cagr:.2%}")
    c4.metric("장기보유 MDD (일별 종가)", f"{bnh_daily_mdd:.2%}")
    st.caption(f"백테스트 기간: {curve.index[0]:%Y-%m-%d} ~ {curve.index[-1]:%Y-%m-%d} | 리밸런싱 거래 발생 주: {trade_events}회 | 총 거래 회전율 합계: {turnover:.1f}배 | 거래비용: 매수·매도 금액 각각 {cost_bps:.2f}%")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curve.index,y=investment*curve["전략"],name="QLD 전략",line=dict(color="#60A5FA",width=3)))
    fig.add_trace(go.Scatter(x=curve.index,y=investment*curve["QLD 장기보유"],name="QLD 장기보유",line=dict(color="#FBBF24",width=2,dash="dash")))
    fig.update_layout(template="plotly_dark",title="자산 성장 비교 (주간 종가)",yaxis_title="평가금액 (원)",xaxis_title="",height=440,legend=dict(orientation="h",y=1.1),margin=dict(l=12,r=10,b=15,t=55))
    if st.toggle("로그 축으로 보기",value=True): fig.update_yaxes(type="log")
    st.plotly_chart(fig,use_container_width=True)
    annual = curve.resample("YE").last().pct_change()
    annual.iloc[0] = curve.resample("YE").last().iloc[0] - 1
    annual.index = annual.index.year
    st.subheader("연도별 수익률")
    st.dataframe(annual.style.format("{:.2%}"),use_container_width=True)
    st.subheader("최근 리밸런싱 기록")
    st.dataframe(trades.tail(15).sort_index(ascending=False).style.format({"목표비중":"{:.0%}","QLD 주간수익률":"{:.2%}","매매회전율":"{:.2%}","전략자산":"{:.3f}","단순보유자산":"{:.3f}"}),use_container_width=True)
    st.download_button("📥 전체 백테스트 CSV 다운로드",trades.to_csv(encoding="utf-8-sig").encode("utf-8-sig"),file_name="qld_backtest.csv",mime="text/csv")
    with st.expander("가정 및 주의사항"):
        st.markdown("""- 무료 Yahoo Finance 시세를 사용하며, 데이터 결측·수정·장애가 발생할 수 있습니다.\n- **원본 `com=20`은 EMA20이 아니라 EMA41과 같은 계수입니다.** 기본값은 원본 재현을 위해 그대로 둡니다.\n- 매주 **금요일 종가에 신호를 확인하고 그 종가에 리밸런싱했다는 이상적 가정**을 사용합니다. 실제 다음 거래일 체결은 차이가 납니다.\n- 미완료 주간 데이터는 성과에서 제외하지만 최신 일봉 신호는 위에 별도로 표시합니다.\n- MDD는 **일별 종가 기준**이며 장중 최대 낙폭은 더 클 수 있습니다.\n- 리밸런싱 비용은 주식 노출 비중의 실제 변화와 매매 이후 비중 드리프트를 반영하며, 현금 연수익률은 사용자 설정값입니다.\n- 과세·환율·환전수수료·슬리피지·배당 관련 별도 세무 효과는 미반영입니다. 이 서비스는 주문을 실행하지 않습니다.\n- yfinance 조정 종가 및 데이터 품질을 확인한 뒤 실전 매매에 활용하세요.""")
except Exception as exc:
    st.error("주가 조회 또는 계산 중 오류가 발생했어요. 종목 코드/인터넷 연결을 확인하고 새로고침해 주세요.")
    st.code(str(exc))

"""QLD 1/2/3 strategy comparison dashboard. Research tool; no orders."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="QLD 1·2·3호 통합 전략", page_icon="📈", layout="wide")
st.markdown("""<style>
.block-container {max-width: 1350px; padding-top: 1.5rem}
/* Avoid fixed dark metric backgrounds: remain legible in both light and dark themes. */
div[data-testid="stMetric"] {border: 1px solid rgba(120,130,140,.35); padding: 15px; border-radius: 12px; background: transparent}
div[data-testid="stMetric"] [data-testid="stMetricLabel"],
div[data-testid="stMetric"] [data-testid="stMetricValue"] {color: inherit !important}
</style>""", unsafe_allow_html=True)

PRESETS = {
    "1호 · SMA5/20/200": {"periods": ("SMA5", "SMA20", "SMA200"), "alloc": (85, 100, 40, 0, 0, 0)},
    "2호 · SMA5/20/200": {"periods": ("SMA5", "SMA20", "SMA200"), "alloc": (70, 100, 50, 0, 20, 10)},
    "3호 · SMA5/EMA/180": {"periods": ("SMA5", "EMA", "SMA180"), "alloc": (70, 100, 40, 0, 40, 10)},
}
ORDER = [(0, 1, 2), (1, 0, 2), (0, 2, 1), (1, 2, 0), (2, 0, 1), (2, 1, 0)]

@st.cache_data(ttl=3600, show_spinner=False)
def prices(start):
    data = yf.download("QLD", start=start, auto_adjust=True, progress=False, threads=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    if data.empty or "Close" not in data:
        raise ValueError("Yahoo Finance에서 QLD 주가를 가져오지 못했습니다.")
    data = data[["Close"]].dropna().sort_index()
    data.index = pd.to_datetime(data.index).tz_localize(None) if data.index.tz is not None else pd.to_datetime(data.index)
    return data


def prepare(price, long_days, ema_original):
    out = price.copy()
    out["SMA5"] = out.Close.rolling(5).mean()
    out["SMA20"] = out.Close.rolling(20).mean()
    out["SMA200"] = out.Close.rolling(200).mean()
    out["SMA180"] = out.Close.rolling(180).mean()
    out["EMA"] = out.Close.ewm(com=20 if ema_original else None, span=None if ema_original else 20, adjust=False).mean()
    # All three strategies use same price history: start only after 200-day indicator available.
    out = out.dropna().copy()
    return out


def state_at(row, names):
    return tuple(sorted(names, key=lambda x: (-float(row[x]), x)))


def run_variant(df, preset, fee, cash_annual, cutoff):
    names, values = preset["periods"], preset["alloc"]
    weights = {tuple(names[j] for j in order): value / 100 for order, value in zip(ORDER, values)}
    states = pd.Series([state_at(row, names) for _, row in df.iterrows()], index=df.index)
    # Weekly decision dates: last available trading close of each fully completed W-FRI week.
    marked = pd.DataFrame({"close": df.Close, "state": states})
    grouped = [g for _, g in marked.groupby(pd.Grouper(freq="W-FRI")) if len(g)]
    closes = [(g.index[-1], g.iloc[-1]["state"]) for g in grouped if g.index[-1].to_period("W-FRI").end_time.normalize() <= cutoff]
    if len(closes) < 3:
        raise ValueError("완료된 주간 데이터가 부족합니다.")
    first, last = closes[0][0], closes[-1][0]
    usable = df.loc[first:last, "Close"]
    rebalance_dates = {date: state for date, state in closes}
    equity = 1.0
    buyhold = 1.0
    invested = 0.0
    last_close = float(usable.iloc[0])
    current_state = rebalance_dates[first]
    # Initial rebalance at first week closing point, initial capital 1 before any purchases.
    target = weights[current_state]
    equity *= 1 - fee * abs(target - invested)
    invested = target
    rows = [{"date": first, "strategy": equity, "buyhold": 1.0, "target": target}]
    turnover, events = abs(target), int(target > 0)
    previous_date = first
    for date, close in usable.iloc[1:].items():
        elapsed = (date - previous_date).days
        qld_ret = float(close) / last_close - 1
        cash_ret = (1 + cash_annual) ** (elapsed / 365.25) - 1
        growth = invested * (1 + qld_ret) + (1 - invested) * (1 + cash_ret)
        equity *= growth
        buyhold *= 1 + qld_ret
        invested = invested * (1 + qld_ret) / growth if growth > 0 else 0
        if date in rebalance_dates and date != last:
            target = weights[rebalance_dates[date]]
            trade = abs(target - invested)
            turnover += trade
            events += int(trade > 1e-10)
            equity *= 1 - fee * trade
            invested = target
        rows.append({"date": date, "strategy": equity, "buyhold": buyhold, "target": invested})
        last_close = float(close)
        previous_date = date
    curve = pd.DataFrame(rows).set_index("date")
    years = (last - first).days / 365.25
    cagr = float(curve.strategy.iloc[-1] ** (1 / years) - 1) if years > 0 else np.nan
    mdd = float((curve.strategy / curve.strategy.cummax() - 1).min())
    final_state = state_at(df.iloc[-1], names)
    confirmed = closes[-1][1]
    return {"curve": curve, "CAGR": cagr, "MDD": mdd, "current": weights[final_state],
            "confirmed": weights[confirmed], "state": final_state, "confirmed_state": confirmed,
            "signal_date": closes[-1][0], "events": events, "turnover": turnover,
            "period_start": first, "period_end": last}

st.title("📈 QLD 1·2·3호 전략별 탭")
st.caption("아프나이델 QLD 전략 3종 동시 계산 · 마지막 완료 주 신호 · 연도별 수익률 · 동일 기간 비교")
with st.sidebar:
    st.header("공통 분석 설정")
    start = st.date_input("데이터 다운로드 시작일", pd.Timestamp("2006-01-01").date())
    capital = st.number_input("투자 기준 금액 (원)", min_value=100_000, max_value=100_000_000_000, value=100_000_000, step=10_000_000)
    fee_pct = st.number_input("편도 거래비용 (%)", min_value=0.0, max_value=2.0, value=0.10, step=0.05)
    cash_pct = st.number_input("현금 연수익률 (%)", min_value=0.0, max_value=15.0, value=0.0, step=0.25)
    st.divider()
    st.subheader("3호 EMA 선택")
    ema_mode = st.radio("3호의 원본 EMA 계산 방식", ["원본 com=20 (EMA41 상당)", "정확한 EMA20 (span=20)"], index=0)
    st.caption("1·2호는 SMA5·SMA20·SMA200을 그대로 사용합니다. 3호는 SMA5·EMA·SMA180입니다.")
    if st.button("↻ 최신 주가 다시 조회", use_container_width=True):
        prices.clear()
        st.rerun()

try:
    raw = prices(str(start))
    d = prepare(raw, 200, ema_mode.startswith("원본"))
    if len(d) < 40:
        st.error("주가 기록이 부족합니다. 시작일을 2006년으로 설정해 주세요.")
        st.stop()
    latest = d.index[-1]
    # Do not allow an incomplete week in completed-week backtest.
    # The previous completed Friday-based week is any weekly period ending no later than today (UTC date).
    today = pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize()
    cutoff = min(today, latest + pd.Timedelta(days=7))
    result = {name: run_variant(d, config, fee_pct / 100, cash_pct / 100, cutoff) for name, config in PRESETS.items()}
    st.info(f"마지막 수집된 QLD 일봉: **{latest:%Y-%m-%d}**. 일봉 잠정 신호와 마지막 완료 주 신호는 다를 수 있습니다.")
    st.subheader("전략별 상세 보기")
    names = list(result)
    tab1, tab2, tab3, tab_compare = st.tabs(["① QLD 1호", "② QLD 2호", "③ QLD 3호", "📊 전체 비교"])
    for tab, name in zip([tab1, tab2, tab3], names):
        r = result[name]
        with tab:
            st.markdown(f"### {name}")
            st.caption("이 탭은 선택한 전략만 표시합니다. 전략 선택을 매주 바꾸면 원래의 기계적 매매 규칙이 달라집니다.")
            a, b, c = st.columns(3)
            a.metric("지난 완료 주 QLD 목표 비중", f"{r['confirmed']:.0%}")
            b.metric("현금 목표 비중", f"{1-r['confirmed']:.0%}")
            c.metric("최신 일봉 잠정 비중", f"{r['current']:.0%}")
            st.info(f"확정 신호 기준일: {r['signal_date']:%Y-%m-%d} | 이동평균선 배열: {' > '.join(r['confirmed_state'])}")
            st.write(f"총 운용금 {capital:,.0f}원 기준: **QLD {capital*r['confirmed']:,.0f}원 / 현금 {capital*(1-r['confirmed']):,.0f}원**")
            a,b,c=st.columns(3)
            a.metric("전체 기간 CAGR", f"{r['CAGR']:.2%}")
            b.metric("전체 기간 MDD (일별 종가)", f"{r['MDD']:.2%}")
            c.metric("매매 발생 횟수", f"{r['events']:,}회")
            st.markdown("#### 올해 성과 (YTD)")
            curve = r["curve"]
            year = curve.index[-1].year
            # Prior-year last close anchors YTD; do not mistake a partial year for annual CAGR.
            before_year = curve[curve.index.year < year]
            if len(before_year):
                prior = before_year.iloc[-1]["strategy"]
                current_year = curve[curve.index.year == year]
                ytd = current_year.strategy.iloc[-1] / prior - 1
                ytd_curve = pd.concat([pd.Series([prior], index=[before_year.index[-1]]), current_year.strategy])
                ytd_mdd = float((ytd_curve / ytd_curve.cummax() - 1).min())
                y1,y2 = st.columns(2)
                y1.metric(f"{year}년 누적 수익률 (YTD)", f"{ytd:.2%}")
                y2.metric(f"{year}년 구간 MDD (일별 종가)", f"{ytd_mdd:.2%}")
                st.caption(f"기준: 전년도 마지막 거래일 ~ {curve.index[-1]:%Y-%m-%d}. 올해 수익률은 연환산 CAGR이 아닌 누적 YTD입니다.")
            else:
                st.caption("작년 마지막 거래일 데이터가 없어 올해 YTD를 계산할 수 없습니다.")
            st.caption("주의: 금요일 신호를 같은 금요일 종가에 체결했다고 가정한 모의 성과입니다. 실제 월요일 매매 결과와 다릅니다.")
            fig_one = go.Figure()
            fig_one.add_trace(go.Scatter(x=curve.index, y=curve.strategy * capital, name=name))
            fig_one.add_trace(go.Scatter(x=curve.index, y=curve.buyhold * capital, name="QLD 장기보유", line=dict(dash="dash")))
            fig_one.update_layout(height=350, hovermode="x unified", yaxis_title="평가액 (원)", margin=dict(l=10,r=10,t=20,b=20))
            st.plotly_chart(fig_one, use_container_width=True, key=f"plot_{name}")
            config = PRESETS[name]
            states = [" > ".join(config["periods"][i] for i in ix) for ix in ORDER]
            with st.expander("이 전략의 6가지 매매 비중 보기"):
                st.dataframe(pd.DataFrame({"이동평균선 배열": states, "QLD 목표 비중 (%)": config["alloc"]}), hide_index=True)
    with tab_compare:
        st.markdown("### QLD 1·2·3호 한눈에 비교")
        st.caption("연구·검증용 비교표입니다. 실전에서는 미리 정한 한 전략만 지속 운영하는 것을 권장합니다.")
        overview = pd.DataFrame([{"전략":name, "완료 주 목표":f"{r['confirmed']:.0%}", "CAGR":f"{r['CAGR']:.2%}", "MDD":f"{r['MDD']:.2%}"} for name,r in result.items()])
        st.dataframe(overview, hide_index=True, use_container_width=True)
    st.divider()
    st.subheader("동일 기간 자산 성장 비교")
    fig = go.Figure()
    colors = ["#2563eb", "#16a34a", "#e11d48"]
    for (name, r), color in zip(result.items(), colors):
        fig.add_trace(go.Scatter(x=r["curve"].index, y=r["curve"].strategy * capital,
                                 name=name, line=dict(color=color, width=2)))
    baseline = next(iter(result.values()))["curve"]
    fig.add_trace(go.Scatter(x=baseline.index, y=baseline.buyhold * capital,
                             name="QLD 장기보유", line=dict(color="#d97706", width=2, dash="dash")))
    logscale = st.toggle("로그 축", value=True)
    fig.update_layout(height=475, yaxis_type="log" if logscale else "linear", yaxis_title="평가액 (원)",
                      hovermode="x unified", legend=dict(orientation="h", y=1.12), margin=dict(l=20, r=20, t=60, b=30))
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("백테스트 비교표")
    comparison = pd.DataFrame({name: {"CAGR": r["CAGR"], "MDD": r["MDD"],
                                    "최종 자산 배수": r["curve"].strategy.iloc[-1],
                                    "매매 발생 횟수": r["events"], "총 회전율": r["turnover"]}
                               for name, r in result.items()}).T
    bnh_cagr = baseline.buyhold.iloc[-1] ** (365.25 / (baseline.index[-1] - baseline.index[0]).days) - 1
    bnh_mdd = (baseline.buyhold / baseline.buyhold.cummax() - 1).min()
    comparison.loc["QLD 장기보유"] = [bnh_cagr, bnh_mdd, baseline.buyhold.iloc[-1], 1, 1]
    st.dataframe(comparison.style.format({"CAGR":"{:.2%}", "MDD":"{:.2%}", "최종 자산 배수":"{:.2f}배", "매매 발생 횟수":"{:.0f}", "총 회전율":"{:.1f}배"}), use_container_width=True)
    st.subheader("연도별 수익률")
    years = pd.DataFrame({name: r["curve"].strategy for name, r in result.items()})
    years["QLD 장기보유"] = baseline.buyhold
    # Annual returns anchored on the actual starting principal.
    annual_end = years.resample("YE").last()
    annual = annual_end.pct_change()
    annual.iloc[0] = annual_end.iloc[0] - 1
    annual.index = annual.index.year
    st.dataframe(annual.style.format("{:.2%}"), use_container_width=True)
    st.download_button("📥 비교 결과 CSV 저장", comparison.to_csv(encoding="utf-8-sig").encode("utf-8-sig"),
                       file_name="QLD_123_comparison.csv", mime="text/csv")
    with st.expander("전략 비중표 및 백테스트 가정"):
        for name, p in PRESETS.items():
            states = [" > ".join(p["periods"][i] for i in ix) for ix in ORDER]
            st.markdown(f"**{name}**")
            st.dataframe(pd.DataFrame({"이동평균선 배열":states, "QLD 목표 비중 (%)":p["alloc"]}), hide_index=True)
        st.markdown("""- 세 전략은 **같은 QLD 일별 가격, 같은 기간**으로 계산됩니다. 이전 원저자의 공식 결과와 같다고 보장되지 않습니다.
- **완료된 주 금요일 마지막 거래일 종가에 신호가 확정되며, 그 시점에 다음 주 목표 비중으로 교체한 이상적 모델**입니다. 신호를 확인한 뒤 같은 종가로 체결하는 것은 현실적으로 불가능하므로 실전 성과는 다릅니다.
- CAGR은 시작 자산 1을 기준으로, MDD는 일별 **종가** 자산곡선으로 계산합니다. 장중 MDD는 더 클 수 있습니다.
- 포트폴리오 내 QLD 비중 변화를 반영한 거래비용을 차감합니다. 현금 수익률은 별도 설정합니다.
- QLD 조정 가격(yfinance)을 사용합니다. 세금, 환율, 슬리피지, 시장 충격은 미반영입니다. 주문 기능은 없습니다.
- 3호의 원본 `ewm(com=20)`은 일반적인 EMA20이 아니라 EMA41 상당입니다.
- 투자 비중은 원본 코드/앞서 확인한 1·2호 설정을 기준으로 했으며, 매수·매도 신호를 실전 적용하기 전 별도 검증이 필요합니다.""")
except Exception as exc:
    st.error("QLD 시세 조회 또는 전략 계산 중 오류가 발생했습니다.")
    st.exception(exc)

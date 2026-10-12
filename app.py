"""RFM 세그먼트 분석 웹앱 (Streamlit)

CSV를 올리면 → 데이터 품질 점검 → 컬럼 지정 → R·F·M 가중치 조절 → 세그먼트 결과
실행: streamlit run app.py
"""
import datetime as dt
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import rfm_core as rc

st.set_page_config(page_title="RFM 세그먼트 분석기", page_icon="📊", layout="wide")

SEG_ORDER = ["VIP", "충성 고객", "이탈 위험 우수고객", "잠재 성장", "일반", "휴면"]
SEG_COLOR = {"VIP": "#2E5BBA", "충성 고객": "#4C9BE8", "이탈 위험 우수고객": "#E8833A",
             "잠재 성장": "#59A96A", "일반": "#A0A4AB", "휴면": "#6B6F76"}
UP, DOWN = "#2E5BBA", "#E8833A"   # 전체보다 높음(파랑) / 낮음(주황)
HOVER_TIP = "막대 위에 마우스를 올리면 정확한 수치를 볼 수 있어요. (모바일은 막대를 눌러 주세요)"
STAT_NAME = {"age": "평균 나이", "median": "중앙값", "mean": "평균", "rate": "참여율", "share": "전체 구매 중 비중"}


# ------------------------------------------------------------------ 숫자 표시 형식 (천 단위 쉼표, 비율 소수 첫째 자리, 큰 금액 만 원)
def fmt_money(v):
    if pd.isna(v):
        return "-"
    return f"{v / 1e4:,.0f}만 원" if abs(v) >= 1e4 else f"{v:,.0f}원"


def fmt_pct(x):
    return "-" if pd.isna(x) else f"{x:.1%}"


def fmt_signed(x):
    if pd.isna(x):
        return "-"
    return "0.0%" if abs(x) < 0.0005 else f"{x:+.1%}"


def fmt_value(v, spec):
    if pd.isna(v):
        return "-"
    k = spec["kind"]
    if k == "age":
        return f"{v:,.1f}세"
    if k in ("rate", "share", "promo"):
        return fmt_pct(v)
    if spec.get("money"):
        return fmt_money(v)
    return f"{v:,.2f}" if abs(v) < 100 else f"{v:,.0f}"


def chart_title(title):
    """그래프 제목과 수치 확인 안내를 그래프 위에 표시한다."""
    st.markdown(f"**{title}**")
    st.caption(HOVER_TIP)

st.title("📊 RFM 세그먼트 분석기")
st.caption("고객 CSV를 올리면 데이터 점검부터 R·F·M 가중치에 따른 고객 세그먼트까지 한 번에 보여줍니다.")

# ------------------------------------------------------------------ 1. 업로드
with st.sidebar:
    st.header("1. 데이터 올리기")
    up = st.file_uploader("CSV 파일", type=["csv", "tsv", "txt"])
    kind = st.radio("데이터 형태", ["고객 합산형 (고객 1명 = 1줄)", "거래형 (거래 1건 = 1줄)"],
                    help="합산형: 최근성·구매 횟수·금액이 고객별로 이미 계산된 표\n\n"
                         "거래형: 주문 내역 원본. 앱이 고객별 R·F·M을 직접 계산합니다.")
    st.divider()
    st.markdown("**RFM이란?**\n\n- **R**ecency: 얼마나 최근에 샀나\n- **F**requency: 얼마나 자주 샀나\n"
                "- **M**onetary: 얼마나 많이 썼나\n\n각각 1~5점으로 매겨 고객을 나눕니다.")

dev_file = os.environ.get("RFM_DEV_FILE")  # 로컬 테스트용: 업로드 없이 파일 경로로 실행
if up is None and not dev_file:
    st.info("왼쪽에서 CSV 파일을 올려 주세요. 쉼표·탭 구분 모두 자동으로 인식합니다.")
    st.stop()


@st.cache_data
def load(raw: bytes) -> pd.DataFrame:
    return rc.read_csv_auto(raw)


df_raw = load(up.getvalue() if up is not None else open(dev_file, "rb").read())
st.success(f"불러오기 완료: {df_raw.shape[0]:,}행 × {df_raw.shape[1]}열")
with st.expander("원본 미리보기"):
    st.dataframe(df_raw.head(20), width="stretch")

# ------------------------------------------------------------------ 2. 품질 점검
st.header("2. 데이터 품질 점검")
st.caption("분석 전에 꼭 확인해야 할 값들입니다. 결측값은 어떤 컬럼을 R·F·M에 쓰는지 정한 뒤(3단계) 처리합니다.")
q = rc.quality_report(df_raw)

c1, c2, c3, c4 = st.columns(4)
c1.metric("결측값 있는 컬럼", len(q["missing"]))
c2.metric("값이 하나뿐인 컬럼", len(q["constant"]))
c3.metric("중복 행", q["duplicated_rows"])
c4.metric("극단값 있는 컬럼", len(q["outliers"]))

df = df_raw.copy()
notes = []

if q["missing"]:
    st.subheader("결측값 (현황)")
    miss = pd.DataFrame({"결측 수": pd.Series(q["missing"])})
    miss["비율"] = (miss["결측 수"] / len(df_raw)).map("{:.1%}".format)
    st.dataframe(miss, width="content")
    st.caption("여기서는 현황만 보여줍니다. R·F·M에 쓰는 컬럼이면 그 고객만 제외하고, "
               "아닌 컬럼은 고객을 빼지 않고 특성 비교에서만 빈 값을 뺍니다. (3단계 컬럼 지정 아래에서 확인)")

if q["constant"]:
    st.subheader("값이 하나뿐인 컬럼")
    st.write("모든 행의 값이 같아서 고객을 구분하는 데 쓸 수 없습니다: " + ", ".join(f"`{c}`" for c in q["constant"]))
    df = df.drop(columns=q["constant"])
    notes.append(f"분석에서 제외한 컬럼: {', '.join(q['constant'])}")

if q["duplicated_rows"]:
    if st.checkbox(f"완전히 같은 중복 행 {q['duplicated_rows']}개 제거", value=True):
        df = df.drop_duplicates()
        notes.append(f"중복 행 {q['duplicated_rows']}개 제거")

if q["outliers"]:
    st.subheader("극단값 (사분위 범위의 3배 밖)")
    st.caption("⚠️ 구매 금액처럼 한쪽으로 치우친 값은 극단값이 아니라 '큰손 고객'일 수 있습니다. "
               "입력 오류로 보이는 컬럼만 고르세요 (예: 출생연도 1901년).")
    ot = pd.DataFrame(q["outliers"]).T[["count", "min", "max"]]
    ot.columns = ["극단값 개수", "최솟값", "최댓값"]
    st.dataframe(ot, width="stretch")
    drop_out_cols = st.multiselect("극단값 행을 제외할 컬럼", [c for c in q["outliers"] if c in df.columns])
    if drop_out_cols:
        df, n = rc.drop_extreme(df, drop_out_cols)
        notes.append(f"극단값 행 {n}개 제외 ({', '.join(drop_out_cols)})")

# ------------------------------------------------------------------ 3. 컬럼 지정
st.header("3. R·F·M 컬럼 지정")
cols = list(df.columns)
num_cols = list(df.select_dtypes("number").columns)


def idx(options, value):
    return options.index(value) if value in options else 0


if kind.startswith("고객"):
    a, b = st.columns(2)
    id_col = a.selectbox("고객 ID", ["(행 번호 사용)"] + cols,
                         index=idx(["(행 번호 사용)"] + cols, rc.guess_columns(df, "customer_id")))
    rec_col = b.selectbox("최근성 (마지막 구매 후 경과일)", num_cols,
                          index=idx(num_cols, rc.guess_columns(df, "recency")))
    f_guess = [c for c in rc.guess_columns(df, "frequency", True) if "discount" not in c.lower()]
    freq_cols = st.multiselect("빈도 (더할 구매 횟수 컬럼)", num_cols, default=f_guess,
                               help="할인 구매처럼 다른 횟수와 겹칠 수 있는 컬럼은 이중 집계가 되지 않는지 확인하세요.")
    money_cols = st.multiselect("금액 (더할 구매 금액 컬럼)", num_cols,
                                default=rc.guess_columns(df, "monetary", True))
    if not (freq_cols and money_cols):
        st.warning("빈도와 금액 컬럼을 하나 이상 골라 주세요.")
        st.stop()
    rfm, miss_info = rc.rfm_from_customer_table(df, None if id_col.startswith("(") else id_col,
                                                rec_col, freq_cols, money_cols)
    used_cols = [c for c in [id_col, rec_col] + freq_cols + money_cols if c in df.columns]
    notes.append(f"빈도 = {' + '.join(freq_cols)}")
else:
    a, b, c = st.columns(3)
    id_col = a.selectbox("고객 ID", cols, index=idx(cols, rc.guess_columns(df, "customer_id")))
    date_col = b.selectbox("거래 일시", cols, index=idx(cols, rc.guess_columns(df, "date")))
    inv_col = c.selectbox("주문 번호 (없으면 날짜로 횟수 계산)", ["(없음)"] + cols,
                          index=idx(["(없음)"] + cols, rc.guess_columns(df, "invoice")))
    mode = st.radio("금액 계산", ["금액 컬럼 사용", "수량 × 단가"], horizontal=True,
                    index=1 if rc.guess_columns(df, "quantity") else 0)
    if mode == "금액 컬럼 사용":
        amt_col = st.selectbox("금액", num_cols, index=idx(num_cols, rc.guess_columns(df, "monetary")))
        qty_col = price_col = None
    else:
        a, b = st.columns(2)
        qty_col = a.selectbox("수량", num_cols, index=idx(num_cols, rc.guess_columns(df, "quantity")))
        price_col = b.selectbox("단가", num_cols, index=idx(num_cols, rc.guess_columns(df, "price")))
        amt_col = None
    ref = st.date_input("분석 기준일 (비우면 마지막 거래 다음 날)", value=None)
    drop_neg = st.checkbox("금액이 0 이하인 거래(취소·반품) 제외", value=True)
    rfm, log = rc.rfm_from_transactions(df, id_col, date_col,
                                        None if inv_col == "(없음)" else inv_col,
                                        amt_col, qty_col, price_col, ref, drop_neg)
    miss_info = {"total": log["start"], "dropped": log["dropped_missing"],
                 "dropped_rate": log["dropped_missing"] / max(log["start"], 1), "unit": "거래"}
    used_cols = [c for c in [id_col, date_col, inv_col, amt_col, qty_col, price_col] if c and c in df.columns]
    notes.append(f"거래 {log['start']:,}건 → 고객 {log['customers']:,}명 (기준일 {log['ref_date']}, "
                 f"결측 {log['dropped_missing']:,}건·0 이하 {log.get('dropped_nonpositive', 0):,}건 제외)")

# ------------------------------------------------------------------ 결측 처리 (컬럼 지정 후 적용)
st.subheader("결측값 처리")
unit = miss_info.get("unit", "고객")
if miss_info["dropped"]:
    st.markdown(f"R·F·M 계산에 쓰는 컬럼이 비어 있는 {unit} **{miss_info['dropped']:,}{'건' if unit == '거래' else '명'}"
                f"({miss_info['dropped_rate']:.1%})** 을 제외했습니다. 컬럼은 그대로 둡니다.")
    if miss_info["dropped_rate"] >= rc.MISSING_EXCLUDE:
        st.warning(f"제외된 {unit}이 전체의 {miss_info['dropped_rate']:.0%}입니다. "
                   "결과가 전체를 대표하지 못할 수 있으니 데이터를 확인해 주세요.")
    notes.append(f"R·F·M 컬럼 결측으로 {unit} {miss_info['dropped']:,}{'건' if unit == '거래' else '명'} "
                 f"({miss_info['dropped_rate']:.1%}) 제외")
else:
    st.markdown("R·F·M 계산에 쓰는 컬럼에는 빈 값이 없습니다.")
other_na = rc.other_missing(df, used_cols)
if len(other_na):
    st.markdown("R·F·M에 쓰지 않는 컬럼은 고객을 빼지 않고, 특성 비교에서만 빈 값을 빼고 계산합니다.")
    st.dataframe(other_na.style.format({"비율": "{:.1%}", "결측 수": "{:,}"}), width="content", hide_index=True)
    for _, r in other_na.iterrows():
        if r["처리"].startswith("사용 ("):
            st.warning(f"`{r['컬럼']}`은 결측이 {r['비율']:.1%}로 많아 특성 비교 결과를 해석할 때 주의가 필요해요.")
        notes.append(f"{r['컬럼']} 결측 {r['결측 수']:,}{'건' if unit == '거래' else '명'}({r['비율']:.1%}) — 제외하지 않음, {r['처리']}")

if len(rfm) < 10:
    st.error("고객 수가 너무 적어 점수를 나눌 수 없습니다.")
    st.stop()

# ------------------------------------------------------------------ 4. 세그먼트 기준 (가중치)
st.header("4. 세그먼트 기준")
st.markdown("최근성(R)은 고객의 **현재 상태**, 가중치는 **'가치 있는 고객'을 판단하는 기준**이에요. "
            "가치 점수 V는 구매 횟수(F)와 금액(M) 점수를 아래 비율로 섞어 계산합니다.")
preset_names = list(rc.PRESETS) + ["직접 조정"]
preset = st.radio(
    "분석 목표", preset_names, key="preset",
    captions=[f"F:M = {w['weights'][0]:g}:{w['weights'][1]:g} — {w['why']}" for w in rc.PRESETS.values()]
    + ["F·M 슬라이더로 비율을 직접 정해요"])
st.caption("이 비율은 정답이 아니라 출발점이에요. 결과를 보면서 사업 목표에 맞게 바꿔 보세요.")
if preset == "직접 조정":
    a, b = st.columns(2)
    wf = a.slider("F (구매 횟수) 가중치", 0.0, 5.0, 1.0, 0.5, key="wf")
    wm = b.slider("M (구매 금액) 가중치", 0.0, 5.0, 1.0, 0.5, key="wm")
    if wf + wm == 0:
        st.warning("가중치가 모두 0입니다. 하나 이상 올려 주세요.")
        st.stop()
else:
    wf, wm = rc.PRESETS[preset]["weights"]
st.info("💡 이탈 방지가 목적이라면 가중치를 바꾸기보다 결과의 **'이탈 위험 우수고객'** 세그먼트를 보세요. "
        "가치는 높은데 최근 발길이 끊긴 고객들입니다.")

scored = rc.score(rfm, (wf, wm))
sens = rc.weight_sensitivity(rfm, (wf, wm))
if sens["changed"]:
    st.markdown(f"**균형(1:1) 대비 {sens['changed']:,}명의 세그먼트가 바뀌었습니다.**")
    cnt = sens["counts"].reindex([s for s in SEG_ORDER if s in sens["counts"].index])
    st.dataframe(cnt.style.format("{:+,}", subset=["차이"]).format("{:,}", subset=["균형 (1:1)", "현재"]),
                 width="content")
else:
    st.caption("균형(1:1)과 세그먼트 결과가 같습니다.")
with st.expander("세그먼트 규칙 보기"):
    st.markdown("\n".join(f"- **{k}**: {v}" for k, v in rc.SEGMENT_RULES.items()))

# ------------------------------------------------------------------ 5. 결과
st.header("5. 세그먼트 결과")
total_m = scored["monetary"].sum()
summary = (scored.groupby("segment")
           .agg(고객수=("customer_id", "size"), 평균_R=("recency", "mean"),
                평균_F=("frequency", "mean"), 평균_M=("monetary", "mean"), 매출합=("monetary", "sum"))
           .reindex([s for s in SEG_ORDER if s in scored["segment"].unique()]))
summary["고객 비중"] = summary["고객수"] / summary["고객수"].sum()
summary["매출 비중"] = summary["매출합"] / total_m

a, b = st.columns(2)
fig1 = px.bar(summary.reset_index(), x="segment", y="고객 비중", color="segment",
              color_discrete_map=SEG_COLOR, title="세그먼트별 고객 비중", text_auto=".0%")
fig2 = px.bar(summary.reset_index(), x="segment", y="매출 비중", color="segment",
              color_discrete_map=SEG_COLOR, title="세그먼트별 매출 비중", text_auto=".0%")
for f in (fig1, fig2):
    f.update_layout(showlegend=False, xaxis_title=None, yaxis_tickformat=".0%")
a.plotly_chart(fig1, width="stretch")
b.plotly_chart(fig2, width="stretch")

top = summary.sort_values("매출 비중", ascending=False).iloc[0]
st.markdown(f"**한눈에 보기:** `{summary['매출 비중'].idxmax()}` 고객이 전체의 {top['고객 비중']:.0%}로 "
            f"매출의 {top['매출 비중']:.0%}를 차지합니다.")

show = summary.copy()
show["고객 비중"] = show["고객 비중"].map("{:.1%}".format)
show["매출 비중"] = show["매출 비중"].map("{:.1%}".format)
show = show.drop(columns="매출합").round(1)
show.insert(0, "설명", [rc.SEGMENT_INFO[s] for s in show.index])
st.dataframe(show, width="stretch")

fig3 = px.density_heatmap(scored, x="R", y="F", z="monetary", histfunc="avg", nbinsx=5, nbinsy=5,
                          title="R·F 점수별 평균 구매 금액", color_continuous_scale="Blues")
st.plotly_chart(fig3, width="stretch")

# 고객 합산형이면 세그먼트별로 다른 특성(소득, 자녀 수 등)도 비교
small_cols, prof_methods, ref_year = [], {}, None
if kind.startswith("고객"):
    seg = scored["segment"]
    dfp = df.loc[scored.index]
    order = [s for s in SEG_ORDER if s in seg.unique()]
    specs, skipped = rc.profile_specs(dfp, used_cols, freq_cols)
    heavy = set(other_na.loc[other_na["처리"].str.startswith("특성"), "컬럼"]) if len(other_na) else set()
    items = {}
    for sp in specs:
        if sp.get("promo"):
            continue
        items[f"나이 ({sp['col']})" if sp["kind"] == "age" else str(sp["col"])] = sp
    promo_cols = [sp["col"] for sp in specs if sp.get("promo")]
    if promo_cols:
        items[f"프로모션 참여 ({len(promo_cols)}개 컬럼)"] = {"kind": "promo", "cols": promo_cols, "col": promo_cols[0]}

    if items:
        st.subheader("세그먼트별 고객 특성")
        st.caption("각 세그먼트가 전체 고객보다 몇 % 높고 낮은지 보여줘요. 빈 값은 이 비교에서만 빼고 계산합니다.")
        if any(sp["kind"] == "age" for sp in items.values()):
            ref_year = st.number_input("나이 계산 기준 연도", 1950, 2100, dt.date.today().year, key="ref_year",
                                       help="출생연도를 나이로 바꿀 때 쓰는 연도예요. 실습 데이터는 2023년 기준이라 바꿀 수 있어요.")
        default = [k for k, sp in items.items() if not set(sp.get("cols", [sp["col"]])) & heavy]
        chosen = st.multiselect("비교할 컬럼", list(items), default=default, key="prof_cols",
                                help="결측이 20% 이상인 컬럼은 기본으로 빠져 있어요. 필요하면 다시 고를 수 있어요.")

        # 컬럼별 세그먼트 값 계산
        vals = {}
        for k in chosen:
            sp = items[k]
            if sp["kind"] == "category":
                prof_methods[k] = "구성 비율"
                continue
            if sp["kind"] == "promo":
                pr = rc.promo_rates(dfp, seg, sp["cols"], order)
                v = pd.DataFrame({"value": pr["하나라도 참여"].drop("전체")})
                v["diff"] = v["value"] / pr.loc["전체", "하나라도 참여"] - 1
                v["n"] = seg.value_counts().reindex(v.index)
                v.attrs["overall"] = pr.loc["전체", "하나라도 참여"]
                prof_methods[k] = "참여율 (하나라도 참여)"
            else:
                v = rc.profile_values(dfp, seg, sp, ref_year, scored["frequency"], order)
                prof_methods[k] = STAT_NAME[sp["kind"]]
            vals[k] = v
            if v["diff"].abs().max() < rc.SMALL_DIFF:
                small_cols.append(k)

        if vals:
            st.markdown("**한눈에 보기: 세그먼트 × 컬럼 (전체 대비 차이)**")
            st.caption("파랑은 전체보다 높음, 주황은 낮음. 차이가 클수록 진하게 칠했어요.")
            table = pd.DataFrame({k: v["diff"] for k, v in vals.items()}).reindex(order)
            table.index.name = "세그먼트"

            def _cell(x):
                # 흰색과 기준색을 섞은 불투명 색. 차이가 클수록(50%에서 최대) 기준색에 가깝게.
                if pd.isna(x):
                    return ""
                a = 0.12 + 0.88 * min(abs(x) / 0.5, 1.0)
                base = (46, 91, 186) if x > 0 else (232, 131, 58)
                r, g, b = (round(255 + (c - 255) * a) for c in base)
                return f"background-color: rgb({r},{g},{b}); color: {'#FFFFFF' if a > 0.55 else '#1F2430'}"

            st.dataframe(table.style.map(_cell).format(fmt_signed), width="stretch")

        if chosen:
            pick = st.selectbox("자세히 볼 컬럼", chosen, key="prof_pick")
            sp = items[pick]
            if sp["kind"] == "category":
                mix = rc.category_mix(dfp, seg, sp["col"], order)
                cnt = pd.crosstab(seg.reindex(dfp[sp["col"]].dropna().index), dfp[sp["col"]].dropna()).reindex(mix.index)
                long = mix.reset_index().melt(id_vars="segment", var_name="값", value_name="비율")
                long["고객 수"] = [f"{cnt.loc[r.segment, r.값]:,}" for r in long.itertuples()]
                long["비율 표시"] = long["비율"].map(fmt_pct)
                chart_title(f"세그먼트별 {pick} 구성")
                fig = px.bar(long, x="segment", y="비율", color="값", custom_data=["값", "비율 표시", "고객 수"])
                fig.update_traces(hovertemplate="세그먼트: %{x}<br>값: %{customdata[0]}<br>"
                                                "비율: %{customdata[1]}<br>고객 수: %{customdata[2]}명<extra></extra>")
                fig.update_layout(yaxis_tickformat=".0%", xaxis_title=None, yaxis_title=None, legend_title=pick)
                st.plotly_chart(fig, width="stretch")
            elif sp["kind"] == "promo":
                pr = rc.promo_rates(dfp, seg, sp["cols"], order)
                long = pr.reset_index(names="segment").melt(id_vars="segment", var_name="프로모션", value_name="참여율")
                long["참여율 표시"] = long["참여율"].map(fmt_pct)
                n_seg = seg.value_counts()
                long["고객 수"] = [f"{(n_seg.get(x) or len(seg)):,}" for x in long["segment"]]
                chart_title("세그먼트별 프로모션 참여율")
                fig = px.bar(long, x="segment", y="참여율", color="프로모션", barmode="group",
                             custom_data=["프로모션", "참여율 표시", "고객 수"])
                fig.update_traces(hovertemplate="세그먼트: %{x}<br>프로모션: %{customdata[0]}<br>"
                                                "참여율: %{customdata[1]}<br>고객 수: %{customdata[2]}명<extra></extra>")
                fig.update_layout(yaxis_tickformat=".0%", xaxis_title=None, yaxis_title=None)
                st.plotly_chart(fig, width="stretch")
                st.caption(f"'하나라도 참여'는 {len(sp['cols'])}개 프로모션 중 하나 이상 참여한 고객 비율이에요.")
            else:
                v = vals[pick]
                mx = v["diff"].abs().max()
                if mx < rc.SMALL_DIFF:
                    st.info(f"`{pick}`은 세그먼트 간 차이가 거의 없어요 (최대 ±{mx:.1%}). "
                            f"전체 {STAT_NAME[sp['kind']]}: {fmt_value(v.attrs['overall'], sp)}")
                else:
                    chart_title(f"세그먼트별 {pick} — 전체 대비 차이 ({STAT_NAME[sp['kind']]} 기준)")
                    cd = np.stack([[fmt_value(x, sp) for x in v["value"]], [fmt_signed(x) for x in v["diff"]],
                                   [f"{int(x):,}" for x in v["n"]]], axis=-1)
                    fig = go.Figure(go.Bar(
                        x=v.index, y=v["diff"], customdata=cd,
                        marker_color=[UP if x >= 0 else DOWN for x in v["diff"]],
                        text=[fmt_signed(x) for x in v["diff"]], textposition="outside", cliponaxis=False,
                        hovertemplate="세그먼트: %{x}<br>" + STAT_NAME[sp["kind"]] + ": %{customdata[0]}<br>"
                                      "전체 대비: %{customdata[1]}<br>고객 수: %{customdata[2]}명<extra></extra>"))
                    fig.add_hline(y=0, line_color="#888", line_width=1)
                    fig.update_layout(yaxis_tickformat="+.0%", yaxis_title="전체 대비 차이", xaxis_title=None,
                                      margin=dict(t=30))
                    st.plotly_chart(fig, width="stretch")
                    st.caption(f"0선 = 전체 고객 {STAT_NAME[sp['kind']]} {fmt_value(v.attrs['overall'], sp)}. "
                               "위로 가면 전체보다 높고, 아래로 가면 낮아요.")

    if money_cols and len(money_cols) > 1:
        mix = df.loc[scored.index, money_cols].join(scored["segment"]).groupby("segment")[money_cols].sum()
        mix = mix.div(mix.sum(axis=1), axis=0).reindex(summary.index)
        fig5 = px.bar(mix.reset_index().melt(id_vars="segment", var_name="품목", value_name="비중"),
                      x="segment", y="비중", color="품목", title="세그먼트별 품목 구매 구성")
        fig5.update_layout(yaxis_tickformat=".0%", xaxis_title=None)
        st.plotly_chart(fig5, width="stretch")

# ------------------------------------------------------------------ 6. 처리 기록과 내려받기
st.header("6. 처리 기록")
st.caption("어떤 판단으로 데이터를 다뤘는지 남겨 두면 결과를 다시 검증할 수 있습니다.")
for n in notes:
    st.markdown(f"- {n}")
st.markdown(f"- 세그먼트 기준: {preset} (F:M = {wf:g}:{wm:g}), 점수는 1~5점 (순위 기준 5등분)")

st.download_button("세그먼트 결과 CSV 내려받기",
                   scored.to_csv(index=False).encode("utf-8-sig"),
                   file_name="rfm_segments.csv", mime="text/csv")

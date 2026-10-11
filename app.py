"""RFM 세그먼트 분석 웹앱 (Streamlit)

CSV를 올리면 → 데이터 품질 점검 → 컬럼 지정 → R·F·M 가중치 조절 → 세그먼트 결과
실행: streamlit run app.py
"""
import os

import pandas as pd
import plotly.express as px
import streamlit as st

import rfm_core as rc

st.set_page_config(page_title="RFM 세그먼트 분석기", page_icon="📊", layout="wide")

SEG_ORDER = ["VIP", "충성 고객", "이탈 위험 우수고객", "잠재 성장", "일반", "휴면"]
SEG_COLOR = {"VIP": "#2E5BBA", "충성 고객": "#4C9BE8", "이탈 위험 우수고객": "#E8833A",
             "잠재 성장": "#59A96A", "일반": "#A0A4AB", "휴면": "#6B6F76"}

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
st.caption("분석 전에 꼭 확인해야 할 값들입니다. 처리 방법은 직접 선택하세요.")
q = rc.quality_report(df_raw)

c1, c2, c3, c4 = st.columns(4)
c1.metric("결측값 있는 컬럼", len(q["missing"]))
c2.metric("값이 하나뿐인 컬럼", len(q["constant"]))
c3.metric("중복 행", q["duplicated_rows"])
c4.metric("극단값 있는 컬럼", len(q["outliers"]))

df = df_raw.copy()
notes = []

if q["missing"]:
    st.subheader("결측값")
    st.dataframe(pd.Series(q["missing"], name="결측 개수"), width="content")
    drop_na_cols = st.multiselect("결측값이 있는 행을 제외할 컬럼", list(q["missing"]),
                                  default=list(q["missing"]))
    if drop_na_cols:
        before = len(df)
        df = df.dropna(subset=drop_na_cols)
        notes.append(f"결측값 행 {before - len(df)}개 제외 ({', '.join(drop_na_cols)})")

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
    rfm = rc.rfm_from_customer_table(df, None if id_col.startswith("(") else id_col,
                                     rec_col, freq_cols, money_cols)
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
    notes.append(f"거래 {log['start']:,}건 → 고객 {log['customers']:,}명 (기준일 {log['ref_date']}, "
                 f"결측 {log['dropped_missing']:,}건·0 이하 {log.get('dropped_nonpositive', 0):,}건 제외)")

if len(rfm) < 10:
    st.error("고객 수가 너무 적어 점수를 나눌 수 없습니다.")
    st.stop()

# ------------------------------------------------------------------ 4. 가중치
st.header("4. R·F·M 가중치")
st.caption("사업 목표에 따라 중요도를 조절하세요. 예) 이탈 방지가 목표면 R↑, 객단가 관리가 목표면 M↑")
a, b, c = st.columns(3)
wr = a.slider("R 가중치", 0.0, 5.0, 1.0, 0.5)
wf = b.slider("F 가중치", 0.0, 5.0, 1.0, 0.5)
wm = c.slider("M 가중치", 0.0, 5.0, 1.0, 0.5)
if wr + wf + wm == 0:
    st.warning("가중치가 모두 0입니다. 하나 이상 올려 주세요.")
    st.stop()

scored = rc.score(rfm, (wr, wf, wm))
sens = rc.weight_sensitivity(rfm, (wr, wf, wm))
if (wr, wf, wm) != (1.0, 1.0, 1.0):
    st.info(f"가중치를 같게 뒀을 때와 비교하면, 상위 20% 고객 {sens['top_n']:,}명 중 "
            f"**{sens['new_in']:,}명이 바뀌었습니다** (겹치는 비율 {sens['overlap_rate']:.0%}). "
            "가중치 선택이 '누구를 핵심 고객으로 볼지'를 바꾼다는 뜻입니다.")

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
if kind.startswith("고객"):
    extra = [c for c in df.select_dtypes("number").columns
             if c not in ([rec_col] + freq_cols + money_cols) and c != id_col]
    if extra:
        st.subheader("세그먼트별 고객 특성")
        pick = st.selectbox("비교할 컬럼", extra)
        prof = df[[pick]].join(scored["segment"]).groupby("segment")[pick].mean() \
            .reindex([s for s in SEG_ORDER if s in scored["segment"].unique()]).reset_index()
        fig4 = px.bar(prof, x="segment", y=pick, color="segment", color_discrete_map=SEG_COLOR,
                      title=f"세그먼트별 평균 {pick}")
        fig4.update_layout(showlegend=False, xaxis_title=None)
        st.plotly_chart(fig4, width="stretch")
        if money_cols and len(money_cols) > 1:
            mix = df[money_cols].join(scored["segment"]).groupby("segment")[money_cols].sum()
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
st.markdown(f"- 가중치 R:F:M = {wr:g}:{wf:g}:{wm:g}, 점수는 1~5점 (순위 기준 5등분)")

st.download_button("세그먼트 결과 CSV 내려받기",
                   scored.to_csv(index=False).encode("utf-8-sig"),
                   file_name="rfm_segments.csv", mime="text/csv")

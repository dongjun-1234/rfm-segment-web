"""RFM 세그먼트 분석 웹앱 (Streamlit)

CSV를 올리면 → 데이터 품질 점검 → 컬럼 지정 → R·F·M 가중치 조절 → 세그먼트 결과
실행: streamlit run app.py
"""
import datetime as dt
import hashlib
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


ITEM_KO = {"alcohol": "주류", "wine": "주류", "fruit": "과일", "meat": "육류", "fish": "수산물",
           "snack": "과자", "sweet": "과자", "general": "잡화", "gold": "잡화"}


def item_name(col):
    """amount_meat → 육류. 모르는 이름은 amount_만 떼고 그대로."""
    key = str(col)
    for pre in ("amount_", "amt_", "mnt_", "Mnt"):
        if key.startswith(pre):
            key = key[len(pre):]
            break
    return ITEM_KO.get(key.lower(), key)


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


raw = up.getvalue() if up is not None else open(dev_file, "rb").read()
df_raw = load(raw)
# 파일마다 다른 값. 위젯 key에 붙여서 새 파일을 올리면 선택칸이 새로 만들어지고 자동 추측값이 다시 적용되게 한다.
# (key가 없으면 컬럼 구성이 같은 파일을 올렸을 때 이전 파일에서 고른 값이 그대로 남는다)
file_sig = hashlib.md5(raw).hexdigest()[:10]
file_name = up.name if up is not None else os.path.basename(dev_file)
# 처리 기록: 화면에서 내린 판단을 소제목별로 모은다 (6. 처리 기록에서 보여주고 .md로 내려받기)
logs = {"데이터": [f"파일 이름: {file_name}", f"원래 크기: {df_raw.shape[0]:,}행 × {df_raw.shape[1]}열",
                  f"데이터 형태: {'고객 합산형' if kind.startswith('고객') else '거래형'}"],
        "컬럼 지정": [], "데이터 정리": [], "세그먼트 기준": [], "특성 비교 설정": []}
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
drop_out_cols, n_extreme, n_dup = [], 0, 0

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
    logs["데이터 정리"].append(f"제외한 컬럼 (값이 하나뿐): {', '.join(map(str, q['constant']))}")

if q["duplicated_rows"]:
    if st.checkbox(f"완전히 같은 중복 행 {q['duplicated_rows']}개 제거", value=True, key=f"dup_{file_sig}"):
        df = df.drop_duplicates()
        n_dup = q["duplicated_rows"]

if q["outliers"]:
    st.subheader("극단값 (사분위 범위의 3배 밖)")
    st.caption("⚠️ 구매 금액처럼 한쪽으로 치우친 값은 극단값이 아니라 '큰손 고객'일 수 있습니다. "
               "입력 오류로 보이는 컬럼만 고르세요 (예: 출생연도 1901년).")
    ot = pd.DataFrame(q["outliers"]).T[["count", "min", "max"]]
    ot.columns = ["극단값 개수", "최솟값", "최댓값"]
    st.dataframe(ot, width="stretch")
    drop_out_cols = st.multiselect("극단값 행을 제외할 컬럼", [c for c in q["outliers"] if c in df.columns],
                                   key=f"extreme_{file_sig}")
    if drop_out_cols:
        df, n_extreme = rc.drop_extreme(df, drop_out_cols)

# ------------------------------------------------------------------ 3. 컬럼 지정
st.header("3. R·F·M 컬럼 지정")
st.caption("웹이 컬럼 이름을 보고 미리 골라 두었어요. 맞는지 확인하고, 다르면 직접 바꿔 주세요.")
cols = list(df.columns)
num_cols = list(df.select_dtypes("number").columns)


def pick(where, label, options, guess, role, help):
    """자동 추측값을 기본으로 보여주는 선택칸. key에 파일 서명을 넣어 파일이 바뀌면 새로 만든다."""
    index = options.index(guess) if guess in options else 0
    return where.selectbox(label, options, index=index, key=f"{role}_{file_sig}", help=help)


if kind.startswith("고객"):
    g_id, g_rec = rc.guess_columns(df, "customer_id"), rc.guess_columns(df, "recency")
    f_guess = [c for c in rc.guess_columns(df, "frequency", True) if "discount" not in str(c).lower()]
    m_guess = rc.guess_columns(df, "monetary", True)
    a, b = st.columns(2)
    id_col = pick(a, "CSV에서 고객 ID가 담긴 컬럼을 선택하세요", ["(행 번호 사용)"] + cols, g_id, "id",
                  "고객 한 명을 구분하는 값이에요. 예) ID, Customer ID, 회원번호")
    rec_col = pick(b, "CSV에서 마지막 구매 후 경과일이 담긴 컬럼을 선택하세요", num_cols, g_rec, "rec",
                   "마지막으로 산 지 며칠 지났는지예요. 예) recency, 최근구매경과일. "
                   "금액이나 횟수 컬럼을 고르면 결과가 틀어져요.")
    freq_cols = st.multiselect("구매 횟수가 담긴 컬럼을 모두 선택하세요 (선택한 값을 더해요)", num_cols,
                               default=f_guess, key=f"freq_{file_sig}",
                               help="예) 웹 구매 횟수 + 매장 구매 횟수. 할인 구매처럼 다른 횟수와 겹치는 컬럼은 "
                                    "빼야 두 번 세지 않아요.")
    money_cols = st.multiselect("구매 금액이 담긴 컬럼을 모두 선택하세요 (선택한 값을 더해요)", num_cols,
                                default=m_guess, key=f"money_{file_sig}",
                                help="예) 품목별 구매 금액을 모두 선택하면 고객의 총 구매 금액이 돼요.")
    if not (freq_cols and money_cols):
        st.warning("빈도와 금액 컬럼을 하나 이상 골라 주세요.")
        st.stop()
    rfm, miss_info = rc.rfm_from_customer_table(df, None if id_col.startswith("(") else id_col,
                                                rec_col, freq_cols, money_cols)
    used_cols = [c for c in [id_col, rec_col] + freq_cols + money_cols if c in df.columns]
    col_log = [("고객 ID", id_col, g_id or "(행 번호 사용)"), ("최근성", rec_col, g_rec),
               ("빈도", freq_cols, f_guess), ("금액", money_cols, m_guess)]
else:
    g_id, g_date, g_inv = (rc.guess_columns(df, r) for r in ("customer_id", "date", "invoice"))
    a, b, c = st.columns(3)
    id_col = pick(a, "CSV에서 고객 ID가 담긴 컬럼을 선택하세요", cols, g_id, "tid",
                  "고객 한 명을 구분하는 값이에요. 예) Customer ID, 회원번호. 거래형은 같은 고객이 여러 줄에 나와도 괜찮아요.")
    date_col = pick(b, "CSV에서 거래 일시가 담긴 컬럼을 선택하세요", cols, g_date, "date",
                    "언제 샀는지예요. 예) InvoiceDate, 주문일시. 고객별 마지막 거래일로 최근성을 계산해요.")
    inv_col = pick(c, "CSV에서 주문 번호가 담긴 컬럼을 선택하세요", ["(없음)"] + cols, g_inv, "inv",
                   "주문 한 건을 구분하는 값이에요. 예) Invoice, 주문번호. 없으면 '(없음)'을 고르면 "
                   "거래한 날짜 수로 구매 횟수를 세요.")
    mode = st.radio("구매 금액을 어떻게 계산할지 선택하세요", ["금액 컬럼 사용", "수량 × 단가"], horizontal=True,
                    index=1 if rc.guess_columns(df, "quantity") else 0, key=f"mode_{file_sig}",
                    help="거래 한 건의 금액 컬럼이 있으면 그대로 쓰고, 없으면 수량과 단가를 곱해요.")
    if mode == "금액 컬럼 사용":
        amt_col = pick(st, "CSV에서 거래 금액이 담긴 컬럼을 선택하세요", num_cols,
                       rc.guess_columns(df, "monetary"), "amt",
                       "거래 한 건의 금액이에요. 예) Amount, 결제금액. 고객별로 더하면 총 구매 금액이 돼요.")
        qty_col = price_col = None
    else:
        a, b = st.columns(2)
        qty_col = pick(a, "CSV에서 수량이 담긴 컬럼을 선택하세요", num_cols, rc.guess_columns(df, "quantity"), "qty",
                       "한 번에 몇 개 샀는지예요. 예) Quantity, 수량. 수량 × 단가로 거래 금액을 계산해요.")
        price_col = pick(b, "CSV에서 단가가 담긴 컬럼을 선택하세요", num_cols, rc.guess_columns(df, "price"), "price",
                         "상품 하나의 가격이에요. 예) Price, UnitPrice, 단가")
        amt_col = None
    ref = st.date_input("분석 기준일 (비우면 마지막 거래 다음 날)", value=None, key=f"ref_{file_sig}")
    drop_neg = st.checkbox("금액이 0 이하인 거래(취소·반품) 제외", value=True, key=f"neg_{file_sig}")
    rfm, log = rc.rfm_from_transactions(df, id_col, date_col,
                                        None if inv_col == "(없음)" else inv_col,
                                        amt_col, qty_col, price_col, ref, drop_neg)
    miss_info = {"total": log["start"], "dropped": log["dropped_missing"],
                 "dropped_rate": log["dropped_missing"] / max(log["start"], 1), "unit": "거래"}
    used_cols = [c for c in [id_col, date_col, inv_col, amt_col, qty_col, price_col] if c and c in df.columns]
    col_log = [("고객 ID", id_col, g_id), ("거래 일시", date_col, g_date), ("주문 번호", inv_col, g_inv or "(없음)")]
    if amt_col:
        col_log.append(("금액", amt_col, rc.guess_columns(df, "monetary")))
    else:
        col_log += [("수량", qty_col, rc.guess_columns(df, "quantity")), ("단가", price_col, rc.guess_columns(df, "price"))]
    logs["데이터"] += [f"거래 {log['start']:,}건 → 고객 {log['customers']:,}명",
                      f"분석 기준일: {log['ref_date']} ({'직접 지정' if ref else '마지막 거래 다음 날'})",
                      f"취소·반품(금액 0 이하) 제외: {log['dropped_nonpositive']:,}건" if drop_neg
                      else "취소·반품(금액 0 이하) 제외: 제외하지 않음"]
for role, chosen, guess in col_log:
    show_ = " + ".join(map(str, chosen)) if isinstance(chosen, list) else str(chosen)
    if isinstance(chosen, list):
        how = "자동 인식 그대로" if list(chosen) == list(guess) else "사용자가 바꿈"
    else:
        how = "자동 인식 그대로" if chosen == guess else f"사용자가 바꿈 (자동 인식: {guess or '없음'})"
    logs["컬럼 지정"].append(f"{role}: {show_} — {how}")

# ------------------------------------------------------------------ 결측 처리 (컬럼 지정 후 적용)
st.subheader("결측값 처리")
unit = miss_info.get("unit", "고객")
if miss_info["dropped"]:
    st.markdown(f"R·F·M 계산에 쓰는 컬럼이 비어 있는 {unit} **{miss_info['dropped']:,}{'건' if unit == '거래' else '명'}"
                f"({miss_info['dropped_rate']:.1%})** 을 제외했습니다. 컬럼은 그대로 둡니다.")
    if miss_info["dropped_rate"] >= rc.MISSING_EXCLUDE:
        st.warning(f"제외된 {unit}이 전체의 {miss_info['dropped_rate']:.0%}입니다. "
                   "결과가 전체를 대표하지 못할 수 있으니 데이터를 확인해 주세요.")
    logs["데이터 정리"].append(f"R·F·M 컬럼 결측으로 제외한 {unit}: {miss_info['dropped']:,}"
                              f"{'건' if unit == '거래' else '명'} ({miss_info['dropped_rate']:.1%})")
else:
    st.markdown("R·F·M 계산에 쓰는 컬럼에는 빈 값이 없습니다.")
    logs["데이터 정리"].append("R·F·M 컬럼 결측: 없음")
other_na = rc.other_missing(df, used_cols)
if len(other_na):
    st.markdown("R·F·M에 쓰지 않는 컬럼은 고객을 빼지 않고, 특성 비교에서만 빈 값을 빼고 계산합니다.")
    st.dataframe(other_na.style.format({"비율": "{:.1%}", "결측 수": "{:,}"}), width="content", hide_index=True)
    for _, r in other_na.iterrows():
        if r["처리"].startswith("사용 ("):
            st.warning(f"`{r['컬럼']}`은 결측이 {r['비율']:.1%}로 많아 특성 비교 결과를 해석할 때 주의가 필요해요.")
        logs["데이터 정리"].append(f"{r['컬럼']} 결측 {r['결측 수']:,}{'건' if unit == '거래' else '명'}({r['비율']:.1%}) — 제외하지 않음, {r['처리']}")

logs["데이터 정리"].append(f"중복 행 제거: {n_dup:,}행" if n_dup else "중복 행 제거: 없음")
logs["데이터 정리"].append(f"극단값 제외: {', '.join(map(str, drop_out_cols))} → {n_extreme:,}행 제외"
                         if drop_out_cols else "극단값 제외 안 함")

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

# 점수별 고객 수: 같은 값은 같은 점수를 받으므로 각 점수가 정확히 20%가 아닐 수 있다
score_tbl = rc.score_table(scored)
ties = rc.tie_conflicts(scored)
with st.expander("점수별 고객 수 보기 (구간이 고르게 나뉘었는지 확인)"):
    st.caption("R·F·M을 1~5점으로 나눌 때 같은 값의 고객은 항상 같은 점수를 받아요. 그래서 동점이 많은 값"
               "(예: 구매 횟수 5회)이 한 점수에 몰리면 각 점수의 인원이 20%에서 벗어날 수 있어요.")
    wide = {}
    for k in rc.SCORE_SOURCE:
        t = score_tbl[score_tbl["점수"] == k].set_index("점")
        wide[f"{k} 고객 수"] = [f"{n:,}명 ({r:.1%})" for n, r in zip(t["고객 수"], t["비율"])]
        wide[f"{k} 원래 값"] = t["원래 값 범위"].tolist()
    st.dataframe(pd.DataFrame(wide, index=pd.Index([f"{p}점" for p in range(1, 6)], name="점수")), width="stretch")
    st.caption(f"같은 원래 값인데 점수가 다른 고객: R {ties['R']}명 · F {ties['F']}명 · M {ties['M']}명")

# ------------------------------------------------------------------ 5. 결과
st.header("5. 세그먼트 결과")
total_m = scored["monetary"].sum()
summary = (scored.groupby("segment")
           .agg(고객수=("customer_id", "size"), 평균_R=("recency", "mean"),
                평균_F=("frequency", "mean"), 평균_M=("monetary", "mean"), 매출합=("monetary", "sum"))
           .reindex([s for s in SEG_ORDER if s in scored["segment"].unique()]))
summary["고객 비중"] = summary["고객수"] / summary["고객수"].sum()
summary["매출 비중"] = summary["매출합"] / total_m

# 고객 비중·매출 비중: 마우스를 올리면 두 값을 함께 보여준다
sm = summary.reset_index()
share_cd = np.stack([[f"{int(n):,}" for n in sm["고객수"]], sm["고객 비중"].map(fmt_pct),
                     sm["매출 비중"].map(fmt_pct)], axis=-1)
share_hover = ("세그먼트: %{x}<br>고객 수: %{customdata[0]}명 (%{customdata[1]})<br>"
               "매출 비중: %{customdata[2]}<extra></extra>")
a, b = st.columns(2)
for col, ycol, title in ((a, "고객 비중", "세그먼트별 고객 비중"), (b, "매출 비중", "세그먼트별 매출 비중")):
    with col:
        chart_title(title)
        f = go.Figure(go.Bar(x=sm["segment"], y=sm[ycol], customdata=share_cd, hovertemplate=share_hover,
                             marker_color=[SEG_COLOR[x] for x in sm["segment"]],
                             text=sm[ycol].map(fmt_pct), textposition="outside", cliponaxis=False))
        f.update_layout(xaxis_title=None, yaxis_tickformat=".0%", margin=dict(t=20))
        st.plotly_chart(f, width="stretch")

top = summary.sort_values("매출 비중", ascending=False).iloc[0]
st.markdown(f"**한눈에 보기:** `{summary['매출 비중'].idxmax()}` 고객이 전체의 {fmt_pct(top['고객 비중'])}로 "
            f"매출의 {fmt_pct(top['매출 비중'])}를 차지합니다.")

show = pd.DataFrame({
    "설명": [rc.SEGMENT_INFO[x] for x in summary.index],
    "고객 수": summary["고객수"].map("{:,}명".format),
    "고객 비중": summary["고객 비중"].map(fmt_pct),
    "매출 비중": summary["매출 비중"].map(fmt_pct),
    "평균 경과일 (R)": summary["평균_R"].map("{:,.1f}일".format),
    "평균 구매 횟수 (F)": summary["평균_F"].map("{:,.1f}회".format),
    "평균 구매 금액 (M)": summary["평균_M"].map(fmt_money),
}, index=summary.index.rename("세그먼트"))
st.dataframe(show, width="stretch")

# R·F 히트맵: 칸마다 평균 구매 금액과 고객 수
chart_title("R·F 점수별 평균 구매 금액")
st.caption("오른쪽(R↑)일수록 최근에 온 고객, 위쪽(F↑)일수록 자주 산 고객이에요. 진할수록 평균 구매 금액이 커요.")
rf = scored.groupby(["F", "R"])["monetary"].agg(["mean", "size"])
z = rf["mean"].unstack("R").reindex(index=range(1, 6), columns=range(1, 6))
n = rf["size"].unstack("R").reindex(index=range(1, 6), columns=range(1, 6)).fillna(0)
cd = np.dstack([z.map(fmt_money).to_numpy(), n.map(lambda x: f"{int(x):,}").to_numpy()])
fig3 = go.Figure(go.Heatmap(
    z=z.to_numpy(), x=list(z.columns), y=list(z.index), customdata=cd, colorscale="Blues",
    colorbar=dict(title="평균 금액"),
    hovertemplate="R 점수: %{x}<br>F 점수: %{y}<br>평균 구매 금액: %{customdata[0]}<br>"
                  "고객 수: %{customdata[1]}명<extra></extra>"))
fig3.update_layout(xaxis_title="R 점수 (최근성)", yaxis_title="F 점수 (구매 횟수)", margin=dict(t=20),
                   xaxis=dict(dtick=1), yaxis=dict(dtick=1))
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
        chosen = st.multiselect("비교할 컬럼", list(items), default=default, key=f"prof_cols_{file_sig}",
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
            pick = st.selectbox("자세히 볼 컬럼", chosen, key=f"prof_pick_{file_sig}")
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
        sums = df.loc[scored.index, money_cols].groupby(scored["segment"]).sum().reindex(summary.index)
        mix = sums.div(sums.sum(axis=1), axis=0)
        long = mix.reset_index(names="segment").melt(id_vars="segment", var_name="col", value_name="비중")
        long["금액 합계"] = [fmt_money(sums.loc[r.segment, r.col]) for r in long.itertuples()]
        long["품목"] = long["col"].map(item_name)
        long["비중 표시"] = long["비중"].map(fmt_pct)
        chart_title("세그먼트별 품목 구매 구성")
        fig5 = px.bar(long, x="segment", y="비중", color="품목", custom_data=["품목", "비중 표시", "금액 합계"])
        fig5.update_traces(hovertemplate="세그먼트: %{x}<br>품목: %{customdata[0]}<br>비중: %{customdata[1]}<br>"
                                         "금액 합계: %{customdata[2]}<extra></extra>")
        fig5.update_layout(yaxis_tickformat=".0%", xaxis_title=None, yaxis_title=None, margin=dict(t=20))
        st.plotly_chart(fig5, width="stretch")

# ------------------------------------------------------------------ 6. 처리 기록과 내려받기
logs["세그먼트 기준"] = [
    f"분석 목표: {preset} (F:M = {wf:g}:{wm:g})",
    f"균형(1:1) 대비 세그먼트가 바뀐 고객: {sens['changed']:,}명",
    "점수: R·F·M을 값 기준으로 5구간으로 나눠 1~5점 (R은 최근일수록 높음). 같은 값은 같은 점수 "
    "(동점 고객의 평균 순위로 구간을 정함)",
    "점수별 고객 수 — " + " / ".join(
        f"{k}: " + ", ".join(f"{int(r['점'])}점 {r['고객 수']:,}명({r['비율']:.1%})"
                            for _, r in score_tbl[score_tbl['점수'] == k].iterrows())
        for k in rc.SCORE_SOURCE),
    f"같은 원래 값인데 점수가 다른 고객: R {ties['R']}명, F {ties['F']}명, M {ties['M']}명",
    "가치 점수 V = (wF×F + wM×M) / (wF + wM)",
    "규칙: " + " / ".join(f"{k}: {v}" for k, v in rc.SEGMENT_RULES.items() if k != "일반") + " / 일반: 나머지",
]
if kind.startswith("고객"):
    if ref_year:
        logs["특성 비교 설정"].append(f"나이 변환 기준 연도: {ref_year}")
    if prof_methods:
        logs["특성 비교 설정"].append("컬럼별 계산 방식: " + ", ".join(f"{k} ({v})" for k, v in prof_methods.items()))
    logs["특성 비교 설정"].append("차이가 작아 문장으로 표시한 컬럼 (최대 차이 5% 미만): "
                              + (", ".join(small_cols) if small_cols else "없음"))
    if skipped:
        logs["특성 비교 설정"].append("비교에서 제외한 컬럼: " + ", ".join(f"{x['col']} ({x['reason']})" for x in skipped))
else:
    logs["특성 비교 설정"].append("거래형 데이터는 고객 특성 컬럼이 없어 특성 비교를 하지 않음")

report = "# RFM 세그먼트 분석 처리 기록\n\n" + "\n\n".join(
    f"## {i}) {sec}\n\n" + "\n".join(f"- {line}" for line in lines)
    for i, (sec, lines) in enumerate(logs.items(), 1)) + "\n"

st.header("6. 처리 기록")
st.caption("어떤 판단으로 데이터를 다뤘는지 남겨 두면 결과를 다시 검증할 수 있습니다. 보고서에 그대로 붙여 넣을 수 있어요.")
for i, (sec, lines) in enumerate(logs.items(), 1):
    st.markdown(f"**{i}) {sec}**\n\n" + "\n".join(f"- {line}" for line in lines))
with st.expander("복사용 텍스트 (오른쪽 위 버튼으로 복사)"):
    st.code(report, language="markdown")

out = scored.rename(columns={"customer_id": "고객 ID"})[
    ["고객 ID", "recency", "frequency", "monetary", "R", "F", "M", "V", "segment"]]
a, b = st.columns(2)
a.download_button("처리 기록 내려받기 (.md)", report.encode("utf-8"),
                  file_name="rfm_처리기록.md", mime="text/markdown")
b.download_button("세그먼트 결과 CSV 내려받기", out.to_csv(index=False).encode("utf-8-sig"),
                  file_name=f"rfm_segments_{preset.replace(' ', '')}.csv", mime="text/csv")

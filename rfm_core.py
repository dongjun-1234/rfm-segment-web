"""RFM 분석 핵심 로직.

웹(app.py)과 분리해 두어 화면 없이도 테스트할 수 있게 했다.
지원하는 CSV 형태
- 고객 합산형: 고객 1명당 1줄 (최근성·구매 횟수·구매 금액이 이미 계산된 상태)
- 거래형: 거래 1건당 1줄 (고객 ID, 거래 일시, 금액)에서 R·F·M을 직접 계산
"""
from __future__ import annotations

import csv
import io
import re

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- 불러오기

def read_csv_auto(raw: bytes) -> pd.DataFrame:
    """쉼표·탭·세미콜론 구분자를 자동으로 판별해 읽는다."""
    text = None
    for enc in ("utf-8-sig", "cp949", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    sample = text[:20000]
    try:
        sep = csv.Sniffer().sniff(sample, delimiters=",\t;|").delimiter
    except csv.Error:
        sep = ","
    return pd.read_csv(io.StringIO(text), sep=sep)


# ---------------------------------------------------------------- 컬럼 추정

_HINTS = {
    "customer_id": [r"^id$", r"cust", r"고객", r"member", r"user"],
    "recency": [r"recen", r"최근", r"last.*(day|purchase)"],
    "frequency": [r"num_purchase", r"freq", r"횟수", r"count", r"orders?$"],
    "monetary": [r"amount", r"amt", r"mnt", r"spend", r"금액", r"sales", r"monetary"],
    "date": [r"date", r"time", r"일시", r"날짜", r"_at$"],
    "invoice": [r"invoice", r"order.?id", r"주문", r"송장"],
    "quantity": [r"qty", r"quantity", r"수량"],
    "price": [r"price", r"단가"],
}


def guess_columns(df: pd.DataFrame, role: str, multi: bool = False):
    hits = [c for c in df.columns if any(re.search(p, str(c), re.I) for p in _HINTS[role])]
    if role in ("frequency", "monetary"):
        hits = [c for c in hits if pd.api.types.is_numeric_dtype(df[c])]
    if multi:
        return hits
    return hits[0] if hits else None


# ---------------------------------------------------------------- 데이터 품질 점검

def quality_report(df: pd.DataFrame) -> dict:
    num = df.select_dtypes("number")
    missing = df.isna().sum()
    constant = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    outliers = {}
    for c in num.columns:
        s = num[c].dropna()
        if s.nunique() <= 2:
            continue  # 0/1 같은 값은 이상값 검사에서 제외
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr == 0:
            continue
        lo, hi = q1 - 3 * iqr, q3 + 3 * iqr  # 극단값만 잡도록 3×IQR 사용
        n = int(((s < lo) | (s > hi)).sum())
        if n:
            outliers[c] = {"count": n, "low": lo, "high": hi, "min": s.min(), "max": s.max()}
    return {
        "rows": len(df),
        "cols": df.shape[1],
        "missing": missing[missing > 0].to_dict(),
        "constant": constant,
        "duplicated_rows": int(df.duplicated().sum()),
        "outliers": outliers,
    }


def drop_extreme(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, int]:
    """선택한 컬럼에서 3×IQR 밖의 극단값이 있는 행을 제거한다."""
    mask = pd.Series(True, index=df.index)
    for c in cols:
        s = df[c]
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr == 0:
            continue
        mask &= s.between(q1 - 3 * iqr, q3 + 3 * iqr) | s.isna()
    return df[mask], int((~mask).sum())


# ---------------------------------------------------------------- R·F·M 만들기

MISSING_WARN = 0.05      # RFM에 안 쓰는 컬럼: 5% 이상이면 해석 주의
MISSING_EXCLUDE = 0.20   # 20% 이상이면 특성 비교에서 기본 제외 / RFM 제외 고객이 20% 이상이면 경고


def rfm_from_customer_table(df, id_col, recency_col, freq_cols, money_cols) -> tuple[pd.DataFrame, dict]:
    """고객 합산형 표에서 R·F·M을 만든다.

    RFM에 쓰는 컬럼(고객 ID, 최근성, 빈도·금액 컬럼) 중 하나라도 비어 있는 고객은 제외한다.
    여러 컬럼을 더할 때 빈 값을 0으로 보지 않도록 min_count로 '모두 있어야 합계'를 낸다.
    """
    out = pd.DataFrame(index=df.index)
    out["customer_id"] = df[id_col] if id_col else df.index
    out["recency"] = df[recency_col]
    out["frequency"] = df[freq_cols].sum(axis=1, min_count=len(freq_cols))
    out["monetary"] = df[money_cols].sum(axis=1, min_count=len(money_cols))
    used = ([id_col] if id_col else []) + [recency_col] + list(freq_cols) + list(money_cols)
    na_by_col = df[used].isna().sum()
    keep = out.notna().all(axis=1)
    info = {"total": len(out), "dropped": int((~keep).sum()),
            "na_by_col": {c: int(n) for c, n in na_by_col.items() if n}}
    info["dropped_rate"] = info["dropped"] / max(info["total"], 1)
    return out[keep], info


def other_missing(df: pd.DataFrame, used_cols) -> pd.DataFrame:
    """RFM에 쓰지 않는 컬럼의 결측 현황과 처리 방법. 행은 제외하지 않는다."""
    rows = []
    for c in df.columns:
        if c in used_cols:
            continue
        n = int(df[c].isna().sum())
        if not n:
            continue
        rate = n / len(df)
        if rate < MISSING_WARN:
            how = "그대로 사용"
        elif rate < MISSING_EXCLUDE:
            how = "사용 (결측이 많아 해석 주의)"
        else:
            how = "특성 비교에서 기본 제외"
        rows.append({"컬럼": c, "결측 수": n, "비율": rate, "처리": how})
    return pd.DataFrame(rows, columns=["컬럼", "결측 수", "비율", "처리"])


def rfm_from_transactions(df, id_col, date_col, invoice_col=None, amount_col=None,
                          qty_col=None, price_col=None, ref_date=None,
                          drop_negative=True) -> tuple[pd.DataFrame, dict]:
    """거래 데이터에서 고객별 R·F·M을 계산한다. 정리 과정에서 제외한 건수도 함께 반환."""
    t = df.copy()
    log = {"start": len(t)}
    t[date_col] = pd.to_datetime(t[date_col], errors="coerce", dayfirst=False)
    if amount_col:
        t["_amount"] = pd.to_numeric(t[amount_col], errors="coerce")
    else:
        t["_amount"] = pd.to_numeric(t[qty_col], errors="coerce") * pd.to_numeric(t[price_col], errors="coerce")
    before = len(t)
    t = t.dropna(subset=[id_col, date_col, "_amount"])
    log["dropped_missing"] = before - len(t)
    if drop_negative:
        before = len(t)
        t = t[t["_amount"] > 0]  # 취소·반품(음수 금액) 제외
        log["dropped_nonpositive"] = before - len(t)
    ref = pd.Timestamp(ref_date) if ref_date else t[date_col].max() + pd.Timedelta(days=1)
    log["ref_date"] = ref.date().isoformat()
    g = t.groupby(id_col)
    out = pd.DataFrame({
        "recency": (ref - g[date_col].max()).dt.days,
        "frequency": g[invoice_col].nunique() if invoice_col else g[date_col].nunique(),
        "monetary": g["_amount"].sum(),
    }).reset_index().rename(columns={id_col: "customer_id"})
    log["customers"] = len(out)
    return out, log


# ---------------------------------------------------------------- 점수화와 세그먼트
#
# 세그먼트는 두 축으로 나눈다.
# - 상태 축: R 점수 (최근에 왔는가). 가중치를 주지 않는다.
# - 가치 축: V = (wF×F + wM×M) / (wF + wM). 구매 횟수와 금액 점수의 가중 평균.
# 가중치는 '가치 있는 고객'을 판단하는 기준이므로 가치 축(V)에만 쓴다.
# R에 가중치를 주면 최근에 온 고객이 위로 올라올 뿐 이탈 고객을 찾는 데 도움이 되지 않는다.

PRESETS = {
    "균형": {"weights": (1.0, 1.0), "why": "구매 횟수와 금액을 똑같이 봐요"},
    "재구매 중시": {"weights": (3.0, 1.0), "why": "자주 오는 고객을 더 중요하게 봐요"},
    "매출 중시": {"weights": (1.0, 3.0), "why": "많이 쓰는 고객을 더 중요하게 봐요"},
}
BASE_WEIGHTS = PRESETS["균형"]["weights"]


def value_score(s: pd.Series, bins: int = 5, ascending: bool = True) -> pd.Series:
    """값을 1~bins점으로 바꾼다. 같은 값이면 반드시 같은 점수.

    동점인 고객들은 평균 순위(백분위)를 함께 쓰고, 그 백분위가 속한 구간의 점수를 받는다.
    그래서 구간 경계는 서로 다른 값 사이에서만 생기고, 각 점수의 인원은 정확히 20%가 아닐 수 있다.
    (예전 방식은 rank(method="first")로 동점자를 데이터 순서대로 잘라 같은 값이 다른 점수를 받았다)
    ascending=False면 값이 작을수록 높은 점수 (R: 최근일수록 높음).
    """
    pct = s.rank(method="average", ascending=ascending, pct=True)
    return np.ceil(pct * bins).clip(1, bins).astype(int)


def score(rfm: pd.DataFrame, weights=BASE_WEIGHTS, bins: int = 5) -> pd.DataFrame:
    """R·F·M을 1~bins점으로 점수화하고, 가치 점수 V와 세그먼트를 붙인다.

    R은 최근일수록(값이 작을수록) 높은 점수. 같은 값은 같은 점수(value_score 참고).
    weights는 (wF, wM) 두 개. 합이 0이면 1:1로 본다.
    """
    r = rfm.copy()
    r["R"] = value_score(r["recency"], bins, ascending=False)
    r["F"] = value_score(r["frequency"], bins)
    r["M"] = value_score(r["monetary"], bins)
    wf, wm = (float(x) for x in weights)
    if wf + wm <= 0:
        wf, wm = BASE_WEIGHTS
    r["V"] = ((wf * r["F"] + wm * r["M"]) / (wf + wm)).round(2)
    r["segment"] = segment_rules(r["R"], r["V"], bins)
    return r


SCORE_SOURCE = {"R": "recency", "F": "frequency", "M": "monetary"}


def score_table(scored: pd.DataFrame, bins: int = 5) -> pd.DataFrame:
    """점수별 고객 수와 그 점수에 해당하는 원래 값 범위. 구간이 고르게 나뉘었는지 확인용."""
    rows = []
    for k, raw in SCORE_SOURCE.items():
        g = scored.groupby(k)[raw]
        for p in range(1, bins + 1):
            n = int((scored[k] == p).sum())
            rng = f"{g.min()[p]:,.0f} ~ {g.max()[p]:,.0f}" if n else "-"
            rows.append({"점수": k, "점": p, "고객 수": n, "비율": n / len(scored), "원래 값 범위": rng})
    return pd.DataFrame(rows)


def tie_conflicts(scored: pd.DataFrame) -> dict:
    """같은 원래 값인데 점수가 다른 고객 수. 0이어야 정상."""
    return {k: int((scored.groupby(raw)[k].transform("nunique") > 1).sum()) for k, raw in SCORE_SOURCE.items()}


SEGMENT_INFO = {
    "VIP": "최근에도 오고, 가치(구매 횟수·금액)도 높은 핵심 고객",
    "충성 고객": "가치가 높고 최근성은 중간인 고객",
    "이탈 위험 우수고객": "가치가 높지만 최근 발길이 끊긴 고객",
    "잠재 성장": "최근에 왔지만 아직 가치가 낮은 고객",
    "휴면": "오래 안 왔고 가치도 낮은 고객",
    "일반": "어느 쪽에도 뚜렷하게 속하지 않는 고객",
}

SEGMENT_RULES = {
    "VIP": "V ≥ 4 그리고 R ≥ 4",
    "충성 고객": "V ≥ 4 그리고 R = 3",
    "이탈 위험 우수고객": "V ≥ 4 그리고 R ≤ 2",
    "잠재 성장": "V ≤ 2 그리고 R ≥ 4",
    "휴면": "V ≤ 2 그리고 R ≤ 2",
    "일반": "위에 해당하지 않음",
}


def segment_rules(R, V, bins=5) -> pd.Series:
    """상태 축(R)과 가치 축(V)의 조합으로 세그먼트를 정한다."""
    hi = bins - 1   # 5점 척도에서 4점 이상
    lo = 2          # 2점 이하
    cond = [
        (V >= hi) & (R >= hi),
        (V >= hi) & (R == 3),
        (V >= hi) & (R <= lo),
        (V <= lo) & (R >= hi),
        (V <= lo) & (R <= lo),
    ]
    names = ["VIP", "충성 고객", "이탈 위험 우수고객", "잠재 성장", "휴면"]
    return pd.Series(np.select(cond, names, default="일반"), index=R.index)


def weight_sensitivity(rfm: pd.DataFrame, weights, base=BASE_WEIGHTS) -> dict:
    """균형(1:1) 대비 세그먼트가 바뀐 고객 수와 세그먼트별 고객 수를 비교한다."""
    a = score(rfm, weights)
    b = score(rfm, base)
    counts = pd.DataFrame({"균형 (1:1)": b["segment"].value_counts(),
                           "현재": a["segment"].value_counts()}).fillna(0).astype(int)
    counts["차이"] = counts["현재"] - counts["균형 (1:1)"]
    return {"changed": int((a["segment"] != b["segment"]).sum()), "counts": counts}


# ---------------------------------------------------------------- 세그먼트별 특성 비교
#
# 세그먼트 값을 '전체 고객 값 대비 몇 % 높고 낮은지'로 바꿔 차이가 눈에 보이게 한다.
# 컬럼 종류에 따라 계산 방식을 다르게 한다.
#   age      출생연도 → 나이로 바꿔 평균
#   median   소득·금액처럼 한쪽으로 치우친 값 → 중앙값
#   mean     그 밖의 숫자 → 평균
#   rate     0/1 컬럼 → 참여율
#   share    빈도 계산에서 뺀 횟수 컬럼(할인 구매 등) → 전체 구매 중 비중
#   category 문자열 범주 → 세그먼트별 구성 비율

_BIRTH = re.compile(r"birth|year|출생|생년", re.I)
_MONEY = re.compile(r"income|salary|amount|amt|mnt|spend|sales|금액|소득|매출", re.I)
_DATE_LIKE = re.compile(r"(^|_)(ym|ymd|yyyymm|date|dt)$|date|signup|join|가입|연월|일자|_at$", re.I)
_PROMO = re.compile(r"promo|campaign|accepted|프로모션|캠페인", re.I)
SMALL_DIFF = 0.05   # 세그먼트 간 최대 차이가 5% 미만이면 그래프 대신 문장으로


def profile_specs(df: pd.DataFrame, used_cols, freq_cols) -> tuple[list[dict], list[dict]]:
    """비교할 컬럼과 계산 방식을 정한다. (비교 목록, 제외 목록[이유 포함])"""
    specs, skipped = [], []
    id_pat = re.compile("|".join(_HINTS["customer_id"]), re.I)
    freq_pat = re.compile("|".join(_HINTS["frequency"]), re.I)
    for c in df.columns:
        if c in used_cols:
            continue
        s = df[c].dropna()
        name = str(c)
        if s.nunique() <= 1:
            skipped.append({"col": c, "reason": "값이 하나뿐"})
            continue
        if not pd.api.types.is_numeric_dtype(s):
            if s.nunique() <= 12:
                specs.append({"col": c, "kind": "category"})
            else:
                skipped.append({"col": c, "reason": "범주가 너무 많음 (ID·날짜성)"})
            continue
        if id_pat.search(name) or (s.nunique() == len(s) and len(s) > 50 and (s % 1 == 0).all()):
            skipped.append({"col": c, "reason": "ID성 컬럼"})
            continue
        if _DATE_LIKE.search(name) and not _BIRTH.search(name):
            skipped.append({"col": c, "reason": "날짜성 컬럼"})
            continue
        if set(s.unique()) <= {0, 1}:
            specs.append({"col": c, "kind": "rate", "promo": bool(_PROMO.search(name))})
        elif _BIRTH.search(name) and s.between(1900, 2010).mean() > 0.95:
            specs.append({"col": c, "kind": "age"})
        elif freq_pat.search(name) and freq_cols:
            specs.append({"col": c, "kind": "share"})
        elif _MONEY.search(name) or (s.nunique() > 10 and s.skew() > 1):
            specs.append({"col": c, "kind": "median", "money": bool(_MONEY.search(name))})
        else:
            specs.append({"col": c, "kind": "mean"})
    return specs, skipped


def _stat(x: pd.DataFrame, spec, ref_year):
    c, k = spec["col"], spec["kind"]
    if k == "age":
        return (ref_year - x[c]).mean()
    if k == "median":
        return x[c].median()
    if k == "share":
        sub = x[[c, "_freq"]].dropna()
        return sub[c].sum() / sub["_freq"].sum() if sub["_freq"].sum() else np.nan
    return x[c].mean()   # mean, rate


def profile_values(df: pd.DataFrame, segment: pd.Series, spec: dict, ref_year=None,
                   frequency: pd.Series | None = None, order=None) -> pd.DataFrame:
    """세그먼트별 값, 전체 값, 전체 대비 차이(비율), 값이 있는 고객 수."""
    c = spec["col"]
    x = df[[c]].copy()
    if frequency is not None:
        x["_freq"] = frequency
    x = x[x[c].notna()]   # 결측은 특성 비교에서만 뺀다
    seg = segment.reindex(x.index)
    overall = _stat(x, spec, ref_year)
    rows = []
    for name, g in x.groupby(seg):
        v = _stat(g, spec, ref_year)
        rows.append({"segment": name, "value": v, "n": len(g),
                     "diff": v / overall - 1 if overall else np.nan})
    out = pd.DataFrame(rows).set_index("segment")
    if order is not None:
        out = out.reindex([s for s in order if s in out.index])
    out.attrs["overall"] = overall
    out.attrs["n_total"] = len(x)
    return out


def promo_rates(df: pd.DataFrame, segment: pd.Series, cols, order=None) -> pd.DataFrame:
    """프로모션(0/1) 컬럼별 참여율과 '하나라도 참여한 비율'."""
    x = df[cols].copy()
    x["하나라도 참여"] = (x.fillna(0).sum(axis=1) > 0).astype(int)
    out = x.groupby(segment.reindex(x.index)).mean()
    out.loc["전체"] = x.mean()
    if order is not None:
        out = out.reindex([s for s in order if s in out.index] + ["전체"])
    return out


def category_mix(df: pd.DataFrame, segment: pd.Series, col, order=None) -> pd.DataFrame:
    """범주형 컬럼의 세그먼트별 구성 비율 (행 합 = 1)."""
    x = df[col].dropna()
    out = pd.crosstab(segment.reindex(x.index), x, normalize="index")
    if order is not None:
        out = out.reindex([s for s in order if s in out.index])
    return out

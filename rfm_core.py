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

def rfm_from_customer_table(df, id_col, recency_col, freq_cols, money_cols) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["customer_id"] = df[id_col] if id_col else df.index
    out["recency"] = df[recency_col]
    out["frequency"] = df[freq_cols].sum(axis=1)
    out["monetary"] = df[money_cols].sum(axis=1)
    return out


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


def score(rfm: pd.DataFrame, weights=BASE_WEIGHTS, bins: int = 5) -> pd.DataFrame:
    """R·F·M을 1~bins점으로 점수화하고, 가치 점수 V와 세그먼트를 붙인다.

    동점이 많아도 구간이 고르게 나뉘도록 순위 기준으로 자른다.
    R은 최근일수록(값이 작을수록) 높은 점수.
    weights는 (wF, wM) 두 개. 합이 0이면 1:1로 본다.
    """
    r = rfm.copy()
    labels = list(range(1, bins + 1))
    r["R"] = pd.qcut(r["recency"].rank(method="first", ascending=False), bins, labels=labels).astype(int)
    r["F"] = pd.qcut(r["frequency"].rank(method="first"), bins, labels=labels).astype(int)
    r["M"] = pd.qcut(r["monetary"].rank(method="first"), bins, labels=labels).astype(int)
    wf, wm = (float(x) for x in weights)
    if wf + wm <= 0:
        wf, wm = BASE_WEIGHTS
    r["V"] = ((wf * r["F"] + wm * r["M"]) / (wf + wm)).round(2)
    r["segment"] = segment_rules(r["R"], r["V"], bins)
    return r


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

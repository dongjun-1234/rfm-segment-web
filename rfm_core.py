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

def score(rfm: pd.DataFrame, weights=(1.0, 1.0, 1.0), bins: int = 5) -> pd.DataFrame:
    """R·F·M을 1~bins점으로 점수화하고 가중 점수를 계산한다.

    동점이 많아도 구간이 고르게 나뉘도록 순위 기준으로 자른다.
    R은 최근일수록(값이 작을수록) 높은 점수.
    """
    r = rfm.copy()
    labels = list(range(1, bins + 1))
    r["R"] = pd.qcut(r["recency"].rank(method="first", ascending=False), bins, labels=labels).astype(int)
    r["F"] = pd.qcut(r["frequency"].rank(method="first"), bins, labels=labels).astype(int)
    r["M"] = pd.qcut(r["monetary"].rank(method="first"), bins, labels=labels).astype(int)
    w = np.array(weights, dtype=float)
    w = w / w.sum() if w.sum() > 0 else np.ones(3) / 3
    r["RFM_score"] = (r[["R", "F", "M"]].to_numpy() * w).sum(axis=1).round(2)
    r["value_tier"] = pd.qcut(r["RFM_score"].rank(method="first", ascending=False), [0, .2, .5, .8, 1.0],
                              labels=["상위 20%", "20~50%", "50~80%", "하위 20%"])
    r["segment"] = segment_rules(r["R"], r["F"], r["M"], bins)
    return r


SEGMENT_INFO = {
    "VIP": "최근에도 자주, 많이 사는 핵심 고객",
    "충성 고객": "자주 사지만 금액이나 최근성은 VIP보다 조금 낮은 고객",
    "이탈 위험 우수고객": "예전엔 자주·많이 샀지만 최근 발길이 끊긴 고객",
    "잠재 성장": "최근에 왔지만 아직 구매가 적은 고객",
    "휴면": "오래 안 왔고 구매도 적은 고객",
    "일반": "어느 쪽에도 뚜렷하게 속하지 않는 고객",
}


def segment_rules(R, F, M, bins=5) -> pd.Series:
    hi = bins - 1   # 5점 척도에서 4점 이상
    lo = 2          # 2점 이하
    FM = (F + M) / 2
    cond = [
        (R >= hi) & (F >= hi) & (M >= hi),
        (R >= 3) & (FM >= hi),
        (R <= lo) & (FM >= hi),
        (R >= hi) & (FM <= lo + 0.5),
        (R <= lo) & (FM <= lo),
    ]
    names = ["VIP", "충성 고객", "이탈 위험 우수고객", "잠재 성장", "휴면"]
    return pd.Series(np.select(cond, names, default="일반"), index=R.index)


def weight_sensitivity(rfm: pd.DataFrame, weights, base=(1, 1, 1)) -> dict:
    """선택한 가중치와 기준 가중치에서 '상위 20%' 고객이 얼마나 겹치는지 비교."""
    a = score(rfm, weights)
    b = score(rfm, base)
    top_a = set(a.loc[a["value_tier"] == "상위 20%", "customer_id"])
    top_b = set(b.loc[b["value_tier"] == "상위 20%", "customer_id"])
    inter = len(top_a & top_b)
    return {"top_n": len(top_a), "overlap": inter,
            "overlap_rate": inter / max(len(top_a), 1),
            "new_in": len(top_a - top_b), "dropped_out": len(top_b - top_a)}

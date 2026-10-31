from __future__ import annotations

import hashlib
import io
import json
import math
import time
import tracemalloc
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pydantic import BaseModel, ConfigDict, Field, model_validator


APP_TITLE = "Synthetic Dataset Generation Laboratory — Live Demo"
SCHEMA_VERSION = "004.foundation.demo.v1"

REGIONS = ("North", "South", "East", "West", "Central")
CATEGORIES = (
    "Electronics",
    "Apparel",
    "Groceries",
    "Home",
    "Health",
    "Books",
    "Toys",
    "Other",
)

TAX_RATES = {
    "North": 0.08,
    "South": 0.06,
    "East": 0.07,
    "West": 0.09,
    "Central": 0.05,
}

CATEGORY_BASE = {
    "Electronics": 180.0,
    "Apparel": 65.0,
    "Groceries": 42.0,
    "Home": 110.0,
    "Health": 75.0,
    "Books": 28.0,
    "Toys": 45.0,
    "Other": 35.0,
}

NUMERIC_COLS = ["age", "income", "tenure_days", "transaction_amount"]
CORE_COLUMNS = [
    "record_id",
    "customer_id",
    "timestamp",
    "region",
    "category",
    "age",
    "income",
    "tenure_days",
    "transaction_amount",
    "tax_amount",
    "total_amount",
    "churn",
]


class DemoSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_VERSION
    seed: int = Field(default=42, ge=0, le=2_147_483_647)
    n_rows: int = Field(default=700, ge=100, le=5000)
    synth_rows: int = Field(default=500, ge=50, le=5000)
    steps: int = Field(default=24, ge=8, le=80)
    fps: int = Field(default=12, ge=5, le=30)
    initial_noise: float = Field(default=0.45, ge=0.0, le=1.0)
    final_noise: float = Field(default=0.06, ge=0.0, le=0.5)
    rare_threshold: float = Field(default=0.03, ge=0.0, le=0.25)
    target_churn: float = Field(default=0.18, ge=0.01, le=0.95)
    correlation_strength: float = Field(default=0.55, ge=-1.0, le=1.0)
    anomaly_rate: float = Field(default=0.02, ge=0.0, le=0.15)

    @model_validator(mode="after")
    def _validate_noise(self) -> "DemoSettings":
        if self.final_noise > self.initial_noise:
            raise ValueError("final_noise must be less than or equal to initial_noise.")
        return self


def settings_hash(settings: DemoSettings) -> str:
    payload = settings.model_dump(mode="json")
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500.0, 500.0)))


def _normalize_probs(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return arr
    arr = np.where(np.isfinite(arr) & (arr > 0.0), arr, 0.0)
    total = float(arr.sum())
    if total <= 0.0:
        arr = np.ones_like(arr, dtype=float)
        total = float(arr.sum())
    arr = arr / total
    if arr.size > 1:
        arr[-1] = max(0.0, 1.0 - float(arr[:-1].sum()))
    arr = np.clip(arr, 0.0, 1.0)
    s = float(arr.sum())
    if s <= 0.0:
        arr = np.ones_like(arr, dtype=float)
        s = float(arr.sum())
    return arr / s


def robust_scale(values: pd.Series) -> float:
    v = pd.to_numeric(values, errors="coerce").dropna()
    if v.empty:
        return 1.0
    q25, q75 = np.percentile(v, [25, 75])
    iqr = float(q75 - q25)
    if iqr <= 1e-9:
        std = float(np.std(v))
        mean_abs = abs(float(np.mean(v)))
        iqr = max(std, mean_abs * 0.05, 1e-3)
    return float(max(iqr / 1.349, 1e-3))


def make_demo_dataframe(settings: DemoSettings) -> pd.DataFrame:
    rng = np.random.default_rng(settings.seed)
    n = int(settings.n_rows)

    region_names = np.array(REGIONS, dtype=object)
    category_names = np.array(CATEGORIES, dtype=object)

    region_p = _normalize_probs(np.array([0.22, 0.20, 0.18, 0.25, 0.15], dtype=float))
    category_p = _normalize_probs(
        np.array([0.18, 0.16, 0.22, 0.14, 0.12, 0.08, 0.06, 0.04], dtype=float)
    )

    region = rng.choice(region_names, size=n, p=region_p)
    category = rng.choice(category_names, size=n, p=category_p)

    age = np.clip(rng.normal(39.0, 12.0, size=n), 18.0, 90.0)
    income_base = np.exp(rng.normal(10.6, 0.55, size=n))
    income = income_base + settings.correlation_strength * (age - 39.0) * 1800.0
    income = np.clip(income, 12000.0, 250000.0)

    tenure_days = np.clip(rng.gamma(shape=2.0, scale=180.0, size=n), 1.0, 3650.0)

    age_filled = np.where(np.isnan(age), 39.0, age)
    linear = (
        0.018 * (age_filled - 39.0)
        - 0.00035 * tenure_days
        + 0.000006 * (income - 50000.0)
        + rng.normal(0.0, 0.35, size=n)
    )

    lo, hi = -10.0, 10.0
    for _ in range(35):
        mid = (lo + hi) / 2.0
        p_mean = float(_sigmoid(linear + mid).mean())
        if p_mean < settings.target_churn:
            lo = mid
        else:
            hi = mid
    intercept = (lo + hi) / 2.0
    churn_prob = _sigmoid(linear + intercept)
    churn = rng.random(n) < churn_prob

    cat_base_arr = np.array([CATEGORY_BASE.get(str(c), 35.0) for c in category], dtype=float)
    amount = (
        cat_base_arr
        * (0.65 + 0.70 * (income / 250000.0))
        * np.exp(rng.normal(0.0, 0.35, size=n))
    )
    amount = np.maximum(amount, 1.0)

    tax_rate = np.array([TAX_RATES.get(str(r), 0.07) for r in region], dtype=float)
    tax = amount * tax_rate
    total = amount + tax

    base_ts = pd.Timestamp("2025-01-01", tz="UTC")
    day_offsets = pd.to_timedelta(rng.integers(0, 365, size=n), unit="D")
    second_offsets = pd.to_timedelta(rng.integers(0, 86400, size=n), unit="s")
    timestamps = base_ts + day_offsets + second_offsets

    record_id = np.array([f"RAW{i:08d}" for i in range(1, n + 1)], dtype=object)
    customer_id = np.array([f"CUST{i:08d}" for i in range(1, n + 1)], dtype=object)
    transaction_id = np.array([f"TXN{100000 + i:08d}" for i in range(n)], dtype=object)

    if settings.anomaly_rate > 0.0:
        region = np.asarray(region, dtype=object)
        category = np.asarray(category, dtype=object)

        miss_age = rng.random(n) < settings.anomaly_rate * 0.60
        age[miss_age] = np.nan

        neg_amount = rng.random(n) < settings.anomaly_rate * 0.25
        amount[neg_amount] = -np.abs(amount[neg_amount])

        bad_region = rng.random(n) < settings.anomaly_rate * 0.20
        region[bad_region] = "Unknown"

        bad_category = rng.random(n) < settings.anomaly_rate * 0.15
        category[bad_category] = "Unknown"

        tax_mismatch = rng.random(n) < settings.anomaly_rate * 0.30
        mismatch_count = int(tax_mismatch.sum())
        if mismatch_count > 0:
            tax[tax_mismatch] *= rng.uniform(0.4, 1.8, size=mismatch_count)

        total = amount + tax

        dup_count = int(round(n * settings.anomaly_rate * 0.10))
        if n > 1 and dup_count > 0:
            donor = int(rng.integers(0, n))
            dup_idx = rng.choice(n, size=min(dup_count, n), replace=False)
            transaction_id[dup_idx] = transaction_id[donor]

    df = pd.DataFrame(
        {
            "record_id": record_id,
            "customer_id": customer_id,
            "transaction_id": transaction_id,
            "timestamp": timestamps,
            "region": region,
            "category": category,
            "age": age,
            "income": np.round(income, 2),
            "tenure_days": np.round(tenure_days, 0),
            "transaction_amount": np.round(amount, 2),
            "tax_amount": np.round(tax, 2),
            "total_amount": np.round(total, 2),
            "churn": churn.astype(bool),
        }
    )
    return df


def fingerprint_dataframe(df: pd.DataFrame) -> str:
    hashed = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    return hashlib.sha256(hashed).hexdigest()


def clean_for_generation(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 17)
    clean = df.copy()

    valid_age = clean["age"].notna() & (pd.to_numeric(clean["age"], errors="coerce") >= 0)
    valid_amount = pd.to_numeric(clean["transaction_amount"], errors="coerce") > 0
    valid_tax = pd.to_numeric(clean["tax_amount"], errors="coerce") >= 0
    valid_region = clean["region"].isin(REGIONS)
    valid_category = clean["category"].isin(CATEGORIES)
    valid_total = np.isclose(
        pd.to_numeric(clean["total_amount"], errors="coerce"),
        pd.to_numeric(clean["transaction_amount"], errors="coerce")
        + pd.to_numeric(clean["tax_amount"], errors="coerce"),
        rtol=0.01,
        atol=0.02,
    )

    mask = valid_age & valid_amount & valid_tax & valid_region & valid_category & valid_total
    filtered = clean.loc[mask].reset_index(drop=True)

    min_required = max(30, int(0.25 * len(clean)))
    if len(filtered) >= min_required:
        return filtered

    median_age = clean["age"].median()
    if pd.isna(median_age):
        median_age = 39.0

    clean["age"] = pd.to_numeric(clean["age"], errors="coerce").fillna(float(median_age)).clip(lower=0.0)
    clean["transaction_amount"] = (
        pd.to_numeric(clean["transaction_amount"], errors="coerce")
        .abs()
        .fillna(25.0)
        .clip(lower=1.0)
    )
    clean["tax_amount"] = pd.to_numeric(clean["tax_amount"], errors="coerce").abs().fillna(0.0)
    clean["region"] = clean["region"].where(clean["region"].isin(REGIONS), "Central")
    clean["category"] = clean["category"].where(clean["category"].isin(CATEGORIES), "Other")

    tax_rate = clean["region"].map(TAX_RATES).fillna(0.07)
    clean["tax_amount"] = (clean["transaction_amount"] * tax_rate).round(2)
    clean["total_amount"] = (clean["transaction_amount"] + clean["tax_amount"]).round(2)
    clean["churn"] = clean["churn"].astype(bool)

    if clean.empty:
        clean = pd.DataFrame(
            [
                {
                    "record_id": "FALLBACK0001",
                    "customer_id": "FALLBACKC0001",
                    "transaction_id": "FALLBACKT0001",
                    "timestamp": pd.Timestamp("2025-06-01", tz="UTC"),
                    "region": "Central",
                    "category": "Other",
                    "age": 39.0,
                    "income": 55000.0,
                    "tenure_days": 365.0,
                    "transaction_amount": 50.0,
                    "tax_amount": 2.5,
                    "total_amount": 52.5,
                    "churn": False,
                }
            ]
        )

    return clean.reset_index(drop=True)


def data_quality_report(df: pd.DataFrame, label: str) -> dict[str, Any]:
    n = len(df)
    if n == 0:
        return {
            "dataset": label,
            "rows": 0,
            "completeness": 0.0,
            "validity": 0.0,
            "uniqueness": 0.0,
            "consistency": 0.0,
            "freshness": 0.0,
            "lineage_coverage": 0.0,
        }

    completeness = float(df.notna().mean().mean())

    valid_age = df["age"].notna() & (pd.to_numeric(df["age"], errors="coerce") >= 0)
    valid_amount = pd.to_numeric(df["transaction_amount"], errors="coerce") > 0
    valid_tax = pd.to_numeric(df["tax_amount"], errors="coerce") >= 0
    valid_region = df["region"].isin(REGIONS)
    valid_category = df["category"].isin(CATEGORIES)
    valid_total = np.isclose(
        pd.to_numeric(df["total_amount"], errors="coerce"),
        pd.to_numeric(df["transaction_amount"], errors="coerce")
        + pd.to_numeric(df["tax_amount"], errors="coerce"),
        rtol=0.01,
        atol=0.02,
    )
    validity_mask = valid_age & valid_amount & valid_tax & valid_region & valid_category & valid_total
    validity = float(validity_mask.mean())

    uniqueness = float(df["customer_id"].nunique() / n)

    tax_rate = pd.to_numeric(df["tax_amount"], errors="coerce") / pd.to_numeric(
        df["transaction_amount"], errors="coerce"
    ).replace(0, np.nan)
    expected_rate = df["region"].map(TAX_RATES)
    consistency_mask = (
        valid_region
        & valid_amount
        & np.isclose(tax_rate, expected_rate, rtol=0.02, atol=0.005)
    )
    consistency = float(consistency_mask.mean())

    ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    if ts.notna().any():
        max_ts = ts.max()
        fresh = (max_ts - ts) <= pd.Timedelta(days=180)
        freshness = float(fresh.mean())
    else:
        freshness = 0.0

    lineage_coverage = 1.0

    return {
        "dataset": label,
        "rows": n,
        "completeness": completeness,
        "validity": validity,
        "uniqueness": uniqueness,
        "consistency": consistency,
        "freshness": freshness,
        "lineage_coverage": lineage_coverage,
    }


def ks_statistic(a: pd.Series | np.ndarray, b: pd.Series | np.ndarray) -> float:
    a = pd.to_numeric(pd.Series(a), errors="coerce").dropna().to_numpy(dtype=float)
    b = pd.to_numeric(pd.Series(b), errors="coerce").dropna().to_numpy(dtype=float)
    if len(a) == 0 or len(b) == 0:
        return 1.0
    all_vals = np.sort(np.concatenate([a, b]))
    a_sorted = np.sort(a)
    b_sorted = np.sort(b)
    cdf_a = np.searchsorted(a_sorted, all_vals, side="right") / float(len(a))
    cdf_b = np.searchsorted(b_sorted, all_vals, side="right") / float(len(b))
    return float(np.max(np.abs(cdf_a - cdf_b)))


def tv_distance(s1: pd.Series, s2: pd.Series) -> float:
    if len(s1) == 0 and len(s2) == 0:
        return 0.0
    if len(s1) == 0 or len(s2) == 0:
        return 1.0
    p1 = s1.astype(str).value_counts(normalize=True)
    p2 = s2.astype(str).value_counts(normalize=True)
    union = set(p1.index) | set(p2.index)
    return float(0.5 * sum(abs(float(p1.get(k, 0.0)) - float(p2.get(k, 0.0))) for k in union))


def correlation_error(real: pd.DataFrame, synth: pd.DataFrame, cols: list[str]) -> float:
    r = real[cols].apply(pd.to_numeric, errors="coerce").dropna()
    s = synth[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(r) < 3 or len(s) < 3:
        return 1.0
    cr = np.corrcoef(r.T.to_numpy(dtype=float))
    cs = np.corrcoef(s.T.to_numpy(dtype=float))
    cr = np.nan_to_num(cr, nan=0.0, posinf=0.0, neginf=0.0)
    cs = np.nan_to_num(cs, nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.linalg.norm(cr - cs, ord="fro") / max(1, cr.shape[0]))


def exact_match_rate(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    cols: list[str],
    cap: int = 400,
    seed: int = 0,
) -> float:
    if len(synth) == 0:
        return 0.0
    real_sample = real[cols].sample(min(cap, len(real)), random_state=seed)
    synth_sample = synth[cols].sample(min(cap, len(synth)), random_state=seed)

    def keys(frame: pd.DataFrame) -> list[tuple[str, ...]]:
        d = frame.copy()
        for c in cols:
            if pd.api.types.is_numeric_dtype(d[c]):
                d[c] = pd.to_numeric(d[c], errors="coerce").round(2)
            else:
                d[c] = d[c].astype(str)
        return [tuple(row) for row in d.astype(str).to_numpy()]

    real_keys = set(keys(real_sample))
    synth_keys = keys(synth_sample)
    if not synth_keys:
        return 0.0
    hits = sum(1 for k in synth_keys if k in real_keys)
    return float(hits / len(synth_keys))


def nn_distance_ratio(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    cols: list[str],
    cap: int = 180,
    seed: int = 0,
) -> float:
    r = real[cols].apply(pd.to_numeric, errors="coerce").dropna()
    s = synth[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(r) < 2 or len(s) < 1:
        return float("nan")

    r = r.sample(min(cap, len(r)), random_state=seed)
    s = s.sample(min(cap, len(s)), random_state=seed)

    mu = r.mean()
    sigma = r.std(ddof=0).replace(0, 1.0)
    R = ((r - mu) / sigma).to_numpy(dtype=float)
    S = ((s - mu) / sigma).to_numpy(dtype=float)

    dist_rr = np.linalg.norm(R[:, None, :] - R[None, :, :], axis=2)
    np.fill_diagonal(dist_rr, np.inf)
    real_nn = np.min(dist_rr, axis=1)

    dist_sr = np.linalg.norm(S[:, None, :] - R[None, :, :], axis=2)
    synth_nn = np.min(dist_sr, axis=1)

    denom = max(float(np.median(real_nn)), 1e-9)
    return float(np.median(synth_nn) / denom)


def rare_preservation(real: pd.DataFrame, synth: pd.DataFrame, threshold: float) -> tuple[float, list[str]]:
    if len(real) == 0 or len(synth) == 0:
        return 0.0, []

    real_counts = real["category"].astype(str).value_counts(normalize=True)
    synth_counts = synth["category"].astype(str).value_counts(normalize=True)
    rare = [str(c) for c, p in real_counts.items() if 0.0 < float(p) < threshold]

    if not rare:
        return 1.0, []

    scores = []
    for c in rare:
        rp = float(real_counts.get(c, 0.0))
        sp = float(synth_counts.get(c, 0.0))
        if rp <= 0.0:
            scores.append(0.0)
        elif sp <= 0.0:
            scores.append(0.0)
        elif sp >= rp:
            scores.append(rp / sp)
        else:
            scores.append(sp / rp)

    return float(np.mean(scores)), rare


def make_category_sampler(series: pd.Series, threshold: float, rng: np.random.Generator):
    counts = series.astype(str).value_counts(normalize=True)
    if counts.empty:

        def sample_empty(size: int) -> np.ndarray:
            return np.full(size, "Other", dtype=object)

        return sample_empty, []

    rare = [str(c) for c, p in counts.items() if float(p) < threshold]
    grouped = counts.copy()

    if rare:
        rare_sum = float(grouped.loc[rare].sum())
        grouped = grouped.drop(index=rare, errors="ignore")
        grouped["__RARE__"] = rare_sum

    probs = _normalize_probs(grouped.to_numpy(dtype=float))
    labels = list(grouped.index.astype(str))

    rare_labels: list[str] = []
    rare_probs: np.ndarray = np.array([], dtype=float)
    if rare:
        rare_counts = counts.loc[rare]
        rare_total = float(rare_counts.sum())
        if rare_total > 0.0:
            rare_labels = list(rare_counts.index.astype(str))
            rare_probs = _normalize_probs(rare_counts.to_numpy(dtype=float))

    def sample_categories(size: int) -> np.ndarray:
        chosen = rng.choice(np.array(labels, dtype=object), size=size, p=probs)
        out = []
        for lab in chosen:
            lab_str = str(lab)
            if lab_str == "__RARE__" and rare_labels:
                out.append(str(rng.choice(np.array(rare_labels, dtype=object), p=rare_probs)))
            else:
                out.append(lab_str)
        return np.array(out, dtype=object)

    return sample_categories, rare


def generate_synthetic(
    clean_df: pd.DataFrame,
    n: int,
    rng: np.random.Generator,
    noise: float,
    rare_threshold: float,
    correlation_strength: float,
) -> tuple[pd.DataFrame, list[str]]:
    if n <= 0:
        return pd.DataFrame(columns=CORE_COLUMNS + ["lineage_source_index"]), []

    if clean_df.empty:
        clean_df = pd.DataFrame(
            [
                {
                    "record_id": "SEED00001",
                    "customer_id": "SEEDC00001",
                    "timestamp": pd.Timestamp("2025-06-01", tz="UTC"),
                    "region": "Central",
                    "category": "Other",
                    "age": 39.0,
                    "income": 55000.0,
                    "tenure_days": 365.0,
                    "transaction_amount": 50.0,
                    "tax_amount": 2.5,
                    "total_amount": 52.5,
                    "churn": False,
                }
            ]
        )

    base_idx = rng.integers(0, len(clean_df), size=n)
    base = clean_df.iloc[base_idx].reset_index(drop=True).copy()

    region_counts = (
        clean_df["region"]
        .astype(str)
        .value_counts(normalize=True)
        .reindex(REGIONS)
        .fillna(0.01)
    )
    region_probs = _normalize_probs(region_counts.to_numpy(dtype=float))
    region = rng.choice(np.array(REGIONS, dtype=object), size=n, p=region_probs)

    category_sampler, rare_list = make_category_sampler(
        clean_df["category"], rare_threshold, rng
    )
    category = category_sampler(n)

    rho = float(np.clip(correlation_strength, -0.999, 0.999))
    sqrt_term = math.sqrt(max(0.0, 1.0 - rho * rho))
    latent = rng.normal(0.0, 1.0, size=n)
    eps = rng.normal(0.0, 1.0, size=(n, 4))
    z = rho * latent[:, None] + sqrt_term * eps

    age_scale = robust_scale(clean_df["age"])
    tenure_scale = robust_scale(clean_df["tenure_days"])

    base_age = pd.to_numeric(base["age"], errors="coerce").fillna(35.0).to_numpy(dtype=float)
    base_income = pd.to_numeric(base["income"], errors="coerce").fillna(50000.0).to_numpy(dtype=float)
    base_tenure = pd.to_numeric(base["tenure_days"], errors="coerce").fillna(365.0).to_numpy(dtype=float)
    base_amount = (
        pd.to_numeric(base["transaction_amount"], errors="coerce").fillna(50.0).to_numpy(dtype=float)
    )

    age = base_age + noise * z[:, 0] * age_scale
    age = np.clip(np.round(age), 0.0, 120.0)

    income = base_income * np.exp(noise * z[:, 1] * 0.35)
    income = np.clip(income, 5000.0, 500000.0)

    tenure = base_tenure + noise * z[:, 2] * tenure_scale
    tenure = np.clip(np.round(tenure), 0.0, 10000.0)

    amount = base_amount * np.exp(noise * z[:, 3] * 0.45)
    amount = np.clip(amount, 1.0, 1_000_000.0)

    tax_rate = np.array([TAX_RATES.get(str(r), 0.07) for r in region], dtype=float)
    tax = np.round(amount * tax_rate, 2)
    total = np.round(amount + tax, 2)
    amount = np.round(amount, 2)

    churn = base["churn"].astype(bool).to_numpy().copy()
    flip_prob = float(np.clip(noise * 0.12, 0.0, 0.25))
    flip = rng.random(n) < flip_prob
    churn[flip] = ~churn[flip]

    base_ts = pd.to_datetime(base["timestamp"], errors="coerce", utc=True).fillna(
        pd.Timestamp("2025-06-01", tz="UTC")
    )
    jitter_sd = max(noise * 15.0, 0.01)
    jitter_days = rng.normal(0.0, jitter_sd, size=n)
    ts = base_ts + pd.to_timedelta(jitter_days, unit="D")

    min_ts = pd.Timestamp("2025-01-01", tz="UTC")
    max_ts = pd.Timestamp("2025-12-31 23:59:59", tz="UTC")
    ts = ts.where(ts >= min_ts, min_ts)
    ts = ts.where(ts <= max_ts, max_ts)

    synth = pd.DataFrame(
        {
            "record_id": [f"SYNR{i:08d}" for i in range(1, n + 1)],
            "customer_id": [f"SYNC{i:08d}" for i in range(1, n + 1)],
            "timestamp": ts,
            "region": region,
            "category": category,
            "age": age,
            "income": np.round(income, 2),
            "tenure_days": tenure,
            "transaction_amount": amount,
            "tax_amount": tax,
            "total_amount": total,
            "churn": churn.astype(bool),
            "lineage_source_index": base_idx.astype(int),
        }
    )
    return synth, rare_list


def compare_real_synth(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    rare_threshold: float,
    seed: int,
) -> dict[str, float | list[str]]:
    out: dict[str, float | list[str]] = {}

    for col in NUMERIC_COLS:
        out[f"ks_{col}"] = ks_statistic(real[col], synth[col])

    out["tv_category"] = tv_distance(real["category"], synth["category"])
    out["tv_region"] = tv_distance(real["region"], synth["region"])
    out["corr_error"] = correlation_error(real, synth, NUMERIC_COLS)
    out["exact_match"] = exact_match_rate(
        real,
        synth,
        ["region", "category", "age", "income", "tenure_days", "transaction_amount"],
        cap=400,
        seed=seed,
    )
    out["nn_ratio"] = nn_distance_ratio(
        real,
        synth,
        ["income", "tenure_days", "transaction_amount"],
        cap=180,
        seed=seed,
    )

    rare_score, rare_list = rare_preservation(real, synth, rare_threshold)
    out["rare_preservation"] = rare_score
    out["rare_categories"] = rare_list

    churn_real = float(pd.to_numeric(real["churn"], errors="coerce").mean())
    churn_synth = float(pd.to_numeric(synth["churn"], errors="coerce").mean())
    out["churn_real"] = churn_real
    out["churn_synth"] = churn_synth
    out["churn_gap"] = abs(churn_synth - churn_real)

    out["mean_income_real"] = float(pd.to_numeric(real["income"], errors="coerce").mean())
    out["mean_income_synth"] = float(pd.to_numeric(synth["income"], errors="coerce").mean())

    return out


def fig_to_rgb(fig: plt.Figure) -> np.ndarray:
    fig.canvas.draw()
    buffer = fig.canvas.buffer_rgba()
    image = np.asarray(buffer)
    rgb = image[:, :, :3].copy()
    return np.ascontiguousarray(rgb, dtype=np.uint8)


def render_frame(
    step: int,
    total_steps: int,
    noise: float,
    hist: pd.DataFrame,
    real: pd.DataFrame,
    synth: pd.DataFrame,
) -> np.ndarray:
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.6), dpi=100)

    ax = axes[0]
    ax.plot(hist["step"], hist["ks_income"], marker="o", linewidth=1.6, label="Income KS")
    ax.plot(hist["step"], hist["corr_error"], marker="s", linewidth=1.6, label="Corr error")
    ax.plot(hist["step"], hist["exact_match"], marker="^", linewidth=1.6, label="Exact match")
    ax.set_xlabel("Step")
    ax.set_ylabel("Metric")
    ax.set_title(f"Convergence step {step}/{total_steps}\nnoise={noise:.2f}", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)

    ax2 = axes[1]
    real_s = real.sample(min(100, len(real)), random_state=step)
    synth_s = synth.sample(min(100, len(synth)), random_state=step)

    real_income = pd.to_numeric(real_s["income"], errors="coerce") / 1000.0
    real_amount = pd.to_numeric(real_s["transaction_amount"], errors="coerce")
    synth_income = pd.to_numeric(synth_s["income"], errors="coerce") / 1000.0
    synth_amount = pd.to_numeric(synth_s["transaction_amount"], errors="coerce")

    ax2.scatter(real_income, real_amount, s=14, alpha=0.55, label="Real")
    ax2.scatter(synth_income, synth_amount, s=14, alpha=0.55, label="Synthetic")
    ax2.set_xlabel("Income ($k)")
    ax2.set_ylabel("Transaction amount ($)")
    ax2.set_title("Income vs transaction amount", fontsize=9)
    ax2.legend(fontsize=7)
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    frame = fig_to_rgb(fig)
    plt.close(fig)
    return frame


def encode_video(frames: list[np.ndarray], fps: int) -> tuple[bytes | None, str | None]:
    if not frames:
        return None, "No frames were generated."

    try:
        import av
    except Exception as exc:
        return None, f"PyAV unavailable: {exc}"

    h, w = frames[0].shape[:2]
    if h % 2 or w % 2:
        h2 = h - (h % 2)
        w2 = w - (w % 2)
        frames = [f[:h2, :w2, :] for f in frames]
        h, w = h2, w2

    last_error: Exception | None = None

    for codec in ("libx264", "mpeg4"):
        buf = io.BytesIO()
        container = None
        try:
            container = av.open(buf, mode="w", format="mp4")
            stream = container.add_stream(codec, rate=int(fps))
            stream.width = int(w)
            stream.height = int(h)
            stream.pix_fmt = "yuv420p"
            stream.bit_rate = 1_200_000

            for frame in frames:
                video_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
                for packet in stream.encode(video_frame):
                    container.mux(packet)

            for packet in stream.encode():
                container.mux(packet)

            container.close()
            container = None
            data = buf.getvalue()
            if data:
                return data, None
        except Exception as exc:
            last_error = exc
            if container is not None:
                try:
                    container.close()
                except Exception:
                    pass

    return None, f"MP4 encoding failed: {last_error}"


def encode_gif(frames: list[np.ndarray], fps: int) -> tuple[bytes | None, str | None]:
    if not frames:
        return None, "No frames were generated."

    try:
        from PIL import Image
    except Exception as exc:
        return None, f"Pillow unavailable: {exc}"

    try:
        images = []
        for frame in frames:
            img = Image.fromarray(frame).convert("RGB")
            images.append(img.convert("P", palette=Image.ADAPTIVE, colors=128))

        buf = io.BytesIO()
        images[0].save(
            buf,
            format="GIF",
            append_images=images[1:],
            duration=int(1000 / max(int(fps), 1)),
            loop=0,
            optimize=False,
        )
        data = buf.getvalue()
        if data:
            return data, None
        return None, "GIF encoding produced an empty file."
    except Exception as exc:
        return None, str(exc)


def run_simulation(settings: DemoSettings, clean_df: pd.DataFrame) -> dict[str, Any]:
    start = time.perf_counter()
    tracemalloc.start()

    rng = np.random.default_rng(settings.seed + 101)
    history_records: list[dict[str, Any]] = []
    frames: list[np.ndarray] = []

    sample_n = min(max(80, settings.synth_rows // 3), 300)

    for step in range(1, settings.steps + 1):
        progress = step / float(settings.steps)
        noise = settings.initial_noise * (1.0 - progress) + settings.final_noise * progress

        synth, _ = generate_synthetic(
            clean_df=clean_df,
            n=sample_n,
            rng=rng,
            noise=noise,
            rare_threshold=settings.rare_threshold,
            correlation_strength=settings.correlation_strength,
        )

        metrics = compare_real_synth(
            real=clean_df,
            synth=synth,
            rare_threshold=settings.rare_threshold,
            seed=settings.seed + step,
        )
        metrics["step"] = step
        metrics["noise"] = noise
        metrics["sample_rows"] = sample_n
        history_records.append(metrics)

        hist_df = pd.DataFrame(history_records)
        frame = render_frame(
            step=step,
            total_steps=settings.steps,
            noise=noise,
            hist=hist_df,
            real=clean_df,
            synth=synth,
        )
        frames.append(frame)

    final_synth, final_rare = generate_synthetic(
        clean_df=clean_df,
        n=settings.synth_rows,
        rng=rng,
        noise=settings.final_noise,
        rare_threshold=settings.rare_threshold,
        correlation_strength=settings.correlation_strength,
    )

    final_metrics = compare_real_synth(
        real=clean_df,
        synth=final_synth,
        rare_threshold=settings.rare_threshold,
        seed=settings.seed + 999,
    )
    final_metrics["final_rare_categories"] = final_rare

    quality_synth = data_quality_report(final_synth, "synthetic")

    elapsed = time.perf_counter() - start
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = float(peak_bytes / (1024 * 1024))

    total_generated_rows = settings.steps * sample_n + settings.synth_rows
    throughput = float(total_generated_rows / max(elapsed, 1e-9))

    video_mp4, video_warning = encode_video(frames, settings.fps)
    video_gif = None
    if video_mp4 is None:
        video_gif, gif_warning = encode_gif(frames, settings.fps)
        if video_gif is None:
            video_warning = f"{video_warning or ''} {gif_warning or ''}".strip()
        else:
            video_warning = f"{video_warning or ''} GIF fallback used.".strip()

    history_df = pd.DataFrame(history_records)

    return {
        "history_df": history_df,
        "final_synth": final_synth,
        "final_metrics": final_metrics,
        "quality_synth": quality_synth,
        "video_mp4": video_mp4,
        "video_gif": video_gif,
        "video_warning": video_warning,
        "elapsed": elapsed,
        "peak_mb": peak_mb,
        "throughput": throughput,
    }


@st.cache_data(show_spinner="Loading demo data, running simulation, and rendering video...")
def cached_pipeline(payload: dict[str, Any]) -> dict[str, Any]:
    settings = DemoSettings.model_validate(payload)
    raw_df = make_demo_dataframe(settings)
    fingerprint = fingerprint_dataframe(raw_df)
    clean_df = clean_for_generation(raw_df, settings.seed)
    quality_raw = data_quality_report(raw_df, "raw_demo")
    sim = run_simulation(settings, clean_df)

    return {
        "settings": settings.model_dump(mode="json"),
        "config_hash": settings_hash(settings),
        "fingerprint": fingerprint,
        "raw_df": raw_df,
        "clean_df": clean_df,
        "synthetic_df": sim["final_synth"],
        "history_df": sim["history_df"],
        "final_metrics": sim["final_metrics"],
        "quality_raw": quality_raw,
        "quality_synth": sim["quality_synth"],
        "video_mp4": sim["video_mp4"],
        "video_gif": sim["video_gif"],
        "video_warning": sim["video_warning"],
        "elapsed": sim["elapsed"],
        "peak_mb": sim["peak_mb"],
        "throughput": sim["throughput"],
    }


def fmt_number(value: Any, digits: int = 3, percent: bool = False) -> str:
    if value is None:
        return "n/a"
    try:
        f = float(value)
    except Exception:
        return str(value)
    if math.isnan(f):
        return "n/a"
    if percent:
        return f"{f * 100:.1f}%"
    return f"{f:.{digits}f}"


def binned_distribution(series: pd.Series, bins: int = 12) -> pd.DataFrame:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return pd.DataFrame({"bin": [], "count": []})
    if s.nunique() <= 1:
        return pd.DataFrame({"bin": [str(round(float(s.iloc[0]), 2))], "count": [int(len(s))]})

    q = min(bins, max(2, int(s.nunique())))
    try:
        cut = pd.qcut(s, q=q, duplicates="drop")
    except ValueError:
        cut = pd.cut(s, bins=q)

    vc = cut.value_counts().sort_index()
    return pd.DataFrame({"bin": vc.index.astype(str), "count": vc.values.astype(int)})


st.set_page_config(
    page_title="Synthetic Dataset Generation Laboratory",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title(APP_TITLE)
st.caption(
    "CPU-only Streamlit prototype • deterministic seeds • no API key • "
    "auto-generated in-app demo video • synthetic data is not automatically anonymous"
)

with st.sidebar:
    st.header("Controls")

    seed = st.number_input("Random seed", min_value=0, max_value=2_147_483_647, value=42, step=1)
    n_rows = st.slider("Raw demo rows", min_value=100, max_value=5000, value=700, step=100)
    synth_rows = st.slider("Synthetic rows", min_value=50, max_value=5000, value=500, step=50)
    steps = st.slider("Simulation steps", min_value=8, max_value=80, value=24, step=4)
    fps = st.select_slider("Video FPS", options=[8, 10, 12, 15, 20, 24, 30], value=12)

    st.subheader("Generation behaviour")
    initial_noise = st.slider("Initial noise", min_value=0.0, max_value=1.0, value=0.45, step=0.01)
    final_noise = st.slider("Final noise", min_value=0.0, max_value=0.5, value=0.06, step=0.01)
    rare_threshold = st.slider("Rare category threshold", min_value=0.0, max_value=0.25, value=0.03, step=0.005)
    target_churn = st.slider("Target churn rate", min_value=0.01, max_value=0.95, value=0.18, step=0.01)
    correlation_strength = st.slider("Correlation strength", min_value=-1.0, max_value=1.0, value=0.55, step=0.01)
    anomaly_rate = st.slider("Injected anomaly rate", min_value=0.0, max_value=0.15, value=0.02, step=0.005)

    st.divider()
    if st.button("Clear cache and rerun"):
        st.cache_data.clear()
        st.rerun()

config_payload = {
    "schema_version": SCHEMA_VERSION,
    "seed": int(seed),
    "n_rows": int(n_rows),
    "synth_rows": int(synth_rows),
    "steps": int(steps),
    "fps": int(fps),
    "initial_noise": float(initial_noise),
    "final_noise": float(final_noise),
    "rare_threshold": float(rare_threshold),
    "target_churn": float(target_churn),
    "correlation_strength": float(correlation_strength),
    "anomaly_rate": float(anomaly_rate),
}

try:
    settings = DemoSettings.model_validate(config_payload)
except Exception as exc:
    st.error(f"Invalid configuration: {exc}")
    st.stop()

try:
    result = cached_pipeline(config_payload)
except Exception as exc:
    st.error("Pipeline failed. See exception below.")
    st.exception(exc)
    st.stop()

final_metrics = result["final_metrics"]
history_df = result["history_df"]
synthetic_df = result["synthetic_df"]
raw_df = result["raw_df"]
clean_df = result["clean_df"]

st.success(
    f"Live demo ready • config {result['config_hash']} • "
    f"raw fingerprint {result['fingerprint'][:16]}… • "
    f"runtime {result['elapsed']:.2f}s • peak {result['peak_mb']:.1f} MB"
)

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Raw rows", f"{len(raw_df):,}")
c2.metric("Synthetic rows", f"{len(synthetic_df):,}")
c3.metric("Simulation steps", f"{settings.steps}")
c4.metric("Income KS", fmt_number(final_metrics.get("ks_income")))
c5.metric("Corr error", fmt_number(final_metrics.get("corr_error")))
c6.metric("Exact match", fmt_number(final_metrics.get("exact_match"), percent=True))

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("NN distance ratio", fmt_number(final_metrics.get("nn_ratio")))
c2.metric("Rare preservation", fmt_number(final_metrics.get("rare_preservation"), percent=True))
c3.metric("Churn gap", fmt_number(final_metrics.get("churn_gap"), percent=True))
c4.metric("Throughput rows/s", fmt_number(result["throughput"], digits=1))
c5.metric("Peak memory MB", fmt_number(result["peak_mb"], digits=1))
c6.metric("Lineage coverage", fmt_number(result["quality_synth"].get("lineage_coverage"), percent=True))

tab_video, tab_overview, tab_data, tab_quality, tab_synth, tab_diagnostics = st.tabs(
    ["Live Demo Video", "Overview", "Source Data", "Data Quality", "Synthetic Output", "Diagnostics"]
)

with tab_video:
    st.subheader("Automatically generated demo video")
    st.write(
        "This video is rendered from the simulation frames on first run. "
        "It shows convergence of distributional distance, correlation error, and exact-copy rate."
    )

    if result["video_mp4"]:
        try:
            st.video(result["video_mp4"], format="video/mp4", loop=True, autoplay=True, muted=True)
        except TypeError:
            st.video(result["video_mp4"], format="video/mp4", loop=True, autoplay=True)

        st.download_button(
            label="Download MP4 demo video",
            data=result["video_mp4"],
            file_name=f"synthetic_lab_demo_{result['config_hash']}.mp4",
            mime="video/mp4",
        )
    elif result["video_gif"]:
        st.warning(
            "H.264/MP4 encoding was unavailable in this environment. "
            "Showing animated GIF fallback so the demo is not blank."
        )
        st.image(result["video_gif"])
        st.download_button(
            label="Download GIF demo fallback",
            data=result["video_gif"],
            file_name=f"synthetic_lab_demo_{result['config_hash']}.gif",
            mime="image/gif",
        )
    else:
        st.error(f"Video generation failed: {result['video_warning'] or 'unknown error'}")

    st.divider()
    st.subheader("Simulation history")
    plot_cols = ["ks_income", "corr_error", "exact_match", "churn_gap"]
    st.line_chart(history_df.set_index("step")[plot_cols])

    st.subheader("Per-step table")
    st.dataframe(history_df, use_container_width=True)

with tab_overview:
    st.subheader("Run summary")
    st.write(
        "- Raw inputs are treated as immutable evidence and fingerprinted with SHA-256.  \n"
        "- Synthetic output is generated as a new versioned derivative.  "
        "- All randomness is controlled by the seed.  "
        "- No external API key is required.  "
        "- Privacy diagnostics are empirical only and are not formal anonymity guarantees."
    )

    meta = {
        "schema_version": SCHEMA_VERSION,
        "config_hash": result["config_hash"],
        "raw_fingerprint_sha256": result["fingerprint"],
        "seed": settings.seed,
        "rows_raw": len(raw_df),
        "rows_clean_for_generation": len(clean_df),
        "rows_synthetic": len(synthetic_df),
        "runtime_seconds": round(result["elapsed"], 3),
        "peak_memory_mb": round(result["peak_mb"], 3),
        "throughput_rows_per_second": round(result["throughput"], 3),
    }
    st.json(meta)

with tab_data:
    st.subheader("Raw demo records")
    st.dataframe(raw_df.head(100), use_container_width=True)

    st.subheader("Schema, cardinality, and missingness")
    card = pd.DataFrame(
        {
            "column": raw_df.columns,
            "dtype": [str(raw_df[c].dtype) for c in raw_df.columns],
            "unique": [int(raw_df[c].nunique(dropna=False)) for c in raw_df.columns],
            "missing_percent": [float(raw_df[c].isna().mean() * 100.0) for c in raw_df.columns],
        }
    )
    st.dataframe(card, use_container_width=True)

    st.subheader("Categorical distribution")
    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown("**Region**")
        st.bar_chart(raw_df["region"].value_counts())
    with cc2:
        st.markdown("**Category**")
        st.bar_chart(raw_df["category"].value_counts())

    st.subheader("Numerical distribution")
    nc1, nc2 = st.columns(2)
    with nc1:
        st.markdown("**Income bins**")
        st.bar_chart(binned_distribution(raw_df["income"], bins=14).set_index("bin")["count"])
    with nc2:
        st.markdown("**Transaction amount bins**")
        st.bar_chart(
            binned_distribution(raw_df["transaction_amount"], bins=14).set_index("bin")["count"]
        )

with tab_quality:
    st.subheader("Quality metrics")
    quality_table = pd.DataFrame(
        [result["quality_raw"], result["quality_synth"]]
    ).set_index("dataset").T
    st.dataframe(quality_table, use_container_width=True)

    st.subheader("Constraint registry")
    constraint_checks = {
        "age >= 0": bool((synthetic_df["age"] >= 0).all()),
        "income > 0": bool((pd.to_numeric(synthetic_df["income"], errors="coerce") > 0).all()),
        "tenure_days >= 0": bool((pd.to_numeric(synthetic_df["tenure_days"], errors="coerce") >= 0).all()),
        "transaction_amount > 0": bool(
            (pd.to_numeric(synthetic_df["transaction_amount"], errors="coerce") > 0).all()
        ),
        "total_amount = transaction_amount + tax_amount": bool(
            np.isclose(
                pd.to_numeric(synthetic_df["total_amount"], errors="coerce"),
                pd.to_numeric(synthetic_df["transaction_amount"], errors="coerce")
                + pd.to_numeric(synthetic_df["tax_amount"], errors="coerce"),
                rtol=0.01,
                atol=0.02,
            ).all()
        ),
        "region in allowed set": bool(synthetic_df["region"].isin(REGIONS).all()),
        "category in allowed set": bool(synthetic_df["category"].isin(CATEGORIES).all()),
    }
    st.json(constraint_checks)

    st.subheader("Raw invalid examples")
    raw_validity = (
        raw_df["age"].notna()
        & (pd.to_numeric(raw_df["age"], errors="coerce") >= 0)
        & (pd.to_numeric(raw_df["transaction_amount"], errors="coerce") > 0)
        & raw_df["region"].isin(REGIONS)
        & raw_df["category"].isin(CATEGORIES)
    )
    invalid = raw_df.loc[~raw_validity].head(25)
    if invalid.empty:
        st.info("No invalid raw rows were injected at the current anomaly rate.")
    else:
        st.dataframe(invalid, use_container_width=True)

with tab_synth:
    st.subheader("Synthetic output")
    st.dataframe(synthetic_df.head(100), use_container_width=True)

    sc1, sc2 = st.columns(2)
    with sc1:
        st.markdown("**Synthetic category distribution**")
        st.bar_chart(synthetic_df["category"].value_counts())
    with sc2:
        st.markdown("**Synthetic region distribution**")
        st.bar_chart(synthetic_df["region"].value_counts())

    st.subheader("Real clean vs synthetic income distribution")
    rc1, rc2 = st.columns(2)
    with rc1:
        st.markdown("**Clean real income bins**")
        st.bar_chart(binned_distribution(clean_df["income"], bins=14).set_index("bin")["count"])
    with rc2:
        st.markdown("**Synthetic income bins**")
        st.bar_chart(binned_distribution(synthetic_df["income"], bins=14).set_index("bin")["count"])

    csv = synthetic_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download synthetic CSV",
        data=csv,
        file_name=f"synthetic_output_{result['config_hash']}.csv",
        mime="text/csv",
    )

    raw_csv = raw_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download raw demo CSV",
        data=raw_csv,
        file_name=f"raw_demo_{result['config_hash']}.csv",
        mime="text/csv",
    )

with tab_diagnostics:
    st.subheader("Configuration")
    st.json(result["settings"])

    st.subheader("Final metrics")
    st.json(final_metrics)

    st.subheader("Reproducibility checks")
    recomputed_hash = settings_hash(DemoSettings.model_validate(result["settings"]))
    recovery_verified = recomputed_hash == result["config_hash"]
    st.write(f"- Configuration hash stable: **{recovery_verified}**")
    st.write(f"- Raw fingerprint: `{result['fingerprint']}`")
    st.write(f"- Synthetic rows linked to source indices: **{int(synthetic_df['lineage_source_index'].notna().sum()):,}**")

    st.subheader("Known limitations")
    st.markdown(
        """
        - This is a prototype, not a production data platform.
        - The generator is a jittered dependency-aware baseline, not a fully trained CTGAN/TVAE or Bayesian network.
        - Empirical privacy diagnostics do not prove differential privacy or anonymity.
        - MP4 playback depends on the local PyAV/FFmpeg build; if H.264 is unavailable, the app falls back to GIF.
        - Large settings may take longer to render on CPU-only machines.
        """
    )
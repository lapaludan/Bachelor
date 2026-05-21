from __future__ import annotations

import os
import time
import warnings
import traceback
from multiprocessing import Pool

import numpy as np
import pandas as pd
import cvxpy as cp

warnings.filterwarnings("ignore")

try:
    from sklearn.covariance import LedoitWolf as _LedoitWolf
except ImportError as e:
    raise ImportError(
        "scikit-learn is required for the linear Ledoit-Wolf model. "
        "Install with: pip install scikit-learn"
    ) from e

try:
    import nonlinshrink as _nls_pkg
    if not callable(getattr(_nls_pkg, "shrink_cov", None)):
        raise ImportError("nonlinshrink.shrink_cov is not callable")
except ImportError as e:
    raise ImportError(
        "nonlinshrink is required for the NLS model. "
        "Install with: pip install nonlinshrink"
    ) from e


REGIONS = ["emea"]

BASE_DIR = (
    r"C:\Users\lapal\OneDrive - CBS - Copenhagen Business School"
    r"\6. Semester\Bachelor\Kode V2"
)

CONFIGS = [
    {
        "tag": "1995_cov756_s1_lo",
        "run_start": 1995,
        "cov_window": 756,
        "scenarios": ["s1_lo"],
    },
    {
        "tag": "1995_cov756_s1_L160",
        "run_start": 1995,
        "cov_window": 756,
        "scenarios": ["s1_L160"],
    },
    {
        "tag": "1995_cov756_s1_L3",
        "run_start": 1995,
        "cov_window": 756,
        "scenarios": ["s1_L3"],
    },
    {
        "tag": "1995_cov756_s1_uncapped",
        "run_start": 1995,
        "cov_window": 756,
        "scenarios": ["s1_uncapped"],
    },
]

TUNING_START = 1990
RUN_END = 2025

TOP_N_UNIVERSE = 500
MIN_ASSETS        = 400
MIN_ASSETS_TUNING = 400

OBS_FRAC = 0.80
NEAR_ZERO_VOL = 1e-10
DUP_CORR_THRESH = 0.99

LINEAR_SHRINKAGE_LABEL = "Ledoit-Wolf auto"

EPO_DEFAULT_W = 0.50
EPO_MIN_HISTORY = 24

_g1 = [round(x, 2) for x in np.arange(0.00, 1.01, 0.05)]
_g2 = [
    0.00, 0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
    0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90, 0.95, 0.99, 1.00,
]
EPO_GRID = sorted(set(_g1 + _g2))

W_GRID_FIXED = EPO_GRID.copy()


SCENARIO_SPECS = {
    "s1_lo": {
        "kind": "lo",
        "sum_target": 1.0,
        "leverage_cap": None,
        "anchor_kind": "ew",
    },
    "s1_L160": {
        "kind": "ls",
        "sum_target": 1.0,
        "leverage_cap": 1.60,
        "anchor_kind": "ew",
    },
    "s1_L3": {
        "kind": "ls",
        "sum_target": 1.0,
        "leverage_cap": 3.0,
        "anchor_kind": "ew",
    },
    "s1_uncapped": {
        "kind": "ls",
        "sum_target": 1.0,
        "leverage_cap": None,
        "anchor_kind": "ew",
    },
}


def _w_to_tag(w: float) -> str:
    return f"{int(round(w * 100)):03d}"


def _base_model_name(cov_kind: str, scenario_key: str) -> str:
    prefix = {
        "sample": "mv_sample",
        "lin": "mv_lin",
        "nls": "mv_nls",
    }[cov_kind]
    return f"{prefix}_{scenario_key}"


def _epo_model_name(base_kind: str, scenario_key: str) -> str:
    return f"epo_{base_kind}_{scenario_key}"


def _epo_fixed_model_name(base_kind: str, scenario_key: str, w: float) -> str:
    return f"{_epo_model_name(base_kind, scenario_key)}_w{_w_to_tag(w)}"


BENCHMARK_MODEL_NAMES = ["ew", "ivol"]

BASE_MODEL_SPECS = []
for cov_kind in ["sample", "lin", "nls"]:
    for scenario_key in SCENARIO_SPECS:
        BASE_MODEL_SPECS.append({
            "name": _base_model_name(cov_kind, scenario_key),
            "cov_kind": cov_kind,
            "scenario_key": scenario_key,
        })

TUNED_EPO_SPECS = []
for base_kind in ["sample", "lin", "nls"]:
    for scenario_key in SCENARIO_SPECS:
        TUNED_EPO_SPECS.append({
            "name": _epo_model_name(base_kind, scenario_key),
            "base_kind": base_kind,
            "scenario_key": scenario_key,
        })

for base_kind in ["cc", "cc_lin", "cc_nls"]:
    for scenario_key in SCENARIO_SPECS:
        TUNED_EPO_SPECS.append({
            "name": _epo_model_name(base_kind, scenario_key),
            "base_kind": base_kind,
            "scenario_key": scenario_key,
        })


def load_inputs(region: str, base_dir: str) -> dict:
    required = {
        "daily_ret": f"daily_returns_{region}_tot_ret_usd.parquet",
        "universe": f"monthly_universe_{region}_top600_adv.parquet",
        "month_ends": f"month_ends_{region}.parquet",
        "signals": f"signals_{region}_mom_2_12_top600_1980_2025.parquet",
        "ff3": "FF3.csv",
    }

    paths = {k: os.path.join(base_dir, v) for k, v in required.items()}

    for k, p in paths.items():
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required input file missing [{k}]:\n  {p}")

    daily_ret = pd.read_parquet(paths["daily_ret"])
    daily_ret.index = pd.to_datetime(daily_ret.index)
    daily_ret.columns = daily_ret.columns.astype(str)
    daily_ret = daily_ret.sort_index()

    universe = pd.read_parquet(paths["universe"])
    universe["month_end"] = pd.to_datetime(universe["month_end"])
    universe["sec_id"] = universe["sec_id"].astype(str)

    me_df = pd.read_parquet(paths["month_ends"])
    month_ends = pd.DatetimeIndex(pd.to_datetime(me_df["month_end"].values))

    signals = pd.read_parquet(paths["signals"])
    signals.index = pd.to_datetime(signals.index)
    signals.columns = signals.columns.astype(str)
    signals = signals.sort_index()

    rf = _load_rf(paths["ff3"])

    print(
        f"  [Inputs] returns={daily_ret.shape} | "
        f"universe rows={len(universe):,} | "
        f"signals={signals.shape} | rf obs={len(rf)}"
    )

    if len(rf) == 0:
        print("  WARNING: RF series is empty — check FF3.csv format.")

    return {
        "daily_ret": daily_ret,
        "universe": universe,
        "month_ends": month_ends,
        "signals": signals,
        "rf": rf,
    }


def _load_rf(path: str) -> pd.Series:
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]

    date_col = df.columns[0]
    df = df.rename(columns={date_col: "date"})

    date_raw = df["date"].astype(str).str.strip()
    parsed = None

    for fmt in ("%d/%m/%Y", "%Y%m%d", "%Y-%m-%d", "%m/%d/%Y"):
        try:
            parsed = pd.to_datetime(date_raw, format=fmt, errors="raise")
            break
        except (ValueError, TypeError):
            continue

    if parsed is None:
        parsed = pd.to_datetime(date_raw, errors="coerce")

    df["date"] = parsed
    df = df.dropna(subset=["date"])

    if "rf" not in df.columns:
        raise ValueError(
            f"FF3.csv missing 'rf' column after lowercasing. "
            f"Found: {list(df.columns)}"
        )

    df["rf"] = pd.to_numeric(df["rf"], errors="coerce")
    df = df.dropna(subset=["rf"])
    df["date"] = df["date"].dt.normalize()

    df = (
        df.sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .set_index("date")
    )

    return df["rf"].sort_index()


def filter_rebalance_universe(
    adv_top600: pd.DataFrame,
    ret_window_all: pd.DataFrame,
    cov_window: int,
) -> tuple[list[str], pd.DataFrame, dict]:
    min_obs = int(np.floor(OBS_FRAC * cov_window))
    ordered = adv_top600.sort_values("adv_rank")["sec_id"].tolist()

    counts = {"start": len(ordered)}

    avail = set(ret_window_all.columns)
    ordered = [s for s in ordered if s in avail]
    counts["after_avail"] = len(ordered)

    if not ordered:
        return [], pd.DataFrame(), counts

    rw = ret_window_all[ordered]
    obs_counts = rw.notna().sum(axis=0)
    pass_obs = set(obs_counts.index[obs_counts >= min_obs])
    ordered = [s for s in ordered if s in pass_obs]
    counts["after_obs"] = len(ordered)

    if not ordered:
        return [], pd.DataFrame(), counts

    rw = ret_window_all[ordered]
    stds = rw.std(axis=0, skipna=True)
    pass_vol = set(stds.index[stds > NEAR_ZERO_VOL])
    ordered = [s for s in ordered if s in pass_vol]
    counts["after_vol"] = len(ordered)

    if not ordered:
        return [], pd.DataFrame(), counts

    rw = ret_window_all[ordered]
    corr = rw.fillna(0.0).corr().values

    n_curr = len(ordered)
    remove = set()

    for i in range(n_curr):
        if ordered[i] in remove:
            continue

        for j in range(i + 1, n_curr):
            if ordered[j] in remove:
                continue

            if abs(corr[i, j]) >= DUP_CORR_THRESH:
                remove.add(ordered[j])

    ordered = [s for s in ordered if s not in remove]
    counts["after_dup"] = len(ordered)

    ordered = ordered[:TOP_N_UNIVERSE]
    counts["final"] = len(ordered)

    if not ordered:
        return [], pd.DataFrame(), counts

    return ordered, ret_window_all[ordered], counts


def _fill_nan_col_mean(X: np.ndarray) -> np.ndarray:
    X = X.copy()

    for j in range(X.shape[1]):
        col = X[:, j]
        mask = np.isnan(col)

        if mask.any():
            mean = np.nanmean(col)
            X[mask, j] = mean if not np.isnan(mean) else 0.0

    return X


def estimate_sample_cov(X: np.ndarray) -> np.ndarray:
    Xf = _fill_nan_col_mean(X)
    cov = np.cov(Xf.T)

    if cov.ndim == 0:
        cov = np.array([[float(cov)]])

    return cov + np.eye(cov.shape[0]) * 1e-10


def estimate_lin_cov(X: np.ndarray) -> tuple[np.ndarray, float]:
    Xf = _fill_nan_col_mean(X)

    if Xf.shape[0] < 2 or Xf.shape[1] < 1:
        raise ValueError(
            f"estimate_lin_cov: insufficient data, shape={Xf.shape}"
        )

    stds = np.std(Xf, axis=0, ddof=1)
    stds = np.maximum(stds, 1e-10)
    Z = Xf / stds[np.newaxis, :]

    lw = _LedoitWolf(
        assume_centered=False,
        block_size=1000,
        store_precision=False,
    )
    lw.fit(Z)

    shrinkage = float(lw.shrinkage_)

    if not (0.0 <= shrinkage <= 1.0):
        raise RuntimeError(
            f"estimate_lin_cov: LW shrinkage out of bounds: {shrinkage}"
        )

    corr_shrunk = np.asarray(lw.covariance_, dtype=float)
    corr_shrunk = 0.5 * (corr_shrunk + corr_shrunk.T)
    d_c = np.sqrt(np.maximum(np.diag(corr_shrunk), 1e-12))
    corr_shrunk = corr_shrunk / np.outer(d_c, d_c)

    cov = corr_shrunk * np.outer(stds, stds)
    cov = cov + np.eye(cov.shape[0]) * 1e-10

    return cov, shrinkage


def estimate_nls_cov(X: np.ndarray) -> np.ndarray:
    Xf = _fill_nan_col_mean(X)

    if Xf.shape[0] < 2 or Xf.shape[1] < 1:
        raise ValueError(
            f"estimate_nls_cov: insufficient data, shape={Xf.shape}"
        )

    stds = np.std(Xf, axis=0, ddof=1)
    stds = np.maximum(stds, 1e-10)
    Z = Xf / stds[np.newaxis, :]

    corr_shrunk = _nls_pkg.shrink_cov(Z)
    corr_shrunk = np.asarray(corr_shrunk, dtype=float)
    corr_shrunk = 0.5 * (corr_shrunk + corr_shrunk.T)
    d_c = np.sqrt(np.maximum(np.diag(corr_shrunk), 1e-12))
    corr_shrunk = corr_shrunk / np.outer(d_c, d_c)

    cov = corr_shrunk * np.outer(stds, stds)

    return cov + np.eye(cov.shape[0]) * 1e-10


def _epo_cov(base_cov: np.ndarray, w: float) -> np.ndarray:
    d = np.sqrt(np.maximum(np.diag(base_cov), 1e-12))
    corr = base_cov / np.outer(d, d)

    corr_epo = (1.0 - w) * corr
    np.fill_diagonal(corr_epo, 1.0)

    sigma_epo = corr_epo * np.outer(d, d)

    return sigma_epo + np.eye(sigma_epo.shape[0]) * 1e-10


def _epo_alpha(
    base_cov: np.ndarray,
    alpha_signal: np.ndarray,
    anchor: np.ndarray,
    w: float,
    gamma: float,
) -> np.ndarray:
    d_vec = np.diag(base_cov)
    return (1.0 - w) * alpha_signal + w * gamma * d_vec * anchor


def _gamma_endogenous(
    base_cov: np.ndarray,
    sigma_model: np.ndarray,
    alpha: np.ndarray,
    anchor: np.ndarray,
) -> float:
    try:
        x = np.linalg.solve(sigma_model, alpha)
    except np.linalg.LinAlgError:
        x = np.linalg.lstsq(sigma_model, alpha, rcond=None)[0]

    num = float(x @ base_cov @ x)
    den = float(anchor @ base_cov @ anchor)

    if den < 1e-12 or num < 0.0:
        return 1.0

    return float(np.sqrt(num / den))


def transform_signal(raw_signal: pd.Series, assets: list[str]) -> np.ndarray:
    mu = raw_signal.reindex(assets).fillna(0.0).values.astype(float)
    mu = mu - mu.mean()

    pos_sum = float(np.maximum(mu, 0.0).sum())

    return mu / pos_sum if pos_sum > 0.0 else np.zeros(len(assets))


def _lookup_signal(signals: pd.DataFrame, rebal_date: pd.Timestamp) -> pd.Series:
    candidates = signals.index[signals.index <= rebal_date]

    if len(candidates) == 0:
        return pd.Series(dtype=float)

    last = candidates[-1]

    if last.to_period("M") == rebal_date.to_period("M"):
        return signals.loc[last]

    return pd.Series(dtype=float)


def _ew(n: int) -> np.ndarray:
    return np.ones(n) / n


def _zero_anchor(n: int) -> np.ndarray:
    return np.zeros(n, dtype=float)


def _get_anchor(n: int, anchor_kind: str) -> np.ndarray:
    if anchor_kind == "ew":
        return _ew(n)

    if anchor_kind == "zero":
        return _zero_anchor(n)

    raise ValueError(f"Unknown anchor_kind: {anchor_kind}")


def _ivol(cov: np.ndarray) -> np.ndarray:
    stds = np.sqrt(np.maximum(np.diag(cov), 0.0))
    stds = np.maximum(stds, 1e-12)

    w = 1.0 / stds

    return w / w.sum()


def _mv_lo(mu: np.ndarray, cov: np.ndarray) -> np.ndarray | None:
    n = len(mu)
    w = cp.Variable(n)

    prob = cp.Problem(
        cp.Minimize(0.5 * cp.quad_form(w, cp.psd_wrap(cov)) - mu @ w),
        [
            w >= 0,
            cp.sum(w) == 1,
        ],
    )

    prob.solve(solver=cp.CLARABEL, verbose=False)

    if prob.status in ("optimal", "optimal_inaccurate") and w.value is not None:
        return np.asarray(w.value, dtype=float)

    return None


def _mv_ls_eq_closed_form(
    mu: np.ndarray,
    cov: np.ndarray,
    sum_target: float,
) -> np.ndarray | None:
    ones = np.ones(len(mu), dtype=float)

    try:
        a = np.linalg.solve(cov, mu)
        b = np.linalg.solve(cov, ones)
    except np.linalg.LinAlgError:
        a = np.linalg.lstsq(cov, mu, rcond=None)[0]
        b = np.linalg.lstsq(cov, ones, rcond=None)[0]

    denom = float(ones @ b)

    if abs(denom) < 1e-12:
        return None

    lam = float((ones @ a - sum_target) / denom)
    w = a - lam * b

    return np.asarray(w, dtype=float)


def _mv_ls(
    mu: np.ndarray,
    cov: np.ndarray,
    sum_target: float,
    leverage_cap: float | None,
) -> np.ndarray | None:
    if leverage_cap is None:
        return _mv_ls_eq_closed_form(mu, cov, sum_target)

    n = len(mu)
    w = cp.Variable(n)

    constraints = [
        cp.sum(w) == sum_target,
        cp.norm1(w) <= leverage_cap,
    ]

    prob = cp.Problem(
        cp.Minimize(0.5 * cp.quad_form(w, cp.psd_wrap(cov)) - mu @ w),
        constraints,
    )

    prob.solve(solver=cp.CLARABEL, verbose=False)

    if prob.status in ("optimal", "optimal_inaccurate") and w.value is not None:
        return np.asarray(w.value, dtype=float)

    return None


def _fallback_weights(n: int, scenario_key: str) -> np.ndarray:
    spec = SCENARIO_SPECS[scenario_key]
    return _get_anchor(n, spec["anchor_kind"]).copy()


def compute_base_weights(
    n: int,
    mu: np.ndarray,
    cov_sample: np.ndarray,
    cov_lin: np.ndarray,
    cov_nls: np.ndarray,
    active_base_specs: list[dict],
) -> dict[str, np.ndarray]:
    weights: dict[str, np.ndarray] = {
        "ew": _ew(n),
        "ivol": _ivol(cov_nls),
    }

    for spec in active_base_specs:
        name = spec["name"]
        scenario_key = spec["scenario_key"]
        scen = SCENARIO_SPECS[scenario_key]

        if spec["cov_kind"] == "sample":
            cov = cov_sample
        elif spec["cov_kind"] == "lin":
            cov = cov_lin
        else:
            cov = cov_nls

        anchor = _get_anchor(n, scen["anchor_kind"])
        gamma = _gamma_endogenous(cov, cov, mu, anchor)
        mu_adj = mu / gamma if gamma > 1e-10 else mu

        if scen["kind"] == "lo":
            w = _mv_lo(mu_adj, cov)
        else:
            w = _mv_ls(
                mu_adj,
                cov,
                sum_target=scen["sum_target"],
                leverage_cap=scen["leverage_cap"],
            )

        weights[name] = w if w is not None else _fallback_weights(n, scenario_key)

    return weights


def compute_epo_grid_weights_for_variant(
    n: int,
    mu: np.ndarray,
    base_cov: np.ndarray,
    scenario_key: str,
) -> dict[float, np.ndarray]:
    scen = SCENARIO_SPECS[scenario_key]
    anchor = _get_anchor(n, scen["anchor_kind"])
    fallback = _fallback_weights(n, scenario_key)

    out: dict[float, np.ndarray] = {}

    for w_val in EPO_GRID:
        cov_epo = _epo_cov(base_cov, w_val)
        gamma_e = _gamma_endogenous(base_cov, cov_epo, mu, anchor)
        alpha_e = _epo_alpha(base_cov, mu, anchor, w_val, gamma_e)
        mu_e = alpha_e / gamma_e if gamma_e > 1e-10 else alpha_e

        if scen["kind"] == "lo":
            w = _mv_lo(mu_e, cov_epo)
        else:
            w = _mv_ls(
                mu_e,
                cov_epo,
                sum_target=scen["sum_target"],
                leverage_cap=scen["leverage_cap"],
            )

        out[w_val] = w if w is not None else fallback.copy()

    return out


def _epo_cov_const_corr(
    base_cov: np.ndarray,
    rho_bar: float,
    w: float,
) -> np.ndarray:
    d = np.sqrt(np.maximum(np.diag(base_cov), 1e-12))
    corr = base_cov / np.outer(d, d)

    corr_cc = (1.0 - w) * corr + w * rho_bar
    np.fill_diagonal(corr_cc, 1.0)

    sigma_cc = corr_cc * np.outer(d, d)
    return sigma_cc + np.eye(sigma_cc.shape[0]) * 1e-10


def compute_epo_cc_grid_weights(
    n: int,
    mu: np.ndarray,
    base_cov: np.ndarray,
    rho_bar: float,
    scenario_key: str,
) -> dict[float, np.ndarray]:
    scen = SCENARIO_SPECS[scenario_key]
    anchor = _get_anchor(n, scen["anchor_kind"])
    fallback = _fallback_weights(n, scenario_key)

    out: dict[float, np.ndarray] = {}

    for w_val in EPO_GRID:
        cov_epo = _epo_cov_const_corr(base_cov, rho_bar, w_val)
        gamma_e = _gamma_endogenous(base_cov, cov_epo, mu, anchor)
        alpha_e = _epo_alpha(base_cov, mu, anchor, w_val, gamma_e)
        mu_e = alpha_e / gamma_e if gamma_e > 1e-10 else alpha_e

        if scen["kind"] == "lo":
            w = _mv_lo(mu_e, cov_epo)
        else:
            w = _mv_ls(
                mu_e,
                cov_epo,
                sum_target=scen["sum_target"],
                leverage_cap=scen["leverage_cap"],
            )

        out[w_val] = w if w is not None else fallback.copy()

    return out


def simulate_drift(
    w0: np.ndarray,
    assets: list[str],
    daily_ret: pd.DataFrame,
    rf_daily: pd.Series,
    impl_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.Series:
    mask = (daily_ret.index >= impl_date) & (daily_ret.index <= end_date)
    hold_ret = daily_ret.loc[mask, assets].fillna(0.0)

    if hold_ret.empty:
        return pd.Series(dtype=float)

    hold_rf = rf_daily.reindex(hold_ret.index, method="ffill").fillna(0.0)

    w = w0.astype(float).copy()
    cash = float(1.0 - w.sum())

    dates: list[pd.Timestamp] = []
    rets: list[float] = []

    for date, row in hold_ret.iterrows():
        r = row.values.astype(float)
        rf = float(hold_rf.loc[date])

        pr = float(w @ r + cash * rf)

        dates.append(date)
        rets.append(pr)

        wealth_growth = 1.0 + pr

        if wealth_growth <= 1e-12:
            break

        risky_val = w * (1.0 + r)
        cash_val = cash * (1.0 + rf)

        w = risky_val / wealth_growth
        cash = float(cash_val / wealth_growth)

    return pd.Series(rets, index=pd.DatetimeIndex(dates))


def _monthly_ret(s: pd.Series) -> float:
    if s.empty:
        return float("nan")

    return float((1.0 + s).prod() - 1.0)


def select_epo_w(
    hist: dict[float, list[float]],
    grid: list[float],
    default: float,
    min_hist: int,
) -> float:
    n_obs = len(hist.get(grid[0], []))

    if n_obs < min_hist:
        return default

    best_w = default
    best_sr = -np.inf

    for w_val in grid:
        r = np.asarray(hist.get(w_val, []), float)

        if len(r) < min_hist:
            continue

        std = r.std()
        sr = r.mean() / std * np.sqrt(12) if std > 1e-12 else 0.0

        if sr > best_sr:
            best_sr = sr
            best_w = w_val

    return best_w


def mean_off_diag_corr(cov: np.ndarray) -> float:
    n = cov.shape[0]
    if n <= 1:
        return float("nan")

    d = np.sqrt(np.maximum(np.diag(cov), 1e-12))
    corr = cov / np.outer(d, d)

    iu = np.triu_indices(n, k=1)
    return float(np.mean(corr[iu]))


def compute_diagnostics(
    w: np.ndarray,
    assets: list[str],
    cov: np.ndarray,
    prev_w: np.ndarray | None,
    prev_assets: list[str] | None,
    lw_shrinkage: float | None = None,
    cond_sample: float | None = None,
    cond_lin: float | None = None,
    cond_nls: float | None = None,
) -> dict:
    n = len(assets)

    hhi = float(np.sum(w ** 2))
    eff = 1.0 / hhi if hhi > 0 else float(n)

    var = float(w @ cov @ w)
    vol = float(np.sqrt(max(var, 0.0) * 252))

    gross_long = float(np.maximum(w, 0.0).sum())
    gross_short = float(np.minimum(w, 0.0).sum())
    gross_exposure = float(np.abs(w).sum())
    net_exposure = float(w.sum())
    cash_weight = float(1.0 - net_exposure)

    if prev_w is not None and prev_assets is not None:
        all_a = list(set(assets) | set(prev_assets))

        new_s = pd.Series(w, index=assets).reindex(all_a, fill_value=0.0)
        old_s = pd.Series(prev_w, index=prev_assets).reindex(all_a, fill_value=0.0)

        turnover = float(0.5 * np.abs(new_s.values - old_s.values).sum())
    else:
        turnover = float("nan")

    return {
        "n_assets": n,
        "hhi": hhi,
        "effective_n": eff,
        "ex_ante_vol_ann": vol,
        "turnover": turnover,
        "gross_long": gross_long,
        "gross_short": gross_short,
        "gross_exposure": gross_exposure,
        "net_exposure": net_exposure,
        "cash_weight": cash_weight,
        "lw_shrinkage": lw_shrinkage if lw_shrinkage is not None else np.nan,
        "cond_sample": cond_sample if cond_sample is not None else np.nan,
        "cond_lin": cond_lin if cond_lin is not None else np.nan,
        "cond_nls": cond_nls if cond_nls is not None else np.nan,
    }


def _record_skip(
    rows: list[dict],
    rebal_date: pd.Timestamp,
    impl_date: pd.Timestamp,
    end_date: pd.Timestamp,
    reason: str,
    active_core_names: list[str],
) -> None:
    for m in active_core_names:
        rows.append({
            "rebal_date": rebal_date,
            "impl_date": impl_date,
            "end_date": end_date,
            "model": m,
            "skipped": True,
            "skip_reason": reason,
            "n_assets": 0,
            "hhi": np.nan,
            "effective_n": np.nan,
            "ex_ante_vol_ann": np.nan,
            "turnover": np.nan,
            "gross_long": np.nan,
            "gross_short": np.nan,
            "gross_exposure": np.nan,
            "net_exposure": np.nan,
            "cash_weight": np.nan,
            "lw_shrinkage": np.nan,
            "cond_sample": np.nan,
            "cond_lin": np.nan,
            "cond_nls": np.nan,
            "mean_corr_sample": np.nan,
            "mean_corr_lin":    np.nan,
            "mean_corr_nls":    np.nan,
            "rho_bar_expand": np.nan,
        })


def run_config(config: dict, region: str, base_dir: str) -> None:
    tag = config["tag"]
    run_start = config["run_start"]
    cov_window = config["cov_window"]

    active_scenarios = set(config.get("scenarios", list(SCENARIO_SPECS.keys())))

    active_base_specs = [
        s for s in BASE_MODEL_SPECS
        if s["scenario_key"] in active_scenarios
    ]

    active_epo_specs = [
        s for s in TUNED_EPO_SPECS
        if s["scenario_key"] in active_scenarios
    ]

    active_fixed_names = [
        _epo_fixed_model_name(spec["base_kind"], spec["scenario_key"], w)
        for spec in active_epo_specs
        for w in W_GRID_FIXED
    ]

    active_core_names = (
        BENCHMARK_MODEL_NAMES
        + [s["name"] for s in active_base_specs]
        + [s["name"] for s in active_epo_specs]
    )

    active_daily_names = active_core_names + active_fixed_names

    t0 = time.time()

    print(f"Config: {tag} | region={region.upper()} | cov_window={cov_window} | run_start={run_start}")
    print(f"  scenarios={sorted(active_scenarios)} | EPO grid={len(EPO_GRID)} | fixed-w={len(W_GRID_FIXED)}")

    inp = load_inputs(region, base_dir)

    daily_ret = inp["daily_ret"]
    universe = inp["universe"]
    signals = inp["signals"]
    rf_daily = inp["rf"]

    rf_aligned = (
        rf_daily
        .reindex(daily_ret.index, method="ffill")
        .shift(1)
        .fillna(0.0)
    )

    for freq in ("ME", "M"):
        try:
            rf_monthly = (1.0 + rf_aligned).resample(freq).prod() - 1.0
            break
        except (ValueError, TypeError):
            continue

    all_dates = daily_ret.index

    univ_me = pd.DatetimeIndex(universe["month_end"].unique()).sort_values()

    rebal_dates = univ_me[
        (univ_me.year >= TUNING_START)
        & (univ_me.year <= RUN_END)
    ]

    n_total = len(rebal_dates)

    print(f"  Rebalance dates: {n_total} (tuning from {TUNING_START})")

    tuning_hist: dict[str, dict[float, list[float]]] = {
        spec["name"]: {w: [] for w in EPO_GRID}
        for spec in active_epo_specs
    }

    prev_w: dict[str, np.ndarray | None] = {
        m: None for m in active_core_names
    }

    prev_a: dict[str, list[str] | None] = {
        m: None for m in active_core_names
    }

    model_daily: dict[str, list[pd.Series]] = {
        m: [] for m in active_daily_names
    }

    holdings_rows: list[dict] = []
    diag_rows: list[dict] = []
    tuning_rows: list[dict] = []
    filter_count_rows: list[dict] = []

    n_run = 0
    n_skip = 0
    did_first_cov_check = False

    for i, rebal_date in enumerate(rebal_dates):
        in_run = run_start <= rebal_date.year <= RUN_END

        impl_idx = all_dates.searchsorted(rebal_date, side="right")

        if impl_idx >= len(all_dates):
            continue

        impl_date = all_dates[impl_idx]

        end_date = rebal_dates[i + 1] if i + 1 < n_total else all_dates[-1]

        if impl_date > end_date:
            continue

        adv_top600 = universe[universe["month_end"] == rebal_date]

        if adv_top600.empty:
            if in_run:
                _record_skip(
                    diag_rows,
                    rebal_date,
                    impl_date,
                    end_date,
                    "no_universe_entry",
                    active_core_names,
                )
                n_skip += 1

            continue

        ret_history = daily_ret.loc[daily_ret.index <= rebal_date]

        if len(ret_history) < cov_window:
            continue

        ret_window_full = ret_history.iloc[-cov_window:]

        assets, ret_win, counts = filter_rebalance_universe(
            adv_top600,
            ret_window_full,
            cov_window,
        )

        n = len(assets)

        filter_count_rows.append({
            "rebal_date":  rebal_date,
            "in_run":      bool(in_run),
            "start":       int(counts.get("start",       0)),
            "after_avail": int(counts.get("after_avail", 0)),
            "after_obs":   int(counts.get("after_obs",   0)),
            "after_vol":   int(counts.get("after_vol",   0)),
            "after_dup":   int(counts.get("after_dup",   0)),
            "final":       int(counts.get("final",       0)),
        })

        if in_run or i % 12 == 0:
            tag_str = "[run]    " if in_run else "[warmup] "

            print(
                f"  {rebal_date.date()} {tag_str} "
                f"600->{counts.get('after_obs', '?'):3}"
                f"->{counts.get('after_vol', '?'):3}"
                f"->{counts.get('after_dup', '?'):3}"
                f"->{n:3} final"
            )

        min_req = MIN_ASSETS if in_run else MIN_ASSETS_TUNING

        if n < min_req:
            if in_run:
                print(f"    SKIP: {n} assets < MIN_ASSETS={MIN_ASSETS}")

                _record_skip(
                    diag_rows,
                    rebal_date,
                    impl_date,
                    end_date,
                    f"too_few_assets_{n}",
                    active_core_names,
                )

                n_skip += 1

            continue

        X = ret_win.values

        cov_s = estimate_sample_cov(X)
        cov_l, lw_shrinkage = estimate_lin_cov(X)
        cov_n = estimate_nls_cov(X)

        ret_expand = daily_ret.loc[daily_ret.index <= rebal_date, assets].fillna(0.0)
        corr_expand = ret_expand.corr().values
        iu = np.triu_indices(n, k=1)
        rho_bar_expand = float(np.mean(corr_expand[iu])) if len(iu[0]) > 0 else 0.0

        mean_corr_sample = mean_off_diag_corr(cov_s)
        mean_corr_lin    = mean_off_diag_corr(cov_l)
        mean_corr_nls    = mean_off_diag_corr(cov_n)

        cond_s = float(np.linalg.cond(cov_s))
        cond_l = float(np.linalg.cond(cov_l))
        cond_n = float(np.linalg.cond(cov_n))

        if not did_first_cov_check:
            import sys as _sys

            print(
                f"    [PID {os.getpid()}] first-rebal cov sanity check "
                f"@ {rebal_date.date()}: N={n} | "
                f"diag(s/l/n) = "
                f"{np.diag(cov_s).mean():.3e}/"
                f"{np.diag(cov_l).mean():.3e}/"
                f"{np.diag(cov_n).mean():.3e} | "
                f"frob(s-l)={np.linalg.norm(cov_s - cov_l):.3e} | "
                f"frob(s-n)={np.linalg.norm(cov_s - cov_n):.3e} | "
                f"LW shrink={lw_shrinkage:.3f} | "
                f"cond(s/l/n)={cond_s:.2e}/{cond_l:.2e}/{cond_n:.2e}"
            )
            _sys.stdout.flush()
            did_first_cov_check = True

        raw_sig = _lookup_signal(signals, rebal_date)
        mu = transform_signal(raw_sig, assets)

        best_w: dict[str, float] = {
            spec["name"]: select_epo_w(
                tuning_hist[spec["name"]],
                EPO_GRID,
                EPO_DEFAULT_W,
                EPO_MIN_HISTORY,
            )
            for spec in active_epo_specs
        }

        core_weights = compute_base_weights(
            n=n,
            mu=mu,
            cov_sample=cov_s,
            cov_lin=cov_l,
            cov_nls=cov_n,
            active_base_specs=active_base_specs,
        )

        epo_grid_weights: dict[str, dict[float, np.ndarray]] = {}
        epo_grid_returns: dict[str, dict[float, pd.Series]] = {}

        for spec in active_epo_specs:
            model_name = spec["name"]

            bk = spec["base_kind"]

            if bk == "sample":
                base_cov = cov_s
            elif bk == "lin":
                base_cov = cov_l
            elif bk == "nls":
                base_cov = cov_n
            elif bk == "cc":
                base_cov = cov_s
            elif bk == "cc_lin":
                base_cov = cov_l
            elif bk == "cc_nls":
                base_cov = cov_n
            else:
                raise ValueError(f"Unknown EPO base_kind: {bk}")

            scenario_key = spec["scenario_key"]

            if bk in ("cc", "cc_lin", "cc_nls"):
                weights_by_w = compute_epo_cc_grid_weights(
                    n=n,
                    mu=mu,
                    base_cov=base_cov,
                    rho_bar=rho_bar_expand,
                    scenario_key=scenario_key,
                )
            else:
                weights_by_w = compute_epo_grid_weights_for_variant(
                    n=n,
                    mu=mu,
                    base_cov=base_cov,
                    scenario_key=scenario_key,
                )

            epo_grid_weights[model_name] = weights_by_w

            returns_by_w: dict[float, pd.Series] = {}

            for w_val in EPO_GRID:
                returns_by_w[w_val] = simulate_drift(
                    w0=weights_by_w[w_val],
                    assets=assets,
                    daily_ret=daily_ret,
                    rf_daily=rf_aligned,
                    impl_date=impl_date,
                    end_date=end_date,
                )

            epo_grid_returns[model_name] = returns_by_w

        for spec in active_epo_specs:
            model_name = spec["name"]
            core_weights[model_name] = epo_grid_weights[model_name][best_w[model_name]]

        core_returns: dict[str, pd.Series] = {}

        for m in BENCHMARK_MODEL_NAMES + [s["name"] for s in active_base_specs]:
            core_returns[m] = simulate_drift(
                w0=core_weights[m],
                assets=assets,
                daily_ret=daily_ret,
                rf_daily=rf_aligned,
                impl_date=impl_date,
                end_date=end_date,
            )

        for spec in active_epo_specs:
            model_name = spec["name"]
            core_returns[model_name] = epo_grid_returns[model_name][best_w[model_name]]

        rf_me_idx = rf_monthly.index[rf_monthly.index <= end_date]
        monthly_rf = float(rf_monthly.loc[rf_me_idx[-1]]) if len(rf_me_idx) > 0 else 0.0

        for spec in active_epo_specs:
            model_name = spec["name"]

            for w_val in EPO_GRID:
                r = _monthly_ret(epo_grid_returns[model_name][w_val])

                if not np.isnan(r):
                    tuning_hist[model_name][w_val].append(r - monthly_rf)

        if in_run:
            n_run += 1

            for m in active_core_names:
                model_daily[m].append(core_returns[m])

            for spec in active_epo_specs:
                for w_val in W_GRID_FIXED:
                    fixed_name = _epo_fixed_model_name(
                        spec["base_kind"],
                        spec["scenario_key"],
                        w_val,
                    )

                    model_daily[fixed_name].append(
                        epo_grid_returns[spec["name"]][w_val]
                    )

            for m in active_core_names:
                wv = core_weights[m]

                for j, sid in enumerate(assets):
                    holdings_rows.append({
                        "rebal_date": rebal_date,
                        "impl_date": impl_date,
                        "end_date": end_date,
                        "model": m,
                        "sec_id": sid,
                        "weight": float(wv[j]),
                    })

            for m in active_core_names:
                wv = core_weights[m]

                d = compute_diagnostics(
                    w=wv,
                    assets=assets,
                    cov=cov_n,
                    prev_w=prev_w[m],
                    prev_assets=prev_a[m],
                    lw_shrinkage=lw_shrinkage,
                    cond_sample=cond_s,
                    cond_lin=cond_l,
                    cond_nls=cond_n,
                )

                d.update({
                    "rebal_date": rebal_date,
                    "impl_date": impl_date,
                    "end_date": end_date,
                    "model": m,
                    "skipped": False,
                    "skip_reason": "",
                    "mean_corr_sample": mean_corr_sample,
                    "mean_corr_lin":    mean_corr_lin,
                    "mean_corr_nls":    mean_corr_nls,
                    "rho_bar_expand":   rho_bar_expand,
                })

                diag_rows.append(d)

            row = {"rebal_date": rebal_date}

            for spec in active_epo_specs:
                row[f"w_{spec['name']}"] = best_w[spec["name"]]

            tuning_rows.append(row)

        for m in active_core_names:
            prev_w[m] = core_weights[m]
            prev_a[m] = assets

    print(f"Loop complete: {n_run} run, {n_skip} skipped")

    out_tag = f"{region}_{tag}"

    _save_outputs(
        base_dir=base_dir,
        out_tag=out_tag,
        model_daily=model_daily,
        holdings_rows=holdings_rows,
        diag_rows=diag_rows,
        tuning_rows=tuning_rows,
        filter_count_rows=filter_count_rows,
        rf_aligned=rf_aligned,
        active_core_names=active_core_names,
        active_epo_specs=active_epo_specs,
    )

    print(f"Config {tag} | {region} done in {time.time() - t0:.0f}s")


def _save_outputs(
    base_dir: str,
    out_tag: str,
    model_daily: dict[str, list[pd.Series]],
    holdings_rows: list[dict],
    diag_rows: list[dict],
    tuning_rows: list[dict],
    filter_count_rows: list[dict],
    rf_aligned: pd.Series,
    active_core_names: list[str],
    active_epo_specs: list[dict],
) -> None:
    print(f"Saving outputs [{out_tag}]...")

    frames: dict[str, pd.Series] = {}

    for m, parts in model_daily.items():
        if parts:
            s = pd.concat(parts).sort_index()
            s = s[~s.index.duplicated(keep="first")]
            frames[m] = s

    if frames:
        daily_df = pd.DataFrame(frames)

        p = os.path.join(base_dir, f"daily_portfolio_returns_{out_tag}.parquet")
        daily_df.to_parquet(p)
        print(f"  -> {p}")

        summary = _summary_metrics(daily_df, rf_aligned)

        p_csv = os.path.join(base_dir, f"summary_metrics_{out_tag}.csv")
        summary.to_csv(p_csv)
        print(f"  -> {p_csv}")

        fixed_grid = _fixed_w_sharpe_grid(
            summary=summary,
            active_epo_specs=active_epo_specs,
        )

        if not fixed_grid.empty:
            p_fixed_csv = os.path.join(base_dir, f"fixed_w_sharpe_grid_{out_tag}.csv")
            fixed_grid.to_csv(p_fixed_csv, index=False)
            print(f"  -> {p_fixed_csv}")

            p_fixed_parquet = os.path.join(base_dir, f"fixed_w_sharpe_grid_{out_tag}.parquet")
            fixed_grid.to_parquet(p_fixed_parquet, index=False)
            print(f"  -> {p_fixed_parquet}")

        core = [m for m in active_core_names if m in summary.index]

        if core:
            print(summary.loc[core].to_string())

    if holdings_rows:
        p = os.path.join(base_dir, f"holdings_{out_tag}.parquet")
        pd.DataFrame(holdings_rows).to_parquet(p, index=False)
        print(f"  -> {p}")

    if diag_rows:
        p = os.path.join(base_dir, f"monthly_diagnostics_{out_tag}.parquet")
        pd.DataFrame(diag_rows).to_parquet(p, index=False)
        print(f"  -> {p}")

    if tuning_rows:
        p = os.path.join(base_dir, f"tuning_w_history_{out_tag}.parquet")
        pd.DataFrame(tuning_rows).to_parquet(p, index=False)
        print(f"  -> {p}")

    if filter_count_rows:
        p = os.path.join(base_dir, f"filter_counts_{out_tag}.parquet")
        pd.DataFrame(filter_count_rows).to_parquet(p, index=False)
        print(f"  -> {p}")

    print(f"Outputs saved for {out_tag}.")


def _fixed_w_sharpe_grid(
    summary: pd.DataFrame,
    active_epo_specs: list[dict],
) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()

    rows: list[dict] = []

    for spec in active_epo_specs:
        base_kind = spec["base_kind"]
        scenario_key = spec["scenario_key"]

        for w_val in W_GRID_FIXED:
            model_name = _epo_fixed_model_name(base_kind, scenario_key, w_val)

            if model_name not in summary.index:
                continue

            metrics = summary.loc[model_name].to_dict()

            rows.append({
                "model": model_name,
                "scenario": scenario_key,
                "epo_base": base_kind,
                "w": float(w_val),
                "ann_sharpe": metrics.get("ann_sharpe", np.nan),
                "ann_return": metrics.get("ann_return", np.nan),
                "ann_vol": metrics.get("ann_vol", np.nan),
                "max_drawdown": metrics.get("max_drawdown", np.nan),
                "n_days": metrics.get("n_days", np.nan),
            })

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .sort_values(["scenario", "epo_base", "w"])
        .reset_index(drop=True)
    )


def _summary_metrics(daily_df: pd.DataFrame, rf_aligned: pd.Series) -> pd.DataFrame:
    rows = []

    rf = rf_aligned.reindex(daily_df.index, method="ffill").fillna(0.0)

    for m in daily_df.columns:
        r = daily_df[m].dropna()

        if len(r) < 2:
            continue

        exc = r - rf.loc[r.index]
        n = len(r)

        ann_r = float((1.0 + r).prod() ** (252 / n) - 1.0)
        ann_v = float(r.std(ddof=1) * np.sqrt(252))

        exc_std = exc.std(ddof=1)

        ann_sr = (
            float(exc.mean() / exc_std * np.sqrt(252))
            if exc_std > 1e-12
            else np.nan
        )

        cum = (1.0 + r).cumprod()
        mdd = float(((cum - cum.cummax()) / cum.cummax()).min())

        rows.append({
            "model": m,
            "ann_return": ann_r,
            "ann_vol": ann_v,
            "ann_sharpe": ann_sr,
            "max_drawdown": mdd,
            "n_days": n,
        })

    return pd.DataFrame(rows).set_index("model") if rows else pd.DataFrame()


def _run_all_regions(cfg: dict) -> str:
    for region in REGIONS:
        try:
            run_config(cfg, region, BASE_DIR)
        except Exception:
            print(f"ERROR in config={cfg.get('tag')} region={region}")
            traceback.print_exc()
            raise

    return cfg["tag"]


def main() -> None:
    t_total = time.time()

    os.makedirs(BASE_DIR, exist_ok=True)

    print(f"Starting backtest: regions={REGIONS}, configs={[c['tag'] for c in CONFIGS]}, workers={len(CONFIGS)}")

    with Pool(processes=len(CONFIGS)) as pool:
        completed = pool.map(_run_all_regions, CONFIGS)

    elapsed_total = time.time() - t_total

    print(f"All configs complete: {completed} | {elapsed_total:.0f}s ({elapsed_total / 3600:.2f}h)")


if __name__ == "__main__":
    main()

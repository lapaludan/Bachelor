import os
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

REGIONS    = ["emea"]
YEAR_START = 1980
YEAR_END   = 2025

PRICE_DATA_DIR  = (
    r"C:\Users\lapal\CBS - Copenhagen Business School"
    r"\Martin Richter - equities\price_data"
)
SEC_MAP_DIR     = (
    r"C:\Users\lapal\CBS - Copenhagen Business School"
    r"\Martin Richter - equities\sec_master"
)
SEC_MASTER_FILE = os.path.join(SEC_MAP_DIR, "sec_master_ext.csv")
OUT_DIR         = (
    r"C:\Users\lapal\OneDrive - CBS - Copenhagen Business School"
    r"\6. Semester\Bachelor\Kode V2"
)

COMPANY_COL = "org_perm_id"

TOP_N             = 600
ADV_LOOKBACK_DAYS = 252

ADV_MIN_OBS_FRAC  = 0.80
ADV_MIN_OBS       = int(ADV_LOOKBACK_DAYS * ADV_MIN_OBS_FRAC)

SET_NEXT_DAY_RETURN_TO_NAN = True

MANUAL_RETURN_EXCLUSIONS = [
    ("39835", "2019-03-25"),
    ("41014", "2019-03-25"),
    ("42945", "2019-03-25"),
    ("44322", "2019-03-25"),
    ("47079", "2019-03-25"),
    ("47262", "2019-03-25"),
]

LOOKBACK_MONTHS  = 11
SKIP_MONTHS      =  1
MIN_FRAC_MONTHS  = 0.80
WINSOR_P         = 0.005
VOL_SCALE_SIGNAL = True

REQUIRED_PRICE_COLS = {"sec_id", "date", "tot_ret_usd", "close_usd", "volume"}

MSCI_DEVELOPED_STATIC = {
    "north_america": {"US", "CA"},
    "emea": {
        "GB", "IE", "CH", "FR", "NL", "DE", "FI", "SE", "BE",
        "IT", "NO", "AT", "ES", "DK",
    },
    "asia": {"JP", "AU", "HK", "SG", "NZ"},
}

MSCI_DEVELOPED_DYNAMIC = {
    "emea": {
        "GR": {"start": pd.Timestamp("2001-05-31"), "end_excl": pd.Timestamp("2013-11-27")},
        "PT": {"start": pd.Timestamp("1997-11-30"), "end_excl": pd.Timestamp.max},
        "IL": {"start": pd.Timestamp("2010-05-31"), "end_excl": pd.Timestamp.max},
    }
}


def load_security_master(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Security master not found:\n  {path}")

    df = pd.read_csv(path, low_memory=False)

    required = ["type_code", "sec_id", "country_code"]
    if COMPANY_COL in df.columns:
        required.append(COMPANY_COL)
    else:
        print(f"  '{COMPANY_COL}' ikke fundet")

    for col in required:
        if col not in df.columns:
            raise ValueError(f"Security master missing required column: '{col}'")

    df = df[df["type_code"] == "EQ"].copy()
    df["sec_id"]       = df["sec_id"].astype(str).str.strip()
    df["country_code"] = df["country_code"].astype(str).str.strip()

    if COMPANY_COL in df.columns:
        df[COMPANY_COL] = df[COMPANY_COL].astype(str).str.strip()

    print(f"  Security master: {len(df):,} EQ rows loaded")
    return df


def build_sec_maps(sec_master: pd.DataFrame, region: str) -> tuple[dict, dict]:
    static_countries  = MSCI_DEVELOPED_STATIC[region]
    dynamic_countries = set(MSCI_DEVELOPED_DYNAMIC.get(region, {}).keys())
    include           = static_countries | dynamic_countries

    subset = (
        sec_master[sec_master["country_code"].isin(include)]
        .drop_duplicates("sec_id", keep="last")
    )

    sec_country_map = subset.set_index("sec_id")["country_code"].to_dict()

    sec_company_map = {}
    if COMPANY_COL in subset.columns:
        sec_company_map = subset.set_index("sec_id")[COMPANY_COL].to_dict()

    return sec_country_map, sec_company_map


def load_price_data(region: str, year_start: int, year_end: int) -> pd.DataFrame:
    frames = []
    for year in range(year_start, year_end + 1):
        fpath = os.path.join(PRICE_DATA_DIR, f"prices_{region}_{year}.csv")
        if not os.path.exists(fpath): continue
        df = pd.read_csv(fpath, low_memory=False)
        frames.append(df[list(REQUIRED_PRICE_COLS)])

    if not frames: raise FileNotFoundError("No price files found.")

    df = pd.concat(frames, ignore_index=True)
    df["date"]        = pd.to_datetime(df["date"])
    df["sec_id"]      = df["sec_id"].astype(str).str.strip()
    df["tot_ret_usd"] = pd.to_numeric(df["tot_ret_usd"], errors="coerce")
    df["close_usd"]   = pd.to_numeric(df["close_usd"],   errors="coerce")
    df["volume"]      = pd.to_numeric(df["volume"],      errors="coerce")

    print(f"  Loaded {len(df):,} rows | {df['sec_id'].nunique():,} securities")
    return df


def filter_eligible_securities(price_df: pd.DataFrame, sec_country_map: dict, region: str) -> pd.DataFrame:
    df = price_df.copy()
    df["country_code"] = df["sec_id"].map(sec_country_map)
    df = df.dropna(subset=["country_code"])

    static_countries = MSCI_DEVELOPED_STATIC[region]
    dynamic_rules    = MSCI_DEVELOPED_DYNAMIC.get(region, {})

    ok = df["country_code"].isin(static_countries)
    for country_code, rule in dynamic_rules.items():
        ok_dynamic = (
            (df["country_code"] == country_code) &
            (df["date"] >= rule["start"]) &
            (df["date"] < rule["end_excl"])
        )
        ok = ok | ok_dynamic

    df = df[ok].copy()
    df.drop(columns=["country_code"], inplace=True)
    df = df.sort_values(["sec_id", "date"]).drop_duplicates(["sec_id", "date"], keep="last").reset_index(drop=True)
    print(f"  After region filter: {len(df):,} rows | {df['sec_id'].nunique():,} securities")
    return df


def identify_month_ends(trading_dates) -> pd.DatetimeIndex:
    dts = pd.DatetimeIndex(sorted(set(pd.to_datetime(trading_dates))))
    if len(dts) == 0: return pd.DatetimeIndex([])
    df = pd.DataFrame({"date": dts})
    df["ym"] = df["date"].dt.to_period("M")
    return pd.DatetimeIndex(df.groupby("ym", sort=True)["date"].max().values)


def compute_monthly_universe(price_df: pd.DataFrame, sec_company_map: dict) -> tuple:
    dv = price_df[["sec_id", "date", "close_usd", "volume"]].copy()
    dv["dollar_volume"] = dv["close_usd"] * dv["volume"]

    dv_wide = dv.pivot_table(index="date", columns="sec_id", values="dollar_volume", aggfunc="last").sort_index()

    print(f"  Rolling ADV (window={ADV_LOOKBACK_DAYS}, min_obs={ADV_MIN_OBS})...")
    adv_mean  = dv_wide.rolling(ADV_LOOKBACK_DAYS, min_periods=ADV_MIN_OBS).mean()
    adv_count = dv_wide.rolling(ADV_LOOKBACK_DAYS, min_periods=1).count()

    month_ends       = identify_month_ends(dv_wide.index)
    records          = []
    ab_dropped_total = 0

    for me in month_ends:
        if me not in adv_mean.index: continue
        adv_row = adv_mean.loc[me].dropna()
        if adv_row.empty: continue

        adv_df = adv_row.reset_index(name="adv")

        if sec_company_map:
            adv_df["company_id"] = adv_df["sec_id"].map(sec_company_map)
            adv_df = adv_df.sort_values("adv", ascending=False)
            pre_count = len(adv_df)
            adv_df = adv_df.drop_duplicates(subset=["company_id"], keep="first")
            ab_dropped_total += (pre_count - len(adv_df))

        adv_df  = adv_df.head(TOP_N)
        top_idx = adv_df["sec_id"].values
        n       = len(top_idx)

        records.append(pd.DataFrame({
            "month_end":      me,
            "sec_id":         top_idx,
            "adv_usd":        adv_df["adv"].values,
            "n_obs_adv":      adv_count.loc[me, top_idx].values.astype(int),
            "adv_rank":       np.arange(1, n + 1),
            "selection_date": me,
        }))

    universe = pd.concat(records, ignore_index=True)
    print(f"  A/B deduplication dropped {ab_dropped_total:,} stock-months")
    print(f"  Universe: {len(universe):,} stock-months over {universe['month_end'].nunique()} months")

    return universe, month_ends


def _build_exclusion_mask(df: pd.DataFrame, exclusions: list) -> pd.Series:
    if not exclusions:
        return pd.Series(False, index=df.index)

    excl_df = pd.DataFrame(exclusions, columns=["sec_id", "date"])
    excl_df["sec_id"] = excl_df["sec_id"].astype(str).str.strip()
    excl_df["date"]   = pd.to_datetime(excl_df["date"])
    excl_set = set(zip(excl_df["sec_id"], excl_df["date"]))

    keys = list(zip(df["sec_id"].astype(str), df["date"]))
    return pd.Series([k in excl_set for k in keys], index=df.index)


def build_daily_returns(price_df: pd.DataFrame, universe_df: pd.DataFrame, region: str) -> pd.DataFrame:
    df = price_df[["sec_id", "date", "tot_ret_usd"]].copy()
    df = df.sort_values(["sec_id", "date"]).reset_index(drop=True)

    df["ret"] = df.groupby("sec_id")["tot_ret_usd"].pct_change()

    excluded = _build_exclusion_mask(df, MANUAL_RETURN_EXCLUSIONS)
    n_excl   = int(excluded.sum())
    print(f"  Manual return exclusions matched: {n_excl} rows (of {len(MANUAL_RETURN_EXCLUSIONS)} requested)")

    if n_excl > 0:
        excl_report = df.loc[excluded, ["sec_id", "date", "tot_ret_usd", "ret"]].copy()
        excl_file   = os.path.join(OUT_DIR, f"manual_exclusions_applied_{region}.csv")
        excl_report.to_csv(excl_file, index=False)
        print(f"  Saved manual-exclusion audit: {excl_file}")

    df.loc[excluded, "ret"] = np.nan

    if SET_NEXT_DAY_RETURN_TO_NAN and n_excl > 0:
        df["_x"]    = excluded
        df["_next"] = df.groupby("sec_id")["_x"].shift(1).fillna(False)
        n_next      = int(df["_next"].sum())
        df.loc[df["_next"], "ret"] = np.nan
        df.drop(columns=["_x", "_next"], inplace=True)
        print(f"  Next-day returns also set to NaN: {n_next}")

    ret_wide = df.pivot_table(index="date", columns="sec_id", values="ret", aggfunc="last").sort_index()
    print(f"  Return matrix: {ret_wide.shape[0]} dates x {ret_wide.shape[1]} stocks")
    return ret_wide


def build_momentum_signals(ret_wide: pd.DataFrame, universe_df: pd.DataFrame) -> pd.DataFrame:
    lo = ret_wide.quantile(WINSOR_P, axis=1)
    hi = ret_wide.quantile(1.0 - WINSOR_P, axis=1)
    ret_wins = ret_wide.clip(lower=lo, upper=hi, axis=0)

    log_ret = np.log1p(ret_wins)

    for freq in ("ME", "M"):
        try:
            log_monthly = log_ret.resample(freq).sum(min_count=1)
            break
        except (ValueError, TypeError):
            continue

    shifted   = log_monthly.shift(SKIP_MONTHS)
    min_valid = int(np.ceil(MIN_FRAC_MONTHS * LOOKBACK_MONTHS))

    mom_log = shifted.rolling(LOOKBACK_MONTHS, min_periods=min_valid).sum()
    mom     = np.expm1(mom_log)

    if VOL_SCALE_SIGNAL:
        # Dividing by realised vol puts the signal in return/vol units, which normalises
        # cross-sectional dispersion and reduces the influence of high-volatility names.
        vol_daily   = ret_wins.rolling(252, min_periods=126).std() * np.sqrt(252)
        vol_monthly = vol_daily.reindex(mom.index, method="ffill")
        mom         = mom / vol_monthly.clip(lower=0.01)

    univ_wide = universe_df.pivot_table(
        index="month_end", columns="sec_id", values="adv_rank", aggfunc="first"
    ).notna()

    common_dates  = mom.index.intersection(univ_wide.index)
    common_stocks = mom.columns.intersection(univ_wide.columns)

    if len(common_dates) == 0:
        aligned = {}
        for ume in univ_wide.index:
            prior = mom.index[mom.index <= ume]
            if len(prior) == 0: continue
            last = prior[-1]
            if last.to_period("M") == ume.to_period("M"):
                aligned[ume] = mom.loc[last]
        if aligned:
            mom           = pd.DataFrame(aligned).T
            common_dates  = mom.index.intersection(univ_wide.index)
            common_stocks = mom.columns.intersection(univ_wide.columns)

    mom_aln  = mom.loc[common_dates, common_stocks]
    mask_aln = univ_wide.reindex(index=common_dates, columns=common_stocks, fill_value=False)
    signals  = mom_aln.where(mask_aln)

    valid = int(signals.notna().sum().sum())
    total = int(mask_aln.sum().sum())
    print(f"  Signal coverage: {valid:,} / {total:,} universe cells ({valid / max(total, 1):.1%})")

    return signals


def run_region(region: str, sec_master: pd.DataFrame) -> str:
    t0 = time.time()
    print(f"\nEquity Data Pipeline | {region.upper()} | {YEAR_START}-{YEAR_END}")

    try:
        print("[1/5] Building country & company maps")
        sec_country_map, sec_company_map = build_sec_maps(sec_master, region)

        print("[2/5] Loading and filtering price data")
        raw    = load_price_data(region, YEAR_START, YEAR_END)
        prices = filter_eligible_securities(raw, sec_country_map, region)
        del raw

        print("[3/5] Building ADV monthly universe")
        universe_df, month_ends = compute_monthly_universe(prices, sec_company_map)

        p_univ = os.path.join(OUT_DIR, f"monthly_universe_{region}_top600_adv.parquet")
        universe_df.to_parquet(p_univ, index=False)
        pd.DataFrame({"month_end": month_ends}).to_parquet(
            os.path.join(OUT_DIR, f"month_ends_{region}.parquet"), index=False
        )

        print("[4/5] Building daily return panel")
        ret_wide = build_daily_returns(prices, universe_df, region)
        ret_wide.to_parquet(os.path.join(OUT_DIR, f"daily_returns_{region}_tot_ret_usd.parquet"))

        print("[5/5] Building momentum signals")
        signals = build_momentum_signals(ret_wide, universe_df)
        signals.to_parquet(os.path.join(OUT_DIR, f"signals_{region}_mom_2_12_top600_1980_2025.parquet"))

        elapsed = time.time() - t0
        print(f"\n{region.upper()} done in {elapsed:.0f}s ({elapsed / 60:.1f} min)")
        return "ok"

    except Exception as e:
        import traceback
        print(f"\nERROR in {region}: {e}\n{traceback.format_exc()}")
        return f"error: {e}"


def main():
    t_total = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading security master...")
    sec_master = load_security_master(SEC_MASTER_FILE)

    for region in REGIONS:
        run_region(region, sec_master)

    elapsed = time.time() - t_total
    print(f"\nTotal runtime: {elapsed:.0f}s")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
East Africa (Kenya) Long-Rains ML Pipeline  — 15 May 2025
"""

# --- std & 3rd-party ----------------------------------------------------------
import argparse, time
from pathlib import Path
from typing import Tuple, Optional

import requests, numpy as np, pandas as pd
import rasterio
from rasterio.windows import from_bounds
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib, matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ──────────────────────────────────────────────────────────────────────────────
# 🔵 0.  CLIMATE-DRIVER FETCH (Niño 3·4 & DMI)                                  │
# ──────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# 0.  CLIMATE-DRIVER FETCH (local CSVs)                                        │
# ─────────────────────────────────────────────────────────────────────────────
def fetch_climate_drivers(
        nino_path: Path = Path("data/raw/nino34.long.anom.csv"),
        dmi_path : Path = Path("data/raw/dmi.had.long.csv"),
) -> pd.DataFrame:
    """
    Return dataframe with columns:
        year , nino34 , dmi
    Values are March–May (MAM) means.

    CSVs must have two columns:
        Date , <index_value>
    Date can be YYYY-MM-DD or M/D/YYYY.
    """

    def _load(csv_path: Path, col_name: str, miss_flag: float) -> pd.Series:
        if not csv_path.exists():
            raise FileNotFoundError(f"{csv_path} not found — download from PSL first.")

        # read, skipping comment lines that start with anything non-numeric
        df = (pd.read_csv(csv_path, comment='#')
                .rename(columns=lambda c: c.strip())          # tidy names
                .replace(miss_flag, np.nan))                  # NaNs for missing

        # Robust date parse (handles 1/1/1900 or 1900-01-01)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])                       # drop bad rows
        df = df.set_index('Date')[df.columns[1]]              # take the value col
        df.name = col_name
        return df

    n34 = _load(nino_path, 'nino34',  -99.99)
    dmi = _load(dmi_path , 'dmi'   , -9999)

    # MAM season mean, one value per hydrological year (year that *starts* in Mar)
    drivers = (
        pd.concat([n34, dmi], axis=1)
          .resample('AS-MAR')       # 1 Mar of each year
          .mean()
    )
    drivers.index = drivers.index.year         # <-- ints, not Timestamps
    drivers.index.name = "year"                # give the index a name
    drivers = drivers.reset_index()    

    return drivers

# ──────────────────────────────────────────────────────────────────────────────
# 1.  CONFIG / CONSTANTS                                                        │
# ──────────────────────────────────────────────────────────────────────────────

DEF_BBOX   = (-4.62, 4.62, 33.5, 41.9)          # south, north, west, east
DATA_DIR   = Path("data/raw")
PROC_DIR   = Path("data/processed")
OUT_DIR    = Path("outputs")
for _d in (DATA_DIR, PROC_DIR, OUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

DL_FILE = DATA_DIR / "chirps_mam.csv" 

CHIRPS_URL = (
    "https://data.chc.ucsb.edu/products/CHIRPS/v3.0/monthly/africa/tifs/chirps-v3.0.{year}.{month:02d}.tif"
)

def download_chirps_month(year: int, month: int,
                          bbox: Tuple[float, float, float, float],
                          retries: int = 3, timeout: int = 60) -> float:
    """Download one CHIRPS monthly GeoTIFF, clip to bbox and return area-mean (mm)."""
    url = CHIRPS_URL.format(year=year, month=month)
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            with rasterio.MemoryFile(resp.content) as mem:
                with mem.open() as src:
                    window = from_bounds(
                        bbox[2], bbox[0], bbox[3], bbox[1], transform=src.transform
                    )
                    data = src.read(1, window=window)
                     # CHIRPS nodata is -9999 (or occasionally -8888)
                    data = np.where(data <= -9990, np.nan, data)
                    
                    return float(np.nanmean(data))
        except Exception as e:
            if attempt == retries:
                raise
            backoff = 5 * attempt
            print(f"🔄  Retry {attempt}/{retries} for {year}-{month:02d} after {backoff}s … {e}")
            time.sleep(backoff)

# ──────────────────────────────────────────────────────────────────────────────
# 2b.  LOOP OVER YEARS (March–May only)                                         │
# ──────────────────────────────────────────────────────────────────────────────
def download_year_range(
        start: int,
        end: int,
        bbox: Tuple[float, float, float, float] = DEF_BBOX,
        force: bool = False
):
    """Download March–May CHIRPS means for each year in [start, end]."""
    # ── 1 ▸ reuse cache if allowed
    if DL_FILE.exists() and not force:
        print("✔️  Using cached CHIRPS table", DL_FILE)
        return                       # nothing else to do

    # ── 2 ▸ otherwise fetch everything
    records: list[dict] = []
    for y in range(start, end + 1):
        for m in (3, 4, 5):          # March-April-May
            try:
                mm = download_chirps_month(y, m, bbox)
                records.append({"year": y, "month": m, "rain_mm": mm})
                print(f"Downloaded {y}-{m:02d}")
            except Exception as e:
                print(f"⚠️  {y}-{m:02d} failed: {e}")

    out = pd.DataFrame(records)
    if out.empty:                    # <- property, **not** callable
        raise RuntimeError("No data downloaded — check years or connection.")

    out.to_csv(DL_FILE, index=False)
    print("📁  Saved", DL_FILE)

# ──────────────────────────────────────────────────────────────────────────────
# 3.  PRE-PROCESS                                                               │
# ──────────────────────────────────────────────────────────────────────────────
def preprocess() -> pd.DataFrame:
    """Clean CHIRPS monthly MAM table, build season totals & basic indices."""
    df = pd.read_csv(DL_FILE)                     # <- already cached
    if df.empty:
        raise ValueError("chirps_mam.csv is empty.")

    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    df["month"] = df["month"].astype(int)
    full_idx = pd.MultiIndex.from_product(
        [range(df.year.min(), df.year.max() + 1), (3, 4, 5)],
        names=["year", "month"]
    )
    df = (df.set_index(["year", "month"])
            .reindex(full_idx)      # missing ⇒ NaN
            .reset_index())

    n_missing = df.rain_mm.isna().sum()
    if n_missing:
        print(f"ℹ️  {n_missing} station-months missing → kept as NaN")

    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    n_neg = (df.rain_mm < 0).sum(skipna=True)
    if n_neg:
        print(f"⚠️  {n_neg} negative values → set to NaN")
        df.loc[df.rain_mm < 0, "rain_mm"] = np.nan
    # keep the cleaned monthly file for future runs
    df.to_csv(DL_FILE, index=False)

    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    season = (df.groupby("year")
                .rain_mm
                .sum(min_count=3)           # if any month NaN ⇒ total NaN
                .rename("total_mm")
                .reset_index())

    # flag / exclude implausible (≤0 mm) seasons from climatological stats
    bad_tot = season.total_mm <= 0
    if bad_tot.any():
        print(f"⚠️  {bad_tot.sum()} seasons have non-positive totals – "
              "excluded from SPI fit")
        season.loc[bad_tot, "total_mm"] = np.nan

    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    clim_mean = season.loc[
        season.year.between(1991, 2020), "total_mm"
    ].mean(skipna=True)

    # fall-back if reference window has gaps
    if np.isnan(clim_mean) or clim_mean == 0:
        clim_mean = season.total_mm.mean(skipna=True)

    season["anom_pct"] = 100 * (season.total_mm - clim_mean) / clim_mean
    season["anom_z"]   = (
        (season.total_mm - season.total_mm.mean(skipna=True)) /
        season.total_mm.std(ddof=0, skipna=True)
    )

    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    valid = season.total_mm.dropna()
    if valid.empty:
        print("⚠️  No valid season totals – skipping SPI computation")
        season["spi3"] = np.nan
    else:
        shp, loc, scl = stats.gamma.fit(valid, floc=0)
        season["spi3"] = stats.norm.ppf(
            stats.gamma.cdf(season.total_mm, shp, loc=loc, scale=scl)
    )

    season.to_csv(PROC_DIR / "season_totals.csv", index=False)
    print(season.head())            # preview first, then return
    return season


# ──────────────────────────────────────────────────────────────────────────────
# 3a.  TREND SIGNIFICANCE (MK + Sen)                                         │
# ──────────────────────────────────────────────────────────────────────────────
def trend_test(season_df: pd.DataFrame):
    if len(season_df) < 20: return
    tau, p = stats.kendalltau(season_df.year, season_df.total_mm)
    slope  = stats.theilslopes(season_df.total_mm, season_df.year)[0]
    if p < 0.05:
        print(f"⭑ Trend: {slope:+.1f} mm / yr (τ={tau:.2f}, p={p:.3f})")
    else:
        print(f"No significant monotonic trend (p={p:.2f}).")

# ──────────────────────────────────────────────────────────────────────────────
# 4. FORECAST                                                            
# ──────────────────────────────────────────────────────────────────────────────

def forecast(season_df: pd.DataFrame, steps: int = 3):
    """ARIMA forecast of season totals. Soft-skips when < 4 seasons."""
    series = season_df.set_index("year")["total_mm"].asfreq("AS-MAR")
    if series.isna().any():
        series = series.interpolate()
    if len(series) < 4:
        print(f"⚠️  Only {len(series)} season(s) available — skipping forecast step.")
        return  # gracefully exit so the rest of the pipeline continues

    model = ARIMA(series, order=(1, 0, 0)).fit()
    fc = model.get_forecast(steps=steps)
    fc_df = fc.summary_frame()
    fc_df.index = [series.index[-1].year + i for i in range(1, steps + 1)]
    fc_df.to_csv(OUT_DIR / "forecast.csv")

    # Plot
    plt.figure(figsize=(7, 4))
    plt.plot(series.index.year, series, label="observed")
    plt.plot(fc_df.index, fc_df["mean"], "--", label="forecast")
    plt.fill_between(fc_df.index, fc_df["mean_ci_lower"], fc_df["mean_ci_upper"], alpha=0.2)
    plt.title("Kenya March–May rainfall — observed & ARIMA forecast")
    plt.xlabel("Year"); plt.ylabel("mm"); plt.legend()
    plt.tight_layout(); plt.savefig(OUT_DIR / "forecast.png", dpi=150)
    plt.close()
# ──────────────────────────────────────────────────────────────────────────────

# 5. CLUSTERING                                                                
# ──────────────────────────────────────────────────────────────────────────────

def cluster(season_df: pd.DataFrame, k: int = 4, *, impute: bool = False):
    """
    K-means on the MAM monthly matrix.
    Set impute=True to mean-fill NaNs instead of dropping seasons.
    """
    df = pd.read_csv(DL_FILE)
    pivot = (
        df.pivot(index="year", columns="month", values="rain_mm")
          .reindex(columns=[3, 4, 5])
    )

    if impute:
        pivot_filled = pivot.fillna(pivot.mean())
        n_dropped = 0
    else:
        n_dropped = pivot.isna().any(axis=1).sum()
        pivot_filled = pivot.dropna()

    if n_dropped:
        print(f"ℹ️  Dropped {n_dropped} season(s) with missing months "
              f"before clustering")

    scaler = StandardScaler()
    X      = scaler.fit_transform(pivot_filled.values)

    km = KMeans(n_clusters=k, n_init="auto", random_state=0).fit(X)
    pivot_filled["cluster"] = km.labels_
    pivot_filled.to_csv(OUT_DIR / "clustered_years.csv")

    joblib.dump(km, OUT_DIR / "kmeans.pkl")
    print("Cluster counts:", np.bincount(km.labels_))

# ──────────────────────────────────────────────────────────────────────────────
# 6. CLASSIFICATION                                                             
# ──────────────────────────────────────────────────────────────────────────────

def classify(season_df: pd.DataFrame):
    q_lo, q_hi = season_df.total_mm.quantile([0.33, 0.66])
    def lab(x): return 0 if x < q_lo else 2 if x > q_hi else 1
    season_df["label"] = season_df.total_mm.apply(lab)
    for lag in (1, 2, 3):
        season_df[f"lag{lag}"] = season_df.total_mm.shift(lag)
    season_df = season_df.dropna()
    if season_df.empty:
        print("⚠️  Not enough labelled seasons to train classifier — skipping class step.")
        return
    X = season_df[[f"lag{l}" for l in (1, 2, 3)]].values
    y = season_df["label"].values
    clf = RandomForestClassifier(n_estimators=300, random_state=0).fit(X, y)
    joblib.dump(clf, OUT_DIR / "rf_classifier.pkl")
    print(f"Training accuracy: {clf.score(X, y):.2f}")

# ──────────────────────────────────────────────────────────────────────────────
# 7.  PLOT                                                                      │
# ──────────────────────────────────────────────────────────────────────────────
def plot_season(season_df: pd.DataFrame, window:Optional[int]=10):
    fig, ax  = plt.subplots(figsize=(10,4))
    ax.plot(season_df.year, season_df.total_mm, lw=1.6, marker="o", label="Season total")

    q_lo,q_hi = season_df.total_mm.quantile([.33,.66])
    ax.axhspan(0,q_lo, 0,1, color="red",  alpha=.07, label="Dry tercile")
    ax.axhspan(q_hi,ax.get_ylim()[1],0,1,color="blue", alpha=.07,label="Wet tercile")

    # % of normal
    ax2 = ax.twinx()
    ax2.plot(season_df.year, season_df.anom_pct, ls="--", lw=1.2, color="grey", label="% of normal")
    ax2.set_ylabel("% of normal"); ax2.axhline(0,color="grey",alpha=.3)

    # SPI bars – outward spine so ticks don’t collide
    ax3 = ax.twinx()
    ax3.spines["right"].set_position(("outward",40))
    ax3.bar(season_df.year, season_df.spi3, width=.6, color="purple", alpha=.3, label="SPI-3")
    ax3.set_ylabel("SPI-3"); ax3.axhline(0,color="purple",lw=.8,alpha=.4)

    if window and len(season_df)>=window:
        ax.plot(season_df.year, season_df.total_mm.rolling(window,center=True).mean(),
                lw=2.2, ls="--", label=f"{window}-season mean")

    ax.set_title("Kenya March–May rainfall"); ax.set_xlabel("Year"); ax.set_ylabel("mm")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True)); ax.grid(alpha=.3)

    # merge legends
    lines,labels = [],[]
    for a in (ax,ax2,ax3):
        l,lbl = a.get_legend_handles_labels(); lines+=l; labels+=lbl
    ax.legend(lines,labels,loc="upper center", bbox_to_anchor=(0.5,-0.15), ncol=3)
    fig.tight_layout(); fig.savefig(OUT_DIR/"season_totals.png", dpi=150); plt.close(fig)

# ──────────────────────────────────────────────────────────────────────────────
# 8. CLI ENTRY-POINT                                                               
# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="chirps",
                    choices=["chirps", "power", "era5"])
    ap.add_argument("--start", type=int, default=2015)
    ap.add_argument("--end",   type=int, default=2024)
    ap.add_argument("--bbox",  type=float, nargs=4,
                    help="south north west east")
    # workflow switches
    ap.add_argument("--run-all",        action="store_true")
    ap.add_argument("--download-only",  action="store_true")
    ap.add_argument("--preprocess-only",action="store_true")
    ap.add_argument("--forecast-only",  action="store_true")
    ap.add_argument("--cluster-only",   action="store_true")
    ap.add_argument("--classify-only",  action="store_true")
    ap.add_argument("--plot-only",      action="store_true",
                    help="Just refresh the season_totals.png plot")
    ap.add_argument("--redownload",     action="store_true",
                    help="Ignore cache and pull CHIRPS rasters again")
    args = ap.parse_args()

    bbox = tuple(args.bbox) if args.bbox else DEF_BBOX

    # ---------------- download ------------------------------------------
    if args.run_all or args.download_only:
        download_year_range(args.start, args.end, bbox,
                            force=args.redownload)

    # ---------------- preprocess ----------------------------------------
    if args.run_all or args.preprocess_only or args.plot_only:
        season  = preprocess()

        drivers = fetch_climate_drivers()
        season  = season.merge(drivers, on="year", how="left")

        merged_path = PROC_DIR / "season_totals_with_drivers.csv"
        season.to_csv(merged_path, index=False)
        trend_test(season)
    else:
        merged_path = PROC_DIR / "season_totals_with_drivers.csv"
        season = pd.read_csv(merged_path) if merged_path.exists() else None

    if season is None:
        print("No processed data – run preprocessing first.")
        return

    # ---------------- downstream steps ----------------------------------
    if args.run_all or args.forecast_only:
        forecast(season)
    if args.run_all or args.cluster_only:
        cluster(season)
    if args.run_all or args.classify_only:
        classify(season)
    if args.run_all or args.plot_only:
        plot_season(season, window=10)

if __name__ == "__main__":
    main()

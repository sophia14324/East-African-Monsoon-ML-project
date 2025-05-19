#!/usr/bin/env python
"""
East Africa (Kenya) Long-Rains ML Pipeline  — 15 May 2025
"""

# --- std & 3rd-party ----------------------------------------------------------
import argparse, gzip, time
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
    drivers.index = drivers.index.year   # int years for merge
    return drivers

# ──────────────────────────────────────────────────────────────────────────────
# 1.  CONFIG / CONSTANTS                                                        │
# ──────────────────────────────────────────────────────────────────────────────
DEF_BBOX   = (-4.62, 4.62, 33.5, 41.9)          # south, north, west, east (Kenya)
DATA_DIR   = Path("data/raw")
PROC_DIR   = Path("data/processed")
OUT_DIR    = Path("outputs")
for d in (DATA_DIR, PROC_DIR, OUT_DIR): d.mkdir(parents=True, exist_ok=True)

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
                    data = src.read(1, window=window, masked=True)
                    return float(data.mean())
        except Exception as e:
            if attempt == retries:
                raise
            backoff = 5 * attempt
            print(f"🔄  Retry {attempt}/{retries} for {year}-{month:02d} after {backoff}s … {e}")
            time.sleep(backoff)

# ──────────────────────────────────────────────────────────────────────────────
# 2b.  LOOP OVER YEARS (March–May only)                                         │
# ──────────────────────────────────────────────────────────────────────────────
def download_year_range(start: int, end: int,
                        bbox: Tuple[float, float, float, float] = DEF_BBOX):
    """Download March-May CHIRPS means for every season in [start, end]."""
    records = []
    for y in range(start, end + 1):
        for m in (3, 4, 5):          # March, April, May
            try:
                mm = download_chirps_month(y, m, bbox)
                records.append({"year": y, "month": m, "rain_mm": mm})
                print(f"Downloaded {y}-{m:02d}")
            except Exception as e:
                print(f"⚠️  {y}-{m:02d} failed: {e}")

    out = pd.DataFrame(records)
    if out.empty:
        raise RuntimeError("No data downloaded — check years or connection.")
    out.to_csv(DATA_DIR / "chirps_mam.csv", index=False)

# ──────────────────────────────────────────────────────────────────────────────
# 3.  PRE-PROCESS                                                               │
# ──────────────────────────────────────────────────────────────────────────────
def preprocess() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR/"chirps_mam.csv")
    if df.empty: raise ValueError("chirps_mam.csv is empty.")

    if (neg := (df.rain_mm<0).sum()):
        print(f"⚠️  {neg} negative values → abs(mm)")
        df["rain_mm"] = df.rain_mm.clip(lower=0); df.to_csv(DATA_DIR/"chirps_mam.csv", index=False)

    season = df.groupby("year").rain_mm.sum().rename("total_mm").reset_index()

    clim_mean = season.loc[season.year.between(1991,2020),"total_mm"].mean()
    if np.isnan(clim_mean) or clim_mean == 0:
        clim_mean = season["total_mm"].mean()
    season["anom_pct"] = 100*(season.total_mm - (clim_mean or season.total_mm.mean()))/clim_mean
    season["anom_z"]   = (season.total_mm - season.total_mm.mean())/season.total_mm.std(ddof=0)

    # SPI-3
    shp, loc, scl = stats.gamma.fit(season.total_mm, floc=0)
    season["spi3"] = stats.norm.ppf(stats.gamma.cdf(season.total_mm, shp, loc=loc, scale=scl))

    season.to_csv(PROC_DIR/"season_totals.csv", index=False)
    return season

# ──────────────────────────────────────────────────────────────────────────────
# 🔵 3a.  TREND SIGNIFICANCE (MK + Sen)                                         │
# ──────────────────────────────────────────────────────────────────────────────
def trend_test(season_df: pd.DataFrame):
    if len(season_df) < 20: return
    tau, p = stats.kendalltau(season_df.year, season_df.total_mm)
    slope  = stats.theilslopes(season_df.total_mm, season_df.year)[0]
    if p < 0.05:
        print(f"⭑ Trend: {slope:+.1f} mm / yr (τ={tau:.2f}, p={p:.3f})")
    else:
        print(f"No significant monotonic trend (p={p:.2f}).")


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

def cluster(season_df: pd.DataFrame, k: int = 4):
    df = pd.read_csv(DATA_DIR / "chirps_mam.csv")
    pivot = df.pivot(index="year", columns="month", values="rain_mm").reindex(columns=[3, 4, 5])
    scaler = StandardScaler(); X = scaler.fit_transform(pivot.values)
    km = KMeans(n_clusters=k, n_init="auto", random_state=0).fit(X)
    pivot["cluster"] = km.labels_
    pivot.to_csv(OUT_DIR / "clustered_years.csv")
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
    ap.add_argument("--dataset", default="chirps", choices=["chirps", "power", "era5"])
    ap.add_argument("--start", type=int, default=2015)
    ap.add_argument("--end", type=int, default=2024)
    ap.add_argument("--bbox", type=float, nargs=4, help="south north west east")
    ap.add_argument("--run-all", action="store_true")
    ap.add_argument("--download-only", action="store_true")
    ap.add_argument("--preprocess-only", action="store_true")
    ap.add_argument("--forecast-only", action="store_true")
    ap.add_argument("--cluster-only", action="store_true")
    ap.add_argument("--classify-only", action="store_true")
    ap.add_argument("--plot-only", action="store_true", help="Just draw/refresh the season_totals.png plot")
    args = ap.parse_args()

    bbox = tuple(args.bbox) if args.bbox else DEF_BBOX

    if args.run_all or args.download_only:
        download_year_range(args.start, args.end, bbox)
    if args.run_all or args.preprocess_only or args.plot_only:
        season = preprocess()
        season = season.merge(fetch_climate_drivers(), on="year", how="left")
        
        merged_path = PROC_DIR / "season_totals_with_drivers.csv"
        season.to_csv(merged_path, index=False)

        trend_test(season)
    else:
        season = pd.read_csv(PROC_DIR/"season_totals_with_drivers.csv") \
                 if (PROC_DIR/"season_totals_with_drivers.csv").exists() else None

    if season is None:
        print("No processed data – run preprocessing first."); return
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

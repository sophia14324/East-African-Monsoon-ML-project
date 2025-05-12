#!/usr/bin/env python
"""
East Africa (Kenya) Long-Rains ML Pipeline — REVISED 12 May 2025
Author : Your-Name-Here
Licence: MIT

▶ What it does
   1. Downloads March–May rainfall for a chosen period (CHIRPS v2 default)
   2. Pre-processes to tidy seasonal tables
   3. Trend analysis & forecasting (ARIMA baseline — auto-skips if < 4 seasons)
   4. Spatial clustering of grid-cells
   5. Simple season-type classification using Random-Forest

Run everything (example 2010-2024):
   python east_africa_monsoon_project.py --start 2010 --end 2024 --run-all
"""
import argparse, gzip, time, datetime as dt
from pathlib import Path
from matplotlib.ticker import MaxNLocator 
from typing import Tuple, Optional
import requests, numpy as np, pandas as pd, rasterio
from rasterio.windows import from_bounds
from statsmodels.tsa.arima.model import ARIMA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib, matplotlib.pyplot as plt
from scipy import stats                      


################################################################################
# CONFIG                                                                        
################################################################################
DEF_BBOX = (-4.62, 4.62, 33.5, 41.9)   # south, north, west, east (Kenya)
DATA_DIR  = Path("data/raw")
PROC_DIR  = Path("data/processed")
OUT_DIR   = Path("outputs")
for d in (DATA_DIR, PROC_DIR, OUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

CHIRPS_URL = (
    "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_monthly/tifs/"
    "chirps-v2.0.{year}.{month:02d}.tif.gz"
)

################################################################################
# 1. DATA DOWNLOAD                                                              
################################################################################

def download_chirps_month(year: int, month: int,
                          bbox: Tuple[float, float, float, float],
                          retries: int = 3, timeout: int = 60) -> float:
    """Download one CHIRPS monthly GeoTIFF, clip to bbox and return area-mean (mm)."""
    url = CHIRPS_URL.format(year=year, month=month)
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            with rasterio.MemoryFile(gzip.decompress(resp.content)) as mem:
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


def download_year_range(start: int, end: int,
                        bbox: Tuple[float, float, float, float] = DEF_BBOX):
    records = []
    for y in range(start, end + 1):
        for m in (3, 4, 5):
            try:
                mm = download_chirps_month(y, m, bbox)
                records.append({"year": y, "month": m, "rain_mm": mm})
                print(f"Downloaded {y}-{m:02d}")
            except Exception as e:
                print(f"⚠️  {y}-{m:02d} failed: {e}")
    out = pd.DataFrame(records)
    if out.empty:
        raise RuntimeError("No data downloaded — check your dates / connection.")
    out.to_csv(DATA_DIR / "chirps_mam.csv", index=False)

################################################################################
# 2. PRE-PROCESSING                                                             
################################################################################

def preprocess():
    df_path = DATA_DIR / "chirps_mam.csv"
    if not df_path.exists():
        raise FileNotFoundError("No raw CSV found — run download stage first.")
    df = pd.read_csv(df_path)
    if df.empty:
        raise ValueError("chirps_mam.csv is empty — all downloads failed?")

    n_neg = (df["rain_mm"] < 0).sum()
    if n_neg:
        print(f"⚠️  Found {n_neg} negative grid-mean values — converting to abs(mm).")
        df["rain_mm"] = df["rain_mm"].abs()
        df.to_csv(df_path, index=False)     

    # --- seasonal aggregation -------------------------------------------------
    season = df.groupby("year")["rain_mm"].agg(total_mm="sum").reset_index()

    clim_mean = season.loc[(season.year >= 1991) & (season.year <= 2020),
                           "total_mm"].mean()
    if np.isnan(clim_mean):                  
        clim_mean = season["total_mm"].mean()
    season["anom_pct"] = 100 * (season.total_mm - clim_mean) / clim_mean

    season["anom_z"] = (season.total_mm - season.total_mm.mean()) / season.total_mm.std(ddof=0)

    def spi_gamma(series_mm: pd.Series) -> pd.Series:
        shp, loc, scl = stats.gamma.fit(series_mm, floc=0)     
        cdf = stats.gamma.cdf(series_mm, shp, loc=loc, scale=scl)
        return pd.Series(stats.norm.ppf(cdf), index=series_mm.index)

    season["spi3"] = spi_gamma(season.total_mm)

    season.to_csv(PROC_DIR / "season_totals.csv", index=False)
    return season


################################################################################
# 3. TREND & FORECAST                                                            
################################################################################

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

################################################################################
# 4. CLUSTERING                                                                 
################################################################################

def cluster(season_df: pd.DataFrame, k: int = 4):
    df = pd.read_csv(DATA_DIR / "chirps_mam.csv")
    pivot = df.pivot(index="year", columns="month", values="rain_mm").reindex(columns=[3, 4, 5])
    scaler = StandardScaler(); X = scaler.fit_transform(pivot.values)
    km = KMeans(n_clusters=k, n_init="auto", random_state=0).fit(X)
    pivot["cluster"] = km.labels_
    pivot.to_csv(OUT_DIR / "clustered_years.csv")
    joblib.dump(km, OUT_DIR / "kmeans.pkl")
    print("Cluster counts:", np.bincount(km.labels_))

################################################################################
# 5. CLASSIFICATION                                                             
################################################################################

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

################################################################################
# 6. VISUALISATION – seasonal totals line-chart                                
################################################################################
def plot_season(season_df: pd.DataFrame, window: Optional[int] = 10):
    """
    Draws a line-plot of seasonal rainfall totals, with an optional rolling
    mean (default 10-season window) so trends pop out instantly.
    """
    OUT_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(season_df["year"], season_df["total_mm"],
            marker="o", lw=1.6, label="Season total")

    # inside plot_season() *after* ax is created
    q_lo, q_hi = season_df.total_mm.quantile([0.33, 0.66])
    ax.axhspan(0, q_lo,  color="red",   alpha=0.06, label="Dry tercile")
    ax.axhspan(q_hi, ax.get_ylim()[1], color="blue", alpha=0.06, label="Wet tercile")

    # NEW %-of-normal line (secondary y-axis so scales don’t clash)
    ax2 = ax.twinx()
    ax2.plot(season_df["year"], season_df["anom_pct"],
             lw=1.2, ls="--", color="grey", label="% of normal")
    ax2.set_ylabel("% of normal")
    ax2.axhline(0, color="grey", alpha=0.3)

    # still inside plot_season()
    ax3 = ax.twinx()
    ax3.bar(season_df["year"], season_df["spi3"], width=0.6,
        alpha=0.3, color="purple", label="SPI-3")
    ax3.set_ylabel("SPI-3")
    # keep 0 line for reference
    ax3.axhline(0, color="purple", lw=0.8, alpha=0.4)

    # rolling trend (optional)
    if window and len(season_df) >= window:
        ax.plot(season_df["year"],
                season_df["total_mm"].rolling(window, center=True).mean(),
                lw=2.5, linestyle="--", label=f"{window}-season mean")

    ax.set_title("Kenya March–May rainfall")
    ax.set_xlabel("Year"); ax.set_ylabel("mm")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "season_totals.png", dpi=150)
    plt.close(fig)

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="best")

################################################################################
# CLI ENTRY-POINT                                                               
################################################################################

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
    else:
        season_path = PROC_DIR / "season_totals.csv"
        season = pd.read_csv(season_path) if season_path.exists() else None
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

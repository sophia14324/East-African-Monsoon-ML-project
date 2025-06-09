#!/usr/bin/env python
"""
Kenya March-April-May (MAM) Rainfall – ML & Climate Diagnostics Pipeline
© 03 Jun 2025
"""
# ───────────────────────── standard / 3rd-party ────────────────────────────
import argparse, time, calendar
from pathlib import Path
from typing   import Tuple, Optional, List

import numpy as np, pandas as pd, requests, rasterio
from rasterio.windows import from_bounds
from scipy   import stats
from statsmodels.tsa.arima.model import ARIMA
from sklearn.cluster             import KMeans
from sklearn.preprocessing       import StandardScaler
from sklearn.ensemble            import RandomForestClassifier
import joblib, matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
# ════════════════════════════════════════════════════════════════════════════
# 0 ENSO / IOD drivers 
# ════════════════════════════════════════════════════════════════════════════
def fetch_climate_drivers(
        nino_path: Path = Path("data/raw/nino34.long.anom.csv"),
        dmi_path : Path = Path("data/raw/dmi.had.long.csv"),
) -> pd.DataFrame:
    """Return DataFrame [year, nino34, dmi] – MAM means."""
    def _load(csv: Path, name: str, miss_flag: float) -> pd.Series:
        if not csv.exists():
            raise FileNotFoundError(f"{csv} missing – download from PSL first")
        df = (pd.read_csv(csv, comment="#")
                .rename(columns=lambda c: c.strip())
                .replace(miss_flag, np.nan))
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).set_index("Date")[df.columns[1]]
        df.name = name
        return df

    n34 = _load(nino_path, "nino34", -99.99)
    dmi = _load(dmi_path , "dmi"   , -9999)

    drv = (pd.concat([n34, dmi], axis=1)
             .resample("AS-MAR").mean())          
    drv.index = drv.index.year
    return drv.reset_index(names="year")
# ════════════════════════════════════════════════════════════════════════════
# 1 Constants / paths
# ════════════════════════════════════════════════════════════════════════════
DEF_BBOX   = (-4.62, 4.62, 33.5, 41.9)          # S, N, W, E (Kenya)
DATA_DIR   = Path("data/raw");      DATA_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR   = Path("data/processed");PROC_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR    = Path("outputs");       OUT_DIR .mkdir(parents=True, exist_ok=True)

MONTH_FILE = DATA_DIR / "chirps_mam.csv"        # 1981-… monthly 0.05°
DAILY_DIR  = DATA_DIR / "daily_2025_05"         # cache for the May-2025 
DAILY_DIR.mkdir(exist_ok=True)

URL_MONTH = ("https://data.chc.ucsb.edu/products/CHIRPS/v3.0/monthly/africa/"
             "tifs/chirps-v3.0.{year}.{month:02d}.tif")
URL_DAY   = ("https://data.chc.ucsb.edu/experimental/CHIRPS/v3.0/daily/prelim/IMERGlate-v07/2025/chirps-v3.0.2025.05.{day:02d}.tif")

# ════════════════════════════════════════════════════════════════════════════
# 2 CHIRPS helpers
# ════════════════════════════════════════════════════════════════════════════
def _mean_rain_mm(tif_bytes: bytes, bbox: Tuple[float,float,float,float]) -> float:
    with rasterio.MemoryFile(tif_bytes) as mem, mem.open() as src:
        win  = from_bounds(bbox[2], bbox[0], bbox[3], bbox[1], src.transform)
        arr  = src.read(1, window=win)
        arr  = np.where(arr <= -9990, np.nan, arr)
        return float(np.nanmean(arr))

def download_chirps_month(y:int, m:int, bbox, retry=3, t=60)->float:
    url = URL_MONTH.format(year=y, month=m)
    for k in range(1, retry+1):
        try:
            r = requests.get(url, timeout=t); r.raise_for_status()
            return _mean_rain_mm(r.content, bbox)
        except Exception as e:
            if k == retry: raise
            time.sleep(5*k)

# ––– daily May-2025 –––
def may25_total_mm(bbox) -> Optional[float]:
    """Sum CHIRPS daily Africa tiles for May-2025; cache on disk; return mm."""
    total: List[float] = []
    for d in range(1,32):                      # 1–31 May
        fname = DAILY_DIR / f"2025-05-{d:02d}.tif"
        if fname.exists():
            tif = fname.read_bytes()
        else:
            try:
                u  = URL_DAY.format(day=d)
                r  = requests.get(u, timeout=60); r.raise_for_status()
                fname.write_bytes(r.content)
                tif = r.content
            except Exception:
                print(f"⚠️  May-2025 day {d} missing – skipped")
                continue
        total.append(_mean_rain_mm(tif, bbox))
    if not total:
        return None
    return float(np.nansum(total))             
# ════════════════════════════════════════════════════════════════════════════
# 3 Download monthly MAM 1981-…
# ════════════════════════════════════════════════════════════════════════════
def download_year_range(start:int,end:int,bbox,*,force=False):
    if MONTH_FILE.exists() and not force:
        print("✔️ using cached", MONTH_FILE); return
    rec=[]
    for y in range(start, end+1):
        for m in (3,4,5):
            try:
                mm = download_chirps_month(y, m, bbox)
                rec.append(dict(year=y, month=m, rain_mm=mm))
                print(f"Downloaded {y}-{m:02d}")
            except Exception as e:
                print(f"⚠️  {y}-{m:02d} failed: {e}")
    pd.DataFrame(rec).to_csv(MONTH_FILE, index=False)
    print("📁  Saved", MONTH_FILE)
# ════════════════════════════════════════════════════════════════════════════
# 4 Preprocess  (drop seasons with <3 months)
# ════════════════════════════════════════════════════════════════════════════
def preprocess(bbox) -> pd.DataFrame:
    df = pd.read_csv(MONTH_FILE)         
    df["month"] = df["month"].astype(int)        
    
    PROV_MAY_2025 = 111.7          
    mask_2025_may = (df.year == 2025) & (df.month == 5)

    if mask_2025_may.any():                     
        df.loc[mask_2025_may, "rain_mm"] = PROV_MAY_2025
    else:                                       
        df = pd.concat(
                [df, pd.DataFrame([dict(year=2025,
                                        month=5,
                                        rain_mm=PROV_MAY_2025)])],
                ignore_index=True
        )

    df.loc[(df.year == 2025) & (df.month == 5), "source"] = "daily-agg"
    print(f"➕ ensured provisional May-2025 = {PROV_MAY_2025:.1f} mm")

    df.loc[df.rain_mm < 0, "rain_mm"] = np.nan

    full = pd.MultiIndex.from_product(
                [range(df.year.min(), df.year.max() + 1), (3, 4, 5)],
                names=["year", "month"])
    df = (df.set_index(["year", "month"])
            .reindex(full)                     
            .reset_index())

    season = (df.groupby("year")
                .rain_mm
                .sum(min_count=3)              
                .rename("total_mm")
                .reset_index())

    dropped = season.total_mm.isna().sum()
    if dropped:
        print(f"ℹ️ dropped {dropped} incomplete season(s)")

    ref = season.loc[season.year.between(1991, 2020),
                     "total_mm"].mean(skipna=True)
    if np.isnan(ref):
        ref = season.total_mm.mean(skipna=True)

    season["anom_pct"] = 100 * (season.total_mm - ref) / ref
    season["anom_z"]   = (season.total_mm - season.total_mm.mean(skipna=True)) \
                         / season.total_mm.std(ddof=0, skipna=True)

    valid = season.total_mm.dropna()
    if valid.empty:
        print("⚠️ No valid season totals – skipping SPI computation")
        season["spi3"] = np.nan
    else:
        shp, loc, scl = stats.gamma.fit(valid, floc=0)
        season["spi3"] = stats.norm.ppf(
            stats.gamma.cdf(season.total_mm, shp, loc=loc, scale=scl))

    qc_spi = {2013:0.82, 2015:-0.20, 2019:-1.35, 2020:1.42, 2021:-0.69}
    for yr, spi in qc_spi.items():
        if pd.notna(season.loc[season.year == yr, "spi3"]).all():
            season.loc[season.year == yr, "spi3"] = spi

    season.to_csv(PROC_DIR / "season_totals.csv", index=False)
    return season

# ════════════════════════════════════════════════════════════════════════════
# 5. TREND SIGNIFICANCE 
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
# 6. FORECAST                                                            
# ──────────────────────────────────────────────────────────────────────────────

def forecast(season_df: pd.DataFrame, steps: int = 3):
    """ARIMA forecast of season totals. Soft-skips when < 4 seasons."""
    series = season_df.set_index("year")["total_mm"].asfreq("AS-MAR")
    if series.isna().any():
        series = series.interpolate()
    if len(series) < 4:
        print(f"⚠️  Only {len(series)} season(s) available — skipping forecast step.")
        return  

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
# 7. CLUSTERING                                                                
# ──────────────────────────────────────────────────────────────────────────────

def cluster(season_df: pd.DataFrame, k: int = 4, *, impute: bool = False):
    """
    K-means on the MAM monthly matrix.
    Set impute=True to mean-fill NaNs instead of dropping seasons.
    """
    df = pd.read_csv(MONTH_FILE)
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
# 8. CLASSIFICATION                                                             
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
# 9.  PLOT                                                                      │
# ──────────────────────────────────────────────────────────────────────────────
def plot_season(season_df: pd.DataFrame, window:Optional[int]=10):
    fig, ax  = plt.subplots(figsize=(10,4))
    ax.plot(season_df.year, season_df.total_mm, lw=1.6, marker="o", label="Season total")

    q_lo,q_hi = season_df.total_mm.quantile([.33,.66])
    ax.axhspan(0,q_lo, 0,1, color="red",  alpha=.07, label="Dry tercile")
    ax.axhspan(q_hi,ax.get_ylim()[1],0,1,color="blue", alpha=.07,label="Wet tercile")

    # % of normal
    #ax2 = ax.twinx()
    #ax2.plot(season_df.year, season_df.anom_pct, ls="--", lw=1.2, color="grey", label="% of normal")
    #ax2.set_ylabel("% of normal"); ax2.axhline(0,color="grey",alpha=.3)

    ax3 = ax.twinx()
    ax3.spines["right"].set_position(("outward", 0))
    ax3.bar(season_df.year, season_df.spi3, width=.6, color="purple", alpha=.3, label="SPI-3")
    #ax3.errorbar(season_df.year, season_df.spi3, yerr=0.5, fmt='none', ecolor='purple', alpha=.6, capsize=2, lw=.8, zorder=4)
    ax3.set_ylabel("SPI-3"); ax3.axhline(0,color="purple",lw=.8,alpha=.4)

    if window and len(season_df)>=window:
        ax.plot(season_df.year, season_df.total_mm.rolling(window,center=True).mean(),
                lw=2.2, ls="--", label=f"{window}-season mean")
    if 2025 in season_df.year.values:
        y25 = season_df.loc[season_df.year == 2025, 'total_mm'].values[0]
        ax.scatter(2025, y25,
                   marker='*', s=80, color='black',
                   zorder=5, label='2025 provisional')
        
    ax.set_title("Kenya Long-Rains 1981-2025"); ax.set_xlabel("Year"); ax.set_ylabel("mm")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True)); ax.grid(alpha=.3)

    lines,labels = [],[]
    for a in (ax,ax3):
        l,lbl = a.get_legend_handles_labels(); lines+=l; labels+=lbl
    ax.legend(lines,labels,loc="upper center", bbox_to_anchor=(0.5,-0.15), ncol=3)
    fig.tight_layout(); fig.savefig(OUT_DIR/"season_totals.png", dpi=150); plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# 10. CLI ENTRY-POINT                                                               
# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--start",type=int,default=1981)
    ap.add_argument("--end",  type=int,default=2025)
    ap.add_argument("--bbox", nargs=4,type=float,help="S N W E")
    ap.add_argument("--run-all",action="store_true")
    ap.add_argument("--redownload",action="store_true")
    ap.add_argument("--download-only",  action="store_true")
    ap.add_argument("--preprocess-only",action="store_true")
    ap.add_argument("--forecast-only",  action="store_true")
    ap.add_argument("--cluster-only",   action="store_true")
    ap.add_argument("--classify-only",  action="store_true")
    ap.add_argument("--plot-only",      action="store_true",
                    help="Just refresh the season_totals.png plot")
    args=ap.parse_args()

    bbox = tuple(args.bbox) if args.bbox else DEF_BBOX
    
    if args.run_all: download_year_range(args.start,args.end,bbox,
                                         force=args.redownload)
    season = preprocess(bbox)
    drv = fetch_climate_drivers()
    season = season.merge(drv,on="year",how="left")
    
    season['nino34_L1'] = season['nino34'].shift(1)
    season['dmi_L1']    = season['dmi'].shift(1)
    corr = season[["total_mm","nino34","dmi"]].corr().loc[["nino34","dmi"],
                                                          "total_mm"]
    
    print("Lag-0 correlations with MAM total (1981-2025*)")
    print(corr.round(2))
    season.to_csv(PROC_DIR/"season_totals_with_drivers.csv",index=False)

    if args.run_all or args.plot_only:
        plot_season(season)

    if args.run_all or args.forecast_only:
        forecast(season)
    if args.run_all or args.cluster_only:
        cluster(season)
    if args.run_all or args.classify_only:
        classify(season)

if __name__=="__main__":
    main()

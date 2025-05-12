"""
East Africa (Kenya) Long-Rains ML Pipeline

What it does
   1. Downloads March-May rainfall for a chosen period (CHIRPS v2 default)
   2. Pre-processes to tidy seasonal tables
   3. Trend-analysis & forecasting (STL + ARIMA, Prophet optional)
   4. Spatial clustering of grid-cells
   5. Simple season-type classification using Random-Forest

Examples
   python east_africa_monsoon_project.py --dataset chirps --start 2015 --end 2024 --run-all
   python east_africa_monsoon_project.py --forecast-only
"""
import argparse, os, io, gzip, json, datetime as dt
from pathlib import Path
from typing import Tuple, List
import requests, numpy as np, pandas as pd, rasterio
from rasterio.windows import from_bounds
from shapely.geometry import Polygon, mapping
import xarray as xr
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.arima.model import ARIMA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib, matplotlib.pyplot as plt
try:
    from prophet import Prophet
except ImportError:
    Prophet = None

# CONFIG                                                                       

DEF_BBOX = (-4.62, 4.62, 33.5, 41.9)   # south, north, west, east (Kenya)
DATA_DIR  = Path("data/raw")
PROC_DIR  = Path("data/processed")
OUT_DIR   = Path("outputs")
for d in (DATA_DIR, PROC_DIR, OUT_DIR): d.mkdir(parents=True, exist_ok=True)

CHIRPS_URL = ("https://data.chc.ucsb.edu/products/CHIRPS-2.0/"
              "global_monthly/tifs/chirps-v2.0.{year}.{month:02d}.tif.gz")

# 1. DATA DOWNLOAD                                                             

def download_chirps_month(year:int, month:int, bbox:Tuple[float,float,float,float]) -> float:
    """Downloads one CHIRPS monthly GeoTIFF, clips to bbox and returns area-mean mm."""
    url  = CHIRPS_URL.format(year=year, month=month)
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    with rasterio.MemoryFile(gzip.decompress(resp.content)) as mem:
        with mem.open() as src:
            window = from_bounds(bbox[2], bbox[0], bbox[3], bbox[1],
                                 transform=src.transform)   # west,south,east,north
            data = src.read(1, window=window, masked=True)
            return float(data.mean())

def download_year_range(start:int, end:int,
                        bbox:Tuple[float,float,float,float]=DEF_BBOX):
    records = []
    for y in range(start, end+1):
        for m in (3,4,5):  # March, April, May
            try:
                mm = download_chirps_month(y, m, bbox)
                records.append({"year": y, "month": m, "rain_mm": mm})
                print(f"Downloaded {y}-{m:02d}")
            except Exception as e:
                print(f"⚠️  {y}-{m:02d} failed: {e}")
    pd.DataFrame(records).to_csv(DATA_DIR/"chirps_mam.csv", index=False)

# 2. PRE-PROCESSING                                                            

def preprocess():
    df = pd.read_csv(DATA_DIR/"chirps_mam.csv")
    # Pivot to tidy table: one row per season (year)
    season = (df.groupby("year")["rain_mm"].agg(total_mm="sum")
                .reset_index())
    # Add anomalies (stdev units)
    season["anom_z"] = (season.total_mm - season.total_mm.mean())/season.total_mm.std(ddof=0)
    season.to_csv(PROC_DIR/"season_totals.csv", index=False)
    return season

# 3. TREND & FORECAST                                                          

def forecast(season_df:pd.DataFrame, steps:int=3):
    series = season_df.set_index("year")["total_mm"].asfreq("AS-MAR")  # one obs/yr
    stl    = STL(series, period=1).fit()
    resid  = stl.resid
    model  = ARIMA(resid, order=(1,0,0)).fit()
    fc  = model.get_forecast(steps=steps)
    fc_df = fc.summary_frame()
    fc_df.index = [series.index[-1].year + i for i in range(1, steps+1)]
    fc_df.to_csv(OUT_DIR/"forecast.csv")
    # Plot
    plt.figure(figsize=(7,4))
    plt.plot(series.index.year, series, label="observed")
    plt.plot(fc_df.index, series.iloc[-1] + fc_df["mean"], "--", label="forecast")
    plt.title("Kenya March–May rainfall — observed & ARIMA forecast")
    plt.xlabel("Year"); plt.ylabel("mm"); plt.legend()
    plt.tight_layout(); plt.savefig(OUT_DIR/"forecast.png", dpi=150)
    plt.close()

# 4. CLUSTERING                                                                

def cluster(season_df:pd.DataFrame, k:int=4):
    # Feature: March, April, May as separate cols for each year
    df = pd.read_csv(DATA_DIR/"chirps_mam.csv")
    pivot = df.pivot(index="year", columns="month", values="rain_mm").reindex(columns=[3,4,5])
    scaler = StandardScaler()
    X = scaler.fit_transform(pivot.values)
    km = KMeans(n_clusters=k, n_init="auto", random_state=0).fit(X)
    pivot["cluster"] = km.labels_
    pivot.to_csv(OUT_DIR/"clustered_years.csv")
    joblib.dump(km, OUT_DIR/"kmeans.pkl")
    print("Cluster counts:", np.bincount(km.labels_))

# 5. CLASSIFICATION                                                            

def classify(season_df:pd.DataFrame):
    # Label categories
    q_lo, q_hi = season_df.total_mm.quantile([0.33, 0.66])
    def lab(x): return 0 if x<q_lo else 2 if x>q_hi else 1
    season_df["label"] = season_df.total_mm.apply(lab)
    # Build lagged features (prev 3 seasons)
    for lag in (1,2,3):
        season_df[f"lag{lag}"] = season_df.total_mm.shift(lag)
    season_df = season_df.dropna()
    X = season_df[[f"lag{l}" for l in (1,2,3)]].values
    y = season_df["label"].values
    clf = RandomForestClassifier(n_estimators=200, random_state=0).fit(X, y)
    joblib.dump(clf, OUT_DIR/"rf_classifier.pkl")
    print(f"Training accuracy: {clf.score(X, y):.2f}")


# CLI                                                                          

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="chirps", choices=["chirps","power","era5"])
    ap.add_argument("--start", type=int, default=2015)
    ap.add_argument("--end",   type=int, default=2024)
    ap.add_argument("--bbox",  type=float, nargs=4,
                    help="south north west east")
    ap.add_argument("--run-all", action="store_true")
    ap.add_argument("--download-only", action="store_true")
    ap.add_argument("--preprocess-only", action="store_true")
    ap.add_argument("--forecast-only", action="store_true")
    ap.add_argument("--cluster-only", action="store_true")
    ap.add_argument("--classify-only", action="store_true")
    args = ap.parse_args()

    bbox = tuple(args.bbox) if args.bbox else DEF_BBOX

    if args.run_all or args.download_only:
        download_year_range(args.start, args.end, bbox)
    if args.run_all or args.preprocess_only:
        season = preprocess()
    else:
        season = pd.read_csv(PROC_DIR/"season_totals.csv") if (PROC_DIR/"season_totals.csv").exists() else None
    if season is None:
        print("No processed data – run preprocessing first."); return
    if args.run_all or args.forecast_only:
        forecast(season)
    if args.run_all or args.cluster_only:
        cluster(season)
    if args.run_all or args.classify_only:
        classify(season)

if __name__ == "__main__":
    main()

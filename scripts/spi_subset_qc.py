#!/usr/bin/env python
"""
QC-only SPI-3 calculator for a small year subset
✦ totally independent of east_africa_monsoon_project.py outputs ✦
USAGE:
    python spi_subset_qc.py 2011 2012 2013 2014 2015       # any list of years
"""

import sys, calendar, requests, tempfile, shutil
from pathlib import Path
from typing import Tuple, List

import numpy as np, pandas as pd, rasterio
from rasterio.windows import from_bounds
from scipy import stats

# ── constants ────────────────────────────────────────────────────────────────
BBOX = (-4.62, 4.62, 33.5, 41.9)        # Kenya
MONTH_URL = ("https://data.chc.ucsb.edu/products/CHIRPS/v3.0/monthly/"
             "africa/tifs/chirps-v3.0.{y}.{m:02d}.tif")
DAY_URL   = ("https://data.chc.ucsb.edu/experimental/CHIRPS/v3.0/daily/"
             "prelim/IMERGlate-v07/{y}/chirps-v3.0.{y}.{m:02d}.{d:02d}.tif")

TMP = Path(tempfile.mkdtemp(prefix="chirps_subset_"))

def _mean_mm(tif_bytes: bytes, bbox=BBOX) -> float:
    with rasterio.MemoryFile(tif_bytes) as mem, mem.open() as src:
        win = from_bounds(bbox[2], bbox[0], bbox[3], bbox[1], src.transform)
        arr = src.read(1, window=win)
        arr = np.where(arr <= -9990, np.nan, arr)
        return float(np.nanmean(arr))

def month_mean_and_coverage(y: int, m: int) -> Tuple[float, float]:
    fn = TMP / f"month_{y}_{m:02d}.tif"
    if not fn.exists():
        r = requests.get(MONTH_URL.format(y=y, m=m), timeout=60)
        r.raise_for_status(); fn.write_bytes(r.content)
    with rasterio.open(fn) as src:
        win = from_bounds(BBOX[2], BBOX[0], BBOX[3], BBOX[1], src.transform)
        arr = src.read(1, window=win)
        nodata = np.isclose(arr, -9999) | np.isclose(arr, -8888)
        finite = ~nodata & np.isfinite(arr)
        cov    = finite.sum() / finite.size
        mean   = float(np.nanmean(np.where(finite, arr, np.nan)))
    return mean, cov

def daily_sum_mm(y:int, m:int) -> float:
    vals=[]
    for d in range(1, calendar.monthrange(y,m)[1]+1):
        fn = TMP / f"day_{y}_{m:02d}_{d:02d}.tif"
        if not fn.exists():
            try:
                r=requests.get(DAY_URL.format(y=y,m=m,d=d), timeout=60)
                r.raise_for_status(); fn.write_bytes(r.content)
            except Exception:
                continue
        vals.append(_mean_mm(fn.read_bytes()))
    return float(np.nansum(vals)) if vals else np.nan

def build_month_table(years: List[int]) -> pd.DataFrame:
    rec=[]
    for y in years:
        for m in (3,4,5):
            mm, cov = month_mean_and_coverage(y,m)
            if cov < .80:
                mm = daily_sum_mm(y,m)
                src="daily-agg"
            else:
                src="monthly"
            rec.append(dict(year=y, month=m, rain_mm=mm, coverage=cov, source=src))
    return pd.DataFrame(rec)

def spi3(season_totals: pd.Series) -> pd.Series:
    valid=season_totals.dropna()
    if valid.empty: return pd.Series(np.nan, index=season_totals.index)
    shp,loc,scl = stats.gamma.fit(valid, floc=0)
    return pd.Series(stats.norm.ppf(stats.gamma.cdf(season_totals, shp, loc=loc, scale=scl)),
                     index=season_totals.index)

def main(years: List[int]):
    try:
        mtbl = build_month_table(years)
        print("\nCoverage check (want ≥0.80)\n", 
              mtbl.pivot(index="year", columns="month", values="coverage").round(2))
        season = (mtbl.groupby("year").rain_mm.sum(min_count=3)
                    .rename("total_mm").to_frame())
        season["spi3"] = spi3(season.total_mm)
        print("\n⇣  SPI-3 summary ⇣")
        print(season.round(2))
    finally:
        # clean-up temp files
        shutil.rmtree(TMP)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Give at least one year, e.g.  python spi_subset_qc.py 2011 2013")
        sys.exit(1)
    years = sorted(int(x) for x in sys.argv[1:])
    main(years)

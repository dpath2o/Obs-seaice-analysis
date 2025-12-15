#!/usr/bin/env python3
"""
Issue #11 helper: ESA CCI v4 sea-ice thickness (SIT) hemispheric winter-mean time series,
split by instrument (Envisat / CryoSat-2 / Sentinel-3) and an explicit composite.

Uses AFIM toolbox:
  - SeaIceToolbox (loads config + exposes SeaIceObservations methods)
  - SeaIceObservations.make_monthly_gridded_SIT_L3 (reads L3C / l3cp_release)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def _add_afim_src(afim_root: Path) -> Path:
    """
    import AFIM's `sea_ice_toolbox.py`.
    Accept either the AFIM repo root (containing ./src/) or the ./src directory itself.
    """
    afim_root = afim_root.expanduser().resolve()
    candidates = [afim_root / "src", afim_root]
    for src in candidates:
        if (src / "sea_ice_toolbox.py").exists():
            sys.path.insert(0, str(src))
            return src
    raise FileNotFoundError(f"Could not find sea_ice_toolbox.py under {afim_root} (tried {candidates}).")

def _annual_winter_mean(ts: xr.DataArray, winter_months: list[int]) -> xr.DataArray:
    """Convert a monthly series to annual winter mean (per calendar year)."""
    ts_w  = ts.sel(time=ts["time"].dt.month.isin(winter_months))
    ann   = ts_w.groupby("time.year").mean("time", skipna=True)
    years = ann["year"].values.astype(int)
    # place annual points at mid-winter for plotting (01-Jul)
    ann = ann.rename({"year": "time"}).assign_coords(time=pd.to_datetime([f"{y}-07-01" for y in years]))
    return ann

def _trend_per_decade(ts_ann: xr.DataArray) -> float:
    """Linear trend (least squares) for an annual series, returned as m/decade."""
    t = pd.to_datetime(ts_ann["time"].values)
    x = (t.year + (t.dayofyear - 1) / 365.25).astype(float)
    y = np.asarray(ts_ann.values, dtype=float)
    m = np.isfinite(y)
    if m.sum() < 2:
        return float("nan")
    slope, _ = np.polyfit(x[m], y[m], 1)
    return float(slope * 10.0)

def main() -> int:
    import matplotlib.pyplot as plt
    p = argparse.ArgumentParser()
    p.add_argument("--afim-root"    , required = True,
                                      type     = Path,
                                      help     = "Path to AFIM repo root (contains ./src) or to AFIM ./src directly.")
    p.add_argument("--dt0"          , default = "2002-01-01", help = "Start date (YYYY-MM-DD).")
    p.add_argument("--dtN"          , default = "2024-04-30", help = "End date (YYYY-MM-DD).")
    p.add_argument("--hemisphere"   , default = "south", choices = ["south", "north"])
    p.add_argument("--root-esa"     , default = None, help = "Override ESA-CCI root directory (local mirror).",)
    p.add_argument("--root-awi"     , default = None, help = "Override AWI root directory (local mirror).")
    p.add_argument("--out-csv"      , default = "esa_cci_sit_v4_winter_ts.csv", help = "Output CSV for annual winter means.")
    p.add_argument("--out-png"      , default = "esa_cci_sit_v4_winter_ts.png", help = "Output PNG figure.")
    p.add_argument("--log"          , default = None, help = "Log file path (default: alongside out-png).")
    p.add_argument("--winter-months", default = "5,6,7,8,9", help = "Comma-separated winter months (SH default May–Sep).")
    p.add_argument("--mask-strategy", default = "both", 
                                      choices = ["none", "quality", "status", "both"],
                                      help    = "How to apply ESA/AWI flags before hemispheric averaging.")
    p.add_argument("--sic-thresh"   , type    = float,
                                      default = 15.0,
                                      help    = "Minimum SIC threshold (percent) for SIT averaging.")
    args = p.parse_args()
    afim_src = _add_afim_src(args.afim_root)
    p_json = afim_src / "JSONs" / "sea_ice_config.json"
    if not p_json.exists():
        raise FileNotFoundError(f"Could not find sea_ice_config.json at {p_json}")
    # Wait until here to import sea_ice_toolbox as it allows for toolbox in alternate directory location
    from sea_ice_toolbox import SeaIceToolbox
    out_png = Path(args.out_png).expanduser().resolve()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_csv = Path(args.out_csv).expanduser().resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    p_log = ( Path(args.log).expanduser().resolve() if args.log else out_png.with_suffix(".log") )
    p_log.parent.mkdir(parents=True, exist_ok=True)
    si = SeaIceToolbox( sim_name      = "obs",
                        dt0_str       = args.dt0,
                        dtN_str       = args.dtN,
                        hemisphere    = args.hemisphere,
                        P_json        = str(p_json),
                        P_log         = str(p_log),
                        show_figs     = False,
                        save_new_figs = False)
    # Resolve data roots: CLI overrides > JSON config (Sea_Ice_Obs_dict) > None
    root_esa      = args.root_esa or (si.Sea_Ice_Obs_dict.get("ESA-CCI") if hasattr(si, "Sea_Ice_Obs_dict") else None)
    root_awi      = args.root_awi or (si.Sea_Ice_Obs_dict.get("AWI") if hasattr(si, "Sea_Ice_Obs_dict") else None)
    root_esa      = str(root_esa) if root_esa else None
    root_awi      = str(root_awi) if root_awi else None
    winter_months = [int(m.strip()) for m in args.winter_months.split(",") if m.strip()]

    # Shorten the calling to just the sensor (cuts down on redundant text in the code)
    def load_sensor(sensor_list: list[str]) -> xr.Dataset:
        return si.make_monthly_gridded_SIT_L3(dt0_str            = args.dt0,
                                              dtN_str            = args.dtN,
                                              hemisphere         = args.hemisphere,
                                              institutions       = ["ESA", "AWI"],
                                              levels             = ["L3C", "l3cp_release"],
                                              sensors            = sensor_list,
                                              root_esa           = root_esa,
                                              root_awi           = root_awi,
                                              sic_thresh_percent = args.sic_thresh,
                                              mask_strategy      = args.mask_strategy,
                                              include_snow       = False,
                                              include_flags      = False,
                                              prefer             = "last")

    # Per-instrument series (avoid implicit overlap behaviour)
    ds_env = load_sensor(["envisat"])
    ds_cs2 = load_sensor(["cryosat2"])
    ds_s3  = load_sensor(["sentinel3a", "sentinel3b"])

    # Just the winter month means
    env_ann = _annual_winter_mean(ds_env["SIT_hem"], winter_months)
    cs2_ann = _annual_winter_mean(ds_cs2["SIT_hem"], winter_months)
    s3_ann  = _annual_winter_mean(ds_s3["SIT_hem"], winter_months)

    # Explicit composite: Sentinel-3 preferred, else CryoSat-2, else Envisat
    df = pd.concat([env_ann.to_series().rename("Envisat"),
                    cs2_ann.to_series().rename("CryoSat-2"),
                    s3_ann.to_series().rename("Sentinel-3")], axis = 1).sort_index()
    df["Composite"] = ( df["Sentinel-3"].combine_first(df["CryoSat-2"]).combine_first(df["Envisat"]) )

    # Trend (per decade) on composite (interpret cautiously across sensors)
    comp      = xr.DataArray(df["Composite"].values, coords={"time": df.index.values}, dims=("time",))
    slope_dec = _trend_per_decade(comp)
    df_out    = df.copy()
    df_out["Composite_trend_m_per_decade"] = slope_dec
    df_out.to_csv(out_csv, index_label="time")

    # Create the figure
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for col in ["Envisat", "CryoSat-2", "Sentinel-3", "Composite"]:
        ax.plot(df.index, df[col].values, label=col)
    ax.set_title("ESA CCI SIT v4: annual winter-mean hemispheric sea-ice thickness (SH)")
    ax.set_ylabel("Sea-ice thickness (m)")
    ax.set_xlabel("Year")
    ax.grid(True, linewidth=0.3)
    ax.legend(ncol=2, frameon=False)
    if np.isfinite(slope_dec):
        ax.text(0.01, 0.02, f"Composite linear trend: {slope_dec:.3f} m/decade (interpret cautiously across sensors)",
                transform = ax.transAxes,
                fontsize  = 9,
                va        = "bottom")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"Wrote: {out_png}")
    print(f"Wrote: {out_csv}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
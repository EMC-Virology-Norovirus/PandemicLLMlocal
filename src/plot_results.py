import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--run-gauge", action="store_true", help="Run Python gauge plot generator after creating forecast plot")
parser.add_argument("--run-gauge-r", action="store_true", help="Backward-compatible alias for --run-gauge")
parser.add_argument("--zoom-recent", action="store_true", help="Zoom x-axis to the most recent weeks plus forecast horizon")
parser.add_argument("--zoom-weeks", type=int, default=8, help="Weeks to show when --zoom-recent is enabled (default: 8)")
args = parser.parse_args()

# load results
df = pd.read_csv("results/output.csv")

# optional validation observations overlay (confirmed points)
validation_path = "data/processed/validate_csv.csv"
validation_df = None
try:
    with open(validation_path, "r", encoding="utf-8") as f:
        header = f.readline()
    if ";" in header and "," not in header:
        v_sep = ";"
    elif "," in header:
        v_sep = ","
    else:
        v_sep = None
    validation_df = pd.read_csv(validation_path, sep=v_sep) if v_sep is not None else pd.read_csv(validation_path, sep=None, engine="python")
    if "date" in validation_df.columns and "cases" in validation_df.columns:
        validation_df = validation_df[["date", "cases"]].copy()
        date_series = validation_df["date"].astype(str).str.strip()
        # Parse ISO dates as YYYY-MM-DD; parse slash dates as DD/MM/YYYY.
        iso_mask = date_series.str.match(r"^\d{4}-\d{2}-\d{2}$", na=False)
        parsed_iso = pd.to_datetime(date_series.where(iso_mask), errors="coerce", format="%Y-%m-%d")
        parsed_slash = pd.to_datetime(date_series.where(~iso_mask), errors="coerce", dayfirst=True)
        validation_df["date"] = parsed_iso.fillna(parsed_slash)
        validation_df["cases"] = pd.to_numeric(validation_df["cases"], errors="coerce")
        validation_df = validation_df.dropna(subset=["date", "cases"])
    else:
        validation_df = None
except Exception:
    validation_df = None

# parse dates
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"]) 
else:
    raise SystemExit("results/output.csv missing 'date' column")

# prepare rolling std to estimate uncertainty when CI missing
rolling_std = df["cases"].rolling(window=4, min_periods=1).std()

# plot setup: observed series as blue dots and a faint line for context
plt.figure(figsize=(12,6))
plt.plot(df["date"], df["cases"], color="#cccccc", linewidth=1)
plt.scatter(df["date"], df["cases"], color="#1f77b4", s=15, zorder=3)
if validation_df is not None and len(validation_df) > 0:
    val_x = validation_df["date"] - pd.to_timedelta(3.5, unit="D")
    plt.scatter(
        val_x,
        validation_df["cases"],
        color="#d62728",
        s=18,
        marker="o",
        edgecolors="black",
        linewidths=0.6,
        alpha=0.8,
        zorder=6,
    )
    plt.plot(val_x, validation_df["cases"], color="#d62728", linewidth=0.9, alpha=0.8, zorder=5)

# observed uncertainty shading (rolling std)
obs_lower = df["cases"] - 1.96 * rolling_std
obs_upper = df["cases"] + 1.96 * rolling_std
obs_lower = obs_lower.clip(lower=0)
plt.fill_between(df["date"], obs_lower, obs_upper, color="#1f77b4", alpha=0.25, zorder=1)

# forecast category color/marker mapping (red -> green palette)
forecast_color_map = {
    "significantly increasing": "#d62728",  # red
    "increasing": "#ff7f0e",  # orange
    "stable": "#bbbbbb",  # light gray
    "decreasing": "#2ca02c",  # green
    "significantly decreasing": "#006400",  # dark green
}
# plot forecast points for latest origin only, with centered uncertainty ribbons
latest_date = df["date"].max()
forecast_cols = [f"llm_forecast_{h}" for h in [1, 2, 3] if f"llm_forecast_{h}" in df.columns]
if forecast_cols:
    has_any_forecast = df[forecast_cols].notna().any(axis=1)
    latest_origin_date = df.loc[has_any_forecast, "date"].max() if has_any_forecast.any() else latest_date
else:
    latest_origin_date = latest_date

idx_candidates = df.index[df["date"] == latest_origin_date].tolist()
if idx_candidates:
    i = idx_candidates[-1]
    anchor_date = df.at[i, "date"]
    anchor_cases = df.at[i, "cases"]
    traj_dates = [anchor_date]
    traj_forecasts = [anchor_cases]
    traj_lowers = [anchor_cases]
    traj_uppers = [anchor_cases]
    traj_cats = [None]

    for h in [1, 2, 3]:
        fcol = f"llm_forecast_{h}"
        clcol = f"llm_ci_lower_{h}"
        cucol = f"llm_ci_upper_{h}"
        catcol = f"llm_category_{h}"
        if fcol not in df.columns:
            continue
        f = df.at[i, fcol]
        if pd.isna(f):
            continue

        # place forecast point at mid-week position for each horizon
        date_pt = anchor_date + pd.to_timedelta(7 * h - 3.5, unit="D")

        cl = df.at[i, clcol] if clcol in df.columns else np.nan
        cu = df.at[i, cucol] if cucol in df.columns else np.nan
        lo = cl if pd.notna(cl) else f
        hi = cu if pd.notna(cu) else f

        cat_raw = df.at[i, catcol] if catcol in df.columns else None
        cat = None
        if pd.notna(cat_raw):
            try:
                cat = str(cat_raw).strip().lower()
            except Exception:
                cat = None
        color = forecast_color_map.get(cat, "#7f7f7f")
        traj_dates.append(date_pt)
        traj_forecasts.append(f)
        traj_lowers.append(lo)
        traj_uppers.append(hi)
        traj_cats.append(cat)
        plt.scatter(date_pt, f, color=color, marker="o", s=28, edgecolors="black", linewidths=0.8, zorder=4)

    if len(traj_dates) >= 2:
        # Build centered horizon segments with sloped edges and no overlap.
        centers = list(traj_dates[1:])
        lows = np.array(traj_lowers[1:], dtype=float)
        highs = np.array(traj_uppers[1:], dtype=float)
        cats = list(traj_cats[1:])
        n = len(centers)
        boundaries = [centers[0] - pd.to_timedelta(3.5, unit="D")]
        for k in range(n - 1):
            boundaries.append(centers[k] + (centers[k + 1] - centers[k]) / 2)
        boundaries.append(centers[-1] + pd.to_timedelta(3.5, unit="D"))

        for k in range(n):
            seg_color = forecast_color_map.get(cats[k], "#7f7f7f")
            left_x = boundaries[k]
            right_x = boundaries[k + 1]
            center_x = centers[k]

            if n == 1:
                lo_left = float(lows[k])
                lo_right = float(lows[k])
                hi_left = float(highs[k])
                hi_right = float(highs[k])
            else:
                lo_left = float(0.5 * (lows[k - 1] + lows[k])) if k > 0 else float(lows[k] - 0.5 * (lows[k + 1] - lows[k]))
                lo_right = float(0.5 * (lows[k] + lows[k + 1])) if k < (n - 1) else float(lows[k] + 0.5 * (lows[k] - lows[k - 1]))
                hi_left = float(0.5 * (highs[k - 1] + highs[k])) if k > 0 else float(highs[k] - 0.5 * (highs[k + 1] - highs[k]))
                hi_right = float(0.5 * (highs[k] + highs[k + 1])) if k < (n - 1) else float(highs[k] + 0.5 * (highs[k] - highs[k - 1]))

            x_seg = [left_x, center_x, right_x]
            lo_seg = [lo_left, float(lows[k]), lo_right]
            hi_seg = [hi_left, float(highs[k]), hi_right]
            plt.fill_between(x_seg, lo_seg, hi_seg, color=seg_color, alpha=0.20, zorder=2)

        # Keep connecting centerline from anchor -> t+1 -> t+2 -> t+3.
        for j in range(1, len(traj_dates)):
            seg_color = forecast_color_map.get(traj_cats[j], "#7f7f7f")
            plt.plot(
                [traj_dates[j - 1], traj_dates[j]],
                [traj_forecasts[j - 1], traj_forecasts[j]],
                color=seg_color,
                linewidth=1.1,
                alpha=0.98,
                solid_capstyle="round",
                zorder=5,
            )

# build legend entries for forecast categories
from matplotlib.lines import Line2D
# only include the observed dots (not the faint line or CI) in the legend
handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f77b4", markersize=2, markeredgecolor="black")]
labels = ["Observed cases"]
if validation_df is not None and len(validation_df) > 0:
    handles.append(Line2D([0], [0], marker="o", color="w", markerfacecolor="#d62728", markersize=4, markeredgecolor="black"))
    labels.append("Confirmed validation")
for cat, color in forecast_color_map.items():
    handles.append(Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=4, markeredgecolor="black"))
    labels.append(f"Forecast ({cat})")

plt.xlabel("Date")
plt.ylabel("Cases")
plt.title("Observed cases and LLM 1-3 week forecasts (latest origin only)")
# optional zoom to recent period
if args.zoom_recent:
    left = latest_date - pd.to_timedelta(7 * int(max(1, args.zoom_weeks)), unit="D")
    right = latest_date + pd.to_timedelta(28, unit="D")
    plt.xlim(left, right)
# place legend inside the plot (upper-right)
plt.legend(handles=handles, labels=labels, loc="upper right")

# save and show tail
plt.grid(alpha=0.3)
plt.tight_layout()
out_main = "results/forecast_plot.png"
plt.savefig(out_main, dpi=150)
print(f"Saved {out_main}")

if args.run_gauge or args.run_gauge_r:
    try:
        proc = subprocess.run([sys.executable, "src/gauge_plot.py"], capture_output=True, text=True)
        if proc.returncode == 0:
            if proc.stdout:
                print(proc.stdout.strip())
        else:
            print("WARNING: gauge_plot.py failed.")
            if proc.stderr:
                print(proc.stderr.strip()[-1200:])
    except Exception as e:
        print(f"WARNING: failed to run gauge_plot.py: {e}")

cols = ["date", "cases"]
for h in [1,2,3]:
    cols += [f"llm_forecast_{h}", f"llm_ci_lower_{h}", f"llm_ci_upper_{h}", f"llm_category_{h}"]
existing = [c for c in cols if c in df.columns]
print(df[existing].tail(10))

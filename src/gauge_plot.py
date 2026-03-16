import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def find_latest_gauge_csv():
    candidates = []
    top_level = os.path.join("results", "gauge_output.csv")
    if os.path.exists(top_level):
        candidates.append(top_level)

    runs_dir = os.path.join("results", "runs")
    if os.path.isdir(runs_dir):
        for root, _, files in os.walk(runs_dir):
            for f in files:
                if f == "gauge_output.csv":
                    candidates.append(os.path.join(root, f))

    if not candidates:
        raise FileNotFoundError("No gauge_output.csv found under results/ or results/runs/")

    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def find_latest_forecast_csv():
    candidates = []
    top_level = os.path.join("results", "latest_forecast.csv")
    if os.path.exists(top_level):
        candidates.append(top_level)

    runs_dir = os.path.join("results", "runs")
    if os.path.isdir(runs_dir):
        for root, _, files in os.walk(runs_dir):
            for f in files:
                if f == "latest_forecast.csv":
                    candidates.append(os.path.join(root, f))

    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def attach_projected_cases(dat_f):
    if "projected_cases" in dat_f.columns and dat_f["projected_cases"].notna().all():
        return dat_f

    forecast_path = find_latest_forecast_csv()
    if forecast_path is None:
        if "projected_cases" not in dat_f.columns:
            dat_f["projected_cases"] = dat_f["risk_score_1_100"]
        return dat_f

    fdf = pd.read_csv(forecast_path)
    if fdf.empty:
        if "projected_cases" not in dat_f.columns:
            dat_f["projected_cases"] = dat_f["risk_score_1_100"]
        return dat_f

    row = fdf.iloc[-1]
    out = dat_f.copy()
    if "projected_cases" not in out.columns:
        out["projected_cases"] = pd.NA
    for h in (1, 2, 3, 4):
        col = f"llm_forecast_{h}"
        if col in row.index:
            out.loc[out["horizon"] == h, "projected_cases"] = row[col]
    out["projected_cases"] = pd.to_numeric(out["projected_cases"], errors="coerce")
    out["projected_cases"] = out["projected_cases"].fillna(out["risk_score_1_100"])
    return out


def risk_color(score):
    if score <= 20:
        return "darkgreen"
    if score <= 40:
        return "green"
    if score <= 60:
        return "grey"
    if score <= 80:
        return "orange"
    return "red"


def build_gauge_figure(dat_f):
    dat_f = dat_f.sort_values("horizon").reset_index(drop=True)
    n = len(dat_f)
    fig = make_subplots(rows=1, cols=n, specs=[[{"type": "indicator"} for _ in range(n)]])

    for idx, row in dat_f.iterrows():
        score = float(row["risk_score_1_100"])
        projected_cases = row.get("projected_cases", score)
        if pd.isna(projected_cases):
            projected_cases = score
        cat_col = risk_color(score)
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=float(projected_cases),
                number={"valueformat": ".0f", "font": {"size": 44}},
                title={
                    "text": (
                        f"<span style='font-size:20px'><b>Horizon {int(row['horizon'])}</b></span><br>"
                        f"<span style='font-size:16px;color:gray'>{row['date']}</span><br>"
                        f"<span style='color:{cat_col}'><b>{str(row['category']).upper()}</b></span>"
                    )
                },
                gauge={
                    "axis": {"range": [0, 100]},
                    # Keep gauge risk-marker behavior tied to score, while number displays projected cases.
                    "bar": {"color": "rgba(0,0,0,0)"},
                    "steps": [
                        {"range": [0, 20], "color": "darkgreen"},
                        {"range": [20, 40], "color": "green"},
                        {"range": [40, 60], "color": "yellow"},
                        {"range": [60, 80], "color": "orange"},
                        {"range": [80, 100], "color": "red"},
                    ],
                    "threshold": {"line": {"color": "black", "width": 6}, "value": score},
                },
            ),
            row=1,
            col=idx + 1,
        )

    fig.update_layout(margin={"t": 120}, template="none")
    return fig


def main():
    gauge_path = find_latest_gauge_csv()
    dat_f = pd.read_csv(gauge_path)
    required = {"horizon", "date", "category", "risk_score_1_100"}
    missing = required - set(dat_f.columns)
    if missing:
        raise ValueError(f"Gauge file missing required columns: {sorted(missing)}")
    dat_f = attach_projected_cases(dat_f)
    dat_f = dat_f.dropna(subset=["horizon", "date", "risk_score_1_100"])
    if dat_f.empty:
        raise ValueError("Gauge file has no rows with required values.")

    fig = build_gauge_figure(dat_f)
    print(f"Using gauge data: {gauge_path}")
    out_png = os.path.join("results", "gauge_plot.png")
    try:
        fig.write_image(out_png, scale=2, width=1400, height=500)
        print(f"Saved {out_png}")
    except Exception as e:
        out_html = os.path.join("results", "gauge_plot.html")
        fig.write_html(out_html, include_plotlyjs="cdn")
        print(f"WARNING: PNG export failed ({e}). Saved {out_html} instead.")


if __name__ == "__main__":
    main()

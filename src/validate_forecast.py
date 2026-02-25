import argparse
import os
import sys
import pandas as pd


def _parse_date(s, label):
    dt = pd.to_datetime(s, errors="coerce")
    if pd.isna(dt):
        raise ValueError(f"Invalid {label}: {s}")
    return pd.Timestamp(dt).normalize()


def _safe_ape(actual, pred):
    if actual == 0:
        return None
    return abs(pred - actual) / abs(actual) * 100.0


def main():
    parser = argparse.ArgumentParser(
        description="Validate a stored 1-3 week forecast against confirmed cases."
    )
    parser.add_argument(
        "--actuals-csv",
        type=str,
        default=None,
        help="CSV with confirmed observations. Must include date and cases columns.",
    )
    parser.add_argument(
        "--actuals-date-col",
        type=str,
        default="date",
        help="Date column name in --actuals-csv.",
    )
    parser.add_argument(
        "--actuals-cases-col",
        type=str,
        default="cases",
        help="Confirmed-cases column name in --actuals-csv.",
    )
    parser.add_argument(
        "--actuals-sep",
        type=str,
        default=None,
        help="Delimiter for --actuals-csv (auto-detected when omitted).",
    )
    parser.add_argument(
        "--actuals-dayfirst",
        action="store_true",
        help="Parse actuals dates as day-first format (e.g., 31/01/2026).",
    )
    parser.add_argument(
        "--origin-date",
        type=str,
        default=None,
        help="Forecast origin date (YYYY-MM-DD). Defaults to latest row with forecasts.",
    )
    parser.add_argument(
        "--actual-date",
        type=str,
        required=False,
        help="Confirmed-data date to score against (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--actual-cases",
        type=float,
        required=False,
        help="Confirmed case count for --actual-date.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/output.csv",
        help="Forecast output CSV path.",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="results/forecast_validations.csv",
        help="Where to append validation records.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.output):
        raise SystemExit(f"Missing forecast file: {args.output}")

    df = pd.read_csv(args.output)
    if "date" not in df.columns:
        raise SystemExit(f"Missing 'date' column in {args.output}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()

    needed = ["llm_forecast_1", "llm_forecast_2", "llm_forecast_3"]
    for c in needed:
        if c not in df.columns:
            raise SystemExit(f"Missing '{c}' in {args.output}")

    results = []

    # Batch mode: score forecasts against a CSV of confirmed values.
    if args.actuals_csv:
        if not os.path.exists(args.actuals_csv):
            raise SystemExit(f"Missing actuals CSV: {args.actuals_csv}")
        if args.actuals_sep is None:
            with open(args.actuals_csv, "r", encoding="utf-8") as f:
                header = f.readline()
            if ";" in header and "," not in header:
                sep = ";"
            elif "," in header:
                sep = ","
            else:
                sep = None
        else:
            sep = args.actuals_sep

        actuals = pd.read_csv(args.actuals_csv, sep=sep) if sep is not None else pd.read_csv(args.actuals_csv, sep=None, engine="python")
        if args.actuals_date_col not in actuals.columns:
            raise SystemExit(
                f"Missing date column '{args.actuals_date_col}' in {args.actuals_csv}"
            )
        if args.actuals_cases_col not in actuals.columns:
            raise SystemExit(
                f"Missing cases column '{args.actuals_cases_col}' in {args.actuals_csv}"
            )
        actuals = actuals[[args.actuals_date_col, args.actuals_cases_col]].copy()
        actuals = actuals.rename(
            columns={
                args.actuals_date_col: "actual_date",
                args.actuals_cases_col: "actual_cases",
            }
        )
        actuals["actual_date"] = pd.to_datetime(
            actuals["actual_date"],
            errors="coerce",
            dayfirst=args.actuals_dayfirst,
        ).dt.normalize()
        actuals = actuals.dropna(subset=["actual_date", "actual_cases"])
        actual_map = {
            pd.Timestamp(r["actual_date"]).normalize(): float(r["actual_cases"])
            for _, r in actuals.iterrows()
        }

        if args.origin_date:
            origins = df[df["date"] == _parse_date(args.origin_date, "origin date")]
            if origins.empty:
                raise SystemExit(f"No row found for origin date {args.origin_date}")
        else:
            has_any = df[needed].notna().any(axis=1)
            origins = df.loc[has_any]

        if origins.empty:
            raise SystemExit("No forecast rows found to validate.")

        for _, row in origins.iterrows():
            origin_dt = pd.Timestamp(row["date"]).normalize()
            for horizon in (1, 2, 3):
                pred_col = f"llm_forecast_{horizon}"
                cat_col = f"llm_category_{horizon}"
                pred = row.get(pred_col)
                if pd.isna(pred):
                    continue
                actual_dt = origin_dt + pd.to_timedelta(7 * horizon, unit="D")
                if actual_dt not in actual_map:
                    continue
                actual = actual_map[actual_dt]
                pred = float(pred)
                err = pred - actual
                abs_err = abs(err)
                ape = _safe_ape(actual, pred)
                results.append(
                    {
                        "origin_date": origin_dt.date().isoformat(),
                        "actual_date": actual_dt.date().isoformat(),
                        "horizon_weeks": int(horizon),
                        "forecast": pred,
                        "actual_cases": actual,
                        "error": err,
                        "abs_error": abs_err,
                        "ape_percent": ape,
                        "category": row.get(cat_col) if pd.notna(row.get(cat_col)) else None,
                    }
                )
        if not results:
            raise SystemExit(
                "No matching validations found. Check date alignment and weekly spacing."
            )
    else:
        # Single-point mode (existing behavior).
        if args.actual_date is None or args.actual_cases is None:
            raise SystemExit(
                "Either provide --actuals-csv, or provide both --actual-date and --actual-cases."
            )
        if args.origin_date:
            origin_dt = _parse_date(args.origin_date, "origin date")
            matches = df[df["date"] == origin_dt]
            if matches.empty:
                raise SystemExit(f"No row found for origin date {origin_dt.date()}")
            row = matches.iloc[-1]
        else:
            has_any = df[needed].notna().any(axis=1)
            if not has_any.any():
                raise SystemExit("No forecast rows found in output CSV.")
            row = df.loc[has_any].iloc[-1]
            origin_dt = pd.Timestamp(row["date"]).normalize()

        actual_dt = _parse_date(args.actual_date, "actual date")
        day_diff = (actual_dt - origin_dt).days
        if day_diff <= 0 or day_diff % 7 != 0:
            expected = [
                (origin_dt + pd.to_timedelta(7 * h, unit="D")).date().isoformat()
                for h in [1, 2, 3]
            ]
            raise SystemExit(
                "actual-date must be a positive weekly horizon from origin-date "
                f"(expected one of: {expected})"
            )

        horizon = day_diff // 7
        if horizon not in (1, 2, 3):
            raise SystemExit(f"Only horizons 1-3 are supported. Got horizon={horizon}.")

        pred_col = f"llm_forecast_{horizon}"
        cat_col = f"llm_category_{horizon}"
        pred = row.get(pred_col)
        category = row.get(cat_col)
        if pd.isna(pred):
            raise SystemExit(
                f"No forecast value in {pred_col} for origin date {origin_dt.date()}."
            )

        pred = float(pred)
        actual = float(args.actual_cases)
        err = pred - actual
        abs_err = abs(err)
        ape = _safe_ape(actual, pred)
        results.append(
            {
                "origin_date": origin_dt.date().isoformat(),
                "actual_date": actual_dt.date().isoformat(),
                "horizon_weeks": int(horizon),
                "forecast": pred,
                "actual_cases": actual,
                "error": err,
                "abs_error": abs_err,
                "ape_percent": ape,
                "category": category if pd.notna(category) else None,
            }
        )

    os.makedirs(os.path.dirname(args.log_file) or ".", exist_ok=True)
    log_df = pd.DataFrame(results)
    if os.path.exists(args.log_file):
        existing = pd.read_csv(args.log_file)
        out = pd.concat([existing, log_df], ignore_index=True)
    else:
        out = log_df
    out.to_csv(args.log_file, index=False)

    print(f"Validation rows: {len(results)}")
    summary = log_df.groupby("horizon_weeks", dropna=False).agg(
        n=("abs_error", "count"),
        MAE=("abs_error", "mean"),
        MAPE=("ape_percent", "mean"),
    ).reset_index()
    print(summary.to_string(index=False))
    print(f"Appended to {args.log_file}")


if __name__ == "__main__":
    try:
        main()
    except ValueError as e:
        print(str(e), file=sys.stderr)
        raise SystemExit(2)

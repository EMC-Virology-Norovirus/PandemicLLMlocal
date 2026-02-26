import pandas as pd
import numpy as np
import json
import re
import argparse
import os
import time
import statistics
import concurrent.futures
from datetime import datetime, timezone
import shutil
import subprocess
import ollama

df = pd.DataFrame()


def run_r_pipeline(r_script_path):
    if not os.path.exists(r_script_path):
        raise FileNotFoundError(f"R pipeline script not found: {r_script_path}")
    rscript = shutil.which("Rscript")
    if rscript is None:
        raise RuntimeError("Rscript was not found on PATH. Install R and ensure Rscript is available.")
    print(f"Running R data pipeline: {r_script_path}")
    proc = subprocess.run([rscript, r_script_path], capture_output=True, text=True)
    if proc.returncode != 0:
        stderr_tail = (proc.stderr or "").strip()[-1000:]
        raise RuntimeError(f"R pipeline failed with code {proc.returncode}: {stderr_tail}")
    print("R data pipeline completed.")


def load_and_prepare_data(data_path, include_social_index=True):
    global df
    df = pd.read_csv(data_path)

    # calculate simple trends
    df["wastewater_change"] = df["ww_mean"].pct_change()
    df["positivity_change"] = df["pct_mean"].pct_change()

    # identify lineage columns (columns starting with "Lineage_")
    lineage_cols = [c for c in df.columns if c.startswith("Lineage_")]
    if lineage_cols:
        # current percentage for top lineage and which lineage it is
        df["top_lineage"] = df[lineage_cols].idxmax(axis=1)
        df["top_lineage_pct"] = df[lineage_cols].max(axis=1)
        # week-over-week percent change for each lineage
        lineage_pct_change = df[lineage_cols].pct_change()

        # extract the pct_change corresponding to the top lineage for each row
        def _top_lineage_pct_change(row):
            top = row["top_lineage"]
            if pd.isna(top):
                return np.nan
            return lineage_pct_change.at[row.name, top]

        df["top_lineage_pct_change"] = df.apply(_top_lineage_pct_change, axis=1)
    else:
        df["top_lineage"] = np.nan
        df["top_lineage_pct"] = np.nan
        df["top_lineage_pct_change"] = np.nan

    # Optional: remove social-index features when unavailable/undesired.
    if not include_social_index:
        drop_cols = [c for c in ["social_index", "social_lag1"] if c in df.columns]
        if drop_cols:
            df = df.drop(columns=drop_cols)

    return df

def _fmt_pct(v, pct=False):
    if pd.isna(v):
        return "N/A"
    if pct:
        return f"{v:.2f}%"
    return f"{v:.2%}"


def create_prompt_for_index(df_local, idx, include_social_index=True):
    # last 4 weeks up to idx (inclusive)
    start = max(0, idx - 3)
    window = df_local.iloc[start : idx + 1]

    lines = [
        "You are an epidemiologist monitoring outbreak risk.",
        "Provide a short assessment based on the most recent four weeks of data (most recent last).",
        "",
        "Weekly data:",
    ]

    for _, r in window.iterrows():
        date = r.get("date", "")
        ww = _fmt_pct(r.get("wastewater_change"))
        pos = _fmt_pct(r.get("positivity_change"))
        cases = r.get("cases", "N/A")
        top = r.get("top_lineage", "N/A")
        top_pct = _fmt_pct(r.get("top_lineage_pct"), pct=True)
        top_change = _fmt_pct(r.get("top_lineage_pct_change"))
        social_segment = ""
        if include_social_index and ("social_index" in df_local.columns):
            social_val = r.get("social_index", "N/A")
            social_lag = r.get("social_lag1", "N/A")
            social_segment = f", Social index {social_val}, Social lag1 {social_lag}"

        lines.append(
            f"{date}: Wastewater change {ww}, Positivity change {pos}, Cases {cases}, Top lineage {top} ({top_pct}), Lineage WoW change {top_change}{social_segment}"
        )

    lines.extend([
        "",
        "Question:",
        "Is outbreak risk significantly increasing, increasing, stable, decreasing, or significantly decreasing based on the last four weeks?",
        "",
        "Task:",
        "Provide numeric forecasts for 1-week, 2-weeks, and 3-weeks ahead (predicted `cases`) and optional 95% CI for each horizon. Also provide a categorical risk label for each horizon chosen from: [\"significantly increasing\", \"increasing\", \"stable\", \"decreasing\", \"significantly decreasing\"].",
        "Return ONLY a JSON object with keys:",
        "  - forecast_1, forecast_2, forecast_3: numeric (predicted cases for 1/2/3 weeks ahead),",
        "  - ci_lower_1, ci_upper_1, ci_lower_2, ci_upper_2, ci_lower_3, ci_upper_3: numeric or null (optional),",
        "  - category_1, category_2, category_3: one of the labels above (strings),",
        "  - explanation: short text explanation (optional)",
        "Example: {\"forecast_1\": 123.4, \"ci_lower_1\": 100, \"ci_upper_1\": 140, \"category_1\": \"increasing\", \"forecast_2\": 150, ...}",
    ])

    return "\n".join(lines)


def parse_llm_response(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.strip()
    raw_text = text
    json_candidate = text
    # Handle common fenced-json responses: ```json { ... } ```
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        json_candidate = fenced.group(1).strip()
    else:
        # Fallback: try to parse from the first {...} object in the response.
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            json_candidate = text[start:end + 1]
    # Try JSON first
    try:
        obj = json.loads(json_candidate)
        return {
            "forecast_1": obj.get("forecast_1"),
            "forecast_2": obj.get("forecast_2"),
            "forecast_3": obj.get("forecast_3"),
            "ci_lower_1": obj.get("ci_lower_1"),
            "ci_upper_1": obj.get("ci_upper_1"),
            "ci_lower_2": obj.get("ci_lower_2"),
            "ci_upper_2": obj.get("ci_upper_2"),
            "ci_lower_3": obj.get("ci_lower_3"),
            "ci_upper_3": obj.get("ci_upper_3"),
            "category_1": obj.get("category_1"),
            "category_2": obj.get("category_2"),
            "category_3": obj.get("category_3"),
            "explanation": obj.get("explanation") if obj.get("explanation") is not None else "",
            "raw": raw_text,
        }
    except Exception:
        lower_text = text.lower()
        def _extract_keyed_float(key):
            m = re.search(rf'"{re.escape(key)}"\s*:\s*(-?\d+(?:\.\d+)?)', text, flags=re.IGNORECASE)
            if m:
                try:
                    return float(m.group(1))
                except Exception:
                    return None
            return None

        f1_keyed = _extract_keyed_float("forecast_1")
        f2_keyed = _extract_keyed_float("forecast_2")
        f3_keyed = _extract_keyed_float("forecast_3")

        ci_lower_1 = _extract_keyed_float("ci_lower_1")
        ci_upper_1 = _extract_keyed_float("ci_upper_1")
        ci_lower_2 = _extract_keyed_float("ci_lower_2")
        ci_upper_2 = _extract_keyed_float("ci_upper_2")
        ci_lower_3 = _extract_keyed_float("ci_lower_3")
        ci_upper_3 = _extract_keyed_float("ci_upper_3")

        labels = [
            "significantly increasing",
            "increasing",
            "stable",
            "decreasing",
            "significantly decreasing",
        ]
        def _extract_category_keyed(key):
            m = re.search(rf'"{re.escape(key)}"\s*:\s*"([^"]+)"', text, flags=re.IGNORECASE)
            if not m:
                return None
            val = m.group(1).strip().lower()
            return val if val in labels else None

        cat1_keyed = _extract_category_keyed("category_1")
        cat2_keyed = _extract_category_keyed("category_2")
        cat3_keyed = _extract_category_keyed("category_3")

        # fallback: find first number in text
        nums = re.findall(r"(-?\d+\.?\d*)", text)
        # map first three numbers to forecasts if available
        f1 = f1_keyed if f1_keyed is not None else (float(nums[0]) if len(nums) > 0 else None)
        f2 = f2_keyed if f2_keyed is not None else (float(nums[1]) if len(nums) > 1 else None)
        f3 = f3_keyed if f3_keyed is not None else (float(nums[2]) if len(nums) > 2 else None)
        # attempt to pick up category labels in text
        cats = []
        for label in labels:
            if label in lower_text:
                cats.append(label)
        cat1 = cat1_keyed if cat1_keyed is not None else (cats[0] if len(cats) > 0 else None)
        cat2 = cat2_keyed if cat2_keyed is not None else (cats[1] if len(cats) > 1 else None)
        cat3 = cat3_keyed if cat3_keyed is not None else (cats[2] if len(cats) > 2 else None)
        return {
            "forecast_1": f1,
            "forecast_2": f2,
            "forecast_3": f3,
            "ci_lower_1": ci_lower_1,
            "ci_upper_1": ci_upper_1,
            "ci_lower_2": ci_lower_2,
            "ci_upper_2": ci_upper_2,
            "ci_lower_3": ci_lower_3,
            "ci_upper_3": ci_upper_3,
            "category_1": cat1,
            "category_2": cat2,
            "category_3": cat3,
            "explanation": text,
            "raw": text,
        }


def _mock_llm_predict(window_df):
    vals = window_df["cases"].dropna()
    if len(vals) == 0:
        return {"forecast_1": None, "forecast_2": None, "forecast_3": None,
                "ci_lower_1": None, "ci_upper_1": None,
                "ci_lower_2": None, "ci_upper_2": None,
                "ci_lower_3": None, "ci_upper_3": None,
                "category_1": None, "category_2": None, "category_3": None,
                "explanation": "no data", "raw": ""}
    last = float(vals.iat[-1])
    mean = float(vals.mean())
    # simple linear trend per week over the window
    if len(vals) > 1:
        trend = (vals.iat[-1] - vals.iat[0]) / max(1, len(vals) - 1)
    else:
        trend = 0.0

    f1 = mean + 1 * trend
    f2 = mean + 2 * trend
    f3 = mean + 3 * trend

    def cat_for(f):
        if last == 0:
            return "stable"
        pct = (f - last) / last
        if pct > 0.2:
            return "significantly increasing"
        if pct > 0.05:
            return "increasing"
        if pct < -0.2:
            return "significantly decreasing"
        if pct < -0.05:
            return "decreasing"
        return "stable"

    # estimate uncertainty from sample std in the window; scale with sqrt(horizon)
    sample_std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    def ci_for(f, h):
        if sample_std == 0:
            # fallback relative CI
            rel = 0.2
            return max(0, f * (1 - rel)), f * (1 + rel)
        se = sample_std * (h ** 0.5)
        lower = max(0, f - 1.96 * se)
        upper = f + 1.96 * se
        return lower, upper

    cl1, cu1 = ci_for(f1, 1)
    cl2, cu2 = ci_for(f2, 2)
    cl3, cu3 = ci_for(f3, 3)

    return {
        "forecast_1": float(f1), "forecast_2": float(f2), "forecast_3": float(f3),
        "ci_lower_1": float(cl1), "ci_upper_1": float(cu1),
        "ci_lower_2": float(cl2), "ci_upper_2": float(cu2),
        "ci_lower_3": float(cl3), "ci_upper_3": float(cu3),
        "category_1": cat_for(f1), "category_2": cat_for(f2), "category_3": cat_for(f3),
        "explanation": "mock forecasts with CI",
        "raw": f"{f1},{f2},{f3}",
    }


def _fill_missing_with_mock(parsed, window_df):
    """Fill missing forecast/category/CI fields from deterministic mock output.

    Preserves any values already returned by the LLM and only fills missing keys.
    """
    if parsed is None:
        parsed = {}
    mock = _mock_llm_predict(window_df)
    filled = dict(parsed)
    keys_to_fill = [
        "forecast_1", "forecast_2", "forecast_3",
        "ci_lower_1", "ci_upper_1",
        "ci_lower_2", "ci_upper_2",
        "ci_lower_3", "ci_upper_3",
        "category_1", "category_2", "category_3",
    ]
    filled_any = False
    for k in keys_to_fill:
        v = filled.get(k, None)
        if v is None or (isinstance(v, float) and pd.isna(v)):
            mv = mock.get(k)
            if mv is not None and not (isinstance(mv, float) and pd.isna(mv)):
                filled[k] = mv
                filled_any = True

    if filled_any:
        prior_expl = filled.get("explanation")
        if prior_expl is None or (isinstance(prior_expl, float) and pd.isna(prior_expl)) or str(prior_expl).strip() == "":
            filled["explanation"] = "partial_llm_filled_with_mock"
        else:
            filled["explanation"] = f"{prior_expl} | partial_fields_filled_with_mock"
    return filled, filled_any


def run(
    llm_mock=False,
    last_only=False,
    model="mistral",
    workers=1,
    timeout=240.0,
    retries=3,
    limit=None,
    num_thread=None,
    progress_every=1,
    max_tokens=256,
    fallback_on_error=True,
    start_date=None,
    rerun_failed=False,
    resume_file="results/output.csv",
    include_social_index=True,
):
    # prepare default empty entry
    default = {
        "forecast_1": None, "forecast_2": None, "forecast_3": None,
        "ci_lower_1": None, "ci_upper_1": None,
        "ci_lower_2": None, "ci_upper_2": None,
        "ci_lower_3": None, "ci_upper_3": None,
        "category_1": None, "category_2": None, "category_3": None,
        "explanation": None, "raw": ""
    }

    # initialize parsed_results with defaults for each row
    parsed_results = [default.copy() for _ in range(len(df))]

    if len(df) == 0:
        return

    def _to_idx_set(mask):
        return set(np.flatnonzero(mask.to_numpy()))

    eligible_set = set(range(len(df)))
    if start_date is not None:
        dt = pd.to_datetime(df["date"], errors="coerce")
        start_dt = pd.to_datetime(start_date)
        eligible_mask = dt >= start_dt
        eligible_set = _to_idx_set(eligible_mask.fillna(False))

    def _seed_from_existing(path):
        failed = set()
        if not os.path.exists(path):
            return failed
        try:
            prev = pd.read_csv(path)
        except Exception:
            return failed

        # map previous rows by index when lengths match; else by date key
        idx_map = {}
        if len(prev) == len(df):
            idx_map = {i: i for i in range(len(df))}
        elif "date" in prev.columns and "date" in df.columns:
            prev_by_date = {}
            for i, d in enumerate(prev["date"]):
                if pd.notna(d) and d not in prev_by_date:
                    prev_by_date[d] = i
            for i, d in enumerate(df["date"]):
                if pd.notna(d) and d in prev_by_date:
                    idx_map[i] = prev_by_date[d]

        llm_cols = [
            "llm_forecast_1", "llm_forecast_2", "llm_forecast_3",
            "llm_ci_lower_1", "llm_ci_upper_1",
            "llm_ci_lower_2", "llm_ci_upper_2",
            "llm_ci_lower_3", "llm_ci_upper_3",
            "llm_category_1", "llm_category_2", "llm_category_3",
            "llm_explanation",
        ]
        for i_df, i_prev in idx_map.items():
            rec = default.copy()
            for c in llm_cols:
                if c in prev.columns:
                    rec_key = c.replace("llm_", "")
                    rec[rec_key] = prev.at[i_prev, c]
            if isinstance(rec.get("explanation"), float) and pd.isna(rec["explanation"]):
                rec["explanation"] = None
            parsed_results[i_df] = rec

            expl = rec.get("explanation")
            has_missing = any(pd.isna(rec.get(k)) for k in ["forecast_1", "forecast_2", "forecast_3", "category_1", "category_2", "category_3"])
            is_failed = False
            if isinstance(expl, str) and (
                expl.startswith("error:")
                or expl.startswith("fallback_after_error:")
            ):
                is_failed = True
            if has_missing:
                is_failed = True
            if is_failed:
                failed.add(i_df)
        return failed

    failed_set = set()
    if rerun_failed:
        failed_set = _seed_from_existing(resume_file)
        print(f"[resume] loaded prior output from {resume_file}; failed_or_missing_rows={len(failed_set)}")

    if last_only:
        if not eligible_set:
            indices = []
        else:
            indices = [max(eligible_set)]
    else:
        indices = sorted(eligible_set)

    if rerun_failed:
        indices = [i for i in indices if i in failed_set]

    timings = []
    run_started = time.time()
    alert_counts = {"success": 0, "timeouts": 0, "errors": 0, "insufficient": 0}

    def _call_with_timeout_and_retries(prompt, model_name, timeout_s, retries_n, num_thread_opt, max_tokens_opt, alert_tag=""):
        # Use Ollama client's native HTTP timeout and retry with backoff.
        last_exc = None
        for attempt in range(1, retries_n + 1):
            try:
                print(f"[call] {alert_tag} attempt {attempt}/{retries_n} (timeout={timeout_s:.0f}s, model={model_name})")
                client = ollama.Client(timeout=timeout_s)
                opts = {}
                if num_thread_opt is not None:
                    opts["num_thread"] = int(num_thread_opt)
                if max_tokens_opt is not None:
                    opts["num_predict"] = int(max_tokens_opt)
                opts["temperature"] = 0
                return client.chat(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    options=opts if opts else None,
                )
            except Exception as e:
                last_exc = e
                print(f"[warn] {alert_tag} attempt {attempt} failed: {repr(e)}")
                backoff = min(5, 0.5 * (2 ** (attempt - 1)))
                time.sleep(backoff)
                continue
        raise last_exc

    def _classify_result(parsed):
        expl = parsed.get("explanation")
        if expl == "Insufficient data":
            return "insufficient"
        if isinstance(expl, str) and expl.startswith("error:"):
            expl_l = expl.lower()
            if "timeout" in expl_l or "timed out" in expl_l:
                return "timeouts"
            return "errors"
        if isinstance(expl, str) and expl.startswith("fallback_after_error:"):
            expl_l = expl.lower()
            if "timeout" in expl_l or "timed out" in expl_l:
                return "timeouts"
            return "errors"
        return "success"

    def _emit_progress(completed, total):
        avg = statistics.mean([t["duration"] for t in timings]) if timings else 0.0
        elapsed = time.time() - run_started
        remaining = max(0, total - completed)
        eta = remaining * avg
        pct = 100.0 * completed / max(1, total)
        print(
            f"[progress] {completed}/{total} ({pct:.1f}%) "
            f"ok={alert_counts['success']} timeout={alert_counts['timeouts']} "
            f"error={alert_counts['errors']} insufficient={alert_counts['insufficient']} "
            f"avg={avg:.2f}s elapsed={elapsed:.1f}s eta={eta:.1f}s"
        )


    def _process_index(idx):
        row = df.iloc[idx]
        if pd.isna(row.wastewater_change):
            pr = default.copy()
            pr["explanation"] = "Insufficient data"
            return idx, pr, 0.0

        prompt = create_prompt_for_index(df, idx, include_social_index=include_social_index)
        start = max(0, idx - 3)
        window = df.iloc[start: idx + 1]
        start_t = time.time()
        parsed = default.copy()
        try:
            if llm_mock:
                parsed = _mock_llm_predict(window)
            else:
                # call with timeout and retries
                resp = _call_with_timeout_and_retries(
                    prompt, model, timeout, retries, num_thread, max_tokens, alert_tag=f"row={idx}"
                )
                text = resp["message"]["content"]
                parsed = parse_llm_response(text)
                parsed, _ = _fill_missing_with_mock(parsed, window)
        except Exception as e:
            if fallback_on_error:
                parsed = _mock_llm_predict(window)
                parsed["explanation"] = f"fallback_after_error: {repr(e)}"
            else:
                parsed = default.copy()
                parsed["explanation"] = f"error: {repr(e)}"
        dur = time.time() - start_t
        return idx, parsed, dur

    # run either serially or with a ThreadPoolExecutor for concurrency
    if limit is not None:
        # restrict indices for quick tests
        indices = indices[:max(0, int(limit))]

    if workers is None or workers <= 1:
        completed = 0
        total = len(indices)
        for idx in indices:
            print(f"[start] row={idx} ({completed + 1}/{total})")
            idxr, parsed, dur = _process_index(idx)
            parsed_results[idxr] = parsed
            timings.append({"index": idxr, "duration": dur})
            alert_counts[_classify_result(parsed)] += 1
            completed += 1
            if completed % max(1, int(progress_every)) == 0 or completed == total:
                _emit_progress(completed, total)
            # save partial results periodically
            if completed % max(1, int(progress_every)) == 0 or completed == total:
                try:
                    _save_partial(parsed_results, timings)
                except Exception:
                    pass
    else:
        total = len(indices)
        completed = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_process_index, idx): idx for idx in indices}
            for fut in concurrent.futures.as_completed(futures):
                idxr, parsed, dur = fut.result()
                parsed_results[idxr] = parsed
                timings.append({"index": idxr, "duration": dur})
                alert_counts[_classify_result(parsed)] += 1
                completed += 1
                if completed % max(1, int(progress_every)) == 0 or completed == total:
                    _emit_progress(completed, total)
                # save partial results periodically
                if completed % max(1, int(progress_every)) == 0 or completed == total:
                    try:
                        _save_partial(parsed_results, timings)
                    except Exception:
                        pass

    # save timings for diagnostics
    os.makedirs("results", exist_ok=True)
    if timings:
        tdf = pd.DataFrame(timings).sort_values("index")
        tdf.to_csv("results/timings.csv", index=False)
        avg_time = float(tdf["duration"].mean())
        print(f"Timing: avg per-call {avg_time:.2f}s over {len(tdf)} calls; saved results/timings.csv")
        # warn if unusually slow
        if avg_time > 5.0 and len(indices) > 10:
            print("WARNING: Average LLM call time is high (>5s). Consider using a smaller model or fewer workers.")

    # helper: save partial outputs to disk for triage
    def _save_partial(parsed_results_list, timings_list):
        try:
            df_partial = df.copy()
            # attach whatever parsed_results we have so far
            df_partial["llm_forecast_1"] = [p.get("forecast_1") for p in parsed_results_list]
            df_partial["llm_forecast_2"] = [p.get("forecast_2") for p in parsed_results_list]
            df_partial["llm_forecast_3"] = [p.get("forecast_3") for p in parsed_results_list]
            df_partial["llm_ci_lower_1"] = [p.get("ci_lower_1") for p in parsed_results_list]
            df_partial["llm_ci_upper_1"] = [p.get("ci_upper_1") for p in parsed_results_list]
            df_partial["llm_ci_lower_2"] = [p.get("ci_lower_2") for p in parsed_results_list]
            df_partial["llm_ci_upper_2"] = [p.get("ci_upper_2") for p in parsed_results_list]
            df_partial["llm_ci_lower_3"] = [p.get("ci_lower_3") for p in parsed_results_list]
            df_partial["llm_ci_upper_3"] = [p.get("ci_upper_3") for p in parsed_results_list]
            df_partial["llm_category_1"] = [p.get("category_1") for p in parsed_results_list]
            df_partial["llm_category_2"] = [p.get("category_2") for p in parsed_results_list]
            df_partial["llm_category_3"] = [p.get("category_3") for p in parsed_results_list]
            df_partial["llm_explanation"] = [p.get("explanation") for p in parsed_results_list]
            os.makedirs("results", exist_ok=True)
            df_partial.to_csv("results/output_partial.csv", index=False)
            if timings_list:
                pd.DataFrame(timings_list).to_csv("results/timings_partial.csv", index=False)
        except Exception:
            pass

    # attach parsed columns for horizons 1-3 (columns for every row; only last row filled if last_only=True)
    df["llm_forecast_1"] = [p.get("forecast_1") for p in parsed_results]
    df["llm_forecast_2"] = [p.get("forecast_2") for p in parsed_results]
    df["llm_forecast_3"] = [p.get("forecast_3") for p in parsed_results]

    df["llm_ci_lower_1"] = [p.get("ci_lower_1") for p in parsed_results]
    df["llm_ci_upper_1"] = [p.get("ci_upper_1") for p in parsed_results]
    df["llm_ci_lower_2"] = [p.get("ci_lower_2") for p in parsed_results]
    df["llm_ci_upper_2"] = [p.get("ci_upper_2") for p in parsed_results]
    df["llm_ci_lower_3"] = [p.get("ci_lower_3") for p in parsed_results]
    df["llm_ci_upper_3"] = [p.get("ci_upper_3") for p in parsed_results]

    df["llm_category_1"] = [p.get("category_1") for p in parsed_results]
    df["llm_category_2"] = [p.get("category_2") for p in parsed_results]
    df["llm_category_3"] = [p.get("category_3") for p in parsed_results]

    df["llm_explanation"] = [p.get("explanation") for p in parsed_results]
def compute_metrics(df_in, max_h=3):
    """Compute MAE, RMSE, MAPE, SMAPE, and MASE for horizons 1..max_h.

    Returns a DataFrame with rows for each horizon and saves to results/metrics.csv.
    """
    df_local = df_in.copy()
    metrics = []

    # naive one-step MAE for MASE denominator (use absolute differences)
    naive_errors = df_local["cases"].diff().abs().dropna()
    naive_mae = float(naive_errors.mean()) if len(naive_errors) > 0 else np.nan

    # If there are not enough LLM forecasts across rows (e.g., only final row),
    # generate a rolling-origin set of mock forecasts for evaluation so metrics
    # can be computed. This avoids NaNs when the pipeline was run with
    # last-only behavior.
    existing_count = 0
    if "llm_forecast_1" in df_local.columns:
        existing_count = int(df_local["llm_forecast_1"].notna().sum())
    metrics_source = "llm_forecasts"
    # use rolling-origin mock only if there are zero real forecasts
    if existing_count < 1:
        # generate eval_* columns using the mock predictor for all indices
        for h in range(1, max_h + 1):
            df_local[f"eval_forecast_{h}"] = np.nan
            df_local[f"eval_ci_lower_{h}"] = np.nan
            df_local[f"eval_ci_upper_{h}"] = np.nan
            df_local[f"eval_category_{h}"] = None

        for idx in range(len(df_local)):
            # require at least one prior week to form a window
            start = max(0, idx - 3)
            window = df_local.iloc[start: idx + 1]
            if window["cases"].dropna().empty:
                continue
            pred = _mock_llm_predict(window)
            for h in range(1, max_h + 1):
                df_local.at[idx, f"eval_forecast_{h}"] = pred.get(f"forecast_{h}")
                df_local.at[idx, f"eval_ci_lower_{h}"] = pred.get(f"ci_lower_{h}")
                df_local.at[idx, f"eval_ci_upper_{h}"] = pred.get(f"ci_upper_{h}")
                df_local.at[idx, f"eval_category_{h}"] = pred.get(f"category_{h}")

        metrics_source = "mock_rolling_origin"

    for h in range(1, max_h + 1):
        # choose forecast column: prefer real LLM forecasts if available, else eval_*
        fcol_llm = f"llm_forecast_{h}"
        fcol_eval = f"eval_forecast_{h}"
        if (fcol_llm in df_local.columns) and (df_local[fcol_llm].notna().sum() >= 1):
            fcol = fcol_llm
        elif fcol_eval in df_local.columns and df_local[fcol_eval].notna().sum() >= 1:
            fcol = fcol_eval
        else:
            metrics.append({"horizon": h, "n": 0, "MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan, "SMAPE": np.nan, "MASE": np.nan, "source": metrics_source})
            continue
        if fcol not in df_local.columns:
            metrics.append({"horizon": h, "n": 0, "MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan, "SMAPE": np.nan, "MASE": np.nan})
            continue

        truth = df_local["cases"].shift(-h)
        mask = df_local[fcol].notna() & truth.notna()
        n = int(mask.sum())
        if n == 0:
            metrics.append({"horizon": h, "n": 0, "MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan, "SMAPE": np.nan, "MASE": np.nan})
            continue

        preds = df_local.loc[mask, fcol].astype(float)
        tr = truth.loc[mask].astype(float)
        errors = preds - tr
        mae = float(errors.abs().mean())
        rmse = float(np.sqrt((errors ** 2).mean()))

        # MAPE: skip zero truths
        nonzero = tr != 0
        if nonzero.any():
            mape = float((errors.abs()[nonzero] / tr[nonzero]).mean() * 100.0)
        else:
            mape = np.nan

        # SMAPE
        denom = preds.abs() + tr.abs()
        valid = denom != 0
        if valid.any():
            smape = float((2.0 * errors.abs()[valid] / denom[valid]).mean() * 100.0)
        else:
            smape = np.nan

        # MASE: scale by naive_mae if available
        if not np.isnan(naive_mae) and naive_mae > 0:
            mase = float(mae / naive_mae)
        else:
            mase = np.nan

        metrics.append({
            "horizon": h,
            "n": n,
            "MAE": mae,
            "RMSE": rmse,
            "MAPE": mape,
            "SMAPE": smape,
            "MASE": mase,
            "source": metrics_source,
        })

    metrics_df = pd.DataFrame(metrics)
    os.makedirs("results", exist_ok=True)
    metrics_df.to_csv("results/metrics.csv", index=False)
    return metrics_df


def build_gauge_output(df_in):
    """Build 3-row gauge output for the latest forecast origin.

    Columns:
      - horizon
      - date (target date for that horizon)
      - category
      - risk_score_1_100 (lower means decreasing risk)
    """
    if "date" not in df_in.columns:
        return pd.DataFrame(columns=["horizon", "date", "category", "risk_score_1_100"])
    required = ["llm_forecast_1", "llm_forecast_2", "llm_forecast_3", "cases"]
    has_cols = all(c in df_in.columns for c in required)
    if not has_cols:
        return pd.DataFrame(columns=["horizon", "date", "category", "risk_score_1_100"])

    tmp = df_in.copy()
    tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce")
    if tmp["date"].dropna().empty:
        return pd.DataFrame(columns=["horizon", "date", "category", "risk_score_1_100"])

    # Anchor gauge to the final surveillance row date.
    row = tmp.sort_values("date").iloc[-1]
    origin_date = pd.Timestamp(row["date"])
    base_cases = row.get("cases", np.nan)

    def _category_from_score(score):
        # Bin edges requested by user:
        # [0,20) significantly decreasing
        # [20,40) decreasing
        # [40,60) stable
        # [60,80) increasing
        # [80,100] significantly increasing
        if pd.isna(score):
            return None
        if score < 20:
            return "significantly decreasing"
        if score < 40:
            return "decreasing"
        if score < 60:
            return "stable"
        if score < 80:
            return "increasing"
        return "significantly increasing"

    out_rows = []
    for h in (1, 2, 3):
        fcol = f"llm_forecast_{h}"
        pred = row.get(fcol, np.nan)
        # Normalize from the same direction signal used by category logic:
        # pct change relative to latest observed cases.
        if pd.isna(pred) or pd.isna(base_cases) or float(base_cases) == 0.0:
            score = np.nan
        else:
            pct = (float(pred) - float(base_cases)) / float(base_cases)
            # Use +/-20% as the significant-change scale from cat_for thresholds.
            score = 50.0 + 50.0 * (pct / 0.2)
            score = float(np.clip(score, 1.0, 100.0))
        cat = _category_from_score(score)
        target_date = (origin_date + pd.to_timedelta(7 * h, unit="D")).date().isoformat()
        out_rows.append(
            {
                "horizon": h,
                "date": target_date,
                "category": cat,
                "risk_score_1_100": score,
            }
        )

    return pd.DataFrame(out_rows, columns=["horizon", "date", "category", "risk_score_1_100"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock-llm", action="store_true", help="Use a mock LLM (no API calls)")
    parser.add_argument("--last-only", action="store_true", help="Compute forecasts only for the final date in the dataset")
    parser.add_argument("--model", type=str, default="mistral", help="LLM model to call (e.g., phi3, mistral)")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker threads for parallel LLM calls")
    parser.add_argument("--timeout", type=float, default=240.0, help="Timeout in seconds per LLM call")
    parser.add_argument("--retries", type=int, default=1, help="Number of retries per LLM call")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of rows processed (for debugging)")
    parser.add_argument("--num-thread", type=int, default=None, help="CPU threads for Ollama inference (e.g., physical core count)")
    parser.add_argument("--progress-every", type=int, default=1, help="Emit progress alerts every N completed rows")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max generated tokens per LLM response (lower is faster)")
    parser.add_argument("--no-fallback", action="store_true", help="Disable deterministic fallback on LLM error/timeout")
    parser.add_argument("--start-date", type=str, default=None, help="Only process rows with date >= YYYY-MM-DD")
    parser.add_argument("--rerun-failed", action="store_true", help="Resume from existing output and rerun only failed/missing rows")
    parser.add_argument("--resume-file", type=str, default="results/output.csv", help="Path to prior output CSV used by --rerun-failed")
    parser.add_argument("--run-id", type=str, default=None, help="Optional run identifier for artifact tracking")
    parser.add_argument("--data-path", type=str, default="data/processed/surveillance_COVID19_weekly.csv", help="Input CSV path")
    parser.add_argument("--r-script-path", type=str, default="scripts/run_pipeline.R", help="R pipeline script to run first")
    parser.add_argument("--skip-r-pipeline", action="store_true", help="Skip running the R pipeline before loading data")
    parser.add_argument("--include-social-index", dest="include_social_index", action="store_true", default=True, help="Include social index columns in prompt construction when available")
    parser.add_argument("--exclude-social-index", dest="include_social_index", action="store_false", help="Exclude social index columns from prompt construction")
    args = parser.parse_args()

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if not args.skip_r_pipeline:
        run_r_pipeline(args.r_script_path)
    load_and_prepare_data(args.data_path, include_social_index=args.include_social_index)

    run(
        llm_mock=args.mock_llm,
        last_only=args.last_only,
        model=args.model,
        workers=args.workers,
        timeout=args.timeout,
        retries=args.retries,
        limit=args.limit,
        num_thread=args.num_thread,
        progress_every=args.progress_every,
        max_tokens=args.max_tokens,
        fallback_on_error=(not args.no_fallback),
        start_date=args.start_date,
        rerun_failed=args.rerun_failed,
        resume_file=args.resume_file,
        include_social_index=args.include_social_index,
    )

    # save forecasts and computed metrics
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/output.csv", index=False)
    run_dir = os.path.join("results", "runs", run_id)
    os.makedirs(run_dir, exist_ok=True)
    df.to_csv(os.path.join(run_dir, "output.csv"), index=False)

    # Last-only mode predicts the final observation, so there is no future ground truth
    # in this file to evaluate against; metrics would be all NaN and misleading.
    if args.last_only:
        latest_cols = [
            "date", "cases",
            "llm_forecast_1", "llm_forecast_2", "llm_forecast_3",
            "llm_ci_lower_1", "llm_ci_upper_1",
            "llm_ci_lower_2", "llm_ci_upper_2",
            "llm_ci_lower_3", "llm_ci_upper_3",
            "llm_category_1", "llm_category_2", "llm_category_3",
            "llm_explanation",
        ]
        df.tail(1)[latest_cols].to_csv("results/latest_forecast.csv", index=False)
        df.tail(1)[latest_cols].to_csv(os.path.join(run_dir, "latest_forecast.csv"), index=False)
        print("Saved forecasts to results/output.csv and results/latest_forecast.csv")
        print("Skipped metrics: --last-only has no in-file future truth for horizon evaluation.")
    else:
        metrics_df = compute_metrics(df, max_h=3)
        metrics_df.to_csv(os.path.join(run_dir, "metrics.csv"), index=False)
        print("Saved forecasts to results/output.csv and metrics to results/metrics.csv")
        print(metrics_df.to_string(index=False))

    gauge_df = build_gauge_output(df)
    gauge_path = os.path.join("results", "gauge_output.csv")
    gauge_df.to_csv(gauge_path, index=False)
    gauge_df.to_csv(os.path.join(run_dir, "gauge_output.csv"), index=False)
    print(f"Saved gauge output to {gauge_path}")
    if not gauge_df.empty:
        print("Latest horizon summary:")
        print(gauge_df.to_string(index=False))

    # copy timing diagnostics into run folder if present
    for timing_name in ["timings.csv", "timings_partial.csv"]:
        src = os.path.join("results", timing_name)
        if os.path.exists(src):
            try:
                pd.read_csv(src).to_csv(os.path.join(run_dir, timing_name), index=False)
            except Exception:
                pass

    # write run manifest for reproducibility and auditability
    expl = df["llm_explanation"].fillna("") if "llm_explanation" in df.columns else pd.Series([], dtype=str)
    manifest = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "args": vars(args),
        "rows_total": int(len(df)),
        "forecast_non_null": int(df["llm_forecast_1"].notna().sum()) if "llm_forecast_1" in df.columns else 0,
        "category_non_null": int(df["llm_category_1"].notna().sum()) if "llm_category_1" in df.columns else 0,
        "fallback_rows": int(expl.str.startswith("fallback_after_error:").sum()) if len(expl) else 0,
        "error_rows": int(expl.str.startswith("error:").sum()) if len(expl) else 0,
        "insufficient_rows": int((expl == "Insufficient data").sum()) if len(expl) else 0,
        "artifacts_dir": run_dir,
    }
    with open(os.path.join(run_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved run artifacts to {run_dir}")

"""Offline NASA EDA quality audit. Never interprets hardware readings as crew health.

Use: python experiment_d1.py --data-dir PATH [--rules experiment_d1_rules.json]
Only accepts original SHA256-pinned CSVs. No WESAD/model code or pickle loading.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RULES_PATH = ROOT / "experiment_d1_rules.json"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_rules(path=RULES_PATH):
    raw = Path(path).read_bytes()
    rules = json.loads(raw)
    if rules["schema_version"] != "d1-rules-v1" or rules["rr9_timeline_allowed"] is not False:
        raise ValueError("Unsupported or unsafe D1 rules")
    if any(x.get("verified") and not x.get("citation") for x in rules["mission_phases"].values()):
        raise ValueError("Verified phases require citations")
    return rules, hashlib.sha256(raw).hexdigest()


def load_source(path, spec, rules):
    path = Path(path)
    if digest(path) != spec["sha256"]:
        raise ValueError(f"SHA256 mismatch: {path.name}")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if spec["kind"] == "summary_only":
        if list(frame.columns) != ["Unnamed: 0", "Max", "Min", "Mean", "Standard Deviation", "Median"]:
            raise ValueError("Unexpected RR-9 summary header")
        return frame
    if list(frame.columns) != rules["telemetry_columns"]:
        raise ValueError("Unexpected telemetry header")
    column = rules["timestamp"]["column"]
    values = frame[column]
    if not values.str.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}").all():
        raise ValueError("Timestamp format must be YYYY-MM-DD HH:MM")
    timestamps = pd.to_datetime(values, format="%Y-%m-%d %H:%M", exact=True, errors="raise", utc=True)
    intervals = timestamps.diff().dt.total_seconds().div(60).iloc[1:]
    if (intervals <= 0).any():
        raise ValueError("Duplicate or out-of-order timestamps")
    frame[column] = timestamps
    for col in rules["telemetry_columns"]:
        if col == column:
            continue
        if not frame[col].str.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?").all():
            raise ValueError(f"Invalid numeric cells in {col}")
        frame[col] = pd.to_numeric(frame[col], errors="raise")
        if not np.isfinite(frame[col].to_numpy(dtype=float)).all():
            raise ValueError(f"Nonfinite values in {col}")
    return frame


def ensure_timeline(spec):
    if spec["kind"] != "telemetry":
        raise ValueError("RR-9 provides summary only; timeline analysis is not supported")


def flat_runs(frame, cols, min_rows):
    """Find maximal identical consecutive values; return inclusive row boundaries."""
    result = []
    for col in cols:
        values = frame[col].to_numpy()
        boundaries = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1, len(values)]
        for start, stop in zip(boundaries[:-1], boundaries[1:]):
            if stop - start >= min_rows:
                result.append({"channel": col, "value": float(values[start]), "start_row": int(start),
                               "end_row": int(stop - 1), "rows": int(stop - start),
                               "start_gmt": frame.iloc[start, 0].strftime("%Y-%m-%d %H:%M"),
                               "end_gmt": frame.iloc[stop - 1, 0].strftime("%Y-%m-%d %H:%M")})
    return sorted(result, key=lambda x: (x["start_row"], x["channel"]))


def analyze(frame, spec, rules, mission):
    ensure_timeline(spec)
    time_col = rules["timestamp"]["column"]
    timestamps = frame[time_col]
    delta = timestamps.diff().dt.total_seconds().div(60)
    gap_min = rules["gaps"]["threshold_minutes"]
    nominal = rules["timestamp"]["expected_nominal_cadence_minutes"]
    # Each sample represents only the interval to its successor; last row has unknown support.
    intervals = (timestamps.shift(-1) - timestamps).dt.total_seconds().div(60).iloc[:-1]
    supported = intervals.le(gap_min) & intervals.gt(0)
    cols = [x for x in rules["telemetry_columns"] if x != time_col]
    runs = flat_runs(frame, cols, rules["flat_runs"]["min_consecutive_rows"])
    flagged_rows = np.zeros(len(frame), dtype=bool)
    channels_at_row = np.zeros(len(frame), dtype=np.int16)
    for run in runs:
        flagged_rows[run["start_row"]:run["end_row"] + 1] = True
        channels_at_row[run["start_row"]:run["end_row"] + 1] += 1
    # Support interval is flagged when either endpoint is within any flat run.
    flagged_intervals = flagged_rows[:-1] | flagged_rows[1:]
    durations = intervals.to_numpy(dtype=float)
    support = supported.to_numpy(dtype=bool)
    co2 = frame[rules["co2_exploratory_row_threshold"]["column"]].to_numpy(dtype=float)
    threshold = rules["co2_exploratory_row_threshold"]["ppm"]
    hi = co2 > threshold
    ge = co2 >= threshold
    def mins(mask):
        return round(float(durations[support & mask].sum()), 6)
    return {
        "mission": mission, "source_kind": "NASA EDA rodent-research hardware telemetry",
        "is_health_assessment": False, "is_astronaut_validation": False,
        "timeline_scope": "unfiltered_export",
        "rows": int(len(frame)), "start_gmt": timestamps.iloc[0].strftime("%Y-%m-%d %H:%M"),
        "end_gmt": timestamps.iloc[-1].strftime("%Y-%m-%d %H:%M"),
        "median_iss_temp_c": float(frame["Temp_degC_ISS"].median()),
        "median_iss_humidity_percent": float(frame["RH_percent_ISS"].median()),
        "median_iss_co2_ppm": float(frame["CO2_ppm_ISS"].median()),
        "mean_iss_co2_ppm": round(float(frame["CO2_ppm_ISS"].mean()), 4),
        "max_iss_co2_ppm": float(frame["CO2_ppm_ISS"].max()),
        "co2_above_4000_rows": int(hi.sum()), "co2_at_or_above_4000_rows": int(ge.sum()),
        "intervals_shorter_than_nominal": int((delta.iloc[1:] < nominal).sum()),
        "gaps_over_10_min": int((delta.iloc[1:] > gap_min).sum()),
        "largest_gap_min": float(delta.iloc[1:].max()),
        "missing_numeric_cells": int(frame[cols].isna().sum().sum()),
        "supported_minutes": mins(np.ones(len(intervals), dtype=bool)),
        "quality_flagged_minutes": mins(flagged_intervals),
        "flagged_minutes_are_not_confirmed_invalid": True,
        "quality_unflagged_minutes": mins(~flagged_intervals),
        "co2_above_4000_supported_minutes": mins(hi[:-1]),
        "co2_above_4000_unflagged_minutes": mins(hi[:-1] & ~flagged_intervals),
        "unknown_gap_minutes": round(float(durations[~support].sum()), 6),
        "flat_runs": runs,
        "multi_channel_flat_rows": int((channels_at_row >= 2).sum()),
        "notes": ["Flat runs are flags, not proof of sensor failure.",
                  "Duration uses forward intervals <= gap threshold; last sample has unknown duration.",
                  "4000 ppm is exploratory only, not a NASA-STD-3001 compliance assessment.",
                  "The provisional 12-row run threshold overflags quantized channels; review sensitivity before exclusion."]
    }


def run(data_dir, rules_path=RULES_PATH, output_dir=None):
    rules, rules_hash = load_rules(rules_path)
    root = Path(data_dir)
    result = {"experiment": "D1", "is_health_assessment": False,
              "is_astronaut_validation": False, "rules_sha256": rules_hash,
              "sources": {}, "missions": {}, "summary_only": {}}
    # Validate all original file bytes *before* writing an output artifact.
    frames = {}
    for mission, spec in rules["sources"].items():
        path = root / spec["filename"]
        frames[mission] = load_source(path, spec, rules)
        result["sources"][mission] = {"filename": spec["filename"], "sha256": spec["sha256"]}
    for mission, spec in rules["sources"].items():
        if spec["kind"] == "summary_only":
            result["summary_only"][mission] = {"rows": len(frames[mission]), "timeline_available": False,
                "radiation_unit_note": "Source radiation label is ambiguous; no dose interpretation."}
        else:
            result["missions"][mission] = analyze(frames[mission], spec, rules, mission)
    target = Path(output_dir) if output_dir else ROOT / rules["output_dir"]
    # Output root must not overwrite verified WESAD model/research directories.
    if target.resolve().name != "experiment_d1_output":
        raise ValueError("D1 output must be in experiment_d1_output")
    target.mkdir(parents=True, exist_ok=True)
    (target / "d1_report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=os.getenv("ASTROSTRESS_NASA_EDA_DIR"))
    parser.add_argument("--rules", default=str(RULES_PATH))
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    if not args.data_dir:
        parser.error("--data-dir or ASTROSTRESS_NASA_EDA_DIR is required")
    report = run(args.data_dir, args.rules, args.output_dir)
    for mission, info in report["missions"].items():
        print(f"{mission}: {info['rows']} rows; {info['gaps_over_10_min']} gaps; {info['co2_above_4000_rows']} CO2 rows >4000")
    print("D1 audit completed; no astronaut health assessment performed")


if __name__ == "__main__":
    main()

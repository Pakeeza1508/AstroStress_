"""
Check that the dashboard's input artifacts are real, current pipeline outputs.

READ-ONLY: this script never writes, renames or deletes anything.

History: an earlier version of this script wrote loso_results.csv,
personalized_results.csv and summary.json from numbers hardcoded from a past
terminal session. That is retired — evaluation artifacts must come from
running the pipeline:

    python astrostress_complete.py --data_dir WESAD   # features.csv
    python astrostress_v2.py                          # evaluation artifacts

Usage:
    python prepare_dashboard_data.py [--output_dir astrostress_output]

Exit code 0 if all evaluation artifacts are verified for the dashboard,
1 otherwise. Validation rules live in astrostress_outputs.load_dashboard_data.
"""

import argparse
import sys

from astrostress_outputs import (load_dashboard_data, FEATURES_FILE, SUMMARY_FILE,
                                 MODEL_COMPARISON_FILE, LOSO_RESULTS_FILE, PERSONALIZED_FILE,
                                 SHAP_FILE, ANOMALY_FILE, RECOMMENDATIONS_FILE)

ARTIFACTS = [
    ('features', FEATURES_FILE, False),
    ('summary', SUMMARY_FILE, True),
    ('comparison', MODEL_COMPARISON_FILE, True),
    ('loso', LOSO_RESULTS_FILE, True),
    ('personalized', PERSONALIZED_FILE, True),
    ('shap', SHAP_FILE, False),
    ('anomalies', ANOMALY_FILE, False),
    ('recommendations', RECOMMENDATIONS_FILE, False),
]


def check(output_dir):
    """Return (ok, report_lines) without modifying anything."""
    data = load_dashboard_data(output_dir)
    lines = [f"Dashboard artifact check: {output_dir}"]
    ok = True
    for key, name, is_evaluation in ARTIFACTS:
        if is_evaluation:
            status = 'OK (verified)' if key in data else 'NOT VERIFIED'
        else:
            status = 'present' if key in data else 'missing'
        if is_evaluation and key not in data:
            ok = False
        lines.append(f"  {name:28s} {status}")
    if 'summary' in data:
        lines.append(f"  run_id: {data['summary']['run_id']}  "
                     f"generated_at: {data['summary']['generated_at']}")
    for msg in data['issues']:
        lines.append(f"  ! {msg}")
    return ok, lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('--output_dir', default='astrostress_output')
    args = parser.parse_args(argv)
    ok, lines = check(args.output_dir)
    print('\n'.join(lines))
    print('\nAll evaluation artifacts verified.' if ok else
          '\nSome evaluation artifacts are missing, legacy or stale — the dashboard will '
          'not show their metrics. Re-run `python astrostress_v2.py`.')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())

# AstroStress — Experiment D1: Verification and Data-Quality Addendum

**Status:** Follow-up inspection of the supplied NASA Environmental Data Application (EDA) exports. This addendum records additional checks reported from local pandas recomputation; the original D1 audit is preserved unchanged. Not a crew-health assessment or astronaut-validation study.

## Reproduction status

The local review reported that all values in `AstroStress_Experiment_D1_initial_data_audit.md` reproduced from the source files, including all source SHA-256 digests, row counts, time spans, descriptive statistics, threshold counts, constant CO₂ runs and the 1,392-minute RR-6 gap. The locally reported test baseline was 85 passing tests before the later security checks. These source CSVs are held outside GitHub and are not included in this commit; independent execution against the files is still required for a final release of D1 results.

## Additional observed flags (reported from local inspection)

| Mission | Channels | Reported observation | Interpretation |
|---|---|---|---|
| RR-1 | ISS temperature + ISS humidity | 26.8 °C and 70% are both unchanged for 314 successive rows, 2014-09-21 10:44 through 2014-09-22 12:49 GMT | Multi-channel flat interval; review instrument and phase context |
| RR-6 | ISS temperature + ISS humidity | 23.0 °C and 50% are both unchanged for 1,738 successive rows from the first observation through 2017-12-12 15:51 GMT | Multi-channel flat interval overlapping the CO₂ plateau |
| RR-1 | Timestamps | Six intervals shorter than five minutes | Nominal cadence is not strictly regular |
| RR-6 | Timestamps | Twelve intervals shorter than five minutes | Row counts must not be treated as elapsed-duration measures |

Long unchanged sequences are *review flags*, not confirmed sensor failures. Multiple simultaneously flat channels should be analyzed jointly; overlapping runs must not be double-counted as separate exposure durations.

## Fixed comparison semantics

For exploratory CO₂ threshold counting, the original audit uses **strictly greater than 4,000 ppm** (`> 4000`).

| Mission | `CO2_ppm_ISS > 4000` | `CO2_ppm_ISS >= 4000` |
|---|---:|---:|
| RR-1 | 993 rows | 996 rows |
| RR-6 | 95 rows | 96 rows |

These are **row counts only**. They are neither exposure-duration estimates nor NASA-STD-3001 compliance judgments. No fixed exposure threshold is scientifically established here.

## RR-9 and mission-phase cautions

RR-9 is a precomputed summary, not time-series telemetry. The source label **`Radiation Accumulated [mGy/day]`** has an apparent ambiguity between an accumulated quantity and a daily rate. Preserve the original label verbatim, do not reinterpret its units, and exclude that value from temporal analysis until metadata clarify it.

The hypothesis that some RR-6 flat runs represent prelaunch conditions or coincide with CRS-13 milestones is **unverified**. Do not use guessed phase boundaries for filtering, event comparisons or scientific claims. Require a primary mission-page citation, timestamp, timezone and explicit verification before enabling phase-aware computations.

## D1 next implementation gate

1. Strictly verify file bytes against the D1 source hashes, column headers and timestamp ordering before processing.
2. Report missing-time gaps, sub-nominal sample intervals, single-channel constant runs and multi-channel overlap on a common timeline.
3. State explicitly how sample intervals and flagged/unknown periods contribute to duration estimates; do not infer the last sample's support interval without a documented convention.
4. Compare full-export results with quality-gated results; distinguish unknown coverage from low exposure.
5. Reject RR-9 timeline requests and keep it as summary-only.
6. Write all D1 outputs separately from WESAD outputs and do not expose window-level WESAD data to a public API.

**Scope:** NASA EDA environmental measurements originated from rodent-research hardware, not validated astronaut cabin exposure. Experiment D1 does not measure crew stress. Any future D2 WESAD overlay must be labeled a counterfactual, unpaired scenario.

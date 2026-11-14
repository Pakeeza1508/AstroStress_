# AstroStress — Experiment D1: Initial NASA EDA Data Audit
**Status:** Initial descriptive audit of user-supplied original NASA EDA CSV exports; **not** a NASA health-risk assessment or an astronaut validation study.

## Source inventory and integrity

| Mission | Source file | SHA-256 | Rows | Source type |
|---|---|---|---:|---|
| RR-1 | `RR-1_EDA_Telemetry_table.csv` | `6f252984805e8e9a1e7d19927f88b70160ebc30fa3cd7e842572f2b806e5a805` | 12,087 | 5-minute nominal time-series telemetry |
| RR-6 | `RR-6_EDA_Telemetry_table.csv` | `7c8d8f9c2c61856e749c3ac64abb3cdbaabf4f35254c2d5198e1c202e5e0f68f` | 17,572 | 5-minute nominal time-series telemetry |
| RR-9 | `RR-9_EDA_Summary_table.csv` | `341c82277a5803f381c603cfe55d899718b753e10bc989ba956e7d10832f360c` | 10 | Precomputed summary only, no timestamped observations |

All columns of RR-1 and RR-6: `Controller_Time_GMT`, `Temp_degC_Ground`, `Temp_degC_ISS`, `RH_percent_Ground`, `RH_percent_ISS`, `CO2_ppm_Ground`, `CO2_ppm_ISS`.

## Telemetry descriptive statistics (ISS columns, over entire exported file)

| Metric | RR-1 | RR-6 |
|---|---:|---:|
| Rows | 12,087 | 17,572 |
| First observation GMT | 2014-09-20 03:59 | 2017-12-06 15:06 |
| Last observation GMT | 2014-11-01 03:54 | 2018-02-06 14:39 |
| Span (days) | 42.00 | 61.98 |
| Median ISS temperature (°C) | 22.6 | 22.8 |
| Median ISS humidity (%) | 43 | 41 |
| Median ISS CO₂ (ppm) | 3,192 | 3,151 |
| Mean ISS CO₂ (ppm) | 3,197.3 | 2,921.1 |
| Max ISS CO₂ (ppm) | 4,667 | 4,187 |
| Observed rows >4,000 ppm (exploratory only) | 993 (8.22%) | 95 (0.54%) |
| Missing numeric cells | 0 | 0 |
| Timestamp gaps >10 minutes | 2 | 2 |
| Largest gap (minutes) | 33 | 1392 |

**Important:** The 4,000 ppm row count above is a descriptive threshold sensitivity check, **not** a NASA-STD-3001 limit determination. No cabin-pressure conversion, hourly averaging, mission-phase filtering, or hardware/crew-habitat equivalence has been validated. Row proportions are not automatically duration-weighted exposure fractions.

## Temporal integrity flags

- **RR-1:** `CO2_ppm_ISS` equals exactly 2,646 ppm for **1,837 consecutive observations**, September 21 04:24 through September 27 13:25 GMT (6 days, 9 hours, 1 minute). Across the file, 2,646 ppm appears in 1,853 rows.
- **RR-6:** `CO2_ppm_ISS` equals exactly 500 ppm for **1,449 consecutive observations**, December 6 15:06 through December 11 15:46 GMT (5 days, 40 minutes); then exactly 3,000 ppm for **1,126 consecutive observations**, December 11 15:51 through December 15 13:40 GMT (3 days, 21 hours, 49 minutes).
- **RR-6:** Timestamp gap from January 10, 2018, 13:27 GMT until January 11, 2018, 12:39 GMT: **1,392 minutes (23.2 hours)**. The CSV has zero empty numeric cells, but this does **not** mean continuous sensor coverage.
- **RR-1:** 20-minute and 33-minute timestamp gaps occur in October 2014.
- These long constant sequences are **flags for provenance/quality review**, not proof of defective sensors. They could reflect phase-specific operational settings, processed/exported values or instrumentation; confirm with the EDA mission metadata before filtering.

## RR-9: Summary-only dataset

- CO2 ISS [PPM]: mean **3022.42**, median **3203.00**, min **400.00**, max **4051.00** (as printed in the exported summary).
- Relative Humidity ISS [%]: mean **40.77**, median **40.00**, min **33.00**, max **67.00** (as printed in the exported summary).
- Temperature ISS [C]: mean **23.55**, median **23.40**, min **20.80**, max **30.00** (as printed in the exported summary).
- Radiation Total [mGy/day]: mean **0.32**, median **0.31**, min **0.27**, max **0.55** (as printed in the exported summary).

RR-9 summary cannot support timestamp gaps, temporal event counts or persistence/recovery calculations. Obtain `RR-9_EDA_Telemetry_table.csv` before including RR-9 in mission-timeline comparisons. Radiation summary must not be treated as a timestamped dose-rate signal.

## Strongest defensible experiment question

**How do long constant-value runs, missing-time intervals and mission phase boundaries influence the environmental-event fingerprints extracted from real NASA EDA telemetry?**

Proposed lightweight method: **Quality-Gated Mission Environmental Exposure Fingerprinting (Q-MEEF)** — a *working name for this project*, not an accepted NASA technique.

1. Verify mission start/end and hardware measurement provenance (ISS versus ground; cabin versus animal hardware).
2. Flag long identical-value runs and timestamp discontinuities; distinguish unavailable/uncertain data from normal environmental conditions.
3. Compute time-aware intensity, persistence, co-occurrence and return-to-reference descriptors on supported intervals only.
4. Report event counts, durations and sensitivity with vs. without flagged segments. Never imply data quality flags automatically prove sensors were defective.
5. For later D2, overlay those **real NASA environmental states** on separate, explicitly counterfactual scenarios involving Experiment C WESAD probabilities. Never call the resulting scenarios paired astronaut validation.

## Key limitations and prerequisites

- Mission files cover different years and durations. Direct causal or controlled ISS-vs-ground / mission-vs-mission comparisons are not supported without matching contexts.
- RR-1 CSV covers September 20–November 1, 2014; provided RR-1 mission milestones include launch September 21 and return October 25. Thus the export spans prelaunch/post-return time; a simple unfiltered file average does not describe exclusively on-orbit operations.
- The EDA data originate from rodent-research mission hardware/context. Do not equate ISS-labelled hardware measurements with crew cabin exposure or astronaut stress without sensor-location/provenance evidence.
- NASA-STD-3001 requirements concern defined conditions and time averaging. Do not mark ppm readings as violations without pressure/time/habitat applicability.
- All results are descriptive and exploratory; no new ML model or medical diagnosis has been evaluated.

**Next action:** download RR-9 Telemetry CSV, confirm RR-1/RR-6 mission-phase dates and each data channel’s instrumentation/measurement location from NASA EDA reference pages, then freeze the D1 event rules before producing final experiment results.
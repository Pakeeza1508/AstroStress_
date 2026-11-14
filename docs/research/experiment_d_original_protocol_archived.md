> **ARCHIVED PROPOSAL — NOT THE CURRENT D1 IMPLEMENTATION (November 14, 2026).**
> This document preserves the original Experiment D/D2 research proposal for provenance.
> Its statements that NASA CSVs were not available, D1 had not yet been implemented,
> or RR-9 telemetry should be downloaded are historical and **no longer reflect current progress**.
> The deployed D1 dashboard now uses verified RR-1 and RR-6 telemetry and RR-9 summary-only data.
> For current findings see [D1 audit](AstroStress_Experiment_D1_initial_data_audit.md),
> [aggregate data](nasa_d1_public_summary.json), and the project README.
> Proposed future D2 scenarios are hypothetical, are **not deployed or validated**,
> and must not be described as NASA astronaut stress predictions.

---

# Experiment D — Mission Environmental Exposure Fingerprinting (MEEF)
Status: research protocol, **not implemented**. Prepared November 14, 2026. Do not report experiment results until the actual NASA CSV is acquired and analyzed.

## Objective
Study whether **duration, intensity, overlap, and return-to-reference** patterns in recorded ISS environmental telemetry can produce a transparent, useful *environmental context* for stress-monitoring decisions. This is a small, rule-based temporal analysis, not additional ML training.

MEEF is a **proposed descriptor/framework**, not a NASA-approved physiological stress metric or a proven original invention.

## Verified sources
1. NASA OSDR Environmental Data Application (EDA): https://visualization.osdr.nasa.gov/eda/
2. NASA's official EDA tutorial: https://github.com/nasa/OSDR_Tutorials/blob/main/source/pages/guides/environmental_data_application.md
3. NASA-STD-3001 Volume 2, Revision F (approved July 14, 2026): https://www.nasa.gov/reference/nasa-std-3001v2/
4. NASA standard section 6.1 [V2 6001] calls for environmental monitoring data compatible with temporal trend analysis.
5. NASA standard section 6.2.1.3 [V2 6004] states the **one-hour average CO2 partial pressure** in habitable volume must not exceed **3 mmHg**.
6. NASA standard section 6.2.3.1 [V2 6012] specifies *joint* temperature/humidity operating limits in Figure 6.2-2, not universal independent thresholds.

The RR mission records come from spaceflight **rodent research hardware and ISS/ground platforms**. Labels, sampling location and units need to be inspected before applying human-habitat limits. Instrument readings must not be presented as astronaut personal exposure, nor should hardware values be declared violations of the crew habitable-volume standard without establishing comparability.

## Source acquisition — first action
- Open the NASA EDA application.
- Select **Data Tables**; select **RR-9** as the first candidate (other RR mission acceptable if more complete).
- Export the **Telemetry** table using **CSV** (not a chart screenshot); export **Summary** as a provenance cross-check; the **Radiation** table is optional and separate.
- Preserve original filename, unchanged copy, export date, mission ID, table name and the exact EDA URL/version if displayed.
- Record column names, timestamp formats/time zone, ISS-vs-Ground labels, instrument/hardware location, units, sampling cadence, coverage and missing values.
- Do not claim there is a validated combined timeline before checking timestamp alignment and gaps.

## D1 — Pure NASA data analysis (no model)
Define **events**, not a medical diagnosis:
- INTENSITY: deviation magnitude relative to a clearly labeled, applicable reference; keep original engineering units.
- PERSISTENCE: duration of contiguous observations meeting an event rule; break at missing readings or excessive timestamp gaps.
- CO-OCCURRENCE: overlap among identifiable CO2, temperature and humidity events with compatible timestamps.
- RETURN: observed time until the parameter returns within the chosen reference range; right-censor events whose ends are unobserved.
- QUALITY: report missing-data coverage and unknown states; missing is not normal.

Do not collapse distinct hazards into an opaque 0-1 clinical score. If a research visualization needs a normalized number, label it an experimental descriptive score and report its input weights and units.

**Standards caveats**:
- CO2: EDA reports ppm; the NASA nominal standard is 1-hour *partial pressure* in mmHg. A ppm conversion requires ambient pressure and assumptions. Without valid pressure, show raw ppm and clearly labeled reference scenarios, not certified NASA compliance.
- Temperature/RH: assess as a pair against the documented NASA joint figure only when applicable and accurately digitized; otherwise descriptive trends without pass/fail claims.
- Radiation: report independently in its native unit/time basis; no universal acute-risk threshold or combination with CO2 without justification.

**D1 results to compute from actual downloaded data**:
1. Number of valid ISS readings, time coverage and data gaps.
2. Descriptive distribution for each measured variable.
3. Event count, duration distribution, peaks and persistence.
4. Multi-variable event overlaps where samples align.
5. Two or three reproducible charts: ISS conditions over time, highlighted event intervals, event fingerprint summary.
6. Comparison to Ground **only where instruments and measurement definitions are comparable**, with no causal interpretation.

## D2 — Optional link to existing Experiment C
Existing WESAD Random Forest and Experiment C remain unchanged. Use only already-saved held-out records (experiment_c_output/experiment_c_predictions.csv) in **explicitly synthetic cross-dataset scenarios** paired with real NASA environmental states. No time alignment or shared participant exists.

Counterfactual scenarios to test:
- Lower model stress estimate / no recorded environment flag.
- Higher model stress estimate / no recorded environment flag.
- Lower model stress estimate / environmental event.
- Higher model stress estimate / environmental event.
- Uncertain or unavailable physiological input / any environment state.

Output: explainable triage *recommendations*, coverage of missing-data states, proportion of scenario outputs changed by context, and disagreements. Safety-related environmental flags must not be suppressed by a low stress probability. Do not claim reduced real-world astronaut false alarms, treatment efficacy, guaranteed conformal coverage or medical autonomy. Do not tune a rule on the same simulation then claim independent validation.

## Deliverables and acceptance criteria
- Source CSV(s) retained verbatim with provenance and integrity hash.
- Clearly named cleaned analytical table; no manual invented samples.
- Reproducible D1 metrics and at least two charts based on real NASA observations.
- Small D2 scenario matrix and ablation only if D1 is done.
- Explicit limitations and 'what this does not prove' paragraph.
- Brief readable results summary suitable for AstroStress's judge-facing overview.

## Scope control
- No new ML algorithm, no model retraining, no clinical allostatic-load claim.
- Do not modify or re-run Experiments A/B/C.
- Do not change deployed production code as part of this research protocol.
- Keep the experiment isolated from the homepage/deployment work on develop-nasa-2026.

## Current blockers
The actual exported RR-9 telemetry CSV has **not yet been obtained**. No measured environmental results or NASA-derived risk flags are available yet. First obtain the CSV, then finalize event rules against its fields, sampling cadence and measurement provenance.

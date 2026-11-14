# AstroStress — Stress Detection from Wearable Physiological Data

> **NASA Space Apps Challenge 2026** | *"Create Health Monitoring Software for Astronauts on Space Missions"*

AstroStress is a stress detection prototype that uses wearable physiological sensors (ECG, EDA) to classify stress states and provide illustrative recommendations. It is trained and evaluated on real physiological data from the WESAD dataset (controlled laboratory study) and demonstrates how such a system could be adapted for astronaut health monitoring on space missions.

## Key Results

<!-- RESULTS:START -->
**Corrected WESAD evaluation (November 12, 2026).** These results were reported by the local pipeline after excluding windows containing undefined WESAD label 0 (run ID: `20261112T113826Z-ab36dc75`).

| Metric | Result |
|---|---:|
| Selected model (highest mean Leave-One-Subject-Out (LOSO) F1) | **Random Forest** |
| Mean Leave-One-Subject-Out (LOSO) accuracy | **88.74%** |
| Mean Leave-One-Subject-Out (LOSO) F1 | **78.35%** |
| Pooled ROC-AUC | **0.9436** |
| Mean Leave-One-Subject-Out (LOSO) precision / recall | 87.81% / 82.41% |
| Participants / valid windows | **15 / 608** |
| Candidate ECG/EDA features | **19** |
| Personalization (same evaluation windows) | **+2.6 percentage points** mean accuracy change |
| Participants whose personalization accuracy improved | **4 / 15** |

Four model families were compared (Random Forest, Gradient Boosting, XGBoost, LightGBM). The separate baseline Random Forest analysis in `astrostress_complete.py` reports slightly different metrics and is **not** the selected-model comparison reported above. Personalization calibration windows had a single class for every participant, so those gains are preliminary. The final model was selected by the same Leave-One-Subject-Out (LOSO) evaluation used to report its performance; the figures are not an independent test-set estimate.

The generated WESAD dataset, trained model, and research output files are **not** committed to GitHub. Reproducing the figures requires the WESAD download and both pipeline commands below.
<!-- RESULTS:END -->

The selected model is the best of the four under this Leave-One-Subject-Out (LOSO) procedure (highest mean F1). It is reported on the same Leave-One-Subject-Out (LOSO) run that selected it, so its figures are somewhat optimistic; it is not an independently validated production model.

## Submission overview (prototype scope)

**Problem:** Long missions can put crew members under physical and psychological strain. A monitoring interface should help people inspect available health signals and understand when a stress prediction may be uncertain.

**Proposed solution:** AstroStress explores ECG and EDA recordings from the WESAD laboratory dataset, uses a Random Forest model to predict stress states, shows SHAP-based explanations, and provides a separate three-question daily self-check with general wellbeing guidance. The research section also investigates whether low-confidence model predictions are more likely to be incorrect.

**Evidence:** In Leave-One-Subject-Out testing across 15 participants and 608 windows, mean participant accuracy was 88.74%, mean F1 was 78.35%, and pooled ROC-AUC was 0.9436. A separate descriptive reliability experiment found pooled accuracy of 88.65% on all held-out windows and 94.24% on the 486 windows retained at an illustrative ≥0.80 confidence cutoff; this was **not** a validated operational threshold.

**What is and is not demonstrated:** The prototype demonstrates recorded-data analysis, user-facing explanations, a manual self-assessment interface, and general guidance. The self-check answers are **not** combined with measurements from the WESAD participant. It does not monitor astronauts, assess a user's vital signs in real time, prescribe medical treatments, or establish safety for spaceflight. Further validation would be required for real missions.

## NASA challenge relevance — focused behavioral-health module

AstroStress addresses **one part** of the [2026 Create Health Monitoring Software for Astronauts on Space Missions challenge](https://www.spaceappschallenge.org/2026/challenges/health-monitoring-software/): exploring physiological stress observations, self-assessment and general guidance. It is **not** a comprehensive astronaut health monitor. Blood pressure, blood glucose, temperature, radiation exposure, bone health and other health conditions are **not** measured by this prototype. Sleep and mood questions are manual self-reports, not sensor-derived measurements.

NASA's Human Research Program identifies [five hazards of human spaceflight](https://www.nasa.gov/hrp/hazards/). The proposed relevance and limits are:

| Human spaceflight hazard | Why it matters here | What AstroStress actually demonstrates |
|---|---|---|
| **Isolation and confinement** (primary focus) | NASA describes sleep loss, work overload, behavioral-health concerns and reduced performance in long-duration isolation. | WESAD-based stress classification, explanations, uncertainty evaluation and a separate daily self-check; **not** prediction of mission mental-health outcomes. |
| **Hostile/closed environments** (secondary context) | Confined habitats can place additional strain on crews. | General rationale for monitoring stress; **no spacecraft environmental sensors or causal environment model**. |
| Space radiation, altered gravity, distance from Earth | Important hazards outside this prototype's measured scope. | No direct assessment or mitigation claim. |

**Proposed astronaut workflow:** inspect a physiological stress example → see which measurements affected the model → understand uncertainty → complete a manual self-check → read general next steps. The demo's lab participant and the person completing the self-check are **different people**; the application does not fuse them into a validated personal risk score.

**Sources:** [NASA HRP: Five Hazards](https://www.nasa.gov/hrp/hazards/), [NASA HRP: Isolation and Confinement](https://www.nasa.gov/hrp/hazard-isolation-and-confinement/), [NASA HRP: Hostile/Closed Environments](https://www.nasa.gov/hrp/hazard-hostile-closed-environments/), and the [official 2026 challenge](https://www.spaceappschallenge.org/2026/challenges/health-monitoring-software/). These NASA sources motivate the problem; **WESAD, not NASA astronaut data, supplies the evaluation recordings**.

## Development provenance and event participation

The repository includes work and commits made **before November 14, 2026**. Those commits are preserved intentionally; this project must **not** be presented as having been built entirely during the November 14–15 hackathon. The submission should clearly distinguish prior preparation from changes actually completed during the event and should follow the rules of the registered local/Universal event. The global [2026 participant terms](https://www.spaceappschallenge.org/legal/) address intellectual property, source citations and challenge relevance, while some local-event guidance requests that actual hackathon project work begin at the event start. Confirm any pre-event work eligibility question with the relevant event organizers before submission.

Third-party materials, including WESAD and software dependencies, must be cited with their applicable usage terms. The WESAD dataset is **not redistributed** in this repository. Verify its exact data-use license/permission before claiming that the dataset can be openly redistributed or that the submission meets all third-party licensing requirements.

## Features

- **Multi-Model Pipeline** — Compares Random Forest, Gradient Boosting, XGBoost and LightGBM with leakage-free Leave-One-Subject-Out (LOSO) validation
- **SHAP Explainability** — Identifies which features drive the selected model's predictions (explains model behaviour, not biological causation)
- **Personalized Detection** — Adds a short calibration period from each participant; generic and personalized models are evaluated on identical, non-overlapping test windows
- **Anomaly Detection** — Isolation Forest flags statistical outliers (the flagged fraction is set by the `contamination` parameter; flags are not validated clinical events)
- **Recommendation Engine** — 4-level protocol (Nominal → Elevated → High → Critical) with illustrative suggestions, not validated medical guidance
- **Browser mission interface** — `index.html` provides the current multi-tab UI, including real recorded-window predictions through the local `app_server.py` API, SHAP explanations, Experiment C analysis, and a manual daily check-in. The saved final model's replay is **in-sample**, not held-out validation or live monitoring.

## Dataset

Evaluated on the **WESAD (Wearable Stress and Affect Detection)** dataset:
- 15 participants (S2–S17, S1 and S12 not released) in a controlled laboratory study
- Chest ECG at 700 Hz + wrist EDA at 4 Hz (WESAD also records other signals that this model does not use)
- Binary classification: stress (TSST condition) vs non-stress (baseline, amusement, meditation)
- 120-second windows with a 60-second step (50% overlap); windows spanning two conditions are skipped
- [Dataset source](https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/)

> Schmidt, P., et al. "Introducing WESAD, a multimodal dataset for wearable stress and affect detection." ICMI 2018.

**Important:** WESAD participants are healthy adults in a lab setting. Performance on astronaut data in microgravity conditions has not been evaluated and would require separate validation.

## Methodology

- **Features (19 per window):** 11 HRV metrics (time, frequency and non-linear, via NeuroKit2), heart-rate mean and SD, and 6 EDA statistics.
- **Leave-One-Subject-Out (LOSO) evaluation:** each participant is held out in turn. Missing-value imputation (median), near-zero-variance feature removal and scaling are fitted **on the training participants only** inside every fold (`astrostress_preprocessing.py`). Earlier versions imputed and filtered features on the full dataset before cross-validation; that leakage is removed, and earlier figures should not be reused.
- **Metrics:** accuracy, precision, recall and F1 are averaged across participants; AUC is pooled over all held-out predictions.
- **Personalization:** for each participant, the first 30% of their windows (chronological) is a calibration set. The generic model is trained on the other participants; the personalized model also gets the calibration windows. Both are scored on the same later windows, and evaluation windows that share raw signal with a calibration window are removed.
- **Final model:** the selected model family is refitted on all windows and saved with its preprocessing and feature schema (`trained_model.joblib`).
- **SHAP:** computed in-sample on that final model.
- **Provenance:** every evaluation artifact carries the same `run_id`, and `summary.json` records the SHA-256 of the `features.csv` it used. The shared artifact loader rejects legacy or cross-run results and outputs built from a different `features.csv`.


## Experiment C — Prediction Reliability (separate exploratory analysis)

Run this **only after** the verified main pipeline has produced `astrostress_output/features.csv` and `summary.json`:

```powershell
python -m unittest discover -s tests
python experiment_c.py
```

This experiment refits the **same Random Forest in 15 Leave-One-Subject-Out folds**, using the existing fold-local preprocessing, to recover the held-out stress probabilities that the original pipeline calculated but did not save. It does **not** re-extract the ~2.5 GB raw WESAD dataset, modify the verified main output files, replace the final trained classifier, or change the UI.

The script checks its reproduced mean participant accuracy, mean F1 and pooled ROC-AUC against the verified prior run (allowing only four-decimal rounding) **before** writing results. A mismatch raises an error and should be investigated; it must not be ignored.

Results are written to a separate `experiment_c_output/` folder:

- `experiment_c_predictions.csv`: held-out subject, recorded window, true label, prediction, probability and confidence.
- `experiment_c_nonoverlap.csv`: timestamp-selected non-overlapping windows, used for a sensitivity check.
- `experiment_c_summary.json`: pooled Brier score, five-bin reliability data, equal-frequency ECE, descriptive confidence/coverage sweep, and participant summaries for both full and non-overlapping samples.

### Observed Experiment C results (local WESAD run, November 13, 2026)

The user-run experiment reproduced the original model-selection evaluation (mean participant accuracy **88.74%**, mean F1 **78.35%**, pooled ROC-AUC **0.9436**) and passed **79 automated tests** at that stage. On 608 held-out windows, the reported Brier score was **0.0837**, and 5-bin equal-frequency ECE was **0.0410**.

| Confidence filter | Windows accepted | Coverage | Accuracy on accepted windows | Stress F1 on accepted windows |
|---|---:|---:|---:|---:|
| No deferral (confidence ≥ 0.50) | 608 / 608 | 100% | **88.65%** | 0.768 |
| Example cutoff (confidence ≥ 0.80) | 486 / 608 | 79.93% | **94.24%** | 0.843 |

As a sensitivity check, the timestamp-selected non-overlapping subset contained 321 windows: **88.79%** accuracy with all windows, and **93.85%** accuracy with 260/321 windows accepted (81.00% coverage) at the same illustrative cutoff.

**Interpretation:** The measured curve suggests higher model confidence is associated with fewer errors on these held-out WESAD windows. **94.24% is not the model's overall accuracy**: it excludes 122 windows. The 0.80 cutoff was examined on the same held-out data and **was not separately validated or chosen for deployment**. Brier and ECE do not establish reliability on individual astronauts; correlated windows and only 15 participants limit the strength of the evidence. These result files are generated locally, not stored in this repository.

After Experiment C finishes, restart `python app_server.py` and open the **Test Results** tab. An optional section called **Can the AI tell when it might be wrong?** will show the held-out accuracy-versus-coverage curve and the five-bin calibration plot. The local server checks the experiment's run ID, original features hash and sample counts; if no matching experiment exists, the tab shows a missing-results message instead of invented figures. The existing saved-model demo remains unchanged.

This is a **descriptive reliability study**, not a validated automated uncertainty alert. Thresholds are shown as a curve and are not tuned on the same held-out results to claim a validated operating point. The final demo still uses a separately trained all-windows, in-sample saved model. Overlapping windows, the small number of participants and model selection on the same Leave-One-Subject-Out evaluation limit the conclusions. The Experiment C output must be inspected before adding any claims to the UI.

## Project Structure

```
astrostress/
├── astrostress_complete.py       # Feature extraction from raw WESAD .pkl files (+ RF Leave-One-Subject-Out (LOSO) report and plots)
├── astrostress_v2.py             # Evaluation: model comparison, personalization, final model, SHAP, anomalies
├── astrostress_preprocessing.py  # Shared fold-local preprocessing and calibration/evaluation split
├── astrostress_outputs.py        # Output schemas, run provenance and validated artifact loading
├── index.html                    # Approved HTML/CSS/JS UI, with optional local model API
├── app_server.py                 # Local-only verified WESAD inference and research endpoints
├── astrostress_signals.py        # Read-only original ECG/EDA waveform slices
├── astrostress_inference.py       # Provenance-checked recorded-window inference
├── prepare_dashboard_data.py     # Read-only validation of generated artifacts
├── requirements.txt              # Python dependencies
├── tests/                        # Automated tests (synthetic data; no WESAD needed)
└── astrostress_output/           # Generated results (not committed)
    ├── features.csv
    ├── feature_extraction.json
    ├── model_comparison.csv
    ├── loso_results.csv
    ├── personalized_results.csv
    ├── shap_analysis.json
    ├── anomaly_results.csv
    ├── recommendations.json
    ├── summary.json
    └── trained_model.joblib
```

## Reproduce the results

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the WESAD dataset

Download from the [WESAD dataset page](https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/) and extract to a `WESAD/` folder (`WESAD/S2/S2.pkl`, …, `WESAD/S17/S17.pkl`, about 13 GB of `.pkl` files).

### 3. Extract features

```bash
python astrostress_complete.py --data_dir WESAD
```

Extracts the 19 features into `astrostress_output/features.csv` and prints a Random Forest Leave-One-Subject-Out (LOSO) report with plots. Takes roughly 30–40 minutes.

### 4. Run the evaluation

```bash
python astrostress_v2.py
```

Runs the model comparison, fair personalization, final model, SHAP and anomaly detection, and writes all research artifacts with one `run_id`. Uses the saved features, so the raw data is not reprocessed.

### 5. Check the outputs (optional)

```bash
python prepare_dashboard_data.py
```

Read-only: reports whether every artifact is a current, verified pipeline output.

### 6. Open the approved interface

**Visual-only preview:** Double-click `index.html` in Chrome or Edge. No build step. Real recorded waveforms and model predictions are unavailable without the local server and trusted WESAD files; the interface does not invent substitute signals.

**Connected research demo:** Once you have already run the analysis pipeline and have the locally generated `astrostress_output/` directory (including `features.csv`, `summary.json`, `trained_model.joblib` and evaluation CSVs):

```bash
python app_server.py
```

Open **http://127.0.0.1:8000/**. The local-only Python server:

- Loads the current trusted saved model **only after** validating run IDs and input feature hashes.
- Populates recorded WESAD participants and their actual available window counts.
- Replays real per-window Random Forest stress probabilities and protocol labels from `features.csv`.
- Shows per-window SHAP contributions computed on the saved final model.
- Fills model-comparison, per-subject Leave-One-Subject-Out (LOSO) and personalization charts from verified local evaluation CSVs. Leave-One-Subject-Out (LOSO) per-subject AUC is not in the saved CSV and is therefore labelled "Not reported" rather than invented.
- Can display **original WESAD ECG and EDA samples** (not generated waveforms) for the selected participant/window **when the trusted raw WESAD files are available locally**. ECG displays a 10-second excerpt at its original 700 Hz resolution; EDA displays the full 120-second window at 4 Hz.

**Interpretation limits:** The saved model was refitted on **all** available WESAD feature windows. A replay is in-sample, *not* a held-out Leave-One-Subject-Out (LOSO) prediction or astronaut validation. Research cross-validation metrics are separate. ECG/EDA waveform plots are only shown if their original raw recordings can be read from a local trusted WESAD folder or ZIP. Without those files, the plot area reports that recordings are unavailable instead of inventing data. Opening the HTML directly cannot load local Python model artifacts. Nothing is uploaded to the frontend.

### Original ECG/EDA signal source (optional for prediction, required for raw plots)

If you already downloaded the ~2.5 GB WESAD archive, **do not download it again**. If you extracted it into `WESAD/S10/S10.pkl` (and similarly for the other subjects) in your project directory, the signal explorer detects it automatically.

Alternatively, set **one** path before running the local server:

```powershell
# Option A — point to your extracted WESAD folder:
$env:WESAD_DATA_DIR = "D:\\Datasets\\WESAD"

# Option B — if you only have the downloaded official ZIP:
# $env:WESAD_ZIP_PATH = "D:\\Downloads\\WESAD.zip"

python app_server.py
```

Use your actual Windows path. Loading a subject from ZIP can use considerable memory; extracted files are generally preferable for repeated exploration. Never load an unknown/untrusted pickle or ZIP. The raw traces stay local and are not uploaded. The Physiology page shows a clear missing-source message if these files cannot be found.

The standard-library HTTP server listens on **127.0.0.1 only**, and only supports GET endpoints. Only load `trained_model.joblib` files you produced locally and trust; joblib deserialization is unsafe for untrusted files.

### Run the tests

```bash
python -m unittest discover -s tests
```

Python tests use synthetic data. The standalone `index.html` needs no frontend build.

## How It Works

1. **Sensor Data** → ECG and EDA from wearable devices
2. **Feature Extraction** → 19 physiological features (HRV time/frequency/nonlinear + EDA statistics) in 120-second windows
3. **Classification** → Best of four models selected via leakage-free Leave-One-Subject-Out (LOSO) cross-validation
4. **SHAP Analysis** → Explains which features influence the model's predictions
5. **Anomaly Detection** → Isolation Forest identifies statistical outliers
6. **Recommendations** → 4-level protocol with illustrative actions

## Limitations

- **Lab data only:** Evaluated on WESAD (controlled lab study, 15 healthy adults), not on astronaut data in microgravity
- **No real-time inference:** Current version is an offline analysis pipeline; real-time monitoring would require additional engineering
- **Optimistic model selection:** The best model is chosen and reported on the same Leave-One-Subject-Out (LOSO) run; there is no separate held-out test set
- **Overlapping windows:** 50% window overlap means Leave-One-Subject-Out (LOSO) windows are not independent observations
- **Anomaly detection threshold:** The 10% anomaly rate reflects the configured `contamination` parameter, not an independently discovered event rate
- **Personalization:** Results are preliminary — calibration class diversity varies by subject due to the WESAD experimental protocol order
- **SHAP:** In-sample, on the final model; it describes model behaviour, not physiological causes
- **No signal quality gating:** ECG windows are processed without explicit noise or artifact quality checks
- **EDA peak detection:** Uses simple local-maxima counting rather than validated SCR decomposition
- **Browser demo:** Directly opened HTML is a visual-only prototype. With the local server running, recorded-window model replay and verified evaluation results are real. ECG/EDA waveform plots require access to original trusted local WESAD recordings; no synthetic substitute traces are drawn. Neither mode uses live astronaut sensors.

## AI Tools Disclosure

This project was developed with the assistance of AI tools (including Claude by Anthropic and ChatGPT by OpenAI) for code generation, debugging, pipeline design, and documentation. All methodology was designed collaboratively, and reported results come from running the pipeline on the WESAD dataset.

## License

[Apache License 2.0](LICENSE)
#   A s t r o S t r e s s _  
 
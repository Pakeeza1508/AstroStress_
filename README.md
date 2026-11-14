# AstroStress — Explainable Stress Research and NASA Environmental Data Explorer

**NASA Space Apps Challenge 2026 · Health-monitoring software concept**  
**Status:** Public, research-focused MVP — **not** a clinical device, astronaut monitor, or flight-qualified system.

[**Launch the live MVP**](https://astrostress-nasa-2026-13af09n01-pakeeza-khalids-projects.vercel.app) · [NASA Space Apps challenge](https://www.spaceappschallenge.org/2026/challenges/health-monitoring-software/) · [Deployment instructions](DEPLOY_VERCEL.md)

> **Two independent research sources, not one paired dataset.** WESAD contains Earth-based laboratory ECG and electrodermal activity (EDA) recordings used to evaluate a stress classifier. NASA Environmental Data Application (EDA) exports contain environmental readings from **rodent-research hardware**, not astronaut physiology. AstroStress presents the two streams separately; it **does not** establish a relationship between NASA hardware readings and human stress.

## What the project demonstrates

AstroStress explores how an explainable, uncertainty-aware physiological stress model *might inform* future crew-wellbeing software. The current web demo combines:

- **WESAD classification research:** comparison of Random Forest, Gradient Boosting, XGBoost and LightGBM using participant-held-out evaluation.
- **Model explainability:** global SHAP feature importance, with attribution interpreted as model behaviour rather than causation.
- **Experiment C — reliability:** accuracy versus confidence/coverage on held-out WESAD windows, including a non-overlapping-window sensitivity check.
- **Daily Check-in:** a fully interactive, browser-side questionnaire about perceived stress, sleep and wellbeing. Advice is general and based on **self-reported answers only**; these answers are not transmitted to the model or combined with laboratory records.
- **Experiment D1 — NASA EDA:** interactive descriptive charts and quality flags for RR-1 and RR-6 environmental hardware exports. RR-9 is presented as summary-only.

The application does **not** predict stress in astronauts, read live wearables, detect illness, assess NASA crew-cabin safety, establish stress caused by CO₂, or offer medical diagnosis or treatment.

## Try the public demo

**Latest tested Preview:** https://astrostress-nasa-2026-13af09n01-pakeeza-khalids-projects.vercel.app

The public UI contains six original pages plus the **NASA EDA** research tab:

| Tab | What visitors can actually do |
|---|---|
| **Overview** | Understand the project scope and navigate to verified research results. |
| **Physiological Data** | Read the WESAD signal-source explanation and navigate to public research summaries. Raw ECG/EDA traces and individual features are **not** published. |
| **AI Analysis** | Select among 15 WESAD participants and inspect **aggregate, held-out, participant-level evaluation**. This is *not* an individual-window prediction. |
| **Daily Check-in** | Answer three self-report questions, receive general next steps, then clear the answers. |
| **Explainability** | Explore the verified **global** SHAP feature-importance chart. |
| **Test Results** | View WESAD model comparison, LOSO research results and Experiment C confidence/reliability plots. |
| **NASA EDA** | Switch RR-1/RR-6 and CO₂/temperature/humidity to inspect weekly medians, observation counts, gaps and flat-sensor cautions. |

**Cold-start caveat:** a fresh Vercel Preview visit has sometimes taken **19–31 seconds** to initialize; subsequent warm visits were faster in recorded checks. The UI says it is connecting to verified research data during startup. This is a current operational limitation, not a guarantee of load time. Prepare the tab before a live demonstration.

### Most recently reported deployment verification

The D1-integrated Preview was built from commit **`66921b593e34b9d637da4dd714e8f6ee0fc8c0c9`** on **`develop-nasa-2026`**. The recorded verification report found **106/106 automated tests** and **143/143 browser checks** passing on desktop and mobile, with no app JavaScript errors. These numbers describe that specific deployment/check run, not an independent external certification. The `main` branch was not changed.

## Verified WESAD model research

**Dataset:** [WESAD — Wearable Stress and Affect Detection](https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/), Schmidt et al., *ICMI 2018*. Fifteen published lab participants, with chest ECG at **700 Hz** and wrist EDA at **4 Hz**. These are **Earth-based lab recordings**, not astronaut data.

The pipeline uses **120-second windows**, a **60-second step**, and **19 candidate ECG/EDA features**. Windows with undefined WESAD label 0 were excluded from the corrected comparison. Stress corresponds to the experimental TSST condition; baseline, amusement and meditation provide non-stress observations. Windows crossing labelled conditions are skipped.

**Selected model:** Random Forest, chosen using mean participant Leave-One-Subject-Out (LOSO) F1 from four compared model families. Verified run: `20261112T113826Z-ab36dc75` (corrected November 12, 2026).

| Metric | Reported result |
|---|---:|
| Participants / valid windows | **15 / 608** |
| Mean participant LOSO accuracy | **88.74%** |
| Mean participant LOSO F1 | **78.35%** |
| Mean participant LOSO precision / recall | **87.81% / 82.41%** |
| Pooled ROC-AUC | **0.9436** |
| Candidate physiological features | **19** |

### Model evaluation methodology

1. Extract ECG/EDA-derived physiological features from trusted WESAD recordings.
2. Leave **one complete participant** out in each evaluation fold to avoid subject overlap between training and evaluation participants.
3. Fit median imputation, near-zero-variance filtering and feature scaling **within each training fold**. This corrects earlier preprocessing leakage.
4. Compare Random Forest, Gradient Boosting, XGBoost and LightGBM using the same LOSO procedure.
5. Refit the selected model on all available windows for **local, in-sample recorded-window replay** and SHAP analysis; that replay must never be described as an independent held-out test.

**Interpretation:** The selected model is also reported on the LOSO run used for model selection, making performance potentially optimistic. Fifty-percent-overlapping windows are correlated. Fifteen participants do not establish clinical or spaceflight generalization. SHAP reveals associations within the model, not biological causes.

**Exploratory personalization:** a chronological initial calibration block is used with the remaining later windows for evaluation, excluding overlapping raw-signal intervals. A reported **+2.6 percentage-point** mean accuracy change occurred on the same evaluation windows, with **4/15** participants improving. Calibration windows had limited class diversity; personalization is not a validated astronaut adaptation.

## Experiment C — Prediction reliability

Experiment C refits the same Random Forest LOSO folds with fold-local preprocessing and extracts **held-out** probabilities. It does not reprocess all original WESAD raw recordings, change the main model artifacts, or substitute for a separately validated risk threshold.

| Measure | Reported result |
|---|---:|
| Held-out windows | **608** |
| Pooled Brier score | **0.0837** |
| Five-bin equal-frequency ECE | **0.0410** |
| Accuracy without confidence deferral | **88.65%** (608/608 windows) |
| Accuracy with illustrative confidence ≥0.80 | **94.24%** (486/608 windows; **79.93% coverage**) |
| Non-overlapping sensitivity subset | **321 windows** |
| Non-overlapping accuracy, all windows | **88.79%** |
| Non-overlapping accuracy, ≥0.80 confidence | **93.85%** (260/321; **81.00% coverage**) |

**Do not say the model has 94.24% overall accuracy.** That figure excludes **122** lower-confidence windows. The 0.80 value was examined on the same evaluation data, was **not independently validated**, and is **not** deployed as an astronaut safety rule. This experiment studies confidence and selective prediction, not medical reliability.

To reproduce after generating and validating the main `astrostress_output/` artifacts:

```bash
python -m unittest discover -s tests
python experiment_c.py
```

Generated `experiment_c_output/` files remain local. The server checks run provenance and the relevant inputs before serving public aggregate results.

## Experiment D1 — NASA environmental telemetry

D1 is a **separate exploratory dataset audit** using the original NASA Environmental Data Application CSV exports for RR-1, RR-6 and RR-9. The sensors belong to **rodent-research mission hardware**. An "ISS" channel label does not prove the sensor was a crew-cabin or human-exposure instrument. D1 data are **not paired** with WESAD participants.

| Source | File type | Rows | ISS-channel median CO₂ | ISS-channel median temperature | ISS-channel median RH |
|---|---|---:|---:|---:|---:|
| RR-1 | Telemetry | **12,087** | **3,192 ppm** | **22.6 °C** | **43%** |
| RR-6 | Telemetry | **17,572** | **3,151 ppm** | **22.8 °C** | **41%** |
| RR-9 | Precomputed summary | **10** | Summary only | Summary only | Summary only |

### Time integrity and exploratory CO₂ flags

| Check | RR-1 | RR-6 |
|---|---:|---:|
| Observations strictly **>4,000 ppm** | **993** | **95** |
| Observations **≥4,000 ppm** | **996** | **96** |
| Timestamp gaps **>10 minutes** | **2** | **2** |
| Largest gap | **33 min** | **1,392 min** |
| Intervals shorter than nominal five minutes | **6** | **12** |

**These are row counts, not duration-weighted exposures or NASA-STD-3001 violations.** The 4,000 ppm cutoff is for exploratory sensitivity comparison only.

The interface plots **weekly medians of sampled ISS-channel temperature, relative humidity and CO₂**. The bins and values are descriptive; they are not adjusted for missing time or verified launch/return phases. The first plotted RR-6 week contains a conspicuous **1,449-row** repeated pattern of 500 ppm CO₂, 23.0 °C and 50% humidity (1,449/2,016 observations). RR-1 has a smaller **314/2,016** simultaneous flat-reading interval in its first week. A dashed chart marker and explanatory caution flag these periods; the underlying medians are **not removed or altered**. Identical readings could reflect processing, operational phases or instrumentation and are **not proof of sensor failure**.

**RR-9:** the supplied file is summary-only and cannot support a timestamped chart, temporal events or missing-time analysis. Its accumulated-radiation field has an ambiguous `mGy/day` label, so no dose-rate or astronaut-exposure inference is made.

Source integrity is documented via SHA-256 hashes and descriptive audits in:

- [D1 original telemetry audit](docs/research/AstroStress_Experiment_D1_initial_data_audit.md)
- [D1 public aggregated summary and flat-run annotations](docs/research/nasa_d1_public_summary.json)

No causal claim between environmental readings and human stress is supported. The working label **Q-MEEF (Quality-Gated Mission Environmental Exposure Fingerprinting)** describes a proposed quality-analysis approach, not a NASA-established method or a completed validated risk model.

## Architecture and code organization

```text
WESAD trusted local files ──> astrostress_complete.py ──> 19-feature windows
                                                           │
                                              astrostress_v2.py
                                               (LOSO + SHAP)
                                                           │
                                              verified artifacts
                                                           │
                                    experiment_c.py ──> reliability outputs
                                                           │
                         app_server.py (trusted local) / app.py (Vercel)
                                                           │
NASA EDA CSV audit ──> verified aggregate D1 summary ───────┤
                                                           │
                                                     index.html
                                      public research dashboard + self-check
```

Useful entry points:

| File or folder | Role |
|---|---|
| `index.html` | Browser UI, charts and manual Daily Check-in |
| `app.py` | Flask/Vercel HTTP adapter |
| `app_server.py` | Trusted local standard-library HTTP server |
| `astrostress_complete.py` | Local WESAD feature extraction |
| `astrostress_v2.py` | Model evaluation and verified research artifact generation |
| `astrostress_preprocessing.py` | Fold-local preprocessing and calibration splits |
| `astrostress_outputs.py` | Artifact schema and provenance validation |
| `astrostress_inference.py` | Trusted local recorded-window inference |
| `astrostress_signals.py` | Trusted local raw ECG/EDA signal slicing |
| `experiment_c.py` | Held-out prediction reliability research |
| `docs/research/nasa_d1_public_summary.json` | Versioned D1 aggregate research values |
| `tests/` | Synthetic-data and UI/API regression tests |
| `DEPLOY_VERCEL.md` | Verified-bundle Preview deployment procedure |

A separate, non-deployed development branch `feature/nasa-d1-offline` contains a provisional offline D1 engine and analysis rules; the published D1 dashboard on `develop-nasa-2026` uses the verified **aggregate JSON**, not those raw CSVs.

## Run locally and reproduce research

**Recommended environment for the saved model:** Python **3.12** and scikit-learn **1.9.1**. The Vercel `requirements.txt` is intentionally inference-only; local training additionally needs the packages imported by the research scripts (for example NeuroKit2, SHAP, XGBoost and LightGBM). A clean research environment may need those installed separately.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests
```

To regenerate WESAD results, obtain the dataset from the official publisher and use trusted, locally extracted subject files arranged like `WESAD/S2/S2.pkl`. Do **not** deserialize untrusted pickle/joblib files.

```bash
python astrostress_complete.py --data_dir WESAD
python astrostress_v2.py
python prepare_dashboard_data.py
python experiment_c.py
python app_server.py
```

Then open **http://127.0.0.1:8000/**. Locally generated model and experiment files are intentionally absent from GitHub. Reproduction of all figures requires the original licensed data, additional research dependencies and the trusted analysis environment; a fresh source checkout alone is not sufficient.

**Local versus public:** a trusted local server with permitted WESAD files can show recorded-window predictions, per-window explanations and raw waveforms where the original recordings are present. On the **public** deployment, those individual records are deliberately restricted; the AI Analysis selector shows **participant-level evaluation** instead.

## Data access, deployment and security

WESAD is supplied by its original researchers for scientific/non-commercial use with attribution. Its terms do not clearly authorize public redistribution of derived individual-window records. Accordingly:

- Original WESAD files, generated `features.csv`, labels, joblib models and per-window explanations are **not committed** to GitHub or offered as public downloads.
- The authenticated deployment build obtains the required artifact ZIP privately; its checksum is pinned to **`734c471c96f20584e74fc5842b3845e8f44f5365a61f03c01b91088c7ee57551`**. A failed checksum blocks deployment.
- Public `/api/window`, `/api/explain` and `/api/physiology-trend` calls return **403 `RESTRICTED_DATA`** by default. Do not enable `ASTROSTRESS_ALLOW_PUBLIC_WESAD_WINDOWS=1` without documented permission.
- Public aggregate research endpoints include `/api/research`, `/api/global-shap`, `/api/reliability` and `/api/status`. The source NASA D1 CSVs and artifact ZIP are not downloadable via the public website.
- The deployed D1 research tab embeds aggregated, source-checked descriptive values; it does not expose the original EDA tables through a file endpoint.
- The deployment is a manually created **Vercel Preview** under project `astrostress-nasa-2026`. It is **not** a Production deployment or an automatically synchronized Git-connected site.
- The Vercel server bundle was last reported at approximately **239.65 MB** versus the **250 MB** function limit; avoid unnecessary backend dependencies.
- This project is distinct from the older `astrostress` Vercel project. The older site has been reported to expose WESAD window records and needs separate owner-side access restriction/retirement.

Detailed reproduction/deployment steps are in [DEPLOY_VERCEL.md](DEPLOY_VERCEL.md).

## Main limitations

- No astronaut, ISS crew medical or prospective human-subject dataset was used to validate stress inference.
- WESAD is a relatively small Earth-laboratory dataset; session protocol and correlated overlapping windows constrain generalization.
- The selected model's reported LOSO evaluation was used for model selection; it is not an untouched final external test set.
- Confidence-based selective accuracy trades coverage for apparent accuracy and has no validated clinical/mission operating threshold.
- In-sample SHAP and model replay do not prove physiology causation or operational performance.
- Manual self-check guidance is not medical advice and does not use or modify a person's ECG, EDA or model predictions.
- NASA RR-1/RR-6 are hardware environmental datasets from different years and phases, not matched controlled experiments or validated crew-cabin exposures.
- NASA D1 charts include flagged long constant-value runs, missing intervals and potential pre-/post-flight observations without verified phase filtering.
- RR-9 is a summary, not a time series. No astronaut radiation-dose claim is made.
- The first public visit can be slow because of server startup; public WESAD individual-window features are intentionally absent.

## NASA challenge framing and provenance

AstroStress addresses the **behavioral/wellbeing and explainability** aspect of the proposed health-monitoring challenge, not every requested biomedical capability. NASA's [five hazards of human spaceflight](https://www.nasa.gov/hrp/hazards/) provide motivation, especially isolation and confinement. Neither NASA's hazard descriptions nor EDA hardware measurements validate the WESAD model on astronauts.

Development in this repository **began before November 14, 2026**. This must be disclosed accurately in any event submission; do not present the entire project as built during the hackathon. Verify event-specific pre-existing-work eligibility with the organizers.

### References and acknowledgements

- Schmidt, P., Reiss, A., Duerichen, R., Marberger, C., & Van Laerhoven, K. (2018). *Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection*. ICMI 2018. [Publisher dataset page](https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/).
- [NASA Space Apps Challenge 2026 — Health Monitoring Software](https://www.spaceappschallenge.org/2026/challenges/health-monitoring-software/).
- [NASA Human Research Program — Five Hazards of Human Spaceflight](https://www.nasa.gov/hrp/hazards/).
- NASA Environmental Data Application RR-1, RR-6 and RR-9 exports; filenames, integrity hashes and quality audit are documented in [D1 research notes](docs/research/AstroStress_Experiment_D1_initial_data_audit.md).
- Open-source libraries used for analysis, visualization and serving are credited via project dependency files and their respective licenses.

### AI-assisted development

Claude (Anthropic) and ChatGPT (OpenAI) assisted with coding, debugging, analysis design, QA and documentation. Scientific values are from the verified experiment artifacts and original D1 exports, not synthetic demo measurements.

## License

Project **source code**: [Apache License 2.0](LICENSE). Third-party datasets, derived data, research papers and model artifacts are **not automatically licensed** under that source-code license; follow the original providers' terms.

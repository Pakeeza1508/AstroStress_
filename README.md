# AstroStress — Explainable AI for Stress Research

A NASA Space Apps Challenge 2026 research prototype exploring physiological stress detection and independent space-environment monitoring.

**[Live Demo (Vercel Preview)](https://astrostress-nasa-2026-33ym1m8wr-pakeeza-khalids-projects.vercel.app/)** · **[2026 NASA Space Apps Challenge](https://www.spaceappschallenge.org/2026/challenges/health-monitoring-software/)**

## Project overview

AstroStress evaluates machine learning on wearable heart and skin signals from the [WESAD dataset](https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html). A separate module explores CO₂, temperature, humidity and data quality from [NASA's Environmental Data Application](https://visualization.osdr.nasa.gov/eda/), using RR-1 and RR-6 environmental telemetry. These two datasets are **not paired or fused**.

## Explore the demo

| Tab | What it shows |
|---|---|
| **Overview** | Project concept and research workflow |
| **Datasets Used** | Research data sources |
| **AI Stress Detection** | Physiological features, 15-participant evaluation and SHAP explainability |
| **Daily Check-in** | Mission-focused, self-reported readiness assessment and example scenarios |
| **Test Results** | Model comparison, participant results and prediction-confidence analysis |
| **Space Environment Monitoring** | RR-1/RR-6 CO₂, temperature and humidity trends, plus data-quality findings |

The Daily Check-in is a **scenario-based demonstration**, not an inference from the trained WESAD model.

## Research results

Using 15 WESAD participants and Leave-One-Subject-Out evaluation, the selected **Random Forest** achieved:

| Metric | Value |
|---|---:|
| Mean accuracy | **88.74%** |
| Mean Stress F1 | **78.35%** |
| Pooled ROC-AUC | **0.9436** |
| Evaluated 120-second windows | **608** |
| Candidate physiological features | **19** |

The project also evaluates prediction-confidence/coverage trade-offs and uses SHAP to explain feature importance. These are experimental research results, not validated medical decisions.

## Run locally

Install dependencies and start the trusted local research server:

```bash
python -m pip install -r requirements.txt
python app_server.py
```

Open `http://127.0.0.1:8000/`.

The verified model/research artifacts must be available locally for research endpoints to work. For further technical information, see the repository's Python research scripts and [deployment notes](DEPLOY_VERCEL.md) (these notes may describe an earlier branch).

## Scope and sources

- **Physiological research:** WESAD laboratory recordings, not astronaut measurements. Individual recorded WESAD windows are not exposed in the public demo.
- **NASA environmental research:** RR-1/RR-6 environmental telemetry, analyzed separately from physiological stress.
- **Current scope:** Research prototype only; no live wearable monitoring, astronaut stress validation, deployed physiological/environmental data fusion or clinical decision-making.
- **Pre-existing work:** Development began before the November 2026 hackathon; the repository includes earlier research and subsequent demo work.

**Sources:** [WESAD](https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html) · [NASA Environmental Data Application](https://visualization.osdr.nasa.gov/eda/) · [NASA Space Apps challenge](https://www.spaceappschallenge.org/2026/challenges/health-monitoring-software/)

**License:** [Apache-2.0](LICENSE) for project source code; third-party datasets retain their own terms.

# AstroStress

**[Live Demo](https://astrostress-nasa-2026-33ym1m8wr-pakeeza-khalids-projects.vercel.app/)**

AstroStress explores explainable AI for physiological stress detection, alongside independent analysis of NASA space-environment telemetry.

## How AstroStress Works

The project studies two separate data sources. **WESAD** provides ECG and skin-conductance recordings for training and evaluating a Random Forest stress classifier. We use Leave-One-Subject-Out testing, SHAP explanations and confidence analysis to understand model performance. **NASA RR-1/RR-6** telemetry provides CO₂, temperature and humidity readings for environmental trend and data-quality analysis. The data sources are displayed together, but have **not been combined into one prediction model**.

The web demo also includes a **Daily Check-in** for self-reported mission-readiness scenarios. This runs separately from the trained sensor-based classifier. Performance charts and dataset details are available directly in the demo.

## Research Experiments

The research is organized into three complementary studies, summarized here; the demo contains the full charts and participant-level results.

**1. Stress detection and model evaluation.** We extracted 19 heart- and skin-signal features from 608 WESAD windows across 15 participants. Four machine-learning models were compared using Leave-One-Subject-Out testing, so each person was evaluated without being included in that fold's training data. The selected Random Forest achieved **88.74% mean accuracy**, **78.35% mean Stress F1** and **0.9436 pooled ROC-AUC**. SHAP analysis shows which features most influenced the trained model.

**2. Prediction confidence and reliability.** We examined how accuracy changes when lower-confidence classifications are set aside. On the held-out evaluation windows, accuracy was **88.65%** overall; it was **94.24%** on the **486 of 608** windows meeting an illustrative 80% confidence cutoff. This is a trade-off between accuracy and coverage, not a higher overall accuracy or a validated safety threshold.

**3. NASA environmental data analysis.** We independently examined RR-1 (**12,087 readings**) and RR-6 (**17,572 readings**) environmental telemetry, including CO₂, temperature, humidity, weekly trends and recording-quality flags. These NASA measurements are **not paired with WESAD stress recordings**, and no combined prediction model has been implemented.

## System Architecture

![AstroStress system architecture](docs/astrostress_architecture.svg)

## Run Locally

Use Python 3.12 and a trusted local copy of the verified research artifacts.

```bash
git clone https://github.com/Pakeeza1508/AstroStress_.git
cd AstroStress_
python -m pip install -r requirements.txt
python app_server.py
```

Open **http://127.0.0.1:8000/**.

The server expects the generated `astrostress_output/` research/model artifacts. To reproduce them, download WESAD from the [dataset source](https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html) and follow the feature-extraction and evaluation scripts (`astrostress_complete.py` and `astrostress_v2.py`). Research training may require additional Python packages beyond the deployment requirements.

## Team

- **Pakeeza Khalid**
- **Afifa Khalid**

## References

- [WESAD — official dataset page](https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html)
- [WESAD — original dataset download](https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx)
- [NASA OSDR — Environmental Data Application](https://visualization.osdr.nasa.gov/eda/)
- [NASA OSDR — environmental data guide](https://github.com/nasa/OSDR_Tutorials/blob/main/source/pages/guides/environmental_data_application.md)
- [NASA Space Apps 2026 — Health Monitoring Software challenge](https://www.spaceappschallenge.org/2026/challenges/health-monitoring-software/)

**License:** [Apache 2.0](LICENSE) for source code; external datasets follow their original terms.

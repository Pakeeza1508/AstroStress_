# AstroStress — isolated Vercel deployment test

Use the `vercel-deployment` branch. Never change the production `main` branch for this test.

- Import `Pakeeza1508/astroStress` with root directory `./`.
- Set **Framework Preset** to Python (Flask detection) and keep the default build/output settings.
- Deploy the `vercel-deployment` branch as a Preview deployment. If the Vercel project is configured to produce Production from `main`, Preview deployment must target this new branch explicitly.
- Vercel reads the inference-only dependencies in `pyproject.toml` (and `requirements.txt`) and the Python 3.12 constraint. It executes `[tool.vercel.scripts].build` to download the exact release bundle, validated by SHA256.
- The root `app.py` exports a Flask `app`, serves unchanged `index.html` and exposes the same-origin `/api/*` endpoints. There are no frontend code changes.
- Public mode deliberately blocks raw WESAD waveform loading. Saved-model window predictions are **in-sample replays**, not held-out validation or real astronaut measurements.

After deployment open `/`, `/healthz`, `/api/status`, `/api/subjects`, `/api/research`, `/api/global-shap`, `/api/reliability`, `/api/window?subject=S10&index=0`, and `/api/explain?subject=S10&index=0`.

If a Vercel build fails, inspect the Build Logs. In particular, confirm that the Vercel build hook runs and that the validated `astrostress_output` files are included in the Python function package. Do not ship without successful `/api/status` and `/api/window` tests.

The saved model was built with scikit-learn 1.9.1. Never downgrade the Python runtime to a scikit-learn version that raises `InconsistentVersionWarning`.

No secrets should be pasted into code or deployment logs.

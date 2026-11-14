# AstroStress: Render Free deployment

Research prototype only. The public app uses prebuilt WESAD-derived features and model outputs, not astronaut readings or raw 17 GB recordings.

## Step 1: Prepare local artifacts (PowerShell)

```powershell
git checkout main
git pull origin main
python -m unittest discover -s tests
python scripts/build_window_shap.py
python scripts/prepare_render_bundle.py
```

The final command creates gitignored astrostress-render-artifacts.zip and prints a SHA256 digest. It contains the trusted saved model, features, verified summary, evaluation tables, global SHAP, precomputed per-window SHAP and Experiment C summary. It does not include original raw WESAD recordings. Never load joblib from an untrusted source.

## Step 2: Host the artifact bundle

Upload the ZIP only to a stable, direct HTTPS download URL at a storage provider you have permission to use. A GitHub Release asset is an example but becomes publicly accessible on a public repository. Check the WESAD terms and privacy implications before publishing derived features or model weights. Do not publish without authorizing disclosure. HTML share pages and expiring links are unsupported. Record the direct URL and exact printed SHA256; do not place passwords in URLs.

## Step 3: Render settings

- New Web Service; connect Pakeeza1508/astroStress.
- Branch: main; Language: Python 3; Root directory: blank.
- Compute: Free $0, not the $7 paid tier.
- Python runtime: .python-version targets Python 3.11.
- Build Command: `pip install -r requirements-render.txt && python scripts/fetch_render_bundle.py`
- Start Command: `python app_server.py`
- Environment: `ASTROSTRESS_PUBLIC` = `1`
- Environment: `ASTROSTRESS_ARTIFACT_URL` = approved direct HTTPS ZIP URL
- Environment: `ASTROSTRESS_ARTIFACT_SHA256` = 64-character digest printed locally

At build time the downloader checks HTTPS, SHA256, file whitelist and size. At startup the server checks saved model and research provenance. Do not skip verification when deployment fails. The 512 MB Free instance may still exhaust RAM; review logs before considering alternatives.

## Step 4: Check the public Render URL

- `/healthz` should return ok: true.
- `/api/status` should report Random Forest and the original run ID.
- `/api/reliability` should report 608 windows and the measured Brier/ECE.
- `/api/global-shap` should return verified saved global feature importance.
- `/api/window?subject=S10&index=9` should return an in-sample recording replay.
- `/api/explain?subject=S10&index=9` should return precomputed real model attributions.
- `/` serves index.html. Test the full site on Render first.

Raw ECG/EDA waveform requests `/api/signals` are disabled on the public backend, so the page must show a missing-source message, not fabricated measurements. A local server with original trusted WESAD files still serves the real waveforms.

## Step 5: Optional Vercel

Once the Render site works, deploy index.html as static frontend to Vercel. Use a same-origin /api proxy rewrite pointing at your confirmed Render HTTPS domain. Do not expose localhost or unapproved backends. Test Vercel separately. Free Render instances can spin down after inactivity, so first responses may take longer.

Keep pre-hackathon commits and honest model provenance; do not claim operational astronaut monitoring, validated confidence thresholds or clinical treatments.
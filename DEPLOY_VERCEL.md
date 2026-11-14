# AstroStress — Vercel deployment (NASA Space Apps 2026 MVP)

- GitHub repository: `Pakeeza1508/AstroStress_`, branch `develop-nasa-2026`.
- Vercel project: `astrostress-nasa-2026` (CLI deployments, no Git integration). Do not reuse or modify the older `astrostress` project.
- `vercel.json` pins `"framework": "flask"`. A project created with `vercel project add` defaults to "Other", and the deploy then fails with "pattern app.py ... doesn't match any Serverless Functions".
- Deploy Preview first. Production needs explicit approval.

## Research bundle (why there is no public download)

The model and verified outputs ship as `astrostress-render-artifacts.zip` (SHA256 `734c471c96f20584e74fc5842b3845e8f44f5365a61f03c01b91088c7ee57551`). It contains WESAD-derived per-window features, labels and subject IDs. WESAD terms allow scientific, non-commercial use and do not clearly permit redistribution, so the bundle is **never** published on GitHub Releases or any other public URL. It is gitignored.

`scripts/fetch_vercel_bundle.py` runs as the Vercel build hook (`[tool.vercel.scripts].build` in `pyproject.toml`) and installs the bundle from one of two private sources. Both are checked against the hash pinned in the script, which the environment cannot override:

1. **Local CLI upload (default):** the zip sits at the root of the snapshot passed to `vercel deploy`. It goes only to the private deployment source and is excluded from the function bundle (`vercel.json`).
2. **Private URL:** set a short-lived HTTPS `ASTROSTRESS_ARTIFACT_URL` as a Vercel environment variable. Never commit it.

If neither source is present, the build fails. A deployment never serves unverified or fabricated results.

## Deploy steps

```bash
git archive --format=tar develop-nasa-2026 | tar -x -C <empty-snapshot-dir>
cp astrostress-render-artifacts.zip <empty-snapshot-dir>/
cd <empty-snapshot-dir>
npx vercel@65.16.0 link --yes --project astrostress-nasa-2026
rm -f .env.local && git checkout -- .gitignore 2>/dev/null || true   # link may write these
npx vercel@65.16.0 deploy --yes --meta gitCommitSha=$(git -C <repo> rev-parse develop-nasa-2026)
```

Deploy from a clean snapshot, never from the working repository (it holds the 17 GB WESAD data and the virtualenv).

## Public data policy

Public deployments default-deny individual WESAD-derived records. `/api/window`, `/api/explain` and `/api/physiology-trend` return `403 RESTRICTED_DATA` unless `ASTROSTRESS_ALLOW_PUBLIC_WESAD_WINDOWS=1` is set. Do not set that variable without confirmed redistribution permission. `/api/status` reports `window_records_public`, and the UI then shows a clear restriction notice instead of per-window predictions.

Summary-level endpoints stay public: `/api/status`, `/api/subjects` (IDs and window counts only), `/api/research`, `/api/global-shap` and `/api/reliability` (Experiment C).

## After deployment

Check `/`, `/healthz`, `/api/status`, `/api/subjects`, `/api/research`, `/api/global-shap` and `/api/reliability` (all 200). Check that `/api/window?subject=S10&index=0`, `/api/explain?subject=S10&index=0` and `/api/physiology-trend?subject=S10` return 403. Check that `/astrostress_output/features.csv` and `/astrostress-render-artifacts.zip` return 404.

The saved model was built with scikit-learn 1.9.1 on Python 3.12. Do not change the runtime to a version that raises `InconsistentVersionWarning`. Predictions are in-sample WESAD laboratory replays, not held-out validation and not astronaut measurements. No secrets belong in code or deployment logs.

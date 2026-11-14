"""Install the exact verified research bundle at Vercel build time.

The bundle contains WESAD-derived per-window features and the trained model,
so it is NOT published at any public URL. It reaches the build in one of two
private ways, and both are checked against the SHA256 pinned below:

1. Local CLI upload (default, MVP): the deployer places the gitignored
   ``astrostress-render-artifacts.zip`` at the project root of the snapshot
   passed to ``vercel deploy``. It is uploaded only to the private Vercel
   deployment source and is excluded from the function bundle (vercel.json).
2. Private URL: set ``ASTROSTRESS_ARTIFACT_URL`` (HTTPS, e.g. a short-lived
   signed URL) as a Vercel environment variable. Never commit such a URL.

The pinned hash cannot be overridden from the environment. The build fails
loudly if no verified bundle is available, so a deployment never serves
fabricated or unverified results.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_render_bundle import ROOT, download, install_verified_zip  # noqa: E402

PINNED_SHA256 = "734c471c96f20584e74fc5842b3845e8f44f5365a61f03c01b91088c7ee57551"
LOCAL_BUNDLE = "astrostress-render-artifacts.zip"


def main(root=ROOT, environ=os.environ):
    root = Path(root)
    local = root / LOCAL_BUNDLE
    url = environ.get("ASTROSTRESS_ARTIFACT_URL", "").strip()
    if local.is_file():
        install_verified_zip(local.read_bytes(), PINNED_SHA256, root)
        source = "local deployment upload"
    elif url:
        install_verified_zip(download(url), PINNED_SHA256, root)
        source = "private artifact URL"
    else:
        raise SystemExit(
            f"No verified research bundle: place the gitignored {LOCAL_BUNDLE} in the "
            "deployment snapshot or set a private ASTROSTRESS_ARTIFACT_URL. "
            "Public redistribution of WESAD-derived files is not permitted."
        )
    print(f"Verified artifact ZIP installed from {source} (SHA256 {PINNED_SHA256[:12]}...); "
          "model provenance rechecked at server startup.")


if __name__ == "__main__":
    main()

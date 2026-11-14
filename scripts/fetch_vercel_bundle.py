"""Install the exact verified, published research bundle at Vercel build time.

Unlike the raw WESAD dataset, this intentionally public demo bundle is small.
This script refuses any ZIP content that does not match its pinned SHA256.
"""
import os

from fetch_render_bundle import main

os.environ["ASTROSTRESS_ARTIFACT_URL"] = (
    "https://github.com/Pakeeza1508/astroStress/releases/download/"
    "v0.1.0-demo/astrostress-render-artifacts.zip"
)
os.environ["ASTROSTRESS_ARTIFACT_SHA256"] = (
    "734c471c96f20584e74fc5842b3845e8f44f5365a61f03c01b91088c7ee57551"
)

if __name__ == "__main__":
    main()

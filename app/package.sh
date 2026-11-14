#!/usr/bin/env bash
# Build app_bundle.tar.gz (top folder artfae_demo/): app, checkpoints, assets, the research modules the
# pipeline imports, tests, requirements. No CLIP weights, no test data, no feature caches.
set -euo pipefail
cd "$(dirname "$0")/.."
STAGE=$(mktemp -d)/artfae_demo
mkdir -p "$STAGE"/{app,artfae_balance,src/artfae/features,src/artfae/models,tests,.streamlit}
cp app/{streamlit_app.py,inference.py,core.py,gallery.py,precompute_gallery.py,i18n.py,ui.py,README.md,requirements.txt,build_assets.py,package.sh} "$STAGE/app/"
cp -r app/assets "$STAGE/app/"
cp -r app/.streamlit "$STAGE/app/"
cp app/.streamlit/config.toml "$STAGE/.streamlit/"
cp app/requirements.txt app/README.md "$STAGE/"
cp artfae_balance/{__init__.py,model.py,heads.py,extract_features.py} "$STAGE/artfae_balance/"
cp src/artfae/__init__.py "$STAGE/src/artfae/"
cp src/artfae/features/{__init__.py,aesthetic.py} "$STAGE/src/artfae/features/"
cp src/artfae/models/{__init__.py,network.py} "$STAGE/src/artfae/models/"
cp tests/test_inference.py "$STAGE/tests/"
find "$STAGE" -name __pycache__ -prune -exec rm -rf {} +
tar -C "$(dirname "$STAGE")" -czf app_bundle.tar.gz artfae_demo
echo "app_bundle.tar.gz: $(stat -c %s app_bundle.tar.gz) bytes, $(tar -tzf app_bundle.tar.gz | grep -vc '/$') files"

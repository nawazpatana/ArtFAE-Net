#!/usr/bin/env bash
# Build app_bundle.tar.gz (top folder artfae_demo/): app, checkpoints, assets, the research modules the
# pipeline imports, tests, requirements. No CLIP weights, no test data, no feature caches.
#
# Refuses to run while app/ or artfae_demo/ have uncommitted changes, so the bundle always matches a commit.
# Afterwards, to update the tracked local copy: extract the bundle in a separate folder and copy it over
# artfae_demo/ (keeping artfae_demo/.venv); never extract it inside the repository.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -n "$(git status --porcelain -- app artfae_demo)" ]; then
  echo "package.sh: app/ or artfae_demo/ has uncommitted changes; commit or stash them first:" >&2
  git status --short -- app artfae_demo >&2
  exit 1
fi
STAGE=$(mktemp -d)/artfae_demo
mkdir -p "$STAGE"/{app,artfae_balance,src/artfae/features,src/artfae/models,tests,.streamlit}
cp app/{streamlit_app.py,inference.py,core.py,gallery.py,precompute_gallery.py,build_text_embeddings.py,verify_cloud_live.py,i18n.py,ui.py,README.md,requirements.txt,build_assets.py,package.sh} "$STAGE/app/"
cp -r app/assets "$STAGE/app/"
cp -r app/.streamlit "$STAGE/app/"
cp app/.streamlit/config.toml "$STAGE/.streamlit/"
cp app/requirements.txt "$STAGE/"
{ printf '%s\n' '> **WARNING: Do NOT extract this bundle inside the ArtFAE-Net repository; it would overwrite tracked files' \
    '> with this snapshot. Extract it in a separate folder.**' '' "Built from commit $(git rev-parse --short HEAD) on $(date -u +%Y-%m-%d)." ''
  cat app/README.md; } > "$STAGE/README.md"
cp "$STAGE/README.md" "$STAGE/app/README.md"
cp artfae_balance/{__init__.py,model.py,heads.py,extract_features.py} "$STAGE/artfae_balance/"
cp src/artfae/__init__.py "$STAGE/src/artfae/"
cp src/artfae/features/{__init__.py,aesthetic.py} "$STAGE/src/artfae/features/"
cp src/artfae/models/{__init__.py,network.py} "$STAGE/src/artfae/models/"
cp tests/test_inference.py "$STAGE/tests/"
find "$STAGE" -name __pycache__ -prune -exec rm -rf {} +
tar -C "$(dirname "$STAGE")" -czf app_bundle.tar.gz artfae_demo
rm -rf "$(dirname "$STAGE")"
echo "app_bundle.tar.gz: $(stat -c %s app_bundle.tar.gz) bytes, $(tar -tzf app_bundle.tar.gz | grep -vc '/$') files"

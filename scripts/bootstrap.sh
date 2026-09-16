#!/usr/bin/env bash
# One-time environment setup. Idempotent — safe to re-run.
#
# Requires network access to github.com: connectorch is not on PyPI, and the
# viewer is a full npm project, so both are vendored as pinned git submodules
# rather than installed from a registry. There is no offline fallback here.
set -euo pipefail
cd "$(dirname "$0")/.."

CONNECTORCH_COMMIT="4bbfb645099aeb85bdbf850e1a87cc094769af87"
VIEWER_COMMIT="38f55332055328d38c29e72474c4ad5b6876101f"

if [ ! -d .git ]; then
  echo "==> initializing git repo"
  git init
fi

if [ ! -f .gitmodules ] || ! grep -q "vendor/connectorch" .gitmodules 2>/dev/null; then
  echo "==> adding connectorch submodule (pinned to ${CONNECTORCH_COMMIT})"
  git submodule add https://github.com/us/connectorch.git vendor/connectorch
fi
( cd vendor/connectorch && git checkout "${CONNECTORCH_COMMIT}" )

if [ ! -f .gitmodules ] || ! grep -q '^\[submodule "viewer"\]' .gitmodules 2>/dev/null; then
  echo "==> adding fly-connectome-template submodule (pinned to ${VIEWER_COMMIT})"
  git submodule add https://github.com/cobanov/fly-connectome-template.git viewer
fi
( cd viewer && git checkout "${VIEWER_COMMIT}" )

echo "==> Python environment"
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip
pip install -e vendor/connectorch
pip install -e ".[dev]"

echo "==> Node environment (viewer)"
if command -v node >/dev/null 2>&1; then
  ( cd viewer && npm ci )
else
  echo "node not found on PATH — install Node.js >=22.18.0 (see .nvmrc) before running the viewer"
fi

echo "==> done. Activate with: source .venv/bin/activate"

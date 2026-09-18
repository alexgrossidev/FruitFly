#!/usr/bin/env bash
# One-time environment setup. Idempotent — safe to re-run.
#
# Requires network access to github.com: connectorch is not on PyPI, and the
# viewer is a full npm project, so both are vendored as pinned git submodules
# rather than installed from a registry. There is no offline fallback here.
#
# Interpreter selection: set PYTHON=/path/to/python3.x to override. Otherwise the
# first of python3.11 / python3.13 / python3.12 / python3 on PATH is used. Do not
# rely on /usr/bin/python3 on macOS — it is an Xcode Command Line Tools shim that
# disappears when the tools are removed, and a venv built on it dies with it.
set -euo pipefail
cd "$(dirname "$0")/.."

CONNECTORCH_COMMIT="4bbfb645099aeb85bdbf850e1a87cc094769af87"
VIEWER_COMMIT="38f55332055328d38c29e72474c4ad5b6876101f"

pick_python() {
  if [ -n "${PYTHON:-}" ]; then echo "$PYTHON"; return; fi
  for candidate in python3.11 python3.13 python3.12 python3; do
    if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1; then
      command -v "$candidate"; return
    fi
  done
  echo "no usable python >= 3.10 found on PATH; set PYTHON=/path/to/python3" >&2
  exit 1
}

if [ ! -d .git ]; then
  echo "==> initializing git repo"
  git init
fi

if [ ! -f .gitmodules ] || ! grep -q "vendor/connectorch" .gitmodules 2>/dev/null; then
  echo "==> adding connectorch submodule (pinned to ${CONNECTORCH_COMMIT})"
  git submodule add https://github.com/us/connectorch.git vendor/connectorch
fi
if [ ! -f .gitmodules ] || ! grep -q '^\[submodule "viewer"\]' .gitmodules 2>/dev/null; then
  echo "==> adding fly-connectome-template submodule (pinned to ${VIEWER_COMMIT})"
  git submodule add https://github.com/cobanov/fly-connectome-template.git viewer
fi
if [ ! -f vendor/connectorch/pyproject.toml ] || [ ! -f viewer/package.json ]; then
  echo "==> fetching submodule contents"
  git submodule update --init
fi
( cd vendor/connectorch && git checkout --quiet "${CONNECTORCH_COMMIT}" )
( cd viewer && git checkout --quiet "${VIEWER_COMMIT}" )

echo "==> Python environment"
PY="$(pick_python)"
echo "    using ${PY} ($("$PY" --version 2>&1))"
if [ -d .venv ] && ! .venv/bin/python -c 'import sys' >/dev/null 2>&1; then
  echo "    existing .venv has a broken interpreter; rebuilding it"
  rm -rf .venv
fi
if [ ! -d .venv ]; then
  "$PY" -m venv .venv
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

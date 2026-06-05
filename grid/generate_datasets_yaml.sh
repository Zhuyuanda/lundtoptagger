#!/bin/bash
# Wrapper: generate configs/config_datasets_grid.yaml from Rucio.
# Run on hypatia-login or lxplus after: lsetup rucio
# (this script will try to run lsetup if rucio is not in PATH)

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

setup_rucio_if_needed() {
  if command -v rucio >/dev/null 2>&1; then
    return 0
  fi
  local atlas_root="/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase/user/atlaslocalRoot.sh"
  if [[ -f "$atlas_root" ]]; then
    echo "Sourcing ATLASLocalRootBase and lsetup rucio ..."
    # shellcheck source=/dev/null
    source "$atlas_root"
    lsetup rucio
  fi
  if ! command -v rucio >/dev/null 2>&1; then
    echo "ERROR: rucio not in PATH."
    echo "  On hypatia-login run:  lsetup rucio"
    echo "  Then:                  bash grid/generate_datasets_yaml.sh"
    exit 1
  fi
}

setup_rucio_if_needed
echo "Using: $(command -v rucio) ($(rucio --version 2>&1 | head -1))"

PYTHON="${PYTHON:-python3}"
exec "$PYTHON" grid/generate_datasets_yaml.py "$@"

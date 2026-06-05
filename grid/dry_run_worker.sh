#!/bin/bash
# Simulate what a PanDA worker does: pack → extract → setup → import check.
# Run from repo root. Two modes:
#   bash grid/dry_run_worker.sh              # local (current environment)
#   bash grid/dry_run_worker.sh --container  # inside centos7 apptainer (recommended)

set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
CONTAINER=0
for arg in "$@"; do [[ "$arg" == "--container" ]] && CONTAINER=1; done

WORK=$(mktemp -d /tmp/panda_dry_XXXXXX)
trap 'rm -rf "$WORK"' EXIT
PASS=0; FAIL=0
ok()  { echo "  [OK]   $*"; PASS=$((PASS+1)); }
bad() { echo "  [FAIL] $*"; FAIL=$((FAIL+1)); }

echo "=== PanDA worker dry run ($(hostname)) ==="
echo "Workdir: $WORK"

# ── Step 1: pack ──────────────────────────────────────────────────────────────
echo ""
echo "[1/5] Packing code (grid/pack_code.sh) ..."
bash "$REPO/grid/pack_code.sh" "$WORK/lundtoptagger_panda.tar.gz"

# ── Step 2: extract (runGen does this, NOT our bash script) ───────────────────
echo ""
echo "[2/5] Extracting tarball (simulates runGen sandbox extraction) ..."
cd "$WORK"
tar xzf lundtoptagger_panda.tar.gz

EXPECTED=(
  "Make_data_LRJ.py"
  "preprocess_LRJ_CPU.py"
  "submit/panda/run_make_data.sh"
  "submit/panda/setup_panda_worker.sh"
  "configs/config_make_data_LRJ_panda.yaml"
  "configs/config_signal_LRJ.yaml"
)
echo "Checking expected files after extraction:"
for f in "${EXPECTED[@]}"; do
  [[ -f "$WORK/$f" ]] && ok "$f" || bad "MISSING: $f"
done

# ── Step 3: show exec string that PanDA would run ─────────────────────────────
echo ""
echo "[3/5] PanDA exec string (decoded from jobPars):"
echo "  IN='<turl>' SIGNAL=top DATA_ID=TopTagging_LRJ bash submit/panda/run_make_data.sh %RNDM:0"
echo "  (%RNDM:0 → 0 for job0 / 1 for job1)"

# ── Step 4: run setup_panda_worker.sh ─────────────────────────────────────────
echo ""
echo "[4/5] Testing setup_panda_worker.sh ..."

SETUP_CMD="
set -x
cd '$WORK'
export ALRB_OSMAJORVER=\${ALRB_OSMAJORVER:-7}
source submit/panda/setup_panda_worker.sh
echo SETUP_EXIT_OK
python3 -c 'import torch, uproot, awkward, yaml; print(\"imports OK: torch uproot awkward yaml\")'
python3 -c 'import torch_geometric; print(\"torch_geometric OK\")' || echo 'torch_geometric not available (non-fatal)'
"

if [[ "$CONTAINER" == "1" ]]; then
  CONT_IMG="/cvmfs/atlas.cern.ch/repo/containers/fs/singularity/x86_64-centos7"
  if [[ ! -d "$CONT_IMG" ]]; then
    bad "Container image not found: $CONT_IMG (need CVMFS)"
  elif ! command -v apptainer >/dev/null 2>&1 && ! command -v singularity >/dev/null 2>&1; then
    bad "apptainer/singularity not in PATH (try: setupATLAS first)"
  else
    RUNNER=$(command -v apptainer 2>/dev/null || command -v singularity)
    echo "  Running in centos7 container via $RUNNER ..."
    if "$RUNNER" exec -B /cvmfs:/cvmfs -B "$WORK:/work" "$CONT_IMG" \
        bash -c "cd /work && export ALRB_OSMAJORVER=7 && source submit/panda/setup_panda_worker.sh && echo SETUP_EXIT_OK && python3 -c 'import torch, uproot, awkward, yaml; print(\"imports OK\")'"; then
      ok "setup + imports inside centos7 container"
    else
      bad "setup/imports failed inside centos7 container (see output above)"
    fi
  fi
else
  echo "  Running locally (not in container - add --container for realistic test)"
  cd "$WORK"
  if bash -c "$SETUP_CMD"; then
    ok "setup + imports locally"
  else
    bad "setup/imports failed locally (see output above)"
  fi
fi

# ── Step 5: Python syntax check ───────────────────────────────────────────────
echo ""
echo "[5/5] Python syntax check on main scripts ..."
for f in Make_data_LRJ.py preprocess_LRJ_CPU.py; do
  if python3 -c "
import ast, sys
with open('$WORK/$f') as fh:
    ast.parse(fh.read())
print('  syntax OK: $f')
" 2>/dev/null; then
    ok "syntax: $f"
  else
    bad "syntax error in: $f"
  fi
done

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "=== Summary: $PASS OK, $FAIL FAIL ==="
[[ $FAIL -eq 0 ]]

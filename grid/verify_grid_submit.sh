#!/bin/bash
# Run on the machine where you will condor_submit (usually lxplus).
# On hypatia: checks Rucio only; EOS/condor will FAIL (expected).

set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

EOS_BASE="${EOS_BASE:-/eos/user/yuanda/lundtoptagger_lrj}"
TARBALL="${TARBALL:-${EOS_BASE}/lundtoptagger_grid.tar.gz}"
PASS=0
FAIL=0
WARN=0

ok()   { echo "[OK]   $*"; PASS=$((PASS+1)); }
bad()  { echo "[FAIL] $*"; FAIL=$((FAIL+1)); }
warn() { echo "[WARN] $*"; WARN=$((WARN+1)); }

echo "========== Grid submit readiness =========="
echo "Host: $(hostname)"
echo "EOS_BASE=$EOS_BASE"
echo ""

# 1. Config
if [[ -f configs/rucio_datasets_lrj.yaml ]] && grep -q "801661" configs/rucio_datasets_lrj.yaml; then
  ok "rucio_datasets_lrj.yaml present (801661)"
else
  bad "configs/rucio_datasets_lrj.yaml missing or incomplete"
fi

# 2. Pack
if bash grid/pack_code.sh /tmp/lundtoptagger_grid_verify.tar.gz >/dev/null 2>&1; then
  ok "grid/pack_code.sh works"
else
  bad "grid/pack_code.sh failed"
fi

# 3. EOS
if [[ -d "/eos/user/yuanda" ]]; then
  ok "EOS mounted (/eos/user/yuanda)"
  if mkdir -p "${EOS_BASE}/logs" "${EOS_BASE}/TopTagging_LRJ" 2>/dev/null; then
    ok "Can create ${EOS_BASE}"
  else
    bad "Cannot write under ${EOS_BASE}"
  fi
else
  bad "No /eos/user/yuanda (submit from lxplus, not hypatia login?)"
fi

# 4. Tarball on EOS
if [[ -f "$TARBALL" ]]; then
  ok "Tarball on EOS: $TARBALL"
else
  warn "Tarball not on EOS yet: $TARBALL (run: bash grid/pack_code.sh && cp ...)"
fi

# 5. Condor
if command -v condor_submit >/dev/null 2>&1; then
  ok "condor_submit in PATH"
else
  bad "condor_submit not found (need lxplus)"
fi

# 6. Rucio
if command -v rucio >/dev/null 2>&1; then
  ok "rucio in PATH"
  if rucio whoami >/dev/null 2>&1; then
    ok "rucio whoami: $(rucio whoami 2>/dev/null | head -1)"
  else
    warn "rucio found but whoami failed (proxy?)"
  fi
elif [[ -f /cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase/user/atlaslocalRoot.sh ]]; then
  warn "rucio not loaded; run: setupATLAS && lsetup rucio"
else
  bad "rucio not available"
fi

# 7. Quick resolve test
if command -v rucio >/dev/null 2>&1 && command -v python3 >/dev/null 2>&1; then
  if python3 grid/rucio_resolve.py --container user.yuanda.mc20_13TeV.801661.e8435_s3681_r13145_p7053.20e_jetm2_ANALYSIS.root/ --max-files 1 2>/dev/null | grep -q '^root://'; then
    ok "rucio_resolve smoke test (1 PFN)"
  else
    warn "rucio_resolve smoke test failed"
  fi
fi

# 8. PFN cache
CACHE="${EOS_BASE}/TopTagging_LRJ/rucio_pfns_top.json"
if [[ -f "$CACHE" ]] && [[ -s "$CACHE" ]]; then
  N=$(python3 -c "import json; print(len(json.load(open('$CACHE'))))" 2>/dev/null || echo "?")
  ok "PFN cache exists ($N entries): $CACHE"
else
  warn "PFN cache not built yet (optional before submit; job 0 will build)"
fi

echo ""
echo "========== Summary: $PASS ok, $WARN warn, $FAIL fail =========="
if [[ $FAIL -eq 0 ]]; then
  echo "READY to condor_submit on this host (fix WARN items if needed)."
  exit 0
else
  echo "NOT READY on this host — see FAIL lines above."
  exit 1
fi

#!/bin/bash
# Submit LRJ Make_data to PanDA (2 jobs: 70% + 30%). Run on hypatia after: setupATLAS && lsetup panda

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

SIGNAL="${SIGNAL:-top}"
DATA_ID="${DATA_ID:-TopTagging_LRJ}"
PANDA_NCORE="${PANDA_NCORE:-1}"

if ! command -v prun >/dev/null 2>&1; then
  echo "ERROR: prun not found. Run: setupATLAS && lsetup panda"
  exit 1
fi

bash grid/pack_code.sh lundtoptagger_panda.tar.gz

IN_DS=$(python3 grid/panda_inDS.py --signal "${SIGNAL}")
# PanDA outDS (logs + output.tar.gz). Default: unique suffix so a prior broken task
# does not block resubmission.
if [[ -n "${OUT_DS:-}" ]]; then
  :
elif [[ "${USE_FIXED_OUTDS:-}" == 1 ]]; then
  OUT_DS="user.yuanda.${DATA_ID}.make_data"
else
  OUT_SUFFIX="${OUT_SUFFIX:-$(date +%Y%m%d_%H%M%S)}"
  OUT_DS="user.yuanda.${DATA_ID}.make_data.${OUT_SUFFIX}"
fi

# Optional: PANDA_MEMORY_MB=32000 to request RAM (MB). Omit for PanDA default (faster matching).
# PANDA_WALLTIME_SEC: max wall time per job in seconds (default: 86400 = 24h).
PANDA_WALLTIME_SEC="${PANDA_WALLTIME_SEC:-86400}"
# PANDA_ARCHITECTURE: require AVX2-capable CPUs (x86_64-v3) to avoid old sites lacking AVX.
PANDA_ARCHITECTURE="${PANDA_ARCHITECTURE:-x86_64-v3}"

echo "inDS=${IN_DS}"
echo "outDS=${OUT_DS}"
echo "Submitting 2 PanDA jobs (70% / 30%), part index via %RNDM:0 ..."
echo "prun nCore=${PANDA_NCORE} (override: PANDA_NCORE=8)"
if [[ -n "${PANDA_MEMORY_MB:-}" ]]; then
  echo "prun memory: ${PANDA_MEMORY_MB} MB"
else
  echo "prun memory: (PanDA default)"
fi

prun -y \
  --noBuild \
  --inTarBall lundtoptagger_panda.tar.gz \
  --exec "IN='%IN' SIGNAL=${SIGNAL} DATA_ID=${DATA_ID} bash submit/panda/run_make_data.sh %RNDM:0" \
  --inDS "${IN_DS}" \
  --outDS "${OUT_DS}" \
  --outputs output.tar.gz \
  --nJobs 2 \
  --nFilesPerJob=1000 \
  --nCore "${PANDA_NCORE}" \
  --maxWalltime "${PANDA_WALLTIME_SEC}" \
  --architecture "${PANDA_ARCHITECTURE}" \
  ${PANDA_MEMORY_MB:+--memory "${PANDA_MEMORY_MB}"}

echo "Submitted. Check bigpanda: https://bigpanda.cern.ch/tasks/?taskname=${OUT_DS}*"
echo "After jobs finish, download outputs:"
echo "  rucio download ${OUT_DS}/"
echo "  tar xzf output._0001.tar.gz && tar xzf output._0002.tar.gz"

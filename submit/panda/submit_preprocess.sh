#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

DATA_ID="${DATA_ID:-TopTagging_LRJ}"
EOS_BASE="${EOS_BASE:-/eos/user/yuanda/lundtoptagger_lrj}"
PANDA_NCORE="${PANDA_NCORE:-1}"

if [[ -n "${OUT_DS:-}" ]]; then
  :
elif [[ "${USE_FIXED_OUTDS:-}" == 1 ]]; then
  OUT_DS="user.yuanda.${DATA_ID}.preprocess"
else
  OUT_SUFFIX="${OUT_SUFFIX:-$(date +%Y%m%d_%H%M%S)}"
  OUT_DS="user.yuanda.${DATA_ID}.preprocess.${OUT_SUFFIX}"
fi

bash grid/pack_code.sh lundtoptagger_panda.tar.gz

echo "outDS=${OUT_DS}"

prun -y \
  --noBuild \
  --inTarBall lundtoptagger_panda.tar.gz \
  --exec "EOS_BASE=${EOS_BASE:-/eos/user/yuanda/lundtoptagger_lrj} DATA_ID=${DATA_ID} bash submit/panda/run_preprocess.sh" \
  --outDS "${OUT_DS}" \
  --nJobs 1 \
  --nCore "${PANDA_NCORE}" \
  ${PANDA_MEMORY_MB:+--memory "${PANDA_MEMORY_MB}"} \
  --destSE CERNPROD

echo "Preprocess submitted. Check: panda check --outDS ${OUT_DS}"

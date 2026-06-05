#!/bin/bash
# PanDA exec: $1 / %RNDM:0 = event_fraction_idx (0=70% train pool, 1=30% test pool)

set -euo pipefail

EVENT_FRACTION_IDX="${1:-0}"
SIGNAL="${SIGNAL:-top}"
DATA_ID="${DATA_ID:-TopTagging_LRJ}"

cd "${PWD}"
echo "=== worker started: PWD=$(pwd) DATE=$(date) ==="

if [[ -f lundtoptagger_panda.tar.gz ]]; then
  echo "=== extracting lundtoptagger_panda.tar.gz ==="
  tar xzf lundtoptagger_panda.tar.gz
fi
echo "=== tar done, contents: $(ls | head -20) ==="

echo "=== sourcing setup_panda_worker.sh ==="
source submit/panda/setup_panda_worker.sh
echo "=== setup done ==="

# PanDA passes input ROOT files in IN (space-separated)
export PANDA_INPUT_FILES="${IN:-${PANDA_INPUT_FILES:-}}"

OUT_LOCAL="./output_part${EVENT_FRACTION_IDX}"
mkdir -p "${OUT_LOCAL}"

echo "part=${EVENT_FRACTION_IDX} signal=${SIGNAL} n_input=$(echo $PANDA_INPUT_FILES | wc -w) out=${OUT_LOCAL}"

python3 Make_data_LRJ.py configs/config_make_data_LRJ_panda.yaml --override \
  signal="${SIGNAL}" \
  id="${DATA_ID}" \
  out_dir="${OUT_LOCAL}/" \
  event_fraction_idx="${EVENT_FRACTION_IDX}"

echo "Make_data part ${EVENT_FRACTION_IDX} done. Packaging output..."
tar czf output.tar.gz "${OUT_LOCAL}/"
echo "Packaged: output.tar.gz ($(du -h output.tar.gz | cut -f1))"

#!/bin/bash
# Run preprocess locally after downloading and extracting make_data outputs.
# Usage: DATA_DIR=/path/to/extracted bash submit/panda/run_preprocess.sh
#
# DATA_DIR should contain:
#   output_part0/  (from output._0001.tar.gz)
#   output_part1/  (from output._0002.tar.gz)

set -euo pipefail

DATA_ID="${DATA_ID:-TopTagging_LRJ}"
DATA_DIR="${DATA_DIR:-./make_data_output}"

BASE_PART0="${DATA_DIR}/output_part0"
BASE_PART1="${DATA_DIR}/output_part1"
OUT_DIR="${DATA_DIR}/preprocessed"

if [[ ! -d "${BASE_PART0}" ]]; then
  echo "ERROR: ${BASE_PART0} not found. Extract make_data tars first."
  exit 1
fi
if [[ ! -d "${BASE_PART1}" ]]; then
  echo "ERROR: ${BASE_PART1} not found. Extract make_data tars first."
  exit 1
fi

mkdir -p "${OUT_DIR}"

echo "Running preprocess: part0=${BASE_PART0} part1=${BASE_PART1} out=${OUT_DIR}"

python3 preprocess_LRJ_CPU.py configs/config_preprocess_LRJ_panda.yaml --override \
  "data.train_graphs=['${BASE_PART0}/graphs_${DATA_ID}_batch*_ln_kT_cut_None_with_pt.pt']" \
  "data.test_graphs=['${BASE_PART1}/graphs_${DATA_ID}_batch*_ln_kT_cut_None_with_pt.pt']" \
  data.meanstd_json="${BASE_PART0}/meanstd_part0.json" \
  data.out_dir="${OUT_DIR}"

echo "Preprocess done. Output: ${OUT_DIR}"

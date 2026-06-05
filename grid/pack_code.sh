#!/bin/bash
# Build tarball for PanDA (--tarBall). Run from repo root: bash grid/pack_code.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT="${1:-lundtoptagger_panda.tar.gz}"
if [[ "$OUT" != /* ]]; then
  OUT="${ROOT}/${OUT}"
fi

tar czf "$OUT" \
  Make_data_LRJ.py \
  preprocess_LRJ_CPU.py \
  tools/GNN_model_weight/utils_newdata.py \
  tools/GNN_model_weight/models.py \
  tools/utils_config.py \
  plotting/utils_plots_matplotlib.py \
  configs/config_make_data_LRJ_panda.yaml \
  configs/config_preprocess_LRJ_panda.yaml \
  configs/rucio_datasets_lrj.yaml \
  configs/config_signal_LRJ.yaml \
  grid/panda_inDS.py \
  submit/panda/run_make_data.sh \
  submit/panda/run_preprocess.sh \
  submit/panda/setup_panda_worker.sh

echo "Created ${OUT} ($(du -h "$OUT" | cut -f1))"

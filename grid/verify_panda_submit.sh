#!/bin/bash
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PASS=0; FAIL=0; WARN=0
ok() { echo "[OK] $*"; PASS=$((PASS+1)); }
bad() { echo "[FAIL] $*"; FAIL=$((FAIL+1)); }
warn() { echo "[WARN] $*"; WARN=$((WARN+1)); }

echo "=== PanDA submit check ($(hostname)) ==="

grep -q "0.7: 1" configs/config_make_data_LRJ.yaml && ok "event_fractions 70/30 in config_make_data_LRJ.yaml" || bad "missing 70/30 split"
grep -q "part0_70percent" configs/config_preprocess_LRJ.yaml && ok "preprocess paths part0/part1" || bad "preprocess paths"

[[ -f submit/panda/submit_make_data.sh ]] && ok "submit/panda scripts" || bad "missing submit/panda"
bash grid/pack_code.sh /tmp/verify_panda.tar.gz >/dev/null 2>&1 && ok "pack_code.sh" || bad "pack failed"

command -v prun >/dev/null && ok "prun in PATH" || warn "prun missing (lsetup panda)"
command -v rucio >/dev/null && ok "rucio in PATH" || warn "rucio missing"
python3 grid/panda_inDS.py --signal top 2>/dev/null | grep -q "user.yuanda\." && \
  ! python3 grid/panda_inDS.py --signal top 2>/dev/null | grep -q "user.yuanda:" && \
  ok "panda_inDS.py (dot names)" || bad "panda_inDS failed (expect user.yuanda.<ds>/, no colon)"

if [[ -d /eos/user/yuanda ]]; then ok "EOS mounted"; else warn "EOS not on this host (OK on Panda worker if destSE set)"; fi

echo "Summary: $PASS ok, $WARN warn, $FAIL fail"
[[ $FAIL -eq 0 ]]

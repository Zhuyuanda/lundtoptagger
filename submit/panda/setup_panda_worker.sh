#!/bin/bash
# Worker environment for PanDA jobs (LCG + torch_geometric).
# Uses ALRB_OSMAJORVER (set by apptainer/ATLAS container infra) for OS detection
# instead of hostnamectl which does not work inside containers.

set -euo pipefail

OS_MAJOR="${ALRB_OSMAJORVER:-}"
if [[ -z "$OS_MAJOR" ]]; then
  # fallback: read from /etc/os-release
  OS_MAJOR=$(grep -oP '(?<=^VERSION_ID=")[0-9]+' /etc/os-release 2>/dev/null | head -1 || echo "7")
fi

echo "Detected OS major version: $OS_MAJOR"

set +u  # LCG setup.sh references unset vars; temporarily disable set -u
if [[ "$OS_MAJOR" == "7" ]]; then
  source /cvmfs/sft.cern.ch/lcg/views/LCG_104/x86_64-centos7-gcc12-opt/setup.sh
elif [[ "$OS_MAJOR" == "8" ]] || [[ "$OS_MAJOR" == "9" ]]; then
  source /cvmfs/sft.cern.ch/lcg/views/LCG_104/x86_64-el9-gcc11-opt/setup.sh
else
  echo "WARNING: unknown OS major version '$OS_MAJOR', trying centos7 LCG"
  source /cvmfs/sft.cern.ch/lcg/views/LCG_104/x86_64-centos7-gcc12-opt/setup.sh || true
fi
set -u

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

if ! python3 -c "import torch_geometric" 2>/dev/null; then
  pip install --user --quiet torch_geometric || echo "WARNING: torch_geometric install failed, continuing"
  export PYTHONPATH="${HOME}/.local/lib/python3.11/site-packages:${PYTHONPATH}"
fi

python3 -c "import torch, uproot, awkward, yaml; print('worker OK')" || \
  echo "WARNING: some imports not available after LCG setup"

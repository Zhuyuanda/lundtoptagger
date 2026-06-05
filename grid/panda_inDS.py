#!/usr/bin/env python3
"""Print comma-separated PanDA --inDS from configs/rucio_datasets_lrj.yaml."""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT = REPO / "configs" / "rucio_datasets_lrj.yaml"


def parse_datasets(path, signal):
    text = Path(path).read_text()
    containers = []
    in_qcd = False
    sig_key = "w_signal:" if signal.upper() == "W" else "top_signal:"
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s == "qcd:":
            in_qcd = True
            continue
        if s.startswith("top_signal:") or s.startswith("w_signal:"):
            in_qcd = False
            if s.startswith(sig_key):
                containers.append(s.split(":", 1)[1].strip().rstrip("/"))
            continue
        if in_qcd and s.startswith("- "):
            containers.append(s[2:].strip().rstrip("/"))
    return containers


def to_panda_ds(container):
    """PanDA --inDS names use dots (DDM), not Rucio scope:name colons."""
    c = container.strip().rstrip("/")
    if c.startswith("user.yuanda:"):
        c = "user.yuanda." + c[len("user.yuanda:") :]
    elif not c.startswith("user.yuanda."):
        c = "user.yuanda." + c
    return c + "/"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signal", default="top", choices=["top", "W", "w"])
    ap.add_argument("--datasets-file", default=str(DEFAULT))
    args = ap.parse_args()
    containers = parse_datasets(args.datasets_file, args.signal)
    if not containers:
        sys.exit("no containers found")
    print(",".join(to_panda_ds(c) for c in containers))


if __name__ == "__main__":
    main()

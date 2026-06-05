#!/usr/bin/env python3
"""
Build configs/config_datasets_grid.yaml from Rucio replicas.

Works when `rucio list-files` has NO --pfns (use list-file-replicas per LFN).
Run on hypatia-login or lxplus where `rucio` works.

Usage:
  python grid/generate_datasets_yaml.py
  python grid/generate_datasets_yaml.py --use-glob   # one *.root per container if same EOS dir
  python grid/generate_datasets_yaml.py --max-files 2   # dry-run: 2 files per dataset
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
SOURCES = REPO / "configs" / "rucio_datasets_lrj.yaml"
OUT = REPO / "configs" / "config_datasets_grid.yaml"

LFN_RE = re.compile(r"user\.yuanda:user\.yuanda\.\d+\._\d+\.ANALYSIS\.root")
URL_RE = re.compile(r"(?:root|davs|gs|s3)://\S+")


def run_rucio(args: list[str], timeout: int = 600) -> str:
    cmd = ["rucio", *args]
    try:
        p = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError:
        sys.exit("ERROR: `rucio` not in PATH. Run on hypatia-login or lxplus.")
    if p.returncode != 0:
        sys.stderr.write(p.stderr)
        sys.exit(f"ERROR: rucio failed: {' '.join(cmd)}")
    return p.stdout + p.stderr


_SUPPORTS_PFN: bool | None = None


def rucio_supports_pfn_flag() -> bool:
    global _SUPPORTS_PFN
    if _SUPPORTS_PFN is not None:
        return _SUPPORTS_PFN
    try:
        p = subprocess.run(
            ["rucio", "list-files", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        _SUPPORTS_PFN = "--pfns" in (p.stdout + p.stderr)
    except FileNotFoundError:
        _SUPPORTS_PFN = False
    return _SUPPORTS_PFN


def parse_lfns_from_list_files(text: str) -> list[str]:
    lfns = []
    for line in text.splitlines():
        m = LFN_RE.search(line)
        if m:
            lfns.append(m.group(0))
    return sorted(set(lfns))


def list_lfns_in_dataset(dataset: str) -> list[str]:
    out = run_rucio(["list-files", dataset])
    return parse_lfns_from_list_files(out)


def list_pfns_via_pfn_flag(dataset: str) -> list[str]:
    out = run_rucio(["list-files", "--pfns", dataset])
    urls = []
    for line in out.splitlines():
        m = URL_RE.search(line)
        if m:
            urls.append(davs_to_root(m.group(0)))
    return sorted(set(urls))


def davs_to_root(url: str) -> str:
    if url.startswith("root://"):
        return url
    if url.startswith("davs://"):
        # davs://eosuser.cern.ch/eos/user/... -> root://eosuser.cern.ch//eos/user/...
        rest = url[len("davs://") :]
        host, _, path = rest.partition("/")
        if path.startswith("eos/"):
            return f"root://{host}//{path}"
        return f"root://{host}/{path}"
    return url


def pfn_for_lfn(lfn: str) -> str | None:
    out = run_rucio(["list-file-replicas", lfn], timeout=120)
    for line in out.splitlines():
        m = URL_RE.search(line)
        if m:
            return davs_to_root(m.group(0))
    return None


def pfns_for_dataset(dataset: str, max_files: int | None) -> list[str]:
    if rucio_supports_pfn_flag():
        urls = list_pfns_via_pfn_flag(dataset)
        if urls:
            if max_files:
                urls = urls[:max_files]
            return urls

    lfns = list_lfns_in_dataset(dataset)
    if max_files:
        lfns = lfns[:max_files]
    pfns = []
    for i, lfn in enumerate(lfns, 1):
        pfn = pfn_for_lfn(lfn)
        if not pfn:
            print(f"WARN: no PFN for {lfn}", file=sys.stderr)
            continue
        pfns.append(pfn)
        if i % 10 == 0:
            print(f"  {dataset}: {i}/{len(lfns)} replicas", file=sys.stderr)
    return pfns


def common_glob(pfns: list[str]) -> str | None:
    if not pfns:
        return None
    dirs = [os.path.dirname(p) for p in pfns]
    if len(set(dirs)) != 1:
        return None
    d = dirs[0]
    names = [os.path.basename(p) for p in pfns]
    if len(set(names)) != len(names):
        return None
    return f"{d}/*.root"


def collect_paths(datasets: list[str], use_glob: bool, max_files: int | None) -> list[str]:
    all_pfns: list[str] = []
    for ds in datasets:
        print(f"Dataset: {ds}", file=sys.stderr)
        pfns = pfns_for_dataset(ds, max_files)
        if use_glob:
            g = common_glob(pfns)
            if g:
                print(f"  -> glob: {g}", file=sys.stderr)
                all_pfns.append(g)
                continue
            print("  -> glob failed (files in multiple dirs); using explicit list", file=sys.stderr)
        all_pfns.extend(pfns)
    return sorted(set(all_pfns))


def main():
    ap = argparse.ArgumentParser(description="Generate config_datasets_grid.yaml from Rucio")
    ap.add_argument("--use-glob", action="store_true", help="One *.root glob per container when possible")
    ap.add_argument("--max-files", type=int, default=None, help="Limit files per container (smoke test)")
    ap.add_argument("-o", "--output", type=Path, default=OUT)
    args = ap.parse_args()

    with open(SOURCES) as f:
        src = yaml.safe_load(f)

    qcd = src["qcd"]
    top_paths = collect_paths(qcd + [src["top_signal"]], args.use_glob, args.max_files)
    w_paths = collect_paths(qcd + [src["w_signal"]], args.use_glob, args.max_files)

    out_doc = {
        "#": "AUTO-GENERATED by grid/generate_datasets_yaml.py — re-run after Rucio changes",
        "path_to_rootfiles_top": top_paths,
        "path_to_rootfiles_W": w_paths,
    }
    # yaml with comment header
    header = (
        "# AUTO-GENERATED by grid/generate_datasets_yaml.py\n"
        "# Re-run on hypatia-login or lxplus: python grid/generate_datasets_yaml.py\n"
        "# Optional: --use-glob (one line per Rucio container)  --max-files 2 (test)\n\n"
    )
    body = yaml.dump(
        {"path_to_rootfiles_top": top_paths, "path_to_rootfiles_W": w_paths},
        default_flow_style=False,
        sort_keys=False,
        width=120,
    )
    args.output.write_text(header + body)
    print(f"Wrote {args.output} ({len(top_paths)} top paths, {len(w_paths)} W paths)")


if __name__ == "__main__":
    main()

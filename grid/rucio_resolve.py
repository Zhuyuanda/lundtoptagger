#!/usr/bin/env python3
"""
Resolve Rucio container name(s) -> root:// PFN list (prefer CERN EOS replicas).

No pre-generated config_datasets_grid.yaml needed. Edit only:
  configs/rucio_datasets_lrj.yaml

Usage:
  python grid/rucio_resolve.py --signal top --write-cache /eos/.../rucio_pfns_top.json
  python grid/rucio_resolve.py --container user.yuanda.mc20_13TeV.801661....root/
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_DATASETS = REPO / "configs" / "rucio_datasets_lrj.yaml"

LFN_RE = re.compile(r"user\.yuanda:user\.yuanda\.\d+\._\d+\.ANALYSIS\.root")
URL_RE = re.compile(r"(?:root|davs|gs|s3)://\S+")

DEFAULT_PREFER = [
    "eosatlas.cern.ch",
    "eosuser.cern.ch",
    "atlas-grid.cern.ch",
]


def run_rucio(args: list[str], timeout: int = 600) -> str:
    try:
        p = subprocess.run(
            ["rucio", *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError:
        sys.exit("ERROR: `rucio` not in PATH. Run: lsetup rucio")
    if p.returncode != 0:
        sys.stderr.write(p.stderr or "")
        sys.exit(f"ERROR: rucio {' '.join(args)}")
    return p.stdout + p.stderr


def normalize_container(name: str) -> str:
    return name.strip().rstrip("/")


def parse_rucio_datasets_yaml(path: Path, signal: str) -> list[str]:
    """Minimal parser — no PyYAML required."""
    text = path.read_text()
    containers: list[str] = []
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
                containers.append(normalize_container(s.split(":", 1)[1]))
            continue
        if in_qcd and s.startswith("- "):
            containers.append(normalize_container(s[2:]))

    if not containers:
        sys.exit(f"ERROR: no containers for signal={signal} in {path}")
    return containers


def davs_to_root(url: str) -> str:
    if url.startswith("root://"):
        return url.rstrip(",")
    if url.startswith("davs://"):
        rest = url[len("davs://") :]
        host, _, path = rest.partition("/")
        if path.startswith("eos/"):
            return f"root://{host}//{path}"
        return f"root://{host}/{path}"
    return url.rstrip(",")


def pick_pfn(urls, prefer_hosts):
    if not urls:
        return None
    for host in prefer_hosts:
        for u in urls:
            if host in u:
                return u
    return urls[0]


def replicas_for_lfn(lfn, prefer_hosts):
    out = run_rucio(["list-file-replicas", lfn], timeout=120)
    urls = [davs_to_root(m.group(0)) for m in URL_RE.finditer(out)]
    return pick_pfn(urls, prefer_hosts)


def pfns_for_container(
    container: str,
    prefer_hosts: list[str],
    max_files=None,
):
    container = normalize_container(container)
    print(f"Rucio container: {container}", file=sys.stderr)
    out = run_rucio(["list-files", container])
    lfns = sorted(set(LFN_RE.findall(out)))
    if max_files:
        lfns = lfns[:max_files]
    pfns = []
    for i, lfn in enumerate(lfns, 1):
        pfn = replicas_for_lfn(lfn, prefer_hosts)
        if not pfn:
            print(f"WARN: no PFN for {lfn}", file=sys.stderr)
            continue
        pfns.append(pfn)
        if i % 20 == 0:
            print(f"  {container}: {i}/{len(lfns)}", file=sys.stderr)
    return pfns


def containers_for_campaign(
    signal: str,
    datasets_file: Path,
    containers_cli=None,
):
    if containers_cli:
        return [normalize_container(c) for c in containers_cli]
    return parse_rucio_datasets_yaml(datasets_file, signal)


def resolve_all(
    containers: list[str],
    prefer_hosts: list[str],
    max_files=None,
):
    all_pfns = []
    for c in containers:
        all_pfns.extend(pfns_for_container(c, prefer_hosts, max_files))
    return sorted(set(all_pfns))


def write_cache(path: str, pfns: list[str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(pfns, f, indent=0)
    os.replace(tmp, path)
    print(f"Wrote {len(pfns)} PFNs -> {path}", file=sys.stderr)


def read_cache(path):
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return None
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list) and data:
        return data
    return None


def ensure_cache(
    cache_path: str,
    containers: list[str],
    prefer_hosts: list[str],
    max_files,
    wait_seconds,
):
    existing = read_cache(cache_path)
    if existing:
        print(f"Using cached PFN list ({len(existing)} files): {cache_path}", file=sys.stderr)
        return existing

    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    lock = cache_path + ".lock"
    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        existing = read_cache(cache_path)
        if existing:
            return existing
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            try:
                pfns = resolve_all(containers, prefer_hosts, max_files)
                if not pfns:
                    sys.exit("ERROR: resolved zero PFNs")
                write_cache(cache_path, pfns)
                return pfns
            finally:
                os.unlink(lock)
        except FileExistsError:
            time.sleep(10)

    existing = read_cache(cache_path)
    if existing:
        return existing
    sys.exit(f"ERROR: timed out waiting for PFN cache {cache_path}")


def main():
    ap = argparse.ArgumentParser(description="Rucio container(s) -> root:// PFN list")
    ap.add_argument("--signal", choices=["top", "W", "w"], help="Campaign (reads rucio_datasets_lrj.yaml)")
    ap.add_argument("--container", action="append", help="Single container (repeatable); overrides --signal")
    ap.add_argument("--datasets-file", type=Path, default=DEFAULT_DATASETS)
    ap.add_argument("--write-cache", metavar="PATH", help="Write JSON PFN list for Make_data")
    ap.add_argument("--print-paths", action="store_true", help="Print Python list repr to stdout")
    ap.add_argument("--max-files", type=int, default=None, help="Limit files per container (test)")
    ap.add_argument("--prefer", nargs="*", default=DEFAULT_PREFER, help="Preferred host substrings in PFN")
    ap.add_argument("--wait", type=int, default=7200, help="Seconds to wait if another job builds cache")
    args = ap.parse_args()

    if args.container:
        containers = [normalize_container(c) for c in args.container]
    elif args.signal:
        containers = containers_for_campaign(args.signal, args.datasets_file, None)
    else:
        ap.error("Provide --signal top|W or --container <name>")

    prefer = args.prefer or DEFAULT_PREFER

    if args.write_cache:
        pfns = ensure_cache(args.write_cache, containers, prefer, args.max_files, args.wait)
    else:
        pfns = resolve_all(containers, prefer, args.max_files)

    if args.print_paths:
        print(repr(pfns))
    elif not args.write_cache:
        for p in pfns:
            print(p)


if __name__ == "__main__":
    main()

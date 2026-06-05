#!/usr/bin/env python3
"""
probe_grid_sites.py – Probe ATLAS grid sites for CPU compatibility and XRootD URL formats.

For each replica of a test file the script:
  1. Retrieves all PFNs via `rucio list-file-replicas --pfns`.
  2. (Optional) cross-checks the XRootD endpoint against CRIC to see the
     site's declared cpu_architecture_level.
  3. Classifies the URL pattern (standard / nested-redirector / colon-in-path).
  4. Attempts an actual XRootD open (stat only – no data transferred).

Usage (on hypatia or lxplus):
    setupATLAS && lsetup rucio xrootd
    python3 grid/probe_grid_sites.py --did user.yuanda:user.yuanda.50577050._000001.ANALYSIS.root

Or probe all replicas of the first file in a dataset:
    python3 grid/probe_grid_sites.py --dataset user.yuanda.50577050

Options:
    --did DID           Rucio DID (scope:name) of a single file to probe
    --dataset DS        Dataset name; script picks the first file automatically
    --min-arch ARCH     Minimum cpu_architecture_level to pass CRIC filter
                        (default: x86_64-v3; set to '' to skip CRIC check)
    --no-open           Skip the actual XRootD open test (faster, offline)
    --timeout N         XRootD open timeout in seconds (default: 15)
"""

import argparse
import json
import re
import ssl
import subprocess
import sys
import urllib.request
from collections import defaultdict


# ---------------------------------------------------------------------------
# URL helpers (same logic as Make_data_LRJ._xrootd_url_variants)
# ---------------------------------------------------------------------------

def _encode_path_colons(url):
    m = re.match(r'^(root://[^/]+)(/.+)$', url)
    if m:
        authority, path = m.group(1), m.group(2)
        if ':' in path:
            return authority + path.replace(':', '%3A')
    return url


def _unwrap_nested(url):
    m = re.match(r'^root://[^/]+//(root://.+)$', url)
    return m.group(1) if m else url


def classify_url(url):
    """Return a short label for the URL pattern."""
    if not isinstance(url, str):
        return "unknown"
    if not url.startswith("root://"):
        return "non-xrootd"
    if re.match(r'^root://[^/]+//root://', url):
        return "nested-redirector"
    m = re.match(r'^root://[^/]+(/.+)$', url)
    if m and ':' in m.group(1):
        return "colon-in-path"
    return "standard"


def xrootd_url_variants(url):
    """Same adaptive variant list as in Make_data_LRJ."""
    if not (isinstance(url, str) and url.startswith("root://")):
        return [url]
    seen, variants = set(), []
    def add(u):
        if u and u not in seen:
            seen.add(u); variants.append(u)
    u1 = _unwrap_nested(url)
    u2 = _encode_path_colons(url)
    u3 = _encode_path_colons(u1)
    add(u1); add(u2); add(u3); add(url)
    return variants


# ---------------------------------------------------------------------------
# CRIC helpers
# ---------------------------------------------------------------------------

# Try multiple CRIC endpoints in order (API paths change between CRIC versions)
_CRIC_URLS = [
    "https://atlas-cric.cern.ch/api/atlas/pandaqueue/query/?json",
    "https://atlas-cric.cern.ch/core/pandaqueue/list/?json",
    "https://atlas-cric.cern.ch/api/atlas/pandaqueue/query/?json&preset=schedconf",
]

def fetch_cric_queues():
    """Return a dict of {queue_name: info} from CRIC, or {} on failure."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    for url in _CRIC_URLS:
        try:
            with urllib.request.urlopen(url, context=ctx, timeout=20) as resp:
                data = json.loads(resp.read())
                print(f"      CRIC OK ({url})")
                return data
        except Exception as exc:
            print(f"  WARNING: CRIC {url} failed ({exc})", file=sys.stderr)
    print("  WARNING: all CRIC endpoints failed; skipping CPU filter", file=sys.stderr)
    return {}


def build_endpoint_to_site_map(queues):
    """Map XRootD hostname fragments → (queue_name, cpu_arch) for quick lookup."""
    mapping = {}  # hostname → (queue_name, arch)
    for qname, info in queues.items():
        arch = info.get("cpu_architecture_level", "")
        # CRIC stores ddm_endpoints; we use the queue name as a rough proxy
        # and also try to match on the copytools/ddm_endpoints when available
        for ep in info.get("ddm_endpoints", {}).values():
            host = ep.get("se", "")
            if host:
                # strip port / path, keep hostname only
                host_only = re.sub(r':[0-9]+.*', '', host).lower()
                mapping[host_only] = (qname, arch)
        # Also store the queue name itself as a key (for direct matching)
        mapping[qname.lower()] = (qname, arch)
    return mapping


# ---------------------------------------------------------------------------
# Rucio helpers
# ---------------------------------------------------------------------------

def rucio_list_replicas_pfns(did):
    """Run `rucio list-file-replicas --pfns DID` and return list of PFNs."""
    result = subprocess.run(
        ["rucio", "list-file-replicas", "--pfns", did],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  ERROR: rucio list-file-replicas failed:\n{result.stderr}", file=sys.stderr)
        return []
    pfns = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith("root://") or line.startswith("gsiftp://"):
            pfns.append(line)
    return pfns


def rucio_list_dataset_files(dataset, limit=1):
    """Return up to `limit` DIDs from a dataset.

    Handles both scope:name and bare name inputs.
    Parses the pipe-delimited table that `rucio list-files` produces.
    """
    # Ensure the dataset has a scope prefix
    if ":" not in dataset:
        # Infer scope from the first component of the dataset name
        # e.g.  user.yuanda.50577050  ->  scope = user.yuanda
        parts = dataset.split(".")
        scope = ".".join(parts[:2]) if len(parts) >= 2 else parts[0]
        dataset_did = f"{scope}:{dataset}"
    else:
        dataset_did = dataset

    result = subprocess.run(
        ["rucio", "list-files", dataset_did],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  ERROR: rucio list-files failed:\n{result.stderr}", file=sys.stderr)
        return []

    dids = []
    for line in result.stdout.splitlines():
        # rucio table format:  | scope:name | GUID | ...
        # strip leading/trailing '|' and whitespace, then grab first token
        stripped = line.strip().strip("|").strip()
        m = re.match(r'^(\S+:\S+)', stripped)
        if m:
            did = m.group(1)
            # Skip header/separator rows
            if did.startswith("-") or did.lower() == "scope:name":
                continue
            dids.append(did)
            if len(dids) >= limit:
                break
    return dids


# ---------------------------------------------------------------------------
# XRootD open test
# ---------------------------------------------------------------------------

def test_open(url, timeout=15):
    """
    Try to open the file with the XRootD Python client (stat only).
    Returns (ok: bool, error_msg: str).
    """
    try:
        from XRootD import client as xrd_client  # type: ignore
        f = xrd_client.File()
        status, _ = f.open(url, timeout=timeout)
        f.close()
        if status.ok:
            return True, ""
        return False, status.message
    except ImportError:
        # Fall back to uproot (heavier but always available in LCG env)
        try:
            import uproot
            with uproot.open(url) as _:
                pass
            return True, ""
        except Exception as exc:
            return False, str(exc)
    except Exception as exc:
        return False, str(exc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--did", nargs="+", help="Rucio DID(s) (scope:name) of file(s) to probe")
    g.add_argument("--dataset", help="Dataset name; probes --nfiles files")
    p.add_argument("--nfiles", type=int, default=1,
                   help="Number of files to probe from dataset (default: 1)")
    p.add_argument("--min-arch", default="x86_64-v3",
                   help="Minimum cpu_architecture_level (default: x86_64-v3; '' to skip)")
    p.add_argument("--no-open", action="store_true",
                   help="Skip actual XRootD open test")
    p.add_argument("--timeout", type=int, default=15,
                   help="XRootD open timeout in seconds (default: 15)")
    return p.parse_args()


def main():
    args = parse_args()

    # 1. Resolve DIDs
    if args.dataset:
        print(f"[1/4] Listing files in dataset {args.dataset} ...")
        dids = rucio_list_dataset_files(args.dataset, limit=args.nfiles)
        if not dids:
            print("ERROR: no files found in dataset", file=sys.stderr)
            sys.exit(1)
        print(f"      Using {len(dids)} file(s)")
    else:
        dids = args.did

    # 2. Get PFNs (aggregate across all DIDs)
    print(f"[2/4] Fetching replicas for {len(dids)} file(s) ...")
    pfns = []
    for did in dids:
        p = rucio_list_replicas_pfns(did)
        print(f"      {did}: {len(p)} replica(s)")
        pfns.extend(p)
    # Deduplicate by hostname (keep first PFN per host to avoid redundancy)
    seen_hosts, unique_pfns = set(), []
    for pfn in pfns:
        m = re.match(r'^root://([^:/]+)', pfn)
        host = m.group(1) if m else pfn
        if host not in seen_hosts:
            seen_hosts.add(host)
            unique_pfns.append(pfn)
    pfns = unique_pfns
    if not pfns:
        print("ERROR: no PFNs found (is rucio set up? are these valid DIDs?)", file=sys.stderr)
        sys.exit(1)
    print(f"      {len(pfns)} unique host(s) to probe")

    # 3. Fetch CRIC data (optional)
    cric_queues = {}
    ep_map = {}
    if args.min_arch:
        print(f"[3/4] Fetching CRIC site data (min arch: {args.min_arch}) ...")
        cric_queues = fetch_cric_queues()
        if cric_queues:
            ep_map = build_endpoint_to_site_map(cric_queues)
            print(f"      Loaded {len(cric_queues)} PanDA queues from CRIC")
    else:
        print("[3/4] Skipping CRIC check (--min-arch not set)")

    # 4. Probe each replica
    print(f"[4/4] Probing replicas ...")
    print()

    results = []  # list of dicts

    for pfn in pfns:
        pattern = classify_url(pfn)
        variants = xrootd_url_variants(pfn)

        # CRIC lookup: try to match hostname to a PanDA queue
        host_match = re.match(r'^root://([^:/]+)', pfn)
        hostname = host_match.group(1).lower() if host_match else ""
        cric_queue, cric_arch = ep_map.get(hostname, ("?", "?"))

        # CPU filter
        arch_ok = True
        if args.min_arch and cric_arch != "?":
            arch_ok = cric_arch >= args.min_arch

        # XRootD open test
        open_ok, open_err, open_variant = None, "", pfn
        if not args.no_open:
            for v in variants:
                ok, err = test_open(v, timeout=args.timeout)
                if ok:
                    open_ok = True
                    open_variant = v
                    break
                open_err = err
            else:
                open_ok = False

        results.append({
            "pfn": pfn,
            "pattern": pattern,
            "hostname": hostname,
            "cric_queue": cric_queue,
            "cric_arch": cric_arch,
            "arch_ok": arch_ok,
            "open_ok": open_ok,
            "open_variant": open_variant,
            "open_err": open_err,
        })

        arch_sym = ("✓" if arch_ok else "✗") if cric_arch != "?" else "?"
        open_sym = ("✓" if open_ok else ("✗" if open_ok is False else "-"))

        print(f"  [{arch_sym} cpu] [{open_sym} open]  {pattern:20s}  {hostname}")
        if cric_queue != "?":
            print(f"    CRIC queue: {cric_queue}  arch: {cric_arch}")
        if open_ok is False:
            print(f"    OPEN ERROR: {open_err[:120]}")
        if open_ok and open_variant != pfn:
            print(f"    URL fix applied: {open_variant[:100]}")

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    by_pattern = defaultdict(list)
    for r in results:
        by_pattern[r["pattern"]].append(r)

    for pat, rs in sorted(by_pattern.items()):
        working = [r for r in rs if r["open_ok"] is True]
        failed  = [r for r in rs if r["open_ok"] is False]
        print(f"\nPattern: {pat}  ({len(rs)} replica(s))")
        for r in rs:
            cpu_str  = f"cpu={r['cric_arch']}" if r['cric_arch'] != '?' else "cpu=unknown"
            open_str = ("open=OK" if r['open_ok'] else
                        ("open=FAIL" if r['open_ok'] is False else "open=untested"))
            print(f"  {r['hostname']:45s}  {cpu_str:15s}  {open_str}")

    good = [r for r in results
            if (r["arch_ok"] or not args.min_arch)
            and (r["open_ok"] is not False)]
    print(f"\nReplicas passing all checks: {len(good)}/{len(results)}")
    if good:
        patterns_seen = sorted({r['pattern'] for r in good})
        print(f"URL patterns that work:      {', '.join(patterns_seen)}")


if __name__ == "__main__":
    main()

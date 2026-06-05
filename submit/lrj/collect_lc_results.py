"""
Collect and summarise results from the unified LRJ LundNet pipeline.

Outputs:
  1. lund_only Learning Curve: AUC vs N_parts (N=5..10)
  2. Full-data experiment comparison (N=10) for top or W tagging

Usage:
  python submit/lrj/collect_lc_results.py --signal top \\
      --score_base /share/lustre/yzhu/IOP_DATA/score_lrj_TopTagging_LRJ

  python submit/lrj/collect_lc_results.py --signal W \\
      --score_base /share/lustre/yzhu/IOP_DATA/score_lrj_WTagging_LRJ
"""

import argparse
import csv
import glob
import os

import numpy as np
import uproot


# ---- Experiment matrices ------------------------------------------------

LC_N_PARTS = [5, 6, 7, 8, 9, 10]   # learning curve points for lund_only

TOP_EXPERIMENTS = [
    ("lund_only",    "only",    1),
    ("lund_toptrans","toptrans",2),
    ("lund_bWP65",   "bWP65",   2),
    ("lund_bWP70",   "bWP70",   2),
    ("lund_bWP77",   "bWP77",   2),
    ("lund_bWP85",   "bWP85",   2),
    ("lund_bWP90",   "bWP90",   2),
    ("lund_fb65",    "fb65",    1),
    ("lund_fb70",    "fb70",    1),
    ("lund_fb77",    "fb77",    1),
    ("lund_fb85",    "fb85",    1),
    ("lund_fb90",    "fb90",    1),
]

W_EXPERIMENTS = [
    ("lund_only",    "only",    1),
    ("lund_wtrans",  "wtrans",  2),
    ("lund_cWP10",   "cWP10",   2),
    ("lund_cWP30",   "cWP30",   2),
    ("lund_cWP50",   "cWP50",   2),
    ("lund_fc10",    "fc10",    1),
    ("lund_fc30",    "fc30",    1),
    ("lund_fc50",    "fc50",    1),
]

TREE = "FlatSubstructureJetTree"


# -------------------------------------------------------------------------

def roc_auc(y_true, y_score):
    """Trapezoidal ROC AUC (no sklearn dependency)."""
    y_true  = np.asarray(y_true,  dtype=np.float32)
    y_score = np.asarray(y_score, dtype=np.float32)
    order   = np.argsort(y_score)[::-1]
    y_true  = y_true[order]
    n_pos   = y_true.sum()
    n_neg   = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    tpr = np.cumsum(y_true) / n_pos
    fpr = np.cumsum(1 - y_true) / n_neg
    return float(np.trapz(tpr, fpr))


def load_auc(score_dir, tag):
    """Load all *_scored.root in score_dir and return (auc, n_jets)."""
    roots = sorted(glob.glob(os.path.join(score_dir, "*_scored.root")))
    if not roots:
        return None, 0
    labels_all, scores_all = [], []
    branch = f"LRJ_{tag}_score"
    for r in roots:
        try:
            with uproot.open(r) as f:
                if TREE not in f:
                    continue
                t = f[TREE]
                labels_all.append(t["labels"].array(library="np"))
                scores_all.append(t[branch].array(library="np"))
        except Exception as exc:
            print(f"  WARNING: could not read {r}: {exc}")
    if not labels_all:
        return None, 0
    labels = np.concatenate(labels_all)
    scores = np.concatenate(scores_all)
    return roc_auc(labels, scores), len(labels)


def score_dir_for(score_base, mode, n_parts):
    return os.path.join(score_base, f"exp_{mode}", f"nparts{n_parts}")


def fmt_auc(auc):
    return f"{auc:.5f}" if auc is not None and not np.isnan(auc) else "N/A"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--signal",     required=True, choices=["top", "W"],
                   help="Signal type: top or W")
    p.add_argument("--score_base", required=True,
                   help="Base dir, e.g. /share/.../score_lrj_TopTagging_LRJ")
    p.add_argument("--csv_out",    default=None,
                   help="Optional CSV output path (default: <score_base>/summary_<signal>.csv)")
    return p.parse_args()


def main():
    args = parse_args()
    signal     = args.signal
    score_base = args.score_base
    experiments = TOP_EXPERIMENTS if signal == "top" else W_EXPERIMENTS

    print(f"\n{'='*60}")
    print(f"  LRJ LundNet results — signal={signal}")
    print(f"  score_base: {score_base}")
    print(f"{'='*60}\n")

    # ---------------------------------------------------------------
    # Part 1: lund_only learning curve (N = 5..10)
    # ---------------------------------------------------------------
    print("--- lund_only Learning Curve ---")
    print(f"  {'N_parts':>8}  {'train_frac':>12}  {'n_jets':>9}  {'AUC':>9}")
    print("  " + "-" * 46)

    lc_rows = []
    for n in LC_N_PARTS:
        sd  = score_dir_for(score_base, "lund_only", n)
        auc, n_jets = load_auc(sd, "only")
        row = {"mode": "lund_only", "n_parts": n,
               "train_frac": f"{n*7}%", "n_jets": n_jets, "AUC": auc}
        lc_rows.append(row)
        print(f"  {n:>8}  {row['train_frac']:>12}  {n_jets:>9}  {fmt_auc(auc):>9}")

    # ---------------------------------------------------------------
    # Part 2: Full-data (N=10) experiment comparison
    # ---------------------------------------------------------------
    print(f"\n--- Full-data experiments (N=10), signal={signal} ---")
    print(f"  {'experiment':<20}  {'n_jets':>9}  {'AUC':>9}")
    print("  " + "-" * 44)

    exp_rows = []
    for mode, tag, _dim in experiments:
        sd  = score_dir_for(score_base, mode, 10)
        auc, n_jets = load_auc(sd, tag)
        row = {"mode": mode, "n_parts": 10, "train_frac": "70%",
               "n_jets": n_jets, "AUC": auc}
        exp_rows.append(row)
        print(f"  {mode:<20}  {n_jets:>9}  {fmt_auc(auc):>9}")

    # ---------------------------------------------------------------
    # Save CSV
    # ---------------------------------------------------------------
    csv_path = args.csv_out or os.path.join(score_base, f"summary_{signal}.csv")
    all_rows = lc_rows + [r for r in exp_rows if r["mode"] != "lund_only"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["mode", "n_parts", "train_frac", "n_jets", "AUC"])
        writer.writeheader()
        for r in all_rows:
            row_out = dict(r)
            row_out["AUC"] = fmt_auc(r["AUC"])
            writer.writerow(row_out)

    print(f"\nCSV saved → {csv_path}\n")


if __name__ == "__main__":
    main()

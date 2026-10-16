#!/usr/bin/env python3
"""Tagger comparison with per-pT-bin working points (PUB-2018-014 style).

Replaces the inclusive-WP figures, which on the balanced test set conflate
discrimination with cross-pT score calibration and run out of weighted
statistics for strongly decorrelated models.

Figures (into --outdir):
  roc_ptbins.png            weighted ROC inside four pT windows
  rejection_vs_pt_exact.png weighted rej @ per-bin 50% WP (exact quantile
                            thresholds, no sigmoid fit) with stat errors
  mass_sculpting_binnedWP.png  bkg mass pass/fail at the pT-parameterized WP
  pareto_binnedWP.png       1/JSD vs overall rejection at the pT-param. WP
  matrix_binnedWP.md
"""
import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/yuanda/lundtoptagger")
sys.path.insert(0, "/home/yuanda/lundtoptagger/submit/lrj")
from plot_lundtag_compare_styled import (load_data_v9, W_CURVES, TOP_CURVES,
                                         MASSBINS, jsd as jsd_hist)  # noqa: E402

_trapz = getattr(np, "trapezoid", None) or np.trapz

PT_WINDOWS = [(200, 500), (500, 1000), (1000, 1500), (1500, 2500)]
PT_BINS = np.array([200, 300, 400, 500, 600, 800, 1000, 1200, 1500, 2000,
                    2500, 3100], dtype=float)


def weighted_roc(s, y, w, n_grid=400):
    """Weighted ROC on a score array: returns eff_s, eff_b arrays and AUC."""
    sig, bkg = y == 1, y == 0
    order = np.argsort(s)[::-1]
    ys, ws = y[order], w[order]
    cs = np.cumsum(ws * (ys == 1)); cb = np.cumsum(ws * (ys == 0))
    tot_s, tot_b = cs[-1], cb[-1]
    eff_s, eff_b = cs / tot_s, cb / tot_b
    # AUC via trapezoid on the full curve
    auc = float(_trapz(eff_s, eff_b))
    # thin out for plotting
    idx = np.unique(np.linspace(0, len(eff_s) - 1, n_grid).astype(int))
    return eff_s[idx], eff_b[idx], auc


def wp_threshold(s_sig, w_sig, eff=0.5):
    order = np.argsort(s_sig)[::-1]
    cw = np.cumsum(w_sig[order])
    idx = min(np.searchsorted(cw, eff * cw[-1]), len(order) - 1)
    return s_sig[order[idx]]


# ---------------------------------------------------------------------------
# v3: pubnote H5 test shards scored by tools/score_h5_multi.py
# ---------------------------------------------------------------------------
SPLITS = "/share/lustre/yzhu/IOP_DATA/LargeR_SRJtag_splits_Aug2026"
V3_PT_BINS = np.array([350, 400, 500, 600, 800, 1000, 1200, 1500, 2000,
                       2500, 3100], dtype=float)
V3_PT_WINDOWS = [(350, 500), (500, 1000), (1000, 1500), (1500, 3100)]
V3_MASSBINS = {"W": np.arange(40, 300 + 5, 5, dtype=float),
               "top": np.arange(40, 600 + 10, 10, dtype=float)}
V3_SIG = {"W": ("wz_801859", ("label_W", "label_Z")), "top": ("top", ("label_top",))}

_V3_COMMON = [
    ("score_pubnote",               "LundNet (pubnote, lr 4e-4)",  "0.45",    "--", "x"),
    ("score_lund_only",             "LundNet",                     "#1f77b4", "-",  "o"),
    ("score_lund_former",           "LundFormer",                  "#d62728", "-",  "o"),
    ("score_lund_ann_lam5",         r"LundNet ANN $\lambda$=5",    "#a1d99b", "-",  "v"),
    ("score_lund_ann_lam25",        r"LundNet ANN $\lambda$=25",   "#41ab5d", "-",  "v"),
    ("score_lund_ann_lam100",       r"LundNet ANN $\lambda$=100",  "#006d2c", "-",  "v"),
    ("score_lund_former_ann_lam25", r"LundFormer ANN $\lambda$=25", "#7b3294", "-", "^"),
    ("score_lund_former_disco_lam10",  r"LundFormer DisCo $\lambda$=10",  "#fdae6b", "-", "D"),
    ("score_lund_former_disco_lam100", r"LundFormer DisCo $\lambda$=100", "#e6550d", "-", "D"),
]
V3_W_CURVES = _V3_COMMON + [
    ("WTrans_score",             "ParT",                        "k",       ":",  "*"),
    ("WTrans_massdec",           "ParT (mass-dec.)",            "0.3",     ":",  "P"),
    ("score_lund_cWP50",         "LundNet + c-tag WP50",        "#1f77b4", "-.", "s"),
    ("score_lund_former_cWP50",  "LundFormer + c-tag WP50",     "#d62728", "-.", "s"),
    ("score_lund_md",            "LundNet flat-mass",           "#8c510a", "-",  "h"),
    ("score_lund_former_md",     "LundFormer flat-mass",        "#c51b7d", "-",  "h"),
    ("score_pubnote_md",         "LundNet flat-mass (pubnote)", "#8c510a", "--", "h"),
]
V3_TOP_CURVES = _V3_COMMON + [
    ("TopTrans_score",           "ParT",                        "k",       ":",  "*"),
    ("score_lund_bWP77",         "LundNet + b-tag WP77",        "#1f77b4", "-.", "s"),
    ("score_lund_former_bWP77",  "LundFormer + b-tag WP77",     "#d62728", "-.", "s"),
]


def _gev(x):
    x = x.astype(np.float64)
    return np.where(x > 1e4, x / 1000.0, x)


def _key(pt, mass):
    """Exact (pt, mass) key from the float32 bit patterns of the GeV values."""
    pb = pt.astype(np.float32).view(np.int32).astype(np.int64)
    mb = mass.astype(np.float32).view(np.int32).astype(np.int64)
    return (pb << 32) | (mb & 0xFFFFFFFF)


def _match(key, raw_key, raw_w):
    """Index into raw for each key; -1 if absent or ambiguous (duplicate
    raw keys with different weights)."""
    order = np.argsort(raw_key, kind="stable")
    rk, rw = raw_key[order], raw_w[order]
    uk, first, cnt = np.unique(rk, return_index=True, return_counts=True)
    wmax = np.maximum.reduceat(rw, first)
    wmin = np.minimum.reduceat(rw, first)
    ok_key = (cnt == 1) | (wmax == wmin)
    pos = np.clip(np.searchsorted(uk, key), 0, len(uk) - 1)
    hit = (uk[pos] == key) & ok_key[pos]
    idx = np.where(hit, order[first[pos]], -1)
    return idx, int((cnt > 1).sum()), int((~ok_key).sum())


def load_v3(scores_h5, channel, sig_def="WZ"):
    """Per-jet scores + physics weights matched from the raw test_1M.h5.

    QCD weight = raw `weight` (xsec-normalised mcEventWeight); signal = 1
    (flat-pT generation). Matching is per class on exact (pt, mass); raw
    duplicates are kept only if their weights agree.
    """
    import h5py
    with h5py.File(scores_h5, "r") as f:
        cols = {k: f[k][:] for k in f.keys()}
    pt, mass, y = cols["pt"], cols["mass"], cols["y"].astype(int)

    sig_dir, labels = V3_SIG[channel]
    with h5py.File(f"{SPLITS}/{sig_dir}/test_1M.h5", "r") as f:
        js = f["jets"][:]
    with h5py.File(f"{SPLITS}/qcd/test_1M.h5", "r") as f:
        jb = f["jets"][:]
    keep = np.zeros(len(js), bool)
    for lab in labels:
        keep |= js[lab].astype(bool)
    js = js[keep]
    k = _key(pt, mass)
    w = np.full(len(y), np.nan)
    isW = np.zeros(len(y), bool)
    for c, lab, jr, rw in ((1, "sig", js, np.ones(len(js))),
                           (0, "bkg", jb, jb["weight"].astype(np.float64))):
        m = y == c
        if c == 1:   # W/Z truth must agree too, so key it into the weight check
            rw = rw + 10.0 * js["label_W"].astype(bool)
        idx, ndup, namb = _match(k[m], _key(_gev(jr["pt"]), _gev(jr["mass"])), rw)
        hit = idx >= 0
        print(f"  {lab}: raw {len(jr):,}, dup keys {ndup:,} (ambiguous {namb:,}); "
              f"matched {hit.sum():,}/{m.sum():,} ({100 * hit.mean():.3f}%)")
        wm = np.full(m.sum(), np.nan)
        if c == 1:
            wm[hit] = 1.0
            isW[np.flatnonzero(m)[hit]] = js["label_W"].astype(bool)[idx[hit]]
        else:
            wm[hit] = rw[idx[hit]]
        w[m] = wm
    found = np.isfinite(w)

    sel = found & (pt >= V3_PT_BINS[0]) & (pt < V3_PT_BINS[-1])
    if channel == "W" and sig_def == "Wonly":
        sel &= (y == 0) | isW
    if channel == "W":
        print(f"  signal in eval range: W {int((sel & isW).sum()):,}, "
              f"Z {int((sel & (y == 1) & ~isW).sum()):,}")
    data = {k_: v[sel] for k_, v in cols.items()}
    data.update(pt=pt[sel], mass=mass[sel], labels=y[sel], weights=w[sel])
    return data


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--merged", help="v2 merged ROOT scores (v9 weights)")
    src.add_argument("--scores-h5", help="v3 scores_{ch}.h5 from tools/score_h5_multi.py")
    ap.add_argument("--channel", choices=["W", "top"], default="W")
    ap.add_argument("--sig-def", choices=["WZ", "Wonly"], default="WZ",
                    help="v3 W channel: W+Z signal (pubnote) or W only")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    if args.scores_h5:
        data = load_v3(args.scores_h5, args.channel, args.sig_def)
        curves = V3_W_CURVES if args.channel == "W" else V3_TOP_CURVES
        sig_txt = {"WZ": "W+Z sig", "Wonly": "W-only sig"}[args.sig_def] \
            if args.channel == "W" else "top sig"
        run_plots(data, curves, args.channel, args.outdir,
                  pt_bins=V3_PT_BINS, pt_windows=V3_PT_WINDOWS,
                  massbins=V3_MASSBINS[args.channel],
                  wtag=f"v3 pubnote-H5 test, phys. weights, {sig_txt}")
        return

    curves = W_CURVES if args.channel == "W" else TOP_CURVES
    data = load_data_v9(args.merged, args.channel, bkg_def="qcdonly")
    if np.nanmedian(data["pt"]) > 1e4:
        data["pt"], data["mass"] = data["pt"] / 1e3, data["mass"] / 1e3
    run_plots(data, curves, args.channel, args.outdir)


def run_plots(data, curves, channel, outdir, pt_bins=PT_BINS,
              pt_windows=PT_WINDOWS, massbins=MASSBINS,
              wtag="v9 weights, QCD-only bkg"):
    PT_BINS, PT_WINDOWS, MASSBINS = pt_bins, pt_windows, massbins
    pt, mass, y, w = data["pt"], data["mass"], data["labels"], data["weights"]
    curves = [c for c in curves if c[0] in data]
    print(f"jets {len(y):,}  sig frac {float(np.average(y==1)):.3f}  models {len(curves)}")

    # ---------- Fig 1: ROC in pT windows ----------
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharex=True)
    for ax, (lo, hi) in zip(axes.ravel(), PT_WINDOWS):
        m = (pt >= lo) & (pt < hi)
        for br, lbl, col, ls, mk in curves:
            s = data[br][m]
            ok = np.isfinite(s)
            es, eb, auc = weighted_roc(s[ok], y[m][ok], w[m][ok])
            keep = es > 0.15
            with np.errstate(divide="ignore"):
                ax.plot(es[keep], 1.0 / np.maximum(eb[keep], 1e-12),
                        color=col, ls=ls, lw=1.6, label=f"{lbl} ({auc:.3f})")
        ax.set_yscale("log"); ax.set_xlim(0.2, 1.0)
        ax.set_title(f"$p_T \\in [{lo}, {hi}]$ GeV", fontsize=11)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6.5, loc="upper right")
    for ax in axes[1]:
        ax.set_xlabel(r"$\varepsilon_{\rm sig}$ (weighted, in-window)")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$1/\varepsilon_{\rm bkg}$ (weighted, in-window)")
    fig.suptitle(f"{channel}-tagging ROC in pT windows — {wtag} "
                 f"(legend: in-window AUC)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "roc_ptbins.png"), dpi=130)
    plt.close(fig)
    print("saved roc_ptbins.png")

    # ---------- per-bin 50% WP thresholds (exact weighted quantile) ----------
    sig, bkg = y == 1, y == 0
    thresholds = {}   # br -> array over PT_BINS
    for br, *_ in curves:
        thr = []
        s = data[br]
        for i in range(len(PT_BINS) - 1):
            mb = sig & (pt >= PT_BINS[i]) & (pt < PT_BINS[i + 1]) & np.isfinite(s)
            thr.append(wp_threshold(s[mb], w[mb]) if mb.sum() > 100 else np.nan)
        thresholds[br] = np.array(thr)

    # ---------- Fig 2: rejection vs pT with stat errors ----------
    fig, ax = plt.subplots(figsize=(10, 7))
    centers = 0.5 * (PT_BINS[:-1] + PT_BINS[1:])
    xerr = 0.5 * np.diff(PT_BINS)
    summary_rows = []
    for br, lbl, col, ls, mk in curves:
        s = data[br]
        rej, rerr = [], []
        for i in range(len(PT_BINS) - 1):
            mb = bkg & (pt >= PT_BINS[i]) & (pt < PT_BINS[i + 1]) & np.isfinite(s)
            wb = w[mb]; tot = wb.sum()
            p = s[mb] >= thresholds[br][i]
            wp_ = wb[p]
            if not np.isfinite(thresholds[br][i]) or wp_.sum() <= 0:
                rej.append(np.nan); rerr.append(0.0); continue
            r = tot / wp_.sum()
            rej.append(r)
            rerr.append(r * np.sqrt((wp_ ** 2).sum()) / wp_.sum())
        ax.errorbar(centers, rej, yerr=rerr, xerr=xerr, color=col, ls="none",
                    marker=mk, ms=4, lw=1.2, capsize=0, label=lbl)
    ax.set_yscale("log"); ax.set_xlabel(r"Large-$R$ jet $p_T$ [GeV]")
    ax.set_ylabel(r"bkg rejection @ $\varepsilon_{\rm sig}$=50% per bin (weighted)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title(f"{channel}-tagging — exact per-bin WPs, {wtag}, stat errors")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "rejection_vs_pt_exact.png"), dpi=130)
    plt.close(fig)
    print("saved rejection_vs_pt_exact.png")

    # ---------- piecewise (pT-parameterized) WP: global pass masks ----------
    fig, ax = plt.subplots(figsize=(10, 7))
    h_incl, _ = np.histogram(mass[bkg], bins=MASSBINS, weights=w[bkg])
    binw = np.diff(MASSBINS)
    ax.fill_between(0.5 * (MASSBINS[:-1] + MASSBINS[1:]),
                    h_incl / h_incl.sum() / binw, step="mid", color="0.8",
                    label="Inclusive background")
    pareto = []
    for br, lbl, col, ls, mk in curves:
        s = data[br]
        ibin = np.clip(np.digitize(pt, PT_BINS) - 1, 0, len(PT_BINS) - 2)
        thr_per_jet = thresholds[br][ibin]
        passed = np.isfinite(s) & np.isfinite(thr_per_jet) & (s >= thr_per_jet)
        bp = bkg & passed; bf = bkg & ~passed
        hp, _ = np.histogram(mass[bp], bins=MASSBINS, weights=w[bp])
        hf, _ = np.histogram(mass[bf], bins=MASSBINS, weights=w[bf])
        j = jsd_hist(hp, hf)
        wb_pass = w[bp]
        rej = w[bkg].sum() / max(wb_pass.sum(), 1e-12)
        rej_err = rej * np.sqrt((wb_pass ** 2).sum()) / max(wb_pass.sum(), 1e-12)
        pareto.append((lbl, col, mk, rej, rej_err, j))
        summary_rows.append((lbl, rej, rej_err, j))
        if hp.sum() > 0:
            ax.step(0.5 * (MASSBINS[:-1] + MASSBINS[1:]),
                    hp / hp.sum() / binw, where="mid", color=col, ls=ls,
                    lw=1.4, label=f"{lbl} (JSD {j:.3f})")
    ax.set_xlabel(r"Large-$R$ jet mass [GeV]")
    ax.set_ylabel(r"norm. bkg density @ pT-param. 50% WP")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    ax.set_title(f"{channel} — mass sculpting at the pT-parameterized WP")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "mass_sculpting_binnedWP.png"), dpi=130)
    plt.close(fig)
    print("saved mass_sculpting_binnedWP.png")

    # ---------- Fig 4: Pareto ----------
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    for lbl, col, mk, rej, rej_err, j in pareto:
        inv_j = 1.0 / max(j, 1e-4)
        ax.errorbar([rej], [inv_j], xerr=[rej_err], color=col, marker=mk,
                    ms=9, capsize=3, label=lbl, ls="none",
                    markeredgecolor="k", markeredgewidth=0.5)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"overall bkg rejection @ pT-parameterized 50% WP (weighted)")
    ax.set_ylabel("1 / JSD (mass decorrelation)")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    ax.set_title(f"{channel} — decorrelation vs performance (per-bin WP definition)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "pareto_binnedWP.png"), dpi=130)
    plt.close(fig)
    print("saved pareto_binnedWP.png")

    # ---------- Fig 5: inclusive ROC at pT-parameterized WPs ----------
    # For each target eff, thresholds are the per-pT-bin signal weighted
    # quantiles; overall weighted sig eff is then exactly the target, and the
    # overall weighted bkg rejection measures discrimination only (no
    # cross-pT calibration artifact).
    eff_grid = np.linspace(0.15, 0.97, 42)
    fig, ax = plt.subplots(figsize=(9.5, 7))
    roc_rows = []
    for br, lbl, col, ls, mk in curves:
        s = data[br]
        ok = np.isfinite(s)
        pass_w = np.zeros_like(eff_grid)
        pass_w2 = np.zeros_like(eff_grid)
        tot_b = 0.0
        for i in range(len(PT_BINS) - 1):
            inb = (pt >= PT_BINS[i]) & (pt < PT_BINS[i + 1]) & ok
            ms = sig & inb
            mb = bkg & inb
            tot_b += w[mb].sum()
            if ms.sum() < 100 or mb.sum() == 0:
                continue
            o = np.argsort(s[ms])[::-1]
            ss, cw = s[ms][o], np.cumsum(w[ms][o])
            thr = ss[np.minimum(np.searchsorted(cw, eff_grid * cw[-1]),
                                len(ss) - 1)]
            ob = np.argsort(s[mb])[::-1]
            sb, wb = s[mb][ob], w[mb][ob]
            cb = np.cumsum(wb)
            cb2 = np.cumsum(wb ** 2)
            npass = np.searchsorted(-sb, -thr, side="right")
            nz = npass > 0
            pass_w[nz] += cb[npass[nz] - 1]
            pass_w2[nz] += cb2[npass[nz] - 1]
        rej = tot_b / np.maximum(pass_w, 1e-12)
        rerr = rej * np.sqrt(pass_w2) / np.maximum(pass_w, 1e-12)
        ax.plot(eff_grid, rej, color=col, ls=ls, lw=1.6, label=lbl)
        ax.fill_between(eff_grid, rej - rerr, rej + rerr, color=col,
                        alpha=0.15, lw=0)
        i50 = int(np.argmin(np.abs(eff_grid - 0.5)))
        i80 = int(np.argmin(np.abs(eff_grid - 0.8)))
        roc_rows.append((lbl, rej[i50], rerr[i50], rej[i80], rerr[i80]))
    ax.axvline(0.5, color="0.6", lw=0.8, ls=":")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\varepsilon_{\rm sig}$ (weighted, inclusive; per-bin quantile thresholds)")
    ax.set_ylabel(r"inclusive bkg rejection $1/\varepsilon_{\rm bkg}$ (weighted)")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    ax.set_title(f"{channel} — inclusive ROC at pT-parameterized WPs "
                 f"({wtag}; band = stat)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "roc_inclusive_ptparam.png"), dpi=130)
    plt.close(fig)
    print("saved roc_inclusive_ptparam.png")

    # ---------- matrix ----------
    with open(os.path.join(outdir, "matrix_binnedWP.md"), "w") as f:
        f.write(f"# {channel} comparison — pT-parameterized 50% WP "
                f"(per-bin exact thresholds), {wtag}\n\n")
        f.write("| Model | Rejection @ WP | stat err | JSD | 1/JSD |\n|---|---|---|---|---|\n")
        for lbl, rej, rej_err, j in summary_rows:
            f.write(f"| {lbl} | {rej:.1f} | ±{rej_err:.1f} | {j:.4f} | {1.0/max(j,1e-4):.1f} |\n")
        f.write("\n## Inclusive ROC @ pT-parameterized WPs\n\n")
        f.write("| Model | Rej @ eff 0.5 | Rej @ eff 0.8 |\n|---|---|---|\n")
        for lbl, r50, e50, r80, e80 in roc_rows:
            f.write(f"| {lbl} | {r50:.1f} ± {e50:.1f} | {r80:.2f} ± {e80:.2f} |\n")
    print("saved matrix_binnedWP.md")


if __name__ == "__main__":
    main()

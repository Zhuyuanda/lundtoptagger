import argparse
import csv
import gc
import json
import os
import glob
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from datetime import datetime
from torch_geometric.loader import DataLoader
from sklearn.utils import shuffle as sk_shuffle
import resource
import torch.multiprocessing

# DataLoader workers pass every PyG tensor attribute through a file descriptor
# by default; with num_workers>=4 this exhausts the 1024 soft limit on batch
# nodes ("Too many open files" -> "Pin memory thread exited unexpectedly").
torch.multiprocessing.set_sharing_strategy("file_system")
_soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
if _soft < _hard:
    resource.setrlimit(resource.RLIMIT_NOFILE, (_hard, _hard))

import numpy as np

from tools.GNN_model_weight.models import (
    LundNet_General, LundNet_FiLM, LundNet_Quantum, LundFormer,
    Adversary_new, mdn_loss_new,
)
from tools.GNN_model_weight.utils_newdata import load_yaml, JSD


def get_extra_features(data, mode):
    device = data.y.device
    batch_size = data.y.shape[0]

    def to_batch_tensor(val):
        if isinstance(val, (float, int)):
            return torch.full((batch_size, 1), float(val), device=device)
        res = val.view(-1, 1)
        if res.shape[0] == 1 and batch_size > 1:
            return res.expand(batch_size, 1)
        return res

    extra_list = [to_batch_tensor(data.Ntrk)]

    if mode in ("lund_only",
                "lund_fb65", "lund_fb70", "lund_fb77", "lund_fb85", "lund_fb90",
                "lund_fc10", "lund_fc30", "lund_fc50"):
        # Ntrk only — filtered modes use filter_attr at load time, no extra features
        pass
    elif mode == "lund_toptrans" or mode == "lund_part":
        extra_list.append(to_batch_tensor(getattr(data, "TopTrans_score", 0.0)))
    elif mode == "lund_wtrans":
        extra_list.append(to_batch_tensor(getattr(data, "WTrans_score", 0.0)))
    elif mode == "lund_bWP65":
        extra_list.append(to_batch_tensor(getattr(data, "has_reco_b_WP65", 0.0)))
    elif mode == "lund_bWP70":
        extra_list.append(to_batch_tensor(getattr(data, "has_reco_b_WP70", 0.0)))
    elif mode == "lund_bWP77":
        extra_list.append(to_batch_tensor(getattr(data, "has_reco_b_WP77", 0.0)))
    elif mode == "lund_bWP85":
        extra_list.append(to_batch_tensor(getattr(data, "has_reco_b_WP85", 0.0)))
    elif mode == "lund_bWP90":
        extra_list.append(to_batch_tensor(getattr(data, "has_reco_b_WP90", 0.0)))
    elif mode == "lund_cWP10":
        extra_list.append(to_batch_tensor(getattr(data, "has_reco_c_WP10", 0.0)))
    elif mode == "lund_cWP30":
        extra_list.append(to_batch_tensor(getattr(data, "has_reco_c_WP30", 0.0)))
    elif mode == "lund_cWP50":
        extra_list.append(to_batch_tensor(getattr(data, "has_reco_c_WP50", 0.0)))
    elif mode == "lund_film":
        extra_list.append(to_batch_tensor(getattr(data, "GN3X_phbb", 0.0)))
    elif mode == "lund_film_part":
        extra_list.append(to_batch_tensor(getattr(data, "TopTrans_score", 0.0)))
    elif mode == "lund_film_ptop":
        ptop = to_batch_tensor(getattr(data, "GN3X_ptop", 1e-6))
        extra_list.append(torch.log(ptop.clamp(min=1e-6)))
    elif mode == "lund_gn3x":
        gn3x_keys = [
            "GN3X_phtautauhad", "GN3X_phbb", "GN3X_phcc", "GN3X_ptop",
            "GN3X_pqcdbb", "GN3X_pqcdbx", "GN3X_pqcdcx", "GN3X_pqcdll", "GN3X_pWqq",
        ]
        for k in gn3x_keys:
            extra_list.append(to_batch_tensor(getattr(data, k, 0.0)))

    return torch.cat(extra_list, dim=1).float()


# ---------------------------------------------------------------------------
# Streaming helpers
# ---------------------------------------------------------------------------

def _count_shard_weights(args):
    """Worker function: count sig/bkg weights in a single shard file (picklable)."""
    p, filter_attr = args
    shards = torch.load(p, weights_only=False)
    w_sig, w_bkg = 0.0, 0.0
    for g in shards:
        if filter_attr and float(getattr(g, filter_attr, 0)) < 0.5:
            continue
        w = float(g.weights)
        if float(g.y) == 1:
            w_sig += w
        else:
            w_bkg += w
    del shards
    gc.collect()
    return w_sig, w_bkg


def compute_scale_factor(shard_paths, filter_attr=None, n_workers=1):
    """Compute sig/bkg weight ratio across all shard files.

    Uses multiprocessing when n_workers > 1 to reduce wall time proportionally.
    Shard-level sampling is intentionally avoided because shards may be pure
    signal or pure background (depending on which ROOT file filled them).
    """
    task_args = [(p, filter_attr) for p in shard_paths]
    if n_workers > 1:
        with Pool(n_workers) as pool:
            results = pool.map(_count_shard_weights, task_args)
    else:
        results = [_count_shard_weights(a) for a in task_args]
    w_sig = sum(r[0] for r in results)
    w_bkg = sum(r[1] for r in results)
    if w_bkg == 0:
        raise ValueError("Background total weight is 0. Check your data (filter_attr too strict?).")
    return w_sig / w_bkg


def load_shards(paths, scale_factor, filter_attr=None):
    """Load a window of shard files; apply bkg scale_factor in-place.

    If filter_attr is set (e.g. 'has_reco_b_WP77'), only graphs where that
    attribute > 0.5 are kept — used for lund_fb*/lund_fc* experiments.
    """
    data_list = []
    for p in paths:
        shards = torch.load(p, weights_only=False)
        for g in shards:
            if filter_attr and float(getattr(g, filter_attr, 0)) < 0.5:
                continue
            g.weights = torch.tensor(float(g.weights))
            if float(g.y) == 0:
                g.weights = g.weights * scale_factor
            data_list.append(g)
        del shards
    gc.collect()
    return data_list


# ---------------------------------------------------------------------------
# Validation (no_grad)
# ---------------------------------------------------------------------------

VAL_METRIC_VERSION = "wclip99.9"   # selection metric: weights clipped at val p99.9


@torch.no_grad()
def validate(loader, model, device, mode, use_amp=False, weight_cap=None):
    """Weighted-BCE validation.

    Returns (loss_sel, loss_raw):
      loss_raw — physics-weighted BCE (historical metric; heavy weight tails
                 make it spike by orders of magnitude when a few high-weight
                 jets are misclassified).
      loss_sel — same but with weights clamped at `weight_cap` (val p99.9).
                 Used for best-model selection and early stopping.
    If weight_cap is None the two are identical.
    """
    model.eval()
    tot_sel, tot_raw, total_n = 0.0, 0.0, 0
    for data in loader:
        data = data.to(device)
        y = data.y.float().view(-1, 1)
        w_raw = data.weights.float().view(-1, 1)
        w_sel = w_raw.clamp(max=weight_cap) if weight_cap else w_raw
        if use_amp:
            with autocast("cuda"):
                extra = get_extra_features(data, mode)
                out = model(data, extra)
            out = out.float()
        else:
            extra = get_extra_features(data, mode)
            out = model(data, extra)
        # Compute loss in float32 — BCE is unsafe inside autocast
        loss_raw = F.binary_cross_entropy(out, y, weight=w_raw)
        loss_sel = (F.binary_cross_entropy(out, y, weight=w_sel)
                    if weight_cap else loss_raw)
        tot_raw += loss_raw.item() * data.num_graphs
        tot_sel += loss_sel.item() * data.num_graphs
        total_n += data.num_graphs
    if total_n == 0:
        return 0.0, 0.0
    return tot_sel / total_n, tot_raw / total_n


# ---------------------------------------------------------------------------
# Adversarial (GMM-ANN) mass decorrelation — PUB-2023-017 recipe
# ---------------------------------------------------------------------------

MASSBINS = np.linspace(40, 300, (300 - 40) // 5 + 1, endpoint=True)


def mass_bins_from_config(arch):
    """architecture.mass_bins: [lo, hi, step] for the JSD monitor (default 40-300/5)."""
    mb = arch.get("mass_bins")
    if not mb:
        return MASSBINS
    lo, hi, step = (float(v) for v in mb)
    return np.linspace(lo, hi, int(round((hi - lo) / step)) + 1, endpoint=True)


def _decor_monitor_cfg(arch):
    """(decor_mass_range, monitor_wp50, mass_bins) from architecture config."""
    mr = arch.get("decor_mass_range")
    mr = (float(mr[0]), float(mr[1])) if mr else None
    return mr, bool(arch.get("monitor_wp50", False)), mass_bins_from_config(arch)


def _bkg_decor_mask(y, mass, mass_range):
    mask = (y < 0.5).view(-1)
    if mass_range is not None:
        m = mass.view(-1)
        mask = mask & (m > mass_range[0]) & (m < mass_range[1])
    return mask


def _adv_losses(model, adv, data, mode, device, detach_score=False, mass_range=None):
    """Classifier BCE + adversary MDN NLL on background jets for one batch.

    mass_range=[lo, hi] restricts the adversary to background jets inside the
    window and maps it linearly onto the MDN's fixed 40-300 GeV support."""
    y = data.y.float().view(-1, 1)
    w = data.weights.float().view(-1, 1)
    logpt = torch.log(data.pt.float().view(-1, 1).clamp(min=1e-6))
    mass  = data.mass.float().view(-1, 1)
    mask  = _bkg_decor_mask(y, mass, mass_range)
    if mass_range is not None:
        lo, hi = mass_range
        mass = 40.0 + (mass - lo) * (260.0 / (hi - lo))

    extra = get_extra_features(data, mode)
    out = model(data, extra).clamp(0, 1)
    out = torch.nan_to_num(out)
    loss_cl = F.binary_cross_entropy(out, y, weight=w)

    score_bkg = out[mask].view(-1, 1)
    if detach_score:
        score_bkg = score_bkg.detach()
    adv_inp = torch.cat([score_bkg, logpt[mask]], dim=1)
    if adv_inp.shape[0] < 2:  # MDN BatchNorm needs >1 background jet
        return loss_cl, None
    pi, sigma, mu = adv(adv_inp)
    loss_adv = mdn_loss_new(device, pi, sigma, mu, mass[mask], w[mask])
    return loss_cl, loss_adv


def _sculpt_monitor(scores, labels, masses, wp50, massbins):
    """Background rejection and tagged/untagged background-mass JSD.

    wp50=False: fixed threshold score >= 0.5 (legacy monitor).
    wp50=True : threshold at 50% signal efficiency (signal-score median)."""
    s, y, m = np.concatenate(scores), np.concatenate(labels), np.concatenate(masses)
    bkg = y < 0.5
    thr = float(np.median(s[~bkg])) if (wp50 and (~bkg).any()) else 0.5
    tagged = s >= thr
    n_tag = int((tagged & bkg).sum())
    rej = (int(bkg.sum()) / n_tag) if n_tag else 0.0
    p, _ = np.histogram(m[bkg & tagged],  bins=massbins, density=True)
    f, _ = np.histogram(m[bkg & ~tagged], bins=massbins, density=True)
    jsd = JSD(p, f)
    if not np.isfinite(jsd):
        jsd = 0.0
    return rej, float(jsd)


@torch.no_grad()
def adv_validate(loader, model, adv, mode, device, mass_range=None,
                 wp50=False, massbins=MASSBINS):
    """Val losses + mass-sculpting monitors: bkg rejection and JSD between
    tagged/untagged background mass spectra (see _sculpt_monitor)."""
    model.eval(); adv.eval()
    tl_cl, tl_adv, n = 0.0, 0.0, 0
    scores, labels, masses = [], [], []
    for data in loader:
        data = data.to(device)
        loss_cl, loss_adv = _adv_losses(model, adv, data, mode, device,
                                        detach_score=True, mass_range=mass_range)
        tl_cl  += loss_cl.item() * data.num_graphs
        tl_adv += (loss_adv.item() if loss_adv is not None else 0.0) * data.num_graphs
        n += data.num_graphs

        extra = get_extra_features(data, mode)
        out = torch.nan_to_num(model(data, extra).clamp(0, 1)).view(-1)
        scores.append(out.cpu().numpy())
        labels.append(data.y.view(-1).cpu().numpy())
        masses.append(data.mass.view(-1).cpu().numpy())

    rej, jsd = _sculpt_monitor(scores, labels, masses, wp50, massbins)
    return tl_cl / max(n, 1), tl_adv / max(n, 1), rej, jsd


def distance_corr(var_1, var_2, normedweight, power=2):
    """Weighted distance correlation between two 1-D tensors (DisCo,
    arXiv:2001.05310, reference implementation). normedweight must average
    to 1 over the batch. power=2 gives dCorr^2 (recommended penalty)."""
    n = len(var_1)
    xx = var_1.view(-1, 1).expand(n, n)
    amat = (xx - xx.t()).abs()
    yy = var_2.view(-1, 1).expand(n, n)
    bmat = (yy - yy.t()).abs()

    amatavg = torch.mean(amat * normedweight, dim=1)
    Amat = (amat - amatavg.view(1, -1).expand(n, n)
                 - amatavg.view(-1, 1).expand(n, n)
                 + torch.mean(amatavg * normedweight))
    bmatavg = torch.mean(bmat * normedweight, dim=1)
    Bmat = (bmat - bmatavg.view(1, -1).expand(n, n)
                 - bmatavg.view(-1, 1).expand(n, n)
                 + torch.mean(bmatavg * normedweight))

    ABavg = torch.mean(Amat * Bmat * normedweight, dim=1)
    AAavg = torch.mean(Amat * Amat * normedweight, dim=1)
    BBavg = torch.mean(Bmat * Bmat * normedweight, dim=1)
    denom = (torch.mean(AAavg * normedweight)
             * torch.mean(BBavg * normedweight)).clamp(min=1e-12)
    if power == 2:
        return (torch.mean(ABavg * normedweight)) ** 2 / denom
    return torch.mean(ABavg * normedweight) / denom.sqrt()


def _disco_losses(model, data, mode, device, lam, mass_range=None):
    """Classifier BCE + lam * dCorr^2(score, mass) on background jets
    (restricted to mass_range=[lo, hi] if given)."""
    y = data.y.float().view(-1, 1)
    w = data.weights.float().view(-1, 1)
    extra = get_extra_features(data, mode)
    out = torch.nan_to_num(model(data, extra).clamp(0, 1))
    loss_cl = F.binary_cross_entropy(out, y, weight=w)

    mask = _bkg_decor_mask(y, data.mass.float().view(-1, 1), mass_range)
    if int(mask.sum()) < 10:
        return loss_cl, None
    s_b = out.view(-1)[mask]
    m_b = data.mass.float().view(-1)[mask]
    # Negative MC weights break the variance terms; clamp for the penalty only
    w_b = data.weights.float().view(-1)[mask].clamp(min=0.0)
    nw  = w_b * len(w_b) / w_b.sum().clamp(min=1e-12)
    dc  = distance_corr(s_b, m_b, nw, power=2)
    return loss_cl, dc


@torch.no_grad()
def disco_validate(loader, model, mode, device, lam, mass_range=None,
                   wp50=False, massbins=MASSBINS):
    """Mirror of adv_validate for DisCo: val BCE, val dCorr^2, bkg rejection
    and pass/fail background-mass JSD (see _sculpt_monitor)."""
    model.eval()
    tl_cl, tl_dc, n = 0.0, 0.0, 0
    scores, labels, masses = [], [], []
    for data in loader:
        data = data.to(device)
        loss_cl, dc = _disco_losses(model, data, mode, device, lam, mass_range=mass_range)
        tl_cl += loss_cl.item() * data.num_graphs
        tl_dc += (dc.item() if dc is not None else 0.0) * data.num_graphs
        n += data.num_graphs

        extra = get_extra_features(data, mode)
        out = torch.nan_to_num(model(data, extra).clamp(0, 1)).view(-1)
        scores.append(out.cpu().numpy())
        labels.append(data.y.view(-1).cpu().numpy())
        masses.append(data.mass.view(-1).cpu().numpy())

    rej, jsd = _sculpt_monitor(scores, labels, masses, wp50, massbins)
    return tl_cl / max(n, 1), tl_dc / max(n, 1), rej, jsd


def run_disco(model, config, args, mode, path_to_save, device,
              train_shards, val_loader, scale_factor, filter_attr,
              shards_per_window, batch_size, num_workers, pin_mem):
    """Single-network mass decorrelation: fine-tune the converged classifier
    with loss = BCE + lambda * dCorr^2(score, mass | background). Same budget
    and seeding protocol as ANN stage B for a like-for-like comparison."""
    arch = config["architecture"]
    lam  = float(arch.get("disco_lambda", 10))
    n_ep = int(arch.get("n_epochs_common", 20))
    lr   = arch["learning_rate"]
    mrange, wp50, mbins = _decor_monitor_cfg(arch)
    tag  = f"{mode}_disco_lam{lam:g}"
    opt  = torch.optim.Adam(model.parameters(), lr=0.4 * lr)

    RESUME = os.path.join(path_to_save, f"resume_{tag}.pt")
    DONE   = os.path.join(path_to_save, f"done_{tag}.flag")
    if os.path.exists(DONE):
        print(f">>> DisCo training already converged ({DONE}). Exiting.", flush=True)
        return

    log_rows, epoch_start = [], 0
    if os.path.exists(RESUME):
        ck = torch.load(RESUME, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        epoch_start, log_rows = ck["epoch"] + 1, ck["log"]
        print(f">>> Resuming DisCo {tag}: epoch {epoch_start + 1}", flush=True)

    def one_epoch():
        model.train()
        el_cl, el_dc, el_tot, n = 0.0, 0.0, 0.0, 0
        epoch_shards = list(train_shards)
        random.shuffle(epoch_shards)
        windows = [epoch_shards[i:i + shards_per_window]
                   for i in range(0, len(epoch_shards), shards_per_window)]
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(load_shards, windows[0], scale_factor, filter_attr)
            for wi, win in enumerate(windows):
                wdata = fut.result()
                if wi + 1 < len(windows):
                    fut = ex.submit(load_shards, windows[wi + 1], scale_factor, filter_attr)
                loader = DataLoader(wdata, batch_size=batch_size, shuffle=True,
                                    num_workers=num_workers, pin_memory=pin_mem)
                for data in loader:
                    data = data.to(device)
                    opt.zero_grad()
                    loss_cl, dc = _disco_losses(model, data, mode, device, lam,
                                                mass_range=mrange)
                    if dc is None or lam == 0:   # lam=0: plain fine-tune control (0*NaN guard)
                        loss = loss_cl
                    else:
                        loss = loss_cl + lam * dc
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    opt.step()
                    el_cl  += loss_cl.item() * data.num_graphs
                    el_dc  += (dc.item() if dc is not None else 0.0) * data.num_graphs
                    el_tot += loss.item() * data.num_graphs
                    n += data.num_graphs
                del wdata, loader
                gc.collect()
        return el_cl / max(n, 1), el_dc / max(n, 1), el_tot / max(n, 1)

    def save_state(epoch):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "epoch": epoch, "log": log_rows}, RESUME)
        with open(os.path.join(path_to_save, f"losses_{tag}.csv"), "w", newline="") as f:
            wcsv = csv.writer(f)
            wcsv.writerow(["epoch", "train_clsf", "train_disco", "train_total",
                           "val_clsf", "val_disco", "val_bkgrej05", "val_jsd"])
            wcsv.writerows(log_rows)

    wp_lbl = "eff50" if wp50 else "0.5"
    print(f"\n>>> DisCo fine-tune | tag={tag} | lambda={lam} | epochs={n_ep} "
          f"| decor_mass_range={mrange} | monitor_wp50={wp50}", flush=True)
    for ep in range(epoch_start, n_ep):
        t_cl, t_dc, t_tot = one_epoch()
        v_cl, v_dc, v_rej, v_jsd = disco_validate(val_loader, model, mode, device, lam,
                                                  mass_range=mrange, wp50=wp50, massbins=mbins)
        log_rows.append([ep + 1, t_cl, t_dc, t_tot, v_cl, v_dc, v_rej, v_jsd])
        print(f"[D] Epoch {ep + 1:03d}/{n_ep} | total {t_tot:.5f} "
              f"| disco {t_dc:.5f}/{v_dc:.5f} | clsf {t_cl:.5f}/{v_cl:.5f} "
              f"| rej@{wp_lbl} {v_rej:.1f} | JSD {v_jsd:.4f}", flush=True)
        torch.save(model.state_dict(),
                   os.path.join(path_to_save, f"model_{tag}_e{ep + 1:03d}.pt"))
        save_state(ep)

    # Same convention as ANN: score with the final fine-tuned epoch
    torch.save(model.state_dict(), os.path.join(path_to_save, f"best_model_{tag}.pt"))
    open(DONE, "w").close()
    print(f">>> DisCo training done: {tag}", flush=True)


def run_adversarial(model, config, args, mode, path_to_save, device,
                    train_shards, val_loader, scale_factor, filter_attr,
                    shards_per_window, batch_size, num_workers, pin_mem):
    arch = config["architecture"]
    lam   = float(args.lambda_parameter if args.lambda_parameter is not None
                  else arch.get("lambda_parameter", 25))
    n_g   = int(arch.get("num_gaussians", 20))
    n_ep_a = int(arch.get("n_epochs_adv", 10))
    n_ep_c = int(arch.get("n_epochs_common", 20))
    loss_par = float(arch.get("loss_parameter", 1))
    lw = arch.get("loss_weights", [1, 1])
    lr = arch["learning_rate"]
    mrange, wp50, mbins = _decor_monitor_cfg(arch)
    wp_lbl = "eff50" if wp50 else "0.5"

    tag = f"{mode}_ann_lam{lam:g}"
    adv = Adversary_new(lam, n_g).to(device)
    opt_adv = torch.optim.Adam(adv.parameters(), lr=5 * lr)
    opt_cl  = torch.optim.Adam(model.parameters(), lr=0.4 * lr)

    RESUME = os.path.join(path_to_save, f"resume_{tag}.pt")
    DONE   = os.path.join(path_to_save, f"done_{tag}.flag")
    if os.path.exists(DONE):
        print(f">>> Adversarial training already converged ({DONE}). Exiting.", flush=True)
        return

    log_rows, stage_start, epoch_start = [], "A", 0
    if os.path.exists(RESUME):
        ck = torch.load(RESUME, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); adv.load_state_dict(ck["adv"])
        opt_adv.load_state_dict(ck["opt_adv"]); opt_cl.load_state_dict(ck["opt_cl"])
        stage_start, epoch_start, log_rows = ck["stage"], ck["epoch"] + 1, ck["log"]
        print(f">>> Resuming ANN {tag}: stage {stage_start}, epoch {epoch_start + 1}", flush=True)

    def one_epoch(stage):
        combined = stage == "B"
        model.train(combined); adv.train()
        el_cl, el_adv, el_tot, n = 0.0, 0.0, 0.0, 0
        epoch_shards = list(train_shards)
        random.shuffle(epoch_shards)
        windows = [epoch_shards[i:i + shards_per_window]
                   for i in range(0, len(epoch_shards), shards_per_window)]
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(load_shards, windows[0], scale_factor, filter_attr)
            for wi, win in enumerate(windows):
                wdata = fut.result()
                if wi + 1 < len(windows):
                    fut = ex.submit(load_shards, windows[wi + 1], scale_factor, filter_attr)
                loader = DataLoader(wdata, batch_size=batch_size, shuffle=True,
                                    num_workers=num_workers, pin_memory=pin_mem)
                for data in loader:
                    data = data.to(device)
                    opt_adv.zero_grad(); opt_cl.zero_grad()
                    loss_cl, loss_adv = _adv_losses(
                        model, adv, data, mode, device, detach_score=not combined,
                        mass_range=mrange)
                    if loss_adv is None:
                        continue
                    if combined:
                        loss = lw[0] * loss_cl + lw[1] * loss_par * loss_adv
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                        torch.nn.utils.clip_grad_norm_(adv.parameters(), 5.0)
                        opt_cl.step(); opt_adv.step()
                    else:
                        loss = loss_adv
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(adv.parameters(), 5.0)
                        opt_adv.step()
                    el_cl  += loss_cl.item() * data.num_graphs
                    el_adv += loss_adv.item() * data.num_graphs
                    el_tot += loss.item() * data.num_graphs
                    n += data.num_graphs
                del wdata, loader
                gc.collect()
        return el_cl / max(n, 1), el_adv / max(n, 1), el_tot / max(n, 1)

    def save_state(stage, epoch):
        torch.save({"model": model.state_dict(), "adv": adv.state_dict(),
                    "opt_adv": opt_adv.state_dict(), "opt_cl": opt_cl.state_dict(),
                    "stage": stage, "epoch": epoch, "log": log_rows}, RESUME)
        with open(os.path.join(path_to_save, f"losses_{tag}.csv"), "w", newline="") as f:
            wcsv = csv.writer(f)
            wcsv.writerow(["stage", "epoch", "train_clsf", "train_adv", "train_total",
                           "val_clsf", "val_adv", "val_bkgrej05", "val_jsd"])
            wcsv.writerows(log_rows)

    print(f"\n>>> ANN stage A (adversary pre-train) | tag={tag} | lambda={lam} "
          f"| gaussians={n_g} | epochs={n_ep_a}", flush=True)
    if stage_start == "A":
        for ep in range(epoch_start, n_ep_a):
            t_cl, t_adv, t_tot = one_epoch("A")
            v_cl, v_adv, v_rej, v_jsd = adv_validate(val_loader, model, adv, mode, device,
                                                      mass_range=mrange, wp50=wp50, massbins=mbins)
            log_rows.append(["A", ep + 1, t_cl, t_adv, t_tot, v_cl, v_adv, v_rej, v_jsd])
            print(f"[A] Epoch {ep + 1:03d}/{n_ep_a} | adv {t_adv:.5f}/{v_adv:.5f} "
                  f"| clsf {t_cl:.5f}/{v_cl:.5f} | rej@{wp_lbl} {v_rej:.1f} | JSD {v_jsd:.4f}",
                  flush=True)
            save_state("A", ep)
        epoch_start = 0

    print(f"\n>>> ANN stage B (combined GRL training) | epochs={n_ep_c}", flush=True)
    for ep in range(epoch_start if stage_start == "B" else 0, n_ep_c):
        t_cl, t_adv, t_tot = one_epoch("B")
        v_cl, v_adv, v_rej, v_jsd = adv_validate(val_loader, model, adv, mode, device,
                                                      mass_range=mrange, wp50=wp50, massbins=mbins)
        log_rows.append(["B", ep + 1, t_cl, t_adv, t_tot, v_cl, v_adv, v_rej, v_jsd])
        print(f"[B] Epoch {ep + 1:03d}/{n_ep_c} | total {t_tot:.5f} "
              f"| adv {t_adv:.5f}/{v_adv:.5f} | clsf {t_cl:.5f}/{v_cl:.5f} "
              f"| rej@{wp_lbl} {v_rej:.1f} | JSD {v_jsd:.4f}", flush=True)
        torch.save(model.state_dict(),
                   os.path.join(path_to_save, f"model_{tag}_e{ep + 1:03d}.pt"))
        save_state("B", ep)

    # Deterministic handle for the scoring config (PUB-2023-017 recipe: use the
    # final combined-training epoch)
    torch.save(model.state_dict(), os.path.join(path_to_save, f"best_model_{tag}.pt"))
    torch.save(adv.state_dict(),   os.path.join(path_to_save, f"adv_{tag}.pt"))
    open(DONE, "w").close()
    print(f">>> ANN training done: {tag}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LRJ LundNet trainer — streaming shard loading")
    parser.add_argument("config", help="Path to config YAML")
    parser.add_argument(
        "--cpu-prep", action="store_true",
        help="CPU-only mode: compute and cache scale_factor.json, then exit (no GPU needed)"
    )
    parser.add_argument(
        "--adversarial", action="store_true",
        help="Run GMM-ANN mass-decorrelation stages after (or on top of) the trained classifier"
    )
    parser.add_argument(
        "--lambda-parameter", type=float, default=None,
        help="Override architecture.lambda_parameter (gradient-reversal strength)"
    )
    args = parser.parse_args()
    config = load_yaml(args.config)

    gpu_id       = config.get("gpu", 0)
    if gpu_id < 0:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    mode         = config["architecture"]["training_mode"]
    path_to_save = config["data"]["path_to_save"]
    os.makedirs(path_to_save, exist_ok=True)

    batch_size        = config["architecture"]["batch_size"]
    schedule          = config["architecture"].get("optimizer_schedule", [8, 16])
    boundaries        = tuple(schedule)
    shards_per_window = config.get("shards_per_window", 20)
    val_frac          = config["architecture"].get("test_size", 0.1)
    num_workers       = config.get("num_workers", 0)
    pin_mem           = device.type == "cuda"
    use_amp           = False  # AMP fp16 causes NaN overflow in this model
    scaler            = GradScaler("cuda") if use_amp else None

    # -----------------------------------------------------------------------
    # Shard discovery + deterministic train/val split
    # -----------------------------------------------------------------------
    filter_attr = config.get("filter_attr", None)

    if "preproc_dir" in config["data"]:
        preproc_dir = config["data"]["preproc_dir"]
        n_parts     = int(config.get("n_parts", 10))
        all_shards  = []
        for i in range(n_parts):
            part_dir = os.path.join(preproc_dir, f"train_part{i}")
            all_shards.extend(sorted(glob.glob(os.path.join(part_dir, "processed_*.pt"))))
        if not all_shards:
            raise FileNotFoundError(
                f"No shards in {preproc_dir}/train_part0..{n_parts-1}/ "
                f"(looked for processed_*.pt)"
            )
        print(f"n_parts mode: using {n_parts} part(s), {len(all_shards)} total shards", flush=True)
    else:
        n_parts    = None
        all_shards = sorted(glob.glob(config["data"]["path_to_trainfiles"]))
        if not all_shards:
            raise FileNotFoundError(f"No shards found: {config['data']['path_to_trainfiles']}")

    rng = random.Random(42)
    shards_shuffled = list(all_shards)
    rng.shuffle(shards_shuffled)

    # -----------------------------------------------------------------------
    # External val set (val_dir) or internal 5% split
    # -----------------------------------------------------------------------
    val_dir = config["data"].get("val_dir", None)
    if val_dir:
        val_shards       = sorted(glob.glob(os.path.join(val_dir, "processed_*.pt")))
        train_shards     = all_shards                    # use full set for training
        if not val_shards:
            raise FileNotFoundError(f"No val shards in {val_dir}")
        # val uses its own scale_factor so val loss is independent of training size
        val_scale_factor = compute_scale_factor(val_shards, filter_attr=filter_attr, n_workers=1)
        print(
            f"Shards — total: {len(all_shards)} | n_parts: {n_parts} "
            f"| train: {len(train_shards)} | val (external): {len(val_shards)} "
            f"| val_sf: {val_scale_factor:.4f} | filter_attr: {filter_attr}",
            flush=True,
        )
    else:
        n_val            = max(1, int(len(shards_shuffled) * val_frac))
        val_shards       = shards_shuffled[:n_val]
        train_shards     = shards_shuffled[n_val:]
        val_scale_factor = None                          # internal split: shared sf, set below
        print(
            f"Shards — total: {len(all_shards)} | n_parts: {n_parts} "
            f"| train: {len(train_shards)} | val: {len(val_shards)} "
            f"| filter_attr: {filter_attr}",
            flush=True,
        )

    # -----------------------------------------------------------------------
    # Scale factor: load from cache if available, otherwise compute and save
    # -----------------------------------------------------------------------
    sf_workers        = config.get("scale_factor_workers", 1)
    scale_factor_path = os.path.join(path_to_save, "scale_factor.json")
    if os.path.exists(scale_factor_path):
        with open(scale_factor_path) as f:
            scale_factor = json.load(f)["scale_factor"]
        print(f">>> Loaded cached scale_factor: {scale_factor:.6f}", flush=True)
    else:
        print(
            f"--- [{datetime.now().strftime('%H:%M:%S')}] Computing scale factor "
            f"({len(train_shards)} shards, {sf_workers} workers)... ---",
            flush=True,
        )
        scale_factor = compute_scale_factor(train_shards, filter_attr=filter_attr, n_workers=sf_workers)
        with open(scale_factor_path, "w") as f:
            json.dump({"scale_factor": scale_factor}, f)
        print(f">>> Scale factor (sig/bkg weight ratio): {scale_factor:.6f}  [saved to {scale_factor_path}]", flush=True)

    if val_scale_factor is None:
        val_scale_factor = scale_factor                  # internal split shares the training sf

    # --cpu-prep mode: cache is now written, nothing more to do on a CPU node
    if args.cpu_prep:
        print(">>> --cpu-prep done. Exiting (GPU training job will pick up from cache).", flush=True)
        sys.exit(0)

    # -----------------------------------------------------------------------
    # Fixed validation set (loaded once, stays in RAM)
    # -----------------------------------------------------------------------
    print(f"--- [{datetime.now().strftime('%H:%M:%S')}] Loading val set ({len(val_shards)} shards) ---", flush=True)
    val_data   = load_shards(val_shards, val_scale_factor, filter_attr=filter_attr)
    val_loader = DataLoader(
        val_data, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_mem,
        persistent_workers=(num_workers > 0),
    )
    print(f">>> Val jets: {len(val_data)}", flush=True)
    _val_w     = np.array([float(g.weights) for g in val_data])
    weight_cap = float(np.percentile(_val_w, 99.9)) if len(_val_w) else None
    print(f">>> Val weight cap (p99.9, selection metric only): {weight_cap:.4g}", flush=True)
    del _val_w

    # -----------------------------------------------------------------------
    # Model & optimizers
    # -----------------------------------------------------------------------
    extra_dim  = config["architecture"]["extra_dim"]
    arch_model = config["architecture"].get("choose_model", "LundNet_General")
    if arch_model == "LundNet_FiLM":
        model = LundNet_FiLM(extra_dim=extra_dim, cond_dim=1).to(device)
    elif arch_model == "LundNet_Quantum":
        n_qubits = config["architecture"].get("n_qubits", 6)
        n_layers = config["architecture"].get("n_layers_vqc", 2)
        model = LundNet_Quantum(extra_dim=extra_dim, n_qubits=n_qubits, n_layers=n_layers).to(device)
    elif arch_model == "LundFormer":
        model = LundFormer(
            extra_dim=extra_dim,
            d_model=config["architecture"].get("d_model", 96),
            n_heads=config["architecture"].get("n_heads", 8),
            n_layers=config["architecture"].get("n_layers_former", 4),
            d_ff=config["architecture"].get("d_ff", 288),
            dropout=config["architecture"].get("dropout", 0.1),
        ).to(device)
    else:
        model = LundNet_General(extra_dim=extra_dim).to(device)
    lr    = config["architecture"]["learning_rate"]
    opts  = [
        torch.optim.Adam(model.parameters(), lr=lr),
        torch.optim.Adam(model.parameters(), lr=4.0 * lr),
        torch.optim.Adam(model.parameters(), lr=10.0 * lr),
    ]

    n_epochs          = config["architecture"]["n_epochs"]
    patience          = config["architecture"].get("patience", 8)
    # Opt-in: avoid aborting in-flight jobs that still have patience:8 in yaml
    early_stop        = bool(config["architecture"].get("early_stop", False))
    # Do not early-stop during the high/mid LR stages (default: after 2nd schedule boundary)
    min_epochs_es     = int(config["architecture"].get("min_epochs_early_stop", boundaries[1]))
    es_reset_stage    = bool(config["architecture"].get("es_reset_per_stage", False))
    grad_clip         = config["architecture"].get("grad_clip")
    grad_clip         = float(grad_clip) if grad_clip else None
    best_val_loss     = float("inf")
    epochs_no_improve = 0
    metrics_log       = []

    # -----------------------------------------------------------------------
    # Checkpoint / resume logic (supports 48h job chaining)
    # -----------------------------------------------------------------------
    do_adversarial = bool(args.adversarial or
                          config["architecture"].get("do_combined_training", False))
    do_disco       = bool(config["architecture"].get("do_disco_training", False))

    RESUME_PATH = os.path.join(path_to_save, f"resume_{mode}.pt")
    DONE_FLAG   = os.path.join(path_to_save, f"done_{mode}.flag")
    BEST_PATH   = os.path.join(path_to_save, f"best_model_{mode}.pt")
    start_epoch = 0

    if os.path.exists(DONE_FLAG):
        if not (do_adversarial or do_disco):
            print(f">>> Training already converged ({DONE_FLAG} present). Exiting.", flush=True)
            sys.exit(0)
        # Classifier stage done — jump straight to the decorrelation stage
        print(f">>> Classifier converged; loading {BEST_PATH} for decorrelation stage.", flush=True)
        model.load_state_dict(torch.load(BEST_PATH, map_location=device, weights_only=True))
        stage_fn = run_disco if do_disco else run_adversarial
        stage_fn(model, config, args, mode, path_to_save, device,
                 train_shards, val_loader, scale_factor, filter_attr,
                 shards_per_window, batch_size, num_workers, pin_mem)
        sys.exit(0)

    # Mid-epoch resume state (populated when the checkpoint was taken between
    # windows; slow models like LundFormer never reach an epoch boundary inside
    # one interruption-free slot, so per-window checkpoints are essential).
    resume_win_start    = 0
    resume_epoch_shards = None
    resume_epoch_loss   = 0.0
    resume_epoch_n      = 0

    if os.path.exists(RESUME_PATH):
        print(f">>> Resuming from checkpoint: {RESUME_PATH}", flush=True)
        ckpt = torch.load(RESUME_PATH, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        for i, o in enumerate(opts):
            o.load_state_dict(ckpt["optimizer_states"][i])
        best_val_loss     = ckpt["best_val_loss"]
        epochs_no_improve = ckpt["epochs_no_improve"]
        metrics_log       = ckpt["metrics_log"]
        if ckpt.get("val_metric") != VAL_METRIC_VERSION:
            print(
                f"    Val metric changed ({ckpt.get('val_metric')} -> {VAL_METRIC_VERSION}): "
                f"resetting best_val_loss and no_improve counter.",
                flush=True,
            )
            best_val_loss     = float("inf")
            epochs_no_improve = 0
        if ckpt.get("win_start", 0) > 0:
            start_epoch         = ckpt["epoch"]          # re-enter same epoch
            resume_win_start    = ckpt["win_start"]
            resume_epoch_shards = ckpt["epoch_shards"]
            resume_epoch_loss   = ckpt["epoch_loss"]
            resume_epoch_n      = ckpt["epoch_n"]
            print(
                f"    Resuming MID-EPOCH {start_epoch + 1}/{n_epochs} at window "
                f"{resume_win_start} | best_val={best_val_loss:.5f}",
                flush=True,
            )
        else:
            start_epoch = ckpt["epoch"] + 1
            print(
                f"    Resuming at epoch {start_epoch + 1}/{n_epochs} | "
                f"best_val={best_val_loss:.5f} | no_improve={epochs_no_improve}",
                flush=True,
            )
    elif os.path.exists(os.path.join(path_to_save, f"best_model_{mode}.pt")):
        # Warm start: load weights from a previous run's best model
        print(f">>> Warm start: loading weights from best_model_{mode}.pt", flush=True)
        model.load_state_dict(torch.load(
            os.path.join(path_to_save, f"best_model_{mode}.pt"),
            map_location=device, weights_only=True,
        ))

    print(
        f"\n>>> Training | mode={mode} | device={device} | "
        f"amp={use_amp} | schedule={boundaries} | shards_per_window={shards_per_window} "
        f"| grad_clip={grad_clip} | early_stop={early_stop} | es_reset_per_stage={es_reset_stage}",
        flush=True,
    )

    converged = False
    for epoch in range(start_epoch, n_epochs):
        if epoch < boundaries[0]:
            opt = opts[2]   # 10× lr — large LR first for fast descent
        elif epoch < boundaries[1]:
            opt = opts[1]   # 4× lr
        else:
            opt = opts[0]   # 1× lr — fine-tune at end

        if es_reset_stage and epoch in boundaries and resume_win_start == 0:
            epochs_no_improve = 0

        if epoch == start_epoch and resume_win_start > 0:
            # Mid-epoch resume: replay the saved shard order, skip done windows
            epoch_shards = list(resume_epoch_shards)
            win_start    = resume_win_start
            epoch_loss, epoch_n = resume_epoch_loss, resume_epoch_n
            resume_win_start = 0
        else:
            epoch_shards = list(train_shards)
            random.shuffle(epoch_shards)
            win_start = 0
            epoch_loss, epoch_n = 0.0, 0

        model.train()

        windows = [
            epoch_shards[i : i + shards_per_window]
            for i in range(0, len(epoch_shards), shards_per_window)
        ]

        def _save_window_ckpt(next_win):
            """Atomic per-window resume save — survives kills shorter than an epoch."""
            tmp = RESUME_PATH + ".tmp"
            torch.save({
                "epoch":             epoch,
                "win_start":         next_win,
                "epoch_shards":      epoch_shards,
                "epoch_loss":        epoch_loss,
                "epoch_n":           epoch_n,
                "model_state_dict":  model.state_dict(),
                "optimizer_states":  [o.state_dict() for o in opts],
                "best_val_loss":     best_val_loss,
                "epochs_no_improve": epochs_no_improve,
                "metrics_log":       metrics_log,
                "val_metric":        VAL_METRIC_VERSION,
            }, tmp)
            # Lustre NFS: atomic rename can fail if the .tmp file is not yet
            # visible due to metadata caching.  Retry once after a short sleep.
            try:
                os.replace(tmp, RESUME_PATH)
            except FileNotFoundError:
                import time; time.sleep(2)
                if os.path.exists(tmp):
                    os.replace(tmp, RESUME_PATH)
                else:
                    # .tmp vanished — write directly (non-atomic fallback)
                    torch.save({
                        "epoch": epoch, "win_start": next_win,
                        "epoch_shards": epoch_shards, "epoch_loss": epoch_loss,
                        "epoch_n": epoch_n,
                        "model_state_dict": model.state_dict(),
                        "optimizer_states": [o.state_dict() for o in opts],
                        "best_val_loss": best_val_loss,
                        "epochs_no_improve": epochs_no_improve,
                        "metrics_log": metrics_log, "val_metric": VAL_METRIC_VERSION,
                    }, RESUME_PATH)
                    print(">>> WARNING: atomic rename failed, wrote directly.", flush=True)

        # Double-buffer: load next window in a background thread while GPU trains
        # on the current window, overlapping I/O and GPU compute.
        # win_start == len(windows) when interrupted after the last window but
        # before the epoch-end checkpoint: go straight to validation.
        with ThreadPoolExecutor(max_workers=1) as executor:
            if win_start < len(windows):
                next_future = executor.submit(
                    load_shards, windows[win_start], scale_factor, filter_attr
                )

            for win_idx in range(win_start, len(windows)):
                window_data = next_future.result()

                # Kick off the next window load immediately
                if win_idx + 1 < len(windows):
                    next_future = executor.submit(
                        load_shards, windows[win_idx + 1], scale_factor, filter_attr
                    )

                window_data = sk_shuffle(
                    window_data, random_state=epoch * 10000 + win_idx * shards_per_window
                )
                loader = DataLoader(
                    window_data, batch_size=batch_size, shuffle=True,
                    num_workers=num_workers, pin_memory=pin_mem,
                    persistent_workers=(num_workers > 0),
                )

                for data in loader:
                    data = data.to(device)
                    opt.zero_grad()
                    if use_amp:
                        with autocast("cuda"):
                            extra = get_extra_features(data, mode)
                            out   = model(data, extra)
                        # Compute loss in float32 — BCE is unsafe inside autocast
                        loss = F.binary_cross_entropy(
                            out.float(), data.y.float().view(-1, 1),
                            weight=data.weights.float().view(-1, 1)
                        )
                        scaler.scale(loss).backward()
                        if grad_clip:
                            scaler.unscale_(opt)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                        scaler.step(opt)
                        scaler.update()
                    else:
                        extra = get_extra_features(data, mode)
                        out   = model(data, extra)
                        loss  = F.binary_cross_entropy(
                            out, data.y.float().view(-1, 1),
                            weight=data.weights.float().view(-1, 1)
                        )
                        loss.backward()
                        if grad_clip:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                        opt.step()
                    epoch_loss += loss.item() * data.num_graphs
                    epoch_n    += data.num_graphs

                del window_data, loader
                gc.collect()
                _save_window_ckpt(win_idx + 1)

        t_loss = epoch_loss / epoch_n if epoch_n > 0 else 0.0
        v_loss, v_loss_raw = validate(
            val_loader, model, device, mode, use_amp=use_amp, weight_cap=weight_cap
        )

        print(
            f"Epoch {epoch + 1:03d}/{n_epochs} | Train Loss: {t_loss:.5f} "
            f"| Val Loss: {v_loss:.5f} (raw {v_loss_raw:.5f})",
            flush=True,
        )
        metrics_log.append([epoch + 1, t_loss, v_loss, v_loss_raw])

        if v_loss < best_val_loss:
            best_val_loss     = v_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(path_to_save, f"best_model_{mode}.pt"))
            print("  [Save] New best model!", flush=True)
        else:
            epochs_no_improve += 1

        # Save resume checkpoint after every epoch so job chaining can continue
        # (atomic write; no win_start key => next resume starts at epoch + 1)
        _tmp = RESUME_PATH + ".tmp"
        torch.save({
            "epoch":             epoch,
            "model_state_dict":  model.state_dict(),
            "optimizer_states":  [o.state_dict() for o in opts],
            "best_val_loss":     best_val_loss,
            "epochs_no_improve": epochs_no_improve,
            "metrics_log":       metrics_log,
            "val_metric":        VAL_METRIC_VERSION,
        }, _tmp)
        try:
            os.replace(_tmp, RESUME_PATH)
        except FileNotFoundError:
            import time; time.sleep(2)
            if os.path.exists(_tmp):
                os.replace(_tmp, RESUME_PATH)
            else:
                torch.save({
                    "epoch": epoch, "model_state_dict": model.state_dict(),
                    "optimizer_states": [o.state_dict() for o in opts],
                    "best_val_loss": best_val_loss,
                    "epochs_no_improve": epochs_no_improve,
                    "metrics_log": metrics_log, "val_metric": VAL_METRIC_VERSION,
                }, RESUME_PATH)
                print(">>> WARNING: atomic rename failed, wrote directly.", flush=True)

        # Flush losses CSV after every epoch (survives mid-run job kills)
        with open(os.path.join(path_to_save, f"losses_{mode}.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss", "val_loss_raw"])
            writer.writerows(metrics_log)

        # Early stop: requires architecture.early_stop: true (opt-in)
        if (
            early_stop
            and patience
            and (epoch + 1) >= min_epochs_es
            and epochs_no_improve >= patience
        ):
            print(
                f"  [EarlyStop] no improve for {epochs_no_improve} epochs "
                f"(patience={patience}, min_epochs={min_epochs_es}). "
                f"best_val={best_val_loss:.5f}",
                flush=True,
            )
            converged = True
            break
    else:
        # for-loop completed without break → max epochs reached
        converged = True

    if converged:
        open(DONE_FLAG, "w").close()

    if (do_adversarial or do_disco) and converged:
        model.load_state_dict(torch.load(BEST_PATH, map_location=device, weights_only=True))
        stage_fn = run_disco if do_disco else run_adversarial
        stage_fn(model, config, args, mode, path_to_save, device,
                 train_shards, val_loader, scale_factor, filter_attr,
                 shards_per_window, batch_size, num_workers, pin_mem)

    print(f"\n>>> Done. Path: {path_to_save}", flush=True)


if __name__ == "__main__":
    main()

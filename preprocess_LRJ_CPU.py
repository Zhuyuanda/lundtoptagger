import os
import json
import gc
import torch
import numpy as np
import uproot
import glob
import matplotlib.pyplot as plt
import sys

from tools.GNN_model_weight.utils_newdata import load_yaml
from tools.utils_config import recursive_update, parse_dot_args
from plotting.utils_plots_matplotlib import hist_with_errors


def sprint(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


def _graph_to_root_path(g_path):
    return g_path.replace("graphs_", "data_").replace("_with_pt", "").replace(".pt", ".root")


def _iter_train_shards(train_graph_files):
    for g_path in sorted(train_graph_files):
        r_path = _graph_to_root_path(g_path)
        with uproot.open(r_path) as f:
            tree = f["FlatSubstructureJetTree"]
            pts = tree["fjet_pt"].array(library="np")
            labels = tree["labels"].array(library="np")
        yield g_path, pts, labels


def _pt_limits_per_label(train_graph_files):
    limits = {0: [np.inf, -np.inf], 1: [np.inf, -np.inf]}
    for _, pts, labels in _iter_train_shards(train_graph_files):
        for lab in (0, 1):
            mask = labels == lab
            if not np.any(mask):
                continue
            p = pts[mask]
            limits[lab][0] = min(limits[lab][0], float(p.min()))
            limits[lab][1] = max(limits[lab][1], float(p.max()))
    for lab in (0, 1):
        if limits[lab][0] > limits[lab][1]:
            limits[lab] = [0.0, 1.0]
    return limits


def _bin_edges_per_label(limits, n_bins):
    edges = {}
    for lab in (0, 1):
        lo, hi = limits[lab]
        if hi <= lo:
            hi = lo + 1.0
        edges[lab] = np.linspace(lo, hi, n_bins + 1)
    return edges


def _digitize_pt(pts, edges, n_bins):
    idx = np.digitize(pts, edges) - 1
    return np.clip(idx, 0, n_bins - 1)


def compute_streaming_flat_weights(train_graph_files, n_bins, iterations=2):
    """Iterative pT flattening without loading all jets into one array."""
    limits = _pt_limits_per_label(train_graph_files)
    bin_edges = _bin_edges_per_label(limits, n_bins)
    weight_by_graph = {}

    for it in range(iterations):
        bin_sums = {0: np.zeros(n_bins, dtype=float), 1: np.zeros(n_bins, dtype=float)}
        for g_path, pts, labels in _iter_train_shards(train_graph_files):
            w = weight_by_graph.get(g_path)
            if w is None:
                w = np.ones(len(pts), dtype=float)
            for lab in (0, 1):
                mask = labels == lab
                if not np.any(mask):
                    continue
                idx = _digitize_pt(pts[mask], bin_edges[lab], n_bins)
                for i in range(n_bins):
                    sel = idx == i
                    if np.any(sel):
                        bin_sums[lab][i] += w[mask][sel].sum()

        for g_path, pts, labels in _iter_train_shards(train_graph_files):
            w = weight_by_graph.get(g_path)
            if w is None:
                w = np.ones(len(pts), dtype=float)
            else:
                w = w.copy()
            for lab in (0, 1):
                mask = labels == lab
                if not np.any(mask):
                    continue
                nz = bin_sums[lab] > 0
                target = bin_sums[lab][nz].mean() if np.any(nz) else 1.0
                idx = _digitize_pt(pts[mask], bin_edges[lab], n_bins)
                scale = np.ones(int(mask.sum()), dtype=float)
                for i in range(n_bins):
                    if bin_sums[lab][i] > 0:
                        scale[idx == i] *= target / bin_sums[lab][i]
                w[mask] *= scale
            weight_by_graph[g_path] = w
        sprint(f"  flatten iteration {it + 1}/{iterations} done")

    return weight_by_graph, bin_edges


def _accumulate_pt_histograms(train_graph_files, weight_by_graph, bin_edges, n_bins):
    """Streaming before/after histograms for validation plots."""
    hist_raw = {0: np.zeros(n_bins), 1: np.zeros(n_bins)}
    hist_flat = {0: np.zeros(n_bins), 1: np.zeros(n_bins)}
    for g_path, pts, labels in _iter_train_shards(train_graph_files):
        w = weight_by_graph[g_path]
        for lab in (0, 1):
            mask = labels == lab
            if not np.any(mask):
                continue
            idx = _digitize_pt(pts[mask], bin_edges[lab], n_bins)
            for i in range(n_bins):
                sel = idx == i
                if np.any(sel):
                    hist_raw[lab][i] += sel.sum()
                    hist_flat[lab][i] += w[mask][sel].sum()
    return hist_raw, hist_flat, bin_edges


def _save_pt_plots_streaming(hist_raw, hist_flat, bin_edges, config, out_dir):
    plot_dir = os.path.join(out_dir, "plots_reweighting")
    os.makedirs(plot_dir, exist_ok=True)
    n_bins = config["n_bins_pt"]

    plt.figure(figsize=(12, 5))
    for subplot, title, use_flat in ((1, "Before Flattening", False), (2, "After Flattening", True)):
        plt.subplot(1, 2, subplot)
        for lab, name, color in ((0, "Bkg", "C0"), (1, "Top", "C1")):
            edges = bin_edges[lab]
            centers = 0.5 * (edges[:-1] + edges[1:])
            counts = hist_flat[lab] if use_flat else hist_raw[lab]
            total = counts.sum()
            if total <= 0:
                continue
            density = counts / total / np.diff(edges)
            plt.step(centers, density, where="mid", label=f"{name} ({'Flat' if use_flat else 'Raw'})", color=color)
        plt.xlabel("Jet pT [GeV]")
        plt.ylabel("Density")
        plt.title(title)
        plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "pT_flattening_check.png"))
    plt.close()
    sprint(f"\n[Check] pT validation plots saved to: {plot_dir}")


def _save_pt_plots(pts, labels, weights_all, config, out_dir):
    plot_dir = os.path.join(out_dir, "plots_reweighting")
    os.makedirs(plot_dir, exist_ok=True)
    sig_mask, bkg_mask = (labels == 1), (labels == 0)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    hist_with_errors(pts[bkg_mask], bins=config["n_bins_pt"], density=True, label="Bkg (Raw)")
    hist_with_errors(pts[sig_mask], bins=config["n_bins_pt"], density=True, label="Top (Raw)")
    plt.xlabel("Jet pT [GeV]")
    plt.ylabel("Density")
    plt.title("Before Flattening")
    plt.legend()
    plt.subplot(1, 2, 2)
    hist_with_errors(pts[bkg_mask], weights=weights_all[bkg_mask], bins=config["n_bins_pt"], density=True, label="Bkg (Flat)")
    hist_with_errors(pts[sig_mask], weights=weights_all[sig_mask], bins=config["n_bins_pt"], density=True, label="Top (Flat)")
    plt.xlabel("Jet pT [GeV]")
    plt.ylabel("Density")
    plt.title("After Flattening")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "pT_flattening_check.png"))
    plt.close()
    sprint(f"\n[Check] pT validation plots saved to: {plot_dir}")


def process_lrj_pt_only(config):
    stats = json.load(open(config["data"]["meanstd_json"]))
    mean_x = torch.from_numpy(np.array(stats["mean_x"], dtype=np.float32))
    std_x = torch.from_numpy(np.array(stats["std_x"], dtype=np.float32))
    mean_ntrk, std_ntrk = float(stats["mean_ntrk"]), float(stats["std_ntrk"])

    base_out_dir = config["data"]["out_dir"]
    n_bins = config["n_bins_pt"]
    iterations = config.get("flatten_iterations", 2)
    use_streaming = config.get("streaming_flatten", True)

    train_graph_files = []
    for p in config["data"]["train_graphs"]:
        train_graph_files.extend(sorted(glob.glob(p)))

    if train_graph_files:
        sprint(f"--- Stage 1: Global pT weights ({len(train_graph_files)} train batches, streaming={use_streaming}) ---")
        if use_streaming:
            weight_by_graph, bin_edges = compute_streaming_flat_weights(
                train_graph_files, n_bins=n_bins, iterations=iterations
            )
            hist_raw, hist_flat, _ = _accumulate_pt_histograms(
                train_graph_files, weight_by_graph, bin_edges, n_bins
            )
        else:
            from tools.GNN_model_weight.utils_newdata import assign_flat_weights

            pts_list, labels_list = [], []
            for g_path in train_graph_files:
                r_path = _graph_to_root_path(g_path)
                with uproot.open(r_path) as f:
                    tree = f["FlatSubstructureJetTree"]
                    pts_list.append(tree["fjet_pt"].array(library="np"))
                    labels_list.append(tree["labels"].array(library="np"))
            pts_all = np.concatenate(pts_list)
            labels_all = np.concatenate(labels_list)
            weights_all = np.zeros_like(pts_all, dtype=float)
            for lab in (0, 1):
                mask = labels_all == lab
                if np.any(mask):
                    weights_all[mask] = assign_flat_weights(pts_all[mask], n_bins=n_bins)
            weight_by_graph = {}
            ptr = 0
            for g_path, pts in zip(train_graph_files, pts_list):
                weight_by_graph[g_path] = weights_all[ptr : ptr + len(pts)]
                ptr += len(pts)

        sprint("\n--- Stage 2: Normalization & pT weights on TRAIN set ---")
        train_out = os.path.join(base_out_dir, "train")
        os.makedirs(train_out, exist_ok=True)

        for g_path in train_graph_files:
            dataset = torch.load(g_path, weights_only=False)
            weights = weight_by_graph[g_path]
            for i, g in enumerate(dataset):
                g.x = ((g.x - mean_x) / std_x).float()
                g.Ntrk = torch.tensor((float(g.Ntrk) - mean_ntrk) / std_ntrk).float()
                g.weights = float(weights[i])
            out_path = os.path.join(train_out, f"processed_{os.path.basename(g_path)}")
            torch.save(dataset, out_path)
            del dataset
            gc.collect()
            sprint(f" Saved Train: {os.path.basename(g_path)}", end="\r")

        if use_streaming:
            _save_pt_plots_streaming(hist_raw, hist_flat, bin_edges, config, train_out)
        else:
            _save_pt_plots(pts_all, labels_all, weights_all, config, train_out)

    if "test_graphs" in config["data"]:
        test_graph_files = []
        for p in config["data"]["test_graphs"]:
            test_graph_files.extend(sorted(glob.glob(p)))

        if test_graph_files:
            sprint(f"\n--- Stage 3: Normalization on {len(test_graph_files)} TEST batches ---")
            test_out = os.path.join(base_out_dir, "test")
            os.makedirs(test_out, exist_ok=True)

            for g_path in test_graph_files:
                dataset = torch.load(g_path, weights_only=False)
                for g in dataset:
                    g.x = ((g.x - mean_x) / std_x).float()
                    g.Ntrk = torch.tensor((float(g.Ntrk) - mean_ntrk) / std_ntrk).float()
                    g.weights = 1.0
                torch.save(
                    dataset,
                    os.path.join(test_out, f"processed_{os.path.basename(g_path)}"),
                )
                del dataset
                gc.collect()
                sprint(f" Saved Test: {os.path.basename(g_path)}", end="\r")

    sprint("\nPreprocessing complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--override", nargs="*", default=[], help="key=value overrides")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    cfg = recursive_update(cfg, parse_dot_args(args.override))
    process_lrj_pt_only(cfg)

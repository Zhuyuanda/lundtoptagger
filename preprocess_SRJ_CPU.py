import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt

from tools.GNN_model_weight.utils_newdata import load_yaml
from plotting.utils_plots_matplotlib import hist_with_errors


# ============================================================
#  Process one dataset (train or test)
# ============================================================
def process_one_split(graph_paths, config,
                      mean_x, std_x, mean_ntrk, std_ntrk,
                      do_flatten, tag, out_dir):

    is_train = (tag == "train")   

    # ------------------------------------
    # Load graph files
    # ------------------------------------
    if isinstance(graph_paths, str):
        graph_paths = [graph_paths]

    dataset = []
    for path in graph_paths:
        print(f"\nLoading {tag} graph file:", path)
        dataset += torch.load(path, weights_only=False)

    print(f"Total {tag} graphs loaded:", len(dataset))

    # ------------------------------------
    # Jet selection (consistent with training)
    # ------------------------------------
    if config["cut_pt_eta_mass"]:
        sig_cfg = load_yaml(config["config_signal_path"])[config["signal"]]

        pt_min, pt_max = sig_cfg["pt_range"]
        m_min, m_max = sig_cfg["mass_range"]
        eta_min = sig_cfg.get("eta_min", 0)
        eta_max = sig_cfg.get("eta_max", 4.0)

        dataset = [
            g for g in dataset
            if pt_min < g.pt < pt_max
            and m_min < g.mass < m_max
            and eta_min <= abs(g.eta) <= eta_max
        ]
        print(f"{tag}: remaining after cuts =", len(dataset))

    # ------------------------------------
    # Standardization
    # ------------------------------------
    print(f"\nStandardizing {tag} graphs...")

    for g in dataset:
        x = g.x.numpy()
        x = (x - mean_x) / std_x
        g.x = torch.from_numpy(x).float()

        g.Ntrk = ((g.Ntrk - mean_ntrk) / std_ntrk).float()

    # ------------------------------------
    # Prepare arrays for plotting / flattening
    # ------------------------------------
    labels = np.array([g.y for g in dataset])
    pts = np.array([g.pt for g in dataset])
    etas = np.array([g.eta for g in dataset])

    # ------------------------------------
    # Flatten (train only)
    # ------------------------------------
    if do_flatten:
        print(f"\nFlattening {tag} dataset...")

        from tools.GNN_model_weight.utils_newdata import (
            assign_flat_weights,
            assign_2d_flat_weights_kde,
        )

        do_2d = config["flatten_pt"] and config["flatten_eta"]
        do_1d = config["flatten_pt"] and not config["flatten_eta"]

        if do_2d:
            weights_sig = assign_2d_flat_weights_kde(etas[labels==1], pts[labels==1])
            weights_bkg = assign_2d_flat_weights_kde(etas[labels==0], pts[labels==0])
        else:
            weights_sig = assign_flat_weights(pts[labels==1], n_bins=config["n_bins_pt"])
            weights_bkg = assign_flat_weights(pts[labels==0], n_bins=config["n_bins_pt"])

        # Store weights
        sig_graphs = [g for g in dataset if g.y == 1]
        bkg_graphs = [g for g in dataset if g.y == 0]

        for g, w in zip(sig_graphs, weights_sig):
            g.weights = float(w)
        for g, w in zip(bkg_graphs, weights_bkg):
            g.weights = float(w)

    else:
        print(f"\nSkipping flattening for {tag} dataset.")
        weights_sig = np.ones(sum(labels == 1))
        weights_bkg = np.ones(sum(labels == 0))


    # ============================================================
    #  Visualization (TRAIN ONLY)
    # ============================================================
    if is_train:
        print("\n=== Saving SRJ distributions BEFORE and AFTER flattening ===")

        plot_dir = os.path.join(config['data']['out_dir'], "plots_before_after")
        os.makedirs(plot_dir, exist_ok=True)

        sig_mask = (labels == 1)
        bkg_mask = (labels == 0)

        pts_sig, pts_bkg = pts[sig_mask], pts[bkg_mask]
        etas_sig, etas_bkg = etas[sig_mask], etas[bkg_mask]

        do_2d = config["flatten_pt"] and config["flatten_eta"]
        do_1d = config["flatten_pt"] and not config["flatten_eta"]

        # ========================================================
        # 1D pT BEFORE
        # ========================================================
        plt.figure()
        hist_with_errors(pts_bkg, bins=config["n_bins_pt"], density=True,
                         fmt=".", capsize=2, label="Background")
        hist_with_errors(pts_sig, bins=config["n_bins_pt"], density=True,
                         fmt=".", label="Signal")
        plt.xlabel("SRJ pT [GeV]")
        plt.ylabel("Density")
        plt.legend()
        plt.savefig(os.path.join(plot_dir, "pT_before.png"))
        plt.close()

        # AFTER only for 1D flatten
        if do_1d:
            plt.figure()
            hist_with_errors(pts_bkg, weights=weights_bkg, bins=config["n_bins_pt"],
                             density=True, fmt=".", capsize=2, label="Background")
            hist_with_errors(pts_sig, weights=weights_sig, bins=config["n_bins_pt"],
                             density=True, fmt=".", label="Signal")
            plt.xlabel("SRJ pT [GeV]")
            plt.ylabel("Density")
            plt.legend()
            plt.savefig(os.path.join(plot_dir, "pT_after.png"))
            plt.close()

        # ========================================================
        # 1D η BEFORE
        # ========================================================
        plt.figure()
        hist_with_errors(etas_bkg, bins=config["n_bins_eta"], density=True,
                         fmt=".", capsize=2, label="Background")
        hist_with_errors(etas_sig, bins=config["n_bins_eta"], density=True,
                         fmt=".", label="Signal")
        plt.xlabel("SRJ η")
        plt.ylabel("Density")
        plt.legend()
        plt.savefig(os.path.join(plot_dir, "eta_before.png"))
        plt.close()

        # ========================================================
        # 2D BEFORE (only plotting BEFORE for train)
        # ========================================================
        plt.figure()
        H, xedges, yedges = np.histogram2d(
            pts_sig, etas_sig,
            bins=(config["n_bins_pt"], config["n_bins_eta"]),
            density=True
        )
        cmin = H[H > 0].min()
        plt.hist2d(pts_sig, etas_sig,
                   bins=(config["n_bins_pt"], config["n_bins_eta"]),
                   density=True, cmin=cmin)
        plt.colorbar(label="Density")
        plt.xlabel("SRJ pT [GeV]")
        plt.ylabel("SRJ η")
        plt.savefig(os.path.join(plot_dir, "2D_sig_before.png"))
        plt.close()

        # ========================================================
        # 2D AFTER (only for 2D flatten)
        # ========================================================
        if do_2d:
            plt.figure()
            H2, _, _ = np.histogram2d(
                pts_sig, etas_sig,
                bins=(config["n_bins_pt"], config["n_bins_eta"]),
                weights=weights_sig, density=True
            )
            cmin2 = H2[H2 > 0].min()
            plt.hist2d(pts_sig, etas_sig,
                       bins=(config["n_bins_pt"], config["n_bins_eta"]),
                       weights=weights_sig, density=True, cmin=cmin2)
            plt.colorbar(label="Density")
            plt.xlabel("SRJ pT [GeV]")
            plt.ylabel("SRJ η")
            plt.savefig(os.path.join(plot_dir, "2D_sig_after.png"))
            plt.close()

        print("\n✓ Finished plotting before/after flattening.\n")

    # ============================================================
    # Save processed dataset
    # ============================================================
    outfile = os.path.join(out_dir, f"processed_SRJ_{tag}.pt")
    torch.save(dataset, outfile)
    print(f"\n✓ Saved {tag} dataset → {outfile}")
    print("-" * 60)



# ============================================================
#  Main function
# ============================================================
def preprocess_SRJ_CPU(config):

    # Load mean/std JSON
    stats = json.load(open(config["data"]["meanstd_json"]))
    mean_x = np.array(stats["mean_x"], dtype=np.float32)
    std_x  = np.array(stats["std_x"], dtype=np.float32)
    mean_ntrk = float(stats["mean_ntrk"])
    std_ntrk  = float(stats["std_ntrk"])

    out_dir = config["data"]["out_dir"]

    # TRAIN (with flatten)
    process_one_split(
        graph_paths = config["data"]["train_graphs"],
        config = config,
        mean_x = mean_x, std_x = std_x,
        mean_ntrk = mean_ntrk, std_ntrk = std_ntrk,
        do_flatten = True,
        tag = "train",
        out_dir = out_dir
    )

    # TEST (no flatten)
    if "test_graphs" in config["data"]:
        process_one_split(
            graph_paths = config["data"]["test_graphs"],
            config = config,
            mean_x = mean_x, std_x = std_x,
            mean_ntrk = mean_ntrk, std_ntrk = std_ntrk,
            do_flatten = False,
            tag = "test",
            out_dir = out_dir
        )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Config YAML")
    args = parser.parse_args()

    config = load_yaml(args.config)
    preprocess_SRJ_CPU(config)

import os
import json
import torch
import numpy as np
import uproot
import glob
import matplotlib.pyplot as plt
import sys

# 请确保这些工具路径在你的环境中正确
from tools.GNN_model_weight.utils_newdata import load_yaml, assign_flat_weights
from plotting.utils_plots_matplotlib import hist_with_errors

def sprint(*args, **kwargs):
    print(*args, **kwargs); sys.stdout.flush()

# ============================================================
# 1. 验证绘图：检查训练集 pT 重权重效果
# ============================================================
def _save_pt_plots(pts, labels, weights_all, config, out_dir):
    plot_dir = os.path.join(out_dir, "plots_reweighting")
    os.makedirs(plot_dir, exist_ok=True)
    sig_mask, bkg_mask = (labels == 1), (labels == 0)
    
    plt.figure(figsize=(12, 5))
    # 重权重前
    plt.subplot(1, 2, 1)
    hist_with_errors(pts[bkg_mask], bins=config["n_bins_pt"], density=True, label="Bkg (Raw)")
    hist_with_errors(pts[sig_mask], bins=config["n_bins_pt"], density=True, label="Top (Raw)")
    plt.xlabel("Jet pT [GeV]"); plt.ylabel("Density"); plt.title("Before Flattening"); plt.legend()
    # 重权重后
    plt.subplot(1, 2, 2)
    hist_with_errors(pts[bkg_mask], weights=weights_all[bkg_mask], bins=config["n_bins_pt"], density=True, label="Bkg (Flat)")
    hist_with_errors(pts[sig_mask], weights=weights_all[sig_mask], bins=config["n_bins_pt"], density=True, label="Top (Flat)")
    plt.xlabel("Jet pT [GeV]"); plt.ylabel("Density"); plt.title("After Flattening"); plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "pT_flattening_check.png"))
    plt.close()
    sprint(f"\n[Check] pT validation plots saved to: {plot_dir}")

# ============================================================
# 2. 核心处理函数
# ============================================================
def process_lrj_pt_only(config):
    # 1. 加载标准化参数 (Mean/Std)
    stats = json.load(open(config["data"]["meanstd_json"]))
    mean_x = torch.from_numpy(np.array(stats["mean_x"], dtype=np.float32))
    std_x = torch.from_numpy(np.array(stats["std_x"], dtype=np.float32))
    mean_ntrk, std_ntrk = float(stats["mean_ntrk"]), float(stats["std_ntrk"])

    base_out_dir = config["data"]["out_dir"]
    
    # ------------------------------------------------------------
    # PART A: 处理训练集 (归一化 + pT Flattening)
    # ------------------------------------------------------------
    train_graph_files = []
    for p in config["data"]["train_graphs"]:
        train_graph_files.extend(sorted(glob.glob(p)))
    
    if train_graph_files:
        sprint(f"--- Stage 1: Calculating Global pT Weights for {len(train_graph_files)} train batches ---")
        pts_list, labels_list = [], []
        for g_path in train_graph_files:
            # 路径转换: graphs_*.pt -> data_*.root
            r_path = g_path.replace("graphs_", "data_").replace("_with_pt", "").replace(".pt", ".root")
            with uproot.open(r_path) as f:
                tree = f["FlatSubstructureJetTree"]
                pts_list.append(tree["fjet_pt"].array(library="np"))
                labels_list.append(tree["labels"].array(library="np"))
        
        pts_all = np.concatenate(pts_list)
        labels_all = np.concatenate(labels_list)
        weights_all = np.zeros_like(pts_all, dtype=float)

        for lab in [0, 1]:
            mask = (labels_all == lab)
            if np.any(mask):
                weights_all[mask] = assign_flat_weights(pts_all[mask], n_bins=config["n_bins_pt"])

        sprint("\n--- Stage 2: Applying Normalization & pT Weights to TRAIN set ---")
        ptr = 0 
        train_out = os.path.join(base_out_dir, "train")
        os.makedirs(train_out, exist_ok=True)

        for g_path in train_graph_files:
            dataset = torch.load(g_path, weights_only=False)
            for g in dataset:
                # 特征标准化
                g.x = ((g.x - mean_x) / std_x).float()
                g.Ntrk = torch.tensor((float(g.Ntrk) - mean_ntrk) / std_ntrk).float()
                g.weights = float(weights_all[ptr])
                ptr += 1
            torch.save(dataset, os.path.join(train_out, f"processed_{os.path.basename(g_path)}"))
            sprint(f" Saved Train: {os.path.basename(g_path)}", end="\r")
        _save_pt_plots(pts_all, labels_all, weights_all, config, train_out)

    # ------------------------------------------------------------
    # PART B: 处理测试集 (仅归一化，权重固定为 1.0)
    # ------------------------------------------------------------
    if "test_graphs" in config["data"]:
        test_graph_files = []
        for p in config["data"]["test_graphs"]:
            test_graph_files.extend(sorted(glob.glob(p)))
        
        if test_graph_files:
            sprint(f"\n--- Stage 3: Applying Normalization to {len(test_graph_files)} TEST batches ---")
            test_out = os.path.join(base_out_dir, "test")
            os.makedirs(test_out, exist_ok=True)

            for g_path in test_graph_files:
                dataset = torch.load(g_path, weights_only=False)
                for g in dataset:
                    # 使用与训练集完全相同的 Mean/Std
                    g.x = ((g.x - mean_x) / std_x).float()
                    g.Ntrk = torch.tensor((float(g.Ntrk) - mean_ntrk) / std_ntrk).float()
                    g.weights = 1.0 # 测试集不重权重
                
                torch.save(dataset, os.path.join(test_out, f"processed_{os.path.basename(g_path)}"))
                sprint(f" Saved Test: {os.path.basename(g_path)}", end="\r")
    
    sprint("\nPreprocessing complete.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    process_lrj_pt_only(load_yaml(args.config))
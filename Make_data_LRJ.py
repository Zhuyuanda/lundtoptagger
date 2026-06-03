import argparse
import os
import json
import glob
import time
import gc

import uproot
import awkward as ak
import numpy as np
import torch

from tools.GNN_model_weight.utils_newdata import (
    load_yaml, 
    lrj_create_train_dataset_pure
)
from tools.utils_config import recursive_update, parse_dot_args

print("Libraries loaded!")

def main():
    parser = argparse.ArgumentParser(description="LRJ Data Prep (SRJ Workflow Style): Pure Extraction")
    parser.add_argument("config", help="job configuration file")
    parser.add_argument('--override', nargs='*', default=[], help='Overrides in the form key.subkey=value')

    args = parser.parse_args()
    config = load_yaml(args.config)
    override_dict = parse_dot_args(args.override)
    config = recursive_update(config, override_dict)

    max_jets_per_batch = config.get("max_jets_per_batch", 10000)

    config_signal = load_yaml(config["signal_config_file"])
    signal = config["signal"]
    signals = [s for s in config_signal.keys() if s != "bkg_histos"] if signal == "all" else [signal]

    path_list = [config["path_to_rootfiles"]] if isinstance(config["path_to_rootfiles"], str) else config["path_to_rootfiles"]
    files = []
    for pat in path_list:
        files.extend(glob.glob(pat))
    if config.get("n_files"):
        files = files[:config["n_files"]]

    print(f"Processing {len(files)} files...")

    # LRJ 高级属性映射 (含 GN3X, Transformers, B-Tagging)
    jet_property_names = {
        "fjet_m": "LRJ_mass", "fjet_pt": "LRJ_pt", "fjet_eta": "LRJ_eta", "fjet_phi": "LRJ_phi", 
        "fjet_truth_label": "LRJ_truthLabel", "fjet_Nconst_Charged": "LRJ_Nconst_Charged",
        "has_inclusive_b": "LRJ_hasInclusiveB", "has_reco_b75": "LRJ_hasRecoB_75", "has_reco_b50": "LRJ_hasRecoB_50",
        "GN3X_phtautauhad": "GN3XPV01_phtautauhad", "GN3X_phbb": "GN3XPV01_phbb", "GN3X_phcc": "GN3XPV01_phcc",
        "GN3X_ptop": "GN3XPV01_ptop", "GN3X_pqcdbb": "GN3XPV01_pqcdbb", "GN3X_pqcdbx": "GN3XPV01_pqcdbx",
        "GN3X_pqcdcx": "GN3XPV01_pqcdcx", "GN3X_pqcdll": "GN3XPV01_pqcdll", "GN3X_pWqq": "GN3XPV01_pWqq",
        "WTrans_massdec": "LRJ_WTransformer_massdec_ConstScore", "WTrans_score": "LRJ_WTransformer_ConstScore",
        "TopTrans_score": "LRJ_TopTransformer_ConstScore",
    }
    
    # 移除这里对 fjet_weight_pt 的计算，仅保留原始 MC 权重，权重重新计算将在 preprocess_lrj.py 中完成
    additional_output_vars = ["EventInfo_mcEventWeight", "EventInfo_mcChannelNumber"]

    def welford_update(mean, M2, count, x):
        count += 1
        delta = x - mean
        mean += delta / count
        M2 += delta * (x - mean)
        return mean, M2, count

    event_fractions = []
    for frac, n_chunks in config["event_fractions"].items():
        event_fractions.extend([frac] * n_chunks)
    
    event_indices = [config["event_fraction_idx"]] if config["event_fraction_idx"] is not None else list(range(len(event_fractions)))

    for frac_idx in event_indices:
        event_fraction = event_fractions[frac_idx]
        percent_str = f"{event_fraction * 100:.2f}".rstrip('0').rstrip('.') + "percent"
        final_out_dir = os.path.join(config["out_dir"].format(id=config["id"]), f"part{frac_idx}_{percent_str}")
        os.makedirs(final_out_dir, exist_ok=True)
        
        print(f"\n>>> Processing Part {frac_idx} (Saving to: {final_out_dir})")

        dataset = []
        primary_Lund_only_one_arr = []
        batch_idx = 0

        # Welford 统计变量
        node_count, node_mean, node_M2 = 0, np.zeros(3), np.zeros(3)
        jet_count, jet_mean_ntrk, jet_M2_ntrk = 0, 0.0, 0.0

        out_tree_dict = {name: ak.Array([]) for name in [*jet_property_names.keys(), *additional_output_vars, "labels"]}

        def save_batch(data_list, tree_dict, b_idx):
            if not data_list: return
            placeholders = dict(id=config["id"], kT_cut=config["kT_cut"], 
                                include_pt="_with_pt" if config["include_pt"] else "", frac=f"_batch{b_idx}")

            torch.save(data_list, os.path.join(final_out_dir, config["out_file_name_graphs"].format(**placeholders)))
            
            tree_dict["labels"] = ak.Array([float(g.y.item()) for g in data_list])
            valid_tree = {k: v for k, v in tree_dict.items() if len(v) > 0}
            with uproot.recreate(os.path.join(final_out_dir, config["out_file_name_root"].format(**placeholders))) as f:
                f["FlatSubstructureJetTree"] = valid_tree
            print(f"Batch {b_idx} saved with {len(data_list)} jets.")

        for file_num, file in enumerate(files, start=1):
            with uproot.open(file) as infile:
                tree = infile["AnalysisTree"]
                dsids = tree["dsid"].array(library="np")
                if dsids[0] in set.intersection(*[set(config_signal[s]["skip_dsids"]) for s in signals]): continue

                total_e = tree.num_entries
                start = int(total_e * sum(event_fractions[:frac_idx]))
                stop = int(total_e * (sum(event_fractions[:frac_idx]) + event_fraction))
                if start >= stop: continue

                # 动态加载分支
                lund_vars = ["jetLundZ", "jetLundKt", "jetLundDeltaR", "jetLundIDParent1", "jetLundIDParent2"]
                avail_branches = [b for b in [*jet_property_names.values(), *lund_vars] if b in tree.keys()]
                
                raw_data = tree.arrays(avail_branches, entry_start=start, entry_stop=stop, library="ak")
                jet_props = {k: ak.flatten(raw_data[k]) for k in raw_data.fields}

                n_jets = ak.num(raw_data[avail_branches[0]])
                jet_props["EventInfo_mcEventWeight"] = np.repeat(tree["mcEventWeight"].array(entry_start=start, entry_stop=stop, library="np"), n_jets)
                jet_props["EventInfo_mcChannelNumber"] = np.repeat(dsids[start:stop], n_jets)

                # 获取信号物理筛选范围
                pt_min = min(config_signal[s]["pt_range"][0] for s in signals)
                pt_max = max(config_signal[s]["pt_range"][1] for s in signals)
                mass_min = min(config_signal[s]["mass_range"][0] for s in signals)
                mass_max = max(config_signal[s]["mass_range"][1] for s in signals)
                e_max = max(config_signal[s]["eta_max"] for s in signals)
                
                # 预筛选掩码用于计算准确的 Welford 统计量
                mask = (
                    (jet_props["LRJ_pt"] > pt_min) & (jet_props["LRJ_pt"] < pt_max)
                    & (jet_props["LRJ_mass"] > mass_min) & (jet_props["LRJ_mass"] < mass_max)
                    & (np.abs(jet_props["LRJ_eta"]) < e_max)
                )
                
                # Welford 更新 Ntrk
                for val in ak.to_numpy(jet_props["LRJ_Nconst_Charged"][mask]).astype(float):
                    jet_mean_ntrk, jet_M2_ntrk, jet_count = welford_update(jet_mean_ntrk, jet_M2_ntrk, jet_count, val)

                # Welford 更新 原始对数特征
                raw_nodes = np.vstack([
                    np.log(1.0 / (ak.to_numpy(ak.flatten(jet_props["jetLundDeltaR"][mask])) + 1e-4)),
                    np.log(1.0 / (ak.to_numpy(ak.flatten(jet_props["jetLundZ"][mask])) + 1e-4)),
                    np.log(ak.to_numpy(ak.flatten(jet_props["jetLundKt"][mask])) + 1e-4)
                ]).T
                for v in raw_nodes: 
                    node_mean, node_M2, node_count = welford_update(node_mean, node_M2, node_count, v)

                # 创建纯净数据集
                passed_selection = []
                aux_scores = {k: jet_props[v] for k, v in jet_property_names.items() 
                              if v in jet_props and ("GN3" in v or "Trans" in v or "has" in v)}

                dataset = lrj_create_train_dataset_pure(
                    dataset, *[jet_props[k] for k in lund_vars],
                    *[jet_props[k] for k in ["LRJ_truthLabel", "EventInfo_mcChannelNumber", "LRJ_Nconst_Charged", "LRJ_pt", "LRJ_mass", "LRJ_eta"]],
                    extra_features = aux_scores,
                    kT_selection=config["kT_cut"], 
                    primary_Lund_only_one_arr=primary_Lund_only_one_arr,
                    passed_selection=passed_selection,
                    signal_jet_truth_labels=set().union(*[config_signal[s]["signal_jet_truth_labels"] for s in signals]),
                    signal_dsids=set().union(*[config_signal[s]["dsids"] for s in signals]),
                    pt_range=(pt_min, pt_max), 
                    mass_range=(mass_min, mass_max),
                    eta_max=e_max,
                    include_pt=config["include_pt"]
                )

                for out_n, in_n in jet_property_names.items():
                    if in_n in jet_props:
                        out_tree_dict[out_n] = ak.concatenate([out_tree_dict[out_n], jet_props[in_n][passed_selection]])
                for var in additional_output_vars:
                    out_tree_dict[var] = ak.concatenate([out_tree_dict[var], jet_props[var][passed_selection]])

                if len(dataset) >= max_jets_per_batch:
                    save_batch(dataset, out_tree_dict, batch_idx)
                    dataset, batch_idx = [], batch_idx + 1
                    out_tree_dict = {name: ak.Array([]) for name in out_tree_dict}
                    gc.collect()

        if dataset:
            save_batch(dataset, out_tree_dict, batch_idx)

        stats = {
            "mean_x": node_mean.tolist(), "std_x": np.sqrt(node_M2 / max(node_count - 1, 1)).tolist(),
            "mean_ntrk": float(jet_mean_ntrk), "std_ntrk": float(np.sqrt(jet_M2_ntrk / max(jet_count - 1, 1))),
            "nodes": int(node_count), "jets": int(jet_count)
        }
        with open(os.path.join(final_out_dir, f"meanstd_part{frac_idx}.json"), "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Part {frac_idx} complete. Stats saved to {final_out_dir}")

if __name__ == "__main__":
    main()
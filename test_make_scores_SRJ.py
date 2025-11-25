import argparse
import os
import glob
import time
import gc

import uproot
import numpy as np
import torch
from torch_geometric.loader import DataLoader

from tools.GNN_model_weight.models import *
from tools.GNN_model_weight.utils_newdata import load_yaml, get_scores
from tools.utils_config import recursive_update, parse_dot_args

print("Libraries loaded!")

def build_model_by_name(name):
    if name == "LundNet":
        return LundNet()
    elif name == "GATNet":
        return GATNet()
    elif name == "GINNet":
        return GINNet()
    elif name == "EdgeGinNet":
        return EdgeGinNet()
    elif name == "PNANet":
        return PNANet()
    elif name == "LundNet_plus_GN2X":
        return LundNet_plus_GN2X()
    elif name == "LundNet_plus_GN3X":
        return LundNet_plus_GN3X()
    else:
        raise ValueError(f"Unknown model type {name}")

def main():
    parser = argparse.ArgumentParser(description='Run SRJ scoring')
    add_arg = parser.add_argument
    add_arg('config', help="job configuration")
    parser.add_argument('--override', nargs='*', default=[])
    args = parser.parse_args()

    config = load_yaml(args.config)
    override_dict = parse_dot_args(args.override)
    config = recursive_update(config, override_dict)

    kT_selection = config['data']['kT_cut']
    placeholder = dict(
        sample=config['data']['sample'],
        kT_cut=kT_selection
    )

    # Resolve input ROOT files
    paths_root = config['data']['paths_to_test_file_root']
    if isinstance(paths_root, str):
        paths_root = [paths_root]
    files_root = []
    for p in paths_root:
        files_root.extend(glob.glob(p.format(**placeholder)))
    files_root.sort()
    print("ROOT input files:", files_root)

    # Resolve graph files
    paths_graphs = config['data']['paths_to_test_file_graphs']
    if isinstance(paths_graphs, str):
        paths_graphs = [paths_graphs]
    files_graphs = []
    for p in paths_graphs:
        files_graphs.extend(glob.glob(p.format(**placeholder)))
    files_graphs.sort()
    print("Graph input files:", files_graphs)

    # Output
    outdir = config['data']['path_to_outdir'].format(**placeholder)
    os.makedirs(outdir, exist_ok=True)
    print("Saving output to:", outdir)

    output_suffix = config['data']['output_suffix'].format(**placeholder)

    # Count entries
    tree_name = "FlatSubstructureJetTree"
    files_and_trees = {fn: tree_name for fn in files_root}
    nentries_total = sum(entry[-1] for entry in uproot.num_entries(files_and_trees))
    nentries_done = 0

    batch_size = config['test']['batch_size']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("\nUsing device:", device)

    t_global = time.time()

    # Main evaluation loop
    for file_number, (fg, fr) in enumerate(zip(files_graphs, files_root), start=1):

        t_start = time.time()
        print(f"\nLoading graph file {file_number}/{len(files_graphs)}:", fg)

        dataset = torch.load(fg, weights_only=False)
        n_jets = len(dataset)
        print("Dataset size:", n_jets)

        test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        del dataset
        torch.cuda.empty_cache()

        multi_scores = {}

        # ---------------------------
        # Run all models in config
        # ---------------------------
        for m in config["test"]["models_to_run"]:
            tag = m["tag"]
            arch = m["arch"]
            ckpt = m["ckpt"]

            print(f"\n=== Running model {tag} ({arch}) ===")
            model = build_model_by_name(arch)
            model.load_state_dict(torch.load(ckpt, map_location=device))
            model.to(device)
            model.eval()

            y_pred = get_scores(test_loader, model, device)
            multi_scores[tag] = np.array(y_pred[:, 0])

            del model, y_pred
            torch.cuda.empty_cache()

        # Load original ROOT tree
        print("\nLoading ROOT tree...")
        filename = os.path.splitext(os.path.basename(fr))[0]
        outfile = os.path.join(outdir, filename) + output_suffix + ".root"
        outfile = outfile.format(**placeholder)

        infile = outfile if os.path.exists(outfile) else fr
        with uproot.open(infile) as f:
            arrays = f[tree_name].arrays()

        # Add all branches from all models
        for tag, scores in multi_scores.items():
            branch_name = config["test"]["scores_branch_name"].format(tag=tag)
            arrays[branch_name] = scores

        # Save to ROOT
        print("Saving scores to:", outfile)
        with uproot.recreate(outfile) as f:
            f[tree_name] = arrays

        del arrays
        gc.collect()

        # Progress logging
        nentries_done += n_jets
        time_per_entry = (time.time() - t_start) / n_jets
        eta = time_per_entry * (nentries_total - nentries_done)
        m, s = divmod(int(eta), 60)
        print(f"Progress: {nentries_done}/{nentries_total}, ETA {m}m {s}s")

    t_total = time.time() - t_global
    m, s = divmod(int(t_total), 60)
    print(f"\nTotal evaluation time: {m} min {s} s")

if __name__ == "__main__":
    main()

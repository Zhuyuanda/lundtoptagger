import argparse
import os
import glob
import time
from datetime import timedelta
import gc
from operator import itemgetter

import uproot
import awkward as ak
import numpy as np
import torch

from tools.GNN_model_weight.utils_newdata import (
    load_yaml,
    srj_create_train_dataset_fulld_new_Ntrk_pt_file
)
from tools.utils_config import recursive_update, parse_dot_args

print("Libraries loaded!")


def main():
    parser = argparse.ArgumentParser(description="Prepare data for classifier input (SRJ version)")
    add_arg = parser.add_argument
    add_arg("config", help="job configuration file")
    add_arg('--override', nargs='*', default=[], help='Overrides in the form key.subkey=value')

    args = parser.parse_args()
    config_file = args.config

    config = load_yaml(config_file)
    override_dict = parse_dot_args(args.override)
    config = recursive_update(config, override_dict)

    # signal config
    config_signal = load_yaml(config["signal_config_file"])
    signal = config["signal"]
    signals = [s for s in config_signal.keys() if s != "bkg_histos"] if signal == "all" else [signal]

    # ROOT files
    path_to_files = config["path_to_rootfiles"]
    files = glob.glob(path_to_files)[:config["n_files"]]

    intreename = "AnalysisTree"
    n_files = len(files)
    print(f"Processing {n_files} files")

    # event fractions
    event_fractions = []
    for frac, n_chunks in config["event_fractions"].items():
        event_fractions.extend([frac] * n_chunks)
    if sum(event_fractions) > 1.0 + 1e-8:
        raise ValueError("Sum of event_fractions exceeds 1.")

    if config["event_fraction_idx"] is not None:
        event_fraction_indices = [config["event_fraction_idx"]]
    else:
        event_fraction_indices = list(range(len(event_fractions)))

    t_start = time.time()

    # ------------------------
    # SRJ jet property mapping
    # ------------------------
    jet_property_names = {
        "fjet_m":              "SRJ_mass",
        "fjet_pt":             "SRJ_pt",
        "fjet_eta":            "SRJ_eta",
        "fjet_phi":            "SRJ_phi",
        "fjet_truth_label":    "SRJ_partonTruthLabel",
        "fjet_Nconst":         "SRJ_Nconst",
        "fjet_Nconst_Charged": "SRJ_Nconst_Charged",
    }

    # additional variables per jet
    additional_output_vars = [
        "labels",
        "EventInfo_mcEventWeight",
        "EventInfo_mcChannelNumber",
    ]


    # Lund variables in SRJ
    lund_vars = [
        "SRJ_jetLundZ",
        "SRJ_jetLundKt",
        "SRJ_jetLundDeltaR",
        "SRJ_jetLundPt1",
        "SRJ_jetLundPt2",
    ]

    # ------------------------
    # Main event loop
    # ------------------------
    for frac_idx in event_fraction_indices:
        event_fraction = event_fractions[frac_idx]
        print(f"\nProcessing event fraction {event_fraction} ({frac_idx}/{len(event_fractions)})")

        dataset = []
        primary_Lund_only_one_arr = []

        out_tree_dict = {
            branch_name: ak.Array([]) for branch_name in
            [*jet_property_names.keys(), *additional_output_vars]
        }

        for file_number, file in enumerate(files, start=1):
            print(f"\nLoading file: {file_number}/{n_files}\n{file}")

            with uproot.open(file) as infile:
                tree = infile[intreename]

                dsids = tree["dsid"].array(library="np")
                dsid_test = dsids[0]

                skip_dsids = set.intersection(*[set(config_signal[s]["skip_dsids"]) for s in signals])
                if dsid_test in skip_dsids:
                    print("Skipping file with DSID", dsid_test)
                    continue

                total_events = tree.num_entries
                prev_fractions = sum(event_fractions[:frac_idx])

                start_entry = int(total_events * prev_fractions)
                stop_entry = int(total_events * (prev_fractions + event_fraction))
                stop_entry = min(stop_entry, total_events)

                if start_entry >= stop_entry:
                    print(f"Skipping: start_entry >= stop_entry")
                    continue

                print(f"Loading entries {start_entry}:{stop_entry} of {total_events}")

                # -------------------------------
                # Load per-jet variables (SRJ)
                # -------------------------------
                jet_properties = {}

                # jet properties from mapping
                for inname in jet_property_names.values():
                    if inname in tree:
                        jet_properties[inname] = ak.flatten(
                            tree[inname].array(entry_start=start_entry, entry_stop=stop_entry, library="ak")
                        )

                # Correct SRJ → internal name mapping
                lund_vars_map = {
                    "SRJ_jetLundZ":       "jetLundZ",
                    "SRJ_jetLundKt":      "jetLundKt",
                    "SRJ_jetLundDeltaR":  "jetLundDeltaR",
                    "SRJ_jetLundIDParent1": "jetLundIDParent1",
                    "SRJ_jetLundIDParent2": "jetLundIDParent2",
                }

                for srj_name, out_name in lund_vars_map.items():
                    if srj_name in tree:
                        jet_properties[out_name] = ak.flatten(
                            tree[srj_name].array(entry_start=start_entry, entry_stop=stop_entry, library="ak")
                        )
                    else:
                        print(f"Warning: {srj_name} not found in file {file}")


                # truth labels (per event → per jet)
                truth_labels_unflattened = tree["SRJ_partonTruthLabel"].array(
                    entry_start=start_entry, entry_stop=stop_entry, library="ak"
                )
                numbers_of_jets_per_event = ak.num(truth_labels_unflattened)

                # event weights expanded per jet
                mcEventWeights = tree["mcEventWeight"].array(
                    entry_start=start_entry, entry_stop=stop_entry, library="np"
                )
                jet_properties["EventInfo_mcEventWeight"] = np.repeat(mcEventWeights, numbers_of_jets_per_event)

                jet_properties["EventInfo_mcChannelNumber"] = np.repeat(
                    dsids[start_entry:stop_entry],
                    numbers_of_jets_per_event
                )


                # -------------------------------
                # Build dataset (graphs)
                # -------------------------------
                print("\nCreating PyTorch graphs:")
                passed_selection = []

                dataset = srj_create_train_dataset_fulld_new_Ntrk_pt_file(
                    dataset,
                    jet_properties["jetLundZ"],
                    jet_properties["jetLundKt"],
                    jet_properties["jetLundDeltaR"],
                    jet_properties["jetLundIDParent1"],
                    jet_properties["jetLundIDParent2"],
                    jet_properties["SRJ_partonTruthLabel"],
                    jet_properties["EventInfo_mcChannelNumber"],
                    jet_properties["SRJ_Nconst_Charged"],
                    jet_properties["SRJ_pt"],
                    jet_properties["SRJ_mass"],
                    jet_properties["SRJ_eta"],
                    kT_selection=config["kT_cut"],
                    primary_Lund_only_one_arr=primary_Lund_only_one_arr,
                    passed_selection=passed_selection,
                    signal_jet_truth_labels=set().union(*[config_signal[s]["signal_jet_truth_labels"] for s in signals]),
                    signal_dsids=set().union(*[config_signal[s]["dsids"] for s in signals]),
                    pt_range=(
                        min(min(config_signal[s]["pt_range"]) for s in signals),
                        max(max(config_signal[s]["pt_range"]) for s in signals)
                    ),
                    mass_range=(
                        min(min(config_signal[s]["mass_range"]) for s in signals),
                        max(max(config_signal[s]["mass_range"]) for s in signals)
                    ),
                    eta_max=max(config_signal[s]["eta_max"] for s in signals),
                    min_splits=min(config_signal[s]["min_splits"] for s in signals),
                    include_pt=config["include_pt"],
                )

                # -------------------------------
                # Fill ROOT output dictionary
                # -------------------------------
                for outname, inname in jet_property_names.items():
                    if inname in jet_properties:
                        out_tree_dict[outname] = ak.concatenate([
                            out_tree_dict[outname],
                            jet_properties[inname][passed_selection]
                        ])

                for output_var in additional_output_vars:
                    if output_var != "labels":
                        if output_var in jet_properties:
                            out_tree_dict[output_var] = ak.concatenate([
                                out_tree_dict[output_var],
                                jet_properties[output_var][passed_selection]
                            ])

                gc.collect()

        # Labels from graph objects
        out_tree_dict["labels"] = ak.Array([g.y for g in dataset])

        print("\nDataset created! len():", len(dataset))
        print("Time:", timedelta(seconds=round(time.time() - t_start)))

        # -------------------------------
        # Save output
        # -------------------------------
        filepath_placeholder_vals = dict(
            id=config["id"],
            kT_cut=config["kT_cut"],
            include_pt="_with_pt" if config["include_pt"] else "",
            frac=f"_part{frac_idx}_{event_fraction * 100:.2f}percent"
            if event_fraction < 1.0 else "",
        )

        out_dir = config["out_dir"].format(**filepath_placeholder_vals)
        os.makedirs(out_dir, exist_ok=True)

        # graphs
        out_file_name_graphs = config["out_file_name_graphs"].format(**filepath_placeholder_vals)
        output_path_graphs = os.path.join(out_dir, out_file_name_graphs)
        torch.save(dataset, output_path_graphs)
        print("Graphs saved to:", output_path_graphs)

        # ROOT file
        outfile_name_root = config["out_file_name_root"].format(**filepath_placeholder_vals)
        output_path_root = os.path.join(out_dir, outfile_name_root)
        with uproot.recreate(output_path_root) as outfile:
            outfile["FlatSubstructureJetTree"] = out_tree_dict

        print("ROOT written:", output_path_root)


if __name__ == "__main__":
    main()

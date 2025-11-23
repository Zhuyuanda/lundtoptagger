import argparse
import csv
from datetime import datetime
import os
import json

import numpy as np
import torch
from torch_geometric.utils import degree
from torch_geometric.loader import DataLoader
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from tools.GNN_model_weight.models import *
from tools.GNN_model_weight.utils_newdata import *
from plotting.utils_plots_matplotlib import hist_with_errors

print("Libraries loaded!")


def main():

    parser = argparse.ArgumentParser(description='Train with configurations')
    add_arg = parser.add_argument
    add_arg('config', help="job configuration")
    add_arg('--ln_kT_cut', type=float, help="minimum value of kT kept for the training graphs")
    add_arg(
        '--do_combined_training',
        type=lambda x: str(x).lower(),
        choices=["true", "false", "yes", "no", "1", "0"],
        help="value can be true/false, yes/no, 0/1, case insensitive"
    )
    args = parser.parse_args()

    config_file = args.config
    config = load_yaml(config_file)
    ln_kT_cut = args.ln_kT_cut if args.ln_kT_cut is not None else config['data']['ln_kT_cut']
    do_combined_training = (
        True if args.do_combined_training in ["true", "yes", "1"] else
        False if args.do_combined_training in ["false", "no", "0"] else
        config['architecture']['do_combined_training']
    )
    
    # load the dataset
    path_to_file = config['data']['path_to_trainfiles']
    dataset = []
    if isinstance(path_to_file, str):
        # path_to_file can be a list of file paths or a single path
        # if it is a single path, convert it to a list
        path_to_file = [path_to_file]
    for file_path in path_to_file:
        file_path = file_path.format(ln_kT_cut=ln_kT_cut)
        print("Loading file", file_path)
        dataset += torch.load(file_path, weights_only=False) # weights_only=False added so that it works with PyTorch 2.6; it used to be the default

    # -------------------------------------------------------
    # Load normalization stats from JSON (slice mean/std)
    # -------------------------------------------------------
    json_path = config['data']['meanstd_json']
    print(f"\nLoading normalization stats from: {json_path}")

    with open(json_path, 'r') as f:
        stats = json.load(f)

    mean_x = np.array(stats["mean_x"], dtype=np.float32)     # [mean_dr, mean_z, mean_kt]
    std_x  = np.array(stats["std_x"], dtype=np.float32)
    mean_ntrk = float(stats["mean_ntrk"])
    std_ntrk  = float(stats["std_ntrk"])

    # apply jet mass, pT, eta cuts
    if config['cut_pt_eta_mass']:
        config_signal = load_yaml(config['config_signal_path'])[config['signal']]
        
        pt_range   = config_signal['pt_range']
        mass_range = config_signal['mass_range']
        eta_min    = config_signal.get('eta_min', 0.0)
        eta_max    = config_signal.get('eta_max', 2.0)  # default if missing

        print("Filtering jets with:")
        print(f"  pT   in {pt_range}")
        print(f"  mass in {mass_range}")
        print(f"  |eta| between {eta_min} and {eta_max}")

        dataset = [
            jet_graph for jet_graph in dataset
            if  (pt_range[0]   < jet_graph.pt   < pt_range[1]) and
                (mass_range[0] < jet_graph.mass < mass_range[1]) and
                (eta_min <= abs(jet_graph.eta) <= eta_max)
        ]

        
    # -------------------------------------------------------
    # Standardization using slice-level JSON stats
    # (after pT + mass cuts to reduce computation)
    # -------------------------------------------------------
    print("Applying standardization to dataset...")

    for g in dataset:
        # normalize node features x: shape (N_nodes, 3)
        x = g.x.numpy()                       # convert to numpy
        x = (x - mean_x) / std_x              # standardize
        g.x = torch.tensor(x, dtype=torch.float)

        # normalize Ntrk
        g.Ntrk = torch.tensor(
            (g.Ntrk - mean_ntrk) / std_ntrk,
            dtype=torch.float
        )

    # check the number of signal and background jets
    labels = np.array([jet_graph.y for jet_graph in dataset])
    num_signal = (labels==1).sum()
    num_background = (labels==0).sum()
    print("")
    print("Signal count:", num_signal)
    print("Background count:", num_background)

    # ===============================================
    #  SRJ flattening: support (pT) or (pT, η)
    # ===============================================
    
    etas = np.array([jet_graph.eta for jet_graph in dataset])
    pts  = np.array([jet_graph.pt  for jet_graph in dataset])

    flatten_eta = config.get('flatten_eta', True)
    flatten_pt  = config.get('flatten_pt', True)

    # -------------------------------
    # Case 1: (pT, η) 2D flattening
    # -------------------------------
    if flatten_pt and flatten_eta:
        print("Applying 2D KDE flattening in (pT, η)…")

        weights_sig = assign_2d_flat_weights_kde(
            mass = etas[labels == 1],   # SRJ: "mass" → use eta
            pt   = pts [labels == 1],
        )

        weights_bkg = assign_2d_flat_weights_kde(
            mass = etas[labels == 0],
            pt   = pts [labels == 0],
        )

    # -------------------------------
    # Case 2: pT-only 1D flattening
    # -------------------------------
    elif flatten_pt and (not flatten_eta):
        print("Applying 1D flattening only on pT...")
        weights_sig = assign_flat_weights(
            pts[labels == 1],
            n_bins = config.get("n_bins_pt", 40)
        )
        weights_bkg = assign_flat_weights(
            pts[labels == 0],
            n_bins = config.get("n_bins_pt", 40)
        )

    # -------------------------------
    # Case 3: η-only flattening → forbidden
    # -------------------------------
    elif flatten_eta and not flatten_pt:
        raise RuntimeError("η-only flattening is not supported. Set flatten_pt=True.")

    # -------------------------------
    # Case 4: No flattening → forbidden
    # -------------------------------
    else:
        raise RuntimeError("SRJ MUST use flatten_pt=True. Set flatten_pt=True in config.")

    # ===============================================
    #  Visualisation: pT / η + 2D distribution plots
    # ===============================================

    path_to_save = config['data']['path_to_save'].format(ln_kT_cut=ln_kT_cut)
    os.makedirs(path_to_save, exist_ok=True)
    print("\nResults will be saved to", path_to_save)

    # -------------------------------
    # Plot pT and η distributions
    # -------------------------------

    # 1. pT distribution
    hist_with_errors(
        pts[labels == 0], label='Background', weights=weights_bkg,
        bins=config['n_bins_pt'], density=True, fmt=".", capsize=2
    )
    hist_with_errors(
        pts[labels == 1], label='Signal', weights=weights_sig,
        bins=config['n_bins_pt'], density=True, fmt="."
    )
    plt.xlabel("LRJ pT [GeV]")
    plt.ylabel("density")
    plt.legend()
    plt.savefig(os.path.join(path_to_save, "pT_distribution.png"))
    plt.close()

    # 2. η distribution
    hist_with_errors(
        etas[labels == 0], label='Background', weights=weights_bkg,
        bins=config['n_bins_eta'], density=True, fmt=".", capsize=2
    )
    hist_with_errors(
        etas[labels == 1], label='Signal', weights=weights_sig,
        bins=config['n_bins_eta'], density=True, fmt="."
    )
    plt.xlabel("LRJ η")
    plt.ylabel("density")
    plt.legend()
    plt.savefig(os.path.join(path_to_save, "eta_distribution.png"))
    plt.close()

    # -------------------------------
    # Plot 2D distributions (pT, η)
    # -------------------------------
    for truth_label, name, w in zip([0, 1], ['background', 'signal'], [weights_bkg, weights_sig]):
        
        H, xedges, yedges = np.histogram2d(
            pts[labels == truth_label],
            etas[labels == truth_label],
            bins=(config['n_bins_pt'], config['n_bins_eta']),
            weights=w, density=True
        )

        # ignore zero bins
        positive_bins = H[H > 0]
        min_bin_count = positive_bins.min() if len(positive_bins) else 0

        print(f"Minimum bin content for {name}: {min_bin_count}")

        plt.hist2d(
            pts[labels == truth_label],
            etas[labels == truth_label],
            bins=(config['n_bins_pt'], config['n_bins_eta']),
            weights=w,
            density=True,
            cmin=min_bin_count
        )
        plt.colorbar(label='density')
        plt.xlabel("LRJ pT [GeV]")
        plt.ylabel("LRJ η")
        plt.savefig(os.path.join(path_to_save, f"pT_eta_distribution_{name}.png"))
        plt.close()


    # rescale the weights so that the total weight of signal jets is equal to the total weight of background jets
    weights_signal_total = weights_sig.sum()
    weights_background_total = weights_bkg.sum()
    print("")
    print("Signal total weight:", weights_signal_total)
    print("Background total weight:", weights_background_total)
    scale_factor = weights_signal_total / weights_background_total
    print("Scale factor:", scale_factor)

    dataset_sig = [jet_graph for jet_graph in dataset if jet_graph.y == 1]
    dataset_bkg = [jet_graph for jet_graph in dataset if jet_graph.y == 0]

    for jet_graph, weight in zip(dataset_sig, weights_sig):
        jet_graph.weights = weight
    for jet_graph, weight in zip(dataset_bkg, weights_bkg):
        jet_graph.weights = weight*scale_factor

    ## define architecture
    batch_size = config['architecture']['batch_size']
    test_size = config['architecture']['test_size']

    dataset= shuffle(dataset_sig+dataset_bkg, random_state=42)
    train_ds, validation_ds = train_test_split(dataset, test_size = test_size, random_state = 144)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=config['num_workers'])
    val_loader = DataLoader(validation_ds, batch_size=batch_size, shuffle=False, num_workers=config['num_workers'])


    print ("train dataset size:", len(train_ds))
    print ("validation dataset size:", len(validation_ds))

    deg = torch.zeros(100, dtype=torch.long)
    for data in dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())


    n_epochs = config['architecture']['n_epochs']
    learning_rate = config['architecture']['learning_rate']
    choose_model = config['architecture']['choose_model']
    save_every_epoch = config['architecture']['save_every_epoch']

    if choose_model == "LundNet":
        model = LundNet()
    if choose_model == "GATNet":
        model = GATNet()
    if choose_model == "GINNet":
        model = GINNet()
    if choose_model == "EdgeGinNet":
        model = EdgeGinNet()
    if choose_model == "PNANet":
        model = PNANet()
    if choose_model == "LundNet_plus_GN2X":
        model = LundNet_plus_GN2X()

    path_to_ckpt = config['retrain']['path_to_ckpt']

    if config['retrain']['flag']:
        path = path_to_ckpt
        model.load_state_dict(torch.load(path))

    if torch.cuda.is_available():
        device_id = 'cuda' if config['gpu'] is None else 'cuda:'+str(config['gpu'])
    else:
        device_id = 'cpu'
    device = torch.device(device_id)
    print(f'\nUsing device: {device}')

    #model = torch.nn.DataParallel(model)
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer_small = torch.optim.Adam(model.parameters(), lr=0.4*learning_rate)
    optimizer2 = torch.optim.Adam(model.parameters(), lr=4*learning_rate)
    optimizer3 = torch.optim.Adam(model.parameters(), lr=10*learning_rate)

    if do_combined_training:
        adv = Adversary_new(config['architecture']['lambda_parameter'], config['architecture']['num_gaussians'])
        adv.to(device)
        optimizer_adv = torch.optim.Adam(adv.parameters(), lr=5*learning_rate)

    train_jds = []
    val_jds = []

    train_bgrej = []
    val_bgrej = []

    model_name = config['data']['model_name'].format(ln_kT_cut=ln_kT_cut)
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []

    timestamp = datetime.now().strftime("%d%m-%H%M")
    metrics_filename = os.path.join(path_to_save, f"losses_{model_name}_{timestamp}.txt")

    for epoch in range(n_epochs):
        train_loss.append(train_clas(train_loader, model, device, optimizer, optimizer2, optimizer3, epoch))
        val_loss.append(my_test(val_loader, model, device))

        print('Epoch: {:03d}, Train Loss: {:.5f}, Val Loss: {:.5f}'.format(epoch, train_loss[epoch], val_loss[epoch]))
        if save_every_epoch or epoch == n_epochs-1:
            model_filename = os.path.join(path_to_save, f"{model_name}_e{epoch+1:03d}_{val_loss[epoch]:.5f}.pt")
            torch.save(model.state_dict(), model_filename)

    metrics = zip(train_loss, val_loss)
    with open(metrics_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Train_Loss", "Val_Loss"])
        writer.writerows(metrics)

    if do_combined_training:
        adv_model_name = config['data']['adv_model_name']
        loss_parameter = config['architecture']['loss_parameter']
        loss_weights = config['architecture']['loss_weights']
        train_loss_clsf = []
        train_loss_adv = []
        train_loss_total = []
        val_loss_clsf = []
        val_loss_adv = []
        val_loss_total = []
        train_acc = []
        val_acc = []

        MASSBINS = np.linspace(40, 300, (300 - 40) // 5 + 1, endpoint=True)
        ############ -------------- adversarial pre-trained ------------- ###############
        n_epochs_adv = config['architecture']['n_epochs_adv']
        for epoch in range(n_epochs_adv):
            ad_lt, clsf_lt, total_lt =  train_adversary_2(train_loader, model, adv, optimizer_adv, device, loss_parameter ,loss_weights) 
            train_loss_adv.append(ad_lt)
            train_loss_clsf.append(clsf_lt)
            train_loss_total.append(total_lt)
            train_acc.append(get_accuracy(train_loader, model, device))

            ad_lv, clsf_lv, total_lv =  test_combined(val_loader, model, adv, device, loss_parameter , loss_weights) 
            val_loss_adv.append(ad_lv)
            val_loss_clsf.append(clsf_lv)
            val_loss_total.append(total_lv)
            val_acc.append(get_accuracy(val_loader, model, device))

            print('Epoch: {:03d}, Train Loss total: {:.5f}, Train Loss adv: {:.5f}, Train Loss clsf: {:.5f}, val_loss_adv: {:.5f}, val_loss_clsf: {:.5f}, val_loss_total: {:.5f},train_acc: {:.5f},val_acc: {:.5f}'.format(epoch, train_loss_total[epoch],train_loss_adv[epoch],train_loss_clsf[epoch], val_loss_adv[epoch], val_loss_clsf[epoch], val_loss_total[epoch],train_acc[epoch],val_acc[epoch]))
            metrics = zip(train_loss_adv, train_loss_clsf, train_loss_total, val_loss_adv, val_loss_clsf, val_loss_total, train_acc, val_acc)
            metrics_filename_adversarial = os.path.join(path_to_save, f"losses_{adv_model_name}_{timestamp}.txt")
            with open(metrics_filename_adversarial, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Train_Loss_adv", "Train_Loss_clsf", "Train_Loss_total", "Val_Loss_Adv", "Val_loss_Class", "val_loss_total", "Train_Acc", "Val_Acc"])
                writer.writerows(metrics)
            if save_every_epoch or epoch == n_epochs_adv-1:
                model_filename_adversarial = os.path.join(path_to_save, f"{model_name}_{adv_model_name}_e{epoch+1:03d}_{val_loss_adv[epoch]:.5f}.pt")
                torch.save(adv.state_dict(), model_filename_adversarial)

        train_loss_clsf = []
        train_loss_adv = []
        train_loss_total = []
        val_loss_clsf = []
        val_loss_adv = []
        val_loss_total = []
        train_acc = []
        val_acc = []
        train_jsdbg = []
        val_jsdbg = []
        n_epochs_common = config['architecture']['n_epochs_common']
        for epoch in range(n_epochs_common):
            print("Epoch:{}".format(epoch))
            if epoch < 12:
                ad_lt, clsf_lt, total_lt =  train_combined_2(train_loader, model, adv, optimizer_small, optimizer_adv, device, loss_parameter,loss_weights)
            else:
                ad_lt, clsf_lt, total_lt =  train_combined_2(train_loader, model, adv, optimizer, optimizer_adv, device, loss_parameter,loss_weights)

            train_loss_clsf.append(clsf_lt)
            train_loss_adv.append(ad_lt)
            train_loss_total.append(total_lt)
            epsilon_bg, jds = aux_metrics(train_loader, model, adv, device, MASSBINS)
            #epsilon_bg, jds = 0,0
            train_jds.append(jds)
            train_bgrej.append(epsilon_bg)
            if jds:
                train_jsdbg.append(epsilon_bg - 1/jds)
            else:
                train_jsdbg.append(0)

            ad_lv, clsf_lv, total_lv =  test_combined(val_loader, model, adv, device, loss_parameter, loss_weights)
            val_loss_adv.append(ad_lv)
            val_loss_clsf.append(clsf_lv)
            val_loss_total.append(total_lv)

            epsilon_bg_test, jds_test = aux_metrics(val_loader, model, adv, device, MASSBINS)
            #epsilon_bg_test, jds_test = 0,0
            val_jds.append(jds_test)
            val_bgrej.append(epsilon_bg_test)
            if jds_test:
                val_jsdbg.append(epsilon_bg_test - 1/jds_test)
            else:
                val_jsdbg.append(0)

            print('Epoch: {:03d}, Train Loss total: {:.5f}, Train Loss adv: {:.5f}, Train Loss clsf: {:.5f}, val_loss_adv: {:.5f}, val_loss_clsf: {:.5f}, val_loss_total: {:.5f},train_jds: {:.5f},val_jds: {:.5f},train_jdsbg: {:.5f},val_jdsbg: {:.5f}'.format(epoch,
                train_loss_total[epoch],train_loss_adv[epoch],train_loss_clsf[epoch], val_loss_adv[epoch], val_loss_clsf[epoch], val_loss_total[epoch], train_jds[epoch], val_jds[epoch],train_jsdbg[epoch],val_jsdbg[epoch]))
            metrics = zip(train_loss_adv, train_loss_clsf, train_loss_total, val_loss_adv, val_loss_clsf, val_loss_total, train_jds, val_jds, train_bgrej, val_bgrej, train_jsdbg, val_jsdbg)
            metrics_filename_comb = os.path.join(path_to_save, f"losses_{model_name}_{adv_model_name}_comb_{timestamp}.txt")
            with open(metrics_filename_comb, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Train_Loss_adv", "Train_Loss_clsf", "Train_Loss_total", "Val_Loss_Adv", "Val_loss_Class", "val_loss_total", "Train_jds", "Val_jds", "Train_bgrej", "Val_bgrej", "Train_jsdbg", "Val_jsdbg"])
                writer.writerows(metrics)
            if save_every_epoch or epoch == n_epochs_common-1:
                model_filename_comb = os.path.join(path_to_save, f"{model_name}_comb_e{epoch+1:03d}_{val_loss_clsf[epoch]:.5f}.pt")
                model_filename_adversarial_comb = os.path.join(path_to_save, f"{model_name}_{adv_model_name}_comb_e{epoch+1:03d}_{val_loss_adv[epoch]:.5f}.pt")
                torch.save(model.state_dict(), model_filename_comb)
                torch.save(adv.state_dict(), model_filename_adversarial_comb)


if __name__ == "__main__":
    main()

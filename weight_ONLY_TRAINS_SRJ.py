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
    path_to_save = config['data']['path_to_save']
    os.makedirs(path_to_save, exist_ok=True)
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

    # rescale the weights so that the total weight of signal jets is equal to the total weight of background jets

    dataset_sig = [g for g in dataset if g.y == 1]
    dataset_bkg = [g for g in dataset if g.y == 0]

    weights_sig_total = sum(g.weights for g in dataset_sig)
    weights_bkg_total = sum(g.weights for g in dataset_bkg)

    scale_factor = weights_sig_total / weights_bkg_total
    print("Scale factor:", scale_factor)

    # multiply scale factor only to background
    for g in dataset_bkg:
        g.weights *= scale_factor

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

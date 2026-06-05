import argparse
import csv
import os
import glob
import torch
import torch.nn.functional as F
from datetime import datetime
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from tools.GNN_model_weight.models import LundNet_General
from tools.GNN_model_weight.utils_newdata import load_yaml, train_clas_general


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

    if mode == "lund_gn3x":
        gn3x_keys = [
            "GN3X_phtautauhad", "GN3X_phbb", "GN3X_phcc", "GN3X_ptop",
            "GN3X_pqcdbb", "GN3X_pqcdbx", "GN3X_pqcdcx", "GN3X_pqcdll", "GN3X_pWqq",
        ]
        for k in gn3x_keys:
            extra_list.append(to_batch_tensor(getattr(data, k, 0.0)))
    elif mode == "lund_b75":
        extra_list.append(to_batch_tensor(getattr(data, "has_reco_b75", 0.0)))
    elif mode == "lund_b50":
        extra_list.append(to_batch_tensor(getattr(data, "has_reco_b50", 0.0)))
    elif mode == "lund_part":
        extra_list.append(to_batch_tensor(getattr(data, "TopTrans_score", 0.0)))

    return torch.cat(extra_list, dim=1).float()


@torch.no_grad()
def validate(loader, model, device, mode):
    model.eval()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        extra = get_extra_features(data, mode)
        out = model(data, extra)
        loss = F.binary_cross_entropy(
            out, data.y.float().view(-1, 1), weight=data.weights.float().view(-1, 1)
        )
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser(description="LRJ LundNet trainer")
    parser.add_argument("config", help="Path to config YAML")
    args = parser.parse_args()
    config = load_yaml(args.config)

    gpu_id = config.get("gpu", 0)
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    mode = config["architecture"]["training_mode"]
    path_to_save = config["data"]["path_to_save"]
    os.makedirs(path_to_save, exist_ok=True)

    schedule = config["architecture"].get("optimizer_schedule", [8, 16])
    boundaries = tuple(schedule)

    print(f"--- [{datetime.now().strftime('%H:%M:%S')}] Loading dataset ---", flush=True)

    all_shards = sorted(glob.glob(config["data"]["path_to_trainfiles"]))
    dataset = []

    for i, shard_path in enumerate(all_shards):
        shards = torch.load(shard_path, weights_only=False)
        for g in shards:
            g.weights = torch.tensor(float(g.weights))
        dataset.extend(shards)
        del shards
        if i % 40 == 0:
            print(f"  > Loaded {i}/{len(all_shards)} shards...", flush=True)

    dataset_sig = [g for g in dataset if g.y == 1]
    dataset_bkg = [g for g in dataset if g.y == 0]
    weights_sig_total = sum(g.weights for g in dataset_sig)
    weights_bkg_total = sum(g.weights for g in dataset_bkg)

    if weights_bkg_total == 0:
        print("Error: Background total weight is 0. Check data.", flush=True)
        return

    scale_factor = weights_sig_total / weights_bkg_total
    print(f">>> Scale factor (Sig/Bkg total weight ratio): {scale_factor:.6f}", flush=True)
    for g in dataset_bkg:
        g.weights *= scale_factor

    print(f"--- [{datetime.now().strftime('%H:%M:%S')}] Shuffling & splitting ---", flush=True)
    dataset = shuffle(dataset_sig + dataset_bkg, random_state=42)
    train_ds, val_ds = train_test_split(
        dataset, test_size=config["architecture"]["test_size"], random_state=144
    )

    num_workers = config.get("num_workers", 0)
    train_loader = DataLoader(
        train_ds,
        batch_size=config["architecture"]["batch_size"],
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["architecture"]["batch_size"],
        shuffle=False,
        num_workers=num_workers,
    )
    print(f">>> Train: {len(train_ds)} | Val: {len(val_ds)}", flush=True)

    model = LundNet_General(extra_dim=config["architecture"]["extra_dim"]).to(device)

    lr = config["architecture"]["learning_rate"]
    opts = [
        torch.optim.Adam(model.parameters(), lr=lr),
        torch.optim.Adam(model.parameters(), lr=4.0 * lr),
        torch.optim.Adam(model.parameters(), lr=10.0 * lr),
    ]

    def forward_fn(m, data):
        return m(data, get_extra_features(data, mode))

    n_epochs = config["architecture"]["n_epochs"]
    patience = config["architecture"].get("patience", 8)
    best_val_loss = float("inf")
    epochs_no_improve = 0
    metrics_log = []

    print(
        f"\n>>> Training | mode={mode} | gpu={gpu_id} | schedule={boundaries}",
        flush=True,
    )
    for epoch in range(n_epochs):
        t_loss = train_clas_general(
            train_loader, model, device, *opts, epoch, forward_fn, boundaries=boundaries
        )
        v_loss = validate(val_loader, model, device, mode)

        print(
            f"Epoch {epoch + 1:03d}/{n_epochs} | Train Loss: {t_loss:.5f} | Val Loss: {v_loss:.5f}",
            flush=True,
        )
        metrics_log.append([epoch + 1, t_loss, v_loss])

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(path_to_save, f"best_model_{mode}.pt"))
            print("  [Save] New best model!", flush=True)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch + 1}", flush=True)
                break

    with open(os.path.join(path_to_save, f"losses_{mode}.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])
        writer.writerows(metrics_log)
    print(f"\n>>> Done. Path: {path_to_save}", flush=True)


if __name__ == "__main__":
    main()

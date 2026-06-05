import argparse
import csv
import gc
import os
import glob
import random
import torch
import torch.nn.functional as F
from datetime import datetime
from torch_geometric.loader import DataLoader
from sklearn.utils import shuffle as sk_shuffle

from tools.GNN_model_weight.models import LundNet_General
from tools.GNN_model_weight.utils_newdata import load_yaml


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

def compute_scale_factor(shard_paths, filter_attr=None):
    """Streaming pass: compute sig/bkg weight ratio without keeping graphs in RAM."""
    w_sig, w_bkg = 0.0, 0.0
    for p in shard_paths:
        shards = torch.load(p, weights_only=False)
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

@torch.no_grad()
def validate(loader, model, device, mode):
    model.eval()
    total_loss, total_n = 0.0, 0
    for data in loader:
        data = data.to(device)
        extra = get_extra_features(data, mode)
        out = model(data, extra)
        loss = F.binary_cross_entropy(
            out, data.y.float().view(-1, 1), weight=data.weights.float().view(-1, 1)
        )
        total_loss += loss.item() * data.num_graphs
        total_n    += data.num_graphs
    return total_loss / total_n if total_n > 0 else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LRJ LundNet trainer — streaming shard loading")
    parser.add_argument("config", help="Path to config YAML")
    args = parser.parse_args()
    config = load_yaml(args.config)

    gpu_id     = config.get("gpu", 0)
    device     = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    mode       = config["architecture"]["training_mode"]
    path_to_save = config["data"]["path_to_save"]
    os.makedirs(path_to_save, exist_ok=True)

    batch_size       = config["architecture"]["batch_size"]
    schedule         = config["architecture"].get("optimizer_schedule", [8, 16])
    boundaries       = tuple(schedule)
    # shards_per_window: tune down if RAM is tight, up for better shuffle coverage
    shards_per_window = config.get("shards_per_window", 20)
    val_frac         = config["architecture"].get("test_size", 0.1)
    num_workers      = config.get("num_workers", 0)

    # -----------------------------------------------------------------------
    # Shard discovery + deterministic train/val split
    # n_parts mode  (new):  use preproc_dir + n_parts to discover shards from
    #                       train_part{0..n_parts-1}/ subdirectories.
    # legacy mode   (old):  path_to_trainfiles glob (backward compatible).
    # -----------------------------------------------------------------------
    filter_attr = config.get("filter_attr", None)   # e.g. "has_reco_b_WP77"

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

    n_val        = max(1, int(len(shards_shuffled) * val_frac))
    val_shards   = shards_shuffled[:n_val]
    train_shards = shards_shuffled[n_val:]

    print(
        f"Shards — total: {len(all_shards)} | n_parts: {n_parts} "
        f"| train: {len(train_shards)} | val: {len(val_shards)} "
        f"| filter_attr: {filter_attr}",
        flush=True,
    )

    # -----------------------------------------------------------------------
    # Pass 1 (streaming): sig/bkg scale factor — no graphs kept in RAM
    # -----------------------------------------------------------------------
    print(f"--- [{datetime.now().strftime('%H:%M:%S')}] Computing scale factor (streaming)... ---", flush=True)
    scale_factor = compute_scale_factor(train_shards, filter_attr=filter_attr)
    print(f">>> Scale factor (sig/bkg weight ratio): {scale_factor:.6f}", flush=True)

    # -----------------------------------------------------------------------
    # Fixed validation set (loaded once, stays in RAM)
    # -----------------------------------------------------------------------
    print(f"--- [{datetime.now().strftime('%H:%M:%S')}] Loading val set ({len(val_shards)} shards) ---", flush=True)
    val_data   = load_shards(val_shards, scale_factor, filter_attr=filter_attr)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    print(f">>> Val jets: {len(val_data)}", flush=True)

    # -----------------------------------------------------------------------
    # Model & optimizers
    # -----------------------------------------------------------------------
    model = LundNet_General(extra_dim=config["architecture"]["extra_dim"]).to(device)
    lr    = config["architecture"]["learning_rate"]
    opts  = [
        torch.optim.Adam(model.parameters(), lr=lr),
        torch.optim.Adam(model.parameters(), lr=4.0 * lr),
        torch.optim.Adam(model.parameters(), lr=10.0 * lr),
    ]

    n_epochs         = config["architecture"]["n_epochs"]
    patience         = config["architecture"].get("patience", 8)
    best_val_loss    = float("inf")
    epochs_no_improve = 0
    metrics_log      = []

    print(
        f"\n>>> Training | mode={mode} | gpu={gpu_id} | "
        f"schedule={boundaries} | shards_per_window={shards_per_window}",
        flush=True,
    )

    for epoch in range(n_epochs):
        # LR schedule: same 3-phase scheme as before
        if epoch < boundaries[0]:
            opt = opts[0]
        elif epoch < boundaries[1]:
            opt = opts[1]
        else:
            opt = opts[2]

        # Shuffle shard order every epoch for cross-shard diversity
        epoch_shards = list(train_shards)
        random.shuffle(epoch_shards)

        model.train()
        epoch_loss, epoch_n = 0.0, 0

        for win_start in range(0, len(epoch_shards), shards_per_window):
            window      = epoch_shards[win_start : win_start + shards_per_window]
            window_data = load_shards(window, scale_factor, filter_attr=filter_attr)
            window_data = sk_shuffle(window_data, random_state=epoch * 10000 + win_start)
            loader      = DataLoader(
                window_data, batch_size=batch_size, shuffle=True, num_workers=num_workers
            )

            for data in loader:
                data = data.to(device)
                opt.zero_grad()
                extra = get_extra_features(data, mode)
                out   = model(data, extra)
                loss  = F.binary_cross_entropy(
                    out, data.y.float().view(-1, 1), weight=data.weights.float().view(-1, 1)
                )
                loss.backward()
                opt.step()
                epoch_loss += loss.item() * data.num_graphs
                epoch_n    += data.num_graphs

            del window_data, loader
            gc.collect()

        t_loss = epoch_loss / epoch_n if epoch_n > 0 else 0.0
        v_loss = validate(val_loader, model, device, mode)

        print(
            f"Epoch {epoch + 1:03d}/{n_epochs} | Train Loss: {t_loss:.5f} | Val Loss: {v_loss:.5f}",
            flush=True,
        )
        metrics_log.append([epoch + 1, t_loss, v_loss])

        if v_loss < best_val_loss:
            best_val_loss     = v_loss
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

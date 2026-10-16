#!/usr/bin/env python
"""Score pubnote test_LRJ_* shards with many trained models in one pass and
write one column per model into a separate scores h5.

Usage:
  python tools/score_h5_multi.py \\
    --test-dir     /share/lustre/yzhu/pubnote/test_LRJ_W_801859_20M \\
    --test-meanstd /share/lustre/yzhu/pubnote/preprocessed_LRJ_W_801859_20M/train/meanstd.json \\
    --out          /share/lustre/yzhu/IOP_DATA/v3_pub/scores_W.h5 \\
    --configs      configs/v3/train_W_v3_*.yaml \\
                   pubnote=configs/pubnote/train_LRJ_W_801859_20M_lund_only.yaml

Per-jet columns (pt, mass, y, b/c flags, transformer scores) are read from the
shard graphs themselves, so rows are in shard order and do not rely on
cut_jets.h5 (which is written in pre-shuffle order by h5_to_lund_shards.py).

Each training yaml gives the checkpoint (path_to_save/best_model_<tag>.pt),
architecture, extra features and the training meanstd
(<preproc_dir>/train/meanstd.json). When the training meanstd differs from the
test-shard meanstd (md models trained on 802017), node features and Ntrk are
mapped back to raw values and re-standardised with the training meanstd.

Column name: score_<name>; name is given as NAME=cfg.yaml or derived from the
config basename minus "train_<ch>_v3_" and ".yaml".
"""
import argparse
import glob
import json
import os
import re
import sys

import h5py
import numpy as np
import torch
import yaml
from torch_geometric.loader import DataLoader

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from tools.GNN_model_weight.models import LundNet_General, LundFormer
from tools.score_h5_shards import get_extra

META = ["pt", "mass", "y", "WTrans_score", "WTrans_massdec", "TopTrans_score",
        "has_reco_b_WP65", "has_reco_b_WP70", "has_reco_b_WP77",
        "has_reco_b_WP85", "has_reco_b_WP90",
        "has_reco_c_WP10", "has_reco_c_WP30", "has_reco_c_WP50"]


def ckpt_name(arch):
    mode = arch["training_mode"]
    if arch.get("do_combined_training"):
        return f"best_model_{mode}_ann_lam{float(arch['lambda_parameter']):g}.pt"
    if arch.get("do_disco_training"):
        return f"best_model_{mode}_disco_lam{float(arch['disco_lambda']):g}.pt"
    return f"best_model_{mode}.pt"


def load_meanstd(path):
    with open(path) as f:
        d = json.load(f)
    return (torch.tensor(d["mean_x"], dtype=torch.float32),
            torch.tensor(d["std_x"], dtype=torch.float32),
            float(d["mean_ntrk"]), float(d["std_ntrk"]))


class Scorer:
    def __init__(self, spec, test_ms, device):
        if "=" in spec:
            self.name, cfg_path = spec.split("=", 1)
        else:
            cfg_path = spec
            self.name = re.sub(r"^train_[A-Za-z]+_v3_", "",
                               os.path.basename(cfg_path)).replace(".yaml", "")
        cfg = yaml.safe_load(open(cfg_path))
        arch = cfg["architecture"]
        self.mode = arch["training_mode"]
        self.ckpt = os.path.join(cfg["data"]["path_to_save"], ckpt_name(arch))
        tag = ckpt_name(arch)[len("best_model_"):-len(".pt")]
        self.done = os.path.join(cfg["data"]["path_to_save"], f"done_{tag}.flag")
        extra_dim = int(arch.get("extra_dim", 1))
        if arch.get("choose_model", "LundNet_General") == "LundFormer":
            self.model = LundFormer(
                extra_dim=extra_dim,
                d_model=arch.get("d_model", 96), n_heads=arch.get("n_heads", 8),
                n_layers=arch.get("n_layers_former", 4), d_ff=arch.get("d_ff", 288),
                dropout=arch.get("dropout", 0.1))
            self.batch_size = 1024
        else:
            self.model = LundNet_General(extra_dim=extra_dim)
            self.batch_size = 2048
        self.cls = type(self.model).__name__
        self.device = device

        self.ms_path = os.path.join(cfg["data"]["preproc_dir"], "train", "meanstd.json")
        train_ms = load_meanstd(self.ms_path)
        self.remap = None
        if not all(torch.allclose(torch.as_tensor(a), torch.as_tensor(b))
                   for a, b in zip(train_ms, test_ms)):
            mx_t, sx_t, mn_t, sn_t = test_ms
            mx_m, sx_m, mn_m, sn_m = train_ms
            # z_model = z_test * a + b
            self.remap = ((sx_t / sx_m).to(device), ((mx_t - mx_m) / sx_m).to(device),
                          sn_t / sn_m, (mn_t - mn_m) / sn_m)

    def load(self):
        sd = torch.load(self.ckpt, map_location="cpu", weights_only=True)
        self.model.load_state_dict(sd)
        self.model.to(self.device).eval()

    @torch.no_grad()
    def score(self, dataset):
        out = []
        for batch in DataLoader(dataset, batch_size=self.batch_size, shuffle=False):
            batch = batch.to(self.device)
            if self.remap is not None:
                ax, bx, an, bn = self.remap
                batch.x = batch.x * ax + bx
                batch.Ntrk = batch.Ntrk * an + bn
            extra = get_extra(batch, self.mode, self.device)
            out.append(self.model(batch, extra).float().cpu().numpy().ravel())
        return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--test-dir", required=True)
    ap.add_argument("--test-meanstd", required=True,
                    help="meanstd.json the test shards were standardised with")
    ap.add_argument("--out", required=True)
    ap.add_argument("--configs", nargs="+", required=True,
                    help="training yamls, optionally NAME=path.yaml")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-score models whose column already exists")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    shards = sorted(glob.glob(os.path.join(args.test_dir, "processed_graphs_MD_batch*.pt")),
                    key=lambda p: int(re.search(r"batch(\d+)_", p).group(1)))
    if not shards:
        raise FileNotFoundError(f"No shards in {args.test_dir}")

    existing, n_jets = {}, None
    if os.path.exists(args.out):
        with h5py.File(args.out, "r") as hf:
            existing = {k: hf[k].attrs.get("ckpt_mtime") for k in hf.keys()}
            n_jets = hf["pt"].shape[0]
            if hf.attrs.get("test_dir") != args.test_dir:
                raise ValueError(f"{args.out} was built from {hf.attrs.get('test_dir')}")
    need_meta = n_jets is None

    test_ms = load_meanstd(args.test_meanstd)
    scorers = []
    for c in args.configs:
        s = Scorer(c, test_ms, device)
        if not (os.path.exists(s.ckpt) and os.path.exists(s.done)):
            print(f"  skip {s.name}: not finished ({s.done})", flush=True)
            continue
        col = f"score_{s.name}"
        if (col in existing and not args.overwrite
                and existing[col] == os.path.getmtime(s.ckpt)):
            print(f"  skip {s.name}: column up to date", flush=True)
            continue
        s.load()
        print(f"  {s.name:28s} {s.cls:16s} {s.mode:11s} remap={s.remap is not None} "
              f"ckpt={s.ckpt}", flush=True)
        scorers.append(s)
    if not scorers and not need_meta:
        print(">>> Nothing to score.")
        return

    print(f">>> {len(shards)} shards (numeric order), {len(scorers)} models", flush=True)
    scores = {s.name: [] for s in scorers}
    meta = {k: [] for k in META}
    for i, sf in enumerate(shards):
        dataset = torch.load(sf, weights_only=False)
        if need_meta:
            for k in META:
                meta[k].append(np.array([float(g[k]) for g in dataset]))
        for s in scorers:
            scores[s.name].append(s.score(dataset))
        if i % 50 == 0:
            print(f"    shard {i}/{len(shards)}", flush=True)
        del dataset

    with h5py.File(args.out, "a") as hf:
        if need_meta:
            for k in META:
                hf.create_dataset(k, data=np.concatenate(meta[k]))
            hf.attrs["test_dir"] = args.test_dir
            hf.attrs["test_meanstd"] = args.test_meanstd
            hf.attrs["row_order"] = "test shards in numeric batch order, graph order within shard"
            n_jets = hf["pt"].shape[0]
            print(f">>> wrote per-jet columns for {n_jets} jets", flush=True)
        for s in scorers:
            arr = np.concatenate(scores[s.name]).astype(np.float32)
            if len(arr) != n_jets:
                raise ValueError(f"{s.name}: {len(arr)} scores vs {n_jets} jets")
            col = f"score_{s.name}"
            if col in hf:
                del hf[col]
            ds = hf.create_dataset(col, data=arr)
            ds.attrs["ckpt"] = s.ckpt
            ds.attrs["ckpt_mtime"] = os.path.getmtime(s.ckpt)
            ds.attrs["train_meanstd"] = s.ms_path
            print(f"  wrote {col}: mean={arr.mean():.4f}", flush=True)
    print(f">>> Done: {args.out}", flush=True)


if __name__ == "__main__":
    main()

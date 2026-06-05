import argparse
import os
import glob
import sys
import uproot
import numpy as np
import torch
from torch_geometric.loader import DataLoader

from tools.GNN_model_weight.models import LundNet_General 
from tools.GNN_model_weight.utils_newdata import load_yaml

def sprint(*args, **kwargs):
    print(*args, **kwargs); sys.stdout.flush()

def calculate_gn3x_standalone(batch):
    """根据优化系数 fi 手动计算 GN3X 判别式 D"""
    f = {
        "pWqq": 0.199979, 
        "phbb": 0.035299, 
        "phcc": 0.083998,
        "phtautauhad": 0.079448, 
        "pqcdbb": 0.009451, 
        "pqcdbx": 0.230861,
        "pqcdcx": 0.098290, 
        "pqcdll": 0.262674
    }

    # 验证：sum(f.values()) = 1.000000
    denom = (
        f["pWqq"] * batch.GN3X_pWqq + f["phbb"] * batch.GN3X_phbb +
        f["phcc"] * batch.GN3X_phcc + f["phtautauhad"] * batch.GN3X_phtautauhad +
        f["pqcdbb"] * batch.GN3X_pqcdbb + f["pqcdbx"] * batch.GN3X_pqcdbx +
        f["pqcdcx"] * batch.GN3X_pqcdcx + f["pqcdll"] * batch.GN3X_pqcdll
    )
    # D = ln( p_top / denom )
    d_gn3x = torch.log(batch.GN3X_ptop / torch.clamp(denom, min=1e-12))
    return d_gn3x.cpu().numpy().flatten()

def build_extra_x(batch, tag):
    """Build extra input features for LundNet inference.

    Mapping:
        only / fb65..fb90 / fc10..fc50  → Ntrk only  (filtered models use Ntrk only at inference)
        toptrans / part (legacy alias)  → Ntrk + TopTrans_score
        wtrans                          → Ntrk + WTrans_score
        bWP65..bWP90                    → Ntrk + has_reco_b_WP{xx}
        cWP10..cWP50                    → Ntrk + has_reco_c_WP{xx}
        gn3x                            → Ntrk + 9 GN3X probs
    """
    ntrk = batch.Ntrk.view(-1, 1).float()

    if tag in ("only",
               "fb65", "fb70", "fb77", "fb85", "fb90",
               "fc10", "fc30", "fc50"):
        return ntrk
    elif tag in ("toptrans", "part"):
        return torch.cat([ntrk, batch.TopTrans_score.view(-1, 1).float()], dim=1)
    elif tag == "wtrans":
        return torch.cat([ntrk, batch.WTrans_score.view(-1, 1).float()], dim=1)
    elif tag == "bWP65":
        return torch.cat([ntrk, batch.has_reco_b_WP65.view(-1, 1).float()], dim=1)
    elif tag == "bWP70":
        return torch.cat([ntrk, batch.has_reco_b_WP70.view(-1, 1).float()], dim=1)
    elif tag == "bWP77":
        return torch.cat([ntrk, batch.has_reco_b_WP77.view(-1, 1).float()], dim=1)
    elif tag == "bWP85":
        return torch.cat([ntrk, batch.has_reco_b_WP85.view(-1, 1).float()], dim=1)
    elif tag == "bWP90":
        return torch.cat([ntrk, batch.has_reco_b_WP90.view(-1, 1).float()], dim=1)
    elif tag == "cWP10":
        return torch.cat([ntrk, batch.has_reco_c_WP10.view(-1, 1).float()], dim=1)
    elif tag == "cWP30":
        return torch.cat([ntrk, batch.has_reco_c_WP30.view(-1, 1).float()], dim=1)
    elif tag == "cWP50":
        return torch.cat([ntrk, batch.has_reco_c_WP50.view(-1, 1).float()], dim=1)
    elif tag == "gn3x":
        gn3x_keys = [
            "GN3X_ptop", "GN3X_pWqq", "GN3X_phbb", "GN3X_phcc",
            "GN3X_phtautauhad", "GN3X_pqcdbb", "GN3X_pqcdbx",
            "GN3X_pqcdcx", "GN3X_pqcdll",
        ]
        gn3x_feats = [getattr(batch, k).view(-1, 1).float() for k in gn3x_keys]
        return torch.cat([ntrk] + gn3x_feats, dim=1)
    return ntrk

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help="LRJ scoring yaml")
    args = parser.parse_args()
    config = load_yaml(args.config)

    files_root = sorted(glob.glob(config['data']['paths_to_test_file_root'][0]))
    files_graphs = sorted(glob.glob(config['data']['paths_to_test_file_graphs'][0]))
    
    outdir = config['data']['path_to_outdir']
    os.makedirs(outdir, exist_ok=True)
    tree_name = "FlatSubstructureJetTree"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for fr, fg in zip(files_root, files_graphs):
        sprint(f"\n>>> Processing Shard: {os.path.basename(fr)}")
        dataset = torch.load(fg, weights_only=False)
        loader = DataLoader(dataset, batch_size=config['test']['batch_size'], shuffle=False)
        
        with uproot.open(fr) as f_in:
            arrays = f_in[tree_name].arrays()

        # --- 1. 计算 Standalone 结果 ---
        sprint("  -> Calculating standalone GN3X and ParT scores...")
        gn3x_standalone = []
        part_standalone = []
        for batch in loader:
            # GN3X Alone: 根据 fi 计算
            gn3x_standalone.append(calculate_gn3x_standalone(batch))
            # ParT Alone: 直接取数据里的 TopTrans_score
            part_standalone.append(batch.TopTrans_score.cpu().numpy().flatten())
            
        arrays["LRJ_GN3X_alone_score"] = np.concatenate(gn3x_standalone)
        arrays["LRJ_ParT_alone_score"] = np.concatenate(part_standalone)

        # --- 2. 运行 LundNet 模型推理 ---
        for m in config["test"]["models_to_run"]:
            tag, dim, ckpt = m["tag"], m["extra_dim"], m["ckpt"]
            sprint(f"  -> Inferencing model: {tag}")
            
            model = LundNet_General(extra_dim=dim)
            model.load_state_dict(torch.load(ckpt, map_location=device))
            model.to(device).eval()

            y_pred_list = []
            with torch.no_grad():
                for batch in loader:
                    batch = batch.to(device)
                    extra_x = build_extra_x(batch, tag)
                    out = model(batch, extra_x)
                    y_pred_list.append(out.cpu().numpy())
            
            branch_name = config["test"]["scores_branch_name"].format(tag=tag)
            arrays[branch_name] = np.concatenate(y_pred_list).flatten()
            del model; torch.cuda.empty_cache()

        # 保存结果
        out_name = os.path.join(outdir, os.path.basename(fr).replace(".root", "_scored.root"))
        with uproot.recreate(out_name) as f_out:
            f_out[tree_name] = arrays
        sprint(f"  Saved to: {os.path.basename(out_name)}")

if __name__ == "__main__":
    main()
#!/usr/bin/env python
"""Generate configs/v3/train_{Top,W}_v3_<model>.yaml.

v3 = lundtag_compare retrain on the pubnote-converted H5 shards, one recipe
for all models: lr 1e-4, grad_clip 5, schedule [8, 16], 35 epochs, no early
stop. Baselines/md pick min val BCE (all val jets, p99.9 weight clip);
ANN/DisCo keep the last epoch.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PUB = "/share/lustre/yzhu/pubnote"
OUT = "/share/lustre/yzhu/IOP_DATA/v3_pub"

CHANNELS = {
    "Top": dict(data="Top_20M", flav="bWP77",
                decor_mass_range=[40, 600], mass_bins=[40, 600, 10]),
    "W":   dict(data="W_801859_20M", flav="cWP50",
                decor_mass_range=[40, 300], mass_bins=[40, 300, 5]),
}

LUNDNET = dict(choose_model="LundNet_General", batch_size=1024)
FORMER = dict(choose_model="LundFormer", d_model=96, n_heads=8,
              n_layers_former=4, d_ff=288, batch_size=512)


def models(ch, flav):
    m = {
        "lund_only":         (LUNDNET, "lund_only", {}),
        "lund_former":       (FORMER,  "lund_only", {}),
        f"lund_{flav}":        (LUNDNET, f"lund_{flav}", {}),
        f"lund_former_{flav}": (FORMER,  f"lund_{flav}", {}),
        "lund_ann_lam25":    (LUNDNET, "lund_only", dict(do_combined_training=True, lambda_parameter=25)),
        "lund_former_ann_lam25": (FORMER, "lund_only", dict(do_combined_training=True, lambda_parameter=25)),
        "lund_former_disco_lam10":  (FORMER, "lund_only", dict(do_disco_training=True, disco_lambda=10)),
        "lund_former_disco_lam100": (FORMER, "lund_only", dict(do_disco_training=True, disco_lambda=100)),
    }
    if ch == "W":
        for lam in (5, 100):
            m[f"lund_ann_lam{lam}"] = (LUNDNET, "lund_only",
                                       dict(do_combined_training=True, lambda_parameter=lam))
        m["lund_md"] = (LUNDNET, "lund_only", dict(_data="W_802017_20M"))
        m["lund_former_md"] = (FORMER, "lund_only", dict(_data="W_802017_20M"))
    return m


def fmt(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return f'"{v}"'
    if isinstance(v, list):
        return "[" + ", ".join(str(x) for x in v) + "]"
    return str(v)


def write(ch, name, net, mode, extra, cc):
    data = extra.pop("_data", cc["data"])
    arch = dict(net)
    arch.update(
        training_mode=mode,
        extra_dim=2 if mode != "lund_only" else 1,
        learning_rate=0.0001,
        n_epochs=35,
        patience=8,
        early_stop=False,
        grad_clip=5.0,
        test_size=0.05,
        optimizer_schedule=[8, 16],
    )
    if extra.get("do_combined_training"):
        arch.update(num_gaussians=20, n_epochs_adv=10, n_epochs_common=20,
                    loss_parameter=1, loss_weights=[1, 1])
    if extra.get("do_disco_training"):
        arch.update(n_epochs_common=20)
    arch.update(extra)
    arch.update(decor_mass_range=cc["decor_mass_range"], mass_bins=cc["mass_bins"],
                monitor_wp50=True)
    choose = arch.pop("choose_model")
    lines = [
        "data:",
        f'  preproc_dir:  "{PUB}/preprocessed_LRJ_{data}"',
        f'  val_dir:      "{PUB}/val_LRJ_{data}"',
        f'  path_to_save: "{OUT}/models_{ch}/exp_{name}/nparts1"',
        "architecture:",
        f'  choose_model: "{choose}"',
    ]
    lines += [f"  {k}: {fmt(v)}" for k, v in arch.items()]
    lines += ["n_parts: 1", "shards_per_window: 40", "gpu: 0",
              "num_workers: 4", "scale_factor_workers: 8", ""]
    path = os.path.join(HERE, f"train_{ch}_v3_{name}.yaml")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path


if __name__ == "__main__":
    for ch, cc in CHANNELS.items():
        for name, (net, mode, extra) in models(ch, cc["flav"]).items():
            print(write(ch, name, net, mode, dict(extra), cc))

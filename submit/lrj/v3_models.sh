# Shared v3 model table, sourced by 08_submit_v3.sh and 09_gate_v3.sh.

REPO=/home/yuanda/lundtoptagger
CFG=$REPO/configs/v3
LOG=/share/lustre/yzhu/IOP_DATA/logs_lrj
V3=/share/lustre/yzhu/IOP_DATA/v3_pub
PUB=/share/lustre/yzhu/pubnote
MEM=120G
CPUS=5   # num_workers 4 + main process

# baselines + md: "ch exp flagtag jobname pubnote_model_dir(scale_factor source)"
BASE=(
  "Top lund_only         lund_only  v3_only_top          $PUB/models_LRJ_Top_20M"
  "Top lund_former       lund_only  v3_former_top        $PUB/models_LRJ_Top_20M"
  "Top lund_bWP77        lund_bWP77 v3_bWP77_top         $PUB/models_LRJ_Top_20M"
  "Top lund_former_bWP77 lund_bWP77 v3_former_bWP77_top  $PUB/models_LRJ_Top_20M"
  "W   lund_only         lund_only  v3_only_W            $PUB/models_LRJ_W_801859_20M"
  "W   lund_former       lund_only  v3_former_W          $PUB/models_LRJ_W_801859_20M"
  "W   lund_cWP50        lund_cWP50 v3_cWP50_W           $PUB/models_LRJ_W_801859_20M"
  "W   lund_former_cWP50 lund_cWP50 v3_former_cWP50_W    $PUB/models_LRJ_W_801859_20M"
  "W   lund_md           lund_only  v3_md_W              $PUB/models_LRJ_W_802017_20M"
  "W   lund_former_md    lund_only  v3_former_md_W       $PUB/models_LRJ_W_802017_20M"
)

# downstream: "ch base_exp exp flagtag jobname worker"
DOWN=(
  "W   lund_only   lund_ann_lam5               lund_only_ann_lam5      v3_ann_lam5_W              03b_train_ann_worker.sh"
  "W   lund_only   lund_ann_lam25              lund_only_ann_lam25     v3_ann_lam25_W             03b_train_ann_worker.sh"
  "W   lund_only   lund_ann_lam100             lund_only_ann_lam100    v3_ann_lam100_W            03b_train_ann_worker.sh"
  "W   lund_former lund_former_ann_lam25       lund_only_ann_lam25     v3_former_ann_lam25_W      03b_train_ann_worker.sh"
  "W   lund_former lund_former_disco_lam10     lund_only_disco_lam10   v3_former_disco_lam10_W    03c_train_disco_worker.sh"
  "W   lund_former lund_former_disco_lam100    lund_only_disco_lam100  v3_former_disco_lam100_W   03c_train_disco_worker.sh"
  "Top lund_only   lund_ann_lam25              lund_only_ann_lam25     v3_ann_lam25_top           03b_train_ann_worker.sh"
  "Top lund_former lund_former_ann_lam25       lund_only_ann_lam25     v3_former_ann_lam25_top    03b_train_ann_worker.sh"
  "Top lund_former lund_former_disco_lam10     lund_only_disco_lam10   v3_former_disco_lam10_top  03c_train_disco_worker.sh"
  "Top lund_former lund_former_disco_lam100    lund_only_disco_lam100  v3_former_disco_lam100_top 03c_train_disco_worker.sh"
)

exp_dir() { echo "$V3/models_$1/exp_$2/nparts1"; }   # ch exp

# conf jobname worker [seed_dir]
v3_sub() {
    sbatch --parsable --job-name="$2" --constraint='a100|l40s|v100' --mem=$MEM --cpus-per-task=$CPUS \
        --export=ALL,CONF="$CFG/$1",SEED_DIR="${4:-}" \
        --output="$LOG/slurm-%j.$2.out" "$REPO/submit/lrj/$3"
}

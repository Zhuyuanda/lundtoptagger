# LRJ on PanDA (hypatia / lxplus)

## Split

| Job | `event_fraction_idx` | Events | Output |
|-----|----------------------|--------|--------|
| 0 | 0 | 70% | `output_part0/` → train |
| 1 | 1 | 30% | `output_part1/` → test |

Train/val: `test_size: 0.1` inside `weight_ONLY_TRAINS_LRJ.py` on the 70% pool.

## Input

Edit [`configs/rucio_datasets_lrj.yaml`](../../configs/rucio_datasets_lrj.yaml) (Rucio container names only).
`grid/panda_inDS.py` maps them to PanDA `--inDS` as `user.yuanda.<name>/` (dots, not `user.yuanda:` colons).

## Step 1: Make_data on PanDA (hypatia)

```bash
conda deactivate
setupATLAS && lsetup panda
voms-proxy-init -voms atlas
cd /path/to/lundtoptagger
bash grid/verify_panda_submit.sh
bash submit/panda/submit_make_data.sh
```

Each job writes output locally on the worker, tars it as `output.tar.gz`, and PanDA uploads it to the outDS.
Monitor at https://bigpanda.cern.ch/

W campaign:
```bash
SIGNAL=W DATA_ID=WTagging_LRJ bash submit/panda/submit_make_data.sh
```

## Step 2: Download make_data outputs (UCL or lxplus)

```bash
# Replace TIMESTAMP with the actual suffix from submission
rucio download user.yuanda.TopTagging_LRJ.make_data.TIMESTAMP/

mkdir -p make_data_output
# PanDA names outputs like output._0001.tar.gz, output._0002.tar.gz
for f in user.yuanda.TopTagging_LRJ.make_data.TIMESTAMP/output.*.tar.gz; do
  tar xzf "$f" -C make_data_output/
done
# Result: make_data_output/output_part0/  and  make_data_output/output_part1/
```

## Step 3: Preprocess (UCL)

```bash
cd /path/to/lundtoptagger
DATA_DIR=./make_data_output bash submit/panda/run_preprocess.sh
# Output: make_data_output/preprocessed/
```

## Step 4: Train (UCL)

```bash
python weight_ONLY_TRAINS_LRJ.py configs/config_ONLY_TRAIN_LRJ.yaml
```

## Tips

- `outDS` gets a timestamp suffix by default (avoids PanDA "broken state" on reuse).
  Force fixed name only after previous task is done/failed/killed:
  `USE_FIXED_OUTDS=1 bash submit/panda/submit_make_data.sh`
- OOM: `max_jets_per_batch: 5000`, `entry_chunk_events: 30000` in `config_make_data_LRJ_panda.yaml`
- Code upload: `--noBuild --inTarBall lundtoptagger_panda.tar.gz`
- Default `--nCore 1`. Override: `PANDA_NCORE=8`
- If OOM: `PANDA_MEMORY_MB=32000 bash submit/panda/submit_make_data.sh`
- Preprocess: `streaming_flatten: True`

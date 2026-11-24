#!/bin/bash
# Download the products of round 37 (the dry rebuild, after_r37.sh) into the repo's data/ (run from
# the repo root on the laptop; ~0.3 GB over a slow link: models, source latents, 12-h replay extracts).
#   bash training/wrf/fetch_r37.sh [REP_SEEDS]   (REP_SEEDS: seeds whose full surface.npz is fetched, e.g. "1 5")
set -euo pipefail
M=/pscratch/sd/m/mwest/aeromeld
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 -o ProxyCommand=false perlmutter"
SCP="scp -q -o BatchMode=yes -o ProxyCommand=false"
mkdir -p data/models data/wrfml_runs/zsrc_dry notes/affx_stats
# models (numpy decoders, checkpoints, run records) and their manifest lines
$SCP "perlmutter:$M/wrfml/models_dry/*_dry.*.npz" "perlmutter:$M/wrfml/models_dry/*_dry.*.json" \
     "perlmutter:$M/wrfml/models_dry/*_dry.*.pt" data/models/
$SSH "cat $M/wrfml/models_dry/MANIFEST.txt" | while read -r line; do
  grep -qxF "$line" data/models/MANIFEST.txt || echo "$line" >> data/models/MANIFEST.txt
done
(cd data/models && shasum -a 256 *_dry.*.npz) | while read -r line; do
  grep -qF "${line##* }" data/models/SHA256SUMS || echo "$line" >> data/models/SHA256SUMS
done
# source latents of the WRF populations
$SCP "perlmutter:$M/transport/zsrc_dry/zsrc_s*.npz" data/wrfml_runs/zsrc_dry/
# test-set K-curves
$SCP perlmutter:$M/wrfml/mixk_r37_lr.json notes/affx_stats/
# 12-h replays: the 0/4/12-h extract of every seed (as surface.npz, the layout transport_schemes.load reads),
# the 30-minute extract (surface_30min.npz) for the representative seeds
for k in 0 1 2 3 4 5 6 7 8 9; do
  mkdir -p data/transport/dry${k}_12h
  case " ${1:-} " in
    *" $k "*) $SCP perlmutter:$M/transport/runs/dry${k}_12h/surface_30min.npz data/transport/dry${k}_12h/surface.npz ;;
    *) $SCP perlmutter:$M/transport/runs/dry${k}_12h/surface_end.npz data/transport/dry${k}_12h/surface.npz ;;
  esac
done
ls -la data/transport/dry*_12h/ data/wrfml_runs/zsrc_dry/ | head -40
echo "models: $(ls data/models/*_dry.*.npz | wc -l)"

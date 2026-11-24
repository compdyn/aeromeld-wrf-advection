#!/bin/bash
# After round 37 (2026-11-16, the clean dry-diameter rebuild: training/tasks/r37/): source latents,
# the 12-hour offline replays of every seed, test-set K-curves, and model collection, for the five
# decoders <model>_dry.<seed>. Run on a Perlmutter login node (it only submits and waits):
#   bash after_r37.sh  (from $M/affx/r37/code, where this and the r37 code were copied)
# SEEDS (default 0-9) limits the encodings and replays to some seeds, PARTS (default "enc rep mixk models")
# the steps, so seeds whose training finished can go ahead:
#   SEEDS="0 1 2 3" PARTS="enc rep" bash after_r37.sh;  later SEEDS="4 5 6 7 8 9" bash after_r37.sh
set -uo pipefail
SEEDS=${SEEDS:-0 1 2 3 4 5 6 7 8 9}
PARTS=${PARTS:-enc rep mixk models}
has() { case " $PARTS " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
M=/pscratch/sd/m/mwest/aeromeld
C=$M/affx/r37/code
L=$M/logs
cd $M/affx
mkdir -p $M/transport/zsrc_dry $M/wrfml/models_dry $M/wrfml/latents_dry
NAMES="twin_k16 twin_k32 twin_pairs twin_none ax_rel1"
# 1. source latents of the three WRF populations (encoder checked against the run's own latents)
jobs=()
for si in $SEEDS; do
  has enc || break
  args=""
  for n in $NAMES; do args="$args ${n}=r37/r37_${n}_s${si}"; done
  jobs+=($(sbatch --parsable -t 00:20:00 -J aeromeld_r37_enc_s$si $C/affx_ana.sbatch \
    "python $C/affx_wrf3_npz.py $M/transport/zsrc_dry/zsrc_s$si.npz $args --seed $si --samples wrf3_jeff.npz 2>&1 | grep -v Warn | grep -E 'numpy decoder|wrote|Error|assert|rror'"))
done
dep=$(IFS=:; echo "${jobs[*]:-}")
echo "encode jobs: $dep"
# 2. the 12-hour replays, every seed, all five decoders (after the encodings)
if has rep; then
  cd $M/transport/code
  arr=$(echo $SEEDS | tr ' ' ',')
  rep=$(sbatch --parsable ${dep:+--dependency=afterok:$dep} --array=$arr -J aeromeld_rep12dry \
        --export=ALL,REP_PREFIX=dry,ZSRC_DIR=zsrc_dry rep12_array.sbatch)
  echo "replays: $rep (seeds $arr)"
fi
has mixk || has models || { echo "after_r37 ($PARTS, seeds $SEEDS) submitted $(date)"; exit 0; }
# 3. test-set K-curves (the paper's error, --logratio, against the dry targets)
cd $M/affx
s() { echo r37/r37_$1_s0,r37/r37_$1_s1,r37/r37_$1_s2,r37/r37_$1_s3,r37/r37_$1_s4,r37/r37_$1_s5,r37/r37_$1_s6,r37/r37_$1_s7,r37/r37_$1_s8,r37/r37_$1_s9; }
has mixk && mk=$(sbatch --parsable -t 01:30:00 -J aeromeld_r37_mixk $C/affx_ana.sbatch \
  "python $C/affx_mixk.py $(for n in $NAMES; do echo -n "${n}_dry=$(s $n) "; done) --K 1 2 3 4 8 16 32 64 128 256 1024 5000 --nmix 2000 --md --pct --logratio --cache $M/affx/cache_dry.npz --json $M/wrfml/mixk_r37_lr.json")
has mixk && echo "mixk: $mk"
has models || exit 0
# 4. models: checkpoints, numpy files, latents
for n in $NAMES; do for si in 0 1 2 3 4 5 6 7 8 9; do
  r=r37/r37_${n}_s$si
  cp -p ${r}_s$si.pt $M/wrfml/models_dry/${n}_dry.$si.pt; cp -p $r.json $M/wrfml/models_dry/${n}_dry.$si.json
  cp -p ${r}_s${si}_Z.npy $M/wrfml/latents_dry/${n}_dry.$si.Z.npy
  echo "${n}_dry.$si ${r}_s$si" >> $M/wrfml/models_dry/MANIFEST.txt
done; done
module load pytorch/2.8.0 >/dev/null 2>&1
(cd $M/wrfml/models_dry && python $C/pt_to_npz.py *_dry.*.pt | tail -1)
echo "after_r37 submitted $(date)"

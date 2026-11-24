#!/bin/bash
# After rounds 34 and 35: the test-set K-curves (affx_mixk.py, the options of after_r33.sh, so the
# mixtures are the same as for the paper's models) of
#   twin_mix0_of   (r34) the KL term, no mixtures      -- trains badly (test error 7-25 %)
#   twin_k16kl0_of (r35) mixtures of up to 16, no KL   -- KL-matched to twin_none_of
# Run in one interactive GPU allocation:
#   salloc -A gen0019_g -C gpu -q interactive -N 1 -G 1 -c 32 -t 01:00:00 bash training/wrf/after_r35.sh
set -uo pipefail
M=/pscratch/sd/m/mwest/aeromeld
T=$M/wrfml/paper/training
cd $M/affx
module load pytorch/2.8.0
s34=r34/r34_twin_mix0_s0,r34/r34_twin_mix0_s1,r34/r34_twin_mix0_s2,r34/r34_twin_mix0_s3   # seeds 4-9 not trained (stopped: the recipe fails)
s35=$(echo r35/r35_twin_k16kl0_s{0..9} | tr ' ' ',')
python $T/affx_mixk.py twin_mix0_of=$s34 twin_k16kl0_of=$s35 --K 1 2 3 4 8 16 32 64 128 256 1024 5000 --nmix 2000 --md --pct \
  --cache /pscratch/sd/m/mwest/aeromeld/affx/cache_optfix.npz --json ../wrfml/mixk_r35.json 2>&1 | grep -v Warn | tail -5
echo "after_r35 done $(date)"

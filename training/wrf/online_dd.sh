#!/bin/bash
# The online check of the final rebuild (2026-11-19): WRF-AeroMELD with per-tracer latent advection
# inside WRF for 12 h, with the source latents of twin_k16_dd.<seed> (round 39: dry diameter, water in the masses), made as
# cares_v3match_twin_k16_of.6.0_pertracer_12h was for twin_k16_of (notes/transport_schemes.md):
#   1. the source latents in build_run.py's layout (write_latents.py: 9 latents, then a 0 for ntot);
#   2. the 4-h run directory from build_all.sh (inputs and emissions for the model);
#   3. to_extensive.py -> a per-tracer copy, with the 12-h run's namelist (run_hours 12, 15-min output);
#   4. WRF on one CPU node (about 5 minutes), then the extract (extract_runs.py --full, z_surf and
#      ntot_surf every 15 minutes) for checks/pertracer_online.py against the replay dd<seed>_12h.
# On a login node (it builds the inputs, a few seconds of I/O, and submits):
#   bash online_dd.sh SEED   (after extract_runs.py POINTS holds the final points A, B, C)
set -euo pipefail
seed=$1
M=/pscratch/sd/m/mwest/aeromeld
W=$M/wrfml
T=$M/affx/r39/code
f=twin_k16_dd.$seed.0
R=$W/work/runs
src=$R/latent_advection_cares_v3match_${f}_5e7
dst=$R/cares_v3match_${f}_pertracer_12h
ref=$R/cares_v3match_twin_k16_of.6.0_pertracer_12h
# 1.
$W/venv/bin/python - $M/transport/zsrc_dd/zsrc_s$seed.npz $W/work/latents/$f <<'EOF'
import sys
from pathlib import Path
import numpy as np
z = np.load(sys.argv[1])
assert (z['pops'] // 25).tolist() == [561, 635, 812], z['pops']
zz = z['twin_k16_z']
assert zz.shape == (3, 9) and (zz >= 0).all(), zz
d = Path(sys.argv[2])
d.mkdir(parents=True, exist_ok=False)
for s, v in zip((561, 635, 812), zz):
    (d / ('latent_scenario_%d_0001_%08d.txt' % (s, 13))).write_text(''.join('%.9e\n' % x for x in list(v) + [0.0]))
print('wrote', d)
EOF
# 2.
bash $W/paper/training/wrf/build_all.sh $f
test -f $src/wrfinput_d01
# 3.
. $W/venv/bin/activate
python $T/to_extensive.py $src $dst $W/work/bin/wrf_pertracer.exe
rm -f $dst/namelist.input
cp $ref/namelist.input $dst/namelist.input
diff <(grep -v -E "run_hours|frames_per_outfile" $src/namelist.input) <(grep -v -E "run_hours|frames_per_outfile" $dst/namelist.input)
sed -e "s/-J pt_12h/-J aeromeld_pt12h_dd/" -e "s#$ref#$dst#g" $ref/batch.sh > $dst/batch.sh
grep -q "$dst" $dst/batch.sh
# 4.
cd $dst
wrf=$(sbatch --parsable batch.sh)
echo "WRF: $wrf"
ext=$(sbatch --parsable --dependency=afterok:$wrf -A gen0019 -C cpu -q shared -c 8 --mem=48G -t 00:30:00 \
  -J aeromeld_extract12h_dd -o $M/logs/%x_%j.out --wrap "cd $W/work/runs && ln -sfn $(basename $dst) latent_advection_cares_v3match_${f}_pertracer12h && cd $W && $W/venv/bin/python $T/extract_runs.py work extracts_full --full $f work/runs/latent_advection_cares_v3match_${f}_pertracer12h; ls -la extracts_full | grep ${f}_pertracer12h")
echo "extract: $ext"

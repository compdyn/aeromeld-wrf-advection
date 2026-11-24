#!/bin/bash
# After round 34 (twin_mix0_of: the decoder with the KL term but no mixtures, ten seeds, on the
# optics-corrected targets): source latents merged into the replays' zsrc files, the models as
# numpy files, the test set and K-curves, and the replays (every scheme, every seed).
# As after_r33.sh, for one model, but every step runs here (no sbatch), so that the whole script
# can run in one interactive GPU allocation:
#   salloc -A gen0019_g -C gpu -q interactive -N 1 -G 1 -c 32 -t 02:00:00 bash training/wrf/after_r34.sh
set -uo pipefail
M=/pscratch/sd/m/mwest/aeromeld
W=$M/wrfml
T=$W/paper/training
VPY=$W/venv/bin/python                    # numpy/cupy/netCDF4 (replays, file merging)
cd $M/affx
for si in 0 1 2 3 4 5 6 7 8 9; do
  test -f r34/r34_twin_mix0_s${si}_s${si}.pt || { echo "missing r34 seed $si"; exit 1; }
done
module load pytorch/2.8.0
echo "== 1. source latents $(date)"
mkdir -p $W/zsrc_mix0
for si in 0 1 2 3 4 5 6 7 8 9; do
  python $T/affx_wrf3_npz.py $W/zsrc_mix0/zsrc_s$si.npz twin_mix0=r34/r34_twin_mix0_s${si} --seed $si \
    --samples wrf3_jeff.npz 2>&1 | grep -v Warn | grep -E 'wrote|Error|assert|rror'
done
$VPY - <<PY
import numpy as np, shutil
from pathlib import Path
for si in range(10):
    dst = Path("$M/transport/zsrc/zsrc_s%d.npz" % si)
    new = dict(np.load("$W/zsrc_mix0/zsrc_s%d.npz" % si))
    old = dict(np.load(dst))
    assert (new["pops"] == old["pops"]).all() and np.allclose(new["X"], old["X"])
    zz = new["twin_mix0_z"]; assert zz.shape == (3, 9) and (zz >= 0).all()
    shutil.copy(dst, dst.with_name(dst.stem + "_before_r34.npz"))
    old.update({k: v for k, v in new.items() if k.startswith("twin_mix0_")})
    np.savez(dst, **old)
print("zsrc merged")
PY
echo "== 2. models $(date)"
for si in 0 1 2 3 4 5 6 7 8 9; do
  r=r34/r34_twin_mix0_s$si
  cp -p ${r}_s$si.pt $W/models/twin_mix0_of.$si.pt; cp -p $r.json $W/models/twin_mix0_of.$si.json
  cp -p ${r}_s${si}_Z.npy $W/latents_all/twin_mix0_of.$si.Z.npy
  echo "twin_mix0_of.$si ${r}_s$si" >> $W/models/MANIFEST.txt
done
(cd $W/models && python $T/wrf/pt_to_npz.py twin_mix0_of.*.pt | tail -1)
echo "== 3. test set and K-curves $(date)"
s=$(echo r34/r34_twin_mix0_s{0..9} | tr ' ' ',')
AFFX_CACHE=cache_optfix.npz python $T/affx_testset.py ../wrfml/testset_of twin_mix0_of=$s 2>&1 | grep -v Warn | tail -3
python $T/affx_mixk.py twin_mix0_of=$s --K 1 2 3 4 8 16 32 64 128 256 1024 5000 --nmix 2000 --md --pct \
  --cache /pscratch/sd/m/mwest/aeromeld/affx/cache_optfix.npz --json ../wrfml/mixk_r34.json 2>&1 | grep -v Warn | tail -3
echo "== 4. replays $(date)"
cd $M/transport/code
for si in 0 1 2 3 4 5 6 7 8 9; do
  srun -n 1 --gpus 1 $VPY run.py --out ../runs/mx$si --gpu --models twin_mix0 --zsrc ../zsrc/zsrc_s$si.npz \
    --schemes oracle,donor,sep,shB > ../runs/mx$si.log 2>&1
  tail -1 ../runs/mx$si.log
done
echo "after_r34 done $(date)"

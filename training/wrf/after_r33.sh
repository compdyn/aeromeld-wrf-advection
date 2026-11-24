#!/bin/bash
# After round 33 (the six models retrained on the optics-corrected targets): source latents,
# WRF-AeroMELD runs, extracts, test-set export and model collection, for fpidx <model>_of.<seed>.0
# ("of" = optics fixed). Run on a Perlmutter login node from anywhere.
set -uo pipefail
M=/pscratch/sd/m/mwest/aeromeld
W=$M/wrfml
T=$W/paper/training
cd $M/affx; mkdir -p ../wrfml/zsrc_of
# 1. source latents (encoder checked against the run's own latents; the default cache: it only
#    checks the samples' dataset diagnostics, and the latents do not depend on the targets)
for si in 0 1 2 3 4 5 6 7 8 9; do
  args=""
  for n in twin_k16 twin_k32 twin_pairs twin_none ax_rel1 ax_k16; do args="$args ${n}=r33/r33_${n}_s${si}"; done
  sbatch -W -t 00:20:00 -J r33_enc_s$si $M/aeromeld-lindiag/alcf/affx/affx_ana.sbatch \
    "python $T/affx_wrf3_npz.py ../wrfml/zsrc_of/zsrc_s$si.npz $args --seed $si --samples wrf3_jeff.npz 2>&1 | grep -v Warn | grep -E 'wrote|Error|assert|rror'" &
done
wait
ls ../wrfml/zsrc_of | wc -l
# 2. latents in build_run's format, build and submit
module load python >/dev/null 2>&1; . $W/venv/bin/activate
python - <<PY
import numpy as np
from pathlib import Path
for si in range(10):
    z = np.load("$W/zsrc_of/zsrc_s%d.npz" % si)
    assert (z["pops"] // 25).tolist() == [561, 635, 812]
    for n in ("twin_k16", "twin_k32", "twin_pairs", "twin_none", "ax_rel1", "ax_k16"):
        zz = z[n + "_z"]; assert zz.shape == (3, 9) and (zz >= 0).all()
        d = Path("$W/work/latents/%s_of.%d.0" % (n, si)); d.mkdir(parents=True, exist_ok=True)
        for s, v in zip((561, 635, 812), zz):
            (d / ("latent_scenario_%d_0001_%08d.txt" % (s, 13))).write_text("".join("%.9e\n" % x for x in list(v) + [0.0]))
print("latents written")
PY
for n in twin_k16 twin_k32 twin_pairs twin_none ax_rel1 ax_k16; do              # one builder per model, in parallel
  bash $T/wrf/build_all.sh --submit $(for s in 0 1 2 3 4 5 6 7 8 9; do echo -n "${n}_of.$s.0 "; done) \
    > $W/logs/after_r33_build_$n.log 2>&1 &
done
wait
cat $W/logs/after_r33_build_*.log | grep -c submitted; cat $W/logs/after_r33_build_*.log | grep -v "build exit 0\|submitted" | head
# 3. wait for the runs; extracts and provenance
while squeue -h -u mwest -o %j | grep -q '_of\.[0-9]\.0_5e7'; do sleep 20; done
grep -L "SUCCESS COMPLETE" $W/work/runs/latent*_of.*/rsl.error.0000
python $T/wrf/extract_runs.py $W/work $W/extracts 2>&1 | grep -c wrote
for r in $W/work/runs/latent*_of.*/; do b=$(basename $r); mkdir -p $W/provenance/runs/$b
  cp -p $r/namelist.input $r/namelist.output $r/batch.sh $r/rsl.error.0000 $r/slurm-*.out $W/provenance/runs/$b/; done
cp -p $W/logs/build_*_of.*.log $W/provenance/runs/
# 4. models: checkpoints, numpy files, latents of every population
cd $M/affx
for n in twin_k16 twin_k32 twin_pairs twin_none ax_rel1 ax_k16; do for si in 0 1 2 3 4 5 6 7 8 9; do
  r=r33/r33_${n}_s$si
  cp -p ${r}_s$si.pt $W/models/${n}_of.$si.pt; cp -p $r.json $W/models/${n}_of.$si.json
  cp -p ${r}_s${si}_Z.npy $W/latents_all/${n}_of.$si.Z.npy
  echo "${n}_of.$si ${r}_s$si" >> $W/models/MANIFEST.txt
done; done
module load pytorch/2.8.0 >/dev/null 2>&1
(cd $W/models && python $T/wrf/pt_to_npz.py *_of.*.pt | tail -1)
# 5. test set (against the corrected targets) and K-curves
s() { echo r33/r33_$1_s0,r33/r33_$1_s1,r33/r33_$1_s2,r33/r33_$1_s3,r33/r33_$1_s4,r33/r33_$1_s5,r33/r33_$1_s6,r33/r33_$1_s7,r33/r33_$1_s8,r33/r33_$1_s9; }
sbatch -t 00:40:00 -J r33_testset $M/aeromeld-lindiag/alcf/affx/affx_ana.sbatch \
  "AFFX_CACHE=cache_optfix.npz python $T/affx_testset.py ../wrfml/testset_of $(for n in twin_k16 twin_k32 twin_pairs twin_none ax_rel1 ax_k16; do echo -n "${n}_of=$(s $n) "; done)"
sbatch -t 01:20:00 -J r33_mixk $M/aeromeld-lindiag/alcf/affx/affx_ana.sbatch \
  "python $T/affx_mixk.py $(for n in twin_k16 twin_k32 twin_pairs twin_none ax_rel1 ax_k16; do echo -n "${n}_of=$(s $n) "; done) --K 1 2 3 4 8 16 32 64 128 256 1024 5000 --nmix 2000 --md --pct --cache /pscratch/sd/m/mwest/aeromeld/affx/cache_optfix.npz --json ../wrfml/mixk_r33.json"
echo "after_r33 done $(date)"

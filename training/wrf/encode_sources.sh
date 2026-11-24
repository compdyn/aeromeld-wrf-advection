# Source latents (the three 1000-particle source files) of every model and seed run in WRF-AeroMELD.
# Run in $SCRATCH/aeromeld/affx (affx_ana.sbatch). Writes ../wrfml/zsrc/zsrc_s<k>.npz.
A=../aeromeld-lindiag/alcf/affx
split() { case $2 in 0|1|2) echo $1;; 3|4) echo $3_s34;; 5|6) echo $3_s56;; 7|8) echo $3_s78;; 9) echo $3_s9;; esac; }
for si in 0 1 2 3 4 5 6 7 8 9; do
  k16=$(split r30/r30_x_nl_mix10_k16 $si r31/r31_x_nl_mix10_k16)
  k32=$(split r31/r31_x_nl_mix10_k32 $si r32/r32_x_nl_mix10_k32)
  pairs=$(split r26/r26_x_nl_mix10 $si r28/r28_x_nl_mix10)
  none=$(split r29/r29_x_nl_kl0_mix0 $si r30/r30_x_nl_kl0_mix0)
  ax=$(split r20/r20_x_rel1_1e5 $si r21/r21_x_rel1_1e5)
  extra=""; if [ $si -le 2 ]; then extra="mix0=r26/r26_x_nl_mix0"; fi
  python $A/affx_wrf3_npz.py ../wrfml/zsrc/zsrc_s$si.npz k16=$k16 k32=$k32 pairs=$pairs none=$none ax=$ax $extra \
     --seed $si --samples wrf3_jeff.npz 2>&1 | grep -v Warn | grep -E "numpy decoder|wrote|Error|assert|rror"
done
ls -la ../wrfml/zsrc

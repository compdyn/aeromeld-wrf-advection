# Copy the checkpoints (and training json) of the six models x seeds run in WRF-AeroMELD
# into ../wrfml/models/<model>.<seed>.pt, with MANIFEST.txt (source run prefix of each).
split() { case $2 in 0|1|2) echo $1;; 3|4) echo $3_s34;; 5|6) echo $3_s56;; 7|8) echo $3_s78;; 9) echo $3_s9;; esac; }
M=../wrfml/models; : > $M/MANIFEST.txt
for si in 0 1 2 3 4 5 6 7 8 9; do
  for spec in "twin_k16 r30/r30_x_nl_mix10_k16 r31/r31_x_nl_mix10_k16" "twin_k32 r31/r31_x_nl_mix10_k32 r32/r32_x_nl_mix10_k32" \
              "twin_pairs r26/r26_x_nl_mix10 r28/r28_x_nl_mix10" "twin_none r29/r29_x_nl_kl0_mix0 r30/r30_x_nl_kl0_mix0" \
              "ax_rel1 r20/r20_x_rel1_1e5 r21/r21_x_rel1_1e5" "twin_mix0 r26/r26_x_nl_mix0 -"; do
    set -- $spec; [ $2 = twin_mix0 ] ; run=$(split $2 $si $3)
    [ "$1" = twin_mix0 ] && [ $si -gt 2 ] && continue
    cp -p ${run}_s$si.pt $M/$1.$si.pt && cp -p ${run}.json $M/$1.$si.json && echo "$1.$si ${run}_s$si" >> $M/MANIFEST.txt
  done
done
wc -l $M/MANIFEST.txt; du -sh $M

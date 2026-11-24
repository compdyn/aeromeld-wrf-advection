# Offline latent transport (the CARES case)

The CARES case's aerosol moved offline on WRF's own mass fluxes, with several
latent-transport schemes side by side. The study and its results are in
`notes/transport_schemes.md`.

- `wrfadv.py`: WRF 4.7's scalar advection (`advect_scalar` / `advect_scalar_mono`):
  RK3 stages, 5th/3rd-order fluxes, the monotonic (Zalesak) limiter, and a
  synchronized limiter for several constraints that share one coefficient per
  face. Works on numpy or cupy arrays.
- `schemes.py`: the schemes (`oracle`, `donor`, `donorW`, `shN`, `shB:<model>`,
  `sep:<model>`) as one `Runner` that advances them all on the same fluxes.
- `run.py`: the replay driver (Perlmutter: fluxes, original wrfout, WRF-PartMC
  control files, source latents). Writes the lowest level every 15 minutes to
  `<out>/surface.npz`.
- `run_gpu.sbatch`: batch job for one GPU.
- `extract_replay.py`: run extracts (`data/wrfml_runs/<model>_<scheme>.npz`,
  the format of `training/wrf/extract_runs.py`) from a replay, so replays are
  ordinary registered runs for the paper's products and figures.
- `test_synthetic.py`: checks on a small synthetic domain (seconds, numpy).
  Uniform fields stay uniform; groups add up to N; the oracle and `shB:frac`
  stay within [0, 1]; `shB` stays on the mixing plane while `sep` leaves it;
  donor-cell has the widest fronts.

On Perlmutter (`$SCRATCH/aeromeld/transport`):

- `wrf_fluxes/` is the rerun of the WRF-AeroMELD meteorology with
  `do_avgflx_em = 1`, 1-minute history, and `drop_hist.txt` keeping only the
  fluxes and MU. Run it once (one CPU node, 20 min).
- `partmc_inputs/` holds the WRF-PartMC run's control files.
- `zsrc_s0.npz` is `data/wrfml_runs/zsrc_of/zsrc_s0.npz`.
- `code/` holds copies of this directory.
- `runs/<name>/` holds the outputs. Copy `surface.npz` to `data/transport/<name>/`
  for the analysis.

Run it (about 3 GPU-minutes):

    cd $SCRATCH/aeromeld/transport/code
    salloc -q interactive -A gen0019_g -C gpu -G 1 -c 32 -t 00:45:00 \
        srun -n 1 --gpus 1 $SCRATCH/aeromeld/wrfml/venv/bin/python run.py --out ../runs/r3 --gpu \
        --models twin_none,twin_pairs,twin_k16,twin_k32,ax_rel1

Validation: `scripts/checks/transport_replay.py`. Figures:
`scripts/plot/transport_schemes.py`.

# Training and evaluating the AeroMELD models (`affx`)

This directory holds the code that trained and evaluated the paper's latent
models (the encoder and the five decoders), and the scripts that ran them in
WRF-AeroMELD on NERSC Perlmutter. The scripts keep the absolute paths of the
authors' Perlmutter workspace (`/pscratch/sd/m/mwest/aeromeld`, `$M` below);
change them to your own.

## The models

Every model has a latent vector of 9 components (plus the number $N$), the
same encoder (a per-particle MLP on square-root species masses, then a
softplus and the number-weighted mean over the particles, so $z \ge 0$ and
mixing populations mixes their latent vectors exactly), and is trained for
1e5 iterations on the PM10 particle dataset with the test case's source
scenarios 561, 635 and 812 held out of training. Ten seeds (0–9), each with
its own 5000 test populations (`data/affx/test_idx.npz` in the data deposit).

| Paper name | Model | Decoder | Training |
|---|---|---|---|
| decoder 1 (no mixtures) | `twin_none` | MLP 9→256→256→456, positive, monotone CCN and frozen fraction | no KL term, single populations |
| decoder 2 (pairs) | `twin_pairs` | same | KL 0.03, mixup on pairs |
| decoder 3 (up to 16, the main model) | `twin_k16` | same | KL 0.03, mixtures of up to 16 populations |
| decoder 4 (up to 32) | `twin_k32` | same | KL 0.03, mixtures of up to 32 populations |
| decoder 5 (affine) | `ax_rel1` | affine, $\hat y = A z + b$, $A, b \ge 0$ | KL 0.03, mixup |

The paper's models are training round 39 (`tasks/r39/`, five decoders × ten
seeds), collected as `<model>_dd.<seed>` (`dd`: the number and speciated-mass
size distributions are binned by dry diameter, with every species' mass,
water included). Round 40 (`tasks/r40/`) holds the two training checks of the
decoder appendices: decoder 3 without the KL term (`twin_k16kl0`, ten seeds)
and the no-mixtures decoder with the KL term (`twin_mix0`, four seeds). The
task files of earlier rounds are kept for reference.

## Training targets

1. The PM10 particle dataset (`04_bwchisamp_pm10.h5`) and its corrected
   optics (`diag_fixed.npz`) are built from the Illinois Data Bank library
   with the `dataset_build` scripts of
   [partnn](https://github.com/ehsansaleh/partnn).
2. `affx.py` writes its cache (`cache.npz`: targets, $N$, particles) from the
   dataset on its first run.
3. `dataset/optfix_cache.py`: the cache with the corrected optics
   (`cache_optfix.npz`).
4. `dataset/dry_cache.py --basis drydiam`: the size distributions binned by
   dry diameter, every species' mass in the speciated-mass distribution
   (`cache_drydiam.npz`, the targets of rounds 39 and 40; also in the data
   deposit).

## Files

| File | Role |
|---|---|
| `affx.py` | The trainer (all options in its docstring). |
| `affx.sbatch`, `dry_decoders.sbatch`, `tasks/*/*.txt` | Perlmutter batch scripts; one task file per job, one line of `affx.py` options per GPU. |
| `paper_errors.py` | The test metrics of the AeroMELD paper. |
| `affx_mixk.py` | Error against the number $K$ of mixed test populations, with the error of decoding mixtures; writes the JSON in `../notes/affx_stats/` (rounds 39 and 40: `mixk_r39_lr.json`, `mixk_r40_lr.json`). |
| `affx_mixk_plot.py`, `affx_plot.py` | K-curve figures and example decoded test populations. |
| `affx_table.py`, `affx_merge.py`, `affx_testset.py` | Test-set tables, merging of seed groups, and per-population test errors. |
| `affx_wrf3_npz.py` | Source latents of the three test-case populations under each model. |
| `resample_wrf3.py` | The diagnostics of the three 1000-particle source populations with the training data's code. |
| `affx_ana.sbatch` | Runs an analysis command in a batch job. |
| `dataset/` | The training targets from the particle dataset (above). |
| `wrf/` | WRF-AeroMELD on Perlmutter for these models (below). |

## WRF-AeroMELD runs and offline replays (`wrf/`)

WRF-AeroMELD is release `wrf-aeromeld-v1.0` of
[compdyn/WRF](https://github.com/compdyn/WRF) (doi:10.5281/zenodo.24167410),
WRF v4.7.1 with the latent advection and per-tracer advection of the latent
state (`latent_per_tracer.patch` is the per-tracer part).

- `build_wrf.sbatch`, `build_wrf_pertracer.sbatch`: compile WRF on a CPU node
  (GNU dmpar, Cray compiler wrappers, `cray-netcdf`).
- `after_r39.sh`: after round 39, the source latents of every model and seed
  (`affx_wrf3_npz.py`), the 12-h offline replays of WRF's advection
  (`../transport/`), the test-set K-curves and the model collection
  (`collect_models.sh`, `pt_to_npz.py`: numpy-only `.npz` copies of the
  checkpoints). `fetch_r39.sh` downloads the results.
- `online_dd.sh SEED`: the online check, WRF-AeroMELD run inside WRF for
  12 h with the representative seed's source latents, then
  `extract_runs.py --full` for `../scripts/checks/pertracer_online.py`.
- `build_all.sh`, `write_latents.py`, `to_extensive.py`,
  `batch_perlmutter.sh`: build and submit a run with
  `../scripts/inputs/build_run.py` from the WRF input templates of the data
  deposit (`wrf_inputs/`).
- `extract_runs.py`, `extract_transport.py`: small extracts of finished runs.

The `after_r3x.sh`, `online_dry.sh`, `r35.sbatch` and `run_r3x_*` scripts are
the same steps for earlier rounds.

# Advecting latent representations of particle-resolved aerosol in WRF

Code for the paper

> J. H. Curtis, W. Tang, N. Riemer and M. West, *Advecting Latent
> Representations of Particle-Resolved Aerosol: Transporting Aerosol Mixing
> State in Three Dimensions with a Learned Latent Space*, submitted to the
> Journal of Advances in Modeling Earth Systems (2026).

WRF-AeroMELD represents the aerosol in each grid cell of WRF by the AeroMELD
state, a number mixing ratio and nine learned latent variables, and advects it
as ten tracers with WRF's standard monotonic advection. This repository holds
the training and evaluation of the AeroMELD models, the offline replays of
WRF's advection, the sectional and modal schemes it is compared with (MOSAIC,
TOMAS, MAM4 and MATRIX), the error analysis, and the figures.

## Related archives

| What | Where |
|---|---|
| WRF-AeroMELD (WRF v4.7.1 with latent advection) | [compdyn/WRF](https://github.com/compdyn/WRF), release `wrf-aeromeld-v1.0`, doi:[10.5281/zenodo.24167410](https://doi.org/10.5281/zenodo.24167410) |
| Training targets, trained models, WRF input templates, transport results and processed data | Zenodo, doi:DATA_DOI |
| The PartMC-MOSAIC scenario library | Illinois Data Bank, doi:[10.13012/B2IDB-2774261_V1](https://doi.org/10.13012/B2IDB-2774261_V1) |
| Building the particle dataset from the library | [partnn](https://github.com/ehsansaleh/partnn), `dataset_build/` |

Unpack the data archive at the root of this repository, so that its `data/`,
`baselines/data/` and `wrf_inputs/` directories sit next to the code.

## Layout

| Directory | Contents |
|---|---|
| `training/` | The AeroMELD encoder and the five decoders: training (`affx.py`), test-set evaluation, the training targets, and the WRF-AeroMELD runs on Perlmutter ([README](training/README.md)). |
| `transport/` | Offline replays of WRF's advection of the latent state on a GPU, with the three transport schemes (donor-cell, shared limiter, separate limiters) and per-source transport ([README](transport/README.md)). |
| `baselines/` | The sectional and modal schemes (`sectional.py`, `tomas.py`, `modal.py`, `matrix.py`), their source states and diagnostics (`diag.py`, `make_sources.py`), their transport (`transport/`), the error terms at 12 h (`static_terms.py`, `transport_terms.py`), the test-set curves (`kcurves.py`) and the MATRIX sensitivity (`matrix_sensitivity.py`). |
| `scripts/` | The error decomposition (`decomposition.py`), the processed data products (`paperfigs/`, `extract/`), the figures (`make_all.py`, `plot/`), checks including every number of the paper (`checks/paper_numbers.py`), and building WRF-AeroMELD runs (`inputs/`) ([README](scripts/README.md)). |
| `notes/affx_stats/` | Test-set statistics of the models (`affx_mixk.py` output); rounds 39 and 40 are the paper's. |
| `data/` | `PRODUCTS.md` (every processed data product) and the model manifest and checksums; the data themselves are in the data archive. |

## Reproducing the paper

The heavy steps ran on NERSC Perlmutter (Slurm, GPU and CPU nodes), and the
batch scripts keep the authors' paths (`/pscratch/sd/m/mwest/aeromeld`) and
allocation; change both to your own. With the data archive, every figure and
number can be regenerated on a workstation from the processed products
(steps 5 and 6).

1. **Particle dataset.** Build `04_bwchisamp_pm10.h5` and `diag_fixed.npz`
   from the Illinois Data Bank library with partnn's `dataset_build/`.
2. **Training targets.** `training/dataset/optfix_cache.py`, then
   `training/dataset/dry_cache.py --basis drydiam` (`cache_drydiam.npz`, also
   in the data archive).
3. **Models.** Train the task files `training/tasks/r39/` (and `r40/` for the
   decoder checks) with `training/affx.sbatch`, then `training/wrf/after_r39.sh`
   for the source latents, the 12-h offline replays, the test-set curves and
   the model collection. The trained models are in the data archive.
4. **Sectional and modal schemes.** `baselines/make_sources.py`, the transport
   runs `baselines/transport/baselines_array.sbatch` and `matrix_run.sbatch`,
   then `baselines/transport_terms.sbatch`, `kcurves.sbatch` and
   `matrix_sensitivity.sbatch`, with `AEROMELD_SIZE_BASIS=drydiam`.
5. **Analysis.** From `scripts/`, with `AEROMELD_SIZE_BASIS=drydiam`:
   `python decomposition.py` and the products of `data/PRODUCTS.md`
   (`python -m paperfigs.products`).
6. **Figures and numbers.** `MPLBACKEND=Agg python make_all.py` writes the
   figures to `figures/`, and `python checks/paper_numbers.py` prints every
   number in the paper.

The online check (WRF-AeroMELD inside WRF, step 3) needs WRF-AeroMELD built
from the release above and the WRF input templates of the data archive
(`training/wrf/online_dd.sh`).

## Environment

Python 3.11 or later with the packages of `scripts/requirements.txt` (numpy,
scipy, xarray, netCDF4, matplotlib, cartopy) for the analysis and figures, and
PyTorch 2.8, h5py and CuPy for training and the offline transport.

## License

MIT (see `LICENSE`). The files in `scripts/vendor/` are copies of code by the
authors and their group, under the same license.

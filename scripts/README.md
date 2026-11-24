# Figure scripts

Every figure in the paper is made by one script in `plot/`, from small
processed data files ("products") in `data/processed/`. Products are built
from the raw model output by `extract/`.
What each product holds: [../data/PRODUCTS.md](../data/PRODUCTS.md).

The WRF-AeroMELD runs themselves are built by `inputs/`: one command
encodes a VAE's populations and writes a checked, ready-to-submit run into
your own workspace. See [inputs/README.md](inputs/README.md).

## Figures

| PDF (`figures/`) | Paper label | Script | Products | Shows |
|---|---|---|---|---|
| `domain.pdf` | `fig:setup` | [plot/domain.py](plot/domain.py) | grid, partmc_fields | PartMC number mixing ratio at 00 UTC and the run-mean surface wind, with the emission source and sampling points A, B, C |
| `number_mixing_ratio_field.pdf` | `fig:number-maps` | [plot/number_mixing_ratio_field.py](plot/number_mixing_ratio_field.py) | wrfml_fields, partmc_fields, grid | total number mixing ratio at 04 UTC: AeroMELD, PartMC, difference |
| `source_number_fraction.pdf` | — | [plot/source_number_fraction.py](plot/source_number_fraction.py) | partmc_fields | number fraction of the background (561), urban (635) and emission (812) populations at each point over the run |
| `latent_space_normalized.pdf` | — | [plot/latent_maps.py](plot/latent_maps.py) | wrfml_fields, grid | maps of nine latents at 04 UTC (a 49-latent run shows the nine with the largest spread) |
| `latent_timeseries.pdf` | — | [plot/latent_timeseries.py](plot/latent_timeseries.py) | wrfml_fields | the same nine latents over the run at points A, B, C |
| `CCN_ML_vs_PR.pdf` | `fig:ccn-fraction` | [plot/ccn_maps.py](plot/ccn_maps.py) | aeromeld_domain, partmc_diag_domain, grid | CCN/CN fraction maps at s_env = 0.1/0.3/0.6/1.0 %: AeroMELD, PartMC, difference |
| `frozen_fraction.pdf` | — | [plot/frozen_fraction.py](plot/frozen_fraction.py) | aeromeld_domain, partmc_diag_domain, grid | frozen-fraction maps at −40/−30/−20/−10 °C (the violins' temperatures): AeroMELD, PartMC, ratio |
| `optics_ML_vs_PR.pdf` | — | [plot/optics_maps.py](plot/optics_maps.py) | aeromeld_domain, partmc_diag_domain, grid | scattering and absorption maps at 400/1000 nm: AeroMELD, PartMC, ratio |
| `summary_symmetric_error.pdf` | — | [plot/summary_symmetric_error.py](plot/summary_symmetric_error.py) | aeromeld_domain, partmc_diag_domain, partmc_fields | violins of the symmetric relative error over mixed cells, SMAPE per condition |
| `diagnostic_ML_vs_PR.pdf` | — | [plot/diagnostic_ML_vs_PR.py](plot/diagnostic_ML_vs_PR.py) | aeromeld_points, partmc_diag_points | CCN/CN, frozen-fraction, scattering and absorption spectra at points A/B/C, at 00, 02 and 04 UTC |
| `diagnostic_panels_ABC.pdf` | `fig:point-panels` | [plot/diagnostic_panels.py](plot/diagnostic_panels.py) | aeromeld_points, partmc_diag_points, partmc_fields | each point at 04 UTC in an outlined block headed by its population mix: number, mass and per-species size distributions, CCN/CN, frozen fraction, optics; PartMC rescaled to the decoded number; panels (a)–(x), one legend |
| `latent_populations.pdf` | `fig:populations` | [plot/latent_populations.py](plot/latent_populations.py) | populations (the run's model) | the three seed populations' latents, and their reconstruction against the 1000-particle sample: size distribution, CCN/CN, frozen fraction, optics, per-species mass (PartMC stacked in g–i, AeroMELD stacked in j–l) |
| `computational_cost.pdf` | — | [plot/computational_cost.py](plot/computational_cost.py) | none (numbers and sources in the script) | values stored per cell and core-minutes, WRF-PartMC vs WRF-AeroMELD |
| `model_comparison.pdf` | — | [plot/model_comparison.py](plot/model_comparison.py) | aeromeld_domain (each run), partmc_diag_domain, partmc_fields | the `paper_compare` runs (two MLP and three affine decoders) at the violin figure's 12 conditions: SMAPE, and the AeroMELD/PartMC ratio's median and p05–p95, over the mixed cells at 04 UTC |

All are at the surface, for the registry's main run. Every domain figure except
`domain` (which shows the start) is at the final instant, 2010-06-17 04:00 UTC.
`latent_populations` involves no WRF run: it reads the three particle files the
runs were seeded from, through the main run's model.
The sampling points are WRF's own 1-based (i, j) indices: (65, 50), (65, 85),
(90, 75).

`diagnostic_panels_ABC` (`fig:point-panels`) replaced the three single-point
figures `diagnostic_panel_{A,B,C}` on 2026-11-09. At 04 UTC, B is pure background, A mostly urban and C a
three-way mixture (`source_number_fraction`). Every figure of `read_wrf_data.ipynb`
that the plan keeps is ported. `model_comparison` is new: it compares the
registry's `[sets] paper_compare` runs (`--set`/`--runs` choose others, which
go to `graphics/by_model/_compare/`).

## Running

Two environments:

- **Plotting from products** (`data/processed/` in place, no raw output):
  Python ≥ 3.11 and [requirements.txt](requirements.txt): numpy, xarray,
  netCDF4, matplotlib, cartopy, shapely and scipy (cartopy needs scipy for the
  wind streamlines in `domain`). From the repository root:

  ```bash
  uv venv --python 3.12 .venv && uv pip install -r scripts/requirements.txt
  cd scripts && ../.venv/bin/python make_all.py --out /tmp/look
  ```

  The first map downloads Natural Earth coastlines and borders into cartopy's
  cache, so it needs network access once.
- **Building products** from the raw output: an environment with torch,
  [partnn](https://github.com/ehsansaleh/partnn) and PyPartMC as well.

The scripts select matplotlib's non-interactive Agg backend themselves
(`figures.setup`).

From this directory. The `export` line is for the `partnn` env, and so is any
command that has to build a product:

```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib
python make_all.py                        # every figure, main run -> figures/
python make_all.py --run 02_mlpprt.0.0_5e7   # another model -> graphics/by_model/<run>/
python plot/ccn_maps.py --format png --out /tmp/look
python -m paperfigs.products list|check|describe
python register_runs.py [--write|--check]   # registry of models and runs
python models.py --sort smape:ff_-30        # model selection, from data/metrics.csv
python models.py --write                    # recompute data/metrics.csv (every registered run)
```

## Choosing models: `models.py` and `data/metrics.csv`

`data/metrics.csv` (committed) holds the error statistics of every registered
run: one row per run per violin condition, over the mixed cells at 04:00, with
the registry columns joined on. The statistics come from
`paperfigs.analysis.compute`, the same function the violin figure uses.
`models.py` prints it as a run × condition table and sorts it, and needs only
the standard library. Every statistic, the ratio's included, is over the mixed
cells, except the `*_all` and `smape_unmixed_pct` columns; the ratio keeps
AeroMELD's sign, so cells decoded to ≤ 0 rank lowest (`n_nonpositive`). For example, `--sort smape:ff` sorts by the mean SMAPE
over the frozen-fraction columns, and `--sort ratio:ccn --show ratio` sorts by
median ML/PMC ratio, closest to 1 first. See `python models.py -h` for the
rest. After registering a new run, `python models.py --write` adds it. That
builds the run's `aeromeld_domain` product: about 3 s on a GPU, or 2.5 min on
CPU.

A plot script builds any product it lacks, or rebuilds a stale one, if the raw
output is reachable (paths in `paperfigs/config.py`). Otherwise it uses the
product it is given. Decoding runs on the GPU when one is available (a few
seconds per run).

## Layout

- `paperfigs/config.py`: paths, the instant, level and sampling points. The one place these are set.
- `paperfigs/registry.toml`: ML models and WRF-ML runs, keyed by run key (e.g. `05_nonlinvlong.0.0_5e7`). `role = "main"` marks the paper's run.
- `paperfigs/products.py`: load/build/staleness of products; `SPECS` lists them all.
- `paperfigs/analysis.py`: the error statistics (symmetric relative error, SMAPE, mixed cells); `compute(run)` feeds both the violins and the metrics table.
- `paperfigs/metrics.py`: writes and reads `data/metrics.csv` (columns documented in its docstring).
- `models.py`: the sortable run table, from `data/metrics.csv`.
- `paperfigs/figures.py`, `paperfigs/fig_palettes.py`: command line, output directory, maps, colours, page sizes.
- `extract/`: product builders (`grid`, `partmc`, `aeromeld`, `wrfml`, `populations`).
- `plot/`: one script per figure.
- `inputs/`: the WRF-AeroMELD run builder (`build_run.py`) and the input
  builders it drives; `ml_project/input_builders/` symlinks to them. See
  [inputs/README.md](inputs/README.md).
- `vendor/`: the copy of `vae_partmc_utils.py` (and the diagnostics it imports)
  that `extract/` and `inputs/` use, plus the `partnn` patch; see
  [vendor/README.md](vendor/README.md).

## Conventions

- Every number concentration and optical coefficient in a product is **per kg
  of dry air**. PartMC's per-m³ values are multiplied by its own ALT.
- Times are matched by exact timestamp, never by index or nearest step.
- Sampling points are 1-based (i, j) in `config.LOCATIONS`. Index arrays only
  through `common.point_index()`.
- Wavelengths are looked up exactly on each model's own grid (they differ).

## The paper's own models (from 2026-11-11)

The main run is now `twin_k16.0.0_5e7`: the nonlinear twin decoder trained on
mixtures of up to 16 populations, seed 0 (`training/README.md`,
`notes/affx_story.md`). Its models and runs are registered with
`backend = "affx"`. Their products are built from small run extracts in
`data/wrfml_runs/` and numpy model files in `data/models/`, so no torch, partnn
or raw WRF output is needed (`extract/affx.py`). The wrfout files stay on
Perlmutter. Sets: `paper_compare` (seed 0 of the five models),
`mixing_sequence`, `twin_vs_affine`, and `<model>_seeds` (ten seeds each).

- `seed_stats.py`: spectrum errors and attribution terms of every seed run, in
  `data/seed_stats.json` (committed).
- `plot/mixing_training.py`: candidate figures for the sections on mixture
  training and the affine decoder (`graphics/candidates/mixing/`):
  `mixing_kcurves`, `mixing_wrf`.

## Latent-transport schemes (offline replay)

`transport/` (repo root) replays the case's aerosol transport offline on WRF's
own mass fluxes with several latent-transport schemes. `notes/transport_schemes.md`
has the study.

- `paperfigs/transport_schemes.py`: loads a replay
  (`data/transport/<run>/surface.npz`), turns each scheme into a latent field,
  decodes it (`extract/affx.decode_domain`) and attributes it
  (`mixing.attribute`). The decoding term is split into "off the mixtures" and
  "decoding mixtures". It attributes against WRF-PartMC (`evaluate`) or against
  the exact answer, per-source transport of the sources' particle-resolved
  diagnostics (`evaluate_exact`, `exact_reference`).
- Replays are also ordinary registered runs (`<model>_<scheme>`; the main run
  is `twin_k16_of.0.0_sep`), from extracts written by
  `transport/extract_replay.py`, so every figure script works on them.
- `checks/transport_replay.py`: the replay against both online models.
- `plot/transport_schemes.py`: candidate figures (`graphics/candidates/transport/`):
  `schemes_errors`, `schemes_models`, `schemes_fronts`, `schemes_budget`,
  `schemes_tradeoff`.

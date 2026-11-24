# Building a WRF-AeroMELD run

These scripts build a ready-to-submit WRF-AeroMELD run of the paper's test
case from the source latents of a model: the latent initial, boundary and
emission files, the run directory and the batch script. Everything is written
into a workspace of your own.

## What gets built

WRF-AeroMELD is WRF with the aerosol carried as a latent vector of 9
components (`ltnt_var1..9`) and a number mixing ratio (`ntot`), advected as
chemical tracers (`chem_opt` 900, `emiss_opt` 21). Only the latents depend
on the model; meteorology, grid, namelist and number are the same for every
model.

- **Grid and run:** 169 × 159 × 39 cells over central California, dx = 4 km,
  starting 2010-06-17 00 UTC.
- **Background population:** scenario 561, everywhere and at the boundaries,
  `ntot` = 1e9 per kg.
- **Urban box:** population 635 at i 50–80, j 70–100, k 1–10 (1-based), with
  a Gaussian `ntot` peaking at 1e10 per kg.
- **Emissions:** population 812 at i 83–113, j 70–100, surface only, at a
  constant rate `E_NTOT` = `--rate` (default 5e7, the paper's).

Every build is checked against [reference_inputs.json](reference_inputs.json):
every non-latent field of the input files must match its SHA-256 from the
paper's runs bit for bit.

## Files

| File | Role |
|---|---|
| `build_run.py` | The driver: builds the inputs and the run directory, and checks the result. `--status` reports every run in a workspace. |
| `encode_populations.py` | Writes `latent_scenario_<s>_0001_00000013.txt` (the latents, then `ntot`) for the three populations with one model. For the paper's models, `training/wrf/write_latents.py` writes these files from the source latents instead. |
| `run_latent_ics.sh`, `make_latent_ic.py` | Fill `wrfinput_d01` and `wrfbdy_d01` from a real.exe template. |
| `run_latent_emis.sh`, `make_latent_emis.py` | Write the 13 hourly `wrfchemi_d01_*` files. |
| `region_profiles.py` | The urban box's `ntot` profile. |
| `reference.py`, `reference_inputs.json` | The checks above. |
| `make_bundle.py`, `compare_inputs.py` | Pack the input templates; compare two runs' inputs variable by variable. |
| `../vendor/` | A copy of the helper code the encoder path imports. See [../vendor/README.md](../vendor/README.md). |

## Setup

1. **Input templates.** Unpack `wrf_inputs.tar.gz` from the data deposit. It
   holds `templates/` (the real.exe output `wrfinput_d01_cares`,
   `wrfbdy_d01_cares` and `namelist.input`), `particles/` (the three
   1000-particle source populations) and `MANIFEST.json` (md5 of every
   file).
2. **WRF-AeroMELD**, compiled: release `wrf-aeromeld-v1.0` of
   [compdyn/WRF](https://github.com/compdyn/WRF) (doi:10.5281/zenodo.24167410).

   ```bash
   git clone https://github.com/compdyn/WRF.git && cd WRF && git checkout wrf-aeromeld-v1.0
   export WRF_EM_CORE=1 WRF_CHEM=1 NETCDF=/path/to/netcdf
   ./configure            # GNU (gfortran/gcc) dmpar, nesting 1
   ./compile em_real >& compile.log
   ```

   `build_run.py` copies `main/wrf.exe` from that tree once, into the
   workspace's `bin/`, and links the physics tables of `test/em_real/`.
   real.exe, WPS and `met_em` files are not needed.
3. **Python** with numpy, xarray and netCDF4 (and torch,
   [partnn](https://github.com/ehsansaleh/partnn) and PyPartMC only for
   `encode_populations.py`).
4. **Point the scripts at them**, from `scripts/`:

   ```bash
   export PAPERFIGS_INPUT_BUNDLE=/path/to/wrf_inputs
   export PAPERFIGS_WRF_DIR=/path/to/WRF
   export AEROMELD_WORK=/path/to/workspace
   ```

## Build and submit

`training/wrf/build_all.sh` is the paper's driver on Perlmutter: the source
latents of model `<name>.<seed>.0` are written to
`$AEROMELD_WORK/latents/<name>.<seed>.0/` by `training/wrf/write_latents.py`,
and then

```bash
python inputs/build_run.py <name>.<seed>.0 --batch-template my_batch.sh
```

builds `runs/latent_advection_cares_v3match_<name>.<seed>.0_5e7/`. The batch
template is a script of your own in which `{name}`, `{run}` and `{mail}` are
replaced; it must `cd {run}` and start `./wrf.exe` under MPI
(`training/wrf/batch_perlmutter.sh` is the paper's). A `CHECK FAILED` line
names the non-latent fields that differ from the paper's inputs: do not
submit until you know why. `build_run.py` never overwrites an existing run.

After the run, `python inputs/build_run.py --status` reports `SUCCESS` from
`rsl.error.0000` (WRF can fail and still exit 0). The 12-h runs of the paper
use the same inputs with `run_hours` 12 in the namelist
(`training/wrf/online_dd.sh`).

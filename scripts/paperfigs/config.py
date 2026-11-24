"""Paths and analysis choices shared by every extractor and plot script.

Only this file says where raw output lives or which instant, level and points a
figure uses. The raw paths are needed only to BUILD products; a plot script run
from existing products never touches them. Each raw path can be overridden with
the environment variable named beside it.
"""
import os
from pathlib import Path


def _path(env, default):
    return Path(os.environ.get(env, default))


# --- this repository ---------------------------------------------------------

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
REPO = SCRIPTS_DIR.parent
DATA_DIR = REPO / 'data'
PROCESSED_DIR = _path('PAPERFIGS_PROCESSED', DATA_DIR / 'processed')
REGISTRY_FILE = Path(__file__).with_name('registry.toml')

#: The paper's figures. A plot script made for the main run writes here.
PAPER_GRAPHICS = REPO / 'figures'
#: Figures for any other registered run: BY_MODEL_GRAPHICS / <run key> / <name>.pdf
BY_MODEL_GRAPHICS = REPO / 'graphics' / 'by_model'
#: Vendored vae_partmc_utils and the diagnostics it imports (vendor/README.md).
VENDOR_DIR = SCRIPTS_DIR / 'vendor'


# --- raw inputs (extractors only) --------------------------------------------

#: An input bundle (inputs/make_bundle.py): everything a WRF-AeroMELD build needs
#: from outside this repository and a WRF build -- models/, particles/ and
#: templates/. When PAPERFIGS_INPUT_BUNDLE is set, MODEL_RESDIR, PARTICLE_DIR
#: and TEMPLATE_DIR default to its subdirectories instead of the defaults below;
#: an explicit PAPERFIGS_* variable for one of them still wins.
INPUT_BUNDLE = Path(os.environ['PAPERFIGS_INPUT_BUNDLE']) if os.environ.get('PAPERFIGS_INPUT_BUNDLE') else None


def _bundled(env, sub, default):
    return _path(env, INPUT_BUNDLE / sub if INPUT_BUNDLE is not None else default)


#: code_partnn notebooks, home of the trained models
#: (22_vaepart/results/<name>_<file:02d>.h5).
PARTNN_NOTEBOOKS = _path('PAPERFIGS_PARTNN_NOTEBOOKS',
                         '/path/to/partnn/notebooks')
#: Where models are looked up by fpidx. Point it at a directory of your own
#: (symlinks to <name>_<file:02d>.h5 are fine) to use archives kept elsewhere.
MODEL_RESDIR = _bundled('PAPERFIGS_MODEL_RESDIR', 'models', PARTNN_NOTEBOOKS / '22_vaepart' / 'results')

#: WRF-ML (AeroMELD) runs: one directory per run under WRF_ML_TEST.
WRF_ML_TEST = _path('PAPERFIGS_WRF_ML_TEST',
                    '/path/to/WRF-AeroMELD/test')
WRF_ML_RUN_PREFIX = 'latent_advection_cares_v3match'
WRFOUT_NAME = 'wrfout_d01_2010-06-17_00:00:00'
WRFCHEMI_NAME = 'wrfchemi_d01_2010-06-17_00:00:00'

#: WRF-PartMC run. Its aerosols file is ~57 GB; only partmc_fields reads it.
PARTMC_RUN_DIR = _path('PAPERFIGS_PARTMC_RUN',
                       '/path/to/WRF-PartMC/test/em_real')
PARTMC_AEROSOLS = PARTMC_RUN_DIR / 'aerosols_d01_2010-06-17_00:00:00'

#: PartMC per-cell diagnostics, computed from the particles (analyze_particles).
#: DIAG_RUN names one independent run (01/02/03 are runs, not times).
DIAG_ROOT = _path('PAPERFIGS_DIAG_ROOT',
                  '/path/to/domain_diagnostics')
DIAG_RUN = '01'
DIAG_Z = 1                       # 1-based level in the diagnostics files

#: The 1000-particle populations WRF-PartMC was initialised from.
PARTICLE_DIR = _bundled('PAPERFIGS_PARTICLE_DIR', 'particles', '/path/to/particles')
PARTICLE_TIME = 13

#: The run's IC, BC and emission control files. Each lists the scenario files
#: its sources came from (num_conc_source_NNN order: ICs, then BCs, then
#: emissions); the emission file also holds num_flux for the source boxes.
IC_CONTROL = PARTMC_RUN_DIR / 'ics_from_file' / 'ic_control.nc'
BC_CONTROL = PARTMC_RUN_DIR / 'bcs_from_file' / 'bc_control.nc'
EMIS_CONTROL = PARTMC_RUN_DIR / 'emis_from_file' / 'emis_control.nc'


# --- WRF-AeroMELD input building (inputs/) -------------------------------------

#: namelist.input and the real.exe templates (the meteorology and grid):
#: wrfinput_d01_cares / wrfbdy_d01_cares for 9 latents, *_cares_l49 for 49.
#: Without a bundle these are the template run's; in a bundle, its templates/.
TEMPLATE_DIR = _bundled('PAPERFIGS_TEMPLATE_DIR', 'templates', WRF_ML_TEST / WRF_ML_RUN_PREFIX)
TEMPLATE_NAMELIST = TEMPLATE_DIR / 'namelist.input'
#: Grid, XLAT/XLONG and global attributes of the wrfchemi files. The IC template
#: has the same grid as latent_advection_cares/wrfinput_d01_cares, which the runs
#: up to 2026-11-10 used; the files differ only in the global attribute
#: USE_THETA_M (0 here, matching the namelist), which WRF never checks on
#: emission input.
EMIS_GRID_TEMPLATE = TEMPLATE_DIR / 'wrfinput_d01_cares'
#: A compiled WRF-AeroMELD tree (github compdyn/WRF, release wrf-aeromeld-v1.0): main/wrf.exe, and the
#: physics-table links WRF's compile puts in test/em_real/. inputs/build_run.py
#: copies wrf.exe into the workspace once, so a later rebuild does not change a run.
WRF_DIR = _path('PAPERFIGS_WRF_DIR', WRF_ML_TEST.parent)
WRF_EXE = WRF_DIR / 'main' / 'wrf.exe'
#: What every build is checked against (inputs/compare_inputs.py --fingerprint):
#: per-variable checksums of the non-latent fields of the reference runs, and the
#: latents of every model encoded so far. Committed; no reference run needed.
INPUT_FINGERPRINTS = SCRIPTS_DIR / 'inputs' / 'reference_inputs.json'


# --- analysis choices ---------------------------------------------------------

#: Emission rate (E_NTOT, # m^-2 s^-1) of every run the paper uses. Runs at any
#: other rate are not registered.
EMISSION_RATE = 5e7

#: Start and end of both simulations. Every domain figure is at TIMESTAMP_FINAL.
TIMESTAMP_START = '2010-06-17_00:00:00'
TIMESTAMP_FINAL = '2010-06-17_12:00:00'      # 04 UTC until 2026-11-13; now the end of the run

#: Columns of diagnostic_ML_vs_PR (CCN, frozen fraction, optics spectra).
TIMESTAMPS_SPECTRA = ('2010-06-17_00:00:00', '2010-06-17_06:00:00', '2010-06-17_12:00:00')

#: 0-based model level of every field (the surface).
LEVEL = 0

#: Sampling points as 1-BASED (i, j) = (west_east, south_north): the exact
#: indices WRF itself uses (Fortran), and those of the PartMC diagnostics files
#: and DiagnosticsIndex. Read into Python, (65, 50) is array element [49, 64].
#: Index arrays only through common.point_index().
#: 2026-11-12: chosen from per-source transport at 04 UTC (the main analysis
#: time) so that each point is a different, balanced mixture away from sharp
#: fronts (smallest 3x3 standard deviation of the source fractions among the
#: cells of its kind): A urban + background (0.39 / 0.61), B emission +
#: background (0.69 / 0.31), C all three (0.28 / 0.32 / 0.40 background /
#: urban / emission). Before: (65, 50), (65, 85), (90, 75), chosen by eye,
#: at 04 UTC 84 % urban, pure background, and a three-way mixture on a front.
#: 2026-11-13: the same rule at 12 UTC (the evaluation time since then), at least 10 cells
#: from the lateral boundaries (checks/pick_points.py): A urban + background (0.68 / 0.32),
#: B emission + background (0.37 / 0.61), C all three (0.21 / 0.37 / 0.42 background /
#: urban / emission). The 04 UTC points were (66, 25), (108, 94), (89, 77).
#: 2026-11-13 (B and C had alike speciated mass, both ~85 % emission by mass): by MASS
#: fractions (checks/pick_points.py), a mixed cell in which each source carries 50-75 % of the
#: mass and another at least 20 %: A urban (mass 0.27 / 0.73 / 0.00 background / urban /
#: emission; number 0.44 / 0.56 / 0.00), B background (mass 0.57 / 0.11 / 0.31; number 0.88 /
#: 0.08 / 0.04), C emission (mass 0.12 / 0.37 / 0.51; number 0.36 / 0.51 / 0.13).
#: 2026-11-16 (the dry rebuild): the same rule on DRY mass (water excluded: emission 3.1, not 11.8,
#: times the background's mass per particle): A urban (mass 0.19 / 0.60 / 0.21 background / urban /
#: emission; number 0.36 / 0.51 / 0.13), B background (mass 0.66 / 0.29 / 0.05; number 0.82 / 0.16 /
#: 0.02), C emission (mass 0.09 / 0.35 / 0.56; number 0.21 / 0.37 / 0.42): (104, 14), (69, 26), (122, 16).
#: 2026-11-19 (the final rebuild, SIZE_BASIS 'drydiam'): the mass with water again (in the dry-diameter
#: bins; emission 11.8 and urban 2.2 times the background's mass per particle), so the 2026-11-13
#: points and fractions above.
LOCATIONS = ((64, 54), (95, 24), (104, 14))
LOC_LABELS = ('A', 'B', 'C')

#: The paper's models (training/): <base><SUFFIX>, e.g. twin_k16_dry. Since 2026-11-16 (a clean
#: dry rebuild) the five decoders are retrained end to end on targets whose number and speciated-mass
#: size distributions are binned by DRY diameter, water excluded from the diameter and the mass
#: (training round 37, training/dataset/dry_cache.py); before, '_of' (round 33, wet diameter).
#: Since 2026-11-19 (the final choice) '_dd': round 39, targets binned by DRY diameter with EVERY
#: species' mass in the bins, water included (dry_cache.py --basis drydiam).
#: SIZE_BASIS is the reference's binning to match (extract/affx.build_populations; baselines/diag.BASES),
#: and the size basis of every scheme's numbers (AEROMELD_SIZE_BASIS in baselines/).
SUFFIX = '_dd'
SIZE_BASIS = {'_dd': 'drydiam', '_dry': 'dry', '_of': 'wet'}[SUFFIX]
#: The 12-hour offline replays of every seed (transport/rep12_array.sbatch), % seed.
REPLAY_12H = {'_dd': 'dd%d_12h', '_dry': 'dry%d_12h', '_of': 'all%d_12h'}[SUFFIX]
BASES = ('twin_none', 'twin_pairs', 'twin_k16', 'twin_k32', 'ax_rel1')


def recipe(base):
    """A decoder's base name (e.g. twin_k16) -> the paper's model name (twin_k16_dry)."""
    return base + SUFFIX


def base_name(name):
    """twin_k16_dry, twin_k16_of, twin_k16_dry.3.0 -> twin_k16 (the replay's model key)."""
    name = name.split('.')[0]
    for s in ('_dd', '_dry', '_of'):
        if name.endswith(s):
            return name[:-len(s)]
    return name


#: Species order of the VAE (vae_partmc_utils.CHEM_SPECIES, which the AeroMELD
#: extractor checks against this). PartMC's species axis is permuted into it by
#: name, never by position: the two orders differ from index 4 on.
ML_SPECIES = ('SO4', 'NO3', 'Cl', 'NH4', 'ARO1', 'ARO2', 'ALK1', 'OLE1',
              'API1', 'Na', 'OIN', 'OC', 'BC', 'MOC', 'H2O')

#: The three populations: initial-condition base, IC box, emissions.
POP_SCENARIOS = (561, 635, 812)
#: Total number concentration (m^-3) of the FULL PartMC population each one was
#: sampled from (AeroState.total_num_conc of scenario_<s>_0001_00000013.nc). The
#: particle files cannot supply it (they keep the sample's own weights, ~1/159 of
#: this); inputs/encode_populations.py turns it into the ntot latent it records.
POP_N_FULL = {561: 2863332000.0, 635: 3043998000.0, 812: 4027310156.672023}
#: Their names in legends and panel tags, in POP_SCENARIOS order.
POP_LABELS = ('Background', 'Urban', 'Emission')
#: The standardized ntot latent the populations product decodes them at (the
#: centre of the model's transformed n_tot space; see extract/populations.py).
POP_NTOT_LATENT = 0.0
#: The population each sampling point is held up against in its diagnostic panel.
POINT_SCENARIOS = (561, 635, 561)

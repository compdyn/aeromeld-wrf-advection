"""Shared helpers for driving the particle VAE against PartMC populations.

Extracted from testing_ML_code.ipynb so the same code can be reused from other
notebooks:

    from partnn.notebooks.vae_partmc_utils import main, take_wrf_output, write_latents

(or just ``import vae_partmc_utils`` when running from the notebooks directory).

The three groups of functionality are:

  * PartMC side -- rebuild an AeroState from an old-format scenario file and
    downsample it into the (n_tot, w_prt, m_prtchm) triple the encoder wants.
  * VAE side -- load encoder/decoder/preprocessors, then either encode real
    data or push a prescribed latent vector through the decoder.
  * Dump/load -- write latent vectors and CCN spectra to plain text so other
    notebooks (and Fortran) can read them back consistently.
"""

import math
import sys
import warnings
from pathlib import Path

import h5py
import numpy as np
import torch
from netCDF4 import Dataset

import PyPartMC as ppmc
from PyPartMC import si

from partnn.io_cfg import data_dir
from partnn.io_utils import resio, read_ymlhdf, get_subdict
from partnn.notebooks.n22_utils import load_riomodel, load_riopp


# =============================================================================
# Configuration
# =============================================================================

_HERE = Path(__file__).resolve().parent

# Bulk optical / frozen-fraction formulas -- same ones the decoder's qs_pop /
# qa_pop / frznfrac_tmp targets were built with. PAPER-REPO COPY: the two
# analyze_particles modules are vendored beside this file (see README.md here),
# so they are imported from this directory instead of an absolute
# analyze_particles path. That is the only change from the original.
sys.path.insert(0, str(_HERE))
# PARTICLE-RESOLVED ONLY. PartMC hands us real particles, so every reference
# curve here is computed per particle and never routed through the 19-bin
# histogram. The qs_pop / qa_pop / frznfrac_tmp training targets are all
# per-particle, and binning first overshoots by a Jensen gap: measured x1.04 /
# x2.09-3.35 / x1.41 on frozen fraction for 561/635/812, and up to x3.5 on
# absorption. Verified 2026-09-09 against the stored targets in
# data/06_prtcls/02_bwchisamp_00.h5 -- the per-particle path reproduces them to
# 0.4%, the histogram path misses by 5.6-12% (scattering) and up to 349% (qa).
from extract_diagnostics_part import (  # noqa: E402
    compute_frozen_fraction_spectrum_part,
    compute_optical_spectrum_part,
)

#: Old-format PartMC output, one directory per scenario.
SCENARIO_ROOT = Path(
    "/path/to/bw_data_chi_sampling_paper/unpacked_data"
)

#: Trained VAE to load by default.
FPIDX = "02_mlpprt.0.0"
RESDIR = _HERE / "22_vaepart" / "results"

#: Training-set HDF5 whose first sample gets overwritten with real PartMC data.
DATA_TREE = "06_prtcls/02_bwchisamp_00.h5"

#: Number of particles handed to the encoder, and species count it expects.
N_PART_SAMPLE = 1000
N_CHEM = 15

#: Diameter grid of the reconstructed number/mass histograms (n_prthst).
DIAM_LOW, DIAM_HIGH, N_BINS = 10 ** (-8.75), 1e-4, 19

#: Supersaturation grid of the reconstructed CCN spectrum (ccn_cdf).
EPS_HISTMIN, EPS_HISTMAX, N_EPSHIST = 1e-3, 1e-1, 100
EPS_HISTBINS = np.exp(
    np.linspace(math.log(EPS_HISTMIN), math.log(EPS_HISTMAX), N_EPSHIST)
)

#: Temperature grid (deg C) of the reconstructed frozen-fraction spectrum
#: (frznfrac_tmp).
TEMPERATURE_C = np.linspace(-40.0, 0.0, 100)

#: Wavelengths (m) the decoder's optics head (qs_pop/qa_pop) was evaluated at --
#: configs/03_prtcls/12_utils/15_eval.yml -> pp/aeroopt/wave_length. NOT the
#: same grid extract_diagnostics.WAVELENGTHS_M uses for its own dumps.
WAVELENGTHS_M = (300e-9, 400e-9, 500e-9, 600e-9, 700e-9, 800e-9, 900e-9, 1000e-9)

#: Canonical latent ordering: z[0..n-1] then ntot, matching WRF's ltnt_var1-9 + ntot.
LATENT_KEYS = ("z", "ntot")


def diam_centers():
    """Geometric bin centers of the n_prthst diameter grid."""
    logdiams = np.linspace(
        np.log10(DIAM_LOW), np.log10(DIAM_HIGH), N_BINS + 1, endpoint=True
    )
    return np.sqrt((10 ** logdiams[:-1]) * (10 ** logdiams[1:]))


# =============================================================================
# PartMC populations
# =============================================================================

def make_aero_data_subset():
    """The 15 species the VAE was trained on, in encoder order."""
    return ppmc.AeroData(
        (
            # density, ions in soln (1), molecular weight, kappa (1), ABIFM_m, ABIFM_c
            {"SO4": [1800 * si.kg / si.m**3, 0, 96.0 * si.g / si.mol, 0.65, 0, 0]},
            {"NO3": [1800 * si.kg / si.m**3, 0, 62.0 * si.g / si.mol, 0.65, 0, 0]},
            {"Cl": [2200 * si.kg / si.m**3, 0, 35.5 * si.g / si.mol, 1.28, 0, 0]},
            {"NH4": [1800 * si.kg / si.m**3, 0, 18.0 * si.g / si.mol, 0.65, 0, 0]},
            {"ARO1": [1400 * si.kg / si.m**3, 0, 150.0 * si.g / si.mol, 0.1, 0, 0]},
            {"ARO2": [1400 * si.kg / si.m**3, 0, 150.0 * si.g / si.mol, 0.1, 0, 0]},
            {"ALK1": [1400 * si.kg / si.m**3, 0, 140.0 * si.g / si.mol, 0.1, 0, 0]},
            {"OLE1": [1400 * si.kg / si.m**3, 0, 140.0 * si.g / si.mol, 0.1, 0, 0]},
            {"API1": [1400 * si.kg / si.m**3, 0, 184.0 * si.g / si.mol, 0.1, 0, 0]},
            {"Na": [2200 * si.kg / si.m**3, 0, 23.0 * si.g / si.mol, 1.28, 0, 0]},
            {"OIN": [2600 * si.kg / si.m**3, 0, 1.0 * si.g / si.mol, 0.1, 0, 0]},
            {"OC": [1000 * si.kg / si.m**3, 0, 1.0 * si.g / si.mol, 0.001, 0]},
            {"BC": [1800 * si.kg / si.m**3, 0, 1.0 * si.g / si.mol, 0, 0, 0]},
            {"MOC": [1000 * si.kg / si.m**3, 0, 1.0 * si.g / si.mol, 0.001, 0]},
            {"H2O": [1000 * si.kg / si.m**3, 0, 18.0 * si.g / si.mol, 0, 0, 0]},
        )
    )


#: Species order m_prtchm / m_chmprthst channels are in -- make_aero_data_subset().species.
CHEM_SPECIES = tuple(make_aero_data_subset().species)


AERO_DIST_INIT = [
    {
        "aikten": {
            "mass_frac": [{"SO4": [1]}, {"NH4": [0.375]}],
            "diam_type": "geometric",
            "mode_type": "log_normal",
            "num_conc": 3.2e9 / si.m**3,
            "geom_mean_diam": 0.02 * si.um,
            "log10_geom_std_dev": 0.161,
        },
        "accumulation": {
            "mass_frac": [{"SO4": [1]}, {"OC": [1.375]}, {"BC": [0.5]}, {"NH4": [0.375]}],
            "diam_type": "geometric",
            "mode_type": "log_normal",
            "num_conc": 2.9e9 / si.m**3,
            "geom_mean_diam": 0.16 * si.um,
            "log10_geom_std_dev": 0.217,
        },
    }
]


def make_particle_population(n_part=2000):
    """Synthetic two-mode urban-plume-ish population, for smoke tests."""
    aero_data_subset = make_aero_data_subset()
    aero_dist_init = ppmc.AeroDist(aero_data_subset, AERO_DIST_INIT)

    aero_state = ppmc.AeroState(aero_data_subset, n_part, "flat")
    aero_state.dist_sample(
        aero_dist_init,
        sample_prop=1.0,
        create_time=0.0,
        allow_doubling=True,
        allow_halving=True,
    )
    return aero_data_subset, aero_state


def scenario_path(scenario, i_time):
    """Path to an old-format PartMC output file."""
    return SCENARIO_ROOT / ("scenario_%i" % scenario) / (
        "%s.nc" % partmc_basename(scenario, i_time)
    )


def convert_old_dataset(scenario, i_time, verbose=True):
    """Rebuild (aero_data, aero_state) from a scenario's old-format PartMC file."""
    return aero_state_from_file(scenario_path(scenario, i_time), verbose=verbose)


def aero_state_from_file(filename, verbose=True):
    """Rebuild (aero_data, aero_state) from an old-format PartMC NetCDF file.

    Any file in that format works, e.g. the 1000-particle
    scenario_*_filtered_*.nc that WRF-PartMC is initialised from. Particles keep
    the file's order and its aero_num_conc weights.

    The AeroState is created via a throwaway mono dist just to get the right
    weighting, then zeroed and refilled with the real particles.
    """
    if verbose:
        print(filename)

    with Dataset(filename, "r") as ncf:
        # Reconstruct AeroData from NetCDF file
        rho = ncf.variables["aero_density"][:]
        kappa = ncf.variables["aero_kappa"][:]
        species = ncf.variables["aero_species"].names.split(",")
        mw = ncf.variables["aero_molec_weight"][:]
        ions_in_solution = ncf.variables["aero_num_ions"][:]
        aero_items = []
        for i, sp in enumerate(species):
            aero_items.append(
                {
                    sp: [
                        rho[i] * si.kg / si.m**3,
                        int(ions_in_solution[i]),
                        float(mw[i]) * si.kg / si.mol,
                        float(kappa[i]),
                        float(0.0),
                        float(0.0),
                    ]
                }
            )
        aero_data = ppmc.AeroData(aero_items)

        # Hack for weighting
        # Eventually we could consider setting the weight and magnitudes
        # weight_magnitude = ncf.variables["weight_magnitude"][:]
        # weight_exponent = ncf.variables["weight_exponent"][:]
        particle_weight = ncf.variables["aero_num_conc"][:]
        dist_init = [
            {
                "temp": {
                    "mass_frac": [{"SO4": [1]}],
                    "diam_type": "geometric",
                    "mode_type": "mono",
                    "num_conc": np.sum(particle_weight) / si.m**3,
                    "diam": 0.02 * si.um,
                },
            }
        ]
        n_part = len(particle_weight)
        aero_dist_init = ppmc.AeroDist(aero_data, dist_init)
        aero_state = ppmc.AeroState(aero_data, n_part, "flat")
        aero_state.dist_sample(
            aero_dist_init,
            sample_prop=1.0,
            create_time=0.0,
            allow_doubling=True,
            allow_halving=True,
        )

        # Zero out AeroState
        aero_state.zero()

        # Add the real particles to AeroState
        var = ncf.variables["aero_particle_mass"][:]
        for i_part in range(n_part):
            particle = ppmc.AeroParticle(
                aero_data, np.array(var[:, i_part]) / aero_data.densities
            )
            particle.weight_group = 1
            particle.weight_class = 1
            particle.new_id()
            # FIXME: We may need to set the particle source
            aero_state.add_particle(particle)

    return aero_data, aero_state


def process_data(aero_state, aero_data=None, count=N_PART_SAMPLE, indices=None):
    """Downsample an AeroState into the encoder's (n_tot, w_prt, m_prtchm, indices).

    m_prtchm: n_part x n_chem, w_prt: n_part (normalized), n_tot: total num conc.

    If aero_data is given, the per-particle masses are reordered/subset onto the
    15 species of make_aero_data_subset(); otherwise the state's own species
    order is used as-is (only correct if it already matches).

    indices: the particles to use, in this order. By default `count` of them are
    drawn at random without replacement. Pass np.arange(len(aero_state)) to take
    an already-sampled file (scenario_*_filtered_*.nc) as it is.
    """
    n_tot = aero_state.total_num_conc
    num_conc = aero_state.num_concs
    if indices is None:
        indices = np.random.choice(len(aero_state), size=count, replace=False)
    count = len(indices)

    if aero_data is None:
        spec_map = None
    else:
        included_species = make_aero_data_subset().species
        spec_map = [aero_data.spec_by_name(name) for name in included_species]

    w_prt = np.zeros(count)
    m_prtchm = []
    for i_part in range(count):
        part_masses = aero_state.particle(indices[i_part]).species_masses
        if spec_map is None:
            m_prtchm.append(part_masses)
        else:
            m_prtchm.append([part_masses[i_spec] for i_spec in spec_map])
        w_prt[i_part] = num_conc[indices[i_part]]

    w_prt = w_prt / sum(w_prt)
    return n_tot, w_prt, np.array(m_prtchm), indices


# =============================================================================
# VAE loading and inference
# =============================================================================

def load_vae_components(fpidx=FPIDX, resdir=RESDIR, device_name="cpu",
                        dtype_name="float32", seed_idx=None, verbose=True):
    """Load encoder, decoder, and their preprocessors from an HDF5 results file.

    seed_idx: if given, extract only that seed; otherwise load all seeds.

    Returns a dict with keys: encoder, decoder, enc_pp, dec_pp, ltnt_dim,
    ltnt_sigtnsfm, n_seeds, rng_seed_list, device.
    """
    name2dtype = dict(
        float64=(torch.double, torch.complex128),
        float32=(torch.float32, torch.complex64),
        float16=(torch.float16, torch.complex32),
    )
    tch_device = torch.device(device_name)
    tch_dtype, tch_cdtype = name2dtype[dtype_name]

    rio = resio(fpidx=fpidx, resdir=str(resdir))
    if verbose:
        print(f"Loading VAE components from {fpidx}...")

    enc_info = load_riomodel(rio, "enc", device_name, dtype_name, tch_device,
                             tch_dtype, seed_idx=seed_idx)
    dec_info = load_riomodel(rio, "dec", device_name, dtype_name, tch_device,
                             tch_dtype, seed_idx=seed_idx)
    enc_pp_info = load_riopp(rio, device_name, dtype_name, tch_device, tch_dtype,
                             tch_cdtype, pp_type="enc", method_loadpp="make",
                             check_mthdeq=False, seed_idx=seed_idx)
    dec_pp_info = load_riopp(rio, device_name, dtype_name, tch_device, tch_dtype,
                             tch_cdtype, pp_type="dec", method_loadpp="make",
                             check_mthdeq=False, seed_idx=seed_idx)

    components = {
        "encoder": enc_info["model"],
        "decoder": dec_info["model"],
        "enc_pp": enc_pp_info["pp"],
        "dec_pp": dec_pp_info["pp"],
        "ltnt_dim": enc_info["ltnt_dim"],
        "ltnt_sigtnsfm": enc_info["ltnt_sigtnsfm"],
        "n_seeds": len(enc_info["rng_seed_list"]),
        "rng_seed_list": enc_info["rng_seed_list"],
        "device": tch_device,
    }
    if verbose:
        print(f"  Encoder: {components['encoder'].__class__.__name__}")
        print(f"  Decoder: {components['decoder'].__class__.__name__}")
        print(f"  Latent dim: {components['ltnt_dim']}")
        print(f"  Latent sigma transform: {components['ltnt_sigtnsfm']}")
        print(f"  Number of seeds: {components['n_seeds']}")
        print(f"  RNG seeds: {components['rng_seed_list']}")
    return components


def default_device():
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def encoder_input_layers(enc_pp):
    """Input nodes the encoder preprocessor expects, or {} if undeterminable."""
    if not hasattr(enc_pp, "pp_layers"):
        return {}
    return {name: info for name, info in enc_pp.pp_layers.items()
            if info.get("type") == "input"}


def create_random_data(enc_pp, n_seeds=1, n_mb=16, device="cpu", verbose=True):
    """Random data matching the encoder's expected input format."""
    input_layers = encoder_input_layers(enc_pp)
    if not input_layers:
        if verbose:
            print("  WARNING: Could not determine input nodes, using generic random data")
        return {
            "m_prtchm": torch.randn(n_seeds, n_mb, N_PART_SAMPLE, N_CHEM, device=device),
            "w_prt": torch.rand(n_seeds, n_mb, N_PART_SAMPLE, 1, device=device),
            "n_tot": torch.rand(n_seeds, n_mb, 1, 1, device=device),
        }

    if verbose:
        print(f"  Creating random data for {len(input_layers)} input nodes:")

    named_dims = {"n_seeds": n_seeds, "n_mb": n_mb,
                  "n_part": N_PART_SAMPLE, "n_chem": N_CHEM}
    data_dict = {}
    for node, layer_info in input_layers.items():
        shape = []
        for dim_name in layer_info.get("shape", []):
            if dim_name in named_dims:
                shape.append(named_dims[dim_name])
            elif isinstance(dim_name, int):
                shape.append(dim_name)
            elif isinstance(dim_name, str) and dim_name.isdigit():
                shape.append(int(dim_name))
            else:
                shape.append(10)  # unknown dimension name -- fallback

        # Weights and counts must be positive; everything else can be negative.
        if "w_prt" in node or "weight" in node:
            data_dict[node] = torch.rand(*shape, device=device)
        elif "n_" in node:
            data_dict[node] = torch.rand(*shape, device=device).abs()
        else:
            data_dict[node] = torch.randn(*shape, device=device)

        if verbose:
            print(f"    {node}: shape {data_dict[node].shape}")
    return data_dict


def load_real_data(scenario_idx=1, time_idx=1, data_tree=DATA_TREE, n_samples=1,
                   verbose=True):
    """Load a training-set sample and overwrite entry 0 with a real PartMC population.

    The HDF5 read is only there to get correctly-shaped arrays for the other
    samples in the batch; the physics comes from the scenario file.

    Returns (data_dict, aero_data, aero_state, sampled_index).
    """
    data_path = Path(data_dir) / data_tree
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found at {data_path}")
    if verbose:
        print(f"  Loading real data from {data_path.name}...")

    with h5py.File(data_path, "r") as h5fp:
        shapevars = read_ymlhdf(h5fp, "dims")
        n_snr = shapevars.get("n_snr", 1000)
        n_t = shapevars.get("n_t", 25)

        data_attrs = read_ymlhdf(h5fp, "attrs")
        data_vars = [key for key in h5fp.keys()
                     if data_attrs.get(f"{key}/shape/str", [None])[0] == "n_pop"]

        total_samples = n_snr * n_t
        if n_samples > total_samples:
            n_samples = total_samples
            print(f"  WARNING: Requested {n_samples} samples but only "
                  f"{total_samples} available")

        # Random indices, sorted because HDF5 fancy-indexing requires it
        indices = np.sort(np.random.choice(total_samples, size=n_samples, replace=False))

        data_dict = {}
        for var_name in data_vars:
            data_dict[var_name] = h5fp[var_name][indices]
            if verbose:
                print(f"    Loaded {var_name}: shape {data_dict[var_name].shape}")

    aero_data, aero_state = convert_old_dataset(scenario_idx, time_idx, verbose=verbose)
    n_tot, w_prt, m_prtchm, sampled_index = process_data(aero_state, aero_data)

    # Lets modify just the first simulation data
    data_dict["n_tot"][0] = np.array(n_tot)
    data_dict["w_prt"][0][:] = np.array(w_prt).reshape(-1, 1)
    data_dict["m_prtchm"][0][:][:] = m_prtchm
    return data_dict, aero_data, aero_state, sampled_index


def _to_x_data(data_np, n_seeds, device):
    """numpy dict -> torch dict with a leading n_seeds dimension."""
    tch_device = torch.device(device)
    x_data = {}
    for key, val in data_np.items():
        val_torch = torch.from_numpy(val).float().to(tch_device)
        # (n_mb, ...) -> (n_seeds, n_mb, ...); -1 keeps all existing dimensions
        expand_shape = [n_seeds] + [-1] * val_torch.ndim
        x_data[key] = val_torch.unsqueeze(0).expand(*expand_shape)
    return x_data


def run_vae_inference(components, x_data, latent_vars=None, verbose=False):
    """Encode x_data, optionally override the latents, then decode.

    latent_vars: if given, an array ordered [z[0], ..., z[n-1], ntot] that
        replaces the encoded latents before decoding. This is how a WRF latent
        state (ltnt_var1-9 + ntot) is pushed through the decoder -- x_data is
        then only used to establish the shapes via enc_shaper.

    Returns dict with 'mu', 'sigma', 'z', 'x_recon'. Sampling uses epsilon=0,
    so 'z' equals 'mu' unless latent_vars overrode it.
    """
    encoder = components["encoder"]
    decoder = components["decoder"]
    enc_pp = components["enc_pp"]
    dec_pp = components["dec_pp"]
    ltnt_sigtnsfm = components["ltnt_sigtnsfm"]
    device = components.get("device", default_device())

    if verbose:
        for name, param in enc_pp.pp_params.items():
            print(name, param)

    with torch.no_grad():
        encoder.eval()
        decoder.eval()

        # 1. Apply encoder preprocessor
        u_enc, enc_shaper = enc_pp.forward(x_data, full=False)

        # 2. Encode to get mu and log-sigma
        musig_dict = encoder(**u_enc)
        mu_dict = get_subdict(musig_dict, "mu", sep=".", pop=True)
        logsigtld_dict = get_subdict(musig_dict, "sig", sep=".", pop=True)

        # Apply latent sigma transformation
        logsigma_dict = {}
        for z_node, logsigtld in logsigtld_dict.items():
            if ltnt_sigtnsfm == "exp":
                logsigma = logsigtld
            elif ltnt_sigtnsfm == "explin":
                logsigma = torch.where(logsigtld < 0, logsigtld, (logsigtld + 1).log())
            else:
                raise ValueError(f"Unknown ltnt_sigtnsfm: {ltnt_sigtnsfm}")
            logsigma_dict[z_node] = logsigma
        sigma_dict = {z_node: logsig.exp() for z_node, logsig in logsigma_dict.items()}

        # 3. Sample latent variables (using epsilon=0 for deterministic inference)
        z_dict = {z_node: mu_dict[z_node] for z_node in mu_dict}

        if latent_vars is not None:
            # Accept a single (N+1,) vector or a batched (n_mb, N+1) array, and
            # always split on the LAST axis -- slicing [0:-1] on a batch would
            # silently take rows instead of the z columns.
            lv = np.asarray(latent_vars, dtype=float)
            if lv.ndim == 1:
                lv = lv[None, :]
            lv = torch.as_tensor(lv, dtype=z_dict["z"].dtype, device=device)
            z_dict["z"] = lv[:, :-1].reshape(z_dict["z"].shape)
            z_dict["ntot"] = lv[:, -1].reshape(z_dict["ntot"].shape)

        # 4. Decode
        u_dec = decoder(**z_dict)

        # 5. Apply decoder inverse preprocessor
        x_recon = dec_pp.inverse(u_dec, enc_shaper, strict=True, full=False)

    return {"mu": mu_dict, "sigma": sigma_dict, "z": z_dict, "x_recon": x_recon}


def main(use_real_data=False, run_inference=False, scenario_idx=1, time_idx=1,
         n_mb=1, seed_idx=0, verbose=True, fpidx=FPIDX):
    """Load the VAE and encode one population.

    fpidx picks the model; the default stays FPIDX so existing callers keep
    decoding with the model their runs were built from.

    Returns (components, results, aero_data, aero_state, sampled_index); the
    last three are None when use_real_data is False.
    """
    device = default_device()
    if verbose:
        print(f"Using device: {device}")

    components = load_vae_components(fpidx=fpidx, device_name=device,
                                     dtype_name="float32", seed_idx=seed_idx,
                                     verbose=verbose)
    enc_pp = components["enc_pp"]

    if verbose:
        print("  Input layers expected by encoder preprocessor:")
        for layer_name, layer_info in encoder_input_layers(enc_pp).items():
            shape_str = " x ".join(str(s) for s in layer_info.get("shape", []))
            print(f"    - {layer_name}: {shape_str}")

    aero_data = aero_state = sampled_index = None
    if use_real_data:
        data_np, aero_data, aero_state, sampled_index = load_real_data(
            scenario_idx=scenario_idx, time_idx=time_idx, n_samples=n_mb,
            verbose=verbose)
        x_data = _to_x_data(data_np, components["n_seeds"], device)
    else:
        x_data = create_random_data(enc_pp, n_seeds=components["n_seeds"],
                                    n_mb=n_mb, device=device, verbose=verbose)

    if not run_inference:
        return components, None, aero_data, aero_state, sampled_index

    results = run_vae_inference(components, x_data)
    if verbose:
        print("  Latent variables (mu):")
        for z_node, mu in results["mu"].items():
            print(f"    {z_node}: shape {mu.shape}, mean={mu.mean().item():.4f}, "
                  f"std={mu.std().item():.4f}")
        print("  Reconstructed data:")
        for x_node, x_recon in results["x_recon"].items():
            print(f"    {x_node}: shape {x_recon.shape}")
    return components, results, aero_data, aero_state, sampled_index


def take_wrf_output(latent_vars, n_mb=None, seed_idx=0, verbose=True,
                    fpidx=FPIDX):
    """Decode a latent vector taken from WRF output.

    latent_vars is ordered [ltnt_var1..ltnt_var9, ntot], with ntot already
    standardized -- see ntot_to_latent, and prefer wrf_latents(..., components)
    which applies that conversion for you.

    n_mb is ignored: the batch size now follows the shape of latent_vars, which
    is the only value that ever worked.

    fpidx must be the model the run's inputs were encoded with.

    This loads the model on every call. To decode many latent vectors, load once
    with load_vae_components() and call decode_latents() in the loop.

    Returns (components, results).
    """
    device = default_device()
    if verbose:
        print(f"Using device: {device}")

    components = load_vae_components(fpidx=fpidx, device_name=device,
                                     dtype_name="float32", seed_idx=seed_idx,
                                     verbose=verbose)
    results = decode_latents(components, latent_vars)

    if verbose:
        print("  Reconstructed data:")
        for x_node, x_recon in results["x_recon"].items():
            print(f"    {x_node}: shape {x_recon.shape}")
    return components, results


def decode_latents(components, latent_vars, verbose=False):
    """Decode prescribed latents with an ALREADY-LOADED model.

    take_wrf_output() reloads encoder, decoder and both preprocessors from disk
    on every call, which is fine once and wasteful in a loop. This is the same
    operation with the load hoisted out, so a scan over many latent vectors
    costs one model load.

    latent_vars: (n_ltnt + 1,) or (n_mb, n_ltnt + 1), ordered [z..., ntot],
        with ntot already STANDARDIZED (see ntot_to_latent).

    Returns the usual dict with 'mu', 'sigma', 'z', 'x_recon'. Random data is
    fed to the encoder purely to establish shapes via enc_shaper.
    """
    lv = np.atleast_2d(np.asarray(latent_vars, dtype=float))
    x_data = create_random_data(
        components["enc_pp"], n_seeds=components["n_seeds"], n_mb=lv.shape[0],
        device=components.get("device", default_device()), verbose=False)
    return run_vae_inference(components, x_data, latent_vars=lv, verbose=verbose)


# =============================================================================
# The ntot latent
#
# n_tot enters the encoder preprocessor through a four-step chain (configs/
# 03_prtcls/12_utils/02_pp.yml, "Part II: The Total Count"):
#
#   t01 = n_tot + toteps            t01/value
#   t02 = sign(t01) * |t01|^totexp  t02/exponent
#   t03 = t02 - mu                  t03shftd/value
#   t04 = t03 / sigma               t04scld/value
#
# so the decoder's ntot latent is a STANDARDIZED quantity of order 1, while
# wrfout's `ntot` field and x_recon['n_tot'] are both PHYSICAL number
# concentrations of order 1e9. Feeding a physical value straight in as the 10th
# latent decodes nonsense, and nothing downstream complains.
#
# The four parameters are fitted, not configured, so they are read back out of
# the loaded preprocessor rather than hardcoded.
# =============================================================================

def ntot_params(components):
    """(eps, exponent, mu, sigma) of the n_tot preprocessing chain."""
    pp_params = components["enc_pp"].pp_params

    def _scalar(key):
        return float(np.asarray(pp_params[key].detach().cpu()).ravel()[0])

    return (_scalar("t01/value"), _scalar("t02/exponent"),
            _scalar("t03shftd/value"), _scalar("t04scld/value"))


def ntot_to_latent(n_phys, components):
    """Physical number concentration (m^-3) -> standardized ntot latent."""
    eps, exponent, mu, sigma = ntot_params(components)
    t01 = np.asarray(n_phys, dtype=float) + eps
    t02 = np.sign(t01) * np.abs(t01) ** exponent
    return (t02 - mu) / sigma


def latent_to_ntot(n_latent, components):
    """Standardized ntot latent -> physical number concentration (m^-3)."""
    eps, exponent, mu, sigma = ntot_params(components)
    t02 = np.asarray(n_latent, dtype=float) * sigma + mu
    t01 = np.sign(t02) * np.abs(t02) ** (1.0 / exponent)
    return t01 - eps


def wrf_latents(ds, i_x, i_y, i_z=0, i_time=-1, n_ltnt=9, components=None):
    """Pull [ltnt_var1..ltnt_varN, ntot] out of an opened wrfout dataset.

    components: pass the loaded VAE so the physical `ntot` field is converted
        to the standardized latent the decoder expects. Leaving it None returns
        ntot as stored -- a ~1e9 physical concentration in the slot where the
        decoder wants an order-1 latent -- which decodes silently and wrongly.
    """
    latent_vars = np.zeros(n_ltnt + 1)
    for i_var in range(n_ltnt):
        latent_vars[i_var] = ds["ltnt_var%i" % (i_var + 1)].values[i_time, i_z, i_y, i_x]
    n_tot_phys = ds["ntot"].values[i_time, i_z, i_y, i_x]

    if components is None:
        warnings.warn(
            "wrf_latents() called without `components`: ntot is returned as a "
            "physical number concentration, NOT the standardized latent the "
            "decoder expects. Pass components=... to convert.",
            stacklevel=2)
        latent_vars[-1] = n_tot_phys
    else:
        latent_vars[-1] = ntot_to_latent(n_tot_phys, components)
    return latent_vars


# =============================================================================
# Mixing
#
# Combining populations means pouring parcels together: the mixture's particles
# are the union of the contributors'. Each contributor i is described by the
# number concentration w_i it donates to the mixture, so
#
#     N_mix = sum_i w_i
#
# and the volume fraction of contributor i in the mixture is phi_i = w_i / N_i,
# where N_i is that population's own total number concentration. phi is NOT
# constrained to sum to 1: adding an emission flux to an existing parcel is
# w = (N_parcel, flux * dt), which is the same algebra with sum(phi) > 1.
#
# Two routes to a mixed diagnostic, and whether they agree is the open question
# these helpers exist to test:
#
#   mix-then-decode   blend the latents, decode once     (what WRF does)
#   decode-then-mix   decode each, blend the diagnostics (the reference)
#
# Latents are INTENSIVE, so blending them is number-weighted, matching the WRF
# advection/emission operators which form ltnt*N internally:
#
#     z_mix = sum_i w_i z_i / sum_i w_i
# =============================================================================

#: How each decoder output combines under mixing. Extensive quantities are
#: per-volume and add by volume fraction; intensive ones are per-particle
#: fractions and average by number.
DIAG_KIND = {
    "n_tot": "extensive",        # m^-3
    "n_prthst": "extensive",     # number per diameter bin, m^-3
    "m_chmprthst": "extensive",  # species mass per diameter bin, kg m^-3
    "qs_pop": "extensive",       # scattering coefficient, m^-1
    "qa_pop": "extensive",       # absorption coefficient, m^-1
    "ccn_cdf": "intensive",      # activated fraction
    "frznfrac_tmp": "intensive", # frozen fraction
}


def recon(results, name, i_mb=0):
    """One decoded diagnostic for one batch member, with the singleton axis
    dropped.

    Shapes out: n_tot (1,), ccn_cdf/frznfrac_tmp (100,), n_prthst (19,),
    qs_pop/qa_pop (8,), m_chmprthst (15, 19).
    """
    arr = np.asarray(results["x_recon"][name].detach().cpu())[0, i_mb]
    return arr[0] if (arr.ndim == 2 and arr.shape[0] == 1) else arr


def population_ntot(results, i_mb=0):
    """Reconstructed physical total number concentration (m^-3)."""
    return float(recon(results, "n_tot", i_mb).ravel()[0])


def mix_weights(results_list, weights=None):
    """Resolve mixing weights to an array of donated number concentrations.

    Default: each population donates its own total number concentration, i.e.
    equal parcel volumes (phi_i = 1 for every contributor).
    """
    if weights is None:
        return np.array([population_ntot(r) for r in results_list], dtype=float)
    weights = np.asarray(weights, dtype=float)
    if len(weights) != len(results_list):
        raise ValueError(
            f"got {len(weights)} weights for {len(results_list)} populations")
    return weights


def mix_by_number_fraction(fracs, n_total):
    """Weights for a PRESCRIBED number split, independent of the populations' own N.

    Contributor i supplies fracs[i] of n_total, so with fracs summing to 1 the
    mixture holds n_total particles per m^3 no matter what N_i the contributors
    natively have -- at fracs = (0.5, 0.5) each really does provide half the
    particles. Contrast mix_weights()'s default, which pours whole parcels
    together and lets N_mix land wherever the populations put it.

    fracs may be (n_pop,) for one mixture or (n_step, n_pop) for a sweep;
    n_total may be a scalar or one value per step.

    Note what this does to the latent vector: with n_total held constant the
    ntot latent is constant too, so the blend reduces to a plain linear
    interpolation of z and the sweep varies composition ALONE.
    """
    fracs = np.asarray(fracs, dtype=float)
    n_total = np.asarray(n_total, dtype=float)
    if n_total.ndim:
        n_total = n_total.reshape(n_total.shape + (1,) * (fracs.ndim - n_total.ndim))
    return fracs * n_total


def mix_latent_vector(results_list, components, weights=None):
    """MIX-THEN-DECODE: number-weighted latent blend -> one [z..., ntot] vector.

    The returned ntot slot is standardized, so the vector can go straight into
    decode_latents().
    """
    w = mix_weights(results_list, weights)
    z = np.stack([np.asarray(r["z"]["z"].detach().cpu()).ravel()
                  for r in results_list])
    z_mix = (w[:, None] * z).sum(axis=0) / w.sum()
    return np.concatenate([z_mix, [ntot_to_latent(w.sum(), components)]])


def mix_recon(results_list, name, weights=None):
    """DECODE-THEN-MIX: combine the decoded diagnostics directly.

    Extensive quantities add by volume fraction phi_i = w_i / N_i; intensive
    ones are averaged by donated number. This is the reference the mix-then-
    decode path is measured against.
    """
    w = mix_weights(results_list, weights)
    diags = np.array([recon(r, name) for r in results_list], dtype=float)
    kind = DIAG_KIND.get(name)

    if kind == "extensive":
        n_own = np.array([population_ntot(r) for r in results_list])
        phi = w / n_own
        coeff = phi
    elif kind == "intensive":
        coeff = w / w.sum()
    else:
        raise KeyError(
            f"{name!r} is not in DIAG_KIND; add it and say whether it is "
            "extensive (per volume) or intensive (a per-particle fraction)")

    return (coeff.reshape((-1,) + (1,) * (diags.ndim - 1)) * diags).sum(axis=0)


def mix_intensive(results_list, components, fracs, name="ccn_cdf", n_total=None):
    """Mix at a PRESCRIBED number split and return one INTENSIVE diagnostic
    by both routes.

    The "ignore what the contributors natively hold" mixture: frac[i] is the
    share of the mixture's PARTICLES that population i supplies, so at
    (0.5, 0.5) each really donates half whatever number concentration the two
    populations happen to carry. Contrast mix_weights()'s default, which pours
    whole parcels together and lets the richer population dominate.
    Rows are normalized to sum to 1, which is what makes them fractions; pass
    donated concentrations to
    mix_latent_vector(weights=...) directly if you want sum(phi) != 1 (the
    emission case).

    fracs: (n_pop,) for a single mixture or (n_step, n_pop) for a sweep.
    name: any intensive DIAG_KIND entry -- "ccn_cdf" (activated fraction vs
        supersaturation, the default) or "frznfrac_tmp" (frozen fraction vs
        temperature). Extensive diagnostics are rejected, because for those
        "take the fractions of the two spectra" is the wrong algebra: they
        combine by volume fraction phi = w/N, not by number fraction.
    n_total: the mixture's number concentration, m^-3. Default is the
        fraction-weighted blend of the contributors' own N, so a pure endpoint
        reproduces that population's own concentration. The choice cannot
        change the answer here -- the decoder factorizes exactly in n_tot, so
        an intensive output is independent of it (`mtd`/`dtm` below are
        bit-stable across n_total; only the `_conc` pair moves).

    Returns a dict of

        fracs    (n_step, n_pop)  the normalized split actually used
        weights  (n_step, n_pop)  donated number concentrations, m^-3
        n_total  (n_step,)        each mixture's N
        latent   (n_step, n_ltnt+1)  the blended latent vectors handed to the
                                  decoder, [z..., ntot] with ntot standardized
        decoded                   the single batched decode_latents() result
        members  (n_pop, n_grid)  each contributor's own decoded spectrum
        mtd      (n_step, n_grid) MIX-THEN-DECODE: blend z, decode once
        dtm      (n_step, n_grid) DECODE-THEN-MIX: sum_i frac_i * spectrum_i
        mtd_conc, dtm_conc        the same two as a CONCENTRATION (m^-3) --
                                  activated or frozen particles per volume --
                                  each on its own route's N
        rel      (n_step,)        relative L2 of mtd against dtm

    `dtm` is the physics reference and is exact given the members' spectra.
    Both supported diagnostics are per-PARTICLE properties: a particle's
    critical supersaturation and its freezing temperature are set by what that
    particle is made of, and pouring two parcels together changes neither. So
    the mixture's activated (or frozen) fraction is precisely the number-
    weighted mean of the parts. `mtd` is what WRF computes when it advects and
    mixes ltnt_var1-9. The gap between them is the decoder's nonlinearity in
    the latents, and nothing else.

    Relative L2 is norm-dominated, which suits ccn_cdf (order 0.1-1 across the
    grid) far better than frznfrac_tmp (which spans decades, so `rel` reports
    essentially only the cold end where the fraction is largest). For frozen
    fraction read the per-point ratio, not just `rel`.
    """
    if DIAG_KIND.get(name) != "intensive":
        raise ValueError(
            f"mix_intensive() blends by number fraction, which is only correct "
            f"for an intensive diagnostic; {name!r} is "
            f"{DIAG_KIND.get(name, 'unknown')!r}. Use mix_recon() for those.")

    fracs = np.atleast_2d(np.asarray(fracs, dtype=float))
    if fracs.shape[1] != len(results_list):
        raise ValueError(f"got {fracs.shape[1]} fractions per row for "
                         f"{len(results_list)} populations")
    row_sum = fracs.sum(axis=1)
    if np.any(row_sum <= 0):
        raise ValueError("every row of fracs must have a positive sum")
    fracs = fracs / row_sum[:, None]

    n_own = np.array([population_ntot(r) for r in results_list])
    totals = (fracs * n_own).sum(axis=1) if n_total is None else np.broadcast_to(
        np.asarray(n_total, dtype=float), (len(fracs),)).copy()

    weights = mix_by_number_fraction(fracs, totals)
    latent = np.stack([mix_latent_vector(results_list, components, weights=w)
                       for w in weights])
    decoded = decode_latents(components, latent)

    members = np.array([recon(r, name) for r in results_list], dtype=float)
    mtd = np.stack([recon(decoded, name, i_mb=i) for i in range(len(fracs))])
    dtm = fracs @ members
    n_mtd = np.array([population_ntot(decoded, i_mb=i) for i in range(len(fracs))])

    return dict(
        fracs=fracs, weights=weights, n_total=totals, latent=latent,
        decoded=decoded, members=members, mtd=mtd, dtm=dtm,
        mtd_conc=mtd * n_mtd[:, None], dtm_conc=dtm * totals[:, None],
        n_decoded=n_mtd,
        rel=np.linalg.norm(mtd - dtm, axis=1)
            / np.maximum(np.linalg.norm(dtm, axis=1), 1e-30),
    )


def mix_ccn(results_list, components, fracs, n_total=None):
    """mix_intensive() on ccn_cdf -- the activated-fraction spectrum."""
    return mix_intensive(results_list, components, fracs, name="ccn_cdf",
                         n_total=n_total)


def mix_frznfrac(results_list, components, fracs, n_total=None):
    """mix_intensive() on frznfrac_tmp -- the frozen-fraction spectrum."""
    return mix_intensive(results_list, components, fracs, name="frznfrac_tmp",
                         n_total=n_total)


# =============================================================================
# Dumping latents and CCN spectra
# =============================================================================

def partmc_basename(scenario, i_time):
    """Matches the PartMC file opened in convert_old_dataset()."""
    return "scenario_%i_0001_%08i" % (scenario, i_time)


def latent_vector(results, key="z"):
    """Flatten a results dict into [z[0], ..., z[n-1], ntot].

    key: "z" for the sampled latents, "mu" / "sigma" for the posterior.
    """
    return np.concatenate(
        [np.asarray(results[key][k].detach().cpu(), dtype=np.float64).ravel()
         for k in LATENT_KEYS]
    )


def latent_filename(scenario, i_time, key="z", outdir="."):
    suffix = "" if key == "z" else "_" + key
    return Path(outdir) / ("latent%s_%s.txt" % (suffix, partmc_basename(scenario, i_time)))


def write_latents(results, scenario, i_time, key="z", outdir=".", verbose=True):
    """Write one latent value per line, bare numbers (Fortran list-directed safe)."""
    vec = latent_vector(results, key)
    out_name = latent_filename(scenario, i_time, key, outdir)
    with open(out_name, "w") as f:
        for value in vec:
            f.write("%.17g\n" % value)
    if verbose:
        print("Wrote", out_name, "(%i values)" % len(vec))
    return out_name


def read_latents(scenario, i_time, key="z", outdir="."):
    """Companion reader -- returns the same [z..., ntot] vector."""
    return np.loadtxt(latent_filename(scenario, i_time, key, outdir))


def ccn_env_state(rel_humidity=0.95, temperature=290, pressure=1e5):
    """Env state used for crit_rel_humids."""
    env_state = ppmc.EnvState(
        {
            "rel_humidity": rel_humidity,
            "latitude": 0,
            "longitude": 0,
            "altitude": 0 * si.m,
            "start_time": 21600 * si.s,
            "start_day": 200,
        }
    )
    env_state.set_temperature(temperature)
    env_state.pressure = pressure
    return env_state


def ccn_cdf_partmc(aero_state, env_state=None, subset=None):
    """Activated number conc vs supersaturation. Returns (cdf, total_conc).

    Vectorized form of the i_sat/i_part loops: searchsorted with side="left"
    reproduces the strict `crit_rel_humids[i] - 1 < bin` test exactly.

    subset: particle indices to restrict to (e.g. sampled_index), or None for all.
    """
    if env_state is None:
        env_state = ccn_env_state()
    s_crit = np.asarray(aero_state.crit_rel_humids(env_state)) - 1.0
    num_concs = np.asarray(aero_state.num_concs)
    if subset is not None:
        s_crit, num_concs = s_crit[subset], num_concs[subset]

    order = np.argsort(s_crit, kind="stable")
    cum = np.cumsum(num_concs[order])
    idx = np.searchsorted(s_crit[order], EPS_HISTBINS, side="left")
    return np.where(idx > 0, cum[np.maximum(idx - 1, 0)], 0.0), num_concs.sum()


def ccn_cdf_ml(results):
    """Reconstructed activated fraction on EPS_HISTBINS."""
    return np.asarray(results["x_recon"]["ccn_cdf"].detach().cpu())[0, 0, 0]


def ccn_filename(scenario, i_time, outdir=".", totals=False):
    stem = "ccn_totals_" if totals else "ccn_"
    return Path(outdir) / (stem + partmc_basename(scenario, i_time) + ".txt")


def write_ccn(results, aero_state, sampled_index, scenario, i_time, outdir=".",
              env_state=None, verbose=True):
    """Write the CCN spectrum as four columns, plus a companion totals file.

    Columns: supersaturation, ML fraction, PartMC full-state fraction, PartMC
    sampled-subset fraction -- the three curves compared in the notebook.
    Totals file holds n_tot_ml, n_tot_partmc_full, n_tot_partmc_subset so the
    fractions can be converted back to absolute concentrations.
    """
    if env_state is None:
        env_state = ccn_env_state()

    frac_ml = ccn_cdf_ml(results)
    n_tot_ml = float(np.asarray(results["x_recon"]["n_tot"].detach().cpu()).ravel()[0])
    cdf_full, tot_full = ccn_cdf_partmc(aero_state, env_state)
    cdf_sub, tot_sub = ccn_cdf_partmc(aero_state, env_state, subset=sampled_index)

    out_name = ccn_filename(scenario, i_time, outdir)
    cols = np.column_stack([EPS_HISTBINS, frac_ml, cdf_full / tot_full, cdf_sub / tot_sub])
    with open(out_name, "w") as f:
        for row in cols:
            f.write("%.17g %.17g %.17g %.17g\n" % tuple(row))

    tot_name = ccn_filename(scenario, i_time, outdir, totals=True)
    with open(tot_name, "w") as f:
        for value in (n_tot_ml, tot_full, tot_sub):
            f.write("%.17g\n" % value)

    if verbose:
        print("Wrote", out_name, "(%i x %i)" % cols.shape)
        print("Wrote", tot_name)
    return out_name


def read_ccn(scenario, i_time, outdir="."):
    """Returns (supersat, frac_ml, frac_partmc_full, frac_partmc_subset), totals."""
    cols = np.loadtxt(ccn_filename(scenario, i_time, outdir))
    totals = np.loadtxt(ccn_filename(scenario, i_time, outdir, totals=True))
    return cols.T, totals


#: Width of one n_prthst bin in log10(D). The grid is uniform in log space.
LOG10_BIN_WIDTH = (math.log10(DIAM_HIGH) - math.log10(DIAM_LOW)) / N_BINS

#: n_prthst is NOT a density -- aerometrics builds it with a plain scatter_add of
#: per-particle number concentrations into diameter bins, with no division by bin
#: width. It is dN per bin (m^-3), and sum(n_prthst) reproduces the population's
#: total number concentration. Divide by the bin width to get dN/dlog10(D).
#:
#: Watch out for plot_results: it compares n_prthst against
#: histogram_1d(..., num_concs / ln(10)). histogram_1d returns dN/dln(D), so the
#: conversion to a per-bin count is a MULTIPLY by the ln bin width
#: (ln(10) * LOG10_BIN_WIDTH = 0.5756), not a divide by ln(10) (0.4343). Those
#: differ by a factor of 1.3257, which is why that PartMC curve sits ~33% below
#: the reconstruction. The functions below use the per-bin-count basis instead.
DNDLOG10_FROM_NPRTHST = 1.0 / LOG10_BIN_WIDTH


def size_dist_ml(results):
    """Reconstructed number per diameter bin (m^-3), on diam_centers().

    Absolute, not a normalized shape: sum() reproduces the population's total
    number concentration. See DNDLOG10_FROM_NPRTHST before labelling an axis.
    """
    return np.asarray(results["x_recon"]["n_prthst"].detach().cpu())[0, 0, 0]


def bin_sum(diams, weights):
    """Plain weighted sum into the n_prthst diameter bins -- no width division.

    This is what aerometrics does with scatter_add, and it is the only binning
    used against reconstructed histograms here. Deliberately not
    ppmc.histogram_1d: that returns a DENSITY (dN/dln(D)), and converting it to
    the per-bin basis needs a multiply by the ln bin width, not a divide by
    ln(10) -- see DNDLOG10_FROM_NPRTHST.
    """
    edges = np.logspace(np.log10(DIAM_LOW), np.log10(DIAM_HIGH), N_BINS + 1)
    counts, _ = np.histogram(np.asarray(diams), bins=edges,
                             weights=np.asarray(weights))
    return counts


def sample_scale(aero_state, sampled_index):
    """Factor putting a sampled-subset EXTENSIVE diagnostic on the full-population
    basis, i.e. N_full / N_sample.

    The basis trap this exists to close: the encoder is handed normalized
    weights plus the full population's n_tot, so every absolute reconstruction
    (n_prthst, m_chmprthst, qs_pop, qa_pop) comes back scaled to the WHOLE
    population -- sum(size_dist_ml(...)) reproduces aero_state.total_num_conc.
    But size_dist_partmc(state, subset=idx) and its siblings weight by the raw
    num_concs of the 1000 sampled particles, which total only ~1/159 of that.

    Comparing the two directly is off by that factor, ~2.2 decades: measured on
    scenario 561, ml vs raw subset gives a median ratio of 179 and a relative L2
    of 156, against 1.13 and 0.25 once rescaled. It looks like catastrophic
    model failure and is arithmetic.

    So the reference to score a reconstruction against is the SAMPLE, rescaled:
    the sample's shape carrying the population's magnitude. The unscaled full
    population is a different question -- it contains coarse particles the
    encoder was never shown (median ratio 0.53 here).

    Intensive diagnostics (ccn_cdf, frznfrac_tmp) are fractions, so the factor
    cancels and they must NOT be rescaled -- see DIAG_KIND.
    """
    num_concs = np.asarray(aero_state.num_concs)
    return float(num_concs.sum() / num_concs[sampled_index].sum())


def size_dist_partmc(aero_state, subset=None):
    """PartMC number per diameter bin (m^-3), matching the n_prthst basis.

    Directly comparable to size_dist_ml() bin for bin, and sums to the
    population's total number concentration.

    Diameters are aero_state.diameters(), i.e. WET -- the encoder's 15-species
    subset includes H2O, so the reconstruction is a wet-diameter histogram and
    a dry-diameter PartMC curve would not line up bin for bin.

    subset: particle indices to restrict to (e.g. sampled_index), or None.
    """
    diams = np.asarray(aero_state.diameters())
    num_concs = np.asarray(aero_state.num_concs)
    if subset is not None:
        diams, num_concs = diams[subset], num_concs[subset]
    return bin_sum(diams, num_concs)


def size_dist_filename(scenario, i_time, outdir="."):
    return Path(outdir) / ("size_dist_" + partmc_basename(scenario, i_time) + ".txt")


def write_size_dist(results, aero_state, sampled_index, scenario, i_time,
                    outdir=".", verbose=True):
    """Write the number size distribution as four columns.

    Columns: diameter bin center (m), ML reconstruction, PartMC full state,
    PartMC sampled subset -- the same layout as write_ccn, all on the
    diam_centers() grid, wet diameter, as number per bin in m^-3
    (multiply by DNDLOG10_FROM_NPRTHST for dN/dlog10(D)).

    Without this dump the reconstructed size distribution exists only inside a
    live `results` dict, so anything downstream has to re-run the decoder just
    to plot it.
    """
    out_name = size_dist_filename(scenario, i_time, outdir)
    cols = np.column_stack([
        diam_centers(),
        size_dist_ml(results),
        size_dist_partmc(aero_state),
        size_dist_partmc(aero_state, subset=sampled_index),
    ])
    with open(out_name, "w") as f:
        for row in cols:
            f.write("%.17g %.17g %.17g %.17g\n" % tuple(row))

    if verbose:
        print("Wrote", out_name, "(%i x %i)" % cols.shape)
    return out_name


def read_size_dist(scenario, i_time, outdir="."):
    """Returns (diam_center, dNdlogD_ml, dNdlogD_partmc_full, dNdlogD_subset)."""
    return np.loadtxt(size_dist_filename(scenario, i_time, outdir)).T


# =============================================================================
# Speciated mass distribution, frozen fraction, optics
#
# These three share one PartMC-truth path: bin the (sub)state onto the same
# diam_centers() / CHEM_SPECIES grid write_size_dist already uses, then run it
# through the exact bulk formulas the decoder's m_chmprthst / frznfrac_tmp /
# qs_pop+qa_pop targets were built with (analyze_particles/extract_diagnostics
# .get_bulkoptical / .get_bulkfrznfrac), rather than re-deriving them here.
# =============================================================================

def mass_hist_partmc(aero_state, subset=None):
    """PartMC per-species mass per diameter bin (kg m^-3), matching m_chmprthst.

    Shape (len(CHEM_SPECIES), N_BINS), row order CHEM_SPECIES, same wet-diameter
    grid and per-bin-total (not density) basis as size_dist_partmc().
    """
    diams = np.asarray(aero_state.diameters())
    num_concs = np.asarray(aero_state.num_concs)
    if subset is not None:
        diams, num_concs = diams[subset], num_concs[subset]
    rows = []
    for name in CHEM_SPECIES:
        mass_per_part = np.asarray(aero_state.masses(include=[name]))
        if subset is not None:
            mass_per_part = mass_per_part[subset]
        rows.append(bin_sum(diams, mass_per_part * num_concs))
    return np.array(rows)


def _diagnostic_prt_tensors(aero_state, subset=None):
    """Per-particle (m_prtchm, n_prt) for extract_diagnostics_part's
    compute_*_part functions, as one population (n_seeds = n_pop = 1).

    Shapes (1, n_part, n_chem) and (1, n_part, 1); the _part helpers unsqueeze a
    leading seed axis onto anything 3-D. Species columns follow CHEM_SPECIES, so
    the reference is subset to the same 15 species the encoder sees.

    There is deliberately no histogram anywhere in this path -- see the module
    header. This replaced _diagnostic_hist_tensors(), whose only two callers
    were frznfrac_partmc() and optics_partmc(), both on the wrong basis.
    """
    num_concs = np.asarray(aero_state.num_concs)
    masses = np.array([np.asarray(aero_state.masses(include=[name]))
                       for name in CHEM_SPECIES])              # (n_chem, n_part)
    if subset is not None:
        num_concs = num_concs[subset]
        masses = masses[:, subset]
    m_prtchm = torch.as_tensor(masses.T[np.newaxis, :, :], dtype=torch.float64)
    n_prt = torch.as_tensor(num_concs[np.newaxis, :, np.newaxis],
                            dtype=torch.float64)
    return m_prtchm, n_prt


def mass_dist_filename(scenario, i_time, outdir="."):
    return Path(outdir) / ("mass_dist_" + partmc_basename(scenario, i_time) + ".txt")


def mass_dist_ml(results):
    """Reconstructed per-species mass per diameter bin (kg m^-3).

    Same shape/basis/row-order as mass_hist_partmc().
    """
    return np.asarray(results["x_recon"]["m_chmprthst"].detach().cpu())[0, 0]


def write_mass_dist(results, aero_state, sampled_index, scenario, i_time,
                    outdir=".", verbose=True):
    """Write the per-species mass size distribution.

    One row per diameter bin: diam center (m), then for each CHEM_SPECIES
    entry three columns -- ML reconstruction, PartMC full state, PartMC
    sampled subset -- generalizing write_size_dist's layout over species.
    """
    ml = mass_dist_ml(results)
    full = mass_hist_partmc(aero_state)
    sub = mass_hist_partmc(aero_state, subset=sampled_index)

    out_name = mass_dist_filename(scenario, i_time, outdir)
    cols = [diam_centers()]
    for i_spec in range(len(CHEM_SPECIES)):
        cols += [ml[i_spec], full[i_spec], sub[i_spec]]
    cols = np.column_stack(cols)
    header = "diam_center(m) " + " ".join(
        "%s_ml %s_full %s_sub" % (s, s, s) for s in CHEM_SPECIES)
    with open(out_name, "w") as f:
        f.write("# " + header + "\n")
        for row in cols:
            f.write(" ".join("%.17g" % v for v in row) + "\n")

    if verbose:
        print("Wrote", out_name, "(%i x %i)" % cols.shape)
    return out_name


def read_mass_dist(scenario, i_time, outdir="."):
    """Returns (diam_center, ml, full, sub), each shaped (n_chem, N_BINS)."""
    raw = np.loadtxt(mass_dist_filename(scenario, i_time, outdir))
    diam_center = raw[:, 0]
    per_species = raw[:, 1:].T.reshape(len(CHEM_SPECIES), 3, -1)
    return diam_center, per_species[:, 0, :], per_species[:, 1, :], per_species[:, 2, :]


def frznfrac_filename(scenario, i_time, outdir="."):
    return Path(outdir) / ("frznfrac_" + partmc_basename(scenario, i_time) + ".txt")


def frznfrac_ml(results):
    """Reconstructed frozen fraction on TEMPERATURE_C."""
    return np.asarray(results["x_recon"]["frznfrac_tmp"].detach().cpu())[0, 0, 0]


def frznfrac_partmc(aero_state, subset=None):
    """Bulk frozen fraction vs TEMPERATURE_C, computed PER PARTICLE.

    frznfrac_tmp's training target is per-particle, so this is the basis the
    decoder head has to be compared against. The previous implementation binned
    into 19 diameter bins first and ran high by x1.04 / x2.09-3.35 / x1.41 for
    561 / 635 / 812 -- a Jensen gap, since surf_oin collapses to
    pi*(6/pi)^(2/3) * v_oin^(2/3), concave in the particle's own INP volume.
    """
    m_prtchm, n_prt = _diagnostic_prt_tensors(aero_state, subset)
    result = compute_frozen_fraction_spectrum_part(
        m_prtchm, n_prt, list(CHEM_SPECIES),
        tmprtr_inpmin=float(TEMPERATURE_C[0]), tmprtr_inpmax=float(TEMPERATURE_C[-1]),
        n_tmprtr=len(TEMPERATURE_C))
    return np.asarray(result["frznfrac_tmp"])[0]


def write_frznfrac(results, aero_state, sampled_index, scenario, i_time,
                   outdir=".", verbose=True):
    """Write frozen fraction vs temperature: ML reconstruction, PartMC sampled
    subset. THREE columns.

    The PartMC reference is the sampled 1000 particles -- the same population
    the encoder was handed -- evaluated per particle. The old `full` column
    (whole ~159k state) was dropped 2026-09-09: the reference is the sample, and
    a full-population curve answers a different question from the one these
    dumps exist to answer.
    """
    out_name = frznfrac_filename(scenario, i_time, outdir)
    cols = np.column_stack([
        TEMPERATURE_C,
        frznfrac_ml(results),
        frznfrac_partmc(aero_state, subset=sampled_index),
    ])
    with open(out_name, "w") as f:
        for row in cols:
            f.write("%.17g %.17g %.17g\n" % tuple(row))

    if verbose:
        print("Wrote", out_name, "(%i x %i)" % cols.shape)
    return out_name


def read_frznfrac(scenario, i_time, outdir="."):
    """Returns (temperature_C, frznfrac_ml, frznfrac_subset).

    Three columns since 2026-09-09; files dumped before that have a fourth
    (PartMC full) between ml and subset and must be regenerated.
    """
    return np.loadtxt(frznfrac_filename(scenario, i_time, outdir)).T


def optics_filename(scenario, i_time, outdir="."):
    return Path(outdir) / ("optics_" + partmc_basename(scenario, i_time) + ".txt")


def optics_ml(results):
    """Reconstructed (scattering, absorption) vs WAVELENGTHS_M, in m^-1."""
    qs = np.asarray(results["x_recon"]["qs_pop"].detach().cpu())[0, 0, 0]
    qa = np.asarray(results["x_recon"]["qa_pop"].detach().cpu())[0, 0, 0]
    return qs, qa


def optics_partmc(aero_state, subset=None):
    """Bulk (scattering, absorption) vs WAVELENGTHS_M, in m^-1, computed PER
    PARTICLE.

    qs_pop / qa_pop training targets are per-particle: recomputing them through
    the 19-bin histogram misses by 5.6-12% on scattering and by up to x3.5 on
    absorption, while the per-particle path reproduces them to 0.4%. Absorption
    is the worse of the two because BC absorption is strongly nonlinear in BC
    content and BC is externally mixed -- exactly the Jensen setup.

    Refractive indices come from DEFAULT_CHEM_REFRACTIVE inside the _part helper
    (it takes no refr argument). That is the same table the old module-level
    _REFR_WVCHM was built from, so dropping it changes nothing.

    COST -- read this before assuming Mie is the expensive part. Measured
    2026-09-11 on a 96-thread node, 1000 particles x 8 wavelengths:

        torch.set_num_threads(1)    1.1 s
        default (96 threads)      318 s      <- 290x slower, bit-identical output

    It is thread oversubscription, not the Mie series. The size-parameter loop
    is a long chain of SMALL tensor ops, so torch opens a 96-thread parallel
    region per op while the interactive session's cgroup grants only 0.75 of a
    core (/sys/fs/cgroup/user.slice/user-<uid>.slice/cpu.max = "75000 100000").
    The threads cannot run concurrently and burn the budget spinning on
    barriers. Cost is linear in n_part either way (~1 ms/particle at 1 thread).

    So call torch.set_num_threads(1) before this -- do not silently do it here,
    since that is the caller's global state. A full ~159k population is then
    ~3 min rather than ~14 h, though the sample is the intended reference.
    """
    m_prtchm, n_prt = _diagnostic_prt_tensors(aero_state, subset)
    result = compute_optical_spectrum_part(
        m_prtchm, n_prt, list(CHEM_SPECIES), wavelengths=list(WAVELENGTHS_M))
    return np.asarray(result["qs_pop"])[0], np.asarray(result["qa_pop"])[0]


def write_optics(results, aero_state, sampled_index, scenario, i_time,
                 outdir=".", verbose=True):
    """Write scattering/absorption vs WAVELENGTHS_M: ML reconstruction, PartMC
    sampled subset. FIVE columns.

    The PartMC reference is the sampled 1000 particles, per particle. The old
    `full` columns were dropped 2026-09-09: per-particle Mie on a whole ~159k
    state is ~17 h on one core, and the sample is the reference anyway.
    """
    qs_ml, qa_ml = optics_ml(results)
    qs_sub, qa_sub = optics_partmc(aero_state, subset=sampled_index)

    out_name = optics_filename(scenario, i_time, outdir)
    cols = np.column_stack([WAVELENGTHS_M, qs_ml, qs_sub, qa_ml, qa_sub])
    header = "wavelength_m qs_ml qs_sub qa_ml qa_sub"
    with open(out_name, "w") as f:
        f.write("# " + header + "\n")
        for row in cols:
            f.write(" ".join("%.17g" % v for v in row) + "\n")

    if verbose:
        print("Wrote", out_name, "(%i x %i)" % cols.shape)
    return out_name


def read_optics(scenario, i_time, outdir="."):
    """Returns (wavelength_m, qs_ml, qs_sub, qa_ml, qa_sub).

    Five columns since 2026-09-09; files dumped before that are seven-column
    (qs_ml, qs_full, qs_sub, qa_ml, qa_full, qa_sub) and must be regenerated.
    """
    return np.loadtxt(optics_filename(scenario, i_time, outdir), comments="#").T


def write_population(results, aero_state, sampled_index, scenario, i_time,
                     outdir=".", verbose=True):
    """Dump everything for one population: z, mu, sigma, CCN, size distribution,
    mass distribution, frozen fraction, optics."""
    for key in ("z", "mu", "sigma"):
        write_latents(results, scenario, i_time, key, outdir, verbose=verbose)
    write_ccn(results, aero_state, sampled_index, scenario, i_time, outdir,
              verbose=verbose)
    write_size_dist(results, aero_state, sampled_index, scenario, i_time, outdir,
                    verbose=verbose)
    write_mass_dist(results, aero_state, sampled_index, scenario, i_time, outdir,
                    verbose=verbose)
    write_frznfrac(results, aero_state, sampled_index, scenario, i_time, outdir,
                   verbose=verbose)
    write_optics(results, aero_state, sampled_index, scenario, i_time, outdir,
                verbose=verbose)


# =============================================================================
# Plotting
# =============================================================================

def create_label_box(axes, string):
    """Re-export of fig_palettes.create_label_box.

    This module carried its own byte-identical copy; kept as a shim so existing
    callers keep working, but the definition now lives in one place.
    """
    from fig_palettes import create_label_box as _create_label_box
    return _create_label_box(axes, string)


def plot_results(results, aero_state, aero_data, sampled_index,
                 species=("SO4", "NH4", "OC", "BC")):
    """Number and per-species mass distributions: reconstruction vs PartMC."""
    import matplotlib
    import matplotlib.pyplot as plt

    aero_data_subset = make_aero_data_subset()
    fig = plt.figure(figsize=(20, 4))
    ncols = len(species) + 1
    spec2 = matplotlib.gridspec.GridSpec(ncols=ncols, nrows=1, figure=fig,
                                         wspace=0.3, hspace=0.2, bottom=0.2)

    # n_prthst / m_chmprthst are per-bin totals, so the PartMC curves are plain
    # weighted bin sums on the same grid. The previous ppmc.histogram_1d(...) /
    # ln(10) form mixed bases and put these curves a factor of 1.3257 low.
    centers = diam_centers()
    diams = aero_state.diameters()
    num_concs = aero_state.num_concs

    f1 = fig.add_subplot(spec2[0])
    data = results["x_recon"]["n_prthst"].cpu()
    f1.plot(centers, data[0, 0, 0], label="ML reconstruction")
    f1.plot(centers, bin_sum(diams, num_concs), label="Aerosol population")
    f1.set_xscale("log")
    create_label_box(f1, "Number")
    f1.legend()

    data = results["x_recon"]["m_chmprthst"].cpu()
    for i_spec in range(1, ncols):
        f1 = fig.add_subplot(spec2[i_spec])
        name = species[i_spec - 1]
        idx = aero_data_subset.spec_by_name(name)
        f1.plot(centers, data[0, 0, idx, :], label="ML reconstruction")
        mass_concs = np.array(aero_state.masses(include=[name])) * np.array(num_concs)
        f1.plot(centers, bin_sum(diams, mass_concs), label="Aerosol population")
        f1.set_xscale("log")
        create_label_box(f1, name)
    return fig


def plot_ccn(results=None, aero_state=None, sampled_index=None, env_state=None,
             axes=None, label_prefix=""):
    """CCN activated fraction: ML reconstruction and/or PartMC truth."""
    import matplotlib.pyplot as plt

    if axes is None:
        _, axes = plt.subplots()
    if results is not None:
        axes.plot(EPS_HISTBINS, ccn_cdf_ml(results),
                  label=(label_prefix + "ML reconstruction").strip())
    if aero_state is not None:
        cdf, total = ccn_cdf_partmc(aero_state, env_state)
        axes.plot(EPS_HISTBINS, cdf / total, "r",
                  label=(label_prefix + "Total aerosol state").strip())
        if sampled_index is not None:
            cdf_sub, tot_sub = ccn_cdf_partmc(aero_state, env_state, subset=sampled_index)
            axes.plot(EPS_HISTBINS, cdf_sub / tot_sub, "r--",
                      label=(label_prefix + "Subset aerosol state").strip())
    axes.set_xscale("log")
    axes.set_xlabel("Environmental supersaturation")
    axes.set_ylabel("Activated fraction")
    axes.grid(True)
    axes.legend()
    return axes


def compare_two_states(aero_state_a, aero_state_b, aero_data=None, env_state=None,
                       labels=("State A", "State B")):
    """CCN spectra of two populations on one axis."""
    import matplotlib.pyplot as plt

    if env_state is None:
        env_state = ccn_env_state()
    for aero_state, label, color in zip((aero_state_a, aero_state_b), labels, "rb"):
        cdf, total = ccn_cdf_partmc(aero_state, env_state)
        plt.plot(EPS_HISTBINS, cdf / total, color, label=label)
    plt.xscale("log")
    plt.xlabel("Environmental supersaturation")
    plt.ylabel("Activated fraction")
    plt.legend()

    print("SS:",
          np.sum(np.array(aero_state_a.masses(include=["Na"])) * np.array(aero_state_a.num_concs)),
          np.sum(np.array(aero_state_b.masses(include=["Na"])) * np.array(aero_state_b.num_concs)))

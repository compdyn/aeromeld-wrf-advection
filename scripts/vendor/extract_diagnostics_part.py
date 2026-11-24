#!/usr/bin/env python

"""
Compute diagnostic spectra from Zheng N1000 particle-resolved aerosol states.

The input particle-resolved state is expected to contain:
    - aero_particle_mass: [sample, aero_species, aero_particle]
    - aero_num_conc:      [sample, aero_particle]

Here each computational particle is treated as one particle-resolved member:
`aero_particle_mass` is the per-particle chemical mass, while
`aero_num_conc` is the particle number-concentration weight. This uses the
particle (`input_type="prt"`) branch of the diagnostic implementation in
`extract_diagnostics.py`.
"""

import math
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import netCDF4 as nc
import numpy as np
import torch

from extract_diagnostics import (
    CCN_EPS_MAX,
    CCN_EPS_MIN,
    CCN_TEMPERATURE_K,
    DEFAULT_CHEM_DENSITY,
    DEFAULT_CHEM_KAPPA,
    DEFAULT_CHEM_REFRACTIVE,
    FREEZE_TEMP_MAX_C,
    FREEZE_TEMP_MIN_C,
    N_EPS,
    N_FREEZE_TEMP,
    SPECIES_SELECT,
    TORCH_DTYPE,
    WAVELENGTHS_M,
    get_apprxccncdf,
    get_bulkfrznfrac,
    get_bulkoptical,
)


DEFAULT_INPUT_FILES = [
    "aero_samples_part_Zheng_N1000.nc",
]
DEFAULT_OUT_FILE = "aero_samples_part_Zheng_N1000_diagnoistics.nc"

# ========================= User Configuration =========================
# Set INPUT_FILE to one concrete path, or keep None to use the first existing
# path in DEFAULT_INPUT_FILES.
INPUT_FILE = "grid_cell_092_096_00000001.nc"

# Set OUTPUT_FILE to one concrete path, or keep None to write DEFAULT_OUT_FILE
# next to the selected input file.
OUTPUT_FILE = None

BATCH_SIZE = 32
# =====================================================================


def _filled_array(var, key=None, dtype=np.float64):
    data = var[:] if key is None else var[key]
    if np.ma.isMaskedArray(data):
        data = data.filled(np.nan)
    return np.asarray(data, dtype=dtype)


def _parse_species_names(raw_names) -> List[str]:
    if isinstance(raw_names, bytes):
        raw_names = raw_names.decode()
    if isinstance(raw_names, str):
        return [name.strip() for name in raw_names.split(",") if name.strip()]
    return [str(name).strip() for name in raw_names]


def resolve_input_file(input_file: Optional[str]) -> str:
    if input_file is not None:
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file does not exist: {input_file}")
        return input_file

    for candidate in DEFAULT_INPUT_FILES:
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        "None of the default input files exist:\n"
        + "\n".join(f"  - {path}" for path in DEFAULT_INPUT_FILES)
    )


def get_species_indices(ds: nc.Dataset, species_select: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
    species_var = ds.variables["aero_species"]
    species_names = _parse_species_names(species_var.getncattr("names"))
    missing = [species for species in species_select if species not in species_names]
    if missing:
        raise ValueError(f"Input file is missing selected species: {missing}")
    species_indices = np.asarray([species_names.index(species) for species in species_select], dtype=np.int64)
    return species_indices, species_names


def validate_input_file(ds: nc.Dataset) -> None:
    required_dimensions = ["sample", "aero_species", "aero_particle"]
    required_variables = ["aero_particle_mass", "aero_num_conc", "aero_density", "aero_species"]
    missing_dimensions = [dim for dim in required_dimensions if dim not in ds.dimensions]
    missing_variables = [var for var in required_variables if var not in ds.variables]
    if missing_dimensions:
        raise ValueError(f"Input file is missing required dimensions: {missing_dimensions}")
    if missing_variables:
        raise ValueError(f"Input file is missing required variables: {missing_variables}")

    expected_shapes = {
        "aero_particle_mass": ("sample", "aero_species", "aero_particle"),
        "aero_num_conc": ("sample", "aero_particle"),
        "aero_density": ("aero_species",),
    }
    for var_name, expected_dims in expected_shapes.items():
        actual_dims = ds.variables[var_name].dimensions
        if actual_dims != expected_dims:
            raise ValueError(f"{var_name} dimensions mismatch: got={actual_dims}, expected={expected_dims}")


def compute_ccn_spectrum_part(
    m_prtchm: torch.Tensor,
    n_prt: torch.Tensor,
    chem_species: List[str],
    rho_chm_values: Optional[np.ndarray] = None,
    eps_histmin: float = CCN_EPS_MIN,
    eps_histmax: float = CCN_EPS_MAX,
    n_epshist: int = N_EPS,
) -> Dict[str, torch.Tensor]:
    device = m_prtchm.device
    dtype = m_prtchm.dtype
    if m_prtchm.ndim == 3:
        m_prtchm = m_prtchm.unsqueeze(0)
        n_prt = n_prt.unsqueeze(0)

    n_seeds, n_pop, n_part, n_chem = m_prtchm.shape
    if rho_chm_values is None:
        rho_chm = torch.tensor([DEFAULT_CHEM_DENSITY.get(chem, 1000.0) for chem in chem_species], device=device, dtype=dtype)
    else:
        rho_chm = torch.as_tensor(rho_chm_values, device=device, dtype=dtype)
    kappa_chm = torch.tensor([DEFAULT_CHEM_KAPPA.get(chem, 0.0) for chem in chem_species], device=device, dtype=dtype)

    result = get_apprxccncdf(
        m_prtchm,
        n_prt,
        eps_histmin,
        eps_histmax,
        n_epshist,
        CCN_TEMPERATURE_K,
        rho_chm,
        kappa_chm,
        chem_species,
        0.0,
        100,
        1e-12,
        n_seeds,
        n_pop,
        n_chem,
        n_part,
        device,
        dtype,
        input_type="prt",
        strict=False,
    )
    ccn_cdf = result["ccn_cdf"].reshape(n_pop, n_epshist) if n_seeds == 1 else result["ccn_cdf"].reshape(n_seeds, n_pop, n_epshist)
    return {"ccn_cdf": ccn_cdf, "eps_histbins": result["eps_histbins"]}


def compute_optical_spectrum_part(
    m_prtchm: torch.Tensor,
    n_prt: torch.Tensor,
    chem_species: List[str],
    rho_chm_values: Optional[np.ndarray] = None,
    wavelengths: Optional[List[float]] = None,
    n_max: int = 100,
) -> Dict[str, torch.Tensor]:
    device = m_prtchm.device
    dtype = m_prtchm.dtype
    cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64
    if m_prtchm.ndim == 3:
        m_prtchm = m_prtchm.unsqueeze(0)
        n_prt = n_prt.unsqueeze(0)

    n_seeds, n_pop, n_part, n_chem = m_prtchm.shape
    if wavelengths is None:
        wavelengths = WAVELENGTHS_M
    len_wv = torch.tensor(wavelengths, device=device, dtype=dtype)
    n_wave = len(wavelengths)
    if rho_chm_values is None:
        rho_chm = torch.tensor([DEFAULT_CHEM_DENSITY.get(chem, 1000.0) for chem in chem_species], device=device, dtype=dtype)
    else:
        rho_chm = torch.as_tensor(rho_chm_values, device=device, dtype=dtype)
    refr_list = [[DEFAULT_CHEM_REFRACTIVE.get(chem, 1.5 + 0j) for chem in chem_species] for _ in wavelengths]
    refr_wvchm = torch.tensor(refr_list, device=device, dtype=cdtype)

    result = get_bulkoptical(
        m_prtchm,
        n_prt,
        rho_chm,
        refr_wvchm,
        len_wv,
        n_max,
        complex(1.5, 0.0),
        chem_species,
        n_seeds,
        n_pop,
        n_chem,
        n_part,
        n_wave,
        device,
        dtype,
        cdtype,
        input_type="prt",
    )
    qs_pop = result["qs_pop"].squeeze(0) if n_seeds == 1 else result["qs_pop"]
    qa_pop = result["qa_pop"].squeeze(0) if n_seeds == 1 else result["qa_pop"]
    return {"qs_pop": qs_pop, "qa_pop": qa_pop}


def compute_frozen_fraction_spectrum_part(
    m_prtchm: torch.Tensor,
    n_prt: torch.Tensor,
    chem_species: List[str],
    rho_chm_values: Optional[np.ndarray] = None,
    tmprtr_inpmin: float = FREEZE_TEMP_MIN_C,
    tmprtr_inpmax: float = FREEZE_TEMP_MAX_C,
    n_tmprtr: int = N_FREEZE_TEMP,
) -> Dict[str, torch.Tensor]:
    device = m_prtchm.device
    dtype = m_prtchm.dtype
    if m_prtchm.ndim == 3:
        m_prtchm = m_prtchm.unsqueeze(0)
        n_prt = n_prt.unsqueeze(0)

    n_seeds, n_pop, n_part, n_chem = m_prtchm.shape
    if rho_chm_values is None:
        rho_chm = torch.tensor([DEFAULT_CHEM_DENSITY.get(chem, 1000.0) for chem in chem_species], device=device, dtype=dtype)
    else:
        rho_chm = torch.as_tensor(rho_chm_values, device=device, dtype=dtype)
    result = get_bulkfrznfrac(
        m_prtchm,
        n_prt,
        tmprtr_inpmin,
        tmprtr_inpmax,
        n_tmprtr,
        rho_chm,
        chem_species,
        n_seeds,
        n_pop,
        n_chem,
        n_part,
        input_type="prt",
    )
    frznfrac_tmp = result["frznfrac_tmp"].squeeze(0) if n_seeds == 1 else result["frznfrac_tmp"]
    return {"frznfrac_tmp": frznfrac_tmp, "temprtr_bins": result["temprtr_bins"]}


def compute_particle_resolved_diagnostics(
    mass_species_particle: np.ndarray,
    numconc_particle: np.ndarray,
    density_species: np.ndarray,
    *,
    species_select: Optional[Sequence[str]] = None,
) -> Dict[str, np.ndarray]:
    """
    Particle-resolved diagnostics for one aerosol state (single timestep/sample).

    Args:
        mass_species_particle: shape (n_species, n_particle), masses per computational particle.
        numconc_particle: shape (n_particle,), number concentration weights.
        density_species: shape (n_species,), bulk density (kg/m^3) aligned with species_select.
        species_select: chemical species names; defaults to SPECIES_SELECT.

    Returns:
        ccn_cdf (eps,), eps_histbins (eps,), qs_pop (wavelength,), qa_pop (wavelength,),
        frznfrac_tmp (temperature,), temprtr_bins (temperature,)
    """
    species_list = list(SPECIES_SELECT if species_select is None else species_select)
    mass_species_particle = np.asarray(mass_species_particle, dtype=np.float64)
    numconc_particle = np.asarray(numconc_particle, dtype=np.float64)
    density_species = np.asarray(density_species, dtype=np.float64)
    if mass_species_particle.ndim != 2:
        raise ValueError(f"mass_species_particle must be 2D (species, particle), got shape {mass_species_particle.shape}")
    if numconc_particle.ndim != 1:
        raise ValueError(f"numconc_particle must be 1D (particle,), got shape {numconc_particle.shape}")
    n_spec, n_part = mass_species_particle.shape
    if n_spec != len(species_list):
        raise ValueError(f"species axis size {n_spec} != len(species_select) {len(species_list)}")
    if density_species.shape[0] != n_spec:
        raise ValueError(f"density_species shape {density_species.shape} mismatches mass species dim {n_spec}")
    if numconc_particle.shape[0] != n_part:
        raise ValueError(f"numconc length {numconc_particle.shape[0]} != n_particle {n_part}")

    mass_species_particle = np.nan_to_num(mass_species_particle, nan=0.0, posinf=0.0, neginf=0.0)
    numconc_particle = np.nan_to_num(numconc_particle, nan=0.0, posinf=0.0, neginf=0.0)

    mass_batch = mass_species_particle[np.newaxis, :, :]
    mass_particle_species = np.moveaxis(mass_batch, 1, 2)
    m_prtchm = torch.as_tensor(mass_particle_species, dtype=TORCH_DTYPE)
    n_prt = torch.as_tensor(numconc_particle[np.newaxis, :, np.newaxis], dtype=TORCH_DTYPE)

    ccn = compute_ccn_spectrum_part(
        m_prtchm,
        n_prt,
        species_list,
        rho_chm_values=density_species,
    )
    optical = compute_optical_spectrum_part(
        m_prtchm,
        n_prt,
        species_list,
        rho_chm_values=density_species,
        wavelengths=WAVELENGTHS_M,
    )
    frozen = compute_frozen_fraction_spectrum_part(
        m_prtchm,
        n_prt,
        species_list,
        rho_chm_values=density_species,
        tmprtr_inpmin=FREEZE_TEMP_MIN_C,
        tmprtr_inpmax=FREEZE_TEMP_MAX_C,
        n_tmprtr=N_FREEZE_TEMP,
    )

    return {
        "ccn_cdf": ccn["ccn_cdf"].detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1),
        "eps_histbins": ccn["eps_histbins"].detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1),
        "qs_pop": optical["qs_pop"].detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1),
        "qa_pop": optical["qa_pop"].detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1),
        "frznfrac_tmp": frozen["frznfrac_tmp"].detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1),
        "temprtr_bins": frozen["temprtr_bins"].detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1),
    }


def compute_diagnostics_batch(
    mass_species_particle: np.ndarray,
    numconc_particle: np.ndarray,
    species_select: List[str],
    density_species: np.ndarray,
    batch_label: str,
) -> Dict[str, np.ndarray]:
    mass_species_particle = np.nan_to_num(mass_species_particle, nan=0.0, posinf=0.0, neginf=0.0)
    numconc_particle = np.nan_to_num(numconc_particle, nan=0.0, posinf=0.0, neginf=0.0)

    # Diagnostic particle routines use [seed, sample, particle, species].
    mass_particle_species = np.moveaxis(mass_species_particle, 1, 2)
    m_prtchm = torch.as_tensor(mass_particle_species, dtype=TORCH_DTYPE)
    n_prt = torch.as_tensor(numconc_particle[..., np.newaxis], dtype=TORCH_DTYPE)

    t_batch = time.time()
    print(f"{batch_label} CCN started", flush=True)
    ccn = compute_ccn_spectrum_part(m_prtchm, n_prt, species_select, rho_chm_values=density_species)
    print(f"{batch_label} CCN finished in {time.time() - t_batch:.1f} s", flush=True)

    t_optical = time.time()
    print(f"{batch_label} optical started", flush=True)
    optical = compute_optical_spectrum_part(
        m_prtchm,
        n_prt,
        species_select,
        rho_chm_values=density_species,
        wavelengths=WAVELENGTHS_M,
    )
    print(f"{batch_label} optical finished in {time.time() - t_optical:.1f} s", flush=True)

    t_frozen = time.time()
    print(f"{batch_label} frozen-fraction started", flush=True)
    frozen = compute_frozen_fraction_spectrum_part(
        m_prtchm,
        n_prt,
        species_select,
        rho_chm_values=density_species,
        tmprtr_inpmin=FREEZE_TEMP_MIN_C,
        tmprtr_inpmax=FREEZE_TEMP_MAX_C,
        n_tmprtr=N_FREEZE_TEMP,
    )
    print(f"{batch_label} frozen-fraction finished in {time.time() - t_frozen:.1f} s", flush=True)

    return {
        "ccn_cdf": ccn["ccn_cdf"].detach().cpu().numpy().astype(np.float32, copy=False),
        "eps_histbins": ccn["eps_histbins"].detach().cpu().numpy().astype(np.float32, copy=False),
        "qs_pop": optical["qs_pop"].detach().cpu().numpy().astype(np.float32, copy=False),
        "qa_pop": optical["qa_pop"].detach().cpu().numpy().astype(np.float32, copy=False),
        "frznfrac_tmp": frozen["frznfrac_tmp"].detach().cpu().numpy().astype(np.float32, copy=False),
        "temprtr_bins": frozen["temprtr_bins"].detach().cpu().numpy().astype(np.float32, copy=False),
    }


def create_output_file(out_path: str, input_file: str, n_sample: int, n_species: int, n_particle: int) -> nc.Dataset:
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(out_path):
        os.remove(out_path)

    ncf = nc.Dataset(out_path, "w", format="NETCDF4")
    ncf.createDimension("samples", None)
    ncf.createDimension("species", n_species)
    ncf.createDimension("aero_particle", n_particle)
    ncf.createDimension("eps", N_EPS)
    ncf.createDimension("wavelength", len(WAVELENGTHS_M))
    ncf.createDimension("temperature", N_FREEZE_TEMP)

    ncv_sample = ncf.createVariable("sample_index", "i4", ("samples",))
    ncv_ccn = ncf.createVariable("ccn_cdf", "f4", ("samples", "eps"), zlib=True)
    ncv_eps = ncf.createVariable("eps_histbins", "f4", ("eps",))
    ncv_qs = ncf.createVariable("optical_scattering", "f4", ("samples", "wavelength"), zlib=True)
    ncv_qa = ncf.createVariable("optical_absorption", "f4", ("samples", "wavelength"), zlib=True)
    ncv_wavelength = ncf.createVariable("wavelength_m", "f4", ("wavelength",))
    ncv_frozen = ncf.createVariable("frozen_fraction", "f4", ("samples", "temperature"), zlib=True)
    ncv_temp = ncf.createVariable("temperature_C", "f4", ("temperature",))
    ncv_species = ncf.createVariable("species", "i4", ("species",))
    ncv_particle = ncf.createVariable("aero_particle", "i4", ("aero_particle",))

    ncv_sample.long_name = "Sample index in particle-resolved aero_samples_part_Zheng_N1000.nc"
    ncv_ccn.long_name = "CCN cumulative distribution function"
    ncv_eps.long_name = "Supersaturation epsilon bins where epsilon = CRH - 1"
    ncv_qs.long_name = "Bulk optical scattering spectrum"
    ncv_qa.long_name = "Bulk optical absorption spectrum"
    ncv_wavelength.long_name = "Optical wavelength in meters"
    ncv_frozen.long_name = "Bulk frozen fraction spectrum"
    ncv_temp.long_name = "Temperature bins in Celsius"
    ncv_species.long_name = "Selected species indices"
    ncv_species.setncattr("names", ",".join(SPECIES_SELECT))
    ncv_particle.long_name = "Particle index in the input particle-resolved state"

    ncv_sample[:] = np.arange(n_sample, dtype=np.int32)
    ncv_wavelength[:] = np.asarray(WAVELENGTHS_M, dtype=np.float32)
    ncv_species[:] = np.arange(n_species, dtype=np.int32)
    ncv_particle[:] = np.arange(n_particle, dtype=np.int32)

    ncf.title = "Aerosol diagnostic spectra derived from Zheng N1000 particle-resolved aerosol states"
    ncf.input_data = input_file
    ncf.note = "Computed directly from particle-resolved mass and number-concentration weights."
    return ncf


def process_file(input_file: str, out_path: str, batch_size: int) -> None:
    with nc.Dataset(input_file) as ds:
        validate_input_file(ds)
        species_indices, species_names = get_species_indices(ds, SPECIES_SELECT)
        n_sample = ds.dimensions["sample"].size
        n_particle = ds.dimensions["aero_particle"].size
        n_species_all = ds.dimensions["aero_species"].size
        density_species = _filled_array(ds.variables["aero_density"], dtype=np.float64)[species_indices]
        batch_size = min(batch_size, n_sample)
        n_batches = math.ceil(n_sample / batch_size)

        print(f"Using particle-resolved aerosol states from: {input_file}", flush=True)
        print(f"Loaded {n_sample} samples, {n_species_all} input species, {n_particle} particles per sample", flush=True)
        print("Selected species: " + ",".join(SPECIES_SELECT), flush=True)
        print("Selected species indices: " + ",".join(str(idx) for idx in species_indices), flush=True)
        print("Input species order: " + ",".join(species_names), flush=True)
        print("Using aero_density from the input file for the selected species.", flush=True)
        print("Diagnostics will be computed directly from particle masses and number-concentration weights.", flush=True)
        print(f"Diagnostic batch size: {batch_size}", flush=True)
        print(f"Writing output to: {out_path}", flush=True)

        ncf_out = create_output_file(out_path, input_file, n_sample, len(SPECIES_SELECT), n_particle)
        t0 = time.time()
        try:
            for i_batch, i_start in enumerate(range(0, n_sample, batch_size), start=1):
                i_end = min(i_start + batch_size, n_sample)
                progress_pct = 100.0 * i_end / n_sample
                elapsed_s = time.time() - t0
                batch_label = (
                    f"[Diagnostics-Part] Batch {i_batch}/{n_batches} | "
                    f"samples {i_start}:{i_end} / {n_sample} | "
                    f"{progress_pct:5.1f}% | elapsed {elapsed_s:7.1f} s |"
                )
                print(f"{batch_label} loading data", flush=True)

                mass_batch_all_species = _filled_array(
                    ds.variables["aero_particle_mass"],
                    key=(slice(i_start, i_end), slice(None), slice(None)),
                    dtype=np.float64,
                )
                mass_batch = mass_batch_all_species[:, species_indices, :]
                numconc_batch = _filled_array(
                    ds.variables["aero_num_conc"],
                    key=(slice(i_start, i_end), slice(None)),
                    dtype=np.float64,
                )
                diagnostics = compute_diagnostics_batch(
                    mass_batch,
                    numconc_batch,
                    SPECIES_SELECT,
                    density_species,
                    batch_label,
                )

                ncf_out.variables["ccn_cdf"][i_start:i_end] = diagnostics["ccn_cdf"]
                ncf_out.variables["optical_scattering"][i_start:i_end] = diagnostics["qs_pop"]
                ncf_out.variables["optical_absorption"][i_start:i_end] = diagnostics["qa_pop"]
                ncf_out.variables["frozen_fraction"][i_start:i_end] = diagnostics["frznfrac_tmp"]
                if i_batch == 1:
                    ncf_out.variables["eps_histbins"][:] = diagnostics["eps_histbins"]
                    ncf_out.variables["temperature_C"][:] = diagnostics["temprtr_bins"]
                ncf_out.sync()
                print(f"{batch_label} written", flush=True)
        finally:
            ncf_out.close()

    print(f"Finished. Wrote {n_sample} samples to {out_path}", flush=True)


def main() -> None:
    input_file = resolve_input_file(INPUT_FILE)
    if BATCH_SIZE <= 0:
        raise ValueError(f"BATCH_SIZE must be positive, got {BATCH_SIZE}")

    out_path = OUTPUT_FILE
    if out_path is None:
        out_path = os.path.join(os.path.dirname(input_file), DEFAULT_OUT_FILE)

    process_file(input_file, out_path, BATCH_SIZE)


if __name__ == "__main__":
    main()

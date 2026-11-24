#!/usr/bin/env bash
# Build the latent-space wrfinput/wrfbdy (WRF-ML) by calling make_latent_ic.py.
# Latent twin of run_ics.sh -- keep NX/NY/NZ and REGION identical between the
# two so the latent run and the particle run describe the same initial condition.
#
# Edit the values below, then run:  ./run_latent_ics.sh
# Needs a conda env with xarray + netCDF4 + numpy (e.g.  conda activate wrf).
set -euo pipefail
ORIG_PWD="$PWD"                                        # for resolving relative latent files
cd "$(dirname "$0")"

# ---- values you want -------------------------------------------------------
CASE_DIR="/path/to/WRF-AeroMELD/test/latent_advection_cares"
OUTDIR="/path/to/latent_inputs"

# These four (and the *_LATENT_FILEs below) can be set in the environment to
# build a second input set without touching the first, e.g. for another model:
#   IC_OUT=/path/to/newrun/wrfinput_d01 ... ./run_latent_ics.sh
IC_TEMPLATE="${IC_TEMPLATE:-$CASE_DIR/wrfinput_d01_cares}"
BC_TEMPLATE="${BC_TEMPLATE:-$CASE_DIR/wrfbdy_d01_cares}"
IC_OUT="${IC_OUT:-$OUTDIR/wrfinput_d01_advection_test}"
BC_OUT="${BC_OUT:-$OUTDIR/wrfbdy_d01_advection_test}"

# Grid is read from the template; the REGION box below is 1-based inclusive.
# CARES: nx=169, ny=159, nz=39 -- same as run_ics.sh.

# ltnt_var1..N. 9 for chem_opt=900 models; 49 for chem_opt=901, whose
# templates need ltnt_var10-49 (extend_latent_template.py; new_model_run.sh
# picks the *_cares_l49 templates and sets N_LATENT itself).
N_LATENT="${N_LATENT:-9}"

# Base population: applied to every cell and to the boundaries.
BASE_VALUE="1e9"                                       # ntot, 1/m^3 (~1000 cm^-3, realistic background)
BASE_LATENT="-0.1027,-2.2868,6.1429,1.4282,-1.0323,0.3196,-1.6883,-1.4831,0.0577"

# Optional box region that swaps in a second population.
# Set REGION="" to skip the region entirely. Keep this identical to run_ics.sh.
# j shifted north (70-100, was 40-70) to sit well clear of the south boundary;
# the emission region (run_latent_emis.sh / run_emis.sh) sits just east of
# this box, i 83-113 at the same j, with a 2-cell gap so the two are almost
# touching but don't overlap.
REGION="50 80 70 100 1 10"                             # i1 i2 j1 j2 k1 k2 (1-based, inclusive)
REGION_VALUE="1e10"                                    # ntot inside region (peak for gaussian); 10x BASE_VALUE
REGION_PROFILE="gaussian"                              # constant | gaussian
REGION_LATENT="0.0667,0.1501,-0.6538,1.8972,-0.3033,1.8006,0.8581,0.9799,-0.0596"

# Instead of the literals above, either latent vector can be read from a text
# file: set the path here, or pass it in the environment, e.g.
#   BASE_LATENT_FILE=input_data/latent_scenario_561_0001_00000013.txt ./run_latent_ics.sh
# The file may hold the values one per line, comma separated, or any mix;
# blank lines and "#" comments are ignored. Only the first N_LATENT values are
# used, so the latent_scenario_*.txt files written by vae_partmc_utils.py
# (N latent values followed by the normalized ntot) can be pointed at directly
# -- their trailing ntot entry is ignored, BASE_VALUE / REGION_VALUE below
# still set the physical number concentration in 1/m^3.
# LATENT_DIR switches the whole set, e.g. LATENT_DIR=input_data/05_linplong.5.0
# for latents encoded by another model (see new_model_run.sh).
LATENT_DIR="${LATENT_DIR:-input_data}"
BASE_LATENT_FILE="${BASE_LATENT_FILE-$LATENT_DIR/latent_scenario_561_0001_00000013.txt}"
REGION_LATENT_FILE="${REGION_LATENT_FILE-$LATENT_DIR/latent_scenario_635_0001_00000013.txt}"
# ---------------------------------------------------------------------------

# Read N_LATENT numbers out of a text file, return them comma separated.
read_latent_file() {
  local file="$1" label="$2"
  # Relative paths are taken from wherever the script was invoked; if that
  # misses, fall back to this directory (we cd'd here at the top).
  if [[ "$file" != /* && -r "$ORIG_PWD/$file" ]]; then
    file="$ORIG_PWD/$file"
  fi
  if [[ ! -r "$file" ]]; then
    echo "$label: cannot read latent file '$file'" >&2
    exit 1
  fi
  local -a vals
  mapfile -t vals < <(sed 's/#.*//' "$file" | tr ',' ' ' | tr -s '[:space:]' '\n' \
                        | sed '/^$/d')
  if (( ${#vals[@]} < N_LATENT )); then
    echo "$label: '$file' has ${#vals[@]} value(s), need $N_LATENT" >&2
    exit 1
  fi
  if (( ${#vals[@]} > N_LATENT )); then
    echo "$label: '$file' has ${#vals[@]} values; using the first $N_LATENT," \
         "ignoring [${vals[*]:N_LATENT}]" >&2
  fi
  local IFS=,
  echo "${vals[*]:0:N_LATENT}"
}

if [[ -n "$BASE_LATENT_FILE" ]]; then
  BASE_LATENT="$(read_latent_file "$BASE_LATENT_FILE" "--base-latent")"
  echo "base latent   <- $BASE_LATENT_FILE"
fi
if [[ -n "$REGION" && -n "$REGION_LATENT_FILE" ]]; then
  REGION_LATENT="$(read_latent_file "$REGION_LATENT_FILE" "--region-latent")"
  echo "region latent <- $REGION_LATENT_FILE"
fi

# Note: the latent vectors are passed with =VALUE so argparse doesn't mistake a
# leading "-0.1027" for a flag.
args=(
  --ic-template "$IC_TEMPLATE" --ic-outfile "$IC_OUT"
  --bc-template "$BC_TEMPLATE" --bc-outfile "$BC_OUT"
  --n-latent "$N_LATENT"
  --base-value "$BASE_VALUE" --base-latent="$BASE_LATENT"
)

if [[ -n "${REGION}" ]]; then
  args+=(
    --region $REGION
    --region-value "$REGION_VALUE"
    --region-profile "$REGION_PROFILE"
    --region-latent="$REGION_LATENT"
  )
fi

python make_latent_ic.py "${args[@]}"

#!/usr/bin/env bash
# Build the latent-space wrfchemi emission files (WRF-ML) by calling
# make_latent_emis.py. Emission companion to run_latent_ics.sh -- keep the grid
# and the REGION box consistent with it so the emitted plume lands where you
# expect relative to the initial condition.
#
# Edit the values below, then run:  ./run_latent_emis.sh
# Needs a conda env with netCDF4 + numpy (e.g.  conda activate wrf).
#
# On the WRF side the run directory needs, in namelist.input:
#   &chem          emiss_opt = 21 (9 latents; 22 with N_LATENT=49 and chem_opt=901),
#                  io_style_emissions = 2, kemit = 1
#   &time_control  io_form_auxinput5 = 2, auxinput5_interval_m = 60,
#                  frames_per_auxinput5 = 1
# The CARES namelist ships with io_style_emissions = 0, which reads NO
# emissions at all -- change it or nothing here will have any effect.
set -euo pipefail
ORIG_PWD="$PWD"                                        # for resolving relative latent files
cd "$(dirname "$0")"

# ---- values you want -------------------------------------------------------
CASE_DIR="/path/to/WRF-AeroMELD/test/latent_advection_cares"
# EMIS_OUTDIR, GRID_TEMPLATE and the *_LATENT_FILEs below can be set in the
# environment to build a second emission set without touching the first.
OUTDIR="${EMIS_OUTDIR:-/path/to/latent_inputs/emis}"

GRID_TEMPLATE="${GRID_TEMPLATE:-$CASE_DIR/wrfinput_d01_cares}"  # grid + XLAT/XLONG + globals
DOMAIN=1

# Time coverage. START must be the run's start time (namelist start_* fields);
# N_TIMES must cover the whole run: run_length / interval + 1.
START="2010-06-17_00:00:00"                            # CARES run start
N_TIMES=13                                             # 12 h run at 60 min -> 13 files
INTERVAL_M=60                                          # = auxinput5_interval_m
KEMIT=1                                                # = kemit (surface-only emissions)

N_LATENT="${N_LATENT:-9}"                              # E_LTNT_VAR1..N: 9 (emiss_opt=21) or 49 (emiss_opt=22)

# Base emitted population: applied to every cell and every emission level.
# 0 = no background emissions, only the region box below emits.
BASE_VALUE="0"                                         # E_NTOT, number/m^2/s
BASE_LATENT="-0.1027,-2.2868,6.1429,1.4282,-1.0323,0.3196,-1.6883,-1.4831,0.0577"

# Optional box region that emits a second population.
# Set REGION="" to skip it. Sits just east of the urban IC box (i 50-80,
# j 70-100 in run_latent_ics.sh / run_ics.sh) at the same j, with a 2-cell gap
# (i 81-82 empty) so the two boxes are almost touching without overlapping.
# Both boxes were moved north together to stay well clear of the south
# boundary. i/j should match run_emis.sh; k is over the KEMIT emission
# levels, NOT the full e_vert column.
REGION="83 113 70 100 1 1"                             # i1 i2 j1 j2 k1 k2 (1-based, inclusive)
REGION_VALUE="${EMIS_REGION_VALUE:-5e7}"               # E_NTOT inside region (env EMIS_REGION_VALUE overrides)
REGION_PROFILE="constant"                              # constant | gaussian
REGION_LATENT="0.0667,0.1501,-0.6538,1.8972,-0.3033,1.8006,0.8581,0.9799,-0.0596"

# Diurnal cycle. "constant" holds the flux fixed; "diurnal" scales it by
# 1 + A*cos(2*pi*(h - peak)/24) so BASE_VALUE / REGION_VALUE are the daily MEAN.
# This only touches E_NTOT -- the latent vectors are stored intensively, so the
# emitted POPULATION is identical at every hour, only its number flux varies.
TIME_PROFILE="constant"                                # constant | diurnal
DIURNAL_AMPLITUDE=""                                   # blank -> 0.75
DIURNAL_PEAK=""                                        # blank -> local hour 15
UTC_OFFSET="-7"                                        # CARES / California local time

# Instead of the literals above, either latent vector can be read from a text
# file: set the path here, or pass it in the environment, e.g.
#   BASE_LATENT_FILE=input_data/latent_scenario_561_0001_00000013.txt ./run_latent_emis.sh
# The file may hold the values one per line, comma separated, or any mix;
# blank lines and "#" comments are ignored. Only the first N_LATENT values are
# used, so the latent_scenario_*.txt files written by vae_partmc_utils.py
# (N latent values followed by the normalized ntot) can be pointed at directly
# -- their trailing ntot entry is ignored, BASE_VALUE / REGION_VALUE below
# still set the physical emission flux.
# LATENT_DIR switches the whole set, e.g. LATENT_DIR=input_data/05_linplong.5.0
# for latents encoded by another model (see new_model_run.sh).
LATENT_DIR="${LATENT_DIR:-input_data}"
BASE_LATENT_FILE="${BASE_LATENT_FILE-$LATENT_DIR/latent_scenario_812_0001_00000013.txt}"
REGION_LATENT_FILE="${REGION_LATENT_FILE-$LATENT_DIR/latent_scenario_812_0001_00000013.txt}"
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
  --grid-template "$GRID_TEMPLATE" --outdir "$OUTDIR" --domain "$DOMAIN"
  --start "$START" --n-times "$N_TIMES" --interval-m "$INTERVAL_M"
  --kemit "$KEMIT" --n-latent "$N_LATENT"
  --base-value "$BASE_VALUE" --base-latent="$BASE_LATENT"
  --time-profile "$TIME_PROFILE" --utc-offset "$UTC_OFFSET"
)

if [[ -n "$REGION" ]]; then
  args+=(
    --region $REGION
    --region-value "$REGION_VALUE"
    --region-profile "$REGION_PROFILE"
    --region-latent="$REGION_LATENT"
  )
fi

[[ -n "$DIURNAL_AMPLITUDE" ]] && args+=(--diurnal-amplitude "$DIURNAL_AMPLITUDE")
[[ -n "$DIURNAL_PEAK" ]]      && args+=(--diurnal-peak "$DIURNAL_PEAK")

python make_latent_emis.py "${args[@]}"

printf -v DOM "d%02d" "$DOMAIN"
cat <<EOF

Next: link or copy the files into the run directory, e.g.
  ln -sf $OUTDIR/wrfchemi_${DOM}_* $CASE_DIR/
and make sure namelist.input has io_style_emissions = 2 (it ships as 0).
EOF

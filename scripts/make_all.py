#!/usr/bin/env python
"""Make every figure for one run (default: the main run), building products as needed.

Usage (from scripts/; the partnn env when products must be built):

    python make_all.py                     # the main run -> figures/
    python make_all.py --run KEY           # another run -> graphics/by_model/KEY/
    python make_all.py --format png --out /tmp/look

Figures are made in the order below; each plot script builds any product it is
missing (see paperfigs/products.py), so a first run on a machine with the raw
output builds them all. Then `python -m paperfigs.products describe` refreshes
data/PRODUCTS.md, and `python models.py --write` refreshes data/metrics.csv
(every registered run, not just this one).
"""
import argparse
import importlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paperfigs import figures  # noqa: E402

#: plot/<module>.py, in the order they are made.
PLOTS = (
    'overview',                  # Figure 1: encode -> transport -> decode (a drawing; replaces schematic 2026-11-13)
    'domain_evolution',          # the set-up and the aerosol through the 12-hour run (Figure 2)
    'source_number_fraction',
    'maps_combined',             # diagnostic_maps: four diagnostics, WRF-AeroMELD, reference, ratio (12 h)
    'summary_errors',            # the recommended configuration's per-cell error, six diagnostics (12 h)
    'diagnostic_panels',         # diagnostic_panels_ABC, the paper's fig:point-panels
    'latent_populations',
    'mixing_training',           # mixing_kcurves: the decoders on the test set (ten seeds)
    'transport_schemes',         # schemes_fronts / _time / _models: the 12-hour replays
    'tradeoffs',                 # the two trade-offs (12 h): tradeoffs_transport, tradeoffs_decoders (+ combined)
    'comparison_scorecard',      # WRF-AeroMELD against the sectional and modal schemes (12 h; baselines/)
    'comparison_tracers',        # their error against the tracers advected per cell
    'comparison_kcurves',        # their error on mixtures of K test populations
)
# Dropped 2026-11-15: ccn_maps, frozen_fraction, optics_maps (CCN_ML_vs_PR, frozen_fraction,
# optics_ML_vs_PR), replaced by maps_combined (diagnostic_maps), one condition of each.
# Dropped 2026-11-15: computational_cost (no cost section; the tracer count is the cost measure,
# comparison_tracers).
# Dropped 2026-11-12: latent_maps, latent_timeseries (the latents are a linear image
# of the source fractions), diagnostic_ML_vs_PR (the point panels show the same spectra),
# model_comparison (models are compared on the test set, schemes on the case).
# Dropped 2026-11-12: number_mixing_ratio_field (it compared WRF-AeroMELD's number with WRF-PartMC's;
# against the exact answer the two are the same tracer to limiter level, stated in the text). The
# reference everywhere is now the exact answer (paperfigs.reference), not WRF-PartMC.


def main(argv=None):
    ap = figures.cli(__doc__)
    ap.add_argument('--only', nargs='+', choices=PLOTS, help='just these plot scripts')
    args = ap.parse_args(argv)
    run = figures.setup(args)
    sys.path.insert(0, str(Path(__file__).resolve().parent / 'plot'))
    for name in args.only or PLOTS:
        t0 = time.perf_counter()
        importlib.import_module(name).make(run)
        print('  %s: %.1f s' % (name, time.perf_counter() - t0), flush=True)


if __name__ == '__main__':
    main()

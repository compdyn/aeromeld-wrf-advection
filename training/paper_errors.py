#!/usr/bin/env python
"""
Compute the AeroMELD paper's test-error table (Table 2, `tab:rcnsterrs` in
paper_diagpart.tex) from a vaepart results h5.

This is a numpy transcription of `cnvrt_physunits` and `calc_aerometrics` in
code_partnn/notebooks/n08_plotting.ipynb (cell 3), restricted to the table rows:

  - Inputs are the stored `eval/identity` diagnostics of all 5000 test
    populations (`var/eval/raw/identity:x:<ycol>/<split>/{orig,rcnst}`), at the
    last stored epoch.
  - Mass and number are clipped at 0, diameter bins are cut to <= 10 um (15 of
    19 bins), and the optics are converted to Mm^-1. The bins are 0.25 decades
    wide from 10^-8.75 m; the number of bins is read from the arrays, so the
    PM10 dataset (15 bins up to 10 um, aeromeld-lindiag/alcf/clean_coarse.py)
    keeps all of its bins and the 19-bin metric is unchanged.
  - Per population, the error is ||y - yhat||_2 / (||y||_2 + ||yhat||_2 + 1e-30),
    taken jointly over channels and bins.
  - CCN is limited to s = 0.1-0.6 % (bins 0-39). The frozen fraction is limited
    to -25 to -10 C (bins 38-75) and uses log(clip(ff, 1e-20)). Optics use
    log(clamp(q, 0.1 Mm^-1)).
  - Errors are averaged over the populations of each seed. The table reports
    the mean over seeds and a 95 % bootstrap interval of the seed means (40
    resamples, as in the notebook's `get_aggdf` call; the resampling RNG differs,
    so interval endpoints can differ slightly).

Usage:
    python paper_errors.py RESULTS.h5 [--split test] [--epoch -1]
"""
import argparse
import math
import h5py
import numpy as np

PAPER = {'CCN Spectrum Relative Error': 2.04, 'Scattering Log-Rel Error': 2.72,
    'Absorption Log-Rel Error': 12.8, 'Frozen Fraction Log-Rel Error': 1.37,
    'Speciated Mass Relative Error': 25.8, 'Number Relative Error': 6.20,
    'Total Mass Relative Error': 19.3, 'Species Bulk Mass Relative Error': 16.6}
ROWS = {'CCN Spectrum Relative Error': 'ccn_cdf', 'Scattering Log-Rel Error': 'logqs_pop',
    'Absorption Log-Rel Error': 'logqa_pop', 'Frozen Fraction Log-Rel Error': 'logfrznfrac_tmp',
    'Speciated Mass Relative Error': 'm_chmprthst', 'Number Relative Error': 'n_prthst',
    'Total Mass Relative Error': 'm_prthst', 'Species Bulk Mass Relative Error': 'm_chmprt'}
YCOLS = ['m_chmprthst', 'n_prthst', 'ccn_cdf', 'qs_pop', 'qa_pop', 'frznfrac_tmp']

n_epshist, n_tmprtr = 100, 100
# Diameter bins of 0.25 decades from 10^-8.75 m: 19 up to 100 um, or 15 up to 10 um (PM10)
log10_diam_low, log10len_bin = -8.75, 0.25
i2_diam_max = round((-5 - log10_diam_low) / log10len_bin)

def get_i2_diam(n_bins):
    """The number of leading bins whose upper edge is at most 10 um."""
    assert n_bins in (15, 19), f'unexpected number of diameter bins: {n_bins}'
    return min(n_bins, i2_diam_max)

def get_binidx(x, x_min, x_max, n_x, tnsfrm='none'):
    x2, lo, hi = (math.log(x), math.log(x_min), math.log(x_max)) if tnsfrm == 'log' else (x, x_min, x_max)
    return round((x2 - lo) * n_x / (hi - lo))

i_eps1, i_eps3 = get_binidx(0.001, 1e-3, 1e-1, n_epshist, 'log'), get_binidx(0.006, 1e-3, 1e-1, n_epshist, 'log')
i_tmp1, i_tmp3 = get_binidx(-25, -40, 0, n_tmprtr), get_binidx(-10, -40, 0, n_tmprtr)

def relerr(y, yhat):
    """Per-population symmetric L2 relative error over the last two dims."""
    num = np.sqrt(((y - yhat) ** 2).sum(axis=(-1, -2)))
    den = np.sqrt((y ** 2).sum(axis=(-1, -2))) + np.sqrt((yhat ** 2).sum(axis=(-1, -2))) + 1e-30
    return num / den

def paper_transform(yv):
    """
    The transformed diagnostics the table errors are computed on. `yv` maps
    the raw diagnostic names (m_chmprthst, n_prthst, ccn_cdf, qs_pop, qa_pop,
    frznfrac_tmp) to arrays of shape (..., C, L) in the dataset's units.
    Returns a dict from table row name to arrays of shape (..., C', L').
    """
    i2_diam = get_i2_diam(yv['m_chmprthst'].shape[-1])
    m = np.clip(yv['m_chmprthst'], 0, None) * (1e9 / log10len_bin)
    n = np.clip(yv['n_prthst'], 0, None) * (1e-6 / log10len_bin)
    y = {'m_chmprt': m.sum(axis=-1, keepdims=True),
         'm_chmprthst': m[..., :i2_diam],
         'm_prthst': m[..., :i2_diam].sum(axis=-2, keepdims=True),
         'n_prthst': n[..., :i2_diam],
         'ccn_cdf': yv['ccn_cdf'][..., i_eps1:i_eps3 + 1],
         'logfrznfrac_tmp': np.log(np.clip(yv['frznfrac_tmp'], 1e-20, None))[..., i_tmp1:i_tmp3 + 1]}
    for q in ['qs_pop', 'qa_pop']:
        y[f'log{q}'] = np.log(np.clip(yv[q] * 1e6, 1e-1, None))
    return {name: y[ycol] for name, ycol in ROWS.items()}

def paper_metrics(yo, yr):
    """
    The per-population table errors between original `yo` and reconstructed
    `yr` diagnostics (both as in `paper_transform`). Returns a dict from table
    row name to errors of shape (...).
    """
    to, tr = paper_transform(yo), paper_transform(yr)
    return {name: relerr(to[name], tr[name]) for name in ROWS}

#: The rows on a logarithmic scale.
LOG_ROWS = ('Scattering Log-Rel Error', 'Absorption Log-Rel Error', 'Frozen Fraction Log-Rel Error')


def logratio_metrics(yo, yr):
    """
    The WRF-AeroMELD paper's errors: as `paper_metrics`, but on the three logarithmic rows
    (frozen fraction and the two optics; same points, clips and units) the error is half the
    RMS of the log ratio, not ||log y - log yhat|| / (||log y|| + ||log yhat||). The latter
    divides by the size of the logarithms themselves, so it depends on the units and shrinks
    an error the more orders of magnitude a diagnostic spans. Half the RMS log ratio reads,
    for small errors, like the linear rows' error (half the relative difference).
    """
    to, tr = paper_transform(yo), paper_transform(yr)
    out = {}
    for name in ROWS:
        if name in LOG_ROWS:
            out[name] = np.sqrt(((to[name] - tr[name]) ** 2).mean(axis=(-1, -2))) / 2
        else:
            out[name] = relerr(to[name], tr[name])
    return out


def report(errs, file=None):
    """Prints the seed mean, a 95 % bootstrap interval, and the paper value of each row."""
    rng = np.random.RandomState(12345)
    print(f'{"metric":34s} {"mean":>7s}  {"95% CI":>17s}  {"paper":>6s}', file=file)
    for name, e in errs.items():
        boot = e[rng.randint(0, len(e), size=(40, len(e)))].mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f'{name:34s} {100 * boot.mean():6.2f}%  [{100 * lo:6.2f}%, {100 * hi:6.2f}%]  {PAPER[name]:5.2f}%', file=file)

def load_eval(grp, key, epoch=-1):
    """
    Reads a stored eval tensor (e.g. `var/eval/raw/identity:x:n_prthst/test/orig/pnts`)
    from a results-h5 partition group at one stored epoch (an index into the
    stored epochs). Returns the array (n_seeds, ...) without the singleton
    sample axis, the epoch, and the rng seed of each row.
    """
    stat_rows = dict(zip(grp['stat/ioidx'][:], grp['stat/block0_values'][:]))
    stat_cols = [c.decode() for c in grp['stat/block0_items'][:]]
    i_ep, i_sd = stat_cols.index('epoch'), stat_cols.index('rng_seed')
    ioidx = grp[f'{key}/ioidx'][:]
    epochs = np.array([stat_rows[i][i_ep] for i in ioidx])
    ep = np.unique(epochs)[epoch]
    rows = np.where(epochs == ep)[0]
    assert len(rows) == rows.max() - rows.min() + 1
    seeds = [int(stat_rows[ioidx[r]][i_sd]) for r in rows]
    arr = grp[f'{key}/data'][rows.min():rows.max() + 1]
    return arr[:, :, 0], int(ep), seeds

if __name__ == '__main__':
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('results')
    my_parser.add_argument('--split', default='test')
    my_parser.add_argument('--epoch', type=int, default=-1,
        help='index into the stored eval epochs (default: the last one)')
    my_parser.add_argument('--part', default='P00000000')
    args = my_parser.parse_args()

    grp = h5py.File(args.results, 'r')[args.part]
    y = dict()
    for ycol in YCOLS:
        for vrnt in ['orig', 'rcnst']:
            y[f'{ycol}/{vrnt}'], epoch, seeds = load_eval(grp,
                f'var/eval/raw/identity:x:{ycol}/{args.split}/{vrnt}/pnts', args.epoch)
            n_pop = y[f'{ycol}/{vrnt}'].shape[1]

    errs = {name: e.mean(axis=-1) for name, e in paper_metrics(
        {c: y[f'{c}/orig'] for c in YCOLS}, {c: y[f'{c}/rcnst'] for c in YCOLS}).items()}
    print(f'{args.results}: split={args.split}, epoch={epoch}, {len(seeds)} seeds, {n_pop} populations per seed')
    report(errs)

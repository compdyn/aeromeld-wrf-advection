#!/usr/bin/env python
"""Sortable table of the registered WRF-ML runs, from data/metrics.csv.

The model-selection tool (it replaces code_partnn's summary_by_run.py). Each
row is a run, each column one condition of the violin figure (CCN/CN at four
s_env, scattering and absorption at 400 and 1000 nm, frozen fraction at four
temperatures), over the mixed cells at the final instant.

Usage (from scripts/):

    python models.py                                  # SMAPE, sorted by its mean
    python models.py --sort smape:ff_-30              # by SMAPE of frozen fraction at -30 C
    python models.py --sort smape:ff                  # by mean SMAPE over the frozen-fraction columns
    python models.py --sort ratio:ccn --show ratio    # median ML/PMC ratio, closest to 1 first
    python models.py --role main,appendix             # only these roles (or --set NAME, --runs A,B)
    python models.py --write                          # recompute data/metrics.csv from the products

--sort METRIC[:WHAT]
    METRIC  a statistic column of metrics.csv, or an alias:
            smape (smape_pct), e (e_median), ratio (ratio_median),
            high (pct_cells_ml_high), unmixed (smape_unmixed_pct), all (smape_all_pct)
    WHAT    a cond_id (ccn_0.1, scat_400, abs_1000, ff_-30, ...), a group
            (ccn, scat, abs, ff: mean over its columns) or mean (every column;
            the default)
    Ratios sort by |log10 r|, i.e. the run closest to 1 first. Everything
    else sorts ascending; --desc reverses.

--show METRIC   the statistic the columns show (default: the sort metric).

Reading the table needs only the standard library. --write needs the products
(and builds any that are missing, which needs the partnn env and the raw
output); it takes a few seconds per run once the products exist.
"""
import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paperfigs import metrics, registry  # noqa: E402

ALIASES = {'smape': 'smape_pct', 'e': 'e_median', 'ratio': 'ratio_median',
           'high': 'pct_cells_ml_high', 'unmixed': 'smape_unmixed_pct', 'all': 'smape_all_pct'}
STATS = tuple(c for c, _ in metrics.STAT_COLUMNS)


def metric_name(name):
    col = ALIASES.get(name, name)
    if col not in STATS:
        raise SystemExit('unknown metric %r; use one of %s or a column: %s'
                         % (name, ', '.join(ALIASES), ', '.join(STATS)))
    return col


def badness(col, value):
    """What --sort orders by: ratios by their distance from 1 in decades."""
    if col.startswith('ratio_') and col != 'ratio_cells':
        return abs(math.log10(value)) if value > 0 else math.inf
    return value


def parse_sort(spec, cond_ids):
    metric, _, what = spec.partition(':')
    col = metric_name(metric)
    what = what or 'mean'
    if what == 'mean':
        cols = list(cond_ids)
    elif what in cond_ids:
        cols = [what]
    elif what in metrics.GROUP_SHORT.values():
        cols = [c for c in cond_ids if c.startswith(what + '_')]
    else:
        raise SystemExit('--sort %s: %r is not a cond_id (%s), a group (%s) or mean'
                         % (spec, what, ', '.join(cond_ids),
                            ', '.join(metrics.GROUP_SHORT.values())))
    return col, cols, '%s:%s' % (metric, what)


def fmt(col, v):
    if col.startswith('n_') or col == 'ratio_cells':
        return '%d' % v
    if col.endswith('_pct'):
        return '%.1f' % v if v < 100 else '%.0f' % v
    if col.startswith('ratio_'):
        return '%.3f' % v if v < 10 else '%.2f' % v
    return '%.3g' % v


def table(runs, cond_ids, by, show, sort_col, sort_cols, sort_label, desc):
    """Lines of the printed table, sorted. The label column appears once some
    model has a label of its own (registry labels default to the model key)."""
    def key(run_key):
        vals = [badness(sort_col, by[run_key][c][sort_col]) for c in sort_cols]
        return sum(vals) / len(vals)

    order = sorted(runs, key=key, reverse=desc)
    left = [('run', '<', lambda r: r), ('role', '<', lambda r: runs[r]['role']),
            ('L', '>', lambda r: '%d' % runs[r]['n_latent'])]
    if any(runs[r]['label'] != runs[r]['model'] for r in order):
        left.insert(1, ('label', '<', lambda r: runs[r]['label']))
    cells = {r: [get(r) for _, _, get in left]
             + [fmt(show, by[r][c][show]) for c in cond_ids] + ['%.4g' % key(r)] for r in order}
    names = [n for n, _, _ in left] + list(cond_ids) + [sort_label]
    align = [a for _, a, _ in left] + ['>'] * (len(cond_ids) + 1)
    widths = [max(len(names[k]), *(len(cells[r][k]) for r in order)) for k in range(len(names))]
    # ' | ' before each condition group and before the sort key, '  ' elsewhere
    seps, prev = [], None
    for k, name in enumerate(names):
        g = name.split('_')[0] if len(left) <= k < len(left) + len(cond_ids) else name
        seps.append('' if k == 0 else ' | ' if k >= len(left) and g != prev else '  ')
        prev = g

    def line(vals):
        return ''.join(s + '{:{}{}}'.format(v, a, w) for s, a, w, v in zip(seps, align, widths, vals))

    by_log = badness(sort_col, 10.0) != 10.0
    head = line(names)
    return (['%s; sorted by %s%s%s' % (show, sort_label,
                                       ' (mean |log10 r|, closest to 1 first)' if by_log else '',
                                       ', descending' if desc else ''),
             head, '-' * len(head)]
            + [line(cells[r]) for r in order])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split('\n\n', 1)[1])
    ap.add_argument('--sort', default='smape:mean', metavar='METRIC[:WHAT]')
    ap.add_argument('--show', metavar='METRIC', help='statistic in the columns (default: the sort metric)')
    ap.add_argument('--desc', action='store_true', help='largest first')
    ap.add_argument('--role', help='comma-separated roles: %s' % ', '.join(registry.ROLES))
    ap.add_argument('--set', dest='set_name', help='a [sets] entry of registry.toml')
    ap.add_argument('--runs', help='comma-separated run keys')
    ap.add_argument('--write', action='store_true',
                    help='recompute %s for every registered run' % metrics.METRICS_FILE.name)
    ap.add_argument('--csv', type=Path, default=metrics.METRICS_FILE, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    reg = registry.load()
    if args.write:
        metrics.write(reg.select(), args.csv)

    rows, warnings = metrics.read(args.csv, reg)
    for msg in warnings:
        print('warning: ' + msg, file=sys.stderr)
    if warnings:
        print('  -> python models.py --write\n', file=sys.stderr)

    wanted = {r.key for r in reg.select(
        keys=args.runs.split(',') if args.runs else None,
        role=args.role.split(',') if args.role else None,
        set_name=args.set_name)}
    cond_ids, by, runs = [], {}, {}
    for row in rows:
        if row['cond_id'] not in cond_ids:
            cond_ids.append(row['cond_id'])
        if row['run'] in wanted:
            by.setdefault(row['run'], {})[row['cond_id']] = row
            runs[row['run']] = row
    if not runs:
        raise SystemExit('no runs in %s match the selection' % args.csv)

    sort_col, sort_cols, sort_label = parse_sort(args.sort, cond_ids)
    show = metric_name(args.show) if args.show else sort_col
    for line in table(runs, cond_ids, by, show, sort_col, sort_cols, sort_label, args.desc):
        print(line)
    stamps = {row['timestamp'] for row in rows}
    print('\n%s, %s cells (%d), %s'
          % (', '.join(sorted(stamps)), rows[0]['smape_over'], int(rows[0]['n_cells']),
             args.csv))


if __name__ == '__main__':
    main()

"""data/metrics.csv: AeroMELD-vs-exact-answer error statistics, one row per run x condition.

Since 2026-11-12 the reference is the exact answer of the passive case
(paperfigs.reference), so only offline replay runs have rows; runs carried
online in WRF (no per-source tracers) are skipped. Before, the reference was
WRF-PartMC (partmc_diag_domain).

The statistics come from analysis.compute(), the function the violin figure
(plot/summary_symmetric_error.py) draws from, so the table and the figure
cannot disagree. Writing the table needs the products (aeromeld_domain for
every registered run, partmc_diag_domain, partmc_fields), but no GPU once they
exist. Reading it needs only the standard library, so anyone with the repo can
sort the models (models.py) without the products; when xarray and the products
are present, read() also checks that the products have not been rebuilt since.

Columns, in file order:

  registry     run, model, label, family, role, n_latent, emission_rate
               (copied from registry.toml when written; read() re-joins the
               current registry, so editing a label or role needs no rewrite)
  condition    timestamp, cond_id (e.g. ff_-30, the handle models.py sorts by),
               group, condition (the nominal value: s_env in %, wavelength in
               nm, temperature in deg C), condition_value (the value on the
               grid: temperatures snap to the 0.4 C grid, so -30 is -29.9),
               smape_over (the cells smape_pct and e_* describe)
  statistics   see analysis.condition_stats. Everything is over the cells
               smape_over names, except smape_unmixed_pct, smape_all_pct,
               ratio_median_all and pct_cells_ml_high_all. The ratio keeps
               AeroMELD's sign (non-positive values rank lowest; n_nonpositive
               counts them). Tables written before 2026-11-08 took ratio_* and
               pct_cells_ml_high over every cell and dropped non-positive values
  provenance   ml_product_created: the `created` attribute of the
               aeromeld_domain file the row was computed from (read() compares
               it with the file on disk); reference: 'exact:<replay>'.
"""
import csv
import os
from pathlib import Path

from . import config

METRICS_FILE = config.DATA_DIR / 'metrics.csv'

#: analysis.GROUPS key -> the prefix of cond_id.
GROUP_SHORT = {'ccn': 'ccn', 'scattering': 'scat', 'absorption': 'abs', 'frozen_fraction': 'ff'}

REGISTRY_COLUMNS = ('run', 'model', 'label', 'family', 'role', 'n_latent', 'emission_rate')
CONDITION_COLUMNS = ('timestamp', 'cond_id', 'group', 'condition', 'condition_value', 'smape_over')
#: (column, format) of each statistic, from analysis.condition_stats.
STAT_COLUMNS = (
    ('n_cells', '%d'), ('smape_pct', '%.4f'),
    ('e_p05', '%.6g'), ('e_median', '%.6g'), ('e_p95', '%.6g'), ('n_below_floor', '%d'),
    ('smape_unmixed_pct', '%.4f'), ('smape_all_pct', '%.4f'),
    ('ratio_cells', '%d'), ('ratio_p05', '%.8g'), ('ratio_median', '%.8g'),
    ('ratio_p95', '%.8g'), ('pct_cells_ml_high', '%.3f'), ('n_nonpositive', '%d'),
    ('n_pmc_nonpositive', '%d'), ('ratio_median_all', '%.8g'), ('pct_cells_ml_high_all', '%.3f'),
)
PROVENANCE_COLUMNS = ('ml_product_created', 'reference')
COLUMNS = (REGISTRY_COLUMNS + CONDITION_COLUMNS + tuple(c for c, _ in STAT_COLUMNS)
           + PROVENANCE_COLUMNS)
NUMERIC = {'n_latent', 'emission_rate', 'condition_value'} | {c for c, _ in STAT_COLUMNS}


def cond_id(group, tick):
    return '%s_%s' % (GROUP_SHORT[group], tick.replace('$', ''))


# --- writing (needs the products) ---------------------------------------------

def run_rows(run):
    """The rows of one registry Run, building its products if needed."""
    from . import analysis, products
    conds, stats, _, _ = analysis.compute(run, mixed_only=True)
    ml = products.attrs('aeromeld_domain', run=run.key)
    m = run.model
    rows = []
    for c, s in zip(conds, stats):
        row = {'run': run.key, 'model': m.key, 'label': m.label, 'family': m.family,
               'role': run.role, 'n_latent': m.n_latent, 'emission_rate': '%g' % run.emission_rate,
               'timestamp': ml['timestamp'], 'cond_id': cond_id(c['group'], c['tick']),
               'group': c['group'], 'condition': c['tick'].replace('$', ''),
               'condition_value': '%g' % c['condition'], 'smape_over': 'mixed',
               'ml_product_created': ml['created'], 'reference': 'exact:%s' % Path(run.wrf_dir).name}
        row.update({col: fmt % s[col] for col, fmt in STAT_COLUMNS})
        rows.append(row)
    return rows


def write(runs, path=METRICS_FILE):
    """Recompute the table for `runs` (registry Runs) and replace the file."""
    rows = []
    runs = [r for r in runs if has_reference(r)]
    for run in runs:
        print('%s ...' % run.key, flush=True)
        rows += run_rows(run)
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator='\n')
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)
    print('wrote %s (%d runs x %d conditions)' % (path, len(runs), len(rows) // max(len(runs), 1)))
    return rows


# --- reading (standard library only) ------------------------------------------

def read(path=METRICS_FILE, reg=None):
    """(rows, warnings). Numeric columns come back as float; the registry
    columns are replaced by the current registry's, and every way the file has
    fallen behind (registry edits, rebuilt products, runs missing) is a warning."""
    from . import registry
    reg = reg or registry.load()
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    warnings = []
    for row in rows:
        for k in NUMERIC:
            row[k] = float(row[k])

    def warn(msg):
        if msg not in warnings:
            warnings.append(msg)

    in_file = {row['run'] for row in rows}
    created = {}                 # (product, run) -> created, one file open each
    for row in rows:
        run = reg.runs.get(row['run'])
        if run is None:
            warn('%s is in %s but not in the registry' % (row['run'], Path(path).name))
            continue
        now = {'model': run.model.key, 'label': run.model.label, 'family': run.model.family,
               'role': run.role, 'n_latent': float(run.model.n_latent),
               'emission_rate': run.emission_rate}
        changed = sorted(k for k, v in now.items() if row[k] != v)
        if changed:
            warn('%s: registry %s changed since the table was written (using the registry\'s)'
                 % (row['run'], ', '.join(changed)))
            row.update(now)
        for col, name, key in (('ml_product_created', 'aeromeld_domain', {'run': row['run']}),):
            ck = (name, key.get('run'))
            if ck not in created:
                created[ck] = _created(name, **key)
            if created[ck] is not None and created[ck] != row[col]:
                warn('%s: the %s product was rebuilt after the table was written'
                     % (row['run'], name))
    for key in sorted(set(reg.runs) - in_file):
        if has_reference(reg.runs[key]):
            warn('%s is registered but not in %s' % (key, Path(path).name))
    return rows, warnings


def has_reference(run):
    """Whether a registry Run is an offline replay, which has the exact reference."""
    p = Path(run.wrf_dir)
    return p.parent.name == 'runs' and p.parent.parent.name == 'transport'


def _created(name, **key):
    """The product file's `created` attribute, or None if it is not on this
    machine (a collaborator with the repo but not the products)."""
    try:
        from . import products          # needs xarray, which a reader may not have
        return products.attrs(name, **key).get('created')
    except (FileNotFoundError, OSError, ImportError):
        return None

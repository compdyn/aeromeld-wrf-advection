"""Processed data products: load, build when missing or stale, and describe.

A product is one NetCDF file in config.PROCESSED_DIR, named
``<name>[_<key value>...].nc``, holding data in condensed form (one level, the
instants a figure uses, the sampling points over time). Plot scripts get data
only through ``load()``:

    ds = products.load('aeromeld_domain', run='05_nonlinvlong.0.0_5e7')

``load`` returns the product in memory (the file is closed). Whether it builds:

  * missing product        -> build it from the raw output; if the raw output is
                              not reachable, raise ProductMissing.
  * stale product          -> rebuild it, if the raw output is reachable;
                              otherwise warn and use it as it is.
  * up-to-date product     -> use it.

A product is stale when its builder's VERSION, its params (the analysis choices
it depends on, e.g. the instant, level, points, model seed) or any source file
it was built from (path, mtime, size) differ from what is stored in its
attributes. A source that does not exist on this machine is not checked -- that
is the collaborator case, who has the products but not the raw output.

Builders live under scripts/extract/ and are imported only to build, so loading
a product never imports torch, PyPartMC or partnn. A builder is a function
``build(**key) -> (xarray.Dataset, [source paths])``. It may load other
products; it then lists their files among its sources, so rebuilding an
upstream product makes its dependents stale.

Command line (from scripts/):

    python -m paperfigs.products list
    python -m paperfigs.products build grid
    python -m paperfigs.products build aeromeld_domain run=05_nonlinvlong.0.0_5e7
    python -m paperfigs.products check
    python -m paperfigs.products describe      # writes data/PRODUCTS.md
"""
import argparse
import datetime
import hashlib
import importlib
import json
import os
import subprocess
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Tuple

import numpy as np
import xarray as xr

from . import config


class ProductMissing(FileNotFoundError):
    pass


@dataclass(frozen=True)
class Spec:
    builder: str                       # 'module:function', importable from scripts/
    key: Tuple[str, ...] = ()          # key arguments; their values form the filename
    version: int = 1                   # bump whenever the builder's output changes
    params: Callable[..., dict] = field(default=lambda **key: {})
    doc: str = ''


SPECS = {}


def register(name, builder, *, key=(), version=1, params=None, doc=''):
    SPECS[name] = Spec(builder=builder, key=tuple(key), version=version,
                       params=params or (lambda **k: {}), doc=doc)


# --- the products --------------------------------------------------------------
#
# params are the analysis choices a product depends on that are NOT visible in
# its source files. Changing one (e.g. config.TIMESTAMP_FINAL) makes the product
# stale. Keep them cheap: config and registry only, no raw I/O.

def _run_params(run):
    from . import registry
    r = registry.load().run(run)
    return {'wrf_dir': r.wrf_dir, 'model': r.model.key, 'seed_idx': r.model.seed_idx}


register('grid', 'extract.grid:build',
         params=lambda: {},
         doc='XLAT/XLONG, map projection, emission source boxes.')

register('partmc_fields', 'extract.partmc:build_fields',
         params=lambda: {'level': config.LEVEL, 'locations': config.LOCATIONS,
                         'instants': [config.TIMESTAMP_START, config.TIMESTAMP_FINAL]},
         doc='WRF-PartMC aerosols file, surface: ntot and per-source number (per kg), ALT, '
             'at the start and final instants and at the sampling points over time; run-mean wind.')

register('partmc_diag_domain', 'extract.partmc:build_diag_domain',
         params=lambda: {'timestamp': config.TIMESTAMP_FINAL, 'diag_run': config.DIAG_RUN,
                         'diag_z': config.DIAG_Z},
         doc='PartMC domain diagnostics at the final instant: CCN/CN spectrum, frozen '
             'fraction, scattering and absorption (per kg dry air).')

register('partmc_diag_points', 'extract.partmc:build_diag_points',
         params=lambda: {'diag_run': config.DIAG_RUN, 'diag_z': config.DIAG_Z,
                         'locations': config.LOCATIONS, 'species': config.ML_SPECIES},
         doc='PartMC domain diagnostics at the sampling points, every diagnostics time: CCN/CN '
             'spectrum, frozen fraction, optics, number and per-species mass size '
             'distributions (densities, per kg dry air).')

register('aeromeld_points', 'extract.aeromeld:build_points', key=('run',),
         params=lambda run: dict(_run_params(run), level=config.LEVEL,
                                 locations=config.LOCATIONS, species=config.ML_SPECIES),
         doc='WRF-ML latents at the sampling points, every wrfout time, decoded with the '
             'run\'s model: CCN/CN spectrum, frozen fraction, optics, number and per-species '
             'mass size distributions (densities, per kg dry air).')

register('wrfml_fields', 'extract.wrfml:build_fields', key=('run',),
         params=lambda run: dict(_run_params(run), timestamp=config.TIMESTAMP_FINAL,
                                 level=config.LEVEL, locations=config.LOCATIONS),
         doc='WRF-ML latents and ntot (not decoded): surface at the final instant, and the '
             'sampling points at every wrfout time.')

def _model_params(model):
    from . import registry
    return {'seed_idx': registry.load().model(model).seed_idx}


register('populations', 'extract.populations:build', key=('model',),
         params=lambda model: dict(_model_params(model), scenarios=config.POP_SCENARIOS,
                                   particle_time=config.PARTICLE_TIME,
                                   ntot_latent=config.POP_NTOT_LATENT,
                                   species=config.ML_SPECIES),
         doc='The three seed populations\' 1000-particle files, encoded and decoded with one '
             'model (latents, reconstructions at ntot latent 0), and the same particles\' PartMC '
             'curves on the file\'s own weights.')

register('aeromeld_domain', 'extract.aeromeld:build_domain', key=('run',),
         params=lambda run: dict(_run_params(run), timestamp=config.TIMESTAMP_FINAL,
                                 level=config.LEVEL),
         doc='WRF-ML latents decoded with the run\'s model at the final instant: CCN/CN '
             'spectrum, frozen fraction, scattering and absorption (per kg dry air).')


# --- paths and provenance -------------------------------------------------------

def _spec(name):
    try:
        return SPECS[name]
    except KeyError:
        raise KeyError('unknown product %r; known: %s' % (name, ', '.join(sorted(SPECS)))) from None


def _check_key(spec, name, key):
    if set(key) != set(spec.key):
        raise TypeError('product %r takes key %s, got %s'
                        % (name, list(spec.key) or '(none)', sorted(key) or '(none)'))


def path(name, **key):
    spec = _spec(name)
    _check_key(spec, name, key)
    stem = '_'.join([name] + [str(key[k]) for k in spec.key])
    return config.PROCESSED_DIR / (stem + '.nc')


def _jsonable(obj):
    """Round-trip through JSON so tuples/lists and numpy scalars compare equal."""
    return json.loads(json.dumps(obj, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))


def _stat(p):
    p = os.path.abspath(p)            # not resolve(): keep the path as given
    st = os.stat(p)
    return [p, round(st.st_mtime, 3), st.st_size]


def _git_state():
    try:
        rev = subprocess.run(['git', '-C', str(config.REPO), 'rev-parse', '--short', 'HEAD'],
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(['git', '-C', str(config.REPO), 'status', '--porcelain', 'scripts'],
                               capture_output=True, text=True, check=True).stdout.strip()
        return rev + ('-dirty' if dirty else '')
    except Exception:
        return 'unknown'


def stale_reason(name, p=None, **key):
    """None if the product at `p` is up to date, else a short reason.

    Sources missing on this machine are skipped (see the module docstring)."""
    spec = _spec(name)
    p = Path(p) if p is not None else path(name, **key)
    with xr.open_dataset(p) as ds:
        attrs = dict(ds.attrs)
    if int(attrs.get('product_version', -1)) != spec.version:
        return 'builder version %s -> %d' % (attrs.get('product_version'), spec.version)
    stored = json.loads(attrs.get('params', '{}'))
    current = _jsonable(spec.params(**key))
    if stored != current:
        changed = sorted(k for k in set(stored) | set(current) if stored.get(k) != current.get(k))
        return 'params changed: %s' % ', '.join(changed)
    for src, mtime, size in json.loads(attrs.get('sources', '[]')):
        if os.path.exists(src):
            st = os.stat(src)
            if round(st.st_mtime, 3) != mtime or st.st_size != size:
                return 'source changed: %s' % src
    return None


def _sources_reachable(p):
    with xr.open_dataset(p) as ds:
        sources = json.loads(ds.attrs.get('sources', '[]'))
    return any(os.path.exists(s) for s, _, _ in sources)


def _read(p):
    with xr.open_dataset(p) as ds:
        return ds.load()


def attrs(name, **key):
    """A product file's global attributes, without loading its data (or building it)."""
    with xr.open_dataset(path(name, **key)) as ds:
        return dict(ds.attrs)


# --- load / build ---------------------------------------------------------------

def load(name, *, rebuild=False, **key):
    """The product as an in-memory xarray.Dataset, built first if needed."""
    p = path(name, **key)
    if p.exists() and not rebuild:
        reason = stale_reason(name, p, **key)
        if reason is None:
            return _read(p)
        if not _sources_reachable(p):
            warnings.warn('%s is stale (%s) but its raw inputs are not on this machine; '
                          'using it as it is' % (p.name, reason), stacklevel=2)
            return _read(p)
        print('%s: stale (%s); rebuilding' % (p.name, reason), flush=True)
    return build(name, **key)


def build(name, **key):
    spec = _spec(name)
    _check_key(spec, name, key)
    p = path(name, **key)
    if str(config.SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(config.SCRIPTS_DIR))
    module, func = spec.builder.split(':')
    builder = getattr(importlib.import_module(module), func)

    print('building %s ...' % p.name, flush=True)
    t0 = datetime.datetime.now()
    try:
        ds, sources = builder(**key)
    except FileNotFoundError as e:
        raise ProductMissing(
            '%s cannot be built here: %s\nAsk for %s and put it in %s, or point the '
            'PAPERFIGS_* paths in paperfigs/config.py at the raw output.'
            % (p.name, e, p.name, config.PROCESSED_DIR)) from e

    ds.attrs.update({
        'product': name,
        'product_key': json.dumps(key, sort_keys=True),
        'product_version': spec.version,
        'params': json.dumps(_jsonable(spec.params(**key)), sort_keys=True),
        'sources': json.dumps([_stat(s) for s in sources]),
        'created': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
        'built_by': 'paper_wrf_ml_advection@%s %s' % (_git_state(), spec.builder),
        'description': spec.doc,
    })
    encoding = {v: {'zlib': True, 'complevel': 4}
                for v in ds.data_vars if ds[v].dtype.kind in 'fiu'}
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.tmp')
    ds.to_netcdf(tmp, encoding=encoding)
    os.replace(tmp, p)
    print('  wrote %s (%.1f MB) in %.0f s'
          % (p.name, p.stat().st_size / 1e6, (datetime.datetime.now() - t0).total_seconds()),
          flush=True)
    return _read(p)


# --- description (data/PRODUCTS.md) ---------------------------------------------

def _sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def describe(out=config.DATA_DIR / 'PRODUCTS.md'):
    """Write a Markdown description of every product file: what it holds, on
    what basis, built from what -- readable without opening the binaries."""
    lines = ['# Processed data products', '',
             'Generated by `python -m paperfigs.products describe`; do not edit.',
             'Files live in `data/processed/` (not committed). The checksums identify',
             'the exact files a figure was made from.', '']
    for p in sorted(config.PROCESSED_DIR.glob('*.nc')):
        with xr.open_dataset(p) as ds:
            a = ds.attrs
            lines += ['## `%s`' % p.name, '',
                      a.get('description', ''), '',
                      '- product `%s`, key `%s`, version %s'
                      % (a.get('product'), a.get('product_key'), a.get('product_version')),
                      '- params `%s`' % a.get('params'),
                      '- built %s by `%s`' % (a.get('created'), a.get('built_by')),
                      '- %.2f MB, sha256 `%s`' % (p.stat().st_size / 1e6, _sha256(p)),
                      '- sources: ' + ', '.join('`%s`' % s for s, _, _ in
                                                json.loads(a.get('sources', '[]'))), '']
            skip = {'product', 'product_key', 'product_version', 'params', 'sources',
                    'created', 'built_by', 'description'}
            other = {k: v for k, v in a.items() if k not in skip}
            if other:
                lines += ['Attributes:', ''] + ['- `%s` = %s' % kv for kv in other.items()] + ['']
            lines += ['| variable | dims | dtype | units | description |',
                      '|---|---|---|---|---|']
            for v in list(ds.coords) + list(ds.data_vars):
                da = ds[v]
                desc = da.attrs.get('long_name', '')
                if 'basis' in da.attrs:
                    desc += ' (basis: %s)' % da.attrs['basis']
                lines.append('| `%s` | %s | %s | %s | %s |'
                             % (v, ' × '.join('%s=%d' % (d, ds.sizes[d]) for d in da.dims) or '—',
                                da.dtype, da.attrs.get('units', ''), desc))
            lines.append('')
    Path(out).write_text('\n'.join(lines))
    print('wrote', out)


# --- command line ---------------------------------------------------------------

def _parse_key(items):
    key = {}
    for item in items:
        k, sep, v = item.partition('=')
        if not sep:
            raise SystemExit('key arguments are name=value, got %r' % item)
        key[k] = v
    return key


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('list', help='known products and the files present')
    b = sub.add_parser('build', help='build (or rebuild) one product')
    b.add_argument('name')
    b.add_argument('key', nargs='*', help='key arguments as name=value')
    sub.add_parser('check', help='report stale product files')
    sub.add_parser('describe', help='write data/PRODUCTS.md')
    args = ap.parse_args(argv)

    if args.cmd == 'list':
        for name, spec in sorted(SPECS.items()):
            print('%-22s key=%-8s v%d  %s' % (name, ','.join(spec.key) or '-', spec.version, spec.doc))
        files = sorted(config.PROCESSED_DIR.glob('*.nc'))
        print('\n%d files in %s' % (len(files), config.PROCESSED_DIR))
        for p in files:
            print('  %-50s %8.2f MB' % (p.name, p.stat().st_size / 1e6))
    elif args.cmd == 'build':
        build(args.name, **_parse_key(args.key))
    elif args.cmd == 'check':
        for p in sorted(config.PROCESSED_DIR.glob('*.nc')):
            with xr.open_dataset(p) as ds:
                name, key = ds.attrs.get('product'), json.loads(ds.attrs.get('product_key', '{}'))
            reason = stale_reason(name, p, **key) if name in SPECS else 'unknown product'
            print('%-50s %s' % (p.name, reason or 'up to date'))
    elif args.cmd == 'describe':
        describe()


if __name__ == '__main__':
    main()

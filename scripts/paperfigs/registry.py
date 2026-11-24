"""The registry of ML models and WRF-ML runs (registry.toml).

A run's directory name is not a reliable identity: the bare
``latent_advection_cares_v3match`` names no model, two runs can share a model
and differ only in emission rate, and the rate is in the name only for runs
built after October 2026. So every product, figure and table refers to runs and
models by their registry KEY, and everything else about them is looked up here.

Two kinds of field live in registry.toml:

  hand-typed   label, family, notes, role, [sets]
  derived      n_latent, emission_rate -- written by register_runs.py from the
               run's own files, and re-checked by verify() whenever an extractor
               uses the run. Never edit these by hand.

Loading the registry needs only the standard library, so plot scripts can sort
and label runs without the raw output. verify() needs netCDF4 and the run dirs.
"""
import re
import sys
from dataclasses import dataclass
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:                                     # pragma: no cover
    import tomli as tomllib

from . import config

ROLES = ('main', 'appendix', 'exploratory', 'superseded')

#: fpidx is '<name>.<file>.<part>': archive <name>_<file:02d>.h5, group P<part:08d>.
_FPIDX_RE = re.compile(r'^(?P<name>\d\d_[A-Za-z0-9]+)\.(?P<file>\d+)\.(?P<part>\d+)$')


class RegistryError(ValueError):
    pass


@dataclass(frozen=True)
class Model:
    key: str                 # fpidx, e.g. '05_nonlinvlong.0.0'
    label: str
    family: str
    notes: str
    seed_idx: int
    n_latent: int            # derived: z dimensions, ntot not counted (9 or 49)
    backend: str = 'partnn'  # 'partnn' (code_partnn archives) or 'affx' (training/, numpy files)

    @property
    def checkpoint(self):
        """The partnn training archive holding this model's weights; for an affx model,
        its numpy file data/models/<name>.<seed>.npz (key <name>.<seed>.0)."""
        if self.backend == 'affx':
            name, seed, _ = self.key.split('.')
            return config.DATA_DIR / 'models' / ('%s.%s.npz' % (name, seed))
        m = _FPIDX_RE.match(self.key)
        if m is None:
            raise RegistryError('model key %r is not an fpidx <name>.<file>.<part>' % self.key)
        return config.MODEL_RESDIR / ('%s_%02d.h5' % (m['name'], int(m['file'])))


@dataclass(frozen=True)
class Run:
    key: str                 # e.g. '05_nonlinvlong.0.0_5e7'
    model: Model
    wrf_dir: str
    role: str
    notes: str
    emission_rate: float     # derived: max E_NTOT of the run's wrfchemi

    @property
    def label(self):
        return self.model.label

    @property
    def path(self):
        return config.WRF_ML_TEST / self.wrf_dir

    @property
    def wrfout(self):
        return self.path / config.WRFOUT_NAME

    @property
    def wrfchemi(self):
        return self.path / config.WRFCHEMI_NAME

    @property
    def extract(self):
        """affx runs: the run's small extract (training/wrf/extract_runs.py), which the
        products are built from; the wrfout itself stays on Perlmutter."""
        return config.DATA_DIR / 'wrfml_runs' / ('%s.npz' % self.key)


class Registry:
    def __init__(self, models, runs, sets):
        self.models = models
        self.runs = runs
        self.sets = sets

    def run(self, key):
        try:
            return self.runs[key]
        except KeyError:
            raise RegistryError('no run %r in the registry; have: %s'
                                % (key, ', '.join(sorted(self.runs)))) from None

    def model(self, key):
        try:
            return self.models[key]
        except KeyError:
            raise RegistryError('no model %r in the registry; have: %s'
                                % (key, ', '.join(sorted(self.models)))) from None

    def main(self):
        mains = [r for r in self.runs.values() if r.role == 'main']
        if len(mains) != 1:
            raise RegistryError('exactly one run must have role = "main"; found %d (%s)'
                                % (len(mains), ', '.join(r.key for r in mains)))
        return mains[0]

    def select(self, keys=None, role=None, set_name=None):
        """Runs by explicit keys, by role(s), or by named set -- in that order of
        precedence; with none of them, every registered run."""
        if keys:
            return [self.run(k) for k in keys]
        if set_name:
            if set_name not in self.sets:
                raise RegistryError('no set %r; have: %s' % (set_name, ', '.join(self.sets)))
            return [self.run(k) for k in self.sets[set_name]]
        runs = sorted(self.runs.values(), key=lambda r: r.key)
        if role:
            roles = {role} if isinstance(role, str) else set(role)
            runs = [r for r in runs if r.role in roles]
        return runs


def load(path=config.REGISTRY_FILE):
    with open(path, 'rb') as f:
        raw = tomllib.load(f)
    models = {}
    for key, m in raw.get('models', {}).items():
        models[key] = Model(key=key, label=m.get('label', key), family=m.get('family', ''),
                            notes=m.get('notes', ''), seed_idx=int(m.get('seed_idx', 0)),
                            n_latent=int(m['n_latent']), backend=m.get('backend', 'partnn'))
    runs = {}
    for key, r in raw.get('runs', {}).items():
        if r['model'] not in models:
            raise RegistryError('run %r refers to unknown model %r' % (key, r['model']))
        role = r.get('role', 'exploratory')
        if role not in ROLES:
            raise RegistryError('run %r has role %r; must be one of %s' % (key, role, ROLES))
        runs[key] = Run(key=key, model=models[r['model']], wrf_dir=r['wrf_dir'], role=role,
                        notes=r.get('notes', ''), emission_rate=float(r['emission_rate']))
    sets = {k: list(v) for k, v in raw.get('sets', {}).items()}
    for name, members in sets.items():
        unknown = [k for k in members if k not in runs]
        if unknown:
            raise RegistryError('set %r lists unknown runs: %s' % (name, ', '.join(unknown)))
    return Registry(models, runs, sets)


# --- deriving and checking (needs netCDF4 and the run directory) -------------

def derive(run_dir):
    """{'n_latent', 'emission_rate', 'completed'} read from a run's own files."""
    import numpy as np
    from netCDF4 import Dataset

    run_dir = Path(run_dir)
    with Dataset(run_dir / config.WRFOUT_NAME) as f:
        n = 0
        while 'ltnt_var%d' % (n + 1) in f.variables:
            n += 1
    with Dataset(run_dir / config.WRFCHEMI_NAME) as f:
        rate = float(np.max(f['E_NTOT'][:]))
    return {'n_latent': n, 'emission_rate': rate, 'completed': run_completed(run_dir)}


def run_completed(run_dir):
    """WRF can exit 0 after a failure, so trust only rsl's last word."""
    rsl = Path(run_dir) / 'rsl.out.0000'
    wrfout = Path(run_dir) / config.WRFOUT_NAME
    if not rsl.exists() or not wrfout.exists() or wrfout.stat().st_size == 0:
        return False
    with open(rsl, 'rb') as f:
        f.seek(max(0, rsl.stat().st_size - 4096))
        return b'SUCCESS COMPLETE WRF' in f.read()


def verify_extract(run):
    """verify() for an affx run, from its extract (the wrfout stays on Perlmutter; the
    extractor took only runs whose rsl.error.0000 ends in SUCCESS COMPLETE WRF)."""
    import numpy as np
    problems = []
    if not run.extract.exists():
        raise RegistryError('run %s: no extract %s' % (run.key, run.extract))
    with np.load(run.extract) as f:
        n = f['z_dom'].shape[1]
        fp = str(f['fpidx'])
    if n != run.model.n_latent:
        problems.append('extract has %d latents, model %s is registered with %d'
                        % (n, run.model.key, run.model.n_latent))
    if fp != run.model.key:
        problems.append('extract was built from model %s, registry says %s' % (fp, run.model.key))
    if not run.model.checkpoint.exists():
        problems.append('model file %s does not exist' % run.model.checkpoint)
    if problems:
        raise RegistryError('run %s (%s):\n  %s' % (run.key, run.extract, '\n  '.join(problems)))
    return {'n_latent': n, 'emission_rate': run.emission_rate, 'completed': True}


def verify(run):
    """Raise RegistryError unless the run's files agree with its registry entry."""
    if run.model.backend == 'affx':
        return verify_extract(run)
    got = derive(run.path)
    problems = []
    if not got['completed']:
        problems.append('rsl.out.0000 does not end in SUCCESS COMPLETE WRF')
    if got['n_latent'] != run.model.n_latent:
        problems.append('wrfout has %d latents, model %s is registered with %d'
                        % (got['n_latent'], run.model.key, run.model.n_latent))
    if not abs(got['emission_rate'] - run.emission_rate) <= 1e-6 * run.emission_rate:
        problems.append('wrfchemi E_NTOT is %g, registry says %g'
                        % (got['emission_rate'], run.emission_rate))
    if not run.model.checkpoint.exists():
        problems.append('model archive %s does not exist' % run.model.checkpoint)
    if problems:
        raise RegistryError('run %s (%s):\n  %s' % (run.key, run.path, '\n  '.join(problems)))
    return got

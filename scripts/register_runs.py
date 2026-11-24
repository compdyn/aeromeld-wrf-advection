#!/usr/bin/env python
"""Find WRF-ML runs that are not in the registry yet, and add stub entries.

Usage (from scripts/, partnn env):

    python register_runs.py            # list what would be added
    python register_runs.py --write    # append the stubs to paperfigs/registry.toml
    python register_runs.py --check    # verify every registered run against its files

A stub has its derived fields (n_latent, emission_rate) read from the run's own
files, role = "exploratory", and label = the model key. Edit label, family,
role and notes by hand afterwards. Runs are skipped when WRF did not finish
(rsl does not end in SUCCESS COMPLETE WRF) or when their emission rate is not
config.EMISSION_RATE.

The model is guessed from the directory name, the only place it is recorded
(latent_advection_cares_v3match[_<fpidx>][_<rate>], with no fpidx meaning
02_mlpprt.0.0). Check the guess before writing.
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paperfigs import config, registry  # noqa: E402

DEFAULT_FPIDX = '02_mlpprt.0.0'
_RUN_RE = re.compile(r'^%s(?:_(?P<fpidx>\d\d_[A-Za-z0-9]+\.\d+\.\d+))?'
                     r'(?:_(?P<rate>\d+(?:\.\d+)?e\d+))?$' % re.escape(config.WRF_ML_RUN_PREFIX))


def short_rate(x):
    """5e7, 2.5e7 -- the way run names spell a rate."""
    import math
    exp = int(math.floor(math.log10(x)))
    return '%ge%d' % (round(x / 10 ** exp, 6), exp)


def discover():
    out = []
    for d in sorted(config.WRF_ML_TEST.glob(config.WRF_ML_RUN_PREFIX + '*')):
        m = _RUN_RE.match(d.name)
        if d.is_dir() and m:
            out.append((d, m['fpidx'] or DEFAULT_FPIDX))
    return out


def stubs(reg):
    known_dirs = {r.wrf_dir for r in reg.runs.values()}
    new_models, new_runs = {}, {}
    for d, fpidx in discover():
        if d.name in known_dirs:
            continue
        got = registry.derive(d)
        if not got['completed']:
            print('skip %s: not completed' % d.name)
            continue
        if abs(got['emission_rate'] - config.EMISSION_RATE) > 1e-6 * config.EMISSION_RATE:
            print('skip %s: emission rate %g, not %g'
                  % (d.name, got['emission_rate'], config.EMISSION_RATE))
            continue
        key = '%s_%s' % (fpidx, short_rate(got['emission_rate']))
        if key in reg.runs or key in new_runs:
            print('skip %s: key %s is already taken' % (d.name, key))
            continue
        if fpidx in reg.models:
            if reg.models[fpidx].n_latent != got['n_latent']:
                print('skip %s: %d latents, but model %s is registered with %d'
                      % (d.name, got['n_latent'], fpidx, reg.models[fpidx].n_latent))
                continue
        elif fpidx not in new_models:
            new_models[fpidx] = got['n_latent']
        new_runs[key] = (fpidx, d.name, got['emission_rate'])
    return new_models, new_runs


def toml_text(new_models, new_runs):
    out = []
    for fpidx, n_latent in sorted(new_models.items()):
        out += ['', '[models."%s"]' % fpidx,
                'label = "%s"' % fpidx,
                'family = ""',
                'notes = ""',
                'seed_idx = 0',
                'n_latent = %d          # derived: z dimensions in the wrfout (ntot not counted)'
                % n_latent]
    for key, (fpidx, wrf_dir, rate) in sorted(new_runs.items()):
        out += ['', '[runs."%s"]' % key,
                'model = "%s"' % fpidx,
                'wrf_dir = "%s"' % wrf_dir,
                'role = "exploratory"',
                'notes = ""',
                'emission_rate = %g    # derived: max E_NTOT of the wrfchemi' % rate]
    return '\n'.join(out) + '\n'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--write', action='store_true', help='append the stubs to registry.toml')
    ap.add_argument('--check', action='store_true', help='verify every registered run')
    args = ap.parse_args(argv)

    reg = registry.load()
    if args.check:
        bad = 0
        for run in reg.select():
            try:
                registry.verify(run)
                print('ok   %-32s %-10s %s' % (run.key, run.role, run.wrf_dir))
            except registry.RegistryError as e:
                bad += 1
                print('FAIL %s' % e)
        reg.main()
        sys.exit(1 if bad else 0)

    new_models, new_runs = stubs(reg)
    if not new_runs:
        print('nothing to add')
        return
    text = toml_text(new_models, new_runs)
    print(text)
    if args.write:
        with open(config.REGISTRY_FILE, 'a') as f:
            f.write(text)
        registry.load()        # must still parse
        print('appended %d run(s), %d model(s) to %s'
              % (len(new_runs), len(new_models), config.REGISTRY_FILE))
        print('then: python models.py --write   (adds them to data/metrics.csv)')
    else:
        print('(dry run; --write to append)')


if __name__ == '__main__':
    main()

#!/usr/bin/env python
"""Pack everything a WRF-AeroMELD build needs from the authors' WRF-AeroMELD tree into one directory.

Usage (from scripts/, partnn env):

    python inputs/make_bundle.py --out DIR [--models FPIDX,...] [--tar]

With the bundle, this repository and a compiled WRF-AeroMELD, build_run.py runs
anywhere: set PAPERFIGS_INPUT_BUNDLE=DIR and pass --wrf-dir. The bundle holds

    templates/   wrfinput_d01_cares, wrfbdy_d01_cares          (9 latents + ntot)
                 wrfinput_d01_cares_l49, wrfbdy_d01_cares_l49  (49 latents + ntot)
                 namelist.input
                 -- the real.exe output the runs start from (meteorology and grid)
                 and the run's namelist; the wrfchemi grid comes from
                 wrfinput_d01_cares too
    particles/   scenario_{561,635,812}_filtered_0001_00000013.nc
                 -- the 1000-particle populations WRF-PartMC was initialised from
    models/      <name>_<NN>.h5 for each model in --models
    MANIFEST.json  every file's size and md5, and where it was copied from
    README.txt

--models defaults to every model with stored latents in reference_inputs.json
(so a fresh encoding of each can be checked), except archives over 1 GB
(02_mlpprt's is 17 GB). About 95 MB of templates, 1 MB of particles, and
7-24 MB per model.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paperfigs import config, registry  # noqa: E402

TEMPLATES = ('wrfinput_d01_cares', 'wrfbdy_d01_cares',
             'wrfinput_d01_cares_l49', 'wrfbdy_d01_cares_l49', 'namelist.input')
MAX_ARCHIVE = 1e9


def archive_name(fpidx):
    m = registry._FPIDX_RE.match(fpidx)
    if m is None:
        raise SystemExit('%r is not an fpidx <name>.<file>.<part>' % fpidx)
    return '%s_%02d.h5' % (m['name'], int(m['file']))


def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def default_models():
    with open(config.INPUT_FINGERPRINTS) as f:
        known = json.load(f)['latents']
    keep = []
    for fpidx in known:
        src = config.MODEL_RESDIR / archive_name(fpidx)
        if src.exists() and src.stat().st_size <= MAX_ARCHIVE:
            keep.append(fpidx)
        else:
            print('  leaving out %s (%s)' % (fpidx, 'no archive' if not src.exists()
                                            else '%.0f GB archive' % (src.stat().st_size / 1e9)))
    return keep


def git_commit(path):
    try:
        return subprocess.run(['git', '-C', str(path), 'rev-parse', '--short', 'HEAD'],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True, type=Path, help='bundle directory to create')
    ap.add_argument('--models', help='comma-separated fpidx list (default: see above)')
    ap.add_argument('--tar', action='store_true', help='also write OUT.tar.gz')
    args = ap.parse_args()
    if config.INPUT_BUNDLE is not None:
        sys.exit('PAPERFIGS_INPUT_BUNDLE is set; unset it so the sources are the originals')
    out = args.out.absolute()
    if out.exists():
        sys.exit('%s exists; refusing to touch it' % out)
    models = args.models.split(',') if args.models else default_models()

    plan = [('templates/' + n, config.TEMPLATE_DIR / n) for n in TEMPLATES]
    plan += [('particles/scenario_%d_filtered_0001_%08d.nc' % (s, config.PARTICLE_TIME),
              config.PARTICLE_DIR / ('scenario_%d_filtered_0001_%08d.nc' % (s, config.PARTICLE_TIME)))
             for s in config.POP_SCENARIOS]
    plan += [('models/' + archive_name(m), config.MODEL_RESDIR / archive_name(m)) for m in models]
    missing = [str(src) for _, src in plan if not src.exists()]
    if missing:
        sys.exit('missing sources:\n  ' + '\n  '.join(missing))

    partial = out.with_name(out.name + '.partial')
    shutil.rmtree(partial, ignore_errors=True)
    files = {}
    for rel, src in plan:
        dst = partial / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        files[rel] = {'bytes': dst.stat().st_size, 'md5': md5(dst), 'from': str(src)}
        print('  %-52s %8.1f MB' % (rel, files[rel]['bytes'] / 1e6))
    manifest = {
        'created': time.strftime('%Y-%m-%d %H:%M'),
        'paper_repo_commit': git_commit(config.REPO),
        'wrf_commit': git_commit(config.WRF_DIR),
        'models': models,
        'files': files,
    }
    (partial / 'MANIFEST.json').write_text(json.dumps(manifest, indent=1) + '\n')
    (partial / 'README.txt').write_text(
        'WRF-AeroMELD input bundle, made by paper_wrf_ml_advection/scripts/inputs/make_bundle.py.\n'
        'Use it with that repository (scripts/inputs/README.md, "Running elsewhere"):\n\n'
        '    export PAPERFIGS_INPUT_BUNDLE=%s\n'
        '    python inputs/build_run.py <fpidx> --work <workspace> --wrf-dir <your WRF-AeroMELD>\n\n'
        'Models: %s\nMANIFEST.json lists every file with its md5.\n'
        'WRF-AeroMELD was at commit %s of the WRF-AeroMELD repository when this was made.\n'
        % ('<this directory>', ', '.join(models), manifest['wrf_commit']))
    partial.rename(out)
    total = sum(f['bytes'] for f in files.values())
    print('wrote %s: %d files, %.0f MB' % (out, len(files), total / 1e6))
    if args.tar:
        tgz = out.with_name(out.name + '.tar.gz')
        with tarfile.open(tgz, 'w:gz') as tar:
            tar.add(out, arcname=out.name)
        print('wrote %s (%.0f MB)' % (tgz, tgz.stat().st_size / 1e6))
    return 0


if __name__ == '__main__':
    sys.exit(main())

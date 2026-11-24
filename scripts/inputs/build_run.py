#!/usr/bin/env python
"""Build a complete WRF-AeroMELD run for one VAE model, in your own workspace.

Usage (from scripts/, partnn env with LD_LIBRARY_PATH=$CONDA_PREFIX/lib):

    python inputs/build_run.py FPIDX [--rate 5e7] [--work DIR] [--wrf-dir DIR]   # build, then check
    python inputs/build_run.py --status [--work DIR]                              # every run in the workspace

Inputs: the model archive, the three particle files, the real.exe templates and
namelist.input (config.MODEL_RESDIR, PARTICLE_DIR, TEMPLATE_DIR; all three come
from an input bundle when PAPERFIGS_INPUT_BUNDLE is set, see make_bundle.py),
and a compiled WRF-AeroMELD tree (--wrf-dir, default config.WRF_DIR).

--work defaults to $AEROMELD_WORK. Nothing outside it is written. For FPIDX at
RATE (the emission box's E_NTOT, # m^-2 s^-1) it creates

    WORK/latents/FPIDX/latent_scenario_*.txt     encoded once per model, reused by every rate
    WORK/emis/FPIDX_RATE/wrfchemi_d01_*           13 hourly emission files (00-12 UTC)
    WORK/runs/latent_advection_cares_v3match_FPIDX_RATE/
        wrfinput_d01, wrfbdy_d01, namelist.input, batch.sh, and links to the
        WRF tree's physics tables (as in its test/em_real/), to WORK/bin/wrf.exe
        and to the emissions
    WORK/bin/wrf.exe (+ wrf.exe.md5)              frozen copy of WRF_DIR/main/wrf.exe, made once

Steps: encode (encode_populations.py; a model with stored latents is checked
against them) -> wrfinput/wrfbdy (run_latent_ics.sh) -> wrfchemi
(run_latent_emis.sh) -> run directory -> every non-latent field checked against
the committed reference checksums (reference.py). The scenario (populations,
boxes, number concentrations) is whatever the two run_latent_*.sh wrappers hold;
this script only points them at the latents, the templates and the workspace.

Run directories are named the way the runs under config.WRF_ML_TEST are
(<prefix>_<fpidx>_<rate>), the naming register_runs.py reads.

All or nothing: if a step fails, the run and emission directories this call
created are removed again; complete latents are kept. An existing run or
emission directory is never touched. Submitting is left to you: the script
prints the sbatch command.
"""
import argparse
import hashlib
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from paperfigs import config  # noqa: E402
import encode_populations  # noqa: E402
import reference  # noqa: E402

#: WRF's two latent options: template suffix, chem_opt, emiss_opt. WRF aborts
#: at start-up (chemics_init) on any other pairing.
LATENT_OPTIONS = {9: ('', 900, 21), 49: ('_l49', 901, 22)}

#: The default batch.sh, for the authors' cluster: 96 ranks on 2 nodes (sesempi needs at
#: least 2); the 4-h run takes ~6 min. Elsewhere pass --batch-template with your
#: own; {name}, {run} and {mail} in it are replaced (nothing else is touched,
#: so shell ${...} is safe).
BATCH = """#!/bin/bash
#SBATCH --job-name=wrf_{name}
#SBATCH -p sesempi
#SBATCH -n 96
#SBATCH -N 2
#SBATCH --sockets-per-node=2
#SBATCH --cores-per-socket=24
#SBATCH --time=10:00:00
#SBATCH --mem-per-cpu=5g
{mail}
# wrf.exe is linked against the system OpenMPI (module mpi/openmpi-x86_64).
export PATH=/usr/lib64/openmpi/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib64/openmpi/lib:${LD_LIBRARY_PATH:-}
cd {run}
mpirun -np 96 ./wrf.exe
"""

#: Submit without the enclosing SLURM job's variables (a JupyterLab or GPU
#: session), which sbatch would otherwise read as options. Harmless outside one.
SUBMIT = ("cd {run} && env $(compgen -e | grep -E '^(SLURM|SBATCH|SALLOC)_' | sed 's/^/-u /') "
          "sbatch batch.sh")


def short_rate(x):
    """5e7, 2.5e7: how run names spell a rate (as register_runs.py does)."""
    exp = int(math.floor(math.log10(x)))
    return '%ge%d' % (round(x / 10 ** exp, 6), exp)


def workspace(arg):
    work = arg or os.environ.get('AEROMELD_WORK')
    if not work:
        sys.exit('give --work DIR or set AEROMELD_WORK (a writable directory on /data, '
                 'not /tmp or $HOME)')
    return Path(work).absolute()


def run_path(work, name):
    return work / 'runs' / ('%s_%s' % (config.WRF_ML_RUN_PREFIX, name))


def n_latent_of(lat_dir):
    path = encode_populations.latent_path(lat_dir, config.POP_SCENARIOS[0])
    return len(path.read_text().split()) - 1


def run_wrapper(script, settings):
    """Run one of the run_latent_*.sh wrappers with this Python first on PATH."""
    env = os.environ.copy()
    env['PATH'] = str(Path(sys.executable).parent) + os.pathsep + env.get('PATH', '')
    env.update({k: str(v) for k, v in settings.items()})
    r = subprocess.run([str(HERE / script)], env=env, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, text=True)
    for line in r.stdout.splitlines():
        if line.startswith('Next:'):        # run_latent_emis.sh's hint about ITS own run dir
            break
        if line.strip() and 'PROJ: proj_create' not in line and 'wrote wrfchemi_d01_' not in line:
            print('    ' + line)
    if r.returncode:
        raise RuntimeError('%s failed (exit %d)' % (script, r.returncode))


def wrf_tables(wrf_dir):
    """{name: target} of the physics-table links WRF's compile made in test/em_real/.

    Those are exactly the files an em_real run needs (and every run so far had);
    the executables are left out, wrf.exe comes from the workspace's copy.
    """
    em_real = Path(wrf_dir) / 'test' / 'em_real'
    links = {f.name: os.path.realpath(f) for f in sorted(em_real.glob('*'))
             if f.is_symlink() and not f.name.endswith('.exe')}
    if not links:
        sys.exit('no table links in %s; compile WRF (./compile em_real) first' % em_real)
    broken = [n for n, t in links.items() if not os.path.exists(t)]
    if broken:
        sys.exit('broken links in %s: %s' % (em_real, ', '.join(broken)))
    return links


def frozen_wrf_exe(work, src):
    exe = work / 'bin' / 'wrf.exe'
    if exe.exists():
        return exe
    exe.parent.mkdir(parents=True, exist_ok=True)
    tmp = exe.with_name('.wrf.exe.partial')
    shutil.copy2(src, tmp)
    md5 = hashlib.md5(tmp.read_bytes()).hexdigest()
    built = time.strftime('%Y-%m-%d %H:%M', time.localtime(src.stat().st_mtime))
    (exe.parent / 'wrf.exe.md5').write_text('%s  wrf.exe\n# copied from %s, built %s\n'
                                           % (md5, src, built))
    tmp.rename(exe)
    print('    wrf.exe: copied %s (built %s, md5 %s)' % (src, built, md5))
    return exe


def assemble(run, emis, exe, tables, name, chem_opt, emiss_opt, mail, batch_template):
    """Links, namelist and batch.sh for one run."""
    for table, target in tables.items():
        os.symlink(target, run / table)
    os.symlink(exe, run / 'wrf.exe')
    for f in sorted(emis.glob('wrfchemi_d01_*')):
        os.symlink(f, run / f.name)

    text = config.TEMPLATE_NAMELIST.read_text()
    for key, value in (('chem_opt', chem_opt), ('emiss_opt', emiss_opt)):
        # Domain 1 only: the first number on the line.
        text, n = re.subn(r'^( *%s *= *)\d+,' % key, r'\g<1>%d,' % value, text, flags=re.M)
        if n != 1:
            raise RuntimeError('expected one %s line in %s, found %d' % (key, config.TEMPLATE_NAMELIST, n))
    (run / 'namelist.input').write_text(text)

    mail_lines = ('#SBATCH --mail-type=END,FAIL\n#SBATCH --mail-user=%s\n' % mail) if mail else ''
    batch = batch_template.read_text() if batch_template else BATCH
    for key, value in (('{name}', name), ('{run}', str(run)), ('{mail}', mail_lines)):
        batch = batch.replace(key, value)
    (run / 'batch.sh').write_text(batch)


def check(run, n_latent, rate):
    """Every non-latent field against the committed reference checksums (reference.py)."""
    ref = reference.load()
    print('    reference: %s (%s)' % (config.INPUT_FINGERPRINTS.name,
                                      ref['reference_runs'][str(n_latent)]))
    ignore = set() if rate == ref['emission_rate'] else {'E_NTOT'}   # another rate SHOULD change it
    if ignore:
        print('    rate %s is not the reference %g: E_NTOT not compared' % (short_rate(rate),
                                                                          ref['emission_rate']))
    return reference.check(run, n_latent, ignore, ref=ref)


def build(fpidx, rate, work, wrf_dir, resdir=None, mail=None, batch_template=None):
    name = '%s_%s' % (fpidx, short_rate(rate))
    run, emis, lat = run_path(work, name), work / 'emis' / name, work / 'latents' / fpidx
    for d in (run, emis):
        if d.exists():
            sys.exit('%s already exists; nothing done (choose another --rate, or move it aside)' % d)
    tables = wrf_tables(wrf_dir)
    if not (Path(wrf_dir) / 'main' / 'wrf.exe').exists():
        sys.exit('no %s/main/wrf.exe; compile WRF-AeroMELD first' % wrf_dir)

    print('[1/5] latents')
    if lat.exists():
        missing = [s for s in config.POP_SCENARIOS
                   if not encode_populations.latent_path(lat, s).exists()]
        if missing:
            sys.exit('%s lacks scenario(s) %s; move it aside and rerun' % (lat, missing))
        print('    reusing %s' % lat)
    else:
        partial = lat.with_name(lat.name + '.partial')
        shutil.rmtree(partial, ignore_errors=True)
        encode_populations.write(encode_populations.encode(fpidx, resdir), partial)
        partial.rename(lat)
        print('    wrote %s' % lat)
    latents = {s: np.loadtxt(encode_populations.latent_path(lat, s)) for s in config.POP_SCENARIOS}
    known = reference.check_latents(fpidx, latents)
    if known is None:
        print('    no stored latents for %s to check the encoding against (a new model)' % fpidx)
    elif known[0]:
        print('    encoder check: matches the stored latents of %s (max |diff| %.2g)' % (fpidx, known[1]))
    else:
        sys.exit('ENCODER CHECK FAILED: %s differs from its stored latents by %.3g (tolerance %g). '
                 'Is the archive the same one, and the partnn/torch environment sound?'
                 % (lat, known[1], reference.LATENT_TOL))
    n_latent = n_latent_of(lat)
    if n_latent not in LATENT_OPTIONS:
        sys.exit('%s has %d latents + ntot; WRF carries 9 (chem_opt 900) or 49 (chem_opt 901)'
                 % (fpidx, n_latent))
    suffix, chem_opt, emiss_opt = LATENT_OPTIONS[n_latent]
    templates = [config.TEMPLATE_DIR / ('%s_d01_cares%s' % (k, suffix)) for k in ('wrfinput', 'wrfbdy')]
    for t in templates + [config.EMIS_GRID_TEMPLATE, config.TEMPLATE_NAMELIST]:
        if not t.exists():
            sys.exit('missing %s' % t)
    print('    %d latents + ntot -> chem_opt=%d, emiss_opt=%d' % (n_latent, chem_opt, emiss_opt))
    exe = frozen_wrf_exe(work, Path(wrf_dir) / 'main' / 'wrf.exe')

    created = []
    try:
        print('[2/5] wrfinput_d01 + wrfbdy_d01')
        run.mkdir(parents=True)
        created.append(run)
        run_wrapper('run_latent_ics.sh', dict(
            IC_TEMPLATE=templates[0], BC_TEMPLATE=templates[1],
            IC_OUT=run / 'wrfinput_d01', BC_OUT=run / 'wrfbdy_d01',
            N_LATENT=n_latent, LATENT_DIR=lat))
        print('[3/5] wrfchemi_d01_* (E_NTOT = %s in the emission box)' % short_rate(rate))
        created.append(emis)
        run_wrapper('run_latent_emis.sh', dict(
            EMIS_OUTDIR=emis, EMIS_REGION_VALUE=short_rate(rate),
            GRID_TEMPLATE=config.EMIS_GRID_TEMPLATE, N_LATENT=n_latent, LATENT_DIR=lat))
        print('[4/5] run directory')
        assemble(run, emis, exe, tables, name, chem_opt, emiss_opt, mail, batch_template)
        print('    %s (%d physics-table links from %s/test/em_real)' % (run, len(tables), wrf_dir))
    except BaseException:
        for d in created:
            shutil.rmtree(d, ignore_errors=True)
        print('failed; removed %s' % ', '.join(str(d) for d in created), file=sys.stderr)
        raise

    print('[5/5] check against the reference inputs')
    ok = check(run, n_latent, rate)
    print()
    if not ok:
        print('CHECK FAILED: fields other than the latents differ from the reference inputs.\n'
              'Do not submit until you know why (the scenario, a template or the namelist changed?).')
        return 1
    print('Built %s. Submit with:\n  %s' % (name, SUBMIT.format(run=run)))
    print('Then:  python inputs/build_run.py --status --work %s' % work)
    print('Decode its output with fpidx=%r.' % fpidx)
    return 0


def status(work):
    runs = sorted((work / 'runs').glob(config.WRF_ML_RUN_PREFIX + '_*'))
    if not runs:
        print('no runs in %s' % (work / 'runs'))
        return 0
    print('%-30s %-10s %s' % ('RUN', 'QUEUE', 'RESULT'))
    for run in runs:
        name = run.name[len(config.WRF_ML_RUN_PREFIX) + 1:]
        try:
            q = subprocess.run(['squeue', '-h', '-u', os.environ.get('USER', ''), '-n', 'wrf_' + name,
                                '-o', '%T'], capture_output=True, text=True).stdout.split()
        except FileNotFoundError:
            q = []
        rsl = run / 'rsl.error.0000'
        if not rsl.exists():
            result = 'not run'
        elif 'SUCCESS COMPLETE WRF' in rsl.read_text(errors='replace'):
            result = 'SUCCESS'
        else:
            result = 'running' if q else 'FAILED: no SUCCESS line in rsl.error.0000'
        for out in sorted(run.glob('wrfout_d01_*')):
            size = out.stat().st_size
            result += (', %s EMPTY (history file never written)' % out.name if size == 0
                       else ', wrfout %.2f GB' % (size / 1e9))
        rsl_out = sum(p.stat().st_size for p in run.glob('rsl.out.*'))
        if rsl_out > 0.5e9:
            result += ', rsl.out.* %.1f GB (rm once checked)' % (rsl_out / 1e9)
        print('%-30s %-10s %s' % (name, q[0] if q else '-', result))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('fpidx', nargs='?', help='model, <name>.<file>.<part> (e.g. 05_nonlinvlong.0.0)')
    ap.add_argument('--rate', type=float, default=config.EMISSION_RATE,
                    help="emission-box E_NTOT in # m^-2 s^-1 (default %g, the paper's)"
                         % config.EMISSION_RATE)
    ap.add_argument('--work', help='workspace directory (default $AEROMELD_WORK)')
    ap.add_argument('--resdir', help='model archive directory (default config.MODEL_RESDIR)')
    ap.add_argument('--wrf-dir', default=str(config.WRF_DIR),
                    help='compiled WRF-AeroMELD tree: wrf.exe from main/, physics tables as linked '
                         'in test/em_real/ (default config.WRF_DIR = %s)' % config.WRF_DIR)
    ap.add_argument('--batch-template', type=Path,
                    help='your own batch.sh with {name}, {run} (and {mail}) placeholders; the '
                         'default is the authors\' cluster\'s (sesempi, 96 ranks, system OpenMPI)')
    ap.add_argument('--mail', default='', help='address for SLURM end/fail mail (default none)')
    ap.add_argument('--status', action='store_true', help='report every run in the workspace')
    args = ap.parse_args()
    work = workspace(args.work)
    if args.status:
        return status(work)
    if not args.fpidx or not re.match(r'^[A-Za-z0-9_]+\.\d+\.\d+$', args.fpidx):
        ap.error('give a model as <name>.<file>.<part>, e.g. 05_nonlinvlong.0.0')
    if not args.rate > 0:
        ap.error('--rate must be positive')
    return build(args.fpidx, args.rate, work, Path(args.wrf_dir).absolute(), args.resdir,
                 args.mail, args.batch_template)


if __name__ == '__main__':
    sys.exit(main())

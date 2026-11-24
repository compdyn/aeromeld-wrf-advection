#!/usr/bin/env python
"""Confirm the cause of the optics difference: code_partnn's get_bulkoptical
(the training data) takes each particle's geometric cross-section from its
SHELL diameter (the volume without BC and OIN), pi/4 d_shell^2, while the Mie
size parameter (and the vendor code, the paper's PartMC side) uses the whole
particle's diameter d. Runs code_partnn's own function on the three source files,
as is and with pi/4 d^2 (a copy patched in memory; code_partnn is not
modified), against the training values (wrf3_jeff.npz) and the vendor code.

Usage (from code_partnn/notebooks, which partnn.io_cfg requires; an env with
torch and netCDF4; PYTHONDONTWRITEBYTECODE=1 so nothing is written there):
    python .../scripts/checks/pmc_optics_crosssection.py PAPER_REPO
"""
import inspect
import math
import sys
from pathlib import Path

import numpy as np
import torch
from netCDF4 import Dataset

REPO = Path(sys.argv[1])
sys.path.insert(0, str(Path.cwd().parent))
import partnn.aerometrics as AM  # noqa: E402

ML = ['SO4', 'NO3', 'Cl', 'NH4', 'ARO1', 'ARO2', 'ALK1', 'OLE1', 'API1', 'Na', 'OIN', 'OC', 'BC', 'MOC', 'H2O']
SCEN = (561, 635, 812)
J = np.load(REPO / 'data' / 'affx' / 'wrf3_jeff.npz')
import h5py  # noqa: E402
with h5py.File('/Users/mwest/git/aeromeld-lindiag/data/datasets/04_bwchisamp_pm10.h5', 'r') as f:
    rho = torch.tensor(f['chem_rho'][()])
    refr = torch.tensor(f['chem_refr_real'][()] + 1j * f['chem_refr_imag'][()])     # (wave, chem)
    wl = torch.tensor(f['wave_len'][()])

src = inspect.getsource(AM.get_bulkoptical)
assert src.count('crossec_prt = d_prtshll.square() * (math.pi / 4)') == 1
ns = dict(vars(AM))
exec(src.replace('crossec_prt = d_prtshll.square() * (math.pi / 4)',
                 'crossec_prt = d_prt.square() * (math.pi / 4)').replace('def get_bulkoptical', 'def fixed'), ns)
fixed = ns['fixed']

for k, s in enumerate(SCEN):
    m = torch.tensor(J['m'][k])[None, None]                    # (1, 1, 1000, 15)
    n = torch.tensor(J['n_prt'][k])[None, None, :, None]
    out = {}
    for name, fn in (('as is', AM.get_bulkoptical), ('pi/4 d^2', fixed)):
        r = fn(m, n, rho, refr, wl, 100, complex(1.5, 0), ML, 1, 1, 15, 1000, 8,
               torch.device('cpu'), torch.float64, torch.complex128, input_type='prt')
        out[name] = (r['qs_pop'].reshape(-1).numpy(), r['qa_pop'].reshape(-1).numpy())
    nt = float(J['ntot'][k])
    T = (J['X'][k][340:348] * nt, J['X'][k][348:356] * nt)
    v = J['m'][k] / rho.numpy()
    core = v[:, [ML.index('BC'), ML.index('OIN')]].sum(1) / v.sum(1)
    w = J['n_prt'][k] * (v.sum(1) ** (2 / 3))
    print('%d: number-weighted mean core volume fraction %.3f; area-weighted (1 - core)^(2/3) %.3f'
          % (s, np.average(core, weights=J['n_prt'][k]), np.average((1 - core) ** (2 / 3), weights=w)))
    for name, (qs, qa) in out.items():
        print('   %-9s scat / training %s   abs / training %s' % (name, np.round(qs / T[0], 4), np.round(qa / T[1], 4)))

# --- the clearest case: particles that are (nearly) all BC and dust --------------------
print('\nparticles whose volume is >= 95% BC + OIN (dust): scattering and absorption at 550-600 nm '
      '(per particle, m^2), training code as is vs pi/4 d^2')
for k, s in enumerate(SCEN):
    m = torch.tensor(J['m'][k])[None, None]
    n = torch.tensor(J['n_prt'][k])[None, None, :, None]
    v = J['m'][k] / rho.numpy()
    core = v[:, [ML.index('BC'), ML.index('OIN')]].sum(1) / v.sum(1)
    sel = np.flatnonzero(core >= .95)
    if not sel.size:
        print('  %d: none' % s)
        continue
    res = {}
    for name, fn in (('as is', AM.get_bulkoptical), ('pi/4 d^2', fixed)):
        r = fn(m, n, rho, refr, wl, 100, complex(1.5, 0), ML, 1, 1, 15, 1000, 8,
               torch.device('cpu'), torch.float64, torch.complex128, input_type='prt')
        res[name] = (r['qscs_prt'].reshape(8, -1)[3].numpy() / J['n_prt'][k],
                     r['qacs_prt'].reshape(8, -1)[3].numpy() / J['n_prt'][k])
    for i in sel[:6]:
        print('  %d particle %4d: D %.2f um, BC+dust %.3f of volume | scat %.3g vs %.3g | abs %.3g vs %.3g'
              % (s, i, (6 / np.pi * v[i].sum()) ** (1 / 3) * 1e6, core[i], res['as is'][0][i], res['pi/4 d^2'][0][i],
                 res['as is'][1][i], res['pi/4 d^2'][1][i]))
    print('  %d: %d such particles of 1000' % (s, sel.size))

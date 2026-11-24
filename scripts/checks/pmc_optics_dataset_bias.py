#!/usr/bin/env python
"""How much the training data's optics are affected by the shell-diameter
cross-section (pmc_optics_crosssection.py), over the dataset: for a random
sample of populations, the population scattering and absorption recomputed with
pi/4 d^2, divided by the stored (training) values.

Usage (from code_partnn/notebooks, PYTHONDONTWRITEBYTECODE=1):
    python .../pmc_optics_dataset_bias.py OUT.npz [N_POP] [SEED]
"""
import inspect
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path.cwd().parent))
# n_max 220: the dataset build's aeroopt/n_max (21_data/01_datasets.yml); 100 truncates for coarse particles
import partnn.aerometrics as AM  # noqa: E402

out = sys.argv[1]
n_pop = int(sys.argv[2]) if len(sys.argv) > 2 else 300
rng = np.random.default_rng(int(sys.argv[3]) if len(sys.argv) > 3 else 0)
torch.set_num_threads(1)
ML = ['SO4', 'NO3', 'Cl', 'NH4', 'ARO1', 'ARO2', 'ALK1', 'OLE1', 'API1', 'Na', 'OIN', 'OC', 'BC', 'MOC', 'H2O']
src = inspect.getsource(AM.get_bulkoptical)
ns = dict(vars(AM))
exec(src.replace('crossec_prt = d_prtshll.square() * (math.pi / 4)',
                 'crossec_prt = d_prt.square() * (math.pi / 4)').replace('def get_bulkoptical', 'def fixed'), ns)
fixed = ns['fixed']
with h5py.File('/Users/mwest/git/aeromeld-lindiag/data/datasets/04_bwchisamp_pm10.h5', 'r') as f:
    rho = torch.tensor(f['chem_rho'][()])
    refr = torch.tensor(f['chem_refr_real'][()] + 1j * f['chem_refr_imag'][()])
    wl = torch.tensor(f['wave_len'][()])
    idx = np.sort(rng.choice(f['n_prt'].shape[0], n_pop, replace=False))
    rs, ra, chk = [], [], []
    for i in idx:
        m = torch.tensor(f['m_prtchm'][i])[None, None]
        n = torch.tensor(f['n_prt'][i])[None, None]
        args = (m, n, rho, refr, wl, 220, complex(1.5, 0), ML, 1, 1, 15, 1000, 8, torch.device('cpu'),
                torch.float64, torch.complex128)
        a = AM.get_bulkoptical(*args, input_type='prt')
        b = fixed(*args, input_type='prt')
        qs, qa = f['qs_pop'][i].reshape(-1), f['qa_pop'][i].reshape(-1)
        chk.append(max(np.abs(a['qs_pop'].reshape(-1).numpy() / qs - 1).max(),
                       np.abs(a['qa_pop'].reshape(-1).numpy() / qa - 1).max()))
        rs.append(b['qs_pop'].reshape(-1).numpy() / qs)
        ra.append(b['qa_pop'].reshape(-1).numpy() / qa)
    i_snr = f['i_snr'][idx]
np.savez(out, idx=idx, i_snr=i_snr, ratio_scat=np.array(rs), ratio_abs=np.array(ra), check=np.array(chk))
rs, ra = np.array(rs), np.array(ra)
print('populations %d; as-is code vs stored values: max rel. diff %.1e' % (len(idx), max(chk)))
for name, r in (('scattering', rs), ('absorption', ra)):
    print('%s corrected / stored, by wavelength 300..1000 nm: median %s, p90 %s, p99 %s, max %s'
          % (name, np.round(np.median(r, 0), 3), np.round(np.percentile(r, 90, 0), 3),
             np.round(np.percentile(r, 99, 0), 3), np.round(r.max(0), 3)))

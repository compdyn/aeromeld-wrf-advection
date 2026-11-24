"""MAM4-like modal representation: Aitken, accumulation, coarse and primary-carbon lognormal
modes with E3SM MAM4's fixed geometric standard deviations, diameter bounds and size adjustment
(calcsize: number relaxed toward the diameter bounds, Aitken <-> accumulation transfer).

Sources (fetched 2026-11-14; notes_representations.md has the details):
  * mode parameters: mam4xx (E3SM's C++ port of MAM4), github.com/eagles-project/mam4xx,
    src/mam4xx/aero_modes.hpp at commit 2c35def02337dbd16aa70b3adfbe603e569c5225 (values from
    E3SM's physprops files mam4_mode{1..4}_*.nc); species per mode: same file,
    mode_aero_species();
  * calcsize: E3SM, github.com/E3SM-Project/E3SM,
    components/eam/src/chemistry/utils/modal_aero_calcsize.F90 at commit
    c25291831fb3e6374f4ce7ad5e43c0d618d5b71e (subroutines modal_aero_calcsize_sub,
    compute_dgn_vol_limits, adjust_num_sizes, aitken_accum_exchange,
    compute_coef_ait_acc_transfer, compute_coef_acc_ait_transfer);
  * primary-carbon ageing criterion used by the projection: E3SM
    components/eam/src/chemistry/modal_aero/modal_aero_amicphys.F90 (same commit),
    mam_pcarbon_aging_1subarea: dr_so4_monolayers_pcage = n_so4_monolayers_pcage * 4.76e-10 m,
    n_so4_monolayers_pcage = 8 in E3SM (mam4xx src/mam4xx/aging.hpp Config).

State. A per-particle state vector x (number fraction and mass per particle of each carried
species in each mode, kg), so that a cell's extensive state is Q = N x (number and kg per kg of
air) and mixing averages x in number. Mode order is MAM4's: 0 accumulation, 1 Aitken, 2 coarse,
3 primary carbon. Per mode: [number, species...]. Variants:
  'mam4'   every mode carries all 15 PartMC species (config.ML_SPECIES order, H2O last): 64.
  'mam4l'  each mode carries MAM4's species (E3SM MAM4 without MOM; see LUMP), lumped from the
           15 PartMC species, plus H2O: 25.
  'mam4a'  as 'mam4' (same layout and calcsize), but fresh carbon particles larger than the
           primary-carbon mode's dgnumhi (100 nm dry) are projected by dry diameter (accumulation)
           instead of to the primary-carbon mode, so every mode starts inside MAM's bounds.
  'mam4la' (added 2026-11-14) MAM4's own species as 'mam4l' (same 25-tracer layout and
           calcsize), with 'mam4a''s projection rule for fresh carbon above the pca dgnumhi.
H2O is carried passively in both (MAM diagnoses aerosol water instead; this case is passive).

Diagnostics: representative particles per mode on an equispaced grid in ln D over the
fixed-sigma lognormal (dgn from the mode's number and dry volume), all with the mode-mean mass
fractions, then baselines/diag.py.
"""
import json
import math

import numpy as np

import diag

SPECIES = diag.SPECIES
NS = diag.NS
SP = diag.SP
RHO = diag.RHO

# --- MAM4 mode parameters (mam4xx aero_modes.hpp, E3SM physprops) ---------------------------
MODES = ('accumulation', 'aitken', 'coarse', 'primary_carbon')
SHORT = ('acc', 'ait', 'coa', 'pca')
SIGMAG = np.array([1.8, 1.6, 1.8, 1.60000002384186])
DGNUM = np.array([1.1e-7, 2.6e-8, 2e-6, 5e-8])            # nominal geometric mean (number), m
DGNUMLO = np.array([5.35e-8, 8.7e-9, 1e-6, 1e-8])
DGNUMHI = np.array([4.4e-7, 5.2e-8, 4e-6, 1e-7])
I_ACC, I_AIT, I_COA, I_PCA = 0, 1, 2, 3
CMN = np.exp(4.5 * np.log(SIGMAG) ** 2) * math.pi / 6.0    # volume = CMN dgn^3 per particle

# --- calcsize constants (modal_aero_calcsize.F90) -------------------------------------------
SECONDS_IN_A_DAY = 86400.0
CLOSE_TO_ONE = 1.0 + 1.0e-15
RELAX_FACTOR = 27.0               # not used: the relaxed bounds act only with cloud-borne aerosol
SZADJ_BLOCK_FAC = 1.0e6

# --- projection ---------------------------------------------------------------------------
#: Aitken/accumulation boundary in dry diameter: the geometric mean of the two nominal dgnum.
D_AIT_ACC = math.sqrt(DGNUM[I_AIT] * DGNUM[I_ACC])          # 53.5 nm
#: Accumulation/coarse boundary: 1 um dry, MAM4's own split of dust and sea-salt emissions
#: between the accumulation and coarse modes (also the coarse mode's dgnumlo).
D_ACC_COA = 1.0e-6
#: Primary-carbon ageing (E3SM): shell of n_so4_monolayers_pcage = 8 monolayers of 4.76e-10 m.
DR_PCAGE = 8 * 4.76e-10
CORE_SPECIES = ('BC', 'OC', 'MOC')
KAPPA_SO4 = diag.KAPPA[SP['SO4']]

# --- lumping (variant mam4l) --------------------------------------------------------------
#: MAM4 species <- PartMC species. The first PartMC species of each group is the one the
#: lumped mass is reported as (whose density, kappa and refractive index the diagnostics use);
#: within each group the members' density, kappa and refractive index are identical, except MOC
#: (refractive index 1.50 vs OC 1.45).
LUMP = {'so4': ('SO4', 'NO3', 'NH4'),
        'pom': ('OC', 'MOC'),
        'soa': ('API1', 'ARO1', 'ARO2', 'ALK1', 'OLE1'),
        'bc': ('BC',),
        'dst': ('OIN',),
        'ncl': ('Cl', 'Na'),
        'h2o': ('H2O',)}
#: Species per mode: E3SM MAM4 (mam4xx mode_aero_species) without MOM (marine organic matter;
#: PartMC's MOC has OC's density and kappa and is lumped into pom).
MODE_SPECIES_L = {I_ACC: ('so4', 'pom', 'soa', 'bc', 'dst', 'ncl', 'h2o'),
                  I_AIT: ('so4', 'soa', 'ncl', 'h2o'),
                  I_COA: ('dst', 'ncl', 'so4', 'bc', 'pom', 'soa', 'h2o'),
                  I_PCA: ('pom', 'bc', 'h2o')}
LUMP_OF = {s: g for g, members in LUMP.items() for s in members}

#: Representative particles: nodes on a fixed grid in log10 WET diameter aligned with the
#: diagnostics' 0.25-decade bins (GRID_PER_BIN midpoints per bin), kept where
#: t = ln(D / dgn) / ln(sigma) lies in [T_LO, T_HI + 3 ln sigma] (number tails to 4.5 sigma, and
#: the volume distribution's upper tail); weights the lognormal density (midpoint rule). Aligned
#: with the bins, the quadrature reproduces the lognormal's binned number and mass to O(h^2);
#: an equispaced grid in t relative to dgn aliased against the bins (number error of the
#: background source 20 % at spacing 0.5 ln sigma, 12 % at 0.25, 10 % at 0.0625).
T_LO, T_HI = -4.5, 4.5
GRID_PER_BIN = 4
#: Diameter in which the node grid is aligned with the diagnostics' bins (2026-11-16): 'wet' when
#: the size distributions are binned by wet diameter, any other basis ('dry', 'drydiam') when by dry
#: diameter (diag.SIZE_BASIS, env AEROMELD_SIZE_BASIS; dist_checks.rep_dists sets it per basis). Misaligned, the midpoint
#: rule's nodes shift number and mass between neighbouring bins (up to 2 points of a source's error).
GRID_BASIS = diag.SIZE_BASIS
LOG10_D0 = math.log10(diag.DIAM_LOW)
DLOG10 = 0.25 / GRID_PER_BIN


class Modal:
    """One modal model ('mam4' or 'mam4l')."""

    def __init__(self, name='mam4', calcsize_dt=20.0):
        if name not in ('mam4', 'mam4l', 'mam4a', 'mam4la'):
            raise ValueError(name)
        self.name = name
        #: mam4l and mam4la lump species; mam4 and mam4a carry all 15 (same layout and calcsize)
        self.lumped = name in ('mam4l', 'mam4la')
        #: mam4a, mam4la: fresh carbon particles larger than the pca mode's dgnumhi go by dry
        #: diameter (accumulation) instead of to primary carbon
        self.pca_max_d = DGNUMHI[I_PCA] if name in ('mam4a', 'mam4la') else None
        self.calcsize_dt = calcsize_dt
        comps = []                                 # (mode, kind, species or None)
        for i in range(4):
            comps.append((i, 'number', None))
            sp = MODE_SPECIES_L[i] if self.lumped else SPECIES
            for s in sp:
                comps.append((i, 'mass', s))
        self.comps = comps
        self.d = len(comps)
        self.i_num = np.array([k for k, c in enumerate(comps) if c[1] == 'number'])
        # per mode: component indices of the species, their densities, dry flags, and the
        # PartMC species each is reported as in the representative particles
        self.mode_idx, self.mode_rho, self.mode_dry, self.mode_rep = [], [], [], []
        for i in range(4):
            idx = [k for k, c in enumerate(comps) if c[0] == i and c[1] == 'mass']
            sp = [comps[k][2] for k in idx]
            rep = [LUMP[s][0] if self.lumped else s for s in sp]
            self.mode_idx.append(np.array(idx))
            self.mode_rho.append(np.array([RHO[SP[r]] for r in rep]))
            self.mode_dry.append(np.array([r != 'H2O' for r in rep]))
            self.mode_rep.append(np.array([SP[r] for r in rep]))
        # Aitken <-> accumulation species pairs (same species name in both modes)
        ait = {comps[k][2]: k for k in self.mode_idx[I_AIT]}
        acc = {comps[k][2]: k for k in self.mode_idx[I_ACC]}
        self.pairs = np.array([(ait[s], acc[s]) for s in ait if s in acc])
        # accumulation species not in the Aitken mode (MAM's noxf_acc2ait), dry only
        self.noxf = np.array([acc[s] for s in acc if s not in ait], dtype=int)
        self.noxf_rho = np.array([RHO[SP[LUMP[s][0] if self.lumped else s]] for s in acc if s not in ait])

    # --- projection -------------------------------------------------------------------------
    def assign(self, m):
        """Mode index (0..3) of each particle m (..., 15)."""
        v = m / RHO
        core = v[..., [SP[s] for s in CORE_SPECIES]].sum(-1)
        noncore = np.array([diag.DRY[k] and SPECIES[k] not in CORE_SPECIES for k in range(NS)])
        shell_eq = (v[..., noncore] * (diag.KAPPA[noncore] / KAPPA_SO4)).sum(-1)
        dcore = (6.0 / math.pi * core) ** (1.0 / 3.0)
        fresh = (core > 0) & (shell_eq * dcore < 6.0 * DR_PCAGE * core)
        dd = diag.dry_diameter(m)
        mode = np.where(dd < D_AIT_ACC, I_AIT, np.where(dd < D_ACC_COA, I_ACC, I_COA))
        if self.pca_max_d is not None:
            fresh = fresh & (dd <= self.pca_max_d)
        return np.where(fresh, I_PCA, mode)

    def project(self, m, w, calcsize_dt=None):
        """Particles m (P, n, 15) kg, w (P, n) -> per-particle states x (P, d), after MAM's
        calcsize once with dt = calcsize_dt (default self.calcsize_dt; None/0 skips it)."""
        m = np.asarray(m, float)
        w = np.asarray(w, float)
        if m.ndim == 2:
            m, w = m[None], w[None]
        wn = w / w.sum(-1, keepdims=True)
        mode = self.assign(m)
        x = np.zeros((len(m), self.d))
        for i in range(4):
            sel = np.where(mode == i, wn, 0.0)                               # (P, n)
            x[:, self.i_num[i]] = sel.sum(-1)
            mass = np.einsum('pn,pnk->pk', sel, m)                           # (P, 15)
            if not self.lumped:
                x[:, self.mode_idx[i]] = mass
            else:
                for g, members in LUMP.items():
                    tot = mass[:, [SP[s] for s in members]].sum(-1)
                    k = self._comp(i, g)
                    if k is None:                    # not carried here: mass to accumulation
                        k = self._comp(I_ACC, g)
                    x[:, k] += tot
        dt = self.calcsize_dt if calcsize_dt is None else calcsize_dt
        if dt:
            x = calcsize(x.T, dt, self).T
        return x

    def _comp(self, mode, species):
        for k in self.mode_idx[mode]:
            if self.comps[k][2] == species:
                return k
        return None

    # --- diagnostics ------------------------------------------------------------------------
    def dgn(self, x):
        """Number geometric mean dry diameter (P, 4) from number and dry volume (unclipped;
        nan where a mode is empty)."""
        x = np.atleast_2d(x)
        out = np.full((len(x), 4), np.nan)
        for i in range(4):
            idx = self.mode_idx[i][self.mode_dry[i]]
            vol = (np.maximum(x[:, idx], 0) / self.mode_rho[i][self.mode_dry[i]]).sum(-1)
            n = x[:, self.i_num[i]]
            ok = (n > 0) & (vol > 0)
            out[ok, i] = (vol[ok] / (CMN[i] * n[ok])) ** (1.0 / 3.0)
        return out

    def particles(self, x):
        """Representative particles of states x (P, d): masses (P, R, 15) kg and weights (P, R)
        (number fractions). Per mode: nodes on the bin-aligned wet-diameter grid (see T_LO,
        GRID_PER_BIN), weights the lognormal number density (normalized); every node has the mode's
        mass fractions (all 15 species incl. H2O), and the node masses are scaled by one factor
        per mode so that the mode's number, dry volume and every species mass are reproduced
        exactly. dgn from number and dry volume; a mode with mass but no number gets MAM's
        diagnosed size for that case (dgn = dgnumhi) and number volume / (CMN dgnumhi^3)."""
        x = np.atleast_2d(np.asarray(x, float))
        P = len(x)
        ms, ws = [], []
        for i in range(4):
            lns = math.log(SIGMAG[i])
            idx, dry, rep = self.mode_idx[i], self.mode_dry[i], self.mode_rep[i]
            mass = np.zeros((P, NS))
            np.add.at(mass.T, rep, np.maximum(x[:, idx], 0).T)                # (P, 15) per particle of pop
            vol = (mass[:, diag.DRY] / RHO[diag.DRY]).sum(-1)                 # dry volume per particle of pop
            wvol = (mass / RHO).sum(-1)                                       # wet volume
            n = np.maximum(x[:, self.i_num[i]], 0)
            nofix = (n <= 0) & (vol > 0)
            n = np.where(nofix, vol / (CMN[i] * DGNUMHI[i] ** 3), n)
            ok = (n > 0) & (vol > 0)
            vbar = np.where(ok, vol / np.where(ok, n, 1), 0.0)                # dry volume per particle of the mode
            dgn = (vbar / CMN[i]) ** (1.0 / 3.0)
            # wet/dry diameter ratio of every node (shared mass fractions)
            rw = np.where(ok, wvol / np.where(ok, vol, 1), 1.0) ** (1.0 / 3.0)
            # nodes: a fixed grid in log10 WET diameter, the midpoints of GRID_PER_BIN equal
            # sub-intervals of each of diag's 0.25-decade bins, over t = (ln D - ln dgn) / ln sigma
            # in [T_LO, T_HI + 3 ln sigma]; weights the lognormal number density x the interval
            # (midpoint rule), normalized
            ra = rw if GRID_BASIS == 'wet' else np.ones_like(rw)              # grid alignment
            lo = np.log10(np.where(ok, dgn * ra, 1.0)) + T_LO * lns / math.log(10)
            k0 = np.floor((lo - LOG10_D0) / DLOG10).astype(int)
            nk = int(math.ceil((T_HI + 3 * lns - T_LO) * lns / math.log(10) / DLOG10)) + 2
            lw = LOG10_D0 + DLOG10 * (k0[:, None] + np.arange(nk)[None, :] + 0.5)   # (P, nk)
            D = 10 ** lw / ra[:, None]                                         # node dry diameters
            t = np.log(D / np.where(ok, dgn, 1.0)[:, None]) / lns
            p = np.where((t >= T_LO) & (t <= T_HI + 3 * lns), np.exp(-0.5 * t ** 2), 0.0)
            p = p / np.maximum(p.sum(-1, keepdims=True), 1e-300)
            v = math.pi / 6 * D ** 3
            c = np.where(ok, vbar / np.maximum((p * v).sum(-1), 1e-300), 0.0)
            # node species masses: (v_j / vbar) c * (mass per particle of the mode)
            per = np.where(ok[:, None], mass / np.where(ok, n, 1)[:, None], 0.0)   # (P, 15)
            scale = np.where(ok[:, None], c[:, None] * v / np.where(ok, vbar, 1)[:, None], 0.0)
            ms.append(scale[:, :, None] * per[:, None, :])
            ws.append(np.where(ok[:, None], n[:, None] * p, 0.0))
            if nofix.any():
                self.last_nofix = int(nofix.sum())
        return np.concatenate(ms, axis=1), np.concatenate(ws, axis=1)

    def layout(self):
        comps = []
        for k, (i, kind, s) in enumerate(self.comps):
            c = {'index': k, 'mode': MODES[i], 'mode_index': i, 'kind': kind}
            if kind == 'number':
                c.update(units_per_particle='1 (fraction of particles in the mode)',
                         units_extensive='# kg-1 (Q = N x)')
            else:
                c.update(species=s, units_per_particle='kg (mass per particle of the population)',
                         units_extensive='kg kg-1 (Q = N x)')
                if self.lumped:
                    c.update(partmc_species=list(LUMP[s]), diagnosed_as=LUMP[s][0])
            comps.append(c)
        return {
            'model': self.name, 'kind': 'modal', 'n_tracers': self.d,
            'components': comps,
            'modes': [{'name': MODES[i], 'sigmag': float(SIGMAG[i]), 'dgnum_m': float(DGNUM[i]),
                       'dgnumlo_m': float(DGNUMLO[i]), 'dgnumhi_m': float(DGNUMHI[i])} for i in range(4)],
            'calcsize': 'baselines/modal.py: calcsize(Q, dt, model, xp=numpy|cupy, do_adjust=True, '
                        'do_aitacc_transfer=True) -> Q; Q (n_tracers, ...) extensive per kg in this '
                        'component order; MAM calls it every step (here: after each 20 s transport step)',
            'calcsize_applied_to_sources_dt_s': self.calcsize_dt,
            'densities_kg_m3': {('%s' % s): float(RHO[SP[LUMP[s][0] if self.lumped else s]])
                                for s in (LUMP if self.lumped else SPECIES)},
        }


# --- calcsize ---------------------------------------------------------------------------------

_MODELS = {}


def _model(model):
    if isinstance(model, Modal):
        return model
    if model not in _MODELS:
        _MODELS[model] = Modal(model)
    return _MODELS[model]


def _xp_of(a):
    try:
        import cupy
        return cupy.get_array_module(a)
    except ImportError:
        return np


def calcsize(Q, dt, model='mam4', xp=None, do_adjust=True, do_aitacc_transfer=True):
    """MAM4's size adjustment (E3SM modal_aero_calcsize_sub) on extensive tracers.

    Q   (n_tracers, ...) per kg of air (or per particle: calcsize is homogeneous of degree one in
        Q), in the component order of Modal(model) / data/state_layout.json; any trailing shape.
    dt  time step (s). tadj = max(1 day, dt); fracadj = dt / tadj.
    xp  numpy or cupy (default: inferred from Q).

    Interstitial aerosol only (no cloud-borne: drv_c = num_c = 0, so adjust_num_sizes takes its
    'activated volume is zero' branch and the relaxed bounds never act). Per mode: dry volume
    drv = sum max(q, 0) / rho over the mode's dry species (rho: the diagnostics' PartMC
    densities, so that sizes agree with the representative particles); number
    num = max(num, 0); if drv <= 0, num = 0; else num += (clip(num, drv v2nmin, drv v2nmax) -
    num) fracadj, with v2nmin/max = 1/(CMN dgnumhi/lo^3); with the Aitken <-> accumulation
    transfer on, the Aitken v2nmin is divided and the accumulation v2nmax multiplied by 1e6
    (compute_dgn_vol_limits). Then the transfer (aitken_accum_exchange) from the adjusted
    numbers: Aitken -> accumulation when num_ait < drv_ait v2n_geomean, accumulation -> Aitken
    when the transferable part of the accumulation mode has num > drv v2n_geomean, with
    v2n_geomean = sqrt(v2n(dgnum_ait) v2n(dgnum_acc)); fractions as in
    compute_coef_*_transfer, applied over dt with rate 1/tadj. Species move between the modes
    where both carry them (H2O, passive here, moves with them); accumulation species absent from
    the Aitken mode (mam4l: pom, bc, dst) stay ('noxf').
    Returns the new Q (a new array)."""
    M = _model(model)
    xp = xp or _xp_of(Q)
    Q = Q.copy()
    tadj = max(SECONDS_IN_A_DAY, dt)
    tadjinv = 1.0 / (tadj * CLOSE_TO_ONE)
    fracadj = max(0.0, min(1.0, dt * tadjinv))
    drv, num = [None] * 4, [None] * 4
    for i in range(4):
        idx = M.mode_idx[i][M.mode_dry[i]]
        rho = M.mode_rho[i][M.mode_dry[i]]
        dv = xp.zeros_like(Q[0])
        for k, r in zip(idx, rho):
            dv = dv + xp.maximum(Q[k], 0.0) / r
        n = xp.maximum(Q[M.i_num[i]], 0.0)
        if do_adjust:
            v2nmin = 1.0 / (CMN[i] * DGNUMHI[i] ** 3)
            v2nmax = 1.0 / (CMN[i] * DGNUMLO[i] ** 3)
            if do_aitacc_transfer:
                if i == I_AIT:
                    v2nmin = v2nmin / SZADJ_BLOCK_FAC
                if i == I_ACC:
                    v2nmax = v2nmax * SZADJ_BLOCK_FAC
            numbnd = xp.maximum(dv * v2nmin, xp.minimum(dv * v2nmax, n))
            n = xp.where(dv > 0, n + (numbnd - n) * fracadj, 0.0)
        Q[M.i_num[i]] = n
        drv[i], num[i] = dv, n
    if not do_aitacc_transfer:
        return Q
    v2n_ait = 1.0 / (CMN[I_AIT] * DGNUM[I_AIT] ** 3)
    v2n_acc = 1.0 / (CMN[I_ACC] * DGNUM[I_ACC] ** 3)
    v2n_geo = math.sqrt(v2n_ait * v2n_acc)
    tiny = 1e-300
    # Aitken -> accumulation (compute_coef_ait_acc_transfer)
    dt_, nt = drv[I_AIT], num[I_AIT]
    do12 = (dt_ > 0) & (nt < dt_ * v2n_geo)
    vol12 = ((nt / xp.maximum(dt_, tiny)) - v2n_geo) / (v2n_acc - v2n_geo)
    num12 = vol12 * (dt_ * v2n_acc / xp.maximum(nt, tiny))
    allx = nt < dt_ * v2n_acc
    zero = (num12 <= 0) | (vol12 <= 0)
    one = (num12 >= 1) | (vol12 >= 1)
    vol12 = xp.where(allx, 1.0, xp.where(zero, 0.0, xp.where(one, 1.0, vol12)))
    num12 = xp.where(allx, 1.0, xp.where(zero, 0.0, xp.where(one, 1.0, num12)))
    xc_num12 = xp.where(do12, num12 * tadjinv, 0.0)
    xc_vol12 = xp.where(do12, vol12 * tadjinv, 0.0)
    xt_num12 = num[I_AIT] * xc_num12
    # accumulation -> Aitken (compute_coef_acc_ait_transfer)
    dt0, nt0 = drv[I_ACC], num[I_ACC]
    first = (dt0 > 0) & (nt0 > dt0 * v2n_geo)
    dv_noxf = xp.zeros_like(dt0)
    for k, r in zip(M.noxf, M.noxf_rho):
        if M.comps[k][2] in ('H2O', 'h2o'):
            continue
        dv_noxf = dv_noxf + xp.maximum(Q[k], 0.0) / r
    v2nlo_acc = 1.0 / (CMN[I_ACC] * DGNUMLO[I_ACC] ** 3)
    nt = xp.where(first, xp.maximum(0.0, nt0 - dv_noxf * v2nlo_acc), nt0)
    dtt = xp.where(first, xp.maximum(0.0, dt0 - dv_noxf), dt0)
    dv_noxf = xp.where(first, dv_noxf, 0.0)
    do21 = (dtt > 0) & (nt > dtt * v2n_geo)
    vol21 = ((nt / xp.maximum(dtt, tiny)) - v2n_geo) / (v2n_ait - v2n_geo)
    num21 = vol21 * (dtt * v2n_ait / xp.maximum(nt, tiny))
    allx = nt > dtt * v2n_ait
    zero = (num21 <= 0) | (vol21 <= 0)
    one = (num21 >= 1) | (vol21 >= 1)
    vol21 = xp.where(allx, 1.0, xp.where(zero, 0.0, xp.where(one, 1.0, vol21)))
    num21 = xp.where(allx, 1.0, xp.where(zero, 0.0, xp.where(one, 1.0, num21)))
    num21 = num21 * nt / xp.maximum(1.0e-37, nt0)
    xc_num21 = xp.where(do21, num21 * tadjinv, 0.0)
    xc_vol21 = xp.where(do21, vol21 * tadjinv, 0.0)
    xt_num21 = num[I_ACC] * xc_num21
    # tendencies over dt (MAM applies dqdt * deltat); masses from the pre-transfer state
    dn = (xt_num12 - xt_num21) * dt
    Q[M.i_num[I_AIT]] = num[I_AIT] - dn
    Q[M.i_num[I_ACC]] = num[I_ACC] + dn
    Q0 = Q.copy()
    for ka, kc in M.pairs:
        t12 = xp.maximum(Q0[ka], 0.0) * xc_vol12
        t21 = xp.maximum(Q0[kc], 0.0) * xc_vol21
        Q[ka] = Q0[ka] + (t21 - t12) * dt
        Q[kc] = Q0[kc] + (t12 - t21) * dt
    return Q


def describe():
    return {m: Modal(m).layout() for m in ('mam4', 'mam4l', 'mam4a', 'mam4la')}


if __name__ == '__main__':
    print(json.dumps(describe()['mam4l'], indent=1)[:2000])

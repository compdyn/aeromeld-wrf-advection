"""MATRIX-like representation (Bauer et al. 2008, ACP 8, 6003; the GISS ModelE aerosol microphysics):
mechanism k's populations, each a lognormal with MATRIX's fixed geometric standard deviation, carrying
MATRIX's species for it, plus total nitrate, ammonium and water shared among the populations as MATRIX
shares them; and MATRIX's state-driven adjustments (`adjust`), which act after transport.

Sources (fetched 2026-11-14; notes_matrix.md has the details and the md5s):
  * Bauer, S. E., et al.: MATRIX ..., Atmos. Chem. Phys., 8, 6003-6035, 2008 (Sect. 2.1, Tables 1a, 2);
  * GISS ModelE 2.1.2 (simplex.giss.nasa.gov/snapshots/modelE2.1.2.tgz, md5 14a5a0fb...; model/TRAMP_*):
      populations per mechanism   TRAMP_param_GISS.f l. 360-387 (MNAME, MSPCS, MODES1..8);
                                  TRAMP_config.F90 l. 13-36, 237-311 (mechanism 1's 54 box variables)
      sigma_g, default diameters  TRAMP_param_GISS.f l. 139-190 (SG_*, DG_*); emission sizes l. 198-249
      coating thresholds          TRAMP_param_GISS.f l. 76-78 (MIVF), l. 342-347 (MIMR), DENSP l. 26,
                                  core densities l. 318-324
      the 53 transported tracers  AmpTracersMetadata.F90 l. 438-532, l. 64-95 (AMP_AERO_MAP);
                                  TRACER_COM.f l. 304-305 (ntmAMP = 53)
      sea-salt number from mass   TRAMP_matrix.f l. 313-324; TRAMP_setup.f l. 812-846
      bulk NO3/NH4/H2O shares     TRAMP_matrix.f l. 452-469; paper Sect. 2 and 2.5
      class transfers             TRAMP_matrix.f l. 1086-1171
      AKK -> ACC transfer         TRAMP_matrix.f l. 140-157, 1180-1252; TRAMP_param_GISS.f l. 53, 62;
                                  DNU: TRAMP_npf.f l. 33-35
      size splits                 sea salt: TRACERS_AEROSOLS_SEASALT.F90 l. 72, 83 (dry radius 1 um);
                                  dust: PBL.f l. 945-960 with TRAMP_param_GISS.f l. 144-150 (4 um);
                                  AKK/ACC: paper Sect. 2.1 (0.1 um ambient diameter)

State. A per-particle vector x (Q = N x per kg of air; mixing averages x in number), in ModelE's AERO order
(TRAMP_config.F90 l. 237-311) without MASS_SSC_SULF, which ModelE does not transport:
  [NO3, NH4, H2O (totals over all populations, kg per particle of the cell)],
  then per population its MATRIX species (kg per particle of the cell) and its number fraction.
Mechanism 1 has 53 components: ModelE 2.1.2's 53 tracers (the paper's 51 plus the SSA and SSC numbers).

Species (MATRIX <- PartMC, `LUMP`; diagnosed as the first PartMC species, except OCAR, as API1):
  SULF <- SO4;  BCAR <- BC;  OCAR <- OC, MOC, ARO1, ARO2, ALK1, OLE1, API1, diagnosed as API1 (MATRIX
  has no SOA species: ModelE emits SOA into OCC's OC, TRACERS_DRV.f l. 6287-6292; MATRIX's OC is
  hygroscopic, soluble fraction 0.78, kappa ~0.14 (TRAMP_actv.f l. 88-110), closest to PartMC's SOA
  (kappa 0.1) and not to its OC (0.001); mam4l reports its lumped soa as API1 too. Coordinator's choice,
  2026-11-14; ocar_as='OC' is the sensitivity);  DUST <- OIN;
  SEAS <- Na, and Cl up to NaCl's ratio (Cl/Na = 35.453/22.990 by mass);
  NO3 (total) <- NO3, and chloride in excess of NaCl (MATRIX has no other chloride);  NH4 <- NH4;
  H2O (total) <- H2O, carried passively (MATRIX's thermodynamics would recompute it; this case is passive).

Projection (per particle; `assign`): by the MATRIX species a particle contains (mass > 0), which is the
receptor of MATRIX's coagulation table for those species (paper Table 1b; TRAMP_config.F90 l. 72-89):
  only SO4/NO3/NH4/H2O/excess Cl      AKK (wet diameter < 0.1 um) or ACC
  DUST                                DD (coating <= 5 %) or DS; 1 (dry diameter < 4 um) or 2
  SEAS                                SSA (dry diameter < 2 um) or SSC
  OCAR                                OCC
  BCAR                                BC1 (coating <= 5 %), BC2 (5-20 %) or BC3
  BCAR + DUST                         DBC
  BCAR + OCAR                         BOC
  anything else                       MXX
"Coating" is MATRIX's: (SO4 + NO3 + NH4 + H2O + excess Cl) mass over core (BC or dust) mass against
MIMR = (1.4 / rho_core) MIVF / (1 - MIVF), here per particle. BCS (and OCS) are populated only through
coagulation of sulfate particles with BC (OC) particles, a process; by composition their particles are
BC3 (OCC). Then MATRIX's own adjustment once (`adjust`, as MAM's calcsize once in modal.py).

Mechanisms 2-8 ('matrix2'..'matrix8'): the same rules on their populations (no BC3: BC2 takes > 5 %; no
DBC/BOC: MXX; no AKK: ACC; one sea-salt population SSS; one dust size: 1).

Diagnostics: representative particles per population on modal.py's bin-aligned grid in log10 wet
diameter over the fixed-sigma lognormal (dgn from number and dry volume; same nodes and weights as
modal.Modal.particles), every node with the population's mass fractions: its own species and its share of
NO3 and NH4 (by sulfate) and of H2O (by sulfate + sea salt), then baselines/diag.py.
"""
import json
import math

import numpy as np

import diag
import modal

SPECIES = diag.SPECIES
NS = diag.NS
SP = diag.SP
RHO = diag.RHO

# --- MATRIX species ---------------------------------------------------------------------------------
#: Principal species per population (TRAMP_param_GISS.f CHEM_SPC_NAME l. 333-334), in AERO order.
MSPC = ('SULF', 'BCAR', 'OCAR', 'DUST', 'SEAS')
#: Totals over all populations (AERO(1:3): MASS_NO3, MASS_NH4, MASS_H2O).
BULK = ('NO3', 'NH4', 'H2O')
#: MATRIX species <- PartMC species; the first is the one the lumped mass is diagnosed as (except OCAR, as
#: API1: DIAG_AS). Cl is split per
#: particle: up to R_CL_NA x Na into SEAS, the rest into NO3 (see split_chloride).
LUMP = {'SULF': ('SO4',), 'BCAR': ('BC',),
        'OCAR': ('OC', 'MOC', 'ARO1', 'ARO2', 'ALK1', 'OLE1', 'API1'),
        'DUST': ('OIN',), 'SEAS': ('Na', 'Cl'),
        'NO3': ('NO3', 'Cl'), 'NH4': ('NH4',), 'H2O': ('H2O',)}
DIAG_AS = {k: v[0] for k, v in LUMP.items()}
#: OCAR is diagnosed as API1 (PartMC's SOA; see the module docstring), not its first member.
DIAG_AS['OCAR'] = 'API1'
#: NaCl's Cl/Na mass ratio.
R_CL_NA = 35.453 / 22.990

# --- populations (TRAMP_param_GISS.f) ------------------------------------------------------------------
#: MNAME (l. 362-366) with MSPCS (l. 369-374): the species each population carries.
POP_SPECIES = {
    'AKK': ('SULF',), 'ACC': ('SULF',),
    'DD1': ('SULF', 'DUST'), 'DS1': ('SULF', 'DUST'), 'DD2': ('SULF', 'DUST'), 'DS2': ('SULF', 'DUST'),
    'SSA': ('SULF', 'SEAS'), 'SSC': ('SULF', 'SEAS'), 'SSS': ('SULF', 'SEAS'),
    'OCC': ('SULF', 'OCAR'), 'BC1': ('SULF', 'BCAR'), 'BC2': ('SULF', 'BCAR'), 'BC3': ('SULF', 'BCAR'),
    'OCS': ('SULF', 'OCAR'), 'DBC': ('SULF', 'BCAR', 'DUST'), 'BOC': ('SULF', 'BCAR', 'OCAR'),
    'BCS': ('SULF', 'BCAR'), 'MXX': ('SULF', 'BCAR', 'OCAR', 'DUST', 'SEAS')}
#: Geometric standard deviations SG_* (l. 173-190); = the emission SG_*_EMIS (l. 232-249), which
#: CONV_DPAM_TO_DGN uses (TRAMP_setup.f l. 1215).
SIGMA = {'AKK': 1.6, 'ACC': 1.8, 'DD1': 1.8, 'DS1': 1.8, 'DD2': 1.8, 'DS2': 1.8, 'SSA': 1.8, 'SSC': 2.0,
         'SSS': 2.0, 'OCC': 1.8, 'BC1': 1.8, 'BC2': 1.8, 'BC3': 1.8, 'OCS': 1.8, 'DBC': 1.8, 'BOC': 1.8,
         'BCS': 1.8, 'MXX': 2.0}
#: Default number geometric mean diameters DG_* (um, l. 139-172): used only for a population with mass and
#: no number (MATRIX: DP = DP0 when N < 1 m-3, TRAMP_matrix.f l. 495-498).
DG_UM = {'AKK': 0.026, 'ACC': 0.110, 'DD1': 1.16, 'DS1': 1.16, 'DD2': 2.0, 'DS2': 2.0, 'SSA': 0.88,
         'SSC': 2.0, 'SSS': 1.38, 'OCC': 0.050, 'BC1': 0.050, 'BC2': 0.100, 'BC3': 0.100, 'OCS': 0.070,
         'DBC': 0.330, 'BOC': 0.100, 'BCS': 0.070, 'MXX': 0.300}
#: Emission lognormals of the sea-salt populations (DG_*_EMIS um, SG_*_EMIS; l. 217-219, 238-240):
#: MATRIX's sea-salt number is NaCl mass / the dry NaCl mass per particle of these (TRAMP_setup.f
#: l. 812-846, l. 830-832 the diameter of average mass DPS, l. 846 with EMIS_DENS_SEAS).
SS_EMIS = {'SSA': (0.88, 1.8), 'SSC': (2.0, 2.0), 'SSS': (1.38, 2.0)}
EMIS_DENS_SEAS = 2165.0                                     # kg m-3 (l. 323)
#: Mechanisms' populations (MODES1..8, l. 382-389; order of TRAMP_config.F90's tables).
MECHANISMS = {
    1: ('AKK', 'ACC', 'DD1', 'DS1', 'DD2', 'DS2', 'SSA', 'SSC', 'OCC', 'BC1', 'BC2', 'BC3', 'DBC', 'BOC', 'BCS', 'MXX'),
    2: ('AKK', 'ACC', 'DD1', 'DS1', 'DD2', 'DS2', 'SSA', 'SSC', 'OCC', 'BC1', 'BC2', 'OCS', 'DBC', 'BOC', 'BCS', 'MXX'),
    3: ('AKK', 'ACC', 'DD1', 'DS1', 'DD2', 'DS2', 'SSA', 'SSC', 'OCC', 'BC1', 'BC2', 'BOC', 'MXX'),
    4: ('ACC', 'DD1', 'DS1', 'DD2', 'DS2', 'SSS', 'OCC', 'BC1', 'BC2', 'MXX'),
    5: ('AKK', 'ACC', 'DD1', 'DS1', 'SSA', 'SSC', 'OCC', 'BC1', 'BC2', 'BC3', 'DBC', 'BOC', 'BCS', 'MXX'),
    6: ('AKK', 'ACC', 'DD1', 'DS1', 'SSA', 'SSC', 'OCC', 'BC1', 'BC2', 'OCS', 'DBC', 'BOC', 'BCS', 'MXX'),
    7: ('AKK', 'ACC', 'DD1', 'DS1', 'SSA', 'SSC', 'OCC', 'BC1', 'BC2', 'BOC', 'MXX'),
    8: ('ACC', 'DD1', 'DS1', 'SSS', 'OCC', 'BC1', 'BC2', 'MXX')}
#: Paper Table 2 (transported species) and ModelE 2.1.2's ntmAMP (TRACER_COM.f l. 303-327).
PAPER_SPECIES = {1: 51, 2: 51, 3: 41, 4: 34, 5: 45, 6: 45, 7: 35, 8: 28}
MODELE_NTMAMP = {1: 53, 2: 51, 3: 41, 4: 34, 5: 45, 6: 45, 7: 35, 8: 28}

# --- thresholds and constants ------------------------------------------------------------------------
DENSP = 1400.0                                  # default ambient particle density, kg m-3 (l. 26)
EMIS_DENS_DUST, EMIS_DENS_BCAR = 2600.0, 1700.0  # l. 320-322
MIVF_DDD, MIVF_BC1, MIVF_BC2 = 0.05, 0.05, 0.20  # l. 76-78
MIMR_DDD = (DENSP / EMIS_DENS_DUST) * MIVF_DDD / (1.0 - MIVF_DDD)    # l. 342-347
MIMR_BC1 = (DENSP / EMIS_DENS_BCAR) * MIVF_BC1 / (1.0 - MIVF_BC1)
MIMR_BC2 = (DENSP / EMIS_DENS_BCAR) * MIVF_BC2 / (1.0 - MIVF_BC2)
#: AKK -> ACC transfer (TRAMP_matrix.f l. 140-157): exponent, cap, AKK minimum number (m-3), DNU (m).
IMTR_EXP, FNUM_MAX, AKK_MINNUM_IMTR, DNU = 4, 0.5, 1.0e6, 3.0e-9
#: Size splits (m): AKK/ACC ambient (wet) diameter; sea salt SSA/SSC and dust 1/2 dry diameter.
D_AKK_ACC_WET = 0.1e-6
D_SSA_SSC_DRY = 2.0e-6
D_DUST12_DRY = 4.0e-6

_BULK_IDX = {s: k for k, s in enumerate(BULK)}


def split_chloride(m):
    """PartMC masses (..., 15) -> (sea-salt Cl, excess Cl): Cl up to NaCl's ratio to Na is sea salt."""
    cl, na = m[..., SP['Cl']], m[..., SP['Na']]
    ss = np.minimum(cl, R_CL_NA * na)
    return ss, cl - ss


def lumped_masses(m):
    """PartMC masses (..., 15) -> dict MATRIX species (MSPC + BULK) -> (...) masses."""
    cl_ss, cl_x = split_chloride(m)
    g = lambda names: sum(m[..., SP[s]] for s in names)
    return {'SULF': m[..., SP['SO4']], 'BCAR': m[..., SP['BC']],
            'OCAR': g(LUMP['OCAR']), 'DUST': m[..., SP['OIN']],
            'SEAS': m[..., SP['Na']] + cl_ss, 'NO3': m[..., SP['NO3']] + cl_x,
            'NH4': m[..., SP['NH4']], 'H2O': m[..., SP['H2O']]}


class Matrix:
    """MATRIX mechanism k ('matrix1' .. 'matrix8'); interface as modal.Modal."""

    def __init__(self, name='matrix1', calcsize_dt=1800.0, ocar_as=None):
        """ocar_as: the PartMC species OCAR is diagnosed as (default DIAG_AS['OCAR'] = 'API1'; 'OC' is the
        sensitivity in notes_matrix.md)."""
        if not (name.startswith('matrix') and name[6:].isdigit() and int(name[6:]) in MECHANISMS):
            raise ValueError(name)
        self.name = name
        self.diag_as = dict(DIAG_AS)
        if ocar_as:
            self.diag_as['OCAR'] = ocar_as
        self.mech = int(name[6:])
        self.pops = MECHANISMS[self.mech]
        self.np = len(self.pops)
        #: dt passed to adjust at projection; MATRIX's adjustments are per call (no dt), so it only marks
        #: "adjust once" (0/None skips, as modal's calcsize_dt=0)
        self.calcsize_dt = calcsize_dt
        self.ss_pops = [p for p in self.pops if p in SS_EMIS]
        two_ss = len(self.ss_pops) == 2
        comps = [(-1, 'mass', s) for s in BULK]                    # (population index, kind, species)
        for i, p in enumerate(self.pops):
            for s in POP_SPECIES[p]:
                if two_ss and p == 'SSC' and s == 'SULF':
                    continue                                     # all sea-salt sulfate is in SSA's
                comps.append((i, 'mass', s))
            comps.append((i, 'number', None))
        self.comps = comps
        self.d = len(comps)
        self.idx = {(c[0], c[1], c[2]): k for k, c in enumerate(comps)}
        self.i_num = np.array([self.idx[(i, 'number', None)] for i in range(self.np)])
        self.i_bulk = np.array([self.idx[(-1, 'mass', s)] for s in BULK])
        self.pi = {p: i for i, p in enumerate(self.pops)}
        self.sigma = np.array([SIGMA[p] for p in self.pops])
        self.cmn = np.exp(4.5 * np.log(self.sigma) ** 2) * math.pi / 6.0
        self.dg_default = np.array([DG_UM[p] for p in self.pops]) * 1e-6
        # sea salt: NaCl mass per emitted particle (TRAMP_setup.f l. 830-846)
        self.ss_mpp = {}
        for p in self.ss_pops:
            dg, sg = SS_EMIS[p]
            dps = 1e-6 * dg * math.exp(1.5 * math.log(sg) ** 2)
            self.ss_mpp[p] = EMIS_DENS_SEAS * math.pi / 6.0 * dps ** 3
        self.two_ss = two_ss
        # class transfers in MATRIX's order (TRAMP_matrix.f l. 1121-1171): (from, to, core, MIMR)
        tr = [('DD1', 'DS1', 'DUST', MIMR_DDD), ('DD2', 'DS2', 'DUST', MIMR_DDD),
              ('BC1', 'BC2', 'BCAR', MIMR_BC1), ('BC2', 'BC3', 'BCAR', MIMR_BC2)]
        self.transfers = [t for t in tr if t[0] in self.pi and t[1] in self.pi]
        self.imtr = 'AKK' in self.pi                             # INTERMODAL_TRANSFER (TRAMP_setup.f l. 239-247)
        self.sulf = np.array([self.idx[(i, 'mass', 'SULF')] for i, p in enumerate(self.pops)
                              if (i, 'mass', 'SULF') in self.idx])

    def comp(self, pop, species):
        """Component index of `species` ('number' for the number) of population `pop` (name), or None."""
        i = self.pi[pop]
        return self.idx.get((i, 'number', None) if species == 'number' else (i, 'mass', species))

    # --- projection --------------------------------------------------------------------------------
    def assign(self, m):
        """Population index of each particle m (..., 15) kg (-1 for a massless particle)."""
        L = lumped_masses(m)
        has = {k: L[k] > 0 for k in ('BCAR', 'OCAR', 'DUST')}
        has['SEAS'] = m[..., SP['Na']] > 0
        coat = L['SULF'] + L['NO3'] + L['NH4'] + L['H2O']        # MATRIX's inorganic coating
        dd = diag.dry_diameter(m)
        dw = diag.wet_diameter(m)
        P = self.pi
        n_other = has['BCAR'].astype(int) + has['OCAR'] + has['DUST'] + has['SEAS']
        out = np.full(m.shape[:-1], P['MXX'])
        # inorganic only
        inorg = n_other == 0
        if 'AKK' in P:
            out = np.where(inorg, np.where(dw < D_AKK_ACC_WET, P['AKK'], P['ACC']), out)
        else:
            out = np.where(inorg, P['ACC'], out)
        # dust (+ inorganics)
        only = lambda k: has[k] & (n_other == 1)
        coated = coat > MIMR_DDD * L['DUST']
        size2 = (dd >= D_DUST12_DRY) & ('DD2' in P)
        dust = np.where(size2, np.where(coated, P.get('DS2', -1), P.get('DD2', -1)),
                        np.where(coated, P['DS1'], P['DD1']))
        out = np.where(only('DUST'), dust, out)
        # sea salt (+ inorganics)
        if 'SSS' in P:
            ss = np.full(out.shape, P['SSS'])
        else:
            ss = np.where(dd < D_SSA_SSC_DRY, P['SSA'], P['SSC'])
        out = np.where(only('SEAS'), ss, out)
        # OC (+ inorganics)
        out = np.where(only('OCAR'), P['OCC'], out)
        # BC (+ inorganics): BC1 <= 5 %, BC2 5-20 %, BC3 > 20 % (no BC3: BC2 > 5 %)
        r = np.where(L['BCAR'] > 0, coat / np.where(L['BCAR'] > 0, L['BCAR'], 1.0), 0.0)
        bc = np.where(r > MIMR_BC1, P['BC2'], P['BC1'])
        if 'BC3' in P:
            bc = np.where(r > MIMR_BC2, P['BC3'], bc)
        out = np.where(only('BCAR'), bc, out)
        # BC + dust, BC + OC
        two = (n_other == 2) & has['BCAR']
        if 'DBC' in P:
            out = np.where(two & has['DUST'], P['DBC'], out)
        if 'BOC' in P:
            out = np.where(two & has['OCAR'], P['BOC'], out)
        out = np.where(m.sum(-1) > 0, out, -1)
        return out

    def project(self, m, w, calcsize_dt=None, number_scale=None):
        """Particles m (P, n, 15) kg, w (P, n) -> per-particle states x (P, d), then MATRIX's adjustment once
        (calcsize_dt = 0 skips it; any other value, or the default, applies it: MATRIX's operations are per
        call). number_scale (P,): the populations' number concentrations (m-3), for the AKK -> ACC transfer's
        1 cm-3 threshold (None: not applied)."""
        m = np.asarray(m, float)
        w = np.asarray(w, float)
        if m.ndim == 2:
            m, w = m[None], w[None]
        wn = w / w.sum(-1, keepdims=True)
        pop = self.assign(m)
        L = lumped_masses(m)
        x = np.zeros((len(m), self.d))
        for s in BULK:
            x[:, self.idx[(-1, 'mass', s)]] = (wn * L[s]).sum(-1)
        for i, p in enumerate(self.pops):
            sel = np.where(pop == i, wn, 0.0)
            x[:, self.idx[(i, 'number', None)]] = sel.sum(-1)
            for s in POP_SPECIES[p]:
                k = self.idx.get((i, 'mass', s))
                if k is None:                                    # SSC's sulfate: in SSA's tracer
                    k = self.idx[(self.pi['SSA'], 'mass', s)]
                x[:, k] += (sel * L[s]).sum(-1)
        # species of a particle its population does not carry: none by construction (assign picks the
        # population carrying every MATRIX species the particle has); checked here
        for i, p in enumerate(self.pops):
            for s in set(MSPC) - set(POP_SPECIES[p]):
                if np.any((pop == i) & (L[s] > 0) & (w > 0)):
                    raise AssertionError('%s: particle with %s assigned to %s' % (self.name, s, p))
        dt = self.calcsize_dt if calcsize_dt is None else calcsize_dt
        if dt:
            x = adjust(x.T, dt, self, xp=np, number_scale=number_scale).T
        return x

    # --- diagnostics -----------------------------------------------------------------------------------
    def population_masses(self, x):
        """States x (P, d) -> per population masses per particle of the cell, (P, npop, 15) PartMC species
        (as diagnosed), and numbers (P, npop). NO3 and NH4 are shared by sulfate, H2O by sulfate + sea salt
        (TRAMP_matrix.f l. 452-469; paper Sect. 2.5); sea-salt sulfate between SSA and SSC by sea salt
        (l. 337-346). If a cell has no sulfate (no sulfate or sea salt for H2O), the total goes by number."""
        x = np.atleast_2d(np.asarray(x, float))
        P = len(x)
        xm = np.maximum(x, 0.0)
        mass = np.zeros((P, self.np, NS))
        for (i, kind, s), k in self.idx.items():
            if i >= 0 and kind == 'mass':
                mass[:, i, SP[self.diag_as[s]]] += xm[:, k]
        if self.two_ss:
            a, c = self.pi['SSA'], self.pi['SSC']
            seas_a, seas_c = xm[:, self.idx[(a, 'mass', 'SEAS')]], xm[:, self.idx[(c, 'mass', 'SEAS')]]
            tot = seas_a + seas_c
            f = np.where(tot > 0, seas_c / np.where(tot > 0, tot, 1.0), 0.0)
            ss_sulf = mass[:, a, SP['SO4']].copy()
            mass[:, c, SP['SO4']] = ss_sulf * f
            mass[:, a, SP['SO4']] = ss_sulf * (1.0 - f)
        num = xm[:, self.i_num]
        sulf = mass[:, :, SP['SO4']]
        hyg = sulf + mass[:, :, SP['Na']]
        byn = num / np.maximum(num.sum(1, keepdims=True), 1e-300)
        for s, share in (('NO3', sulf), ('NH4', sulf), ('H2O', hyg)):
            tot = share.sum(1, keepdims=True)
            f = np.where(tot > 0, share / np.where(tot > 0, tot, 1.0), byn)
            mass[:, :, SP[self.diag_as[s]]] += f * xm[:, self.idx[(-1, 'mass', s)]][:, None]
        return mass, num

    def dgn(self, x):
        """Number geometric mean dry diameter (P, npop) from number and dry volume (nan where empty)."""
        mass, num = self.population_masses(x)
        vol = (mass[..., diag.DRY] / RHO[diag.DRY]).sum(-1)
        ok = (num > 0) & (vol > 0)
        out = np.full(num.shape, np.nan)
        out[ok] = (vol[ok] / (np.broadcast_to(self.cmn, num.shape)[ok] * num[ok])) ** (1.0 / 3.0)
        return out

    def particles(self, x):
        """Representative particles of states x (P, d): masses (P, R, 15) kg and weights (P, R). Per
        population: modal.py's nodes and weights (lognormal_nodes) for its sigma, dgn from its number and dry
        volume; a population with mass and no number gets MATRIX's default diameter DG_*."""
        mass, num = self.population_masses(x)
        ms, ws = [], []
        for i in range(self.np):
            mi, wi = lognormal_nodes(mass[:, i], num[:, i], self.sigma[i], self.dg_default[i])
            ms.append(mi)
            ws.append(wi)
        return np.concatenate(ms, axis=1), np.concatenate(ws, axis=1)

    def layout(self):
        comps = []
        for k, (i, kind, s) in enumerate(self.comps):
            c = {'index': k, 'population': 'total' if i < 0 else self.pops[i], 'population_index': int(i),
                 'kind': kind}
            if kind == 'number':
                c.update(units_per_particle='1 (fraction of particles in the population)',
                         units_extensive='# kg-1 (Q = N x)')
            else:
                c.update(species=s, partmc_species=list(LUMP[s]), diagnosed_as=self.diag_as[s],
                         units_per_particle='kg (mass per particle of the cell)',
                         units_extensive='kg kg-1 (Q = N x)')
                if s == 'SEAS':
                    c['partmc_note'] = 'Na, and Cl up to Cl/Na = %.4f (NaCl)' % R_CL_NA
                if s == 'NO3':
                    c['partmc_note'] = 'NO3, and Cl in excess of NaCl'
                if i < 0:
                    c['shared'] = ('by sulfate' if s != 'H2O' else 'by sulfate + sea salt') + \
                        ' (TRAMP_matrix.f l. 452-469)'
                if self.two_ss and i == self.pi.get('SSA') and s == 'SULF':
                    c['note'] = 'sea-salt sulfate of SSA and SSC, split by sea salt mass (TRAMP_matrix.f l. 337-346)'
            comps.append(c)
        return {
            'model': self.name, 'kind': 'matrix', 'mechanism': self.mech, 'n_tracers': self.d,
            'paper_transported_species': PAPER_SPECIES[self.mech], 'modelE212_ntmAMP': MODELE_NTMAMP[self.mech],
            'components': comps,
            'populations': [{'name': p, 'sigmag': SIGMA[p], 'dg_default_m': DG_UM[p] * 1e-6,
                             'species': list(POP_SPECIES[p])} for p in self.pops],
            'calcsize': 'baselines/matrix.py: adjust(Q, dt, model, xp=numpy|cupy) -> Q; MATRIX\'s state-driven '
                        'operations (sea-salt number from mass, class transfers, AKK -> ACC); per call',
            'calcsize_applied_to_sources': bool(self.calcsize_dt),
            'thresholds': {'MIMR_DDD': MIMR_DDD, 'MIMR_BC1': MIMR_BC1, 'MIMR_BC2': MIMR_BC2},
            'densities_kg_m3': {s: float(RHO[SP[self.diag_as[s]]]) for s in MSPC + BULK},
        }


def lognormal_nodes(mass, n, sigma, dg_default):
    """modal.Modal.particles' representative particles for one lognormal population of fixed sigma:
    mass (P, 15) per particle of the cell, number n (P,) -> masses (P, R, 15), weights (P, R). Nodes on the
    bin-aligned grid in log10 wet diameter (modal.T_LO, T_HI, GRID_PER_BIN), weights the lognormal density,
    node masses scaled so that number, dry volume and every species mass are exact. A population with mass
    and no number gets dgn = dg_default and the implied number."""
    P = len(mass)
    lns = math.log(sigma)
    cmn = math.exp(4.5 * lns ** 2) * math.pi / 6.0
    vol = (mass[:, diag.DRY] / RHO[diag.DRY]).sum(-1)
    wvol = (mass / RHO).sum(-1)
    n = np.maximum(n, 0)
    nofix = (n <= 0) & (vol > 0)
    n = np.where(nofix, vol / (cmn * dg_default ** 3), n)
    ok = (n > 0) & (vol > 0)
    vbar = np.where(ok, vol / np.where(ok, n, 1), 0.0)
    dgn = (vbar / cmn) ** (1.0 / 3.0)
    rw = np.where(ok, wvol / np.where(ok, vol, 1), 1.0) ** (1.0 / 3.0)
    ra = rw if modal.GRID_BASIS == 'wet' else np.ones_like(rw)          # grid alignment
    lo = np.log10(np.where(ok, dgn * ra, 1.0)) + modal.T_LO * lns / math.log(10)
    k0 = np.floor((lo - modal.LOG10_D0) / modal.DLOG10).astype(int)
    nk = int(math.ceil((modal.T_HI + 3 * lns - modal.T_LO) * lns / math.log(10) / modal.DLOG10)) + 2
    lw = modal.LOG10_D0 + modal.DLOG10 * (k0[:, None] + np.arange(nk)[None, :] + 0.5)
    D = 10 ** lw / ra[:, None]
    t = np.log(D / np.where(ok, dgn, 1.0)[:, None]) / lns
    p = np.where((t >= modal.T_LO) & (t <= modal.T_HI + 3 * lns), np.exp(-0.5 * t ** 2), 0.0)
    p = p / np.maximum(p.sum(-1, keepdims=True), 1e-300)
    v = math.pi / 6 * D ** 3
    c = np.where(ok, vbar / np.maximum((p * v).sum(-1), 1e-300), 0.0)
    per = np.where(ok[:, None], mass / np.where(ok, n, 1)[:, None], 0.0)
    scale = np.where(ok[:, None], c[:, None] * v / np.where(ok, vbar, 1)[:, None], 0.0)
    return scale[:, :, None] * per[:, None, :], np.where(ok[:, None], n[:, None] * p, 0.0)


# --- MATRIX's state-driven operations ------------------------------------------------------------------

_MODELS = {}


def _model(model):
    if isinstance(model, Matrix):
        return model
    if model not in _MODELS:
        _MODELS[model] = Matrix(model)
    return _MODELS[model]


def _xp_of(a):
    try:
        import cupy
        return cupy.get_array_module(a)
    except ImportError:
        return np


def adjust(Q, dt=None, model='matrix1', xp=None, do_seasalt_number=True, do_class=True, do_imtr=True,
           number_scale=None, stats=None):
    """MATRIX's state-driven operations, i.e. what one call of TRAMP_matrix.f's MATRIX does to a state when
    no process acts (no emission, nucleation, coagulation, condensation, cloud sulfate or thermodynamics):

      1. sea-salt numbers from NaCl mass: N(SSA/SSC/SSS) = SEAS / (NaCl mass per emitted particle)
         (l. 313-324, written back at l. 997);
      2. class transfers of whole populations, in MATRIX's order DD1 -> DS1, DD2 -> DS2, BC1 -> BC2,
         BC2 -> BC3, when SULF x OPTOT > MIMR x core (l. 1112-1171), OPTOT = 1 + (NO3 + NH4 + H2O) / total
         SULF (the cell's totals, l. 1118-1119); a zero core with sulfate transfers (MATRIX floors the core
         at 1e-30, l. 1113-1116);
      3. AKK -> ACC (l. 1194-1252, IMTR_METHOD 1): FNUM = clip(((Dp_AKK - DNU) / (Dp_ACC - DNU))^4, 0,
         0.5) if Dp_AKK >= DNU else 0, of AKK's number and sulfate, Dp the diameter of average mass of
         SULF x OPTOT / N at DENSP = 1400 kg m-3; only if N_AKK > 1e6 m-3 when number_scale (m-3 per unit
         of Q's number) is given, else for any AKK number.
    The sea-salt sulfate split and merge (l. 337-346, 1257-1264), MASSADJ (l. 1277-1280) and the floors
    (l. 1286) are identities or negligible here and not applied.

    Q (d, ...) extensive (per kg of air, or per particle: homogeneous of degree one except the 1 cm-3
    threshold), component order of Matrix(model). dt is ignored: these operations are per call (ModelE
    calls MATRIX once per physics step, DTsrc = 1800 s). xp: numpy or cupy (elementwise only). stats: a dict
    to which the count of cells where each operation acted is added (numpy ints / cupy scalars).
    Returns a new array."""
    M = _model(model)
    xp = xp or _xp_of(Q)
    Q = Q.copy()
    if do_seasalt_number:
        for p in M.ss_pops:
            Q[M.comp(p, 'number')] = xp.maximum(Q[M.comp(p, 'SEAS')], 0.0) / M.ss_mpp[p]
    tot_sulf = xp.zeros_like(Q[0])
    for k in M.sulf:
        tot_sulf = tot_sulf + Q[k]
    bulk = Q[M.i_bulk[0]] + Q[M.i_bulk[1]] + Q[M.i_bulk[2]]
    optot = 1.0 + xp.where(tot_sulf > 0, bulk / xp.where(tot_sulf > 0, tot_sulf, 1.0), 0.0)
    if do_class:
        for src, dst, core, mimr in M.transfers:
            ks, kc = M.comp(src, 'SULF'), M.comp(src, core)
            cond = Q[ks] * optot > mimr * xp.maximum(Q[kc], 0.0)
            if stats is not None:
                stats['%s->%s' % (src, dst)] = stats.get('%s->%s' % (src, dst), 0) + cond.sum()
            for s in POP_SPECIES[src] + ('number',):
                a, b = M.comp(src, s), M.comp(dst, s)
                Q[b] = Q[b] + xp.where(cond, Q[a], 0.0)
                Q[a] = xp.where(cond, 0.0, Q[a])
    if do_imtr and M.imtr:
        na, nc = Q[M.comp('AKK', 'number')], Q[M.comp('ACC', 'number')]
        sa, sc = Q[M.comp('AKK', 'SULF')], Q[M.comp('ACC', 'SULF')]
        c3 = 6.0 / (math.pi * DENSP)
        dpa = (c3 * xp.maximum(sa, 0.0) * optot / xp.where(na > 0, na, 1.0)) ** (1.0 / 3.0)
        dpc = (c3 * xp.maximum(sc, 0.0) * optot / xp.where(nc > 0, nc, 1.0)) ** (1.0 / 3.0)
        ratio = (dpa - DNU) / xp.where(dpc != DNU, dpc - DNU, 1e-300)
        fnum = xp.where(dpa >= DNU, xp.minimum(xp.maximum(ratio ** IMTR_EXP, 0.0), FNUM_MAX), 0.0)
        fnum = xp.where((na > 0) & (nc > 0), fnum, 0.0)          # N_ACC = 0: Dp_ACC infinite, no transfer
        if number_scale is not None:
            fnum = xp.where(na * number_scale > AKK_MINNUM_IMTR, fnum, 0.0)
        if stats is not None:
            stats['AKK->ACC'] = stats.get('AKK->ACC', 0) + (fnum > 0).sum()
        dm, dn = sa * fnum, na * fnum
        Q[M.comp('AKK', 'SULF')] = sa - dm
        Q[M.comp('ACC', 'SULF')] = sc + dm
        Q[M.comp('AKK', 'number')] = na - dn
        Q[M.comp('ACC', 'number')] = nc + dn
    return Q


#: modal.calcsize's signature, for scripts that call calcsize(Q, dt, model)
calcsize = adjust


def describe():
    return {'matrix%d' % k: Matrix('matrix%d' % k).layout() for k in MECHANISMS}


if __name__ == '__main__':
    L = Matrix('matrix1').layout()
    print(L['n_tracers'], [c['population'] + ':' + str(c.get('species', 'N')) for c in L['components']])

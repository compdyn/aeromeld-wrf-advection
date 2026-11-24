"""TOMAS-40: GEOS-Chem's TOMAS sectional aerosol scheme with 40 bins, as a baseline
representation (key 'tomas40'), and its state-driven adjustment MNFIX (`mnfix`).

Sources: GEOS-Chem 14.8.0 (github.com/geoschem/geos-chem, tag 14.8.0 = commit
a4551f9442183bb572b23e9c2d362e2d34420d3a, released 2026-09-15; files fetched 2026-11-14, read
only). Line numbers refer to that commit.
  GeosCore/tomas_mod.F90 (md5 7545ba73...)
    L52-60      "boundaries between size bins are defined in terms of dry aerosol mass such that
                the actual sizes of the sections move as water is added to or lost from the aerosol"
    L107-117    species order SRTSO4 1, SRTNACL 2, SRTECOB 3, SRTECIL 4, SRTOCOB 5, SRTOCIL 6,
                SRTDUST 7, SRTNH4 8, SRTH2O 9
    L7040-7046  INIT_TOMAS: Mo = 1e-21 kg * 2**(-10) for TOMAS40
    L7048-7075  ICOMP = 9 (7 species + NH4 + H2O), IDIAG = 2 (NH4 and H2O: "diagnostic")
    L7231-7239  Xk(k) = Mo * 2**(k-1), k = 1..IBINS+1 (mass doubling); AVGMASS(k) = sqrt(Xk(k) Xk(k+1))
    L8428-9059  MNFIX (Peter Adams 2000, Win Trivitayanurak 2007): bins whose mean dry mass per
                particle (DRYMASS = sum of the ICOMP-IDIAG species, i.e. without NH4 and water)
                lies outside [Xk(k), Xk(k+1)] are fixed (see `mnfix` below)
    L637-650    AEROPHYS: EZWATEREQM, then "Fix any inconsistencies in M/N distribution (because
                of advection)": MNFIX on the incoming (transported) state; also L834, L881, L919
    L261-282    DO_TOMAS: AEROPHYS, CHECKMN (= MNFIX on every grid box, L8203-8408), dry
                deposition, CHECKMN again
    L9380-9434  NH4BULKTOBIN: the bulk NH4 tracer distributed over the bins in proportion to
                each bin's sulfate, capped at 0.375 x total sulfate ((NH4)2SO4)
    L1628 etc.  AERODENS(MSO4, 0 (nitrate), ...) (L1628, 1841, 2840, 4788, 5453, 9165): nitrate is
                never part of a TOMAS particle
    L7750-7815  EZWATEREQM: aerosol water from RH on sulfate (as NH4HSO4), NaCl and OCIL only
    L4351-4655  SOACOND: SOA condenses into OCIL (TMCOND(..., SRTOCIL, ...), L4584)
  GeosCore/carbon_mod.F90 (md5 7757ad9f...)
    L4553-4563  EMITSGC: emitted EC 20 % ECIL / 80 % ECOB, OC 50 % / 50 %, whatever the size
    L1449-1500  AGING_CARB: ECOB -> ECIL and OCOB -> OCIL, e-folding TAU_HYDRO = 1.15 day (L1479)
  GeosCore/chemistry_mod.F90 L629-645: Do_TOMAS is called inside Do_Chemistry.
  Interfaces/GCClassic/main.F90: Do_Transport (L1158), ..., Do_Mixing (L1433), Do_Convection
    (L1457), Do_Chemistry (L1514), Do_WetDep (L1540): MNFIX acts on the transported state at
    the next chemistry step. GeosCore/transport_mod.F90 has no TOMAS code: TOMAS tracers are
    advected as independent tracers.
  run/shared/setupConfigFiles.sh L316-744: a TOMAS40 run advects H2SO4 and 40 bins of NK, SF,
    SS, ECOB, ECIL, OCOB, OCIL, DUST and AW (361 species) besides fullchem's bulk species (NH4,
    NIT, ...).
Literature: Adams and Seinfeld (2002, JGR 107, 4370; TOMAS-30, 10 nm-10 um, mass doubling);
Lee, Pierce and Adams (2013, GMD 6, 1221; TOMAS-40 down to ~1 nm); Pierce et al. (2007, ACP 7,
5447; carbonaceous species); Trivitayanurak et al. (2008, ACP 8, 3149; TOMAS in GEOS-Chem);
Kodros et al. (2016, ACP 16, 1049). Cited from memory; the numbers here are the code's.

Bins. 40 bins of TOMAS dry mass per particle: Xk(1) = 9.765625e-25 kg (1.0 nm at 1800 kg m-3)
to Xk(41) = 1.0737e-12 kg (10.4 um), each bin a mass doubling. The binned mass is the sum of
TOMAS's own dry species (sulfate, sea salt, EC, OC, dust): it excludes water and ammonium
(TOMAS's two 'diagnostic' species), and nitrate, which TOMAS does not carry. Particles are
binned by this mass exactly (not by diameter; densities differ). A particle outside the range
goes to an end bin, whose mnfix then acts as TOMAS's does (none of the three sources'
particles is outside: their binned masses are 2.4e-22..2.4e-16 kg, bins 8..28).

Species (TOMAS's), and the lumping of PartMC's 15 species (config.ML_SPECIES) onto them:

  TOMAS (per bin)  PartMC                                           diagnosed as
  NK               number                                           -
  SF  sulfate      SO4                                              SO4
  SS  sea salt     Cl, Na                                           Cl (identical properties)
  ECOB             BC of fresh (uncoated) particles                 BC
  ECIL             BC of the other particles                        BC
  OCOB             OC, MOC of fresh particles                       OC
  OCIL             OC, MOC of the other particles; the SOA species  API1 (SOA)
                   API1, ARO1, ARO2, ALK1, OLE1 of every particle
  DUST             OIN                                              OIN
  AW  water        H2O (passive, as for every representation)      H2O
  bulk (one tracer each, not per bin):
  NH4              NH4: per bin in proportion to SF (NH4BULKTOBIN)  NH4
  NIT              NO3: per bin in proportion to SF                 NO3

  * Fresh carbon: E3SM MAM4's primary-carbon ageing criterion applied to one particle (fewer
    than 8 monolayers of kappa-weighted SO4-equivalent coating on the BC+OC+MOC core: modal.py,
    Modal('mam4').assign(m) == I_PCA), the same rule as the modal baselines. GEOS-Chem emits
    fixed fractions (EC 80 %, OC 50 % hydrophobic) and ages them with a 1.15-day e-folding time,
    a bulk stand-in for coating; our sources carry their particles' actual coatings, so the
    coating criterion, particle by particle, is the analogue. It selects 0.5 / 5 / 96 % of the
    BC and 92 / 45 / 97 % of the OC+MOC (background / urban / emission) as hydrophobic.
  * OCIL is diagnosed as PartMC's SOA species (density 1400 kg m-3 = TOMAS's organic density,
    AERODENS L9474; kappa 0.1): TOMAS treats OCIL as hygroscopic (EZWATEREQM) and condenses SOA
    into it. OCOB is diagnosed as OC (kappa 0.001). ECOB and ECIL are both BC: within a bin all
    species are internally mixed, so the split does not change the diagnostics; it only decides
    where aged OC is diagnosed.
  * NH4 and NO3: GEOS-Chem carries them as bulk tracers; TOMAS distributes NH4 over its bins in
    proportion to sulfate (capped at (NH4)2SO4 for its microphysics) and never sees nitrate. Here
    both are carried as bulk tracers and distributed in proportion to SF for the diagnostics,
    without the cap (all their mass is placed; NH4/SO4 is 0.37 / 0.86 / 0.50 in the sources,
    above the cap in two of them because of NH4NO3). A population with no SF puts them in
    proportion to the binned mass. They do not count in the bin bounds, as in TOMAS.
  * Tracers: 40 x 9 (NK + 7 dry species + AW) + 2 bulk = 362. TOMAS-40 in GEOS-Chem advects 361
    species (the 360 bin tracers and H2SO4 gas) plus the bulk ones.

State. Per-particle vector x in TOMAS's tracer order (species-major, as GEOS-Chem's NK01..NK40,
SF01..SF40, ...): x[j * 40 + k] for j = 0 NK (fraction of particles in bin k), 1 SF, 2 SS,
3 ECOB, 4 ECIL, 5 OCOB, 6 OCIL, 7 DUST, 8 AW (mass per particle of the population in bin k,
kg), then x[360] NH4 and x[361] NO3 (bulk, kg per particle of the population). Q = N x;
mixing averages x in number, so under mixing alone every bin's mean dry mass stays inside its
bounds; only the limiters can move it out.

Representative particles: those of the existing sectional (sectional.Sectional.particles; the
coag paper's n_rep = 20 nodes, max-entropy weights matching the bin's number and mean binned
mass), with nodes log-uniform in TOMAS dry mass between Xk(k) and Xk(k+1). Every node has the
bin's mass fractions (SF, SS, ..., the NH4 and NO3 attached to the bin, and water), so a node's
dry volume is its binned mass divided by the bin's mean dry density (binned mass per dry volume,
NH4 and NO3 included): the mass bounds become dry-volume bounds through that density.

Adjustments between transport steps (principle: state-driven ones applied, process-driven ones
skipped; this case has no processes):
  applied   MNFIX: acts on any state, after every 20-s transport step here (GEOS-Chem: at the
            start of AEROPHYS, once per chemistry step, after transport)
  skipped   EZWATEREQM (water from RH; water is passive here, as for every representation);
            AGING_CARB (1.15-day ageing time scale); nucleation, condensation, coagulation
            (AEROPHYS), SOACOND, AQOXID (in-cloud sulfate), sub-grid coagulation of emissions
            (EMITSGC), dry and wet deposition: process-driven
"""
import math

import numpy as np

import diag
import sectional

SPECIES = diag.SPECIES
NS = diag.NS
SP = diag.SP
RHO = diag.RHO

# --- bins (tomas_mod.F90 INIT_TOMAS, TOMAS40) -------------------------------------------------
NBINS = 40
MO = 1.0e-21 * 2.0 ** (-10)                                  # L7041
XK = MO * 2.0 ** np.arange(NBINS + 1)                        # L7232-7234, kg dry mass per particle
AVGMASS = np.sqrt(XK[:-1] * XK[1:])                          # L7237-7239

# --- TOMAS species (tracer order) -------------------------------------------------------------
TSPECIES = ('NK', 'SF', 'SS', 'ECOB', 'ECIL', 'OCOB', 'OCIL', 'DUST', 'AW')
J_NK, J_SF, J_SS, J_ECOB, J_ECIL, J_OCOB, J_OCIL, J_DUST, J_AW = range(9)
DRY_J = slice(1, 8)                                          # SF..DUST: TOMAS's DRYMASS (ICOMP-IDIAG)
BULK = ('NH4', 'NIT')
I_NH4, I_NO3 = 9 * NBINS, 9 * NBINS + 1
D = 9 * NBINS + len(BULK)                                    # 362
#: PartMC species of each TOMAS species (ECOB/ECIL and OCOB/OCIL split by the fresh-carbon rule)
LUMP = {'SF': ('SO4',), 'SS': ('Cl', 'Na'), 'ECOB': ('BC',), 'ECIL': ('BC',),
        'OCOB': ('OC', 'MOC'), 'OCIL': ('OC', 'MOC', 'API1', 'ARO1', 'ARO2', 'ALK1', 'OLE1'),
        'DUST': ('OIN',), 'AW': ('H2O',), 'NH4': ('NH4',), 'NIT': ('NO3',)}
#: The PartMC species each TOMAS species is diagnosed as
DIAG_AS = {'SF': 'SO4', 'SS': 'Cl', 'ECOB': 'BC', 'ECIL': 'BC', 'OCOB': 'OC', 'OCIL': 'API1',
           'DUST': 'OIN', 'AW': 'H2O', 'NH4': 'NH4', 'NIT': 'NO3'}
SOA = ('API1', 'ARO1', 'ARO2', 'ALK1', 'OLE1')
#: PartMC species whose mass counts in the bin bounds: every dry species but NH4 and NO3
BIN_SEL = np.array([diag.DRY[k] and s not in ('NH4', 'NO3') for k, s in enumerate(SPECIES)])

# --- MNFIX constants (tomas_mod.F90 L8457-8460) and their per-kg-of-air analogues -------------
#: TOMAS's floors are per grid box: EPS = 1e-20 particles, EPS2 = 1e-28 kg; a bin below either
#: is replaced by EPS particles of sulfate per box, i.e. emptied to working precision. Here the
#: state is per kg of air; the floors are divided by M_AIR_CELL, the order of a WRF surface
#: cell's air mass (4 km x 4 km x ~60 m x 1.2 kg m-3), and such a bin is emptied.
M_AIR_CELL = 1.0e9
NK_FLOOR = 1.0e-20 / M_AIR_CELL
M_FLOOR = 1.0e-28 / M_AIR_CELL


def fresh_carbon(m):
    """Fresh (uncoated) carbon particles: MAM4's ageing criterion per particle (modal.py), the
    primary-carbon assignment of Modal('mam4') (no size limit)."""
    import modal
    return modal.Modal('mam4').assign(m) == modal.I_PCA


def tomas_masses(m, fresh):
    """PartMC masses (..., 15) -> TOMAS species masses (..., 8: SF..AW) and bulk (..., 2)."""
    g = lambda names: m[..., [SP[s] for s in names]].sum(-1)
    bc, oc = m[..., SP['BC']], g(('OC', 'MOC'))
    t = np.stack([g(('SO4',)), g(('Cl', 'Na')),
                  np.where(fresh, bc, 0.0), np.where(fresh, 0.0, bc),
                  np.where(fresh, oc, 0.0), np.where(fresh, 0.0, oc) + g(SOA),
                  m[..., SP['OIN']], m[..., SP['H2O']]], -1)
    bulk = np.stack([m[..., SP['NH4']], m[..., SP['NO3']]], -1)
    return t, bulk


class Tomas(sectional.Sectional):
    """TOMAS-40 ('tomas40'). project(m, w) -> x (P, 362); particles(x) -> representative
    particles (via sectional.Sectional.particles, binned in TOMAS dry mass); layout()."""

    def __init__(self, name='tomas40', n_rep=sectional.N_REP):
        if name != 'tomas40':
            raise ValueError(name)
        super().__init__(name, n_rep=n_rep, edges=XK, basis='dry')
        self.basis = 'tomas_dry_mass'
        self.d = D
        self.i_num = np.arange(NBINS) + J_NK * NBINS
        #: (bin or None, kind, TOMAS species) per component, for transport_terms' fit weights
        self.comps = ([(k, 'number', None) for k in range(NBINS)] +
                      [(k, 'mass', s) for s in TSPECIES[1:] for k in range(NBINS)] +
                      [(None, 'mass', s) for s in BULK])

    # --- binning --------------------------------------------------------------------------------
    def binned_mass(self, m):
        """TOMAS dry mass per particle (..., 15) -> (...): SF + SS + EC + OC + DUST in PartMC terms."""
        return m[..., BIN_SEL].sum(-1)

    def bin_of(self, m):
        b = np.searchsorted(self.edges, self.binned_mass(m), side='right') - 1
        return np.clip(b, 0, self.nb - 1)

    def project(self, m, w, mnfix_once=True):
        """Particles m (P, n, 15) kg, w (P, n) -> per-particle states x (P, 362), then MNFIX once,
        as Modal.project applies calcsize once (a no-op unless particles lie outside the range:
        exact binning puts every bin's mean inside its bounds; mnfix_once=False skips it)."""
        m = np.asarray(m, float)
        w = np.asarray(w, float)
        if m.ndim == 2:
            m, w = m[None], w[None]
        wn = w / w.sum(-1, keepdims=True)
        t, bulk = tomas_masses(m, fresh_carbon(m))
        b = self.bin_of(m)
        x = np.zeros((len(m), 9, NBINS))
        for k in range(NBINS):
            sel = np.where(b == k, wn, 0.0)
            x[:, J_NK, k] = sel.sum(-1)
            x[:, 1:, k] = np.einsum('pn,pnj->pj', sel, t)
        x = np.concatenate([x.reshape(len(m), -1), np.einsum('pn,pnj->pj', wn, bulk)], 1)
        if mnfix_once:
            x = mnfix(x.T).T
        return x

    # --- diagnostics ----------------------------------------------------------------------------
    def expand(self, x):
        """TOMAS states x (P, 362) -> sectional states (P, 40 * 16) in PartMC species (number,
        then 15 species per bin), each TOMAS species as DIAG_AS, the bulk NH4 and NO3 per bin in
        proportion to SF (else to the binned mass)."""
        x = np.atleast_2d(np.asarray(x, float))
        P = len(x)
        a = np.maximum(x[:, :9 * NBINS].reshape(P, 9, NBINS), 0.0)
        out = np.zeros((P, NBINS, 1 + NS))
        out[..., 0] = a[:, J_NK]
        for j, s in enumerate(TSPECIES[1:], 1):
            out[..., 1 + SP[DIAG_AS[s]]] += a[:, j]
        sf = a[:, J_SF]
        dry = a[:, DRY_J].sum(1)
        tot_sf, tot_dry = sf.sum(-1, keepdims=True), dry.sum(-1, keepdims=True)
        share = np.where(tot_sf > 0, sf / np.where(tot_sf > 0, tot_sf, 1.0),
                         np.where(tot_dry > 0, dry / np.where(tot_dry > 0, tot_dry, 1.0), 0.0))
        for i, s in ((I_NH4, 'NH4'), (I_NO3, 'NO3')):
            out[..., 1 + SP[s]] += np.maximum(x[:, i], 0.0)[:, None] * share
        return out.reshape(P, -1)

    def measure(self, mass):
        """Binned variable of the expanded state's bins: TOMAS dry mass (no NH4, NO3, H2O)."""
        return mass[..., BIN_SEL].sum(-1)

    def node_measure(self, D):
        """nodes() of the mass edges are masses already."""
        return D

    def particles(self, x):
        return super().particles(self.expand(x))

    def dry_mass_per_particle(self, x):
        """Mean TOMAS dry mass per particle in each bin (P, 40), nan where a bin is empty."""
        x = np.atleast_2d(np.asarray(x, float))
        a = x[:, :9 * NBINS].reshape(len(x), 9, NBINS)
        n, dry = a[:, J_NK], a[:, DRY_J].sum(1)
        with np.errstate(invalid='ignore', divide='ignore'):
            return np.where(n > 0, dry / np.where(n > 0, n, 1.0), np.nan)

    def layout(self):
        comps = []
        for idx, (k, kind, s) in enumerate(self.comps):
            c = {'index': idx, 'kind': kind}
            if k is not None:
                c.update(bin=k, dry_mass_per_particle_kg=[float(XK[k]), float(XK[k + 1])])
            if kind == 'number':
                c.update(species='NK', units_per_particle='1 (fraction of particles in the bin)',
                         units_extensive='# kg-1 (Q = N x)')
            else:
                c.update(species=s, partmc_species=list(LUMP[s]), diagnosed_as=DIAG_AS[s],
                         units_per_particle='kg (mass per particle of the population%s)'
                         % ('' if k is not None else '; bulk, distributed over the bins in proportion to SF'),
                         units_extensive='kg kg-1 (Q = N x)')
            comps.append(c)
        return {'model': self.name, 'kind': 'sectional', 'scheme': 'TOMAS-40 (GEOS-Chem 14.8.0)',
                'n_tracers': self.d, 'basis': 'TOMAS dry mass per particle (SF+SS+EC+OC+DUST; '
                'no NH4, NO3, H2O)', 'tomas_species': list(TSPECIES), 'bulk_species': list(BULK),
                'xk_kg': [float(v) for v in XK], 'avgmass_kg': [float(v) for v in AVGMASS],
                'outside_range': 'particles below Xk(1) / above Xk(41) go to the end bins (then mnfix)',
                'hydrophobic_rule': 'fresh carbon by MAM4\'s 8-monolayer criterion per particle (modal.py)',
                'n_rep': self.n_rep, 'components': comps, 'calcsize': None,
                'mnfix': 'baselines/tomas.py: mnfix(Q, xp=numpy|cupy) -> Q; Q (362, ...) extensive per kg '
                         'in this component order; GEOS-Chem calls it at each chemistry step (here: after '
                         'each 20 s transport step)'}


# --- MNFIX ------------------------------------------------------------------------------------

def _xp_of(a):
    try:
        import cupy
        return cupy.get_array_module(a)
    except ImportError:
        return np


MNFIX_EVENTS = ('negative', 'floor', 'top_mass_removed', 'bottom_number_removed', 'shift_up',
                'shift_up_top_mass_removed', 'shift_down', 'shift_down_bottom_number_removed',
                'inconsistent_emptied')


def mnfix(Q, xp=None, inplace=False, stats=None, water='with_dry'):
    """TOMAS's MNFIX (tomas_mod.F90 L8428-9059) on extensive tracers.

    Q   (362, ...) per kg of air (or per particle) in Tomas().layout() order; any trailing shape.
        Only the 360 bin tracers are touched; the bulk NH4 and NO3 are not TOMAS's.
    xp  numpy or cupy (default: inferred). inplace: modify Q itself. stats: a dict that gets the
        number of (cell, bin) events of each kind added (MNFIX_EVENTS).
    Returns the fixed Q.

    Steps, as in the code (DRYMASS = SF+SS+ECOB+ECIL+OCOB+OCIL+DUST; AVG = DRYMASS / NK):
      0. negative values set to 0 (TOMAS: values above -1 particle / -1e-5 kg per box; below,
         it stops the model);
      1. floors: a bin with NK <= EPS or DRYMASS < EPS2 is replaced by EPS particles of sulfate
         per box (L8588-8638), here emptied (NK_FLOOR, M_FLOOR per kg of air);
      2. every bin (L8666-8706): AVG > Xk(41): all masses scaled so AVG = Xk(41) / 1.2 (mass
         removed); AVG < Xk(1): NK = DRYMASS / (1.2 Xk(1)) (number removed);
      3. bins k = 1..40 in order (L8746-8899), AVG from the current state:
         AVG > Xk(k+1): KK = the first bin >= k with Xk(KK+1)/1.1 > AVG; NSHIFT particles of
           dry mass XNEW = Xk(KK+1)/1.1 move to KK, leaving bin k at XOLD = AVGMASS(k):
           NSHIFT = (DRYMASS - XOLD NK) / (XNEW - XOLD); number and dry mass conserved, each
           species in the bin's proportions. No such bin (AVG >= Xk(41)/1.1): masses scaled
           so AVG = Xk(k+1)/1.1 (mass removed).
         AVG < Xk(k): KK = the last bin <= k with 1.1 Xk(KK) < AVG; NSHIFT = NK - DRYMASS/XOLD
           particles move to KK with dry mass XNEW = 1.1 Xk(KK) each, in the bin's proportions;
           bin k keeps all its mass (code since 10/20/08, L8873-8875): number conserved, mass
           MSHIFT = XNEW NSHIFT created in bin KK. No such bin: NK = DRYMASS / (1.2 Xk(1)).
      4. negatives set to 0; 5. a bin with number but no dry mass, or dry mass but no number,
         emptied (L8994-9010).
    Water (passive here). TOMAS moves AW with the bin's proportions in the downward shift and
    in the mass scalings, but not in the upward shift, where it stays behind; GEOS-Chem
    recomputes AW from RH (EZWATEREQM) right before MNFIX, so this never matters there. Here
    water is carried, and water='with_dry' (default) moves it with the dry mass in the upward
    shift too, keeping each particle's water; water='tomas' leaves it behind, as the code does.
    """
    xp = xp or _xp_of(Q)
    if not inplace:
        Q = Q.copy()
    Q = xp.ascontiguousarray(Q)                                          # the same array if contiguous
    A = Q.reshape(Q.shape[0], -1)[:9 * NBINS].reshape(9, NBINS, -1)      # view: (species, bin, cell)
    count = (lambda key, mask: stats.__setitem__(key, stats.get(key, 0) + int(mask.sum()))) \
        if stats is not None else (lambda key, mask: None)

    def clip_negatives():
        for j in range(9):
            count('negative', A[j] < 0)
            xp.maximum(A[j], 0.0, out=A[j])

    def empty(mask):
        for j in range(9):
            A[j] = xp.where(mask, 0.0, A[j])

    # 0. negatives
    clip_negatives()
    # 1. floors
    dry = A[DRY_J].sum(0)                                                # (bin, cell)
    nk = A[J_NK]
    floor = ((nk <= NK_FLOOR) | (dry < M_FLOOR)) & ((nk > 0) | (dry > 0))
    count('floor', floor)
    if bool(floor.any()):
        empty(floor)
    # 2. any bin out of the whole range
    dry = A[DRY_J].sum(0)
    nk = A[J_NK]
    occ = nk > 0
    avg = xp.where(occ, dry / xp.where(occ, nk, 1.0), 0.0)
    top = occ & (avg > XK[-1])
    bot = occ & (avg < XK[0])
    count('top_mass_removed', top)
    count('bottom_number_removed', bot)
    if bool(top.any()):
        f = xp.where(top, nk * (XK[-1] / 1.2) / xp.where(top, dry, 1.0), 1.0)
        for j in range(1, 9):                                            # J = 1..ICOMP: water too
            A[j] *= f
    if bool(bot.any()):
        A[J_NK] = xp.where(bot, dry / (XK[0] * 1.2), A[J_NK])
    del dry, nk, occ, avg, top, bot
    # 3. bin by bin
    xk_up = xp.asarray(XK[1:] / 1.1)                                     # XNEW candidates, upward
    xk_dn = xp.asarray(XK[:-1] * 1.1)                                    # XNEW candidates, downward
    up_species = range(1, 9) if water == 'with_dry' else range(1, 8)
    for k in range(NBINS):
        nk = A[J_NK, k]
        dry = A[DRY_J, k].sum(0)
        occ = nk > 0
        avg = xp.where(occ, dry / xp.where(occ, nk, 1.0), AVGMASS[k])
        hi = avg > XK[k + 1]
        lo = avg < XK[k]
        if bool(hi.any()):
            c = xp.nonzero(hi)[0]
            a, n, dm = avg[c], nk[c], dry[c]
            kk = xp.searchsorted(xk_up, a, side='right')                 # first KK with Xk(KK+1)/1.1 > AVG
            room = kk < NBINS
            count('shift_up', room)
            count('shift_up_top_mass_removed', ~room)
            if bool((~room).any()):                                      # no bin can take them: scale
                ct = c[~room]                                            # every mass (water too) so
                f = n[~room] * (XK[k + 1] / 1.1) / dm[~room]             # that AVG = Xk(k+1)/1.1
                for j in range(1, 9):
                    A[j, k, ct] *= f
            if bool(room.any()):
                cs, kk, n, dm = c[room], kk[room], n[room], dm[room]
                xold = AVGMASS[k]
                xnew = xk_up[kk]
                nshift = (dm - xold * n) / (xnew - xold)
                mshift = xnew * nshift
                nnew = n - nshift
                A[J_NK, k, cs] = nnew
                A[J_NK, kk, cs] += nshift
                for j in up_species:                                     # J = 1..ICOMP-IDIAG (+ water)
                    fj = A[j, k, cs] / dm
                    A[j, k, cs] = xold * nnew * fj
                    A[j, kk, cs] += mshift * fj
        if bool(lo.any()):
            c = xp.nonzero(lo)[0]
            a, n, dm = avg[c], nk[c], dry[c]
            kk = xp.searchsorted(xk_dn, a, side='left') - 1              # last KK with 1.1 Xk(KK) < AVG
            room = kk >= 0
            count('shift_down', room)
            count('shift_down_bottom_number_removed', ~room)
            if bool((~room).any()):
                A[J_NK, k, c[~room]] = dm[~room] / (XK[0] * 1.2)
            if bool(room.any()):
                cs, kk, n, dm = c[room], kk[room], n[room], dm[room]
                xold = AVGMASS[k]
                xnew = xk_dn[kk]
                nshift = n - dm / xold
                mshift = xnew * nshift
                nnew = n - nshift
                A[J_NK, k, cs] = nnew
                A[J_NK, kk, cs] += nshift
                for j in range(1, 9):                                    # J = 1..ICOMP: water too
                    fj = A[j, k, cs] / dm
                    A[j, k, cs] = xold * nnew * fj
                    A[j, kk, cs] += mshift * fj
    # 4. negatives after fixing; 5. bins with number but no dry mass or the reverse
    clip_negatives()
    dry = A[DRY_J].sum(0)
    nk = A[J_NK]
    bad = ((nk != 0) & (dry == 0)) | ((nk == 0) & (dry != 0))
    count('inconsistent_emptied', bad)
    if bool(bad.any()):
        empty(bad)
    return Q


def describe():
    return Tomas().layout()


if __name__ == '__main__':
    import json
    L = describe()
    print(json.dumps({k: v for k, v in L.items() if k != 'components'}, indent=1)[:3000])

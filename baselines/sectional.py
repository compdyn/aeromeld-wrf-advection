"""Sectional (MOSAIC-like) representation: number and the mass of each of the 15 species
(incl. H2O) per bin, bins in DRY diameter.

Bins (dry diameter, MOSAIC-like):
  sec8   MOSAIC 8 bins, WRF-Chem's edges 0.039, 0.078, 0.156, 0.3125, 0.625, 1.25, 2.5, 5, 10 um
  sec4   MOSAIC 4 bins, 0.039, 0.156, 0.625, 2.5, 10 um
  sec20  20 bins log-spaced 0.001-10 um
and in WET diameter (all 15 species incl. H2O), the coag paper's sectional exactly:
  sec20w 20 bins log-spaced 0.001-10 um (bins, nodes and node volumes in wet diameter)
Dry diameter: (6/pi sum_{k != H2O} m_k / rho_k)^(1/3), the diagnostics' densities. Particles
below the first edge go to the first bin, above the last edge to the last bin (number and mass
kept; for sec20w none is outside: wet diameters 7 nm-1.05 um).

State. Per-particle vector x: for each bin b, [fraction of particles in b, mass per particle of
the population of each species in b (kg), 15 species in config.ML_SPECIES order]; layout
x[b * 16 + 0] number, x[b * 16 + 1 + k] species k. Extensive Q = N x (# and kg per kg of air);
mixing averages x in number.

Representative particles (coag paper, section "Sectional baseline"; dry diameter except sec20w):
n_rep nodes log-uniform across the bin including both edges, weights p_j = q_j exp(lam v_j) /
sum (trapezoidal prior q, maximum relative entropy) matching the bin's number and mean dry
volume; every node has the bin's mass fractions (all 15 species, H2O included), so number, dry
volume and every species mass of the bin are reproduced. If the bin's mean dry volume lies
outside the node range (possible only in an end bin holding particles from outside the range),
the bin gets one particle at its mean dry volume instead.
"""
import math

import numpy as np

import diag

SPECIES = diag.SPECIES
NS = diag.NS
RHO = diag.RHO
UM = 1e-6
EDGES = {
    'sec8': np.array([0.039, 0.078, 0.156, 0.3125, 0.625, 1.25, 2.5, 5.0, 10.0]) * UM,
    'sec4': np.array([0.039, 0.156, 0.625, 2.5, 10.0]) * UM,
    'sec20': np.logspace(-3, 1, 21) * UM,
    'sec20w': np.logspace(-3, 1, 21) * UM,        # the coag paper's sectional: WET diameter
}
#: Representations whose bins and nodes are in wet diameter (all 15 species incl. H2O).
WET = ('sec20w',)
#: Nodes per bin: the coag paper's 20 (10 changes the sources' errors by <= 1 point except the
#: number distribution of sec4/sec20 (up to 4.5 points); 40 by <= 1.6 points).
N_REP = 20


class Sectional:
    def __init__(self, name='sec8', n_rep=N_REP, edges=None, basis=None):
        """basis 'dry' (MOSAIC-like): bins and nodes in dry diameter; 'wet' (sec20w, the coag
        paper's): in wet diameter over all 15 species incl. H2O. Default from the name."""
        basis = basis or ('wet' if name in WET else 'dry')
        self.name = name
        self.basis = basis
        self.sel = diag.DRY if basis == 'dry' else np.ones(NS, bool)
        self.edges = np.asarray(edges if edges is not None else EDGES[name], float)
        self.nb = len(self.edges) - 1
        self.n_rep = n_rep
        self.d = self.nb * (1 + NS)

    def bin_of(self, m):
        """Bin index of each particle (..., 15) by dry (or wet) diameter; outside -> end bins."""
        dd = diag.dry_diameter(m) if self.basis == 'dry' else diag.wet_diameter(m)
        b = np.searchsorted(self.edges, dd, side='right') - 1
        return np.clip(b, 0, self.nb - 1)

    def project(self, m, w):
        """Particles m (P, n, 15) kg, w (P, n) -> per-particle states x (P, d)."""
        m = np.asarray(m, float)
        w = np.asarray(w, float)
        if m.ndim == 2:
            m, w = m[None], w[None]
        wn = w / w.sum(-1, keepdims=True)
        b = self.bin_of(m)
        x = np.zeros((len(m), self.nb, 1 + NS))
        for k in range(self.nb):
            sel = np.where(b == k, wn, 0.0)
            x[:, k, 0] = sel.sum(-1)
            x[:, k, 1:] = np.einsum('pn,pnk->pk', sel, m)
        return x.reshape(len(m), -1)

    def measure(self, mass):
        """The binned variable of a bin's particles from species masses (..., 15): dry (or wet)
        volume. A hook for subclasses (tomas.Tomas bins in dry mass); particles() uses it."""
        return (mass[..., self.sel] / RHO[self.sel]).sum(-1)

    def node_measure(self, D):
        """The binned variable at the nodes() values D: volume pi/6 D^3 (tomas.Tomas: D are masses)."""
        return math.pi / 6 * D ** 3

    def nodes(self):
        """Node dry diameters (nb, n_rep) and the trapezoidal prior (n_rep,)."""
        j = np.arange(self.n_rep) / (self.n_rep - 1)
        D = self.edges[:-1, None] * (self.edges[1:, None] / self.edges[:-1, None]) ** j[None, :]
        q = np.full(self.n_rep, 1.0 / (self.n_rep - 1))
        q[[0, -1]] *= 0.5
        return D, q

    def particles(self, x):
        """States x (P, d) -> representative particles: masses (P, nb * (n_rep + 1), 15) kg and
        weights (P, nb * (n_rep + 1)) (number fractions); the extra slot per bin is the
        single-particle fallback (weight 0 unless used)."""
        x = np.atleast_2d(np.asarray(x, float)).reshape(-1, self.nb, 1 + NS)
        P = len(x)
        f = np.maximum(x[..., 0], 0.0)                                     # (P, nb)
        mass = np.maximum(x[..., 1:], 0.0)                                 # (P, nb, 15)
        vol = self.measure(mass)                                           # dry volume per particle of pop
        ok = (f > 0) & (vol > 0)
        vbar = np.where(ok, vol / np.where(ok, f, 1.0), 0.0)               # mean dry volume in the bin
        D, q = self.nodes()
        v = self.node_measure(D)                                           # (nb, n_rep)
        v1, vn = v[:, 0], v[:, -1]
        t = (vbar - v1) / (vn - v1)                                        # target in [0, 1] if inside
        s = (v - v1[:, None]) / (vn - v1)[:, None]                         # (nb, n_rep) in [0, 1]
        inside = ok & (t > 1e-12) & (t < 1 - 1e-12)
        lam = _solve_lambda(s, np.log(q), np.where(inside, t, 0.5))       # (P, nb)
        a = np.log(q)[None, None, :] + lam[..., None] * s[None]
        a -= a.max(-1, keepdims=True)
        p = np.exp(a)
        p /= p.sum(-1, keepdims=True)                                      # (P, nb, n_rep)
        vmean = (p * v[None]).sum(-1)
        # nodes: p_j f, masses (v_j / vbar) * (mass per particle in the bin); exact up to
        # vmean / vbar (the solve's tolerance, ~1e-12), absorbed by rescaling the volumes
        per = np.where(ok[..., None], mass / np.where(ok, f, 1.0)[..., None], 0.0)    # (P, nb, 15)
        scale = np.where(inside[..., None], v[None] / np.where(inside, vmean, 1.0)[..., None], 0.0)
        m_nodes = scale[..., None] * per[:, :, None, :]                    # (P, nb, n_rep, 15)
        w_nodes = np.where(inside[..., None], f[..., None] * p, 0.0)
        single = ok & ~inside
        m_one = np.where(single[..., None], per, 0.0)[:, :, None, :]       # (P, nb, 1, 15)
        w_one = np.where(single, f, 0.0)[..., None]
        self.last_single = int(single.sum())
        m_all = np.concatenate([m_nodes, m_one], axis=2).reshape(P, -1, NS)
        w_all = np.concatenate([w_nodes, w_one], axis=2).reshape(P, -1)
        return m_all, w_all

    def layout(self):
        comps = []
        for b in range(self.nb):
            lo, hi = self.edges[b], self.edges[b + 1]
            comps.append({'index': b * (1 + NS), 'bin': b, 'kind': 'number',
                          '%s_diameter_m' % self.basis: [float(lo), float(hi)],
                          'units_per_particle': '1 (fraction of particles in the bin)',
                          'units_extensive': '# kg-1 (Q = N x)'})
            for k, s in enumerate(SPECIES):
                comps.append({'index': b * (1 + NS) + 1 + k, 'bin': b, 'kind': 'mass', 'species': s,
                              '%s_diameter_m' % self.basis: [float(lo), float(hi)],
                              'units_per_particle': 'kg (mass per particle of the population)',
                              'units_extensive': 'kg kg-1 (Q = N x)'})
        return {'model': self.name, 'kind': 'sectional', 'n_tracers': self.d, 'basis': self.basis,
                'diameter': 'wet (all 15 species incl. H2O)' if self.basis == 'wet' else 'dry (H2O excluded)',
                '%s_diameter_edges_m' % self.basis: [float(e) for e in self.edges],
                'outside_range': 'particles below the first / above the last edge go to the end bins',
                'n_rep': self.n_rep, 'components': comps, 'calcsize': None}


def _solve_lambda(s, logq, t, iters=200):
    """lam (P, nb) with sum_j p_j(lam) s_j = t, p_j ~ q_j exp(lam s_j); s (nb, n) in [0, 1].
    Bisection in lam on [-L, L] (monotone), L large enough for t within 1e-12 of the ends."""
    lo = np.full(t.shape, -1.0e5)
    hi = np.full(t.shape, 1.0e5)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        a = logq[None, None, :] + mid[..., None] * s[None]
        a -= a.max(-1, keepdims=True)
        p = np.exp(a)
        mean = (p * s[None]).sum(-1) / p.sum(-1)
        big = mean > t
        hi = np.where(big, mid, hi)
        lo = np.where(big, lo, mid)
        if np.all(hi - lo < 1e-13 * np.maximum(1.0, np.abs(mid))):
            break
    return 0.5 * (lo + hi)

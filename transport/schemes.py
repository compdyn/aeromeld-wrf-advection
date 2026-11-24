"""Latent-transport schemes for AeroMELD, on WRF's own advection (wrfadv.py).

The aerosol state of a cell is its number mixing ratio N and its latent vector
z, the number-weighted mean of the particles' encodings, so the conserved
quantities are N and Q = N z (per kg of dry air). Mixing two parcels is exact
when Q and N are moved by the same operator. The schemes differ only in the
final RK3 stage, where WRF limits the fluxes (chem_adv_opt = 2):

  oracle  each WRF-PartMC particle class (= source) is its own tracer with its
          own monotonic limiter: what WRF-PartMC does, in expectation. Needs
          source tags, so it is the reference, not an AeroMELD option.
  donor   N with WRF's monotonic scheme; z moved donor-cell by N's face fluxes
          (WRF-AeroMELD's design). Exact mixing, first-order in z.
  donorW  donor as coded in WRF-AeroMELD (latent_advection 838cfd48), whose
          vertical inflows are weighted by the donor layer's 1/d(eta) instead
          of the receiver's (on the stretched CARES grid the surface layer gets
          70% of its inflow from above).
  sep     N and every Q_m with WRF's monotonic scheme, each with its own
          limiter. High order, but the latent leaves the mixing plane.
  shN     Q with N's limiter coefficients (one per face, shared). Exactly
          affine (z stays on the mixing plane), but z can overshoot.
  shB     one coefficient per face shared by N and all Q_m, the largest that
          keeps N within WRF's bounds and every z_m within the range of the
          cell and its upwind neighbours (synchronized FCT). Exactly affine
          and bounded.

For the shared schemes (donor, shN, shB) one step is the same linear operator
on every tracer, so carrying the three source groups' numbers n_g instead of
(N, Q) gives the same z = sum_g (n_g / N) z*_g exactly; the runs carry n_g,
which gives the source fractions directly. sep has to carry Q.

A tracer set is (initial, boundary, emission) maps from the inputs:
  IC sources (background, urban), the BC source (background), and the
  emission sources (two, both the Emissions population).
"""
import numpy as np

from wrfadv import (add, apply_coef, cell_contributions, hi_fluxes, lo_fluxes, mono_coef, sub, sync_coef,
                    tendency, update, upwind_bounds)  # noqa: F401

#: Group of each input: IC background, IC urban, BC background, emission 1, emission 2.
INPUT_GROUP = (0, 1, 0, 2, 2)
N_INPUTS = 5
#: WRF-PartMC's particle classes, one per input (IC_FROM_FILE_01/02, BC_FROM_FILE_01,
#: EMIS_FROM_FILE_01/02).
ORACLE_NAMES = ('ic_background', 'ic_urban', 'bc_background', 'emissions_1', 'emissions_2')


def group_map():
    """(3 groups x 5 inputs) 0/1 map."""
    m = np.zeros((3, N_INPUTS))
    for i, gidx in enumerate(INPUT_GROUP):
        m[gidx, i] = 1.0
    return m


def stage_masses(g, M0, F_air, dt):
    T = tendency(g, *F_air)
    return [M0 + d * g.msfty * T for d in (dt / 3., dt / 2., dt)]


class Runner:
    """Advances all schemes together on one stream of WRF mass fluxes."""

    def __init__(self, g, Z, xp, inputs0, bc_edges, schemes=('oracle', 'donor', 'sep', 'shN', 'shB'),
                 sep_models=None, shB_models=None):
        """Z: {model: (3, d) source latents}; inputs0 (5, nz, ny, nx) initial
        number mixing ratio of each input (0 for BC and emissions); bc_edges:
        dict w/e (nz, ny), s/n (nz, nx) of the BC source's number mixing ratio.
        sep and shB run for sep_models / shB_models (default: every model in Z);
        the model 'frac' (Z = identity) bounds the source fractions themselves."""
        self.g, self.xp, self.Z = g, xp, {k: np.asarray(v, float) for k, v in Z.items()}
        self.schemes = schemes
        sep_models = list(self.Z) if sep_models is None else sep_models
        shB_models = list(self.Z) if shB_models is None else shB_models
        G = group_map()
        sets = {}
        # Bundle of independently limited tracers: N, the oracle classes, sep's Q per model.
        W = [np.ones(N_INPUTS)]                               # N
        self.idx = {'N': 0}
        if 'oracle' in schemes:
            self.idx['oracle'] = slice(len(W), len(W) + N_INPUTS)
            W += list(np.eye(N_INPUTS))
        if 'sep' in schemes:
            for mk in sep_models:
                z = self.Z[mk]
                self.idx['sep:' + mk] = slice(len(W), len(W) + z.shape[1])
                W += list(z.T @ G)                            # Q_m = sum_g z*_gm n_g
        self.mono = np.asarray(W)
        self.shared = {}
        for s in ('donor', 'donorW', 'shN'):
            if s in schemes:
                self.shared[s] = G.copy()
        if 'shB' in schemes:
            for mk in shB_models:
                self.shared['shB:' + mk] = G.copy()
        to = lambda a: xp.asarray(a)
        ins = to(inputs0)
        self.q_mono = xp.tensordot(to(self.mono), ins, axes=1)
        self.q_sh = {k: xp.tensordot(to(W_), ins, axes=1) for k, W_ in self.shared.items()}
        bce = {k: to(v) for k, v in bc_edges.items()}
        # boundary value of each tracer: (its weight on the BC input) x BC number
        self.bc_w = {'mono': to(self.mono[:, 2]), **{k: to(W_[:, 2]) for k, W_ in self.shared.items()}}
        self.bce = bce
        self.em_w = {'mono': to(self.mono[:, 3:5]), **{k: to(W_[:, 3:5]) for k, W_ in self.shared.items()}}

    # --- boundary conditions (flow_dep_bdy_chem: inflow -> BC, outflow -> interior)
    def bc(self, q, key, F):
        xp = self.xp
        ru, rv = F[0], F[1]
        w = self.bc_w[key][:, None, None]                     # (T, 1, 1)
        ny, nx = self.g.ny, self.g.nx
        # south / north rows (corners included), then west / east columns
        inner = q[..., 1, :].copy()
        inner[..., 0], inner[..., -1] = inner[..., 1], inner[..., -2]
        q[..., 0, :] = xp.where(rv[:, 0, :] < 0, inner, w * self.bce['s'])
        inner = q[..., ny - 2, :].copy()
        inner[..., 0], inner[..., -1] = inner[..., 1], inner[..., -2]
        q[..., ny - 1, :] = xp.where(rv[:, ny, :] > 0, inner, w * self.bce['n'])
        inner = q[..., :, 1].copy()
        inner[..., 0], inner[..., -1] = inner[..., 1], inner[..., -2]
        q[..., :, 0] = xp.where(ru[:, :, 0] < 0, inner, w * self.bce['w'])
        inner = q[..., :, nx - 2].copy()
        inner[..., 0], inner[..., -1] = inner[..., 1], inner[..., -2]
        q[..., :, nx - 1] = xp.where(ru[:, :, nx] > 0, inner, w * self.bce['e'])

    def _stages12(self, q0, key, F, M, dt):
        g = self.g
        q1 = update(g, q0, M[0], M[1], dt / 3., hi_fluxes(g, q0, *F))
        self.bc(q1, key, F)
        q2 = update(g, q0, M[0], M[2], dt / 2., hi_fluxes(g, q1, *F))
        self.bc(q2, key, F)
        return q2

    def step(self, F, M0, dt):
        """One 20 s step with face mass fluxes F = (ru, rv, ww); returns M_new."""
        g, xp = self.g, self.xp
        M = [M0] + stage_masses(g, M0, F, dt)              # M0, M1, M2, M3

        # independent monotonic limiters (WRF's chem_adv_opt = 2)
        q0 = self.q_mono
        q2 = self._stages12(q0, 'mono', F, M, dt)
        FL = lo_fluxes(g, q0, *F)
        A = sub(hi_fluxes(g, q2, *F), FL)
        coef = mono_coef(g, q0, M[0], M[3], dt, FL, A, *F)
        F_tot = add(FL, apply_coef(A, coef))
        self.q_mono = update(g, q0, M[0], M[3], dt, F_tot)
        self.bc(self.q_mono, 'mono', F)
        coef_N = tuple(c[0] for c in coef)
        FN = tuple(f[0] for f in F_tot)                         # N's total face fluxes

        for key, q0 in list(self.q_sh.items()):
            if key == 'donor':
                Fg = self._donor_fluxes(q0, FN)
                q3 = update(g, q0, M[0], M[3], dt, Fg)
            elif key == 'donorW':
                q3 = self._donor_wrf(q0, FN, M[3], self.q_mono[0], dt)
            else:
                q2 = self._stages12(q0, key, F, M, dt)
                FL = lo_fluxes(g, q0, *F)
                A = sub(hi_fluxes(g, q2, *F), FL)
                if key == 'shN':
                    c = coef_N
                else:
                    c = self._sync_coef(q0, M, dt, FL, A, F, self.Z[key.split(':', 1)[1]])
                q3 = update(g, q0, M[0], M[3], dt, add(FL, apply_coef(A, tuple(cc[None] for cc in c))))
            self.bc(q3, key, F)
            self.q_sh[key] = q3
        return M[3]

    def _donor_fluxes(self, q0, FN):
        """Per-group face fluxes: N's total face flux times the donor cell's
        group fraction (WRF-AeroMELD's latent update)."""
        xp = self.xp
        N0 = q0.sum(0)
        f = q0 / xp.where(N0 > 0, N0, 1.0)
        Fx, Fy, Fz = (xp.zeros((q0.shape[0],) + a.shape) for a in FN)
        Fx[..., 1:-1] = FN[0][None, ..., 1:-1] * xp.where(FN[0][None, ..., 1:-1] > 0, f[..., :-1], f[..., 1:])
        Fy[..., 1:-1, :] = FN[1][None, ..., 1:-1, :] * xp.where(FN[1][None, ..., 1:-1, :] > 0, f[..., :-1, :], f[..., 1:, :])
        # vertical: positive flux is downward (from the cell above, k, into k-1)
        Fz[..., 1:-1, :, :] = FN[2][None, ..., 1:-1, :, :] * xp.where(FN[2][None, ..., 1:-1, :, :] > 0,
                                                                      f[..., 1:, :, :], f[..., :-1, :, :])
        return Fx, Fy, Fz

    def _donor_wrf(self, q0, FN, M3, N3, dt):
        """donorW: WRF-AeroMELD's latent update exactly as coded (solve_em.F,
        latent_advection 838cfd48), in its incremental form
            f_new = f_old + dt msfty sum_inflow w (f_nb - f_old) / (M_new N_new),
        which is donor-cell except that a vertical inflow is weighted by the
        donor layer's 1/d(eta) (flux_out_top/bottom are normalized by the
        originating cell's rdzw) instead of the receiving layer's."""
        g, xp = self.g, self.xp
        nz, ny, nx = g.nz, g.ny, g.nx
        N0 = q0.sum(0)
        f = q0 / xp.where(N0 > 0, N0, 1.0)
        d = xp.zeros(q0.shape)
        wx = dt * g.msfty * g.msftx * g.rdx
        wy = dt * g.msfty * g.msftx * g.rdy
        Fx = FN[0][..., 1:nx]
        d[..., 1:nx] += xp.where(Fx > 0, wx[:, 1:nx] * Fx, 0) * (f[..., 0:nx - 1] - f[..., 1:nx])
        d[..., 0:nx - 1] += xp.where(Fx < 0, -wx[:, 0:nx - 1] * Fx, 0) * (f[..., 1:nx] - f[..., 0:nx - 1])
        Fy = FN[1][..., 1:ny, :]
        d[..., 1:ny, :] += xp.where(Fy > 0, wy[1:ny, :] * Fy, 0) * (f[..., 0:ny - 1, :] - f[..., 1:ny, :])
        d[..., 0:ny - 1, :] += xp.where(Fy < 0, -wy[0:ny - 1, :] * Fy, 0) * (f[..., 1:ny, :] - f[..., 0:ny - 1, :])
        Fz = FN[2][1:nz]                                         # faces 1..nz-1
        ar = -g.rdnw                                             # |1/d(eta)| (nz, 1, 1)
        wz_lo = dt * g.msfty * ar[1:nz]                          # weight on the face's upper cell's rdzw ...
        wz_hi = dt * g.msfty * ar[0:nz - 1]                      # ... and its lower cell's
        # downward flux (Fz > 0): inflow to the lower cell k-1 from k, weighted by rdzw(k) (donor)
        d[..., 0:nz - 1, :, :] += xp.where(Fz > 0, wz_lo * Fz, 0) * (f[..., 1:nz, :, :] - f[..., 0:nz - 1, :, :])
        # upward flux (Fz < 0): inflow to the upper cell k from k-1, weighted by rdzw(k-1) (donor)
        d[..., 1:nz, :, :] += xp.where(Fz < 0, -wz_hi * Fz, 0) * (f[..., 0:nz - 1, :, :] - f[..., 1:nz, :, :])
        f3 = f + d / (M3 * xp.where(N3 > 0, N3, 1.0))
        return xp.where(g.interior, f3 * N3, q0)

    def _sync_coef(self, q0, M, dt, FL, A, F, Z):
        """shB: one coefficient per face that keeps N within WRF's bounds and
        each z_m within the range of the cell and its upwind neighbours."""
        g, xp = self.g, self.xp
        Zx = xp.asarray(Z)                                      # (3, d)
        N0 = q0.sum(0)
        Q0 = xp.tensordot(Zx.T, q0, axes=1)                     # (d, ...)
        z0 = Q0 / xp.where(N0 > 0, N0, 1.0)
        Nmin, Nmax = upwind_bounds(g, N0, *F)
        zmin, zmax = upwind_bounds(g, z0, *F)
        ph = M[0] * q0 + dt * g.msfty * tendency(g, *FL)        # (3, ...) low-order M_new q
        phN = ph.sum(0)
        phQ = xp.tensordot(Zx.T, ph, axes=1)
        G0 = xp.concatenate([(Nmax * M[3] - phN)[None], (phN - Nmin * M[3])[None],
                             zmax * phN - phQ, phQ - zmin * phN])  # (2 + 2d, ...)
        contrib = []
        nz, ny, nx = g.nz, g.ny, g.nx
        sides = ((slice(0, nx - 1), slice(1, nx)), (slice(0, ny - 1), slice(1, ny)), (slice(0, nz - 1), slice(1, nz)))
        for ax, (lo_, hi_), (s_lo, s_hi) in zip((-1, -2, -3), cell_contributions(g, dt, *A), sides):
            pair = []
            for c, s in ((lo_, s_lo), (hi_, s_hi)):
                cN = c.sum(0)
                cQ = xp.tensordot(Zx.T, c, axes=1)
                sl = [slice(None)] * 4
                sl[ax] = s
                sl = tuple(sl)
                zx, zn = zmax[sl], zmin[sl]
                pair.append(xp.concatenate([-cN[None], cN[None], zx * cN - cQ, cQ - zn * cN]))
            contrib.append(tuple(pair))
        return sync_coef(g, G0, contrib)

    def emit(self, dn):
        """Add each emission input's increment dn (2, ny, nx) at the lowest level."""
        xp = self.xp
        self.q_mono[:, 0] += xp.tensordot(self.em_w['mono'], dn, axes=1)
        for key in self.q_sh:
            self.q_sh[key][:, 0] += xp.tensordot(self.em_w[key], dn, axes=1)

    def surface(self, level=0):
        """{name: (T, ny, nx)} at one level, as numpy."""
        xp = self.xp
        get = (lambda a: a.get()) if xp is not np else (lambda a: a)
        out = {'N': get(self.q_mono[0, level])}
        if 'oracle' in self.idx:
            out['oracle'] = get(self.q_mono[self.idx['oracle'], level])
        for k, s in self.idx.items():
            if k.startswith('sep:'):
                out[k] = get(self.q_mono[s, level])
        for k, q in self.q_sh.items():
            out[k] = get(q[:, level])
        return out

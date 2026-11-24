"""WRF's scalar advection, replayed offline (dyn_em/module_advect_em.F, WRF 4.7).

Arrays are (..., nz, ny, nx) on mass points, leading axes for several tracers
at once; faces are staggered: x faces (..., nz, ny, nx+1), y faces
(..., nz, ny+1, nx), vertical faces (..., nz+1, ny, nx). Face f lies between
cells f-1 and f (0-based), as in WRF's fqx(i) between i-1 and i.

The mass fluxes are WRF's mu-coupled velocities ru, rv, ww (Pa m/s, Pa/s);
cell "mass" is M = c1h mu + c2h (Pa per unit eta). One update over dt with face
fluxes F (tracer flux = mass flux x face value):

    M_new q_new = M_old q_old + dt msfty T(F),
    T(F) = -msftx (rdx dx F_x + rdy dy F_y) - rdnw dz F_z

(rk_update_scalar with advect_scalar's tendency). Air mass evolves with the same
operator on the mass fluxes themselves, so a uniform q stays uniform exactly.

Orders, as configured for the CARES runs (h_sca_adv_order = 5,
v_sca_adv_order = 3, specified lateral boundaries): 5th-order horizontal fluxes,
degraded to 3rd and 2nd order at the two faces next to each lateral boundary;
3rd-order vertical, 2nd at the faces next to the bottom and top; zero flux
through the domain's outer faces (the boundary ring is reset from the boundary
conditions after every stage, flow_dep_bdy_chem).

`mono_coef` is WRF's monotonic flux limiter (advect_scalar_mono, used in the
final RK3 stage when chem_adv_opt = 2): Zalesak's FCT with the low-order
(upwind) solution, bounds from each cell and its upwind neighbours, and one
coefficient per face. `sync_coef` is the same limiter with several linear
constraints per cell sharing one coefficient per face (the shared limiters of
transport/schemes.py).
"""
import numpy as np

EPS = 1e-20


class Grid:
    """Static grid quantities, all as arrays of the chosen array module xp."""

    def __init__(self, xp, msftx, msfty, rdx, rdy, rdnw, fnm, fnp, c1h, c2h):
        self.xp = xp
        f = lambda a: xp.asarray(a, dtype=xp.float64)
        self.msftx, self.msfty = f(msftx), f(msfty)              # (ny, nx)
        self.rdx, self.rdy = float(rdx), float(rdy)
        self.rdnw = f(rdnw)[:, None, None]                         # (nz, 1, 1), negative
        self.fnm, self.fnp = f(fnm), f(fnp)                        # (nz,)
        self.c1h, self.c2h = f(c1h)[:, None, None], f(c2h)[:, None, None]
        self.nz = self.rdnw.shape[0]
        self.ny, self.nx = self.msftx.shape
        # interior: everything but the specified-boundary ring (spec_zone = 1)
        m = np.zeros((self.ny, self.nx), bool)
        m[1:-1, 1:-1] = True
        self.interior = xp.asarray(m)

    def mass(self, mu):
        """M = c1h mu + c2h for the full dry column mass mu (ny, nx)."""
        return self.c1h * mu[None] + self.c2h


def _flux5(qm3, qm2, qm1, q0, qp1, qp2, s):
    return ((37. / 60.) * (q0 + qm1) - (2. / 15.) * (qp1 + qm2) + (1. / 60.) * (qp2 + qm3)
            - s * (1. / 60.) * ((qp2 - qm3) - 5. * (qp1 - qm2) + 10. * (q0 - qm1)))


def _flux3(qm2, qm1, q0, qp1, s):
    return ((7. / 12.) * (q0 + qm1) - (1. / 12.) * (qp1 + qm2)
            + s * (1. / 12.) * ((qp1 - qm2) - 3. * (q0 - qm1)))


def _sign(xp, v):
    """Fortran SIGN(1., v): +1 for v >= 0."""
    return xp.where(v >= 0, 1.0, -1.0)


def hi_fluxes(g, q, ru, rv, ww):
    """High-order face fluxes (advect_scalar / the high-order part of
    advect_scalar_mono) of q (..., nz, ny, nx)."""
    xp = g.xp
    lead = q.shape[:-3]
    nz, ny, nx = g.nz, g.ny, g.nx
    Fx = xp.zeros(lead + (nz, ny, nx + 1))
    Fy = xp.zeros(lead + (nz, ny + 1, nx))
    Fz = xp.zeros(lead + (nz + 1, ny, nx))

    # x: 5th order on faces 3..nx-3, 3rd on 2 and nx-2, 2nd on 1 and nx-1
    u = ru[..., 3:nx - 2]
    Fx[..., 3:nx - 2] = u * _flux5(q[..., 0:nx - 5], q[..., 1:nx - 4], q[..., 2:nx - 3],
                                   q[..., 3:nx - 2], q[..., 4:nx - 1], q[..., 5:nx], _sign(xp, u))
    for f in (2, nx - 2):
        u = ru[..., f]
        Fx[..., f] = u * _flux3(q[..., f - 2], q[..., f - 1], q[..., f], q[..., f + 1], _sign(xp, u))
    for f in (1, nx - 1):
        Fx[..., f] = 0.5 * ru[..., f] * (q[..., f - 1] + q[..., f])

    # y
    v = rv[..., 3:ny - 2, :]
    Fy[..., 3:ny - 2, :] = v * _flux5(q[..., 0:ny - 5, :], q[..., 1:ny - 4, :], q[..., 2:ny - 3, :],
                                      q[..., 3:ny - 2, :], q[..., 4:ny - 1, :], q[..., 5:ny, :], _sign(xp, v))
    for f in (2, ny - 2):
        v = rv[..., f, :]
        Fy[..., f, :] = v * _flux3(q[..., f - 2, :], q[..., f - 1, :], q[..., f, :], q[..., f + 1, :], _sign(xp, v))
    for f in (1, ny - 1):
        Fy[..., f, :] = 0.5 * rv[..., f, :] * (q[..., f - 1, :] + q[..., f, :])

    # vertical: 3rd order on faces 2..nz-2 (velocity -ww along k), 2nd on 1 and nz-1
    w = ww[..., 2:nz - 1, :, :]
    Fz[..., 2:nz - 1, :, :] = w * _flux3(q[..., 0:nz - 3, :, :], q[..., 1:nz - 2, :, :], q[..., 2:nz - 1, :, :],
                                         q[..., 3:nz, :, :], _sign(xp, -w))
    for k in (1, nz - 1):
        Fz[..., k, :, :] = ww[..., k, :, :] * (g.fnm[k] * q[..., k, :, :] + g.fnp[k] * q[..., k - 1, :, :])
    return Fx, Fy, Fz


def lo_fluxes(g, q, ru, rv, ww):
    """First-order upwind face fluxes (flux_upwind) on the faces WRF computes."""
    xp = g.xp
    lead = q.shape[:-3]
    nz, ny, nx = g.nz, g.ny, g.nx
    Fx = xp.zeros(lead + (nz, ny, nx + 1))
    Fy = xp.zeros(lead + (nz, ny + 1, nx))
    Fz = xp.zeros(lead + (nz + 1, ny, nx))
    u = ru[..., 1:nx]
    Fx[..., 1:nx] = u * xp.where(u >= 0, q[..., 0:nx - 1], q[..., 1:nx])
    v = rv[..., 1:ny, :]
    Fy[..., 1:ny, :] = v * xp.where(v >= 0, q[..., 0:ny - 1, :], q[..., 1:ny, :])
    w = ww[..., 1:nz, :, :]
    Fz[..., 1:nz, :, :] = w * xp.where(-w >= 0, q[..., 0:nz - 1, :, :], q[..., 1:nz, :, :])
    return Fx, Fy, Fz


def tendency(g, Fx, Fy, Fz):
    """T(F) = -msftx (rdx dx F_x + rdy dy F_y) - rdnw dz F_z, on mass points."""
    return (-g.msftx * (g.rdx * (Fx[..., 1:] - Fx[..., :-1]) + g.rdy * (Fy[..., 1:, :] - Fy[..., :-1, :]))
            - g.rdnw * (Fz[..., 1:, :, :] - Fz[..., :-1, :, :]))


def air_fluxes(ru, rv, ww):
    """The mass fluxes themselves: the tracer fluxes of q = 1 (any order)."""
    return ru, rv, ww


def update(g, q_old, M_old, M_new, dt, F):
    """(M_old q_old + dt msfty T(F)) / M_new in the interior; the boundary ring
    keeps q_old (it is reset by the boundary conditions afterwards)."""
    xp = g.xp
    q_new = (M_old * q_old + dt * g.msfty * tendency(g, *F)) / M_new
    return xp.where(g.interior, q_new, q_old)


def upwind_bounds(g, q_old, ru, rv, ww):
    """(qmin, qmax): each cell and its upwind neighbours (advect_scalar_mono),
    over the faces WRF computes."""
    xp = g.xp
    nz, ny, nx = g.nz, g.ny, g.nx
    lo = q_old.copy()
    hi = q_old.copy()
    neg, pos = -xp.inf, xp.inf

    def upd(sl_dst, cand_lo, cand_hi):
        lo[sl_dst] = xp.minimum(lo[sl_dst], cand_lo)
        hi[sl_dst] = xp.maximum(hi[sl_dst], cand_hi)

    e = (Ellipsis,)
    # x faces 1..nx-1: flow +x -> cell f gets q[f-1]; otherwise cell f-1 gets q[f]
    p = ru[..., 1:nx] > 0
    upd(e + (slice(1, nx),), xp.where(p, q_old[..., 0:nx - 1], pos), xp.where(p, q_old[..., 0:nx - 1], neg))
    upd(e + (slice(0, nx - 1),), xp.where(~p, q_old[..., 1:nx], pos), xp.where(~p, q_old[..., 1:nx], neg))
    p = rv[..., 1:ny, :] > 0
    upd(e + (slice(1, ny), slice(None)), xp.where(p, q_old[..., 0:ny - 1, :], pos), xp.where(p, q_old[..., 0:ny - 1, :], neg))
    upd(e + (slice(0, ny - 1), slice(None)), xp.where(~p, q_old[..., 1:ny, :], pos), xp.where(~p, q_old[..., 1:ny, :], neg))
    p = -ww[..., 1:nz, :, :] > 0                               # cr = -vel > 0: from below
    upd(e + (slice(1, nz), slice(None), slice(None)), xp.where(p, q_old[..., 0:nz - 1, :, :], pos),
        xp.where(p, q_old[..., 0:nz - 1, :, :], neg))
    upd(e + (slice(0, nz - 1), slice(None), slice(None)), xp.where(~p, q_old[..., 1:nz, :, :], pos),
        xp.where(~p, q_old[..., 1:nz, :, :], neg))
    return lo, hi


def cell_contributions(g, dt, Ax, Ay, Az):
    """Each face's antidiffusive flux as seen by its two cells, in units of
    M_new q_new (dt msfty T): returns, per direction, (to the cell before the
    face, to the cell after it), on the faces 1..n-1."""
    nz, ny, nx = g.nz, g.ny, g.nx
    wx = dt * g.msfty * g.msftx * g.rdx                    # (ny, nx)
    wy = dt * g.msfty * g.msftx * g.rdy
    wz = dt * g.msfty * g.rdnw                             # (nz, ny, nx)
    ax, ay, az = Ax[..., 1:nx], Ay[..., 1:ny, :], Az[..., 1:nz, :, :]
    x = (-wx[:, 0:nx - 1] * ax, wx[:, 1:nx] * ax)
    y = (-wy[0:ny - 1, :] * ay, wy[1:ny, :] * ay)
    z = (-wz[0:nz - 1] * az, wz[1:nz] * az)
    return x, y, z


def mono_coef(g, q_old, M_old, M_new, dt, FL, A, ru, rv, ww):
    """WRF's monotonic limiter: per-face coefficients (cx, cy, cz) in [0, 1] on
    the faces 1..n-1, each tracer limited on its own (q_old (..., nz, ny, nx))."""
    xp = g.xp
    nz, ny, nx = g.nz, g.ny, g.nx
    qmin, qmax = upwind_bounds(g, q_old, ru, rv, ww)
    ph_up = M_old * q_old + dt * g.msfty * tendency(g, *FL)
    (xl, xr), (yl, yr), (zl, zr) = cell_contributions(g, dt, *A)
    zero = lambda: xp.zeros(q_old.shape)
    inflow, outflow = zero(), zero()
    for (lo_, hi_), sl_lo, sl_hi in (((xl, xr), (Ellipsis, slice(0, nx - 1)), (Ellipsis, slice(1, nx))),
                                     ((yl, yr), (Ellipsis, slice(0, ny - 1), slice(None)), (Ellipsis, slice(1, ny), slice(None))),
                                     ((zl, zr), (Ellipsis, slice(0, nz - 1), slice(None), slice(None)),
                                      (Ellipsis, slice(1, nz), slice(None), slice(None)))):
        inflow[sl_lo] += xp.maximum(lo_, 0)
        outflow[sl_lo] -= xp.minimum(lo_, 0)
        inflow[sl_hi] += xp.maximum(hi_, 0)
        outflow[sl_hi] -= xp.minimum(hi_, 0)
    ph_hi = M_new * qmax - ph_up
    ph_low = ph_up - M_new * qmin
    s_in = xp.where(inflow > ph_hi, xp.maximum(0., ph_hi / (inflow + EPS)), 1.)
    s_out = xp.where(outflow > ph_low, xp.maximum(0., ph_low / (outflow + EPS)), 1.)
    s_in = xp.where(g.interior, s_in, 1.)
    s_out = xp.where(g.interior, s_out, 1.)
    # A face's flux leaves one cell and enters the other.
    cx = xp.where(xl < 0, xp.minimum(s_out[..., 0:nx - 1], s_in[..., 1:nx]),
                  xp.minimum(s_in[..., 0:nx - 1], s_out[..., 1:nx]))
    cy = xp.where(yl < 0, xp.minimum(s_out[..., 0:ny - 1, :], s_in[..., 1:ny, :]),
                  xp.minimum(s_in[..., 0:ny - 1, :], s_out[..., 1:ny, :]))
    cz = xp.where(zl < 0, xp.minimum(s_out[..., 0:nz - 1, :, :], s_in[..., 1:nz, :, :]),
                  xp.minimum(s_in[..., 0:nz - 1, :, :], s_out[..., 1:nz, :, :]))
    # WRF leaves the x (y) fluxes of the boundary rows (columns) unlimited
    cx = xp.where(g.interior[:, 1:nx] | g.interior[:, 0:nx - 1], cx, 1.)
    cy = xp.where(g.interior[1:ny, :] | g.interior[0:ny - 1, :], cy, 1.)
    cz = xp.where(g.interior, cz, 1.)
    return cx, cy, cz


def sync_coef(g, G0, contrib):
    """One coefficient per face for several linear constraints per cell
    (Zalesak's limiter, synchronized). G0 (C, nz, ny, nx): each constraint's
    value after the low-order update (>= 0); contrib: per direction, (to the
    cell before, to the cell after) of shape (C, faces...), each constraint's
    change per unit coefficient. Returns (cx, cy, cz) on faces 1..n-1 such that
    every constraint stays >= 0."""
    xp = g.xp
    nz, ny, nx = g.nz, g.ny, g.nx
    sls = ((Ellipsis, slice(0, nx - 1)), (Ellipsis, slice(1, nx))), \
          ((Ellipsis, slice(0, ny - 1), slice(None)), (Ellipsis, slice(1, ny), slice(None))), \
          ((Ellipsis, slice(0, nz - 1), slice(None), slice(None)), (Ellipsis, slice(1, nz), slice(None), slice(None)))
    neg = xp.zeros(G0.shape)
    for (lo_, hi_), (s_lo, s_hi) in zip(contrib, sls):
        neg[s_lo] -= xp.minimum(lo_, 0)
        neg[s_hi] -= xp.minimum(hi_, 0)
    R = xp.where(neg > 0, xp.clip(xp.maximum(G0, 0) / (neg + EPS), 0, 1), 1.)
    R = xp.where(g.interior, R, 1.)                          # (C, nz, ny, nx)
    out = []
    for (lo_, hi_), (s_lo, s_hi) in zip(contrib, sls):
        r_lo = xp.where(lo_ < 0, R[s_lo], 1.).min(axis=0)
        r_hi = xp.where(hi_ < 0, R[s_hi], 1.).min(axis=0)
        out.append(xp.minimum(r_lo, r_hi))
    return tuple(out)


def apply_coef(A, coef):
    """Limited antidiffusive fluxes: A on faces 1..n-1 times the coefficients."""
    (Ax, Ay, Az), (cx, cy, cz) = A, coef
    Ax, Ay, Az = Ax.copy(), Ay.copy(), Az.copy()
    nx, ny, nz = Ax.shape[-1] - 1, Ay.shape[-2] - 1, Az.shape[-3] - 1
    Ax[..., 1:nx] *= cx
    Ay[..., 1:ny, :] *= cy
    Az[..., 1:nz, :, :] *= cz
    return Ax, Ay, Az


def add(F, G):
    return tuple(a + b for a, b in zip(F, G))


def sub(F, G):
    return tuple(a - b for a, b in zip(F, G))

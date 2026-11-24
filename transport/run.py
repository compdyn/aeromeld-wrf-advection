#!/usr/bin/env python
"""Replay the CARES case's aerosol transport offline with every scheme (schemes.py).

Inputs (Perlmutter, $SCRATCH/aeromeld/transport):
  wrf_fluxes/wrfout_d01_*  a rerun of the WRF-AeroMELD meteorology (identical MU
      to the paper's runs) writing WRF's time-averaged mu-coupled fluxes
      AVGFLX_RUM/RVM/WWM every minute (do_avgflx_em = 1; the averages of the
      ru_m, rv_m, ww_m that scalar advection uses, over the 3 steps of each
      minute), with MU, MUB and the grid constants;
  the original run's wrfout, for ALT and the lowest layer depth (emissions);
  partmc_inputs/{ic,bc,emis}_control.nc  WRF-PartMC's per-source initial
      number mixing ratio, boundary number and emission number flux (identical
      to WRF-AeroMELD's ntot initial and boundary values, checked);
  the source latents of each model (zsrc_s<seed>.npz).

Every scheme sees the same 720 steps of 20 s, with the fluxes of each minute;
air mass is evolved with the same fluxes, so a uniform mixing ratio stays
uniform. Emissions are added after each step at the lowest level, as number
flux x alt dt / dz (WRF-Chem's conv_rho_aer; PartMC's flux dt / height).

Writes <out>/surface.npz: the lowest level every 15 minutes, per tracer set
(N, the oracle's 5 classes, sep:<model> Q, donor / shN / shB:<model> groups),
and the run's settings.

With --levels K1,K2,...: also level_<k>.npz, the same records at those model
levels (0-based).

Usage: python run.py --out DIR [--gpu] [--models twin_k16,ax_rel1] [--steps 720] [--levels 4,7,11]
"""
import argparse
import json
import time
from pathlib import Path

import netCDF4
import numpy as np

import schemes
from wrfadv import Grid

SCR = Path('/pscratch/sd/m/mwest/aeromeld')
FLUXES = SCR / 'transport/wrf_fluxes/wrfout_d01_2010-06-17_00:00:00'
ORIG = SCR / 'wrfml/work/runs/latent_advection_cares_v3match_twin_k16_of.0.0_5e7/wrfout_d01_2010-06-17_00:00:00'
PMC = SCR / 'transport/partmc_inputs'
ZSRC = SCR / 'transport/zsrc_s0.npz'
G_ACC = 9.81
DT = 20.0
STEPS_PER_FRAME = 3            # 1-minute flux averages, 20 s steps
OUT_EVERY = 45                 # steps (15 minutes)


def times_of(ds):
    return [b''.join(t).decode() for t in ds['Times'][:]]


def load_static(fx, xp):
    v = lambda name: np.asarray(fx[name][0], dtype=np.float64)
    g = Grid(xp, v('MAPFAC_MX'), v('MAPFAC_MY'), 1.0 / fx.DX, 1.0 / fx.DY, v('RDNW'), v('FNM'), v('FNP'),
             v('C1H'), v('C2H'))
    return g


def emission_factor(orig):
    """alt dt / dz at the lowest level, on the original run's output times: (times (s), (nt, ny, nx))."""
    t = np.arange(len(times_of(orig))) * 900.0
    # WRF's equation of state (use_theta_m = 0): alt = (r_d / p0) (T + t0) (1 + rv/rd qv) ((P + PB) / p0)^cvpm
    r_d, r_v, p0 = 287.0, 461.6, 1e5
    cp = 3.5 * r_d
    cvpm = -(cp - r_d) / cp
    th = np.asarray(orig['T'][:, 0], np.float64) + 300.0
    qv = np.asarray(orig['QVAPOR'][:, 0], np.float64)
    p = np.asarray(orig['P'][:, 0] + orig['PB'][:, 0], np.float64)
    alt = (r_d / p0) * th * (1 + r_v / r_d * qv) * (p / p0) ** cvpm
    ph = np.asarray(orig['PH'][:, 0:2] + orig['PHB'][:, 0:2], np.float64)
    dz = (ph[:, 1] - ph[:, 0]) / G_ACC
    return t, alt * DT / dz


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', required=True)
    ap.add_argument('--gpu', action='store_true')
    ap.add_argument('--models', default='twin_k16,ax_rel1')
    ap.add_argument('--steps', type=int, default=720)
    ap.add_argument('--schemes', default='oracle,donor,donorW,sep,shN,shB')
    ap.add_argument('--zsrc', default=str(ZSRC), help="the models' source latents (one seed's zsrc_s<k>.npz)")
    ap.add_argument('--fluxes', default=str(FLUXES), help='WRF output with the 1-minute averaged mass fluxes')
    ap.add_argument('--orig', default=str(ORIG), help='WRF output every 15 minutes with T, QVAPOR, P, PB, PH, PHB '
                    '(for the emission factor), covering the run')
    ap.add_argument('--levels', default='', help='extra model levels (0-based, comma-separated) to write, '
                    'each as level_<k>.npz in the format of surface.npz (without M_offline_over_wrf)')
    args = ap.parse_args()
    if args.gpu:
        import cupy as xp
    else:
        xp = np
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    models = args.models.split(',')

    fx = netCDF4.Dataset(args.fluxes)
    orig = netCDF4.Dataset(args.orig)
    if len(times_of(orig)) < args.steps // OUT_EVERY + 1:
        raise ValueError('the 15-minute output does not cover the run')
    if times_of(fx)[0] != '2010-06-17_00:00:00' or len(times_of(fx)) < args.steps // STEPS_PER_FRAME + 1:
        raise ValueError('flux file does not cover the run')
    g = load_static(fx, xp)
    nz, ny, nx = g.nz, g.ny, g.nx

    # inputs: IC background, IC urban, BC background, emissions 1, 2
    ic = np.transpose(np.asarray(netCDF4.Dataset(PMC / 'ic_control.nc')['num_conc'][:], np.float64), (0, 3, 2, 1))
    ins = np.zeros((schemes.N_INPUTS, nz, ny, nx))
    ins[0:2] = ic
    bc = netCDF4.Dataset(PMC / 'bc_control.nc')
    if bc.dimensions['n_source'].size != 1 or bc.dimensions['n_time'].size != 1:
        raise ValueError('expected one time-constant BC source')
    edge = lambda name: np.asarray(bc[name][0, 0], np.float64).T          # (nz, along)
    bce = {'w': edge('num_conc_xs'), 'e': edge('num_conc_xe'), 's': edge('num_conc_ys'), 'n': edge('num_conc_ye')}
    em = netCDF4.Dataset(PMC / 'emis_control.nc')
    em_t = np.asarray(em['emis_time'][:], np.float64)
    em_flux = np.transpose(np.asarray(em['num_flux'][:, :, :, :, 0], np.float64), (0, 1, 3, 2))   # (2, nt, ny, nx)
    ef_t, ef = emission_factor(orig)

    with np.load(args.zsrc) as f:
        if list(f['pops'] // 25) != [561, 635, 812]:
            raise ValueError('source latents are not for scenarios 561, 635, 812')
        Z = {m: f[m + '_z'].astype(np.float64) for m in models}
    Z['frac'] = np.eye(3)

    sch = tuple(args.schemes.split(','))
    r = schemes.Runner(g, Z, xp, ins, bce, schemes=sch, sep_models=models, shB_models=models + ['frac'])
    mu0 = xp.asarray(np.asarray(fx['MU'][0] + fx['MUB'][0], np.float64))
    M = g.mass(mu0)

    rec = {}
    levels = [int(k) for k in args.levels.split(',') if k]
    rec_lv = {k: {} for k in levels}
    t_out = []
    c1h0, c2h0 = float(fx['C1H'][0][0]), float(fx['C2H'][0][0])

    def record(step):
        s = r.surface(0)
        for k, v in s.items():
            rec.setdefault(k, []).append(v.astype(np.float32))
        for lv in levels:
            for k, v in r.surface(lv).items():
                rec_lv[lv].setdefault(k, []).append(v.astype(np.float32))
        mu_wrf = np.asarray(fx['MU'][step // STEPS_PER_FRAME] + fx['MUB'][step // STEPS_PER_FRAME], np.float64)
        M0 = (M.get() if xp is not np else M)[0]
        rec.setdefault('M_offline_over_wrf', []).append((M0 / (c1h0 * mu_wrf + c2h0)).astype(np.float32)[1:-1, 1:-1])
        t_out.append(step * DT)

    record(0)
    t0 = time.time()
    for step in range(args.steps):
        frame = step // STEPS_PER_FRAME + 1
        F = tuple(xp.asarray(np.asarray(fx[v][frame], np.float64)) for v in ('AVGFLX_RUM', 'AVGFLX_RVM', 'AVGFLX_WWM'))
        M = r.step(F, M, DT)
        t = step * DT
        irec = max(int(np.searchsorted(em_t, t, side='right')) - 1, 0)
        i0 = min(int(t // 900), len(ef_t) - 2)
        a = (t - ef_t[i0]) / (ef_t[i0 + 1] - ef_t[i0])
        conv = (1 - a) * ef[i0] + a * ef[i0 + 1]                           # (ny, nx)
        r.emit(xp.asarray(em_flux[:, irec] * conv[None]))
        if (step + 1) % OUT_EVERY == 0:
            record(step + 1)
        if step % 45 == 0:
            print('step %d/%d  %.1f s elapsed' % (step, args.steps, time.time() - t0), flush=True)
    print('done in %.1f s' % (time.time() - t0))
    np.savez_compressed(out / 'surface.npz', time_s=np.array(t_out), **{k: np.stack(v) for k, v in rec.items()},
                        settings=json.dumps(dict(models=models, schemes=sch, steps=args.steps, dt=DT,
                                                 fluxes=str(args.fluxes), zsrc=str(args.zsrc))))
    print('wrote', out / 'surface.npz')
    for lv in levels:
        np.savez_compressed(out / ('level_%d.npz' % lv), time_s=np.array(t_out),
                            **{k: np.stack(v) for k, v in rec_lv[lv].items()},
                            settings=json.dumps(dict(models=models, schemes=sch, steps=args.steps, dt=DT,
                                                     fluxes=str(args.fluxes), zsrc=str(args.zsrc), level=lv)))
        print('wrote', out / ('level_%d.npz' % lv))


if __name__ == '__main__':
    main()

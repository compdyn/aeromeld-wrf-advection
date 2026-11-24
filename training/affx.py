#!/usr/bin/env python
"""
Affine-x decoder experiments (Nov 2026): a standalone trainer for AeroMELD
models whose decoder is affine in physical per-particle units, so decoded
diagnostics mix exactly, with options aimed at physically valid output.

Model (as code_partnn's mlpencprt + depth-0 mlpdecprt):
    u_p   = (sqrt(m_p) - shift) / scale         per particle, 15 species
    phi_p = MLP(u_p)  (15 -> width x depth -> L-1 [+ L-1 for sigma])
    z     = mean_p phi_p                        exact under mixing
    y_hat = A z + b,  x_hat = s * y_hat         456 per-particle diagnostics

Targets (per particle, i.e. extensive ones divided by n_tot):
    M (15 species x 15 bins, kg), N (15 bins, fraction), C (100, CCN
    fraction), S, A (8 wavelengths, m^2), F (100 temperatures, frozen
    fraction); the loss also has Mtot (15 bins, M summed over species).
    s = the training mean of each element (so y ~ 1).

Options:
  --loss lin   0.5 ((y_hat - y) s / std)^2 per element (the current affine-x
               loss with element scales), mean per block, summed over blocks.
  --loss log   0.5 huber(T(y_hat) - T(y))^2 with T(y) = log(y + eps) for y >= 0
               and log(eps) + y / eps below (steep, so negative values cost a
               lot); CCN stays 'lin' unless --ccn log. Floors eps: F, S, A:
               --eps-q times the 1st percentile of the element; M, Mtot:
               --eps-m times the population's total mass per particle; N:
               --eps-n (fraction of particles).
  --pos        phi = softplus(.) >= 0, A = softplus(.) >= 0, b = softplus(.) >= 0,
               so every decoded value is >= 0 for every population and mixture.
  --kl W       VAE KL weight (mean over latent dims, as cri/kl/rdctn mean);
               --prior learned: N(m, I) with a learned mean m (for --pos,
               whose z cannot be centred at 0); fixed: N(0, I).
  --mixup W    mixup on number-weighted pairs (exact mixtures as targets).
  --mixk KMAX  (with --mixup) mixtures of K = 2, 4, ..., KMAX populations (K drawn
               uniformly per iteration, Dirichlet(1) volume fractions) from the
               pool of the minibatch and the mixup partners (same encoder cost).
  --frozen F   use fixed latents from F (affx_latents.py) instead of training
               an encoder; only A, b are trained (with --pos: the latents are
               shifted to >= 0 and scaled to unit std on the training populations).
  --mono       (with --pos) frozen fraction and the non-activated fraction
               1 - CCN are cumulative sums (from the warm / high-s end) of
               nonnegative increments, so FF falls with T, CCN rises with s and
               CCN <= 1 for every population and mixture (still affine).
  --box K      (with --pos) the last K latent dims have sigmoid features in
               [0, 1] and signed weights; b gets the sum of the negative ones,
               so outputs stay >= 0 over the whole box.
  --simplex    archetype model: softmax features, z on the simplex (L
               archetypes, L - 1 degrees of freedom), decoded columns physical
               by construction: M, S, A >= 0, N sums to 1, CCN and FF monotone
               in [0, 1]; every population and mixture is then physical.
  --inputs     encoder inputs per species: sqrt (as code_partnn), lin, log mass.
  --linblocks  blocks that keep the lin loss under --loss log (default C).
  --wcold W    weight of the frozen-fraction elements at T <= -30 C (the F
               block's mean weight stays 1).
  --oracle     no encoder: a free latent per population (softplus with --pos),
               trained jointly with A, b on all populations (test ones too), so
               the result bounds what any encoder could reach with this decoder
               form and loss (use a large --mb: each latent is only updated
               when its population is drawn).

Evaluation on each seed's test populations (the same 5000 as code_partnn's
seed with that rng seed): the paper's eight metrics (paper_errors.py), the
frozen fraction over the full -40..0 C range (log, floor 1e-14), the share of
populations with any negative value per diagnostic, mix_eval's validity flags,
and 16-way Dirichlet mixtures (for an affine decoder, the mixture of the
decoded populations).

Usage (from anywhere, with code_partnn's venv or any torch + h5py):
    python affx.py --name NAME --out DIR [options]
"""
import argparse
import json
import math
import os
import sys
import time
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paper_errors import paper_metrics, ROWS

p = argparse.ArgumentParser()
p.add_argument('--name', required=True)
p.add_argument('--out', required=True)
p.add_argument('--data', default='/pscratch/sd/m/mwest/aeromeld/data/04_bwchisamp_pm10.h5')
p.add_argument('--cache', default='/pscratch/sd/m/mwest/aeromeld/affx/cache.npz')
p.add_argument('--splits', default='/pscratch/sd/m/mwest/aeromeld/models/test_idx.npz')
p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2], help='indices into the 10 split seeds')
p.add_argument('--iters', type=int, default=30000)
p.add_argument('--lr', type=float, default=1e-3)
p.add_argument('--lr-min', type=float, default=1e-5)
p.add_argument('--lr-dec', type=float, default=1.0, help='learning-rate multiplier for the decoder parameters')
p.add_argument('--warmup', type=int, default=0, help='linear learning-rate warmup iterations')
p.add_argument('--clip', type=float, default=0.0, help='gradient-norm clipping (0: none)')
p.add_argument('--araw0', type=float, default=-4.0, help='--pos: initial raw decoder weights (softplus(-4) = 0.018)')
p.add_argument('--mb', type=int, default=16)
p.add_argument('--L', type=int, default=10, help='latent dim counting n_tot (z has L - 1)')
p.add_argument('--inputs', default='sqrt', help='encoder inputs per species, standardized: any of sqrt, lin, log')
p.add_argument('--width', type=int, default=256)
p.add_argument('--depth', type=int, default=2)
p.add_argument('--loss', choices=['lin', 'log'], default='lin')
p.add_argument('--ccn', choices=['lin', 'log'], default='lin')
p.add_argument('--linblocks', default='C', help='with --loss log: blocks that keep the lin loss (C unless --ccn log)')
p.add_argument('--mloss', choices=['', 'rel2', 'rel1'], default='', help='mass loss: the paper metric\'s '
               'per-population relative L2 error r = |dm| / (|m| + |m_hat|) of speciated, total and bulk mass '
               '(0.5 r^2 or r) instead of the M and Mtot losses of --loss')
p.add_argument('--nloss', choices=['', 'rel2', 'rel1'], default='', help='number loss: as --mloss, for n_prthst')
p.add_argument('--eps-q', type=float, default=0.1)
p.add_argument('--eps-m', type=float, default=1e-3)
p.add_argument('--eps-n', type=float, default=1e-3)
p.add_argument('--huber', type=float, default=3.0, help='huber threshold on log differences (0: plain squares)')
p.add_argument('--wcold', type=float, default=1.0, help='weight of the frozen-fraction elements at T <= -30 C')
p.add_argument('--cpen', type=float, default=0.0, help='penalty weight on CCN and frozen fraction above 1')
p.add_argument('--wblock', default='', help='block weights, e.g. "F=2,S=1"')
p.add_argument('--pos', action='store_true')
p.add_argument('--box', type=int, default=0, help='with --pos: the last BOX latent dims have features in [0, 1] '
               '(sigmoid) and signed weights, with b raised so every output stays >= 0 on the whole box')
p.add_argument('--dec', choices=['affine', 'mlp'], default='affine', help='mlp (with --pos): the nonlinear twin, '
               'softplus(MLP(z)) in place of A z + b, with the same output constraints (--mono)')
p.add_argument('--bounded', action='store_true', help='--dec mlp: softmax heads, so number fractions sum to 1, '
               'CCN and FF are monotone in [0, 1] for every z (not possible for an affine decoder with these constraints)')
p.add_argument('--dec-width', type=int, default=256)
p.add_argument('--dec-depth', type=int, default=2)
p.add_argument('--mono', action='store_true', help='with --pos: C and F as sums of nonnegative increments (monotone)')
p.add_argument('--simplex', action='store_true', help='archetype model: z on the simplex (L archetypes), decoded '
               'columns physical (M, S, A >= 0; N sums to 1; C, F monotone in [0, 1])')
p.add_argument('--kl', type=float, default=0.0)
p.add_argument('--prior', choices=['fixed', 'learned'], default='fixed')
p.add_argument('--mixup', type=float, default=0.0)
p.add_argument('--mixk', type=int, default=2)
p.add_argument('--frozen', default=None, help='npz with z (n_seeds_all, n_pop, d) from affx_latents.py')
p.add_argument('--init-from', default=None, help='a .pt state from an earlier run (same seeds): start the encoder there')
p.add_argument('--freeze-enc', action='store_true', help='with --init-from: train only the decoder')
p.add_argument('--zero-water', action='store_true', help='after training, set the element scales s of the '
               'speciated mass block\'s H2O row (species 14 of 15, all bins) to 0, so the decoder gives exactly no '
               'water: for the dry-diameter targets (dataset/dry_cache.py), where that row is 0 (round 37)')
p.add_argument('--oracle', action='store_true', help='no encoder: a free latent per population (softplus with '
               '--pos), trained with A, b on all populations (test ones too): the capacity of the decoder form')
p.add_argument('--exclude-snr', default='', help='scenarios never trained on, all 25 times (e.g. 561,635,812: '
               'the WRF-AeroMELD source populations, kept as test data)')
p.add_argument('--nmix', type=int, default=1000)
p.add_argument('--device', default='cuda:0')
p.add_argument('--log-every', type=int, default=2000)
args = p.parse_args()
assert args.mixk & (args.mixk - 1) == 0 and 2 <= args.mixk <= 2 * args.mb, '--mixk: a power of 2, <= 2 mb'
LINB = set(filter(None, args.linblocks.split(',')))
if args.ccn == 'log':
    LINB.discard('C')
assert not (args.simplex and (args.pos or args.kl > 0 or args.frozen)), '--simplex excludes --pos, --kl, --frozen'
assert not args.mono or args.pos, '--mono needs --pos'
assert args.dec == 'affine' or (args.pos and not args.box and not args.simplex), '--dec mlp needs --pos (no --box, --simplex)'
assert not args.bounded or args.dec == 'mlp', '--bounded needs --dec mlp'
assert not args.box or (args.pos and args.kl == 0 and not args.frozen), '--box needs --pos, no --kl, no --frozen'
assert not args.oracle or (args.kl == 0 and not args.frozen and not args.simplex and not args.box), \
    '--oracle excludes --kl, --frozen, --simplex, --box'
ENC = args.frozen is None and not args.oracle          # latents from the particle encoder
dev = torch.device(args.device)
os.makedirs(args.out, exist_ok=True)
t_start = time.time()

########################################## Data ##########################################
BLOCKS = ['M', 'N', 'C', 'S', 'A', 'F']           # decoder outputs, in this order
SHAPES = {'M': (15, 15), 'N': (1, 15), 'C': (1, 100), 'S': (1, 8), 'A': (1, 8), 'F': (1, 100)}
SIZES = {k: int(np.prod(v)) for k, v in SHAPES.items()}
OFFS = dict(zip(BLOCKS, np.cumsum([0] + [SIZES[k] for k in BLOCKS])[:-1]))
D_OUT = sum(SIZES.values())
DSNAME = {'M': 'm_chmprthst', 'N': 'n_prthst', 'C': 'ccn_cdf', 'S': 'qs_pop', 'A': 'qa_pop', 'F': 'frznfrac_tmp'}
EXT = {'M', 'N', 'S', 'A'}                         # extensive: divided by n_tot

if not os.path.exists(args.cache):
    print('building cache', args.cache, flush=True)
    with h5py.File(args.data, 'r') as f:
        ntot = f['n_tot'][:].reshape(-1)
        X = np.concatenate([(f[DSNAME[k]][:].reshape(len(ntot), -1) / (ntot[:, None] if k in EXT else 1))
                            for k in BLOCKS], axis=1)
        m = f['m_prtchm'][:].astype(np.float32)
        w = f['w_prt'][:].reshape(len(ntot), -1)
        assert np.allclose(w, w[:, :1]), 'the encoder below assumes equal particle weights'
        i_snr = f['i_snr'][:]
    np.savez(args.cache, X=X, ntot=ntot, m=m, i_snr=i_snr)
    del m
c = np.load(args.cache)
X64, ntot, i_snr = c['X'], c['ntot'], c['i_snr']
n_pop = len(ntot)
splits = np.load(args.splits)
assert X64.shape == (n_pop, D_OUT)
m_prt = torch.from_numpy(c['m']).to(dev) if ENC else None   # (n_pop, 1000, 15) float32
U_prt = None
print(f'data: {n_pop} populations, {D_OUT} outputs, loaded in {time.time() - t_start:.0f} s', flush=True)
frozen_z = np.load(args.frozen)['z'] if args.frozen else None

wblock = {k: 1.0 for k in BLOCKS + ['Mtot', 'Mbulk']}
for kv in filter(None, args.wblock.split(',')):
    k, v = kv.split('='); wblock[k] = float(v)
wblock['over1'] = 1.0

######################################### Model ##########################################
class Model(nn.Module):
    def __init__(self, s, ymean, d_z, n_in=15, extra=None):
        super().__init__()
        self.d_z = d_z
        if args.oracle:
            g0 = torch.Generator().manual_seed(1)
            self.zraw = nn.Parameter(0.5 * torch.randn(n_pop, d_z, generator=g0))
        if ENC:
            dims = [n_in] + [args.width] * args.depth
            layers = []
            for a, b in zip(dims[:-1], dims[1:]):
                layers += [nn.Linear(a, b), nn.ReLU()]
            self.mlp = nn.Sequential(*layers, nn.Linear(dims[-1], 2 * d_z if args.kl > 0 else d_z))
        self.register_buffer('s', s)
        g = torch.Generator().manual_seed(0)
        if args.simplex:
            # archetype columns: M, S, A rows softplus (y ~ 1 at start); N, C, F from softmaxes
            self.R = nn.Parameter(math.log(math.expm1(1.0)) + 0.3 * torch.randn(D_OUT, d_z, generator=g))
            for k in ['N', 'C', 'F']:
                lg = torch.log(extra['w0_' + k].clamp(min=1e-12))[:, None]
                setattr(self, 'R' + k, nn.Parameter(lg + 0.3 * torch.randn(len(lg), d_z, generator=g)))
            self.register_buffer('s_inv', (1 / s).float())
        elif args.pos:
            # softplus(a) small, softplus(b) = the mean of y (~1)
            if args.dec == 'affine':
                self.Araw = nn.Parameter(args.araw0 + 0.5 * torch.randn(D_OUT, d_z - args.box, generator=g))
            if args.box:
                self.Abox = nn.Parameter(0.02 * torch.randn(D_OUT, args.box, generator=g))
            bm = ymean.clone()
            if args.mono:
                # C, F rows are increments, in units of their mean increments
                for k in ['C', 'F']:
                    self.register_buffer('G' + k, extra['G' + k])
                    bm[OFFS[k]:OFFS[k] + SIZES[k]] = extra['incr' + k]
                self.register_buffer('cC', extra['cC'])
            if args.dec == 'affine':
                self.braw = nn.Parameter(torch.log(torch.expm1(bm.clamp(min=1e-6))))
            else:
                # the nonlinear twin: code_partnn's mlpdecprt shape (d_z -> width x depth -> outputs, ReLU),
                # its raw outputs through the same softplus (and monotone sums) as A z + b; starts at the mean
                dims = [d_z] + [args.dec_width] * args.dec_depth
                layers = []
                for a_, b2 in zip(dims[:-1], dims[1:]):
                    layers += [nn.Linear(a_, b2), nn.ReLU()]
                last = nn.Linear(dims[-1], D_OUT + (2 if args.bounded else 0))
                b0 = torch.log(torch.expm1(bm.clamp(min=1e-6)))
                if args.bounded:
                    # softmax logits at the log of the mean training shares: N over its 15 bins, C over its 100
                    # increments from the low-s end plus the share never activated, F over its 100 increments
                    # from the warm end plus the share never frozen (the last two raw outputs)
                    lg = lambda v: torch.log(v.clamp(min=1e-12))
                    b0 = b0.clone()
                    b0[OFFS['N']:OFFS['N'] + 15] = lg(extra['w0_N'])
                    b0[OFFS['C']:OFFS['C'] + 100] = lg(extra['w0_C'][:100])
                    b0[OFFS['F']:OFFS['F'] + 100] = lg(extra['w0_F'][:100])
                    b0 = torch.cat([b0, lg(extra['w0_C'][100:]).to(b0), lg(extra['w0_F'][100:]).to(b0)])
                    self.register_buffer('s_inv', (1 / s).float())
                with torch.no_grad():
                    last.weight.mul_(0.01)
                    last.bias.copy_(b0)
                self.dmlp = nn.Sequential(*layers, last)
        else:
            self.A = nn.Parameter(0.1 * torch.randn(D_OUT, d_z, generator=g) / math.sqrt(d_z))
            self.b = nn.Parameter(ymean.clone())
        self.prior_m = nn.Parameter(torch.zeros(d_z), requires_grad=(args.prior == 'learned'))

    def phi(self, u):
        """Per-particle features (..., d_z) and raw sigma features (..., d_z) or None."""
        h = self.mlp(u)
        mu, sig = (h[..., :self.d_z], h[..., self.d_z:]) if args.kl > 0 else (h, None)
        if args.simplex:
            mu = F.softmax(mu, dim=-1)
        elif args.pos:
            nb = self.d_z - args.box
            mu = torch.cat([F.softplus(mu[..., :nb]), torch.sigmoid(mu[..., nb:])], dim=-1) if args.box else F.softplus(mu)
        return mu, sig

    def zlat(self, idx):
        """--oracle: the free latents of populations idx."""
        z = self.zraw[idx]
        return F.softplus(z) if args.pos else z

    def encode(self, u):
        """u (n, n_part, 15) -> mu (n, d_z), logsig (n, d_z) or None; z is the particle mean."""
        mu, sig = self.phi(u)
        mu = mu.mean(dim=1)
        if sig is None:
            return mu, None
        sig = sig.mean(dim=1)
        logsig = torch.where(sig < 0, sig, torch.log1p(sig.clamp(min=0)))   # 'explin', as code_partnn
        return mu, logsig

    def AB(self):
        """Effective A (D_OUT, d_z) and b (D_OUT,), in units of s."""
        if args.simplex:
            A = F.softplus(self.R)
            si = self.s_inv[:, None]
            parts = [A[:OFFS['N']]]
            parts.append(F.softmax(self.RN, dim=0) * si[OFFS['N']:OFFS['N'] + 15])
            parts.append(torch.cumsum(F.softmax(self.RC, dim=0)[:100], dim=0) * si[OFFS['C']:OFFS['C'] + 100])
            parts.append(A[OFFS['S']:OFFS['F']])
            wf = F.softmax(self.RF, dim=0)[:100]
            parts.append(torch.flip(torch.cumsum(torch.flip(wf, [0]), dim=0), [0]) * si[OFFS['F']:])
            return torch.cat(parts), torch.zeros(D_OUT, device=A.device, dtype=A.dtype)
        if args.pos:
            A, b = F.softplus(self.Araw), F.softplus(self.braw)
            if args.box:
                # z_box in [0, 1]: the smallest value of A_box z_box is sum of the negative weights
                A, b = torch.cat([A, self.Abox], dim=1), b + F.relu(-self.Abox).sum(1)
            if args.mono:
                A, b = A.clone(), b.clone()
                for k in ['C', 'F']:
                    sl = slice(OFFS[k], OFFS[k] + SIZES[k]); G = getattr(self, 'G' + k)
                    A[sl], b[sl] = G @ A[sl], G @ b[sl]
                b[OFFS['C']:OFFS['C'] + 100] += self.cC
            return A, b
        return self.A, self.b

    def decode(self, z):
        """z (..., d_z) -> y_hat (..., D_OUT), in units of s."""
        if args.dec == 'mlp' and args.bounded:
            r = self.dmlp(z)
            out = []
            for k in BLOCKS:
                sl = slice(OFFS[k], OFFS[k] + SIZES[k]); si = self.s_inv[sl]
                if k == 'N':
                    out.append(F.softmax(r[..., sl], dim=-1) * si)
                elif k == 'C':                     # activated fraction: cumulative shares from the low-s end
                    w = F.softmax(torch.cat([r[..., sl], r[..., D_OUT:D_OUT + 1]], -1), dim=-1)[..., :100]
                    out.append(torch.cumsum(w, -1) * si)
                elif k == 'F':                     # frozen fraction: cumulative shares from the warm end
                    w = F.softmax(torch.cat([r[..., sl], r[..., D_OUT + 1:]], -1), dim=-1)[..., :100]
                    out.append(torch.flip(torch.cumsum(torch.flip(w, [-1]), -1), [-1]) * si)
                else:
                    out.append(F.softplus(r[..., sl]))
            return torch.cat(out, -1)
        if args.dec == 'mlp':
            v = F.softplus(self.dmlp(z))
            if not args.mono:
                return v
            parts = []
            for k in BLOCKS:
                vk = v[..., OFFS[k]:OFFS[k] + SIZES[k]]
                if k in ('C', 'F'):
                    vk = vk @ getattr(self, 'G' + k).T + (self.cC if k == 'C' else 0)
                parts.append(vk)
            return torch.cat(parts, dim=-1)
        A, b = self.AB()
        return z @ A.T + b

def T_log(y, eps):
    """log(y + eps) for y >= 0, continued linearly (slope 1 / eps) below 0."""
    return torch.where(y >= 0, torch.log(y.clamp(min=0) + eps), torch.log(eps) + y / eps)

def huber(d, delta):
    if delta <= 0:
        return 0.5 * d.square()
    a = d.abs()
    return torch.where(a <= delta, 0.5 * d.square(), delta * (a - 0.5 * delta))

######################################## Metrics #########################################
def to_dataset_units(Xpp, n):
    """per-particle (n_pop, D_OUT) -> dict of dataset diagnostics (n_pop, C, L)."""
    out = dict()
    for k in BLOCKS:
        v = Xpp[:, OFFS[k]:OFFS[k] + SIZES[k]].reshape(-1, *SHAPES[k])
        out[DSNAME[k]] = v * n[:, None, None] if k in EXT else v
    return out

def relerr(y, yhat):
    num = np.sqrt(((y - yhat) ** 2).sum(axis=(-1, -2)))
    den = np.sqrt((y ** 2).sum(axis=(-1, -2))) + np.sqrt((yhat ** 2).sum(axis=(-1, -2))) + 1e-30
    return num / den

def evaluate(Xo, Xr, n):
    """Per-population metrics between true Xo and decoded Xr (per particle, (n_pop, D_OUT))."""
    yo, yr = to_dataset_units(Xo, n), to_dataset_units(Xr, n)
    e = {ROWS[k]: v for k, v in paper_metrics(yo, yr).items()}
    lf = lambda v: np.log(np.clip(v, 1e-14, None))
    e['ff_full'] = relerr(lf(yo['frznfrac_tmp']), lf(yr['frznfrac_tmp']))
    for k in BLOCKS:
        e[f'neg_{k}'] = (Xr[:, OFFS[k]:OFFS[k] + SIZES[k]] < 0).any(axis=1).astype(float)
    e['neg_any'] = (Xr < 0).any(axis=1).astype(float)
    # mix_eval's validity flags (with its tolerance)
    tol = 1e-3
    bad = np.zeros(len(n), bool)
    for c in ['m_chmprthst', 'n_prthst', 'qs_pop', 'qa_pop']:
        x = yr[c]; bad |= x.min(axis=(-1, -2)) < -tol * (np.abs(x).max(axis=(-1, -2)) + 1e-300)
    ccn = yr['ccn_cdf'][:, 0]
    bad |= (ccn.min(-1) < -tol) | (ccn.max(-1) > 1 + tol) | (np.diff(ccn, axis=-1).min(-1) < -tol)
    ff = yr['frznfrac_tmp'][:, 0]
    bad |= (ff.min(-1) < -tol * np.abs(ff).max(-1)) | (ff.max(-1) > 1 + tol)
    e['viol_any'] = bad.astype(float)
    e['ccn_gt1'] = (ccn.max(-1) > 1 + tol).astype(float)
    e['ff_nonmono'] = (np.diff(ff, axis=-1).max(-1) > tol * np.abs(ff).max(-1)).astype(float)
    return e

GEO = ['ccn_cdf', 'logqs_pop', 'logqa_pop', 'logfrznfrac_tmp', 'm_chmprthst', 'n_prthst', 'm_prthst', 'm_chmprt']

def summarize(e):
    out = {k: float(np.mean(v)) for k, v in e.items()}
    out['geo'] = float(np.exp(np.mean([np.log(out[k]) for k in GEO])))
    return out

################################# Training, one seed at a time ###########################
results = dict(args=vars(args), seeds=dict())
for si in args.seeds:
    seed = int(splits['seeds'][si])
    torch.manual_seed(seed); rng = np.random.RandomState(seed)
    tst = np.sort(splits['idx'][si]); is_tst = np.zeros(n_pop, bool); is_tst[tst] = True
    trn = np.where(~is_tst)[0]
    if args.oracle:
        trn = np.arange(n_pop)                                 # a capacity bound: fit every population
    if args.exclude_snr:
        trn = trn[~np.isin(i_snr[trn], [int(v) for v in args.exclude_snr.split(',')])]
    # Element scales s, std, floors, from the training populations
    Xt = X64[trn]
    # Element scales: the training mean; an element that is always 0 in training (9 species/bin
    # pairs of M in the two smallest bins) gets 1e-3 of the largest mean in its bin (M) or block
    s = Xt.mean(axis=0)
    fb = np.zeros(D_OUT)
    fb[OFFS['M']:OFFS['M'] + SIZES['M']] = 1e-3 * np.broadcast_to(
        s[OFFS['M']:OFFS['M'] + SIZES['M']].reshape(15, 15).max(axis=0), (15, 15)).reshape(-1)
    for k in BLOCKS[1:]:
        fb[OFFS[k]:OFFS[k] + SIZES[k]] = 1e-3 * s[OFFS[k]:OFFS[k] + SIZES[k]].max()
    s = np.where(s > 0, s, fb)
    sd = Xt.std(axis=0); sd = np.where(sd > 0, sd, s)
    Y = X64 / s                                                # (n_pop, D_OUT)
    eps = np.zeros(D_OUT)
    for k in ['S', 'A', 'F', 'C']:
        sl = slice(OFFS[k], OFFS[k] + SIZES[k])
        eps[sl] = args.eps_q * np.percentile(Xt[:, sl], 1, axis=0) / s[sl]
    eps[OFFS['N']:OFFS['N'] + SIZES['N']] = args.eps_n / s[OFFS['N']:OFFS['N'] + SIZES['N']]
    eps = np.clip(eps, 1e-30, None)
    # Mass floors and the total-mass block, as float64 ratios (per-particle masses are ~1e-20 kg,
    # and their products and quotients over- or underflow in float32)
    sl_M = slice(OFFS['M'], OFFS['M'] + SIZES['M'])
    sM = s[sl_M]
    Mbins = X64[:, sl_M].reshape(n_pop, 15, 15).sum(axis=1)                   # (n_pop, 15) mass per bin
    Mtot_pp = Mbins.sum(axis=1)                                                # total mass per particle
    Mref = Mtot_pp[trn].mean()
    sMt = Mbins[trn].mean(axis=0); sMt = np.where(sMt > 0, sMt, Mref)
    sdMt = Mbins[trn].std(axis=0); sdMt = np.where(sdMt > 0, sdMt, sMt)
    r_mt = (sM.reshape(15, 15) / sMt[None, :]).reshape(-1)                     # y_M -> Mtot in units of sMt
    assert np.isfinite(Y).all() and np.isfinite(eps).all()

    tY = torch.from_numpy(Y).float().to(dev)
    t_s = torch.from_numpy(s).double().to(dev)                  # kept in the state dict: x = s * y
    t_sdr = torch.from_numpy(s / sd).float().to(dev)
    t_s32 = torch.from_numpy(s).float().to(dev)             # (y - y') * s / sd = scaled difference
    t_eps = torch.from_numpy(eps).float().to(dev)
    t_mtn = torch.from_numpy(Mtot_pp / Mref).float().to(dev)                   # total mass / Mref
    t_Mref_sM = torch.from_numpy(np.clip(Mref / sM, 0, 1e30)).float().to(dev)
    t_Mref_sMt = torch.from_numpy(np.clip(Mref / sMt, 0, 1e30)).float().to(dev)
    t_rmt = torch.from_numpy(r_mt).float().to(dev)
    t_sMn = torch.from_numpy(sM / sM.max()).float().to(dev)                    # y_M -> mass in units of max(sM)
    sN_ = s[OFFS['N']:OFFS['N'] + SIZES['N']]
    t_sNn = torch.from_numpy(sN_ / sN_.max()).float().to(dev)
    t_sdMtr = torch.from_numpy(sMt / sdMt).float().to(dev)
    t_n = torch.from_numpy(ntot / ntot.mean()).float().to(dev)
    welem = np.ones(D_OUT)
    welem[OFFS['F']:OFFS['F'] + 100][np.linspace(-40, 0, 100) <= -30 + 1e-9] = args.wcold
    welem[OFFS['F']:OFFS['F'] + 100] /= welem[OFFS['F']:OFFS['F'] + 100].mean()
    t_w = torch.from_numpy(welem).float().to(dev)

    # Encoder input scaling (polyprt: sqrt mass, per-species shift and scale over training particles)
    if ENC:
        mt = m_prt[torch.from_numpy(trn[::10]).to(dev)].reshape(-1, 15)
        mfl = torch.stack([mt[:, j][mt[:, j] > 0].mean() if (mt[:, j] > 0).any() else mt.new_tensor(1e-20)
                           for j in range(15)]) * 1e-6           # log floors: 1e-6 of the mean nonzero mass
        TR = {'sqrt': torch.sqrt, 'lin': lambda m: m * 1e18, 'log': lambda m: torch.log(m + mfl)}
        kinds = args.inputs.split(',')
        u_sh, u_sc = [], []
        for kd in kinds:
            v = TR[kd](mt)
            u_sh.append(v.mean(0)); u_sc.append(v.std(0).clamp(min=1e-30))
        del mt, v
        def inputs(idx):
            m = m_prt[idx]
            return torch.cat([(TR[kd](m) - a) / b for kd, a, b in zip(kinds, u_sh, u_sc)], dim=-1)
    elif args.frozen is not None:
        Zf = torch.from_numpy(frozen_z[si]).float().to(dev)

    d_z = args.L - 1 if args.frozen is None else frozen_z.shape[-1]
    extra = dict()
    if args.frozen is not None and args.pos:
        ztr = Zf[torch.from_numpy(trn).to(dev)]
        Zf = (Zf - ztr.min(0).values) / ztr.std(0)                   # z >= 0 (and ~1) on the training populations
    xC, xF = Xt[:, OFFS['C']:OFFS['C'] + 100], Xt[:, OFFS['F']:OFFS['F'] + 100]
    dC = np.diff(xC, axis=1, prepend=0).mean(0)                     # C(j) = sum_{i<=j} dC_i
    dF = -np.diff(xF, axis=1, append=0).mean(0)                     # F(j) = sum_{i>=j} dF_i
    if args.mono:
        # FF(j) = sum_{i>=j} dF_i and CCN(j) = 1 - U(j), U(j) = sum_{i>=j} dU_i (the fraction not
        # activated), increments dF, dU = A z + b >= 0 in units of their training means: FF falls
        # with T, CCN rises with s and stays <= 1, and the coefficients of z can vanish where
        # every population has CCN = 1 (a cumulative CCN with A >= 0 could not)
        dU = np.append(np.diff(xC, axis=1).mean(0), 1 - xC[:, -1].mean())
        for k, d, sgn, mean_k in [('C', dU, -1.0, 1 - xC.mean(0)), ('F', dF, 1.0, s[OFFS['F']:OFFS['F'] + 100])]:
            sk = s[OFFS[k]:OFFS[k] + 100]
            sd_ = np.maximum(d, 1e-3 * mean_k)
            extra['G' + k] = torch.from_numpy(sgn * np.triu(np.ones((100, 100))) * sd_[None, :] / sk[:, None]).float().to(dev)
            extra['incr' + k] = torch.from_numpy(np.maximum(d, 0) / sd_).float().to(dev)
        extra['cC'] = torch.from_numpy(1 / s[OFFS['C']:OFFS['C'] + 100]).float().to(dev)
    if args.simplex or args.bounded:
        d_z = args.L if args.simplex else d_z
        extra['w0_N'] = torch.from_numpy(Xt[:, OFFS['N']:OFFS['N'] + 15].mean(0)).float()
        extra['w0_C'] = torch.from_numpy(np.append(dC, max(1 - xC[:, -1].mean(), 1e-6))).float()
        extra['w0_F'] = torch.from_numpy(np.append(dF, max(1 - xF[:, 0].mean(), 1e-6))).float()
    model = Model(t_s, tY[torch.from_numpy(trn).to(dev)].mean(0), d_z, n_in=15 * len(args.inputs.split(',')),
                  extra=extra).to(dev)
    if args.init_from:
        st = torch.load(args.init_from.replace('SEED', str(si)), map_location=dev)
        model.mlp.load_state_dict({k[4:]: v for k, v in st.items() if k.startswith('mlp.')})
    named = [(n_, q) for n_, q in model.named_parameters() if q.requires_grad and not (args.freeze_enc and n_.startswith('mlp.'))]
    p_enc = [q for n_, q in named if n_.startswith('mlp.')]
    p_dec = [q for n_, q in named if not n_.startswith('mlp.')]
    groups = ([dict(params=p_enc, lr=args.lr)] if p_enc else []) + [dict(params=p_dec, lr=args.lr * args.lr_dec)]
    opt = torch.optim.Adam(groups)
    def lr_factor(it):
        w = min(1.0, (it + 1) / args.warmup) if args.warmup else 1.0
        r = args.lr_min / args.lr
        return w * (r + (1 - r) * 0.5 * (1 + math.cos(math.pi * it / args.iters)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    params = [q for _, q in named]
    trn_t = torch.from_numpy(trn).to(dev)

    def block_losses(yhat, y, mtot):
        """Per-block losses (scalars) between y_hat and targets y (b, D_OUT); mtot (b,) total mass per particle."""
        L = dict()
        for k in BLOCKS:
            sl = slice(OFFS[k], OFFS[k] + SIZES[k])
            if args.loss == 'lin' or k in LINB:
                L[k] = (0.5 * ((yhat[:, sl] - y[:, sl]) * t_sdr[sl]).square() * t_w[sl]).mean()
            else:
                e_ = (args.eps_m * mtot[:, None] * t_Mref_sM).clamp(min=1e-30) if k == 'M' else t_eps[sl]
                L[k] = (huber(T_log(yhat[:, sl], e_) - T_log(y[:, sl], e_), args.huber) * t_w[sl]).mean()
        # Total mass per bin (sum over species), in units of its own mean
        sl = slice(OFFS['M'], OFFS['M'] + SIZES['M'])
        mt = lambda yy: (yy[:, sl] * t_rmt).reshape(-1, 15, 15).sum(1)
        mth, mtt = mt(yhat), mt(y)
        if args.loss == 'lin' or 'M' in LINB:
            L['Mtot'] = 0.5 * ((mth - mtt) * t_sdMtr).square().mean()
        else:
            e_ = (args.eps_m * mtot[:, None] * t_Mref_sMt).clamp(min=1e-30)
            L['Mtot'] = huber(T_log(mth, e_) - T_log(mtt, e_), args.huber).mean()
        if args.mloss:
            # the paper's mass metrics: per population, |m - m_hat| / (|m| + |m_hat|) over species x bins,
            # bins (summed over species) and species (summed over bins), in physical (relative) units
            mh, mo = ((yy[:, sl] * t_sMn).reshape(-1, 15, 15) for yy in (yhat, y))
            for key, a, b_ in [('M', mh, mo), ('Mtot', mh.sum(1), mo.sum(1)), ('Mbulk', mh.sum(2), mo.sum(2))]:
                a, b_ = a.reshape(len(a), -1), b_.reshape(len(b_), -1)
                nrm = lambda v: (v.square().sum(1) + 1e-24).sqrt()          # no 0/0 in the gradient
                r = nrm(a - b_) / (nrm(a) + nrm(b_))
                L[key] = (0.5 * r.square() if args.mloss == 'rel2' else r).mean()
        if args.nloss:
            sl_N = slice(OFFS['N'], OFFS['N'] + SIZES['N'])
            a, b_ = yhat[:, sl_N] * t_sNn, y[:, sl_N] * t_sNn
            nrm = lambda v: (v.square().sum(1) + 1e-24).sqrt()
            r = nrm(a - b_) / (nrm(a) + nrm(b_))
            L['N'] = (0.5 * r.square() if args.nloss == 'rel2' else r).mean()
        if args.cpen > 0:
            # CCN and frozen fractions (intensive, x = s y) above 1, squared, mean over elements
            ov = [F.relu(yhat[:, OFFS[k]:OFFS[k] + 100] * t_s32[OFFS[k]:OFFS[k] + 100] - 1) for k in ['C', 'F']]
            L['over1'] = args.cpen * 0.5 * torch.cat(ov, 1).square().mean() * 1e4
        return L

    hist = []
    t0 = time.time()
    model.train()
    for it in range(args.iters):
        idx = trn_t[torch.randint(len(trn), (args.mb,), device=dev)]
        if ENC:
            mu, logsig = model.encode(inputs(idx))
        else:
            mu, logsig = (model.zlat(idx) if args.oracle else Zf[idx]), None
        z = mu if logsig is None else mu + logsig.exp() * torch.randn_like(mu)
        if args.pos and logsig is not None:
            z = z.clamp(min=0)                     # the decoder only ever sees z >= 0
        Lb = block_losses(model.decode(z), tY[idx], t_mtn[idx])
        loss = sum(wblock[k] * v for k, v in Lb.items())
        if args.mixup > 0:
            # pairs (idx, j) mixed in volume fractions lam; per-particle quantities and z
            # mix with the number fractions alpha (exact; the decoded mixture is affine in z)
            j = trn_t[torch.randint(len(trn), (args.mb,), device=dev)]
            lam = torch.rand(args.mb, 1, device=dev)
            alpha = lam * t_n[idx, None] / (lam * t_n[idx, None] + (1 - lam) * t_n[j, None])
            muj = model.encode(inputs(j))[0] if ENC else model.zlat(j) if args.oracle else Zf[j]
            if args.mixk > 2:
                # K-way mixtures (K = 2, 4, ..., mixk) of distinct members of the pool (idx, j)
                pool, pmu = torch.cat([idx, j]), torch.cat([mu, muj])
                K = 2 ** int(torch.randint(1, int(round(math.log2(args.mixk))) + 1, ()))
                pick = torch.rand(args.mb, len(pool), device=dev).topk(K, 1).indices
                e = -torch.rand(args.mb, K, device=dev).clamp_min(1e-12).log()      # Dirichlet(1)
                al = e * t_n[pool][pick]
                al = al / al.sum(1, keepdim=True)
                zmix = torch.einsum('mk,mkd->md', al, pmu[pick])
                Lm = block_losses(model.decode(zmix), torch.einsum('mk,mkd->md', al, tY[pool][pick]),
                                  (al * t_mtn[pool][pick]).sum(1))
            else:
                zmix = alpha * mu + (1 - alpha) * muj
                Lm = block_losses(model.decode(zmix), alpha * tY[idx] + (1 - alpha) * tY[j],
                                  alpha[:, 0] * t_mtn[idx] + (1 - alpha[:, 0]) * t_mtn[j])
            loss = loss + args.mixup * sum(wblock[k] * v for k, v in Lm.items())
        if logsig is not None:
            kl = 0.5 * ((mu - model.prior_m).square() + (2 * logsig).exp() - 2 * logsig - 1).mean()
            loss = loss + args.kl * kl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(params, args.clip)
        opt.step(); sched.step()
        if it % args.log_every == 0 or it == args.iters - 1:
            rec = {k: round(v.item(), 5) for k, v in Lb.items()}
            hist.append(dict(it=it, loss=loss.item(), **rec))
            print(f'[{args.name} s{si}] it {it:6d} loss {loss.item():.4f} '
                  + ' '.join(f'{k}={v:.4f}' for k, v in rec.items())
                  + f'  ({time.time() - t0:.0f} s)', flush=True)

    if args.zero_water:
        # the dry targets' H2O row is 0; its outputs, trained towards 0 at the fallback scale, are set to 0
        assert np.all(X64[:, OFFS['M'] + 210:OFFS['M'] + 225] == 0), '--zero-water: the cache has water'
        s[OFFS['M'] + 210:OFFS['M'] + 225] = 0.0
        with torch.no_grad():
            model.s[OFFS['M'] + 210:OFFS['M'] + 225] = 0.0
    # Evaluation on this seed's test populations
    model.eval()
    with torch.no_grad():
        Z = []
        for i0 in range(0, n_pop, 500):
            ii = torch.arange(i0, min(i0 + 500, n_pop), device=dev)
            Z.append(model.encode(inputs(ii))[0] if ENC else model.zlat(ii) if args.oracle else Zf[ii])
        Z = torch.cat(Z)
        Yhat = model.decode(Z).double().cpu().numpy()
    Xhat = Yhat * s
    e_tst = evaluate(X64[tst], Xhat[tst], ntot[tst])
    # 16-way mixtures of test populations, Dirichlet(1) volume fractions; exact for an affine decoder
    K = 16
    pick = np.stack([rng.choice(tst, K, replace=False) for _ in range(args.nmix)])
    lam = rng.dirichlet(np.ones(K), size=args.nmix)
    nmix = (lam * ntot[pick]).sum(1)
    alpha = lam * ntot[pick] / nmix[:, None]
    # mix16: the decoded mixed latent against the true mixture (what transport gives); for an affine
    # decoder it equals the mixture of the decoded populations (mix16_dm); comm16 compares the two
    # (the decoding-mixtures error)
    mixo = np.einsum('mk,mkd->md', alpha, X64[pick]); mixdm = np.einsum('mk,mkd->md', alpha, Xhat[pick])
    with torch.no_grad():
        zmix = torch.einsum('mk,mkd->md', torch.from_numpy(alpha).to(Z), Z[torch.from_numpy(pick).to(dev)])
        mixr = (model.decode(zmix).double().cpu().numpy()) * s
    e_mix = evaluate(mixo, mixr, nmix)
    r = dict(seed=seed, train_s=time.time() - t0, test=summarize(e_tst), mix16=summarize(e_mix),
             mix16_dm=summarize(evaluate(mixo, mixdm, nmix)), comm16=summarize(evaluate(mixdm, mixr, nmix)),
             hist=hist, z_range=[float(Z.min()), float(Z.max())])
    results['seeds'][si] = r
    print(f'[{args.name} s{si}] test geo {r["test"]["geo"]:.2%}  mix16 geo {r["mix16"]["geo"]:.2%}  '
          f'neg_any {r["test"]["neg_any"]:.1%}  viol {r["test"]["viol_any"]:.1%}  ff_full {r["test"]["ff_full"]:.2%}', flush=True)
    enc_pp = dict() if not ENC else dict(u_kinds=args.inputs, u_shift=torch.cat(u_sh), u_scale=torch.cat(u_sc))
    if args.dec == 'affine':
        with torch.no_grad():
            A_eff, b_eff = model.AB()
        enc_pp.update(A_eff=A_eff.detach(), b_eff=b_eff.detach())
    else:
        np.save(f'{args.out}/{args.name}_s{si}_X.npy', Xhat.astype(np.float32))    # decoded, per particle
    torch.save({**model.state_dict(), **enc_pp, 'dec': args.dec, 'mono': bool(args.mono), 'bounded': bool(args.bounded)},
               f'{args.out}/{args.name}_s{si}.pt')
    np.save(f'{args.out}/{args.name}_s{si}_Z.npy', Z.cpu().numpy().astype(np.float32))

# Mean over seeds
agg = dict()
for part in ['test', 'mix16', 'mix16_dm', 'comm16']:
    keys = results['seeds'][args.seeds[0]][part].keys()
    agg[part] = {k: float(np.mean([results['seeds'][si][part][k] for si in args.seeds])) for k in keys}
results['mean'] = agg
results['wall_s'] = time.time() - t_start
with open(f'{args.out}/{args.name}.json', 'w') as fp:
    json.dump(results, fp, indent=1)
t = agg['test']; m = agg['mix16']
print(f'== {args.name}: test geo {t["geo"]:.2%} | ' + ' '.join(f'{k} {t[k]:.1%}' for k in GEO)
      + f' | ff_full {t["ff_full"]:.1%} | neg_any {t["neg_any"]:.1%} '
      + ' '.join(f'neg_{k} {t["neg_" + k]:.1%}' for k in BLOCKS)
      + f' | viol {t["viol_any"]:.1%} | mix16 geo {m["geo"]:.2%} neg {m["neg_any"]:.1%}'
      + f' | decoding mixtures (comm16) geo {agg["comm16"]["geo"]:.2%} | {results["wall_s"]:.0f} s', flush=True)

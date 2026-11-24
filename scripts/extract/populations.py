"""`populations_<model>`: the three seed populations, encoded and decoded by one model.

Everything comes from ONE particle file per population: the 1000 particles
WRF-PartMC was initialised from (config.PARTICLE_DIR, scenario_<s>_filtered_
0001_<PARTICLE_TIME>.nc; raw_data/filter.sh cut them out of the full PartMC
state). The model's encoder turns them into the latents, its decoder into the
reconstructions, and the PartMC curves are computed from the same particles. So
all three describe the same particles, whatever model is used. Nothing here
reads a WRF run, a latent dump or the diagnostics.

Every particle in the file goes to the encoder, in file order, with no
resampling. The encoder's inputs are all overwritten with the file's data; the
build refuses a model whose encoder expects any other input, so no random
placeholder can reach the latents.

ntot. Each population is decoded at ntot latent = NTOT_LATENT (0, the centre of
the model's transformed n_tot space). The particle files cannot supply a total:
they keep each particle's original weight, so they sum to 1.8e7 / 1.8e7 / 3.6e7
m^-3, about 1/159 of the populations they were drawn from, and that would put
the ntot latent near -9.4 for 05_nonlinvlong.0.0, far outside training. The
encoder is still handed the file's own sum; z does not depend on it. The
number concentration latent 0 stands for is read back from the decoder's own
n_tot output (`ml_ntot`).

Basis. The two sides are stored on their own bases, as in the other products:

  ml_*   decoder outputs at NTOT_LATENT: per m^3 (the training data's basis),
         totalling ml_ntot.
  pmc_*  the file's particles with their own weights: per m^3, totalling
         pmc_ntot, ~1/159 of the sampled population.

A plot puts PartMC on the decoder's basis by multiplying its EXTENSIVE
quantities (size and mass distributions, optics) by ml_ntot / pmc_ntot. CCN/CN
and frozen fraction are fractions and are never rescaled.

Size distributions are densities, dN/dlog10(D) and dM/dlog10(D), on the
decoder's bins (0.25 decade from 10^-8.75 m; 19 bins for 02_mlpprt, the first
15 for the 05/06/11 models), wet diameter. PartMC is binned onto the same bins
by plain weighted sums (vae_partmc_utils.bin_sum), then divided by the width;
the build checks that the bins past the decoder's last hold no particle. Both
use the 15 species the encoder sees, in config.ML_SPECIES order; the diameter
is the particle's wet diameter over all of the file's species.

PartMC frozen fraction and optics are evaluated per particle (the basis the
decoder heads were trained on), with torch at one thread: threaded, the Mie
loop is ~290x slower under the interactive CPU quota.

Ported from read_wrf_data.ipynb cell 29 (populations_from_particles).
"""
import sys
import time

import numpy as np
import xarray as xr

from paperfigs import config, registry

#: The standardized ntot latent every population is decoded at (module docstring).
NTOT_LATENT = config.POP_NTOT_LATENT

#: The encoder inputs this build fills; any other input node is refused.
ENCODER_INPUTS = ('m_prtchm', 'w_prt', 'n_tot')


def particle_file(scenario):
    return config.PARTICLE_DIR / ('scenario_%d_filtered_0001_%08d.nc'
                                  % (scenario, config.PARTICLE_TIME))


def encode(vpu, comp, files, model):
    """Encode particle files with an already-loaded model: every particle, in file order.

    Returns (pops, x, z, n_file): the (aero_data, aero_state) of each file, the
    encoder input dict (decoding reuses it for its shapes), z as (n_files,
    n_latent), and each file's own total number concentration. The WRF-AeroMELD
    inputs (inputs/encode_populations.py) are encoded by this same function, so
    the runs' latents and this product's are the same numbers.
    """
    import torch

    if tuple(vpu.CHEM_SPECIES) != tuple(config.ML_SPECIES):
        raise ValueError('vae_partmc_utils.CHEM_SPECIES %s is not config.ML_SPECIES'
                         % (vpu.CHEM_SPECIES,))
    n_pop = len(files)
    pops = [vpu.aero_state_from_file(str(f), verbose=False) for f in files]
    x = vpu.create_random_data(comp['enc_pp'], n_seeds=comp['n_seeds'], n_mb=n_pop,
                               device=comp['device'], verbose=False)
    if set(x) != set(ENCODER_INPUTS):
        raise ValueError('%s\'s encoder takes %s; this build fills only %s'
                         % (model, sorted(x), list(ENCODER_INPUTS)))
    n_tot = np.zeros(n_pop)
    for i, (aero_data, aero_state) in enumerate(pops):
        n_file, w_prt, m_prtchm, _ = vpu.process_data(aero_state, aero_data,
                                                      indices=np.arange(len(aero_state)))
        if m_prtchm.shape != tuple(x['m_prtchm'].shape[2:]):
            raise ValueError('%s holds %s particle masses; the encoder takes %s'
                             % (files[i], m_prtchm.shape, tuple(x['m_prtchm'].shape[2:])))
        x['m_prtchm'][:, i] = torch.as_tensor(m_prtchm, dtype=x['m_prtchm'].dtype,
                                              device=comp['device'])
        x['w_prt'][:, i] = torch.as_tensor(w_prt[:, None], dtype=x['w_prt'].dtype,
                                           device=comp['device'])
        x['n_tot'][:, i] = n_file
        n_tot[i] = n_file
    z = vpu.run_vae_inference(comp, x)['mu']['z'].cpu().numpy().reshape(n_pop, -1)
    return pops, x, z, n_tot


def build(model):
    from extract.aeromeld import load_model

    m = registry.load().model(model)
    if m.backend == 'affx':                       # the paper's own models (extract/affx.py)
        from extract import affx
        return affx.build_populations(m)
    files = [particle_file(s) for s in config.POP_SCENARIOS]
    for f in files:
        if not f.exists():
            raise FileNotFoundError(f)
    vpu, comp = load_model(m)           # also sets torch to one thread
    scenarios = list(config.POP_SCENARIOS)
    n_pop = len(scenarios)

    # --- encode -------------------------------------------------------------
    t0 = time.perf_counter()
    pops, x, z, pmc_ntot = encode(vpu, comp, files, model)
    if z.shape[1] != m.n_latent:
        raise registry.RegistryError('%s encodes %d latents, registry says %d'
                                     % (model, z.shape[1], m.n_latent))

    # --- decode at NTOT_LATENT ------------------------------------------------
    lv = np.column_stack([z, np.full(n_pop, NTOT_LATENT)])
    rec = {k: v.cpu().numpy()[0] for k, v in
           vpu.run_vae_inference(comp, x, latent_vars=lv)['x_recon'].items()}
    print('  encoded and decoded %d populations with %s on %s in %.1f s'
          % (n_pop, model, comp['device'], time.perf_counter() - t0))

    n_bins = rec['n_prthst'].shape[-1]
    width = vpu.LOG10_BIN_WIDTH
    edges = np.log10(vpu.DIAM_LOW) + width * np.arange(n_bins + 1)
    grids = {'eps': np.asarray(vpu.EPS_HISTBINS), 'temperature': np.asarray(vpu.TEMPERATURE_C),
             'wavelength': np.asarray(vpu.WAVELENGTHS_M)}
    for node, dim in (('ccn_cdf', 'eps'), ('frznfrac_tmp', 'temperature'),
                      ('qs_pop', 'wavelength'), ('qa_pop', 'wavelength')):
        if rec[node].shape[-1] != grids[dim].size:
            raise ValueError('%s: %s has %d values, the %s grid %d'
                             % (model, node, rec[node].shape[-1], dim, grids[dim].size))
    if rec['m_chmprthst'].shape[1:] != (len(config.ML_SPECIES), n_bins) or n_bins > vpu.N_BINS:
        raise ValueError('unexpected size-distribution shapes: n_prthst %s, m_chmprthst %s'
                         % (rec['n_prthst'].shape, rec['m_chmprthst'].shape))

    # --- PartMC, per particle, on the file's own weights -------------------------
    pmc = {k: [] for k in ('ccn_cdf', 'frozen_fraction', 'scattering', 'absorption',
                           'dNdlogDp', 'dMdlogDp')}
    t0 = time.perf_counter()
    for s, (_, aero_state) in zip(scenarios, pops):
        n_hist = vpu.size_dist_partmc(aero_state)
        if n_hist[n_bins:].any():
            raise ValueError('scenario %d has particles past %s\'s %d size bins'
                             % (s, model, n_bins))
        cdf, n_tot = vpu.ccn_cdf_partmc(aero_state)
        qs, qa = vpu.optics_partmc(aero_state)
        pmc['ccn_cdf'].append(cdf / n_tot)
        pmc['frozen_fraction'].append(vpu.frznfrac_partmc(aero_state))
        pmc['scattering'].append(qs)
        pmc['absorption'].append(qa)
        pmc['dNdlogDp'].append(n_hist[:n_bins] / width)
        pmc['dMdlogDp'].append(vpu.mass_hist_partmc(aero_state)[:, :n_bins] / width)
    print('  PartMC diagnostics of %d x %d particles in %.1f s'
          % (n_pop, len(pops[0][1]), time.perf_counter() - t0))

    ml = {'ccn_cdf': rec['ccn_cdf'][:, 0], 'frozen_fraction': rec['frznfrac_tmp'][:, 0],
          'scattering': rec['qs_pop'][:, 0], 'absorption': rec['qa_pop'][:, 0],
          'dNdlogDp': rec['n_prthst'][:, 0] / width, 'dMdlogDp': rec['m_chmprthst'] / width}
    ml_basis = 'decoder output at ntot latent %g, per m^3 (training basis); totals ml_ntot' % NTOT_LATENT
    pmc_basis = ('the 1000-particle file with its own weights, per m^3; totals pmc_ntot '
                 '(~1/159 of the population sampled). Rescale extensive quantities by '
                 'ml_ntot / pmc_ntot')
    meta = {  # name: (dims after scenario, units, long name)
        'ccn_cdf': (('eps',), '1', 'CCN/CN fraction activated at s_env'),
        'frozen_fraction': (('temperature',), '1', 'frozen fraction'),
        'scattering': (('wavelength',), 'm-1', 'bulk scattering coefficient'),
        'absorption': (('wavelength',), 'm-1', 'bulk absorption coefficient'),
        'dNdlogDp': (('bin',), 'm-3', 'number size distribution dN/dlog10(D), wet diameter'),
        'dMdlogDp': (('species', 'bin'), 'kg m-3',
                     'per-species mass size distribution dM/dlog10(D), wet diameter'),
    }
    data = {}
    for name, (dims, units, long_name) in meta.items():
        for side, src, basis, who in (('ml', ml, ml_basis, 'AeroMELD reconstruction'),
                                      ('pmc', pmc, pmc_basis, 'PartMC')):
            attrs = {'units': units, 'long_name': '%s: %s' % (who, long_name)}
            if units != '1':
                attrs['basis'] = basis
            data['%s_%s' % (side, name)] = (('scenario',) + dims,
                                            np.asarray(src[name], dtype=np.float32), attrs)
    data['z'] = (('scenario', 'latent'), z.astype(np.float32),
                 {'units': '1', 'long_name': 'encoder mean latent vector (WRF ltnt_var1..n)'})
    data['ml_ntot'] = (('scenario',), rec['n_tot'].reshape(n_pop).astype(np.float32),
                       {'units': 'm-3', 'long_name': 'decoder n_tot output at ntot latent %g'
                        % NTOT_LATENT})
    data['pmc_ntot'] = (('scenario',), pmc_ntot.astype(np.float32),
                        {'units': 'm-3', 'long_name': 'sum of the particle file\'s weights '
                         '(also the n_tot handed to the encoder)'})

    ds = xr.Dataset(
        data,
        coords={
            'scenario': ('scenario', np.array(scenarios, np.int32),
                         {'long_name': 'PartMC scenario of the population'}),
            'latent': ('latent', np.arange(1, m.n_latent + 1, dtype=np.int32),
                       {'long_name': '1-based latent index (WRF ltnt_var<n>)'}),
            'eps': ('eps', grids['eps'].astype(np.float32),
                    {'units': '1', 'long_name': 'supersaturation s_env (fraction)'}),
            'temperature': ('temperature', grids['temperature'].astype(np.float32), {'units': 'degC'}),
            'wavelength': ('wavelength', grids['wavelength'].astype(np.float32), {'units': 'm'}),
            'species': ('species', list(config.ML_SPECIES), {'long_name': 'species, in the VAE order'}),
            'log10_d_lo': ('bin', edges[:-1], {'units': 'log10(m)',
                                               'long_name': 'bin lower edge, log10 wet diameter'}),
            'log10_d_hi': ('bin', edges[1:], {'units': 'log10(m)',
                                              'long_name': 'bin upper edge, log10 wet diameter'}),
        },
        attrs={'model': model, 'seed_idx': m.seed_idx, 'n_latent': m.n_latent,
               'ntot_latent': NTOT_LATENT, 'particle_time': config.PARTICLE_TIME,
               'n_particles': len(pops[0][1]), 'device': str(comp['device'])},
    )
    return ds, files + [m.checkpoint]

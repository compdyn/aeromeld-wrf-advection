"""One style for the representations in every baseline figure.

The paper's colours are taken: transport schemes red/blues, decoders a green ramp + magenta
(affine), source populations grey/red/blue, sampling points oranges. The baselines get their
own hues: sectional browns (dashed), modal purples (dash-dot). AeroMELD is the paper's
decoder 3 (recommended configuration) in its own green.

  STYLES[key] = dict(label, color, ls, marker)
"""
STYLES = {
    'aeromeld': dict(label='WRF-AeroMELD (recommended configuration)', short='AeroMELD',
                     color='#238b45', ls='-', marker='^'),
    'sec20w': dict(label='Sectional, 20 wet bins (coag paper)', short='Sectional 20 wet',
                   color='#8c510a', ls='--', marker='s'),
    'sec8': dict(label='Sectional, MOSAIC 8 dry bins', short='MOSAIC 8',
                 color='#bf812d', ls='--', marker='s'),
    'sec4': dict(label='Sectional, MOSAIC 4 dry bins', short='MOSAIC 4',
                 color='#dfc27d', ls='--', marker='s'),
    'sec20': dict(label='Sectional, 20 dry bins', short='Sectional 20 dry',
                  color='#a6611a', ls=':', marker='s'),
    'mam4': dict(label='Modal, MAM4 (all species)', short='MAM4',
                 color='#542788', ls='-.', marker='D'),
    'mam4a': dict(label='Modal, MAM4, large fresh carbon to accumulation', short='MAM4 (a)',
                  color='#8073ac', ls='-.', marker='D'),
    'mam4l': dict(label='Modal, MAM4 species', short='MAM4 species',
                  color='#b2abd2', ls='-.', marker='D'),
    # added 2026-11-14 (provisional): TOMAS-40 a darker brown than sec20w/sec8, its own marker;
    # mam4la a mid purple between mam4 and mam4a
    'tomas40': dict(label='Sectional, TOMAS-40 (GEOS-Chem)', short='TOMAS-40',
                    color='#543005', ls='--', marker='v'),
    'mam4la': dict(label='Modal, MAM4 species, large fresh carbon to accumulation', short='MAM4 species (a)',
                   color='#6a51a3', ls='-.', marker='d'),
    # added 2026-11-14 (provisional): MATRIX, modal populations (dash-dot) in BrBG's teal, away from
    # the MAM4 purples and the sectional browns
    'matrix1': dict(label='Modal, MATRIX mechanism 1 (GISS ModelE)', short='MATRIX',
                    color='#01665e', ls='-.', marker='h'),
}
#: Advected tracers per grid cell (number + species per bin or mode; AeroMELD N and N z).
TRACERS = {'aeromeld': 10, 'sec4': 64, 'sec8': 128, 'sec20': 320, 'sec20w': 320,
           'mam4': 64, 'mam4a': 64, 'mam4l': 25}
#: tomas40: 40 bins x (number + 7 dry species + water) + bulk NH4 and NO3; mam4la as mam4l
TRACERS.update(tomas40=362, mam4la=25)
#: matrix1: ModelE 2.1.2's 53 MATRIX tracers (total NO3, NH4, H2O; 16 numbers incl. SSA and SSC; 34
#: population masses)
TRACERS.update(matrix1=53)

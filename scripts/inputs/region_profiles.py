#!/usr/bin/env python3
"""Shared num_conc region profiles for the WRF-ML and WRF-PartMC input builders.

Both make_latent_ic.py (WRF-ML) and make_ic_control.py (WRF-PartMC) describe the
same scenario -- a base population everywhere plus a box "region" that swaps in a
second population. A *profile* fills the number-concentration field INSIDE that
box. Keeping the profiles here means a new bump shape is written once and both
tools pick it up.

A profile is a callable f(ni, nj, nk, base_value, region_value, sigma, vscale)
that returns an f8 array in canonical (k, j, i) order -- shape (nk, nj, ni) --
where i = west_east, j = south_north, k = bottom_top. Callers request their own
axis order from region_profile(..., order=...):

  * WRF-ML wrfinput vars are (Time, bottom_top, south_north, west_east) -> "kji"
  * WRF-PartMC num_conc is (n_source, west_east, south_north, bottom_top) -> "ijk"

``sigma`` sets the horizontal Gaussian width (cells); ``vscale`` sets the
vertical exponential scale height (levels) for profiles that decrease with
height. To add a profile, write it below and register it in PROFILES.
"""

import numpy as np

_ORDERS = {
    "kji": (0, 1, 2),   # (nk, nj, ni)  -- wrfinput layout
    "ijk": (2, 1, 0),   # (ni, nj, nk)  -- num_conc layout
}


def _profile_constant(ni, nj, nk, base_value, region_value, sigma=None, vscale=None):
    """Uniform region_value everywhere in the box."""
    return np.full((nk, nj, ni), region_value, dtype="f8")


def _profile_gaussian(ni, nj, nk, base_value, region_value, sigma=None, vscale=None):
    """Horizontal Gaussian bump that decreases exponentially with height.

    The enhancement (region_value - base_value) is a Gaussian in the horizontal
    (peak at the box center, tapering to base_value at the edges) multiplied by
    exp(-k / vscale) in the vertical, where k = 0 is the box bottom (lowest
    model level). So the field is base_value at the box edges and aloft, rising
    to region_value only at the box center on the bottom level.

    ``sigma``  : horizontal Gaussian width in cells (default min(ni, nj) / 6).
    ``vscale`` : vertical exponential scale height in levels (default nk / 2).
    """
    jj, ii = np.meshgrid(np.arange(nj), np.arange(ni), indexing="ij")
    cj, ci = (nj - 1) / 2.0, (ni - 1) / 2.0
    if sigma is None:
        sigma = max(min(ni, nj) / 6.0, 1e-9)
    hbump = np.exp(-((ii - ci) ** 2 + (jj - cj) ** 2) / (2.0 * sigma ** 2))  # (nj, ni)

    if vscale is None:
        vscale = max(nk / 2.0, 1e-9)
    vdecay = np.exp(-np.arange(nk) / vscale)                                 # (nk,), 1 at k=0

    enh = (region_value - base_value) * vdecay[:, None, None] * hbump[None, :, :]
    return (base_value + enh).astype("f8")


PROFILES = {
    "constant": _profile_constant,
    "gaussian": _profile_gaussian,
}


def region_profile(name, ni, nj, nk, base_value, region_value,
                   sigma=None, vscale=None, order="kji"):
    """Build a num_conc field for a box of size (ni, nj, nk) in the given order.

    Parameters
    ----------
    name : str
        Profile name (see PROFILES): "constant", "gaussian", ...
    ni, nj, nk : int
        Box extent in west_east (i), south_north (j), bottom_top (k).
    base_value, region_value : float
        Background value and the region peak/plateau value.
    sigma : float, optional
        Horizontal Gaussian width in cells (default min(ni, nj) / 6).
    vscale : float, optional
        Vertical exponential scale height in levels (default nk / 2); larger =
        slower decrease with height.
    order : {"kji", "ijk"}
        Axis order of the returned array (see module docstring).
    """
    try:
        field = PROFILES[name](ni, nj, nk, base_value, region_value, sigma, vscale)
    except KeyError:
        raise ValueError("unknown region profile %r (have: %s)"
                         % (name, ", ".join(sorted(PROFILES))))
    try:
        axes = _ORDERS[order]
    except KeyError:
        raise ValueError("unknown axis order %r (have: %s)"
                         % (order, ", ".join(sorted(_ORDERS))))
    return np.transpose(field, axes)

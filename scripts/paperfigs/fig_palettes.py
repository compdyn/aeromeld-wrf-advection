"""Colour palettes and shared figure style for the WRF-ML / AeroMELD figures.

Everything here is presentation-only and free of notebook state, so a figure
script can ``from fig_palettes import ...`` without dragging in the analysis
stack. Heavy imports (pyplot, cartopy) are made inside the functions that need
them, so importing this module for its colours alone stays cheap.

The paper has two unrelated families of three, and they must never be mistaken
for each other:

  * three sampling POINTS in the domain (A, B, C)
  * three latent POPULATIONS used to seed the runs (scenarios 561, 635, 812)

Both used to be drawn on near-identical hues (points on matplotlib's C0/C1/C2 =
blue/orange/green, populations on #eb6834/#2a78d6/#1baf7a = orange/blue/aqua),
so a reader flipping between figures reads a correspondence that does not exist.

The fix is a *register* split rather than six ad-hoc hues:

    points       -> a WARM one-hue ramp (orange), light -> dark for A -> B -> C
    populations  -> a COOL one-hue ramp (blue),   light -> dark for 561 -> 812

Hue tells you which family; lightness tells you which member. That choice is
forced, not aesthetic: three equally-light warm hues cannot be told apart by a
protanope or a deuteranope (orange vs red measures OKLab dE 5.6 under simulated
deuteranopia, against a floor of 8), so a "warm triad" for either family is
not available. A lightness ramp is invariant under every form of colour-vision
deficiency and survives greyscale printing, which a hue triad does not.

Which family gets which ramp is decided by the domain map, the only panel where
a family is drawn over a filled viridis field. Measured against all 256 viridis
steps, worst case per ramp step:

    warm ramp   worst background dE 22.5 / 27.1 / 21.7
    cool ramp   worst background dE 18.4 / 11.2 /  5.6  <- dark end disappears

Viridis owns the dark blue corner of the space, so the blue ramp's dark end is
5.6 from the field it sits on and point C simply vanished. The points are the
family that has to survive that field, so the points take the warm ramp; the
populations are only ever drawn on white and take the cool one.

Measured with the data-viz validator (OKLab dE x100, Machado-Oliveira-Fernandes
2009 at severity 1.0, white surface). These are properties of the ramp PAIR, so
they are unchanged by which family wears which:

    within warm ramp    adjacent steps dE 16.1 / 16.7 normal, >= 14.0 protan+deutan
    within cool ramp    adjacent steps dE 20.0 / 19.5 normal, >= 19.0 protan+deutan
    across families     worst of all 9 cross pairs dE 22.5 normal, 17.8 protan,
                        19.6 deutan, 24.5 tritan
    all six at once     worst pair dE 16.1 normal, 14.0 protan (target >= 8)

i.e. even if all six landed in one axes they would still separate; across
figures the warm/cool split is unmissable.

Caveat carried by the light end of each ramp: ``POINT_COLORS[0]`` (2.18:1) and
``POP_COLORS[0]`` (2.11:1) sit below 3:1 against white. That clears the ordinal
light-end floor (2:1) but obliges a relief channel -- keep the legend or the
direct A/B/C labels visible, and do not drop these two below ~1.2 pt line width.
``LIGHT_END_LW`` is here for that.

Orange steps are a one-hue OKLCH ramp anchored on the documented orange
#eb6834 (hue 40.6 deg), stepped to L = 0.77 / 0.615 / 0.455 with chroma
maximised in sRGB. Blue steps are the documented sequential blue ramp
(steps 250 / 450 / 650).
"""

import os

import matplotlib
import matplotlib.offsetbox

# --- where figures go, and at what resolution -------------------------------
#
# One control point for every figure the project writes. Change these two names
# here (or set the environment variables) and every save_figure() call follows;
# nothing else should call fig.savefig() directly.

#: Directory save_figure() writes into, created on demand. Relative paths
#: resolve against the working directory, so "." keeps figures next to the
#: notebook. Point it elsewhere in one line::
#:
#:     import fig_palettes; fig_palettes.FIGURE_DIR = "../graphics"
#:
#: or set WRFML_FIG_DIR in the environment before importing.
FIGURE_DIR = os.environ.get("WRFML_FIG_DIR", ".")

#: Raster resolution. The figures are PDFs, so this does NOT affect the vector
#: content -- it sets the resolution of the ``rasterized=True`` artists, which
#: is what every pcolormesh field and map feature is drawn as. 600 was already
#: the value on most call sites.
FIGURE_DPI = int(os.environ.get("WRFML_FIG_DPI", "600"))

#: Extension appended when a figure name carries none.
FIGURE_FORMAT = "pdf"


def figure_path(name):
    """Path save_figure() would write ``name`` to, extension filled in."""
    if not os.path.splitext(name)[1]:
        name = "%s.%s" % (name, FIGURE_FORMAT)
    return os.path.join(FIGURE_DIR, name)


def save_figure(fig, name, *, dpi=None, verbose=False, **kwargs):
    """Write ``fig`` to FIGURE_DIR/``name`` at FIGURE_DPI.

    ``fig`` may be None to save the current pyplot figure. ``dpi`` overrides
    FIGURE_DPI for this one call; any other keyword (bbox_inches, transparent,
    ...) is passed straight through to savefig. Returns the path written, so a
    bare call at the end of a cell echoes where the figure went.
    """
    if fig is None:
        import matplotlib.pyplot as plt
        fig = plt.gcf()

    path = figure_path(name)
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    fig.savefig(path, dpi=FIGURE_DPI if dpi is None else dpi, **kwargs)
    if verbose:
        print("Wrote", path)
    return path


# --- page geometry ----------------------------------------------------------
#
# Journal column widths, in INCHES -- matplotlib's figsize unit. Given in mm
# first because that is how the journal specifies them.

_MM_PER_INCH = 25.4

#: The text width of the AGU template (agujournal2019.cls: \textwidth 5.5 in), so that a figure
#: included at \textwidth prints at 100 % and its font sizes are the printed ones (until
#: 2026-11-13 170 mm, printed at 82 %). SINGLE_COLUMN is 0.6 of it, the width at which the
#: narrow figures are included (width=0.6\textwidth).
TWO_COLUMN = 5.5                     # 139.7 mm
SINGLE_COLUMN = 0.6 * TWO_COLUMN     # 83.8 mm
MAX_DEPTH = 228 / _MM_PER_INCH       # 228 mm


# --- shared axis ranges -----------------------------------------------------

#: Scattering and absorption axes (y-range or colour scale): one 4-decade range
#: per quantity, the same in every figure that shows it (optics_ML_vs_PR,
#: diagnostic_ML_vs_PR, diagnostic_panels_ABC, latent_populations), so each reads
#: alike everywhere. Absorption's is scattering's shifted down one decade.
#: Units: Mm^-1 m^3 kg^-1, except latent_populations (Mm^-1 at the decoded
#: number), whose values happen to fit the same ranges.
#:
#: Chosen 2026-11-09 for the main run (05_nonlinvlong.0.0). Every plotted line
#: is in range: at the points (00, 02, 04 UTC), scattering 0.73-344 and
#: absorption 0.061-16; in latent_populations, 2.1-535 and 0.20-24. In the
#: 04 UTC maps, only 0.1 % of scattering cells saturate (the plume core at
#: 400 nm, up to 1500); absorption spans 0.034-79. A single range for both
#: would need 5 decades (1e-2..1e3) to stay decade-aligned and in range, and it
#: left the absorption maps in the dark half of the colour map.
#:
#: 2026-11-15: three decades each, 1e0-1e3 and 1e-1-1e2, everywhere. For
#: the main run (twin_k16_of.1.0_sep12h, 12 h) every line is in range: at the
#: points scattering 1.8-79 and absorption 0.11-2.7; in latent_populations
#: 2.1-772 and 0.20-34. In the 12-h maps at 700 nm, 0.7 % (scattering, up to
#: 1960) and 0.2 % (absorption, up to 132) of the cells saturate; none fall
#: below.
SCATTERING_RANGE = (1e0, 1e3)
ABSORPTION_RANGE = (1e-1, 1e2)

#: Frozen fraction against temperature (latent_populations, diagnostic_panels_ABC,
#: diagnostic_ML_vs_PR): range and labelled ticks, every 5 decades.
FROZEN_FRACTION_RANGE = (1e-15, 1.0)
FROZEN_FRACTION_TICKS = (1e-15, 1e-10, 1e-5, 1.0)


def decade_ticks(lo, hi):
    """10^n for every n from lo to hi (both powers of ten): log-axis major ticks."""
    import numpy as np
    a, b = (int(round(np.log10(v))) for v in (lo, hi))
    return 10.0 ** np.arange(a, b + 1)


# --- shared axis labels -----------------------------------------------------
#
# One spelling per quantity across the figures, with the paper's names and
# symbols (s, T, lambda). Fractions carry no "(1)". The CCN spectrum is the
# activated fraction against supersaturation (until 2026-11-14 "CCN/CN fraction"
# against s_env).

CCN_LABEL = 'Activated fraction'
FROZEN_FRACTION_LABEL = 'Frozen fraction'
SUPERSATURATION_LABEL = r'Supersaturation $s$ (%)'
TEMPERATURE_LABEL = r'Temperature $T$ ($^{\circ}$C)'
WAVELENGTH_LABEL = r'Wavelength $\lambda$ (nm)'
#: Fits the narrowest optics panel (diagnostic_panels_ABC; 400/600/800/1000 ran
#: together there) and marks the maps' and violins' 400 and 1000 nm.
WAVELENGTH_TICKS = (400, 700, 1000)
#: The size distributions' diameter: by dry diameter since the dry rebuild (2026-11-16,
#: config.SIZE_BASIS 'dry', then 'drydiam' with water in the masses; the decoder is trained on
#: dry-diameter distributions), before by wet diameter.
from . import config as _config  # noqa: E402
DIAMETER_LABEL = (r'Dry diameter ($\mathrm{\mu}$m)' if _config.SIZE_BASIS != 'wet'
                  else r'Wet diameter ($\mathrm{\mu}$m)')
TIME_LABEL = 'Time (h)'


def use_paper_style():
    """Apply the rcParams every figure in the paper shares.

    Deliberately a call rather than an import side effect: importing a module
    should not silently reach into another module's global state.
    """
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8})


# --- the two families -------------------------------------------------------

#: Sampling points A, B, C -- warm ramp, light -> dark. Warm because these are
#: the marks that land on the viridis-filled domain map (see module docstring).
POINT_COLORS = ("#fd946e", "#db520f", "#923406")

#: Source populations (background, urban, emission), in config.POP_LABELS order:
#: gray, red, blue, the colours that mix into the composition maps of Figure 2
#: (plot/domain_evolution.py, which lightens the background so that the plumes
#: stand out). Until 2026-11-12 a cool ramp ("#86b6ef", "#2a78d6", "#104281"),
#: which made blue mean emission in Figure 2 and any population elsewhere.
POP_COLORS = ("#636363", "#d7301f", "#2166ac")

#: The same populations as area fills on maps (Figure 2's composition maps, the
#: schematic's grid): the background lightened, so that the plumes stand out.
POP_MAP_COLORS = ("#e6e6e6",) + POP_COLORS[1:]

#: Minimum line width for the light end of either ramp (contrast relief).
LIGHT_END_LW = 1.2

# --- neutrals for things that are annotation, not series --------------------
#
# Anything that is not a point and not a population must stay out of both hue
# families, or it re-introduces the confusion from the other side. The emission
# source boxes on the domain map are the case in point: drawn in tab:red and
# tab:orange they read as population colours. Draw them neutral and let line
# STYLE carry which source it is.

#: Ink for annotations on a WHITE panel. Do not use this on a filled map --
#: apply_map_features already draws coastlines, borders and state lines in black,
#: so a black annotation there stops reading as an annotation and starts reading
#: as geography. Filled maps take MAP_ANNOTATION_COLOR below.
ANNOTATION_COLOR = "#0b0b0b"
ANNOTATION_STYLES = ("-", "--", ":")

# --- filled maps are a different surface ------------------------------------
#
# Everything above is validated against WHITE. A pcolormesh panel is not white:
# viridis runs dark blue -> green -> yellow, so it swallows the dark end of the
# cool ramp, and the basemap has already spent black on geography. No single
# colour survives a full-range colormap, so on a filled map the marks keep their
# family colour and gain an achromatic HALO -- that is the relief channel the
# sub-3:1 steps were always going to need, made explicit.

#: Annotation on a filled map: white core, dark casing (see halo()). Achromatic,
#: so it collides with neither hue family, and thicker + cased so it never reads
#: as a coastline.
MAP_ANNOTATION_COLOR = "#ffffff"
MAP_ANNOTATION_STROKE = "#0b0b0b"


def halo(width=2.0, color="#ffffff"):
    """path_effects that keep a mark legible over any colormap value.

    Use on text and lines drawn over a filled field::

        ax.annotate(..., color=point_color(i), path_effects=halo())
        ax.plot(..., color=MAP_ANNOTATION_COLOR,
                path_effects=halo(color=MAP_ANNOTATION_STROKE))
    """
    import matplotlib.patheffects as pe
    return [pe.withStroke(linewidth=width, foreground=color)]


def halo_for(color, width=2.0):
    """halo() whose colour is chosen to contrast with ``color`` itself.

    A fixed white halo works for the dark end of a ramp and washes out the pale
    end -- point A's #86b6ef against white is only 2.1:1, so a white-haloed A is
    a pale smudge. Pick the halo per step instead: pale marks get an ink halo,
    dark marks get a white one. The mark then carries its own contrast and no
    longer depends on what the colormap put behind it.
    """
    r, g, b = (int(color.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4))

    def _lin(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    lum = 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)
    # Relative luminance 0.18 is the crossover where white and black backing
    # give equal WCAG contrast.
    return halo(width, MAP_ANNOTATION_STROKE if lum > 0.18 else MAP_ANNOTATION_COLOR)


# --- a third family: the ML models of the cross-model figure ------------------
#
# Same register idea as above: hue says the decoder FAMILY, lightness the member.
#
#   MLP decoders     achromatic   (ink-dark, mid grey)
#   affine decoders  magenta ramp, light -> dark as the exponent p rises
#                    (0.1, 0.7, 1); OKLCH hue 350, L = 0.76 / 0.62 / 0.48,
#                    chroma maximised in sRGB
#
# Why these: no chromatic hue clears BOTH existing families under simulated
# CVD -- green collapses onto the orange point ramp (protan/deutan dE 1.6-2.0),
# violet onto the blue population ramp. Achromatic is the one register distinct
# from both under every simulation (dE >= 10), so the main family takes it.
# Magenta clears both families in normal vision (dE >= 13.4 to the points,
# >= 20.6 to the populations) but falls to 2.7-4.6 from the orange ramp under
# protan/deutan; the two never share a figure, and the marker shape (circle and
# diamond for MLP, triangles and square for affine) carries identity on its own.
#
# Measured 2026-11-08 with the data-viz validator's Python twin (OKLab dE x100,
# Machado 2009 at severity 1.0, surface #fcfcfb), in the figure's dodge order
# (the adjacent pairlist: every model keeps its own offset in each slot, like
# grouped bars): CVD worst adjacent 11.6, normal-vision worst adjacent 15.1,
# both passing. Off-gate by design: the greys sit outside the hue band / below
# the chroma floor (achromatic on purpose); #ff7fc0 is 2.26:1 against white,
# above the 2:1 light-end floor, so the legend must stay and the markers carry
# a dark edge. All-pairs CVD is 5.4 (#eb009b vs #7a7a7a): legal only because
# the shapes differ.
MLP_COLORS = ('#262626', '#7a7a7a')
AFFINE_COLORS = ('#ff7fc0', '#eb009b', '#a7006d')

#: Model key -> (colour, marker). Colour follows the model, never its position
#: in a set, so dropping a model from a figure repaints nothing.
MODEL_STYLES = {
    '05_nonlinvlong.0.0': (MLP_COLORS[0], 'o'),
    '06_nonlinregvlong.3.0': (MLP_COLORS[1], 'D'),
    '05_linuvlong.0.0': (AFFINE_COLORS[0], '^'),
    '11_linpvlong.5.0': (AFFINE_COLORS[1], 'v'),
    '05_linxvlong10.1.0': (AFFINE_COLORS[2], 's'),
}
#: The paper's own models (training/, from 2026-11-11), by model NAME (every seed
#: of a model looks the same): the nonlinear twins as a green ramp ordered by how
#: much mixture training they had (none, pairs, up to 16, up to 32; light to
#: dark; a grey ramp until 2026-11-13, when the lightest blended with the grid), the affine
#: AX-rel1 in magenta.
#: Each also has its own marker, so identity never rests on the grey level alone.
AFFX_STYLES = {
    'twin_none': ('#74c476', 'v'),
    'twin_pairs': ('#41ab5d', 'D'),
    'twin_k16': ('#238b45', '^'),
    'twin_k32': ('#00441b', 's'),
    'ax_rel1': (AFFINE_COLORS[1], 'o'),
    'ax_k16': (AFFINE_COLORS[2], 'P'),
}
#: Markers for a model without an entry (ad hoc --runs); drawn in neutral grey.
#: None of them is a marker MODEL_STYLES uses.
SPARE_MARKERS = ('P', 'X', 'h', '*', '<', '>', 'p')


def model_style(model_key, i_spare=0):
    """(colour, marker) of a model; unlisted models get grey and a spare marker."""
    if model_key in MODEL_STYLES:
        return MODEL_STYLES[model_key]
    name = model_key.split('.')[0]
    for suffix in ('_of', '_dry', '_dd'):                         # retrained (optics fixed, dry, dry diameter): same look
        name = name[:-len(suffix)] if name.endswith(suffix) else name
    if name in AFFX_STYLES:
        return AFFX_STYLES[name]
    return '#a6a6a6', SPARE_MARKERS[i_spare % len(SPARE_MARKERS)]


def point_color(i_loc):
    """Colour for sampling point ``i_loc`` (0-based)."""
    return POINT_COLORS[i_loc % len(POINT_COLORS)]


def pop_color(i_pop):
    """Colour for latent population ``i_pop`` (0-based, in listed order)."""
    return POP_COLORS[i_pop % len(POP_COLORS)]


def point_lw(i_loc, base=1.0):
    """Line width for a point, floored so the pale first step stays visible."""
    return max(base, LIGHT_END_LW) if i_loc % len(POINT_COLORS) == 0 else base


def pop_lw(i_pop, base=1.0):
    """Line width for a population, floored so the pale first step stays visible."""
    return max(base, LIGHT_END_LW) if i_pop % len(POP_COLORS) == 0 else base


# --- per-species colours for stacked composition panels ---------------------
#
# A third family, and a different KIND of family: these key chemical species in
# the stacked mass panels, never a point or a population. They are categorical
# (tab20 hues), not a ramp, because species have no natural order.
#
# Converted from a tab20-based YAML scheme found elsewhere. Each name keys a
# tab20 hue; "bright" is that hue's paired light shade. The five SOA-like
# species (API1, ARO1, ARO2, ALK1, OLE1) are all the SAME light green,
# distinguished only by alpha (1.00 down to 0.32), so they read as one organic
# family in the stacked panels rather than five unrelated hues.
#
# NOTE: Cl and MOC both land on tab20 "purple" -- a real collision within
# ML_SPECIES, inherited from the source palette. Flagging it rather than
# silently changing your palette; let me know if you want MOC moved off it.

#: 20 RGB triples: 10 hues x (dark, bright).
_TAB20 = matplotlib.colormaps["tab20"].colors
_TAB20_HUES = ['blue', 'orange', 'green', 'red', 'purple',
               'brown', 'pink', 'gray', 'olive', 'cyan']
TAB20_COLORS = {}
for _i, _hue in enumerate(_TAB20_HUES):
    TAB20_COLORS[_hue] = _TAB20[2 * _i]
    TAB20_COLORS[_hue + 'bright'] = _TAB20[2 * _i + 1]

_GREEN_ALPHAS = [1.00, 0.83, 0.66, 0.49, 0.32]  # API1, ARO1, ARO2, ALK1, OLE1
GREEN_FAMILY = [(*TAB20_COLORS['greenbright'], _a) for _a in _GREEN_ALPHAS]

SPECIES_COLORS = {
    'SO4':  TAB20_COLORS['red'],
    'NO3':  TAB20_COLORS['blue'],
    'Cl':   TAB20_COLORS['purple'],
    'NH4':  TAB20_COLORS['orange'],
    'API1': GREEN_FAMILY[0],
    'ARO1': GREEN_FAMILY[1],
    'ARO2': GREEN_FAMILY[2],
    'ALK1': GREEN_FAMILY[3],
    'OLE1': GREEN_FAMILY[4],
    'Na':   TAB20_COLORS['olive'],
    'OIN':  TAB20_COLORS['brown'],
    'OC':   TAB20_COLORS['green'],
    'BC':   'black',
    'MOC':  TAB20_COLORS['purple'],
    'H2O':  TAB20_COLORS['bluebright'],
    'OGN':  TAB20_COLORS['green'],
    'OA':   TAB20_COLORS['green'],
    'MSA':  TAB20_COLORS['redbright'],
    'API2': TAB20_COLORS['orangebright'],
    'LIM1': TAB20_COLORS['purplebright'],
    'LIM2': TAB20_COLORS['cyanbright'],
    'CO3':  TAB20_COLORS['brownbright'],
    'Ca':   TAB20_COLORS['gray'],
}
SPECIES_COLORS['Dust'] = SPECIES_COLORS['OIN']
SPECIES_COLORS['POA'] = SPECIES_COLORS['OC']

#: Species whose names are chemical formulas, with their subscripts. The lumped
#: organic species (ARO1, API1, ...) keep their digits: those are names, not formulas.
_SPECIES_TEX = {'SO4': r'SO$_4$', 'NO3': r'NO$_3$', 'NH4': r'NH$_4$', 'H2O': r'H$_2$O',
                'CO3': r'CO$_3$'}


def species_label(name):
    """A species' name for a legend, chemical formulas subscripted ('SO4' -> 'SO$_4$')."""
    return _SPECIES_TEX.get(name, name)


# --- shared panel furniture -------------------------------------------------

#: Named anchors for create_label_box. Both sit the box ON the axes' top edge;
#: they differ only in which end it is flush with. matplotlib loc codes: 4 is
#: lower-right of the box, 2 is upper-left.
LABEL_BOX_CORNERS = {
    "above right": (4, (1.0, 1.0)),
    "above left": (2, (0.0, 1.0)),
}


def create_label_box(axes, string, corner="above right"):
    """Boxed panel tag riding on the top edge of ``axes``.

    ``corner`` selects which end it is flush with, per LABEL_BOX_CORNERS. The
    "above left" variant used to be a separate copy of this function named
    create_label_inside; it is a parameter now so the two cannot drift.
    """
    try:
        loc, bbox_to_anchor = LABEL_BOX_CORNERS[corner]
    except KeyError:
        raise ValueError("corner must be one of %s, got %r"
                         % (sorted(LABEL_BOX_CORNERS), corner)) from None

    box1 = matplotlib.offsetbox.TextArea(string)
    anchored_box = matplotlib.offsetbox.AnchoredOffsetbox(
        loc=loc, child=box1, pad=0.2, frameon=True,
        bbox_to_anchor=bbox_to_anchor, bbox_transform=axes.transAxes,
        borderpad=0.0)
    axes.add_artist(anchored_box)
    return axes


def _clipped_to_view(axes, feature, pad_deg=2.0):
    """`feature` (a lon/lat NaturalEarthFeature) cut to a lon/lat box `pad_deg`
    beyond the axes' current view, with the feature's own styling. Call after
    the view is set (figures.plot_field pins it)."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import shapely.geometry as sgeom

    lon0, lon1, lat0, lat1 = axes.get_extent(ccrs.PlateCarree())
    box = sgeom.box(lon0 - pad_deg, lat0 - pad_deg, lon1 + pad_deg, lat1 + pad_deg)
    geoms = [g.intersection(box) for g in feature.geometries() if g.intersects(box)]
    return cfeature.ShapelyFeature([g for g in geoms if not g.is_empty], ccrs.PlateCarree(),
                                   **feature.kwargs)


def apply_map_features(axes):
    """Coastlines, borders, state lines, lakes and land/ocean fill on a GeoAxes.

    Everything is drawn in black and rasterized -- see the ANNOTATION_COLOR
    note above for why a black annotation on top of this stops reading as an
    annotation.
    """
    import cartopy
    import cartopy.feature as cfeature

    axes.coastlines(resolution='50m', color='black', linewidth=1, rasterized=True)
    states_provinces = cartopy.feature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='50m',
        facecolor='none')
    axes.add_feature(states_provinces, linestyle=':', edgecolor='black', rasterized=True)

    axes.add_feature(cfeature.BORDERS, linestyle='-', edgecolor='black', rasterized=True)
    axes.coastlines(rasterized=True)
    # LAND and OCEAN clipped to the view first: projecting the global 110m
    # polygons into the WRF Lambert grid took ~48 s per process, for fills that
    # lie under the data (zorder -1) and mostly outside the view anyway.
    for feature in (cfeature.LAND, cfeature.OCEAN):
        axes.add_feature(_clipped_to_view(axes, feature), rasterized=True)

    lakes = cfeature.NaturalEarthFeature(
        category='physical',
        name='lakes',
        scale='50m',  # Options: '10m', '50m', '110m'
        facecolor='None',
    )
    axes.add_feature(lakes, rasterized=True)
    return

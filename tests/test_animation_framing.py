"""Frame geometry for animations.

Two properties have to hold simultaneously, and they pull against each other:

1. **Every frame in one movie must have identical pixel dimensions.** h264 rejects a stream
   whose frame size changes, which is why ``bbox_inches="tight"`` is not used when saving
   frames -- it crops to content, so dimensions drift with the data.
2. **The domain should fill the frame.** Satisfying (1) with a single hardcoded ``figsize``
   drew a ~2:1 global map on a 7x5 canvas, leaving ~45% of every frame as white margin.

``_domain_figsize`` resolves both by deriving one canvas aspect from the projected extent of
the domain, up front, and reusing it for every frame. These tests pin that behaviour: the
aspect tracks the projection, the height lands on an even pixel count, and anything it cannot
measure degrades to the previous fixed canvas instead of raising mid-render.
"""

import cartopy.crs as ccrs
import numpy as np
import pytest

from marEx.plotX.animation import (
    _DEFAULT_FIG_ASPECT,
    _FIG_WIDTH_INCHES,
    _FRAME_DPI,
    _MAX_FRAME_PIXELS,
    _MIN_FIG_ASPECT,
    _domain_figsize,
    _even_height,
)

GLOBAL_LON = np.arange(-179.5, 180.0, 1.0)
GLOBAL_LAT = np.arange(-89.5, 90.0, 1.0)
REGIONAL_LON = np.arange(-80.0, 20.0, 0.25)
REGIONAL_LAT = np.arange(-10.0, 60.0, 0.25)

DEFAULT_SIZE = (_FIG_WIDTH_INCHES, _even_height(_FIG_WIDTH_INCHES * _DEFAULT_FIG_ASPECT))


def _pixels(figsize):
    return round(figsize[0] * _FRAME_DPI), round(figsize[1] * _FRAME_DPI)


class TestEvenHeight:
    """h264 needs even dimensions; the helper must never hand back an odd pixel height."""

    @pytest.mark.parametrize("inches", [0.001, 1.0, 2.8655, 3.57, 5.0, 7.13, 12.0])
    def test_height_lands_on_even_pixels(self, inches):
        assert round(_even_height(inches) * _FRAME_DPI) % 2 == 0

    def test_rounding_stays_close_to_the_request(self):
        # Never move by more than one pixel-pair, or the aspect correction is undone.
        for inches in (1.0, 2.8655, 3.57, 7.13):
            assert abs(_even_height(inches) - inches) <= 2.0 / _FRAME_DPI

    def test_degenerate_height_is_clamped_positive(self):
        assert _even_height(0.0) > 0
        assert _even_height(-5.0) > 0


class TestDomainFigsize:
    """The canvas aspect must follow the projected domain, not the raw lon/lat box."""

    def test_global_robinson_is_wide_not_seven_by_five(self):
        """The actual regression: a global map used to be drawn on a 7x5 canvas."""
        figsize = _domain_figsize(ccrs.Robinson(), GLOBAL_LON, GLOBAL_LAT, show_colorbar=False)
        aspect = figsize[1] / figsize[0]

        assert aspect < _DEFAULT_FIG_ASPECT, "global domain should be wider than the old fixed canvas"
        # Robinson draws the globe at roughly 2:1, so the canvas should be near that.
        assert 0.40 < aspect < 0.62

    def test_width_is_held_constant(self):
        """Only the height varies; a fixed width keeps output resolution predictable."""
        for proj, lon, lat in [
            (ccrs.Robinson(), GLOBAL_LON, GLOBAL_LAT),
            (ccrs.PlateCarree(), REGIONAL_LON, REGIONAL_LAT),
        ]:
            for colorbar in (True, False):
                assert _domain_figsize(proj, lon, lat, colorbar)[0] == _FIG_WIDTH_INCHES

    def test_taller_domain_gives_taller_canvas(self):
        """Monotonicity: the canvas has to track the domain rather than a constant."""
        wide = _domain_figsize(ccrs.PlateCarree(), np.arange(-180.0, 180.0, 1.0), np.arange(-10.0, 10.0, 1.0), False)
        tall = _domain_figsize(ccrs.PlateCarree(), np.arange(-20.0, 20.0, 1.0), np.arange(-80.0, 80.0, 1.0), False)
        assert tall[1] > wide[1]

    def test_platecarree_aspect_tracks_the_lonlat_box(self):
        """PlateCarree is linear, so the projected aspect is the lon/lat aspect."""
        lon = np.arange(-100.0, 100.0, 1.0)  # 199 deg wide
        lat = np.arange(-25.0, 25.0, 1.0)  # 49 deg tall
        figsize = _domain_figsize(ccrs.PlateCarree(), lon, lat, show_colorbar=False)
        # height/width of the canvas should be proportional to the domain ratio, scaled by
        # the fraction of the figure the axes box occupies.
        axes_ratio = (figsize[1] * 0.77) / (figsize[0] * 0.775)
        assert axes_ratio == pytest.approx(np.ptp(lat) / np.ptp(lon), rel=0.02)

    def test_colorbar_narrows_the_map_so_the_canvas_is_shorter(self):
        """A colorbar takes width from the map; at equal aspect the canvas must shrink."""
        with_cb = _domain_figsize(ccrs.Robinson(), GLOBAL_LON, GLOBAL_LAT, show_colorbar=True)
        without_cb = _domain_figsize(ccrs.Robinson(), GLOBAL_LON, GLOBAL_LAT, show_colorbar=False)
        assert with_cb[1] < without_cb[1]

    @pytest.mark.parametrize(
        "proj",
        [ccrs.Robinson(), ccrs.PlateCarree(), ccrs.Mollweide(), ccrs.EqualEarth()],
    )
    def test_aspect_stays_above_the_floor(self, proj):
        for lon, lat in [(GLOBAL_LON, GLOBAL_LAT), (REGIONAL_LON, REGIONAL_LAT)]:
            figsize = _domain_figsize(proj, lon, lat, show_colorbar=False)
            assert figsize[1] / figsize[0] >= _MIN_FIG_ASPECT

    def test_result_is_deterministic(self):
        """Same domain must give the same canvas -- this is what keeps frames uniform."""
        a = _domain_figsize(ccrs.Robinson(), GLOBAL_LON, GLOBAL_LAT, show_colorbar=True)
        b = _domain_figsize(ccrs.Robinson(), GLOBAL_LON, GLOBAL_LAT, show_colorbar=True)
        assert a == b

    def test_every_result_has_even_pixel_height(self):
        for proj in (ccrs.Robinson(), ccrs.PlateCarree(), ccrs.Mollweide()):
            for lon, lat in [(GLOBAL_LON, GLOBAL_LAT), (REGIONAL_LON, REGIONAL_LAT)]:
                for colorbar in (True, False):
                    assert _pixels(_domain_figsize(proj, lon, lat, colorbar))[1] % 2 == 0


class TestDomainFigsizeDegradesQuietly:
    """Sizing must never be the reason a render dies; fall back to the previous canvas."""

    @pytest.mark.parametrize(
        "lon,lat",
        [
            (None, None),
            (GLOBAL_LON, None),
            (None, GLOBAL_LAT),
            (np.array([]), np.array([])),
            (np.array([10.0]), np.array([10.0])),  # single point: zero extent
            (np.array([np.nan, np.nan]), np.array([np.nan, np.nan])),
        ],
    )
    def test_degenerate_input_returns_the_default_canvas(self, lon, lat):
        assert _domain_figsize(ccrs.Robinson(), lon, lat, show_colorbar=True) == DEFAULT_SIZE

    def test_a_projection_that_raises_returns_the_default_canvas(self):
        class ExplodingProjection:
            def transform_points(self, *args, **kwargs):
                raise RuntimeError("projection blew up")

        assert _domain_figsize(ExplodingProjection(), GLOBAL_LON, GLOBAL_LAT, show_colorbar=False) == DEFAULT_SIZE

    def test_a_projection_returning_non_finite_points_returns_the_default_canvas(self):
        class NonFiniteProjection:
            def transform_points(self, src, x, y):
                return np.full((x.size, 3), np.inf)

        assert _domain_figsize(NonFiniteProjection(), GLOBAL_LON, GLOBAL_LAT, show_colorbar=False) == DEFAULT_SIZE


class TestFrameFitsWithinH264Limits:
    """H.264 levels cap a single dimension at 4096 px.

    libx264 encodes an oversized frame without complaining, so the failure surfaces at
    playback in a browser -- which is precisely where these movies are served. Assert the
    pixel ceiling directly: asserting the *aspect* against its own constant is a tautology
    that would have let a 2100x4200 frame through.
    """

    # 10 deg lon by 80 deg lat is an ordinary regional strip (a boundary current), and it
    # projects to aspect ~8 -- comfortably past the ceiling before clamping.
    TALL_NARROW = (np.arange(-5.0, 5.0, 0.1), np.arange(-40.0, 40.0, 0.1))

    @pytest.mark.parametrize("proj", [ccrs.PlateCarree(), ccrs.Mollweide(), ccrs.Robinson()])
    @pytest.mark.parametrize("colorbar", [True, False])
    def test_tall_narrow_domain_stays_within_the_pixel_cap(self, proj, colorbar):
        width_px, height_px = _pixels(_domain_figsize(proj, *self.TALL_NARROW, show_colorbar=colorbar))
        assert width_px <= _MAX_FRAME_PIXELS
        assert height_px <= _MAX_FRAME_PIXELS

    def test_the_cap_actually_binds_for_this_domain(self):
        """Guards the guard: if this stops clamping, the test above proves nothing."""
        _, height_px = _pixels(_domain_figsize(ccrs.PlateCarree(), *self.TALL_NARROW, show_colorbar=False))
        assert height_px == _MAX_FRAME_PIXELS

    def test_clamped_height_is_still_even(self):
        _, height_px = _pixels(_domain_figsize(ccrs.PlateCarree(), *self.TALL_NARROW, show_colorbar=False))
        assert height_px % 2 == 0

    def test_every_realistic_domain_is_within_the_cap(self):
        for proj in (ccrs.Robinson(), ccrs.PlateCarree(), ccrs.Mollweide(), ccrs.EqualEarth()):
            for lon, lat in [(GLOBAL_LON, GLOBAL_LAT), (REGIONAL_LON, REGIONAL_LAT), self.TALL_NARROW]:
                for colorbar in (True, False):
                    for dim in _pixels(_domain_figsize(proj, lon, lat, colorbar)):
                        assert dim <= _MAX_FRAME_PIXELS

"""Map rendering: grid detection, plotter dispatch, colour handling, titles, ID plots and unstructured grid files.

Rendering rows draw on a real Agg figure and read the property off the artist, so a configuration field that
stops reaching the picture fails here. Global state (the ``specify_grid`` settings, the unstructured grid
cache, open figures) is restored after every test.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm

import marEx
from marEx.exceptions import ConfigurationError, DataValidationError, VisualisationError
from marEx.plotX import PlotConfig, PlotXAccessor, _detect_grid_type, specify_grid
from marEx.plotX.base import PlotterBase
from marEx.plotX.gridded import GriddedPlotter
from marEx.plotX.mesh import pcolormesh_seam_safe
from marEx.plotX.unstructured import _GRID_CACHE, UnstructuredPlotter, _load_ckdtree, _load_triangulation, clear_cache

from ..conftest import assert_no_failures, check

DEFAULT_MAP = {"time": "time", "y": "lat", "x": "lon"}
CELL_DIMS = {"time": "time", "x": "cell"}
CELL_COORDS = {"time": "time", "x": "lon", "y": "lat"}


@pytest.fixture(autouse=True)
def _isolated_plot_state(monkeypatch):
    """No test sees, or leaves behind, a grid specification, cached grid files or an open figure."""
    for name in ("_grid_type", "_fpath_tgrid", "_fpath_ckdtree"):
        monkeypatch.setattr(marEx.plotX, name, None)
    clear_cache()
    yield
    clear_cache()
    plt.close("all")


class _Plotter(PlotterBase):
    """A concrete plotter that draws nothing."""

    def plot(self, ax, cmap="viridis", clim=None, norm=None):
        return ax, None


def _gridded(nt=3, nlat=4, nlon=5, lat=(-20, 20), lon=(-25, 25), seed=0):
    return xr.DataArray(
        np.random.default_rng(seed).random((nt, nlat, nlon)),
        dims=["time", "lat", "lon"],
        coords={"time": range(nt), "lat": np.linspace(*lat, nlat), "lon": np.linspace(*lon, nlon)},
    )


def _cells(nt=3, ncell=100, seed=0, cell="cell"):
    rng = np.random.default_rng(seed)
    return xr.DataArray(
        rng.random((nt, ncell)),
        dims=["time", cell],
        coords={
            "time": range(nt),
            "lat": (cell, rng.uniform(-90, 90, ncell)),
            "lon": (cell, rng.uniform(-180, 180, ncell)),
        },
    )


def _ids_on_five_degree_grid(values=None, n_ids=None, lon_start=2.5):
    lon = np.arange(lon_start, 360, 5.0)
    lat = np.arange(-87.5, 90, 5.0)
    if values is None:
        values = np.zeros((lat.size, lon.size), dtype=int)
        values.flat[:n_ids] = np.arange(1, n_ids + 1)
    return xr.DataArray(values, dims=("lat", "lon"), coords={"lat": lat, "lon": lon})


def test_accessor_dispatches_on_grid_type():
    """The accessor picks the plotter class from the data, carries dims, coords and data, and rejects unknown names."""
    failures = []

    def plotter_has(plotter, cls, dims, coords, data):
        assert type(plotter) is cls, f"got {type(plotter).__name__}, expected {cls.__name__}"
        assert plotter.dimensions == dims, f"dimensions {plotter.dimensions}, expected {dims}"
        assert plotter.coordinates == coords, f"coordinates {plotter.coordinates}, expected {coords}"
        assert plotter.da is data, "the plotter holds a different array"

    def default_gridded():
        data = _gridded()
        plotter_has(data.plotX(), GriddedPlotter, DEFAULT_MAP, DEFAULT_MAP, data)
        plotter_has(PlotXAccessor(data)(), GriddedPlotter, DEFAULT_MAP, DEFAULT_MAP, data)

    def default_unstructured():
        data = _cells()
        plotter_has(data.plotX(), UnstructuredPlotter, CELL_DIMS, DEFAULT_MAP, data)

    def custom_unstructured_dims():
        data = _cells()
        plotter_has(data.plotX(dimensions=CELL_DIMS, coordinates=CELL_COORDS), UnstructuredPlotter, CELL_DIMS, CELL_COORDS, data)

    def custom_names_gridded():
        data = xr.DataArray(
            np.zeros((3, 4, 5)),
            dims=["t", "latitude", "longitude"],
            coords={"t": range(3), "latitude": np.linspace(-20, 20, 4), "longitude": np.linspace(-25, 25, 5)},
        )
        names = {"time": "t", "x": "longitude", "y": "latitude"}
        plotter_has(PlotXAccessor(data)(dimensions=names, coordinates=names), GriddedPlotter, names, names, data)

    def custom_names_unstructured():
        data = xr.DataArray(
            np.zeros((3, 100)),
            dims=["t", "cells"],
            coords={"t": range(3), "x_coord": ("cells", np.zeros(100)), "y_coord": ("cells", np.zeros(100))},
        )
        dims = {"time": "t", "x": "cells"}
        coords = {"time": "t", "x": "x_coord", "y": "y_coord"}
        plotter_has(PlotXAccessor(data)(dimensions=dims, coordinates=coords), UnstructuredPlotter, dims, coords, data)

    def detection():
        assert _detect_grid_type(_gridded()) == "gridded"
        assert _detect_grid_type(_cells()) == "unstructured"
        coords = {"time": range(2), "lat": range(3), "lon": range(4)}
        bare = xr.DataArray(np.zeros((2, 3, 4)), dims=["time", "lat", "lon"], coords=coords)
        assert _detect_grid_type(bare) == "gridded", "ambiguous coordinates must fall back to gridded"
        data = xr.DataArray(
            np.zeros((3, 4, 5)),
            dims=["t", "spatial_y", "spatial_x"],
            coords={"t": range(3), "spatial_y": np.linspace(-20, 20, 4), "spatial_x": np.linspace(-25, 25, 5)},
        )
        with_y = {"time": "t", "x": "spatial_x", "y": "spatial_y"}
        assert _detect_grid_type(data, dimensions=with_y) == "gridded"
        assert _detect_grid_type(data, dimensions={"time": "t", "x": "spatial_x"}) == "unstructured"

    def base_and_unstructured_plotters_initialise():
        data = _gridded()
        default = _Plotter(data)
        assert default.da is data and default.dimensions == DEFAULT_MAP and default.coordinates == DEFAULT_MAP
        assert default._land is not None and default._coastlines is not None
        custom = {"time": "time", "x": "lon", "y": "lat"}
        explicit = _Plotter(data, dimensions=custom, coordinates=custom)
        assert explicit.dimensions == custom and explicit.coordinates == custom
        cells = _cells()
        plotter = UnstructuredPlotter(cells, dimensions=CELL_DIMS, coordinates=CELL_COORDS)
        assert plotter.da is cells and plotter.dimensions == CELL_DIMS and plotter.coordinates == CELL_COORDS

    def unknown_names_are_rejected():
        data = _gridded()
        accessor = PlotXAccessor(data)
        for kwargs in (
            {"dimensions": {"time": "invalid_time", "x": "invalid_x", "y": "invalid_y"}},
            {"coordinates": {"time": "invalid_time", "x": "invalid_x", "y": "invalid_y"}},
            {"dimensions": {"time": "time", "x": "invalid_x", "y": "lat"}},
        ):
            with pytest.raises(VisualisationError):
                accessor(**kwargs)

    def accessor_is_registered():
        data = _gridded()
        assert isinstance(data.plotX, PlotXAccessor)
        for method in ("single_plot", "multi_plot", "animate"):
            assert callable(getattr(data.plotX, method, None)), f"the accessor has no callable {method}"

    check(failures, "default gridded", default_gridded)
    check(failures, "default unstructured", default_unstructured)
    check(failures, "custom unstructured dims", custom_unstructured_dims)
    check(failures, "custom names, gridded", custom_names_gridded)
    check(failures, "custom names, unstructured", custom_names_unstructured)
    check(failures, "detection", detection)
    check(failures, "plotter initialisation", base_and_unstructured_plotters_initialise)
    check(failures, "unknown names", unknown_names_are_rejected)
    check(failures, "accessor registration", accessor_is_registered)
    assert_no_failures(failures)


def test_specify_grid_overrides_detection(tmp_path):
    """A global ``specify_grid`` decides the plotter class (case-insensitively); a plotter's own ``specify_grid`` sets its paths."""
    gridded = _gridded()
    cells = _cells()
    failures = []

    def global_type_decides_the_class():
        # Each value disagrees with what the data's structure alone would give, so only the setting can decide.
        for value, cls, data, dims, coords in [
            ("gridded", GriddedPlotter, cells, CELL_DIMS, CELL_COORDS),
            ("unstructured", UnstructuredPlotter, gridded, None, None),
            ("GRIDDED", GriddedPlotter, cells, CELL_DIMS, CELL_COORDS),
            ("UNSTRUCTURED", UnstructuredPlotter, gridded, None, None),
        ]:
            specify_grid(grid_type=value)
            with pytest.warns(UserWarning, match="differs from detected type"):
                got = PlotXAccessor(data)(dimensions=dims, coordinates=coords)
            assert type(got) is cls, f"specify_grid({value!r}) gave {type(got).__name__}, expected {cls.__name__}"
        specify_grid(grid_type=None)
        got = PlotXAccessor(gridded)()
        assert type(got) is GriddedPlotter, f"after clearing the type, detection gave {type(got).__name__}"
        got = PlotXAccessor(cells)()
        assert type(got) is UnstructuredPlotter, f"after clearing the type, detection gave {type(got).__name__}"

    def invalid_type_is_refused():
        with pytest.raises(ConfigurationError, match="Invalid grid type specification"):
            specify_grid(grid_type="invalid")

    def global_paths_reach_the_plotter():
        tgrid, ckdtree = tmp_path / "tgrid.nc", tmp_path / "ckdtree"
        specify_grid(grid_type="unstructured", fpath_tgrid=tgrid, fpath_ckdtree=ckdtree)
        plotter = PlotXAccessor(cells)(dimensions=CELL_DIMS, coordinates=CELL_COORDS)
        assert type(plotter) is UnstructuredPlotter
        assert str(plotter.fpath_tgrid) == str(tgrid) and str(plotter.fpath_ckdtree) == str(ckdtree)

    def per_plotter_paths():
        plotter = UnstructuredPlotter(cells, dimensions=CELL_DIMS, coordinates=CELL_COORDS)
        plotter.specify_grid(fpath_tgrid="/path/to/triangulation.nc", fpath_ckdtree="/path/to/ckdtree")
        assert Path(plotter.fpath_tgrid) == Path("/path/to/triangulation.nc")
        assert Path(plotter.fpath_ckdtree) == Path("/path/to/ckdtree")
        as_paths = (Path("/path/to/a.nc"), Path("/path/to/ckd"))
        plotter.specify_grid(fpath_tgrid=as_paths[0], fpath_ckdtree=as_paths[1])
        assert plotter.fpath_tgrid == as_paths[0] and plotter.fpath_ckdtree == as_paths[1]
        plotter.specify_grid(fpath_tgrid="/path/only.nc")
        assert Path(plotter.fpath_tgrid) == Path("/path/only.nc")
        assert plotter.fpath_ckdtree is None, "a path left out must be cleared"
        plotter.specify_grid(fpath_tgrid="/some/path", fpath_ckdtree="/another/path")
        plotter.specify_grid(fpath_tgrid=None, fpath_ckdtree=None)
        assert plotter.fpath_tgrid is None and plotter.fpath_ckdtree is None

    try:
        check(failures, "global type decides the class", global_type_decides_the_class)
        check(failures, "invalid type", invalid_type_is_refused)
        check(failures, "global paths", global_paths_reach_the_plotter)
        check(failures, "per-plotter paths", per_plotter_paths)
    finally:
        specify_grid(grid_type=None, fpath_tgrid=None, fpath_ckdtree=None)
    assert_no_failures(failures)


def test_plot_config_defaults_and_overrides():
    """``PlotConfig`` defaults, overrides, and the one derived rule: ID plots switch the colorbar off."""
    failures = []

    def defaults():
        config = PlotConfig()
        expected = {
            "title": None,
            "var_units": "",
            "issym": False,
            "cmap": None,
            "cperc": [4, 96],
            "clim": None,
            "show_colorbar": True,
            "grid_lines": True,
            "grid_labels": False,
            "dimensions": DEFAULT_MAP,
            "coordinates": DEFAULT_MAP,
            "norm": None,
            "plot_IDs": False,
            "extend": "both",
            "framerate": 10,
            "ckdtree_res": 0.3,
        }
        for field, want in expected.items():
            assert getattr(config, field) == want, f"{field} is {getattr(config, field)!r}, expected {want!r}"
        assert isinstance(config.projection, ccrs.Robinson), f"default projection is {config.projection!r}"

    def overrides():
        dims = {"time": "time_var", "x": "x_var", "y": "y_var"}
        coords = {"time": "time_coord", "x": "x_coord", "y": "y_coord"}
        given = {
            "title": "Test Plot",
            "var_units": "degC",
            "issym": True,
            "cmap": "viridis",
            "cperc": [5, 95],
            "clim": (0, 10),
            "grid_lines": False,
            "grid_labels": True,
            "extend": "neither",
            "dimensions": dims,
            "coordinates": coords,
        }
        config = PlotConfig(show_colorbar=False, **given)
        for field, want in given.items():
            assert getattr(config, field) == want, f"{field} is {getattr(config, field)!r}, expected {want!r}"
        assert config.show_colorbar is False

    def ids_disable_the_colorbar():
        assert PlotConfig(plot_IDs=True, show_colorbar=True).show_colorbar is False
        assert PlotConfig(plot_IDs=False, show_colorbar=True).show_colorbar is True

    check(failures, "defaults", defaults)
    check(failures, "overrides", overrides)
    check(failures, "plot_IDs disables the colorbar", ids_disable_the_colorbar)
    assert_no_failures(failures)


def test_single_plot_config_reaches_the_figure():
    """A real Agg render: each configuration field lands on the artist (cmap, norm, clim, colorbar, units, title, axes)."""
    rng = np.random.default_rng(0)
    lat, lon = np.linspace(-10, 10, 3), np.linspace(0, 20, 3)

    def field(values, dims=("time", "lat", "lon"), **coords):
        values = np.asarray(values)
        coord = {"lat": lat[: values.shape[-2]], "lon": lon[: values.shape[-1]]}
        if "time" in dims:
            coord["time"] = [0]
        coord.update(coords)
        return xr.DataArray(values, dims=dims, coords=coord)

    noise = field(rng.standard_normal((1, 3, 3)))
    failures = []

    def draw(data, **config):
        fig, ax, im = GriddedPlotter(data).single_plot(PlotConfig(**config))
        return fig, ax, im

    def defaults():
        fig, ax, im = draw(noise)
        assert im.cmap.name == "viridis", f"cmap is {im.cmap.name}"
        assert len(fig.axes) == 2, f"{len(fig.axes)} axes, expected the map and a colorbar"
        want = np.percentile(noise.values, [4, 96])
        assert np.allclose(im.get_clim(), want), f"clim {im.get_clim()}, expected percentiles {want}"

    def named_cmap():
        assert draw(noise, cmap="plasma")[2].cmap.name == "plasma"

    def custom_norm():
        norm = LogNorm(vmin=0.1, vmax=10)
        positive = field(np.abs(rng.standard_normal((1, 3, 3))) + 1)
        assert draw(positive, norm=norm, clim=(0.1, 10))[2].norm is norm

    def colorbar_off():
        fig, ax, im = draw(noise, show_colorbar=False)
        assert len(fig.axes) == 1, f"{len(fig.axes)} axes with the colorbar off"

    def units_on_the_colorbar():
        fig, ax, im = draw(noise, var_units="Test Units [degC]", show_colorbar=True)
        assert len(fig.axes) == 2 and fig.axes[-1].get_ylabel() == "Test Units [degC]", f"label {fig.axes[-1].get_ylabel()!r}"

    def explicit_clim():
        assert draw(noise, clim=(0.2, 0.8))[2].get_clim() == (0.2, 0.8)

    def symmetric_scale():
        data = field([[[-6, -2, 0], [2, 4, 5]]], lat=[0, 1])
        im = draw(data, issym=True)[2]
        vmin, vmax = im.get_clim()
        want = np.abs(np.percentile(data.values, [4, 96])).max()
        assert vmin == -vmax and np.isclose(vmax, want), f"clim {(vmin, vmax)}, expected +-{want}"
        assert im.cmap.name == "RdBu_r", f"cmap is {im.cmap.name}"

    def minimal_grid():
        data = field([[[1, 2], [3, 4]]], lat=[0, 1], lon=[0, 1])
        shown = np.ma.getdata(draw(data)[2].get_array())
        np.testing.assert_array_equal(np.asarray(shown).reshape(2, 2), [[1, 2], [3, 4]])

    def no_time_dimension():
        data = field(rng.standard_normal((3, 3)), dims=("lat", "lon"))
        shown = np.ma.getdata(draw(data)[2].get_array())
        np.testing.assert_array_equal(np.asarray(shown).reshape(3, 3), data.values)

    def id_plot():
        data = field([[[0, 1, 2], [1, 2, 0], [2, 0, 1]]])
        config = PlotConfig(plot_IDs=True, show_colorbar=True)
        fig, ax, im = GriddedPlotter(data).single_plot(config)
        assert config.show_colorbar is False and len(fig.axes) == 1, f"{len(fig.axes)} axes for an ID plot"
        assert im.cmap.N == 2 and isinstance(im.norm, BoundaryNorm), f"cmap N {im.cmap.N}, norm {type(im.norm).__name__}"
        assert np.ma.getmaskarray(im.get_array()).sum() == 3, "the three zero cells must be masked"

    def title():
        assert draw(noise, title="Sea level")[1].get_title() == "Sea level"
        assert draw(noise)[1].get_title() == "", "no title was requested"

    def reuses_a_given_axes():
        fig0 = plt.figure()
        ax0 = fig0.add_subplot(projection=ccrs.Robinson())
        fig, ax, im = GriddedPlotter(noise).single_plot(PlotConfig(), ax=ax0)
        assert ax is ax0 and fig is fig0, "the given axes were not reused"
        assert plt.get_fignums() == [fig0.number], f"figures {plt.get_fignums()}, expected only the caller's"
        assert im.axes is ax0

    for case, row in {
        "defaults": defaults,
        "named cmap": named_cmap,
        "custom norm": custom_norm,
        "colorbar off": colorbar_off,
        "units on the colorbar": units_on_the_colorbar,
        "explicit clim": explicit_clim,
        "symmetric scale": symmetric_scale,
        "minimal 2x2 grid": minimal_grid,
        "no time dimension": no_time_dimension,
        "ID plot": id_plot,
        "title": title,
        "given axes": reuses_a_given_axes,
    }.items():
        check(failures, case, row)
        plt.close("all")
    assert_no_failures(failures)


def test_clim_robust():
    """Robust colour limits equal the percentiles; symmetric takes the larger magnitude; a 0th percentile pins to 0."""
    plotter = _Plotter(_gridded())
    skewed = np.concatenate([np.arange(-10.0, 3.0, 0.5), [np.nan]])
    failures = []

    def symmetric():
        clim = plotter.clim_robust(skewed, issym=True, percentiles=[10, 90])
        bound = np.abs(np.nanpercentile(skewed, [10, 90])).max()
        np.testing.assert_allclose(clim, [-bound, bound])
        assert bound > np.nanpercentile(skewed, 90), "the larger magnitude is the lower percentile for this data"

    def asymmetric():
        data = np.array([0, 1, 2, 3, 4, 5])
        clim = plotter.clim_robust(data, issym=False, percentiles=[10, 90])
        np.testing.assert_allclose(clim, np.percentile(data, [10, 90]))

    def zero_percentile():
        data = np.array([1, 2, 3, 4, 5])
        clim = plotter.clim_robust(data, issym=False, percentiles=[0, 90])
        np.testing.assert_allclose(clim, [0, np.percentile(data, 90)])

    check(failures, "symmetric", symmetric)
    check(failures, "asymmetric", asymmetric)
    check(failures, "0th percentile", zero_percentile)
    assert_no_failures(failures)


def test_wrap_lon():
    """A cyclic column is added only for global longitudes; regional data is returned unchanged."""

    def field(lon):
        return xr.DataArray(
            np.arange(2 * 3 * len(lon), dtype=float).reshape(2, 3, len(lon)),
            dims=["time", "lat", "lon"],
            coords={"time": range(2), "lat": np.linspace(-90, 90, 3), "lon": np.array(lon, dtype=float)},
        )

    failures = []
    for case, (lon, wraps) in {
        "global every 60 degrees": ([0, 60, 120, 180, 240, 300], True),
        "span of almost 360 degrees": ([0, 120, 240, 359.9], True),
        "regional": ([0, 30, 60, 90], False),
    }.items():

        def row(lon=lon, wraps=wraps):
            data = field(lon)
            wrapped = GriddedPlotter(data).wrap_lon(data)
            assert wrapped.dims == data.dims
            if not wraps:
                assert wrapped.shape == data.shape, f"shape changed from {data.shape} to {wrapped.shape}"
                return
            assert wrapped.shape[2] == data.shape[2] + 1, f"shape {wrapped.shape} from {data.shape}"
            assert wrapped.lon.values[-1] == data.lon.values[0] + 360
            np.testing.assert_array_equal(wrapped.isel(lon=-1).values, data.isel(lon=0).values)

        check(failures, case, row)
    assert_no_failures(failures)


def test_titles_name_the_slice():
    """Titles give the date for the time dimension and exactly ``dim=value`` for any other dimension."""
    data = xr.DataArray(
        np.zeros((3, 2, 2, 4, 5)),
        dims=["time", "depth", "level", "lat", "lon"],
        coords={
            "time": pd.date_range("2020-01-01", periods=3),
            "depth": [0, 10],
            "level": [100, 200],
            "lat": np.linspace(-10, 10, 4),
            "lon": np.linspace(0, 20, 5),
        },
    )
    plotter = GriddedPlotter(data, dimensions={"time": "time", "y": "lat", "x": "lon"})
    failures = []
    for case, index, column, expected in [
        ("first date", 0, "time", "2020-01-01"),
        ("second date", 1, "time", "2020-01-02"),
        ("depth", 0, "depth", "depth=0"),
        ("second depth", 1, "depth", "depth=10"),
        ("level", 0, "level", "level=100"),
    ]:

        def row(index=index, column=column, expected=expected):
            title = plotter._get_title(time_index=index, col_name=column)
            assert title == expected, f"title {title!r}, expected {expected!r}"

        check(failures, case, row)
    assert_no_failures(failures)


def test_global_grid_with_seam_cell_renders():
    """A global 5-degree grid whose centres include the antimeridian cell renders in full instead of crashing in shapely."""
    lon = np.arange(0.0, 360, 5.0)
    lat = np.arange(-87.5, 90, 5.0)
    data = xr.DataArray(np.random.default_rng(0).random((lat.size, lon.size)), dims=("lat", "lon"), coords={"lat": lat, "lon": lon})
    failures = []

    def render():
        fig, ax, im = data.plotX.single_plot(PlotConfig(var_units="x", cmap="hot_r", cperc=[0, 96]))
        fig.canvas.draw()
        # One column for the cyclic wrap, one for the seam cell drawn as two halves.
        assert np.asarray(im.get_array()).size == lat.size * (lon.size + 2), "the seam cell must be drawn as two halves"

    check(failures, "global grid, centres from 0 degrees", render)
    assert_no_failures(failures)


def test_cell_across_seam_is_split_in_two():
    """The seam-straddling cell is drawn as two halves, one at each edge of the map.

    Read from the rendered pixels, not from the mesh's array: how cartopy distributes the cells that touch
    the wrap between its artists differs between releases, the picture must not.
    """
    from matplotlib.colors import ListedColormap

    lon = np.arange(-180.0, 180, 5.0)  # centred on 0, the seam of a projection centred on 180
    lat = np.arange(-87.5, 90, 5.0)
    values = np.tile((lon == 0.0).astype(float), (lat.size, 1))  # the seam cell is 1, every other cell 0

    fig = plt.figure(figsize=(14.4, 7.2), dpi=100)
    try:
        ax = fig.add_axes([0, 0, 1, 1], projection=ccrs.PlateCarree(central_longitude=180))
        mesh = pcolormesh_seam_safe(
            ax,
            lon,
            lat,
            values,
            transform=ccrs.PlateCarree(),
            shading="auto",
            cmap=ListedColormap(["white", "red"]),
            vmin=0,
            vmax=1,
        )
        ax.set_global()
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())

        def is_red(cell_lon, cell_lat):
            x, y = ax.projection.transform_point(cell_lon, cell_lat, ccrs.PlateCarree())
            px, py = ax.transData.transform((x, y))
            r, g, b, _ = rgba[int(rgba.shape[0] - py), int(px)]
            return (int(r), int(g), int(b)) == (255, 0, 0)

        assert np.asarray(mesh.get_array()).reshape(lat.size, -1).shape[1] == lon.size + 1
        for half in (-1.25, 1.25):
            assert all(is_red(half, y) for y in lat), f"half of the seam cell at lon {half} is not drawn"
        for neighbour in (-5.0, 5.0):
            assert not any(is_red(neighbour, y) for y in lat), f"the seam cell's value bled into lon {neighbour}"
    finally:
        plt.close(fig)


def test_grid_without_seam_cell_is_untouched():
    """No cell across the seam, no cartopy projection, or 2-D coordinates: the call is forwarded exactly as given."""
    from unittest.mock import MagicMock

    failures = []
    lon = np.arange(2.5, 360, 5.0)
    lat = np.arange(-87.5, 90, 5.0)
    values = np.zeros((lat.size, lon.size))

    def forwarded(ax, lons, lats):
        result = pcolormesh_seam_safe(ax, lons, lats, values, shading="auto")
        ax.pcolormesh.assert_called_once()
        args, kwargs = ax.pcolormesh.call_args
        assert result is ax.pcolormesh.return_value, "the mesh pcolormesh drew is not returned"
        assert args[0] is lons and args[1] is lats and args[2] is values, "the coordinates or values were not forwarded as given"
        assert kwargs == {"shading": "auto"}, kwargs

    def no_seam_cell():
        ax = MagicMock()
        ax.projection.proj4_params = {"lon_0": 0.0}
        forwarded(ax, lon, lat)

    def no_cartopy_projection():
        forwarded(MagicMock(spec=["pcolormesh"]), lon, lat)

    def two_dimensional_coords():
        ax = MagicMock()
        ax.projection.proj4_params = {"lon_0": 0.0}
        lon2d, lat2d = np.meshgrid(lon, lat)
        forwarded(ax, lon2d, lat2d)

    for row in (no_seam_cell, no_cartopy_projection, two_dimensional_coords):
        check(failures, row.__name__, row)
    assert_no_failures(failures)


def test_id_plot_warns_only_when_empty(marex_records):
    """An ID field with no objects plots a fully masked map and warns; one with objects shows them and does not."""
    records = marex_records(logging.WARNING)
    failures = []

    def warnings_about_empty():
        return [r.getMessage() for r in records if r.levelno >= logging.WARNING and "no objects" in r.getMessage()]

    def empty_field():
        records.clear()
        fig, ax, im = _ids_on_five_degree_grid(np.zeros((36, 72), dtype=int)).plotX.single_plot(PlotConfig(plot_IDs=True))
        fig.canvas.draw()
        assert np.ma.getmaskarray(im.get_array()).all(), "an empty ID field must plot an entirely masked map"
        assert len(warnings_about_empty()) == 1, f"warnings {warnings_about_empty()}"

    def field_with_objects():
        values = np.zeros((36, 72), dtype=int)
        values[10:14, 20:30] = 3
        values[20:24, 40:50] = 7
        records.clear()
        fig, ax, im = _ids_on_five_degree_grid(values).plotX.single_plot(PlotConfig(plot_IDs=True))
        fig.canvas.draw()
        assert set(np.unique(np.ma.compressed(im.get_array()))) == {3.0, 7.0}
        assert not warnings_about_empty(), f"warned about an empty field: {warnings_about_empty()}"

    check(failures, "empty ID field", empty_field)
    check(failures, "ID field with objects", field_with_objects)
    assert_no_failures(failures)


def test_id_plot_colour_table():
    """A listed map is used as given when it has colours enough; a shorter one cycles so neighbouring IDs differ."""
    rng = np.random.default_rng(0)
    failures = []

    def render(n_ids, cmap):
        fig, ax, im = _ids_on_five_degree_grid(n_ids=n_ids).plotX.single_plot(PlotConfig(plot_IDs=True, cmap=cmap))
        fig.canvas.draw()
        return im

    def listed_with_one_colour_per_id():
        cmap = ListedColormap(rng.random((5, 3)))
        im = render(5, cmap)
        assert im.cmap.N == 5 and im.cmap is cmap, f"cmap N {im.cmap.N}, not the map as given"
        np.testing.assert_array_equal(im.cmap.colors, cmap.colors)

    def listed_longer_than_the_ids():
        cmap = ListedColormap(rng.random((10, 3)))
        assert render(5, cmap).cmap is cmap, "a map with more colours than IDs must be used as given"

    def shorter_map_cycles(cmap, n_colours):
        im = render(300, cmap)
        colours = im.cmap(np.arange(300))
        assert im.cmap.N == 300, f"cmap N {im.cmap.N}"
        assert not np.array_equal(colours[0], colours[1]), "neighbouring IDs share a colour"
        np.testing.assert_array_equal(colours[0], colours[n_colours])

    check(failures, "listed, one colour per ID", listed_with_one_colour_per_id)
    check(failures, "listed, longer than the IDs", listed_longer_than_the_ids)
    check(failures, "tab10 cycles", lambda: shorter_map_cycles("tab10", 10))
    check(failures, "5-colour listed cycles", lambda: shorter_map_cycles(ListedColormap(rng.random((5, 3))), 5))
    assert_no_failures(failures)


def test_named_cmap_gives_one_bin_per_id():
    """A named colormap yields one bin per ID, also for 300 IDs, more than the 256 colours a BoundaryNorm would accept."""
    failures = []
    for n_ids in (5, 300):

        def row(n_ids=n_ids):
            fig, ax, im = _ids_on_five_degree_grid(n_ids=n_ids).plotX.single_plot(PlotConfig(plot_IDs=True, cmap="viridis"))
            fig.canvas.draw()
            assert im.cmap.N == n_ids, f"cmap has {im.cmap.N} colours"

        check(failures, f"n_ids={n_ids}", row)
    assert_no_failures(failures)


def _write_triangulation(path):
    """Four vertices and two triangles, with 1-based vertex indices as in an ICON grid file."""
    xr.Dataset(
        {
            "vertex_of_cell": (("nv", "cell"), np.array([[1, 1], [2, 3], [3, 4]])),
            "clon": ("vertex", np.array([0.0, 10.0, 10.0, 0.0])),
            "clat": ("vertex", np.array([0.0, 0.0, 10.0, 10.0])),
            "other": ("vertex", np.zeros(4)),
        }
    ).to_netcdf(path)


def _write_ckdtree(directory, res=0.3):
    """A 2 (lat) by 3 (lon) lookup table into a 4-cell field."""
    directory.mkdir()
    xr.Dataset(
        {"ickdtree_c": ("point", np.array([3, 2, 1, 0, 2, 3]))},
        coords={"lon": ("lon", np.array([0.0, 30.0, 60.0])), "lat": ("lat", np.array([-30.0, 30.0]))},
    ).to_netcdf(directory / f"res{res:3.2f}.nc")


def test_unstructured_grid_files_load_and_cache(tmp_path):
    """Real tiny grid files: triangulation and lookup table load, are cached by path, interpolate as indexed, and clear."""
    tgrid = tmp_path / "tgrid.nc"
    ckdtree = tmp_path / "ckdtree"
    _write_triangulation(tgrid)
    _write_ckdtree(ckdtree)
    failures = []

    def triangulation():
        tri = _load_triangulation(tgrid)
        np.testing.assert_array_equal(tri.triangles, [[0, 1, 2], [0, 2, 3]])
        np.testing.assert_array_equal(tri.x, [0.0, 10.0, 10.0, 0.0])
        np.testing.assert_array_equal(tri.y, [0.0, 0.0, 10.0, 10.0])
        assert str(tgrid) in _GRID_CACHE["triangulation"], "the triangulation was not cached by path"
        tgrid.rename(tmp_path / "moved.nc")
        assert _load_triangulation(tgrid) is tri, "the second load did not come from the cache"

    def lookup_table():
        loaded = _load_ckdtree(ckdtree, 0.3)
        np.testing.assert_array_equal(loaded["indices"], [3, 2, 1, 0, 2, 3])
        np.testing.assert_array_equal(loaded["lon"], [0.0, 30.0, 60.0])
        np.testing.assert_array_equal(loaded["lat"], [-30.0, 30.0])
        assert (str(ckdtree), 0.3) in _GRID_CACHE["ckdtree"], "the lookup table was not cached by path and resolution"
        assert _load_ckdtree(ckdtree, 0.3) is loaded, "the second load did not come from the cache"

    def interpolation():
        plotter = UnstructuredPlotter(_cells(ncell=4), dimensions=CELL_DIMS, coordinates=CELL_COORDS)
        plotter.specify_grid(fpath_ckdtree=ckdtree)
        grid_lon, grid_lat, grid_data = plotter._interpolate_with_ckdtree(np.array([10.0, 20.0, 30.0, 40.0]), res=0.3)
        np.testing.assert_array_equal(grid_data, [[40.0, 30.0, 20.0], [10.0, 30.0, 40.0]])
        np.testing.assert_array_equal(grid_lon, [[0.0, 30.0, 60.0]] * 2)
        np.testing.assert_array_equal(grid_lat, [[-30.0] * 3, [30.0] * 3])

    def clearing():
        assert _GRID_CACHE["triangulation"] and _GRID_CACHE["ckdtree"], "premise: both caches are filled"
        clear_cache()
        assert not _GRID_CACHE["triangulation"] and not _GRID_CACHE["ckdtree"]

    check(failures, "triangulation", triangulation)
    check(failures, "lookup table", lookup_table)
    check(failures, "interpolation", interpolation)
    check(failures, "clear_cache", clearing)
    assert_no_failures(failures)


def test_unstructured_grid_bad_inputs_raise(tmp_path):
    """A grid file missing variables, a missing lookup-table file and a missing lookup-table path each raise a clear error."""
    incomplete = tmp_path / "incomplete.nc"
    xr.Dataset({"other_var": ("x", np.zeros(3))}).to_netcdf(incomplete)
    failures = []

    def missing_variables():
        with pytest.raises(DataValidationError, match="Invalid triangulation grid file format"):
            _load_triangulation(incomplete)

    def missing_file():
        with pytest.raises(DataValidationError, match="KDTree file not found"):
            _load_ckdtree(tmp_path / "nowhere", 0.25)

    def no_path():
        plotter = UnstructuredPlotter(_cells(), dimensions=CELL_DIMS, coordinates=CELL_COORDS)
        plotter.fpath_ckdtree = None
        with pytest.raises(VisualisationError, match="KDTree path not specified"):
            plotter._interpolate_with_ckdtree(np.array([10, 20, 30]), res=0.3)

    check(failures, "triangulation file without its variables", missing_variables)
    check(failures, "lookup-table file does not exist", missing_file)
    check(failures, "no lookup-table path", no_path)
    assert_no_failures(failures)


def test_single_plot_refuses_an_extra_dimension():
    """A field carrying a further dimension is refused with the dimension named; a selected level still plots."""
    dims = {"time": "time", "x": "lon", "y": "lat"}
    field = xr.DataArray(
        np.random.default_rng(0).random((2, 2, 3, 4)),
        dims=["time", "depth", "lat", "lon"],
        coords={"time": range(2), "depth": [0.0, 10.0], "lat": np.linspace(-10, 10, 3), "lon": np.linspace(0, 30, 4)},
    )
    failures = []

    def extra_dimension_raises():
        plotter = field.isel(time=0).plotX(dimensions=dims, coordinates=dims)
        with pytest.raises(VisualisationError, match="depth"):
            plotter.single_plot(PlotConfig(title="3-D"))

    def selected_level_plots():
        level = field.isel(time=0, depth=0, drop=True)
        fig, ax, im = level.plotX(dimensions=dims, coordinates=dims).single_plot(PlotConfig(title="level 0"))
        try:
            np.testing.assert_array_equal(np.asarray(im.get_array()).reshape(level.shape), level.values)
        finally:
            plt.close(fig)

    check(failures, "extra dimension", extra_dimension_raises)
    check(failures, "selected level", selected_level_plots)
    assert_no_failures(failures)

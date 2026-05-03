"""Plot configuration dataclass for the plotX visualisation system."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from ..logging_config import configure_logging

# Handle optional dependencies for plotting
try:
    import cartopy.crs as ccrs

    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    ccrs = None

try:
    from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    BoundaryNorm = None
    ListedColormap = None
    Normalize = None


@dataclass
class PlotConfig:
    """Configuration class for plot parameters

    Attributes:
        title: Plot title
        var_units: Variable units for colorbar label
        issym: Whether data is symmetric (centers colormap at 0)
        cmap: Colormap name or ListedColormap object
        cperc: Percentile range for automatic color limits [min, max]
        clim: Manual color limits (vmin, vmax)
        show_colorbar: Whether to display colorbar
        grid_lines: Whether to display grid lines
        grid_labels: Whether to display grid labels
        dimensions: Mapping of conceptual to actual dimension names
        coordinates: Mapping of conceptual to actual coordinate names
        norm: Custom normalization (BoundaryNorm or Normalize)
        plot_IDs: Whether to plot object IDs with random colors
        extend: Colorbar extension ('neither', 'both', 'min', 'max')
        verbose: Enable verbose logging
        quiet: Enable quiet logging
        projection: Cartopy projection for map plots
        framerate: Frames per second for animations (default 10)
        ckdtree_res: Resolution (degrees) of the pre-computed ckdtree grid used
            for unstructured interpolation, matching the ``res<value>.nc`` file
            naming (default 0.3)
    """

    title: Optional[str] = None
    var_units: str = ""
    issym: bool = False
    cmap: Optional[Union[str, ListedColormap]] = None
    cperc: List[int] = None
    clim: Optional[Tuple[float, float]] = None
    show_colorbar: bool = True
    grid_lines: bool = True
    grid_labels: bool = False
    dimensions: Dict[str, str] = None
    coordinates: Dict[str, str] = None
    norm: Optional[Union[BoundaryNorm, Normalize]] = None
    plot_IDs: bool = False
    extend: str = "both"
    verbose: Optional[bool] = None
    quiet: Optional[bool] = None
    projection: Optional[Any] = None
    framerate: int = 10
    ckdtree_res: float = 0.3

    def __post_init__(self) -> None:
        """Initialise default values and configure logging."""
        if self.cperc is None:
            self.cperc = [4, 96]
        if self.dimensions is None:
            self.dimensions = {"time": "time", "y": "lat", "x": "lon"}
        if self.coordinates is None:
            self.coordinates = {"time": "time", "y": "lat", "x": "lon"}
        if self.plot_IDs:
            self.show_colorbar = False
        if self.projection is None and HAS_CARTOPY:
            self.projection = ccrs.Robinson()

        # Configure logging if verbose/quiet parameters are provided
        if self.verbose is not None or self.quiet is not None:
            configure_logging(verbose=self.verbose, quiet=self.quiet)

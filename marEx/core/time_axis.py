"""
Time-axis helpers shared across marEx.

Provides calendar detection and :func:`add_decimal_year`, the continuous-time
coordinate used by polynomial and harmonic detrending. This module is a leaf in
the dependency graph and imports only third-party libraries and the package
logger.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

from ..exceptions import ConfigurationError
from ..logging_config import get_logger

# Get module logger
logger = get_logger(__name__)

# Number of day-of-year slots the daily cycle is resolved on. 366 rather than 365 so a
# leap day is a group of its own rather than colliding with 1 March.
_DAYS_PER_YEAR = 366


@dataclass(frozen=True)
class SeasonalCycle:
    """The repeating within-year axis a climatology or seasonal threshold is resolved on.

    marEx was written for daily data, where that axis is ``dayofyear`` with 366 slots.
    This dataclass states the axis explicitly so monthly and sub-daily series can use
    the same code paths with a ``month`` or ``hourofyear`` axis instead.

    Attributes
    ----------
    index_name : str
        Name of the cycle dimension/coordinate: ``"dayofyear"``, ``"month"`` or
        ``"hourofyear"``. This is the *label* used for the output dimension, so it
        must stay ``"dayofyear"`` for daily data -- it appears in the golden zarr
        schema and in every shipped example notebook.
    length : int
        Number of slots in one cycle (366 daily, 12 monthly, ``366 * steps_per_day``
        sub-daily).
    step_days : float
        Median spacing of the time axis, in days. ``1.0`` for daily data.
    """

    index_name: str
    length: int
    step_days: float

    @property
    def labels(self) -> np.ndarray:
        """1-based cycle labels, ``[1, ..., length]`` -- the ``expected_groups`` axis."""
        return np.arange(1, self.length + 1, dtype=np.int32)

    @property
    def is_daily(self) -> bool:
        """Whether this cycle is the daily (``dayofyear``) index."""
        return self.index_name == "dayofyear"

    @property
    def is_subdaily(self) -> bool:
        """Whether this cycle is the sub-daily (``hourofyear``) index."""
        return self.index_name == "hourofyear"

    @property
    def steps_per_day(self) -> int:
        """Cycle slots per calendar day (1 for daily, 24 for hourly, 0 for monthly)."""
        if self.is_subdaily:
            return max(1, self.length // _DAYS_PER_YEAR)
        return 1 if self.is_daily else 0

    def index_of(self, time_coord: xr.DataArray) -> xr.DataArray:
        """Map a time coordinate onto this cycle's 1-based index.

        The returned DataArray is *named* :attr:`index_name`, because flox and
        ``groupby`` take the output dimension's name from the grouper.
        """
        if self.index_name == "month":
            index = time_coord.dt.month
        elif self.index_name == "dayofyear":
            index = time_coord.dt.dayofyear
        else:
            steps_per_day = self.steps_per_day
            day_fraction = (time_coord.dt.hour * 3600 + time_coord.dt.minute * 60 + time_coord.dt.second) / 86400.0
            within_day = np.floor(day_fraction * steps_per_day).astype(np.int32)
            index = (time_coord.dt.dayofyear - 1) * steps_per_day + within_day + 1
        return index.rename(self.index_name)

    def steps_for_days(self, days: float, *, odd: bool = False, name: str = "window") -> int:
        """Convert a physical duration in days into a whole number of cycle steps.

        ``window_days`` and ``smooth_days`` are physical durations, not step counts.
        On daily data the conversion is the identity, which is what keeps every
        existing result bit-identical.

        Parameters
        ----------
        days : float
            Requested duration in days.
        odd : bool, default False
            Force the result odd. Required where the window must be symmetric about
            a centre step (the seasonal-percentile window); **not** applied to the
            climatology smoothing or the standardisation window, where an even step
            count is legitimate and forcing it odd would move existing daily output.
        name : str, default "window"
            Parameter name used in the clamp warning.
        """
        step_days = self.step_days if self.step_days > 0 else 1.0
        steps = int(round(days / step_days))
        if odd and steps % 2 == 0:
            steps += 1
        steps = max(1, steps)
        realised = steps * step_days
        if abs(realised - days) > 0.5 * step_days:
            logger.warning(
                f"{name}={days} days cannot be represented on a {step_days:g}-day time axis: "
                f"using {steps} step(s) = {realised:g} days. "
                f"The realised {name} differs from the request by {abs(realised - days):g} days."
            )
        return steps

    def window_steps(self, days: float, name: str = "window_days") -> int:
        """:meth:`steps_for_days` with the result forced odd (symmetric windows)."""
        return self.steps_for_days(days, odd=True, name=name)


#: The daily cycle -- marEx's historical, and still overwhelmingly common, resolution.
DAILY_CYCLE = SeasonalCycle("dayofyear", _DAYS_PER_YEAR, 1.0)


def _step_days(time_coord: xr.DataArray) -> Optional[np.ndarray]:
    """Spacings of ``time_coord`` in days, or ``None`` if they cannot be derived."""
    values = np.asarray(time_coord.values).reshape(-1)
    if values.size < 2:
        return None
    if np.issubdtype(values.dtype, np.datetime64):
        return np.diff(values.astype("datetime64[ns]")).astype("timedelta64[s]").astype(np.float64) / 86400.0
    if values.dtype == np.dtype("O"):
        try:
            return np.array([(b - a).total_seconds() for a, b in zip(values[:-1], values[1:])], dtype=np.float64) / 86400.0
        except (AttributeError, TypeError):
            return None
    return None


def infer_cycle(time_coord: xr.DataArray, dim: str = "time") -> SeasonalCycle:
    """Infer the :class:`SeasonalCycle` a time axis should be resolved on.

    ``step_days >= 28`` gives a monthly cycle, ``>= 1`` the daily cycle, and anything
    finer a sub-daily ``hourofyear`` cycle of ``366 * round(1 / step_days)`` slots.

    The cadence is the **median** spacing, never the first: a single gap (a missing
    file, a concat seam) must not redefine the cycle. A genuinely irregular axis --
    one whose interquartile spread of spacings exceeds the median -- raises
    :class:`~marEx.exceptions.ConfigurationError` rather than guessing, and the caller
    can override with the ``cycle=`` escape hatch.

    An axis too short to measure (fewer than two timesteps), or one that is not
    datetime-like, falls back to :data:`DAILY_CYCLE`.

    **Sub-daily ceiling.** ``extremes/histogram.py`` builds its flox ``expected_groups``
    as ``uint16``, so a cycle longer than 65535 slots would wrap. That bound is
    ``steps_per_day > 179``, i.e. finer than roughly 8-minute data; hourly (8784 slots)
    and even 1-minute-of-15 (35136) are well inside it. Anything finer is rejected here
    rather than silently wrapping thousands of timesteps into the wrong group.
    """
    diffs = _step_days(time_coord)
    if diffs is None or diffs.size == 0:
        logger.debug("Cannot infer time cadence from '%s'; assuming daily.", dim)
        return DAILY_CYCLE

    step_days = float(np.median(diffs))
    if step_days <= 0:
        raise ConfigurationError(
            f"Time axis '{dim}' has a non-positive median spacing ({step_days} days)",
            details="marEx needs a monotonically increasing time axis to infer the seasonal cycle.",
            suggestions=[f"Sort the data along '{dim}'", "Remove duplicate timesteps"],
            context={"median_step_days": step_days},
        )

    # Irregularity guard. The IQR is used rather than the range so that a handful of
    # gaps (leap-day omissions, a missing month, a concat seam) do not trip it, while a
    # genuinely mixed-cadence axis does: gaps live in the tails and leave the middle
    # half untouched, whereas two interleaved cadences push the quartiles apart.
    #
    # The threshold is HALF the median, not the median. A plain `iqr > median` test is
    # very nearly unfireable: for a two-valued spacing distribution the IQR is bounded
    # by `max - min`, which is below the median whenever the larger spacing holds at
    # least half the mass -- so the case this guard exists for (a daily series concatenated
    # with a monthly one) slips straight through. Half the median fires on that while
    # leaving every regular axis far outside: a monthly axis, the most ragged regular
    # case, has spacings of 28-31 days, an IQR of ~1 against a threshold of ~15.
    if diffs.size >= 4:
        q75, q25 = np.percentile(diffs, [75, 25])
        iqr = float(q75 - q25)
        if iqr > 0.5 * step_days:
            raise ConfigurationError(
                f"Time axis '{dim}' is too irregular to infer a seasonal cycle",
                details=(
                    f"The interquartile spread of the timestep spacings ({iqr:g} days) exceeds half the "
                    f"median spacing ({step_days:g} days), so no single cadence describes the axis."
                ),
                suggestions=[
                    "Resample the data onto a regular time axis",
                    "Pass an explicit cycle=marEx.SeasonalCycle(...) to override the inference",
                ],
                context={"median_step_days": step_days, "iqr_step_days": iqr},
            )

    if step_days >= 28:
        cycle = SeasonalCycle("month", 12, step_days)
    elif step_days >= 1:
        cycle = SeasonalCycle("dayofyear", _DAYS_PER_YEAR, step_days)
    else:
        steps_per_day = max(1, int(round(1.0 / step_days)))
        if _DAYS_PER_YEAR * steps_per_day > np.iinfo(np.uint16).max:
            raise ConfigurationError(
                f"Time axis '{dim}' is too finely resolved for a seasonal cycle",
                details=(
                    f"A median spacing of {step_days:g} days is {steps_per_day} steps per day, giving a cycle of "
                    f"{_DAYS_PER_YEAR * steps_per_day} slots. The histogram path labels its groups as uint16, so "
                    f"cycles above {np.iinfo(np.uint16).max} slots would wrap silently."
                ),
                suggestions=[
                    "Coarsen the time axis before calling marEx (e.g. resample to hourly)",
                    "Use method_extreme='global_percentile', which needs no within-year cycle",
                ],
                context={"step_days": step_days, "steps_per_day": steps_per_day},
            )
        cycle = SeasonalCycle("hourofyear", _DAYS_PER_YEAR * steps_per_day, step_days)

    logger.debug(
        "Inferred seasonal cycle '%s' (%d slots) from a median spacing of %g days.",
        cycle.index_name,
        cycle.length,
        step_days,
    )
    return cycle


def _median_step_days(time_coord: xr.DataArray) -> float:
    """Median spacing of a time axis in days, defaulting to 1.0 when unmeasurable."""
    diffs = _step_days(time_coord)
    if diffs is None or diffs.size == 0:
        return 1.0
    return float(np.median(diffs))


def is_subdaily_axis(da: xr.DataArray, coord_name: str, cycle: Optional[SeasonalCycle] = None) -> bool:
    """Whether a time axis is finer than daily, without demanding a full cycle.

    Deliberately does **not** go through :func:`infer_cycle`: that raises on a
    mixed-cadence axis, and a caller only asking "is this sub-daily?" must not be made
    to fail on an axis it could otherwise process. An axis too short or too irregular to
    describe is reported as not sub-daily -- the conservative answer, since the guards
    built on this reject work rather than accepting it.
    """
    if cycle is not None:
        return cycle.is_subdaily
    return bool(_median_step_days(da[coord_name]) < 1.0)


def resolve_cycle(da: xr.DataArray, coord_name: str, cycle: Optional[SeasonalCycle] = None) -> SeasonalCycle:
    """Return ``cycle`` if the caller supplied one, otherwise infer it from ``da``."""
    if cycle is not None:
        return cycle
    return infer_cycle(da[coord_name], coord_name)


def _is_cftime_coord(time_coord: xr.DataArray) -> bool:
    """Return True if ``time_coord`` holds cftime datetime objects.

    Only genuine cftime coordinates (object dtype containing ``cftime.datetime``
    instances) take the calendar-aware branch in :func:`add_decimal_year`; plain
    numeric or numpy-datetime64 coordinates fall through to the legacy path.
    """
    values = np.asarray(time_coord.values)
    if values.dtype != np.dtype("O") or values.size == 0:
        return False
    try:
        import cftime
    except ImportError:  # pragma: no cover - cftime ships with netcdf4
        return False
    return isinstance(values.reshape(-1)[0], cftime.datetime)


def add_decimal_year(
    da: xr.DataArray, dim: str = "time", coord: Optional[str] = None, subdaily: Optional[bool] = None
) -> xr.DataArray:
    """
    Add decimal year coordinate to DataArray for trend analysis.

    Parameters
    ----------
    da : xarray.DataArray
        Input data with datetime coordinate
    dim : str, optional
        Name of the time dimension
    coord : str, optional
        Name of the time coordinate (if different from dimension name)
    subdaily : bool, optional
        Whether to resolve the within-year fraction to sub-day precision. ``None``
        (the default) infers it: sub-day resolution is used only when the axis is
        finer than daily.

        **This conditional is load-bearing and must not be made unconditional.**
        Timestamps are routinely stamped away from midnight (marEx's own gridded
        fixture sits at 12:00:00 and the unstructured one at 23:59:59), so adding a
        sub-day term on daily data shifts ``decimal_year`` by up to a day, changes
        the harmonic design matrix, and moves every detrended result.

    Returns
    -------
    xarray.DataArray
        Input data with added 'decimal_year' coordinate
    """
    # Use coordinate name if provided, otherwise use dimension name
    coord_name = coord if coord is not None else dim
    time_coord = da[coord_name]

    if subdaily is None:
        diffs = _step_days(time_coord)
        subdaily = bool(diffs is not None and diffs.size > 0 and float(np.median(diffs)) < 1.0)

    if _is_cftime_coord(time_coord):
        # cftime / non-standard calendars (noleap, 360_day, ...): pd.to_datetime
        # cannot parse these. Derive the decimal year from day-of-year directly,
        # using a calendar-aware days-in-year so the fraction is correct. Matches the
        # legacy path's definition: year + (dayofyear - 1) / days_in_year.
        year = time_coord.dt.year
        dayofyear = time_coord.dt.dayofyear
        fixed_length = {"360_day": 360, "noleap": 365, "365_day": 365, "all_leap": 366, "366_day": 366}
        calendar = getattr(time_coord.dt, "calendar", "standard")
        if calendar in fixed_length:
            days_in_year = xr.full_like(year, fixed_length[calendar], dtype=float)
        else:
            # Calendars with real leap years (standard/gregorian/julian/proleptic_gregorian)
            days_in_year = 365.0 + time_coord.dt.is_leap_year.astype(float)
        # Sub-day term threaded through this branch identically to the datetime64 one:
        # a calendar-dependent split here would make the anomaly depend on the calendar
        # attribute rather than on the cadence.
        elapsed = dayofyear - 1
        if subdaily:
            elapsed = elapsed + (time_coord.dt.hour * 3600 + time_coord.dt.minute * 60 + time_coord.dt.second) / 86400.0
        decimal_year = np.asarray((year + elapsed / days_in_year).values)
    else:
        # Legacy path (unchanged; bit-identical for datetime64 and numeric coordinates).
        time = pd.to_datetime(time_coord)
        start_of_year = pd.to_datetime(time.year.astype(str) + "-01-01")
        start_of_next_year = pd.to_datetime((time.year + 1).astype(str) + "-01-01")
        if subdaily:
            year_elapsed = (time - start_of_year).total_seconds() / 86400.0
        else:
            year_elapsed = (time - start_of_year).days
        year_duration = (start_of_next_year - start_of_year).days
        decimal_year = time.year + year_elapsed / year_duration

    return da.assign_coords(decimal_year=(dim, decimal_year))

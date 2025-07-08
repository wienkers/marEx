"""
MarEx Exception Hierarchy: Error Handling
===================================================================================

This module provides a structured exception hierarchy for the marEx package.
"""

from typing import Any, Dict, List, Optional, Type


class MarExError(Exception):
    """
    Base exception class for all MarEx-specific errors.

    This is the root of the MarEx exception hierarchy and provides
    common functionality for all marEx exceptions including:

    * Structured error context
    * Exception chaining support
    * Consistent error formatting

    Parameters
    ----------
    message : str
        Primary error message describing what went wrong
    details : str, optional
        Additional technical details about the error
    suggestions : list of str, optional
        Actionable suggestions for resolving the error
    error_code : str, optional
        Structured error code for programmatic handling
    context : dict, optional
        Additional context information (e.g., parameter values, data shapes)
    """

    def __init__(
        self,
        message: str,
        details: Optional[str] = None,
        suggestions: Optional[List[str]] = None,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Initialise the Error."""
        self.message = message
        self.details = details
        self.suggestions = suggestions or []
        self.error_code = error_code
        self.context = context or {}

        # Build comprehensive error message
        full_message = self._format_error_message()
        super().__init__(full_message)

    def _format_error_message(self) -> str:
        """Format a comprehensive error message with all available information."""
        parts = [self.message]

        if self.details:
            parts.append(f"Details: {self.details}")

        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            parts.append(f"Context: {context_str}")

        if self.suggestions:
            suggestions_str = "\n".join(f"  - {s}" for s in self.suggestions)
            parts.append(f"Suggestions:\n{suggestions_str}")

        if self.error_code:
            parts.append(f"Error Code: {self.error_code}")

        return "\n".join(parts)

    def add_suggestion(self, suggestion: str) -> None:
        """Add an additional suggestion for resolving the error."""
        self.suggestions.append(suggestion)

    def add_context(self, key: str, value: Any) -> None:
        """Add additional context information."""
        self.context[key] = value


class DataValidationError(MarExError):
    """
    Raise exception for input data validation issues.

    This exception covers problems with input data structure, format,
    content, or compatibility with marEx processing requirements.

    Common scenarios:

    * Non-Dask arrays when Dask is required
    * Missing required coordinates or dimensions
    * Invalid data types or ranges
    * Incompatible chunking strategies
    * Malformed input datasets

    Examples
    --------
    >>> raise DataValidationError(
    ...     "Input DataArray must be Dask-backed",
    ...     details="Found numpy array, but marEx requires chunked Dask arrays",
    ...     suggestions=["Use da.chunk() to convert to Dask array",
                          "Load data with dask chunking: xr.open_dataset(...).chunk()"],
    ...     context={"data_type": type(data), "shape": data.shape}
    ... )
    """

    def __init__(
        self,
        message: str,
        details: Optional[str] = None,
        suggestions: Optional[List[str]] = None,
        error_code: str = "DATA_VALIDATION",
        context: Optional[Dict[str, Any]] = None,
    ):
        """Initialise the Error."""
        super().__init__(message, details, suggestions, error_code, context)


class CoordinateError(MarExError):
    """
    Raise exception for coordinate system problems.

    This exception handles issues with geographic coordinates including
    unit mismatches, invalid ranges, missing coordinate information,
    and coordinate system inconsistencies.

    Common scenarios:

    * Latitude/longitude values outside valid ranges
    * Unit mismatches (degrees vs radians)
    * Missing coordinate dimensions
    * Inconsistent coordinate systems between datasets
    * Auto-detection failures
    """

    def __init__(
        self,
        message: str,
        details: Optional[str] = None,
        suggestions: Optional[List[str]] = None,
        error_code: str = "COORDINATE_ERROR",
        context: Optional[Dict[str, Any]] = None,
    ):
        """Initialise the Error."""
        super().__init__(message, details, suggestions, error_code, context)


class ProcessingError(MarExError):
    """
    Raise exception for computational and algorithmic issues.

    This exception covers problems that occur during data processing,
    including numerical computation errors, algorithm convergence issues,
    and memory/performance problems.

    Common scenarios:

    * Insufficient memory for computation
    * Numerical instability or overflow
    * Algorithm convergence failures
    * Chunking strategy problems
    * Dask computation errors
    """

    def __init__(
        self,
        message: str,
        details: Optional[str] = None,
        suggestions: Optional[List[str]] = None,
        error_code: str = "PROCESSING_ERROR",
        context: Optional[Dict[str, Any]] = None,
    ):
        """Initialise the Error."""
        super().__init__(message, details, suggestions, error_code, context)


class ConfigurationError(MarExError):
    """
    Raise exception for parameter and setup issues.

    This exception handles problems with function parameters, configuration
    settings, and setup requirements that prevent proper operation.

    Common scenarios:

    * Invalid parameter values or combinations
    * Missing required configuration
    * Incompatible parameter settings
    * Environment setup issues

    Examples
    --------
    >>> raise ConfigurationError(
    ...     "Invalid threshold percentile value",
    ...     details="threshold_percentile must be between 0 and 100",
    ...     suggestions=["Use percentile value between 50-99 for extreme events",
                          "Common values: 90 (moderate), 95 (strong), 99 (severe)"],
    ...     context={"provided_value": 150, "valid_range": [0, 100]}
    ... )
    """

    def __init__(
        self,
        message: str,
        details: Optional[str] = None,
        suggestions: Optional[List[str]] = None,
        error_code: str = "CONFIGURATION_ERROR",
        context: Optional[Dict[str, Any]] = None,
    ):
        """Initialise the Error."""
        super().__init__(message, details, suggestions, error_code, context)


class DependencyError(MarExError):
    """
    Raise exception for missing or incompatible dependencies.

    This exception handles issues with optional or required dependencies
    that are missing, incompatible, or incorrectly configured.

    Common scenarios:

    * Missing optional dependencies (JAX, ffmpeg)
    * Version incompatibilities
    * Import failures
    * System dependency issues

    Examples
    --------
    >>> raise DependencyError(
    ...     "JAX acceleration not available",
    ...     details="JAX package not found or incompatible version",
    ...     suggestions=["Install JAX: pip install marEx[full]",
                          "Check CUDA compatibility for GPU acceleration",
                          "Processing will continue with NumPy backend"],
    ...     context={"requested_feature": "GPU acceleration", "available": False}
    ... )
    """

    def __init__(
        self,
        message: str,
        details: Optional[str] = None,
        suggestions: Optional[List[str]] = None,
        error_code: str = "DEPENDENCY_ERROR",
        context: Optional[Dict[str, Any]] = None,
    ):
        """Initialise the Error."""
        super().__init__(message, details, suggestions, error_code, context)


class TrackingError(MarExError):
    """
    Raise exception for object tracking and identification issues.

    This exception covers problems specific to the tracking module
    including binary object identification, temporal linking,
    and merge/split handling.

    Common scenarios:

    * Invalid binary input data
    * Tracking parameter conflicts
    * Temporal continuity issues
    * Memory overflow during tracking
    * Checkpoint/resume failures

    Examples
    --------
    >>> raise TrackingError(
    ...     "Tracking failed due to excessive memory usage",
    ...     details="Event fragmentation created >100,000 objects per timestep",
    ...     suggestions=["Increase area_filter_quartile to remove small events",
                          "Apply stronger spatial smoothing before tracking",
                          "Consider processing shorter time periods"],
    ...     context={"objects_per_timestep": 150000, "memory_limit_gb": 32}
    ... )
    """

    def __init__(
        self,
        message: str,
        details: Optional[str] = None,
        suggestions: Optional[List[str]] = None,
        error_code: str = "TRACKING_ERROR",
        context: Optional[Dict[str, Any]] = None,
    ):
        """Initialise the Error."""
        super().__init__(message, details, suggestions, error_code, context)


class VisualisationError(MarExError):
    """
    Raise exception for plotting and visualisation problems.

    This exception handles issues with the plotX visualisation system
    including matplotlib configuration, cartopy projections,
    and animation generation.

    Common scenarios:

    * Missing plotting dependencies
    * Cartopy projection issues
    * Invalid plot configuration
    * Animation encoding failures
    * Grid type detection problems

    Examples
    --------
    >>> raise VisualisationError(
    ...     "Animation creation failed",
    ...     details="ffmpeg encoder not found for MP4 generation",
    ...     suggestions=["Install ffmpeg system package",
                          "Use alternative format: save_format='gif'",
                          "Install ffmpeg via conda: conda install ffmpeg"],
    ...     context={"requested_format": "mp4", "available_encoders": ["png", "gif"]}
    ... )
    """

    def __init__(
        self,
        message: str,
        details: Optional[str] = None,
        suggestions: Optional[List[str]] = None,
        error_code: str = "VISUALISATION_ERROR",
        context: Optional[Dict[str, Any]] = None,
    ):
        """Initialise the Error."""
        super().__init__(message, details, suggestions, error_code, context)


# Backward compatibility aliases and specific error constructors


def create_data_validation_error(message: str, data_info: Optional[Dict[str, Any]] = None, **kwargs) -> DataValidationError:
    """
    Create DataValidationError with common data context.

    Parameters
    ----------
    message : str
        Error message
    data_info : dict, optional
        Dictionary with data information (type, shape, dtype, etc.)
    **kwargs
        Additional arguments passed to DataValidationError

    Returns
    -------
    DataValidationError
        Configured exception with data context
    """
    context = kwargs.get("context", {})
    if data_info:
        context.update(data_info)
    kwargs["context"] = context
    return DataValidationError(message, **kwargs)


def create_coordinate_error(
    message: str,
    coordinate_ranges: Optional[Dict[str, tuple]] = None,
    detected_system: Optional[str] = None,
    **kwargs,
) -> CoordinateError:
    """
    Create CoordinateError with coordinate context.

    Parameters
    ----------
    message : str
        Error message
    coordinate_ranges : dict, optional
        Dictionary with coordinate ranges (e.g., {'lat': (-90, 90), 'lon': (0, 360)})
    detected_system : str, optional
        Auto-detected coordinate system
    **kwargs
        Additional arguments passed to CoordinateError

    Returns
    -------
    CoordinateError
        Configured exception with coordinate context
    """
    context = kwargs.get("context", {})
    if coordinate_ranges:
        context["coordinate_ranges"] = coordinate_ranges
    if detected_system:
        context["detected_system"] = detected_system
    kwargs["context"] = context
    return CoordinateError(message, **kwargs)


def create_processing_error(message: str, computation_info: Optional[Dict[str, Any]] = None, **kwargs) -> ProcessingError:
    """
    Create ProcessingError with computation context.

    Parameters
    ----------
    message : str
        Error message
    computation_info : dict, optional
        Dictionary with computation information (memory usage, chunk sizes, etc.)
    **kwargs
        Additional arguments passed to ProcessingError

    Returns
    -------
    ProcessingError
        Configured exception with computation context
    """
    context = kwargs.get("context", {})
    if computation_info:
        context.update(computation_info)
    kwargs["context"] = context
    return ProcessingError(message, **kwargs)


# Exception type mapping for backward compatibility
EXCEPTION_MAP: Dict[str, Type[MarExError]] = {
    "ValueError": DataValidationError,
    "RuntimeError": ProcessingError,
    "TypeError": DataValidationError,
    "KeyError": ConfigurationError,
    "AttributeError": ConfigurationError,
    "ImportError": DependencyError,
    "ModuleNotFoundError": DependencyError,
}


def wrap_exception(
    original_exception: Exception,
    message: Optional[str] = None,
    marex_exception_type: Optional[Type[MarExError]] = None,
) -> MarExError:
    """
    Wrap a generic exception in an appropriate MarEx exception.

    This function helps maintain backward compatibility while migrating
    to the new exception hierarchy by wrapping generic exceptions in
    appropriate MarEx-specific exceptions.

    Parameters
    ----------
    original_exception : Exception
        The original exception to wrap
    message : str, optional
        Custom message (uses original message if not provided)
    marex_exception_type : type, optional
        Specific MarEx exception type to use

    Returns
    -------
    MarExError
        Wrapped exception with original as cause
    """
    if marex_exception_type is None:
        exception_name = type(original_exception).__name__
        marex_exception_type = EXCEPTION_MAP.get(exception_name, MarExError)

    if message is None:
        message = str(original_exception)

    wrapped = marex_exception_type(
        message,
        details=f"Original {type(original_exception).__name__}: {str(original_exception)}",
        context={"original_exception_type": type(original_exception).__name__},
    )

    # Chain the original exception
    wrapped.__cause__ = original_exception
    return wrapped


# Export all exceptions
__all__ = [
    # Main exception hierarchy
    "MarExError",
    "DataValidationError",
    "CoordinateError",
    "ProcessingError",
    "ConfigurationError",
    "DependencyError",
    "TrackingError",
    "VisualisationError",
    # Convenience constructors
    "create_data_validation_error",
    "create_coordinate_error",
    "create_processing_error",
    "wrap_exception",
]

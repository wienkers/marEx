"""
MarEx Logging Configuration Module
==================================

The logger is designed to be efficient and non-intrusive, with minimal
performance impact on computations.
"""

import functools
import logging
import logging.handlers
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Dict, Optional, Union

import psutil
import xarray as xr

# Handle optional tqdm dependency for progress bars
try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    tqdm = None
    HAS_TQDM = False

# Default configuration
DEFAULT_LOG_LEVEL = logging.INFO
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - [PID:%(process)d] - %(message)s"
DEFAULT_VERBOSE_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - [PID:%(process)d] - %(funcName)s:%(lineno)d - %(message)s"
DEFAULT_QUIET_FORMAT = "%(levelname)s - %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Environment variable to control logging level
MAREX_LOG_LEVEL = os.environ.get("MAREX_LOG_LEVEL", "INFO").upper()
MAREX_LOG_FILE = os.environ.get("MAREX_LOG_FILE", None)
MAREX_VERBOSE = os.environ.get("MAREX_VERBOSE", "false").lower() in ("true", "1", "yes")
MAREX_QUIET = os.environ.get("MAREX_QUIET", "false").lower() in ("true", "1", "yes")

# Global logger instance and configuration state
_logger_configured = False
_current_verbosity_level = "normal"  # Can be 'quiet', 'normal', 'verbose'


def get_logger(name: str = "marEx") -> logging.Logger:
    """
    Get a configured logger instance for MarEx.

    Args:
        name: Logger name, typically the module name

    Returns:
        Configured logger instance
    """
    global _logger_configured

    logger = logging.getLogger(name)

    # Configure the root marEx logger only once
    if not _logger_configured and name == "marEx":
        configure_logging()
        _logger_configured = True

    return logger


def configure_logging(
    level: Optional[Union[int, str]] = None,
    format_str: Optional[str] = None,
    date_format: Optional[str] = None,
    log_file: Optional[Union[str, Path]] = None,
    console_output: bool = True,
    disable_external_loggers: bool = True,
    verbose: Optional[bool] = None,
    quiet: Optional[bool] = None,
) -> None:
    """
    Configure logging for the MarEx package.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_str: Custom log format string
        date_format: Date format for timestamps
        log_file: Optional file path for logging output
        console_output: Whether to log to console
        disable_external_loggers: Whether to suppress noisy external library logs
        verbose: Enable verbose mode (detailed logging with DEBUG level)
        quiet: Enable quiet mode (minimal logging with WARNING level)
    """
    global _current_verbosity_level

    # Handle verbose/quiet mode precedence
    if verbose is None:
        verbose = MAREX_VERBOSE
    if quiet is None:
        quiet = MAREX_QUIET

    # Quiet mode takes precedence over verbose
    if quiet and verbose:
        verbose = False

    # Determine log level based on verbose/quiet mode
    if quiet:
        _current_verbosity_level = "quiet"
        if level is None:
            level = logging.WARNING
    elif verbose:
        _current_verbosity_level = "verbose"
        if level is None:
            level = logging.DEBUG
    else:
        _current_verbosity_level = "normal"
        if level is None:
            level = getattr(logging, MAREX_LOG_LEVEL, DEFAULT_LOG_LEVEL)

    if isinstance(level, str):
        level = getattr(logging, level.upper(), DEFAULT_LOG_LEVEL)

    # Use environment variable for log file if not specified
    if log_file is None and MAREX_LOG_FILE:
        log_file = Path(MAREX_LOG_FILE)

    # Configure format based on verbosity level
    if format_str is None:
        if _current_verbosity_level == "verbose":
            format_str = DEFAULT_VERBOSE_FORMAT
        elif _current_verbosity_level == "quiet":
            format_str = DEFAULT_QUIET_FORMAT
        else:
            format_str = DEFAULT_LOG_FORMAT
    if date_format is None:
        date_format = DEFAULT_DATE_FORMAT

    formatter = logging.Formatter(format_str, datefmt=date_format)

    # Get root marEx logger
    root_logger = logging.getLogger("marEx")
    root_logger.setLevel(level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Add console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # Add file handler if requested
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        # Use rotating file handler to prevent huge log files
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=50 * 1024 * 1024,  # 50 MB
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Prevent propagation to avoid duplicate messages
    root_logger.propagate = False

    # Suppress noisy external library loggers
    if disable_external_loggers:
        _configure_external_loggers()

    # Log configuration
    root_logger.info(f"MarEx logging configured - Level: {logging.getLevelName(level)}, Mode: {_current_verbosity_level}")
    if log_file:
        root_logger.info(f"Logging to file: {log_file}")


def set_verbose_mode(verbose: bool = True) -> None:
    """
    Enable or disable verbose logging mode.

    Args:
        verbose: If True, enables verbose mode with DEBUG level logging
    """
    configure_logging(verbose=verbose, quiet=False)


def set_quiet_mode(quiet: bool = True) -> None:
    """
    Enable or disable quiet logging mode.

    Args:
        quiet: If True, enables quiet mode with WARNING level logging
    """
    configure_logging(quiet=quiet, verbose=False)


def set_normal_logging() -> None:
    """Reset logging to normal mode (INFO level)."""
    configure_logging(verbose=False, quiet=False)


def get_verbosity_level() -> str:
    """
    Get the current verbosity level.

    Returns:
        Current verbosity level: 'quiet', 'normal', or 'verbose'
    """
    return _current_verbosity_level


def is_verbose_mode() -> bool:
    """Check if verbose mode is currently enabled."""
    return _current_verbosity_level == "verbose"


def is_quiet_mode() -> bool:
    """Check if quiet mode is currently enabled."""
    return _current_verbosity_level == "quiet"


def _configure_external_loggers() -> None:
    """Configure external library loggers to reduce noise."""
    external_loggers = [
        "distributed.scheduler",
        "distributed.worker",
        "distributed.shuffle._scheduler_plugin",
        "distributed.comm",
        "distributed.core",
        "tornado.access",
        "asyncio",
        "matplotlib.font_manager",
        "PIL.PngImagePlugin",
    ]

    for logger_name in external_loggers:
        logging.getLogger(logger_name).setLevel(logging.ERROR)


def get_memory_usage() -> Dict[str, float]:
    """
    Get current memory usage information.

    Returns:
        Dictionary with memory usage statistics in MB
    """
    process = psutil.Process()
    memory_info = process.memory_info()

    return {
        "rss_mb": memory_info.rss / 1024 / 1024,  # Resident Set Size
        "vms_mb": memory_info.vms / 1024 / 1024,  # Virtual Memory Size
        "percent": process.memory_percent(),
        "available_mb": psutil.virtual_memory().available / 1024 / 1024,
    }


def log_memory_usage(logger: logging.Logger, message: str = "", level: int = logging.DEBUG) -> None:
    """
    Log current memory usage.

    Args:
        logger: Logger instance to use
        message: Additional message to include
        level: Log level to use
    """
    memory = get_memory_usage()
    log_msg = (
        f"Memory Usage - RSS: {memory['rss_mb']:.1f}MB, "
        f"Virtual: {memory['vms_mb']:.1f}MB, "
        f"Percent: {memory['percent']:.1f}%, "
        f"Available: {memory['available_mb']:.1f}MB"
    )

    if message:
        log_msg = f"{message} - {log_msg}"

    logger.log(level, log_msg)


@contextmanager
def log_timing(
    logger: logging.Logger,
    operation: str,
    level: int = logging.INFO,
    log_memory: bool = False,
    show_progress: bool = False,
):
    """
    Context manager to log operation timing.

    Args:
        logger: Logger instance to use
        operation: Description of the operation being timed
        level: Log level to use
        log_memory: Whether to log memory usage before and after
        show_progress: Whether to show progress information (if in verbose mode)

    Example:
        >>> logger = get_logger(__name__)
        >>> with log_timing(logger, "Data preprocessing", log_memory=True):
        ...     # expensive operation
        ...     pass
    """
    start_time = time.perf_counter()

    # Enhanced logging for verbose mode
    if is_verbose_mode() and show_progress:
        logger.debug(f"Initializing {operation}")

    if log_memory and (is_verbose_mode() or level <= logging.INFO):
        log_memory_usage(logger, f"Before {operation}", level=logging.DEBUG)

    logger.log(level, f"Starting {operation}")

    try:
        yield
        end_time = time.perf_counter()
        duration = end_time - start_time

        if log_memory and (is_verbose_mode() or level <= logging.INFO):
            log_memory_usage(logger, f"After {operation}", level=logging.DEBUG)

        # More detailed timing in verbose mode
        if is_verbose_mode():
            logger.debug(f"Completed {operation} - Duration: {duration:.3f}s, " f"Performance: {1/duration:.2f} ops/sec")
        else:
            logger.log(level, f"Completed {operation} in {duration:.2f}s")

    except Exception as e:
        end_time = time.perf_counter()
        duration = end_time - start_time
        logger.error(f"Failed {operation} after {duration:.2f}s: {e}")
        raise


def create_progress_bar(
    total: Optional[int] = None,
    desc: str = "Processing",
    unit: str = "it",
    disable: Optional[bool] = None,
) -> Optional[tqdm]:
    """
    Create a progress bar if tqdm is available and not in quiet mode.

    Args:
        total: Total number of iterations
        desc: Description for the progress bar
        unit: Unit of measurement
        disable: Explicitly disable progress bar

    Returns:
        tqdm instance or None if not available/disabled
    """
    if disable is None:
        # Auto-disable in quiet mode or if tqdm not available
        disable = is_quiet_mode() or not HAS_TQDM

    if disable or not HAS_TQDM:
        return None

    # Only show progress bar in normal or verbose mode
    return tqdm(
        total=total,
        desc=desc,
        unit=unit,
        disable=disable,
        ascii=True if os.environ.get("TERM") != "xterm-256color" else False,
    )


@contextmanager
def progress_bar(
    total: Optional[int] = None,
    desc: str = "Processing",
    unit: str = "it",
    logger: Optional[logging.Logger] = None,
):
    """
    Context manager for progress bars with logging integration.

    Args:
        total: Total number of iterations
        desc: Description for the progress bar
        unit: Unit of measurement
        logger: Optional logger for fallback progress reporting

    Example:
        >>> with progress_bar(100, "Processing data") as pbar:
        ...     for i in range(100):
        ...         # do work
        ...         if pbar:
        ...             pbar.update(1)
    """
    pbar = create_progress_bar(total=total, desc=desc, unit=unit)

    try:
        yield pbar
    finally:
        if pbar is not None:
            pbar.close()
        elif logger and not is_quiet_mode():
            # Fallback logging if no progress bar
            logger.info(f"Completed {desc}")


def log_progress(
    logger: logging.Logger,
    current: int,
    total: int,
    operation: str = "Processing",
    frequency: int = 10,
) -> None:
    """
    Log progress information without using progress bars.

    Args:
        logger: Logger instance
        current: Current progress count
        total: Total count
        operation: Description of the operation
        frequency: Log every N percent (default: 10%)
    """
    if is_quiet_mode():
        return

    if total <= 0:
        return

    percentage = (current / total) * 100

    # Log at frequency intervals
    if percentage % frequency == 0 or current == total:
        if is_verbose_mode():
            logger.debug(
                f"{operation}: {current}/{total} ({percentage:.1f}%) - " f"Rate: {current/(time.perf_counter()):.2f} items/sec"
            )
        else:
            logger.info(f"{operation}: {percentage:.0f}% complete ({current}/{total})")


def log_function_call(logger: Optional[logging.Logger] = None, level: int = logging.DEBUG):
    """
    Log function calls with parameters and timing.

    Args:
        logger: Logger instance (defaults to function's module logger)
        level: Log level to use

    Example:
        >>> @log_function_call()
        ... def my_function(x, y=10):
        ...     return x + y
    """

    def decorator(func: Callable) -> Callable:
        nonlocal logger
        if logger is None:
            logger = get_logger(func.__module__)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Log function call
            func_name = f"{func.__module__}.{func.__name__}"

            # Create parameter summary (limit length for readability)
            args_repr = [repr(arg) for arg in args[:3]]  # First 3 args only
            if len(args) > 3:
                args_repr.append(f"... +{len(args)-3} more")

            kwargs_repr = [f"{k}={repr(v)}" for k, v in list(kwargs.items())[:3]]
            if len(kwargs) > 3:
                kwargs_repr.append(f"... +{len(kwargs)-3} more")

            params = ", ".join(args_repr + kwargs_repr)
            if len(params) > 200:
                params = params[:200] + "..."

            logger.log(level, f"Calling {func_name}({params})")

            # Time the execution
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                end_time = time.perf_counter()
                duration = end_time - start_time
                logger.log(level, f"Completed {func_name} in {duration:.3f}s")
                return result
            except Exception as e:
                end_time = time.perf_counter()
                duration = end_time - start_time
                logger.error(f"Failed {func_name} after {duration:.3f}s: {e}")
                raise

        return wrapper

    return decorator


def log_dask_info(
    logger: logging.Logger,
    da_or_ds: Union["xr.DataArray", "xr.Dataset"],
    message: str = "",
) -> None:
    """
    Log information about Dask arrays/datasets.

    Args:
        logger: Logger instance
        da_or_ds: Dask-backed xarray object
        message: Additional context message
    """
    try:
        from dask.base import is_dask_collection

        if hasattr(da_or_ds, "chunks"):
            chunks_info = str(da_or_ds.chunks)
            if len(chunks_info) > 100:
                chunks_info = chunks_info[:100] + "..."

            nbytes = da_or_ds.nbytes if hasattr(da_or_ds, "nbytes") else "unknown"

            log_msg = f"Dask object - Shape: {da_or_ds.shape}, " f"Chunks: {chunks_info}, " f"Size: {nbytes}"

            if message:
                log_msg = f"{message} - {log_msg}"

            logger.debug(log_msg)

            # Log task graph size if available
            if is_dask_collection(da_or_ds):
                graph_size = len(da_or_ds.__dask_graph__())
                logger.debug(f"Dask graph size: {graph_size} tasks")

    except Exception as e:
        logger.debug(f"Could not log Dask info: {e}")


# Convenience function for backward compatibility
def setup_logging(*args, **kwargs) -> None:
    """Alias for configure_logging for backward compatibility."""
    configure_logging(*args, **kwargs)

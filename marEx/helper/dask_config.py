"""
Dask configuration for HPC environments.
--------------------------------------------------------------------

Provides the default Dask configuration tuned for large-scale climate
data processing and the :func:`configure_dask` helper used to apply it.
This is the leaf module of the helper package: it has no cross-module
dependencies within ``marEx.helper``.
"""

from getpass import getuser
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Optional, Union

import dask

from ..logging_config import get_logger

# Get module logger
logger = get_logger(__name__)


DEFAULT_DASK_CONFIG = {
    # Array processing
    "array.slicing.split_large_chunks": False,
    "array.chunk-size": "24MiB",  # Optimal chunk size for oceanographic data
    # Worker memory management
    "distributed.worker.memory.target": 0.4,  # Target memory threshold for spilling to disk
    "distributed.worker.memory.spill": 0.5,  # Spill to disk threshold
    "distributed.worker.memory.pause": 0.6,  # Pause worker threshold
    "distributed.worker.memory.terminate": 0.8,  # Terminate worker threshold
    "distributed.worker.memory.recent-to-old-time": "10s",  # Time to consider data old
    "distributed.worker.daemon": False,  # Workers are not daemons
    # Scheduler stability settings
    "distributed.scheduler.allowed-failures": 50,  # Allow many retries (common on HPC)
    "distributed.scheduler.work-stealing": False,  # Disable for deterministic execution
    "distributed.scheduler.worker-ttl": "600s",  # Keep workers alive for 10 minutes
    # Communication timeouts - increased for HPC network latency
    "distributed.comm.timeouts.connect": "300s",  # Connection timeout
    "distributed.comm.timeouts.tcp": "300s",  # TCP timeout
    "distributed.comm.retry.count": 15,  # More retries before giving up
    "distributed.comm.retry.delay.min": "3s",  # Min delay between retries
    "distributed.comm.retry.delay.max": "30s",  # Max delay between retries
    # Admin and logging
    "distributed.admin.log-format": "%(name)s - %(levelname)s - %(message)s",  # Log format
}

DKRZ_SCRATCH_PATH = Path("/scratch") / getuser()[0] / getuser() / "clients"


def configure_dask(
    scratch_dir: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> TemporaryDirectory:  # pragma: no cover
    """
    Configure Dask with appropriate settings for HPC environments.

    Parameters
    ----------
    scratch_dir : str or Path, optional
        Directory to use for temporary files.
    config : dict, optional
        Additional Dask configuration settings to apply.

    Returns
    -------
    TemporaryDirectory
        Temporary directory object that should be kept alive while Dask is in use.
    """
    logger.info("Configuring Dask for HPC environment")

    # Use provided scratch directory or default to DKRZ scratch
    scratch_path = Path(scratch_dir) if scratch_dir else DKRZ_SCRATCH_PATH
    logger.debug(f"Using scratch directory: {scratch_path}")

    # Create temporary directory
    if not scratch_path.exists():
        logger.debug(f"Creating scratch directory: {scratch_path}")
        scratch_path.mkdir(parents=True, exist_ok=True)

    temp_dir = TemporaryDirectory(dir=scratch_path)
    logger.info(f"Dask temporary directory: {temp_dir.name}")

    # Apply default configuration
    dask.config.set(temporary_directory=temp_dir.name)

    # Apply default settings
    logger.debug("Applying default Dask configuration")
    for key, value in DEFAULT_DASK_CONFIG.items():
        dask.config.set({key: value})
        logger.debug(f"Set Dask config: {key} = {value}")

    # Apply any additional configuration
    if config:
        logger.debug(f"Applying additional Dask configuration: {config}")
        dask.config.set(config)

    logger.info("Dask configuration completed")
    return temp_dir

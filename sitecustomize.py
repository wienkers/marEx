"""Coverage subprocess initialisation for multiprocessing support."""

import os

import coverage

# Start coverage in subprocesses if COVERAGE_PROCESS_START is set
if "COVERAGE_PROCESS_START" in os.environ:
    coverage.process_startup()

"""Test configuration and fixtures for marEx package.

One bounded Dask cluster per xdist worker (``client``), the library's own memory thresholds,
and memory warnings that stay visible. Everything set on dask's config is scoped to the
session and restored afterwards.
"""

import contextlib
import gc
import logging
import os
import time
import traceback
import uuid
from pathlib import Path

import dask
import numpy as np
import psutil
import pytest
from dask.distributed import Client, LocalCluster
from dask.utils import format_bytes, parse_bytes

logger = logging.getLogger(__name__)

N_CLUSTER_WORKERS = 2
MEMORY_LOGGERS = ("distributed.worker.memory", "distributed.nanny.memory")
CLIENT_FIXTURES = frozenset(("client",))
# Module-private fixtures that return their own distributed Client. A test that requests one
# computes on that Client, so it is not pinned to the threaded scheduler either.
PRIVATE_CLIENT_FIXTURES = frozenset(("inprocess_client", "ltm_client"))

# The session client, once a test has asked for it. Autouse fixtures read it from here so that
# modules which never request it do not start a cluster.
_SESSION = {}


def pytest_configure(config):
    """Configure pytest with custom logging settings."""
    # Configure matplotlib to use non-interactive backend for CI/testing
    # This must be done before importing pyplot to avoid TkAgg errors on Windows
    import matplotlib

    matplotlib.use("Agg")

    # Tornado's access log is noise; the distributed loggers are left alone so that memory
    # warnings ("Worker ... exceeded 95% memory budget") stay visible.
    logging.getLogger("tornado").setLevel(logging.ERROR)


def _n_xdist():
    return int(os.environ.get("PYTEST_XDIST_WORKER_COUNT", "1"))


def _usable_cpus():
    """CPUs this process may run on; ``os.cpu_count()`` reports the whole node under SLURM."""
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def _cgroup_memory_limit():
    """Smallest cgroup memory limit on the path from this process's cgroup to the root, or None.

    A SLURM job's limit sits on the job's cgroup, a few levels above the task's own, whose
    ``memory.max`` reads ``max``; Dask and psutil see neither and report the whole node.
    """
    try:
        lines = Path("/proc/self/cgroup").read_text().splitlines()
    except OSError:
        return None
    limits = []
    for line in lines:
        _, controllers, path = line.split(":", 2)
        if controllers == "":  # cgroup v2
            root, name = Path("/sys/fs/cgroup"), "memory.max"
        elif "memory" in controllers.split(","):  # cgroup v1
            root, name = Path("/sys/fs/cgroup/memory"), "memory.limit_in_bytes"
        else:
            continue
        node = root / path.lstrip("/")
        while True:
            try:
                value = (node / name).read_text().strip()
            except OSError:
                value = ""
            if value.isdigit():
                limits.append(int(value))
            if node == root or root not in node.parents:
                break
            node = node.parent
    return min(limits) if limits else None


def _memory_ceiling():
    """The memory this run may use: the smallest of RAM, the cgroup limit and SLURM's grant."""
    sources = {"psutil": psutil.virtual_memory().total}
    cgroup = _cgroup_memory_limit()
    if cgroup is not None:
        sources["cgroup"] = cgroup
    slurm = os.environ.get("SLURM_MEM_PER_NODE", "")
    if slurm.isdigit():
        sources["SLURM_MEM_PER_NODE"] = int(slurm) * 2**20
    source = min(sources, key=sources.get)
    return sources[source], source


def worker_memory_limit(n_xdist=None):
    """Per-worker ``memory_limit`` in bytes for the test cluster, and where it came from.

    ``MAREX_TEST_WORKER_MEMORY`` (bytes or a dask string such as ``"2.5GB"``) wins; otherwise
    60 % of the usable memory is shared between every worker of every xdist process.
    """
    requested = os.environ.get("MAREX_TEST_WORKER_MEMORY")
    if requested:
        return parse_bytes(requested), "MAREX_TEST_WORKER_MEMORY"
    total, source = _memory_ceiling()
    n_xdist = n_xdist or _n_xdist()
    limit = int(0.6 * total / (n_xdist * N_CLUSTER_WORKERS))
    return limit, f"0.6 x {format_bytes(total)} ({source}) / ({n_xdist} xdist x {N_CLUSTER_WORKERS} workers)"


def _threaded_num_workers():
    # The seasonal histogram path runs one budgeted tile (~50M elements, several int
    # intermediates) per thread: one thread per core SIGKILLed a 16 GB interactive node, and
    # every xdist process spawning a full pool oversubscribes the machine. An explicit
    # ``DASK_NUM_WORKERS`` may lower the cap, never raise it.
    cap = max(1, min(4, _usable_cpus() // _n_xdist()))
    requested = dask.config.get("num_workers", None)
    return min(cap, int(requested)) if requested else cap


@pytest.fixture(scope="session", autouse=True)
def configure_dask():
    """Configure Dask for testing environment; restored when the session ends."""
    with dask.config.set(
        {
            # Distributed clients ignore this; it bounds the local threaded scheduler.
            "num_workers": _threaded_num_workers(),
            "array.chunk-size": "32MB",
            "array.slicing.split_large_chunks": True,
            # Add optimisation settings for complex computations
            "optimization.fuse.ave-width": 2,
            "optimization.fuse.max-width": 4,
            "optimization.fuse.max-depth": 4,
            # Optimise for memory-intensive reductions
            "array.chunk-options.split-every": {"reduction-dimension": 4},
        }
    ):
        yield


def pytest_report_header(config):
    n = getattr(config.option, "numprocesses", None)
    limit, origin = worker_memory_limit(n if isinstance(n, int) and n > 0 else 1)
    return f"marEx test cluster: {N_CLUSTER_WORKERS} workers x {format_bytes(limit)} per xdist process, from {origin}"


@pytest.fixture(scope="session")
def client(configure_dask, tmp_path_factory):
    """The one Dask cluster of this xdist process: 2 single-threaded worker processes, bounded memory."""
    limit, origin = worker_memory_limit()
    cluster = LocalCluster(
        n_workers=N_CLUSTER_WORKERS,
        threads_per_worker=1,
        processes=True,
        memory_limit=limit,
        local_directory=str(tmp_path_factory.mktemp("dask-worker-space")),
        dashboard_address=None,
        silence_logs=logging.WARNING,
    )
    session_client = Client(cluster)
    try:
        session_client.wait_for_workers(N_CLUSTER_WORKERS, timeout=60)
        # Dask caps a requested limit at what it believes the machine has, so read back the
        # limit the workers actually run under rather than trusting the request.
        effective = sorted(w["memory_limit"] for w in session_client.scheduler_info()["workers"].values())
        logger.info(f"Test cluster: {N_CLUSTER_WORKERS} workers, memory_limit {format_bytes(limit)} per worker ({origin})")
        assert effective == [limit] * N_CLUSTER_WORKERS, (
            f"workers run with memory_limit {[format_bytes(e) for e in effective]}, requested {format_bytes(limit)} ({origin})"
        )
        with capture_memory_warnings(session_client):  # fails here unless worker and nanny warnings arrive
            pass
        _SESSION["client"] = session_client
        yield session_client
    finally:
        _SESSION.pop("client", None)
        session_client.close()
        cluster.close()


def _held(session_client):
    """The session client's live futures: key -> its future state (a new object for a key made again after a release).

    Keeping a state object does not keep its future alive: the client counts references to the key.
    """
    return dict(session_client.futures.copy())


@pytest.fixture(scope="module", autouse=True)
def _client_holds_no_futures():
    """A module that leaves futures on the session client fails; the client itself stays up.

    A future counts against the module when it is held at the end and is not the same future that was
    already held at the start: a leak is reported by the module that made it, and a later module that
    leaks the same key again, after the first copy was released, is reported too.

    The start snapshot is taken after collecting garbage. Without that, an earlier module's leak held only
    by an uncollected reference cycle would still be in the snapshot, and this module re-leaking the same
    key would reuse that future and go unreported.
    """
    session_client = _SESSION.get("client")
    before = {}
    if session_client is not None:
        gc.collect()
        before = _held(session_client)
    yield
    session_client = _SESSION.get("client")
    if session_client is None:
        return
    gc.collect()

    def new(held):
        return {key for key, state in held.items() if before.get(key) is not state}

    # A dropped future is released on the client's event loop, not at once: let that drain.
    deadline = time.monotonic() + 10
    while new(held := _held(session_client)) and time.monotonic() < deadline:
        time.sleep(0.05)
    leaked = sorted(str(key) for key in new(held))
    assert not leaked, f"{len(leaked)} futures still held by the session client after this module: {leaked[:5]}"


# The pin a test that does not request the cluster runs under (``_threaded_unless_the_cluster_is_requested``).
THREADED = {"scheduler": "threads", "array.rechunk.method": "tasks"}


def on_client(client):
    """True when dask's current scheduler is ``client``."""
    return getattr(dask.base.get_scheduler(), "__self__", None) is client


def check(failures, case, func):
    """Run one table row's assertions; record the first that fails, or the error the row raised, under its case id."""
    try:
        func()
    except (AssertionError, pytest.fail.Exception) as e:
        failures.append(f"{case}: {str(e).strip().splitlines()[0] if str(e).strip() else 'assertion failed'}")
    except Exception as e:  # noqa: BLE001 - a row that raises is a failing row; the others still run
        failures.append(f"{case}: raised {type(e).__name__}: {e}")


def assert_no_failures(failures):
    assert not failures, f"{len(failures)} row(s) failed: " + " | ".join(failures)


def released(run, what="the run"):
    """Return ``run()``; if it raises, raise an ``AssertionError`` carrying the error and where it came from, as text.

    The traceback of a failed run holds its persisted results (a tracker's intermediates, a pipeline's
    output), which would leak futures past the module and turn the real failure into a teardown error.
    ``run`` keeps them in its own frame; only strings leave it: the error's type and message, and the
    last three frames of its traceback.
    """
    try:
        return run()
    except Exception as e:  # noqa: BLE001 - re-raised below without its traceback
        failure = f"{type(e).__name__}: {e}"
        frames = traceback.extract_tb(e.__traceback__)[-3:]
        # The locations also go on the first line: a table row (``check``) keeps only that line.
        at = "; ".join(f"{Path(f.filename).name}:{f.lineno} in {f.name}" for f in frames)
        where = "".join(traceback.format_list(frames))
    raise AssertionError(f"{what} raised {failure} (at {at})\n{where}")


# Statistical Test Helper Functions
def assert_percentile_frequency(
    frequency,
    expected_percentile,
    tolerance_std=2.0,
    sample_size=None,
    description=None,
):
    """
    Test that observed frequency matches expected percentile with statistical
    significance.

    For a percentile threshold (e.g., 95th percentile), we expect the frequency
    to be close to (100 - percentile)/100. This uses a binomial distribution to
    test if the observed frequency is statistically reasonable.

    Parameters:
    -----------
    frequency : float
        Observed frequency of extreme events (between 0 and 1)
    expected_percentile : float
        The percentile threshold used (e.g., 95 for 95th percentile)
    tolerance_std : float, default=2.0
        Number of standard deviations to allow as tolerance
    sample_size : int, optional
        Sample size for more precise statistical test. If None, uses approximate test.

    Raises:
    -------
    AssertionError : If frequency is outside statistically expected range
    """
    expected_frequency = (100 - expected_percentile) / 100.0

    if sample_size is not None:
        # Use exact binomial test with known sample size
        # Calculate confidence interval for binomial distribution
        std_error = np.sqrt(expected_frequency * (1 - expected_frequency) / sample_size)
        lower_bound = expected_frequency - tolerance_std * std_error
        upper_bound = expected_frequency + tolerance_std * std_error

        desc_suffix = f" ({description})" if description else ""
        assert lower_bound <= frequency <= upper_bound, (
            f"Extreme frequency {frequency:.4f} is outside statistically "
            f"expected range [{lower_bound:.4f}, {upper_bound:.4f}] for "
            f"{expected_percentile}th percentile (expected: "
            f"{expected_frequency:.4f}, sample_size: {sample_size})"
            f"{desc_suffix}"
        )
    else:
        # Use approximate test with reasonable tolerance for unknown sample size
        # For typical dataset sizes, allow broader tolerance
        relative_tolerance = 0.20  # 20% relative tolerance
        absolute_tolerance = max(0.005, expected_frequency * relative_tolerance)

        lower_bound = expected_frequency - absolute_tolerance
        upper_bound = expected_frequency + absolute_tolerance

        desc_suffix = f" ({description})" if description else ""
        assert lower_bound <= frequency <= upper_bound, (
            f"Extreme frequency {frequency:.4f} is outside expected range "
            f"[{lower_bound:.4f}, {upper_bound:.4f}] for "
            f"{expected_percentile}th percentile (expected: "
            f"{expected_frequency:.4f} ± {absolute_tolerance:.4f})"
            f"{desc_suffix}"
        )


def assert_reasonable_bounds(
    value,
    expected_value,
    tolerance_relative=0.1,
    tolerance_absolute=None,
    description="value",
):
    """
    Test that a value is within reasonable bounds of an expected value.

    Uses relative tolerance by default, with optional absolute tolerance override.

    Parameters:
    -----------
    value : float
        Observed value to test
    expected_value : float
        Expected/reference value
    tolerance_relative : float, default=0.1
        Relative tolerance as fraction (e.g., 0.1 = 10%)
    tolerance_absolute : float, optional
        If provided, uses absolute tolerance instead of relative
    description : str
        Description of what is being tested for error messages
    """
    if tolerance_absolute is not None:
        lower_bound = expected_value - tolerance_absolute
        upper_bound = expected_value + tolerance_absolute
        tolerance_desc = f"±{tolerance_absolute}"
    else:
        tolerance_abs = abs(expected_value * tolerance_relative)
        lower_bound = expected_value - tolerance_abs
        upper_bound = expected_value + tolerance_abs
        tolerance_desc = f"±{tolerance_relative * 100:.1f}%"

    assert lower_bound <= value <= upper_bound, (
        f"{description} {value} is outside reasonable bounds "
        f"[{lower_bound:.4f}, {upper_bound:.4f}] "
        f"(expected: {expected_value} {tolerance_desc})"
    )


def assert_count_in_reasonable_range(count, expected_count, tolerance=2):
    """
    Test that an integer count is within a reasonable range of expected count.

    For counting discrete objects (events, objects, etc.) where small variations
    are expected due to algorithmic differences or test data variations.

    Parameters:
    -----------
    count : int
        Observed count
    expected_count : int
        Expected count
    tolerance : int, default=2
        Absolute tolerance for count differences
    """
    lower_bound = expected_count - tolerance
    upper_bound = expected_count + tolerance

    assert lower_bound <= count <= upper_bound, (
        f"Count {count} is outside reasonable range [{lower_bound}, {upper_bound}] (expected: {expected_count} ± {tolerance})"
    )


def assert_statistical_consistency(
    data_array,
    expected_property,
    test_type="mean",
    tolerance_std=2.0,
    description="data",
):
    """
    Test statistical properties of data arrays for consistency.

    Parameters:
    -----------
    data_array : array-like
        Data to test
    expected_property : float
        Expected value of the statistical property
    test_type : str
        Type of test: 'mean', 'std', 'median', 'percentile_95', etc.
    tolerance_std : float
        Number of standard deviations for tolerance
    description : str
        Description for error messages
    """
    data = np.array(data_array).flatten()
    data = data[~np.isnan(data)]  # Remove NaN values

    if test_type == "mean":
        observed = np.mean(data)
        # Use t-test for mean comparison
        std_error = np.std(data) / np.sqrt(len(data))
        tolerance = tolerance_std * std_error
    elif test_type == "std":
        observed = np.std(data)
        tolerance = expected_property * 0.2  # 20% tolerance for std
    elif test_type == "median":
        observed = np.median(data)
        tolerance = np.std(data) / np.sqrt(len(data)) * tolerance_std
    else:
        raise ValueError(f"Unsupported test_type: {test_type}")

    lower_bound = expected_property - tolerance
    upper_bound = expected_property + tolerance

    assert lower_bound <= observed <= upper_bound, (
        f"{description} {test_type} {observed:.4f} is outside expected range "
        f"[{lower_bound:.4f}, {upper_bound:.4f}] (expected: {expected_property:.4f})"
    )


PROBE_TEXT = "marex-memory-capture-probe"


class _RecordList(logging.Handler):
    """Collect LogRecords at or above ``level``; capture probes only when ``probes``."""

    def __init__(self, level=logging.WARNING, probes=False):
        super().__init__(level=level)
        self.records = []
        self.probes = probes

    def emit(self, record):
        if self.probes or PROBE_TEXT not in record.getMessage():
            self.records.append(record)


# Memory-warning captures currently open, and the test phase running now, for the report.
_ACTIVE_CAPTURES = []
_PHASE = {}


def _report_text(records):
    return "\n".join(f"{r.name} {r.levelname}: {r.getMessage()}" for r in records)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Put the memory warnings recorded so far into the report of the phase that just ran."""
    outcome = yield
    records = [r for handler in _ACTIVE_CAPTURES for r in handler.records]
    if records:
        outcome.get_result().sections.append((f"distributed memory warnings ({call.when})", _report_text(records)))


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    _PHASE["item"] = item
    yield
    _PHASE.pop("item", None)


def _probe(handler, client, label, nanny, timeout):
    """Emit one identifiable WARNING from every worker (and on the parent's nanny logger), wait for all.

    A worker's probe is re-sent each second until it arrives: the first ``forward_logging`` of a
    client subscribes to its topic asynchronously, and a record published before the scheduler
    has the subscription is dropped.
    """
    token = f"{PROBE_TEXT} {label} {uuid.uuid4().hex}"
    pending = {f"{token} from {address}": address for address in client.scheduler_info()["workers"]}
    if nanny:
        pending[f"{token} from nanny"] = None
        logging.getLogger(MEMORY_LOGGERS[1]).warning(f"{token} from nanny")
    deadline, resend = time.monotonic() + timeout, 0.0
    while True:
        pending = {m: a for m, a in pending.items() if m not in {r.getMessage() for r in list(handler.records)}}
        if not pending:
            break
        if time.monotonic() > deadline:
            raise TimeoutError(f"memory-warning capture is not live: no probe record within {timeout} s for {sorted(pending)}")
        if time.monotonic() >= resend:
            for message, address in pending.items():
                if address is not None:
                    client.run(logging.getLogger(MEMORY_LOGGERS[0]).warning, message, workers=[address])
            resend = time.monotonic() + 1.0
        time.sleep(0.02)
    handler.records[:] = [r for r in handler.records if PROBE_TEXT not in r.getMessage()]


@contextlib.contextmanager
def capture_memory_warnings(client, probe=True, timeout=30):
    """Collect the ``distributed.{worker,nanny}.memory`` WARNINGs raised while the block runs.

    Workers are separate processes, so their records are forwarded to this process with
    ``client.forward_logging``; a ``LocalCluster``'s nannies run here, so their logger is
    read directly. With ``probe``, one identifiable WARNING is injected from every worker and
    on the nanny logger before the block runs and must arrive within ``timeout`` seconds,
    otherwise the capture is not live and this raises; on exit every worker is flushed the same
    way, so a warning raised inside the block cannot arrive after ``.records`` is read. The
    probes are removed: ``.records`` holds only what the block produced. Anything recorded is
    attached to the current test's report.
    """
    handler = _RecordList(logging.WARNING, probes=probe)
    _ACTIVE_CAPTURES.append(handler)
    for name in MEMORY_LOGGERS:
        logging.getLogger(name).addHandler(handler)
    try:
        client.forward_logging(MEMORY_LOGGERS[0], level=logging.WARNING)
        if probe:
            _probe(handler, client, "entry", nanny=True, timeout=timeout)
        yield handler
        if probe:
            _probe(handler, client, "exit", nanny=False, timeout=timeout)
    finally:
        for name in MEMORY_LOGGERS:
            logging.getLogger(name).removeHandler(handler)
        _ACTIVE_CAPTURES.remove(handler)
        item = _PHASE.get("item")
        if handler.records and item is not None:
            item.add_report_section("call", "distributed memory warnings", _report_text(handler.records))


@pytest.fixture(autouse=True)
def _report_memory_warnings(request):
    """Attach any distributed memory warning raised during a test that uses the session client."""
    if not CLIENT_FIXTURES.intersection(request.fixturenames):
        yield
        return
    with capture_memory_warnings(request.getfixturevalue("client"), probe=False):
        yield


@pytest.fixture(autouse=True)
def _threaded_unless_the_cluster_is_requested(request):
    """A test that does not ask for the session client runs on dask's threaded scheduler.

    Once a test has started the session client it is every later test's default scheduler in
    this process, so without this a test's speed, and the rechunk method dask picks for it,
    would depend on which tests ran before it: a table of small computes ran several times
    slower through two single-threaded worker processes. Rechunks are pinned to ``tasks``
    because dask picks P2P whenever a distributed client exists, and P2P needs a worker. The
    threaded scheduler keeps the ``num_workers`` cap set in ``configure_dask``. A test that
    requests a module's own client fixture (``PRIVATE_CLIENT_FIXTURES``) is left on that client.
    """
    if (CLIENT_FIXTURES | PRIVATE_CLIENT_FIXTURES).intersection(request.fixturenames):
        yield
        return
    with dask.config.set({"scheduler": "threads", "array.rechunk.method": "tasks"}):
        yield


class _PinnedHandlers(list):
    """A handler list that keeps its ``pinned`` handlers through ``clear()``, which ``configure_logging`` calls."""

    def __init__(self, handlers, pinned):
        super().__init__(handlers)
        self.pinned = list(getattr(handlers, "pinned", [])) + [pinned]
        self.append(pinned)

    def clear(self):
        super().clear()
        self.extend(self.pinned)


@pytest.fixture
def marex_records():
    """``marex_records(level)`` -> the list of LogRecords the ``marEx`` logger handles from now on.

    ``caplog`` cannot see them: ``configure_logging`` sets ``propagate = False`` on the package
    logger, and clears its handlers. The capture survives that; the logger's level is set to
    ``level`` here and is whatever ``configure_logging`` sets after that. Level, propagation and
    handlers are restored on teardown.
    """
    target = logging.getLogger("marEx")
    saved = (target.level, target.propagate, target.handlers)

    def capture(level=logging.DEBUG):
        handler = _RecordList(level)
        target.handlers = _PinnedHandlers(target.handlers, handler)
        target.setLevel(min(level, target.getEffectiveLevel()))
        return handler.records

    yield capture
    target.setLevel(saved[0])
    target.propagate = saved[1]
    target.handlers = saved[2]


class _MarExWarningCapture(logging.Handler):
    """Collect WARNING records straight off the ``marEx`` logger.

    ``caplog`` cannot see them: :func:`marEx.logging_config.configure_logging` sets
    ``propagate = False`` on the package root logger, so nothing reaches pytest's
    handler on the true root.
    """

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())

    def __enter__(self):
        self.messages.clear()
        logging.getLogger("marEx").addHandler(self)
        return self

    def __exit__(self, *exc):
        logging.getLogger("marEx").removeHandler(self)
        return False


@pytest.fixture
def marex_warnings():
    """Context manager yielding the marEx WARNING records emitted inside its block."""
    return _MarExWarningCapture()

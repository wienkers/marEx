"""The marEx logging system: verbosity state, destinations and environment, the timing, progress,
function-call and dask/memory helpers, the external-logger levels, and the verbose/quiet flags of
each pipeline stage.

Every test starts at normal verbosity and leaves the logging state (the verbosity level, the
configured flag, the package logger's level, propagation and handlers, and the external loggers'
levels) as it found it. A "flag took effect" row first forces the opposite mode, so a flag that is
ignored fails the row instead of matching a default.
"""

import contextlib
import importlib.util
import io
import logging
import os
import re
import subprocess
import sys
import time
from unittest import mock

import dask
import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import marEx
import marEx.logging_config as lc
from marEx.logging_config import (
    _configure_external_loggers,
    configure_logging,
    create_progress_bar,
    get_logger,
    get_memory_usage,
    get_verbosity_level,
    is_quiet_mode,
    is_verbose_mode,
    log_dask_info,
    log_function_call,
    log_memory_usage,
    log_progress,
    log_timing,
    progress_bar,
    set_normal_logging,
    set_quiet_mode,
    set_verbose_mode,
    setup_logging,
)

from .conftest import THREADED, assert_no_failures, check, on_client

# Every logger ``_configure_external_loggers`` sets a level on.
EXTERNAL_LOGGERS = (
    "distributed.scheduler",
    "distributed.worker",
    "distributed.shuffle._scheduler_plugin",
    "distributed.comm",
    "distributed.core",
    "tornado.access",
    "asyncio",
    "matplotlib.font_manager",
    "PIL.PngImagePlugin",
    "distributed.worker.memory",
    "distributed.nanny.memory",
)
MEMORY_LOGGERS = ("distributed.worker.memory", "distributed.nanny.memory")
NOISY_LOGGERS = EXTERNAL_LOGGERS[:9]

# A logger outside the ``marEx`` tree, so ``configure_logging`` never touches it.
PROBE = "tests.logging_probe"


def _logging_state():
    root = logging.getLogger("marEx")
    return {
        "verbosity": lc._current_verbosity_level,
        "configured": lc._logger_configured,
        "level": root.level,
        "propagate": root.propagate,
        "handlers": list(root.handlers),
        "external": {name: logging.getLogger(name).level for name in EXTERNAL_LOGGERS},
    }


def _restore_logging(saved):
    root = logging.getLogger("marEx")
    for handler in list(root.handlers):
        if handler not in saved["handlers"]:
            handler.close()
    lc._current_verbosity_level = saved["verbosity"]
    lc._logger_configured = saved["configured"]
    root.setLevel(saved["level"])
    root.propagate = saved["propagate"]
    root.handlers = list(saved["handlers"])
    for name, level in saved["external"].items():
        logging.getLogger(name).setLevel(level)


@pytest.fixture(autouse=True)
def _logging_state_restored():
    """Start at normal verbosity; put the whole logging state back afterwards."""
    saved = _logging_state()
    set_normal_logging()
    yield
    _restore_logging(saved)


class _Records(logging.Handler):
    def __init__(self, level):
        super().__init__(level=level)
        self.records = []

    def emit(self, record):
        self.records.append(record)


@contextlib.contextmanager
def _recorded(name=PROBE, level=logging.DEBUG):
    """The records ``name`` handles at ``level`` and above while the block runs."""
    logger = logging.getLogger(name)
    handler = _Records(level)
    saved = (logger.level, logger.propagate)
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    try:
        yield handler.records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(saved[0])
        logger.propagate = saved[1]


def _messages(records, level=None):
    return [r.getMessage() for r in records if level is None or r.levelno == level]


def _run_rows(failures, rows):
    """Run each row from normal verbosity, so no row inherits the mode an earlier one set."""
    for row in rows:

        def from_normal(row=row):
            set_normal_logging()
            row()

        check(failures, row.__name__, from_normal)


def _stream_handlers():
    """The marEx console handlers (``RotatingFileHandler`` is a ``StreamHandler`` subclass, so match the type)."""
    return [h for h in logging.getLogger("marEx").handlers if type(h) is logging.StreamHandler]


def test_verbosity_state_and_configuration():
    """Modes, quiet over verbose, the setup_logging alias, string levels, the configured flag, PlotConfig flags."""
    root = logging.getLogger("marEx")
    failures = []

    def modes():
        for setter, level in ((set_verbose_mode, "verbose"), (set_quiet_mode, "quiet"), (set_normal_logging, "normal")):
            setter()
            assert get_verbosity_level() == level, f"{setter.__name__} left {get_verbosity_level()!r}"
            assert is_verbose_mode() == (level == "verbose") and is_quiet_mode() == (level == "quiet"), level

    def precedence():
        set_verbose_mode()
        configure_logging(verbose=True, quiet=True)
        assert is_quiet_mode() and not is_verbose_mode(), f"verbose and quiet gave {get_verbosity_level()!r}"
        assert root.level == logging.WARNING, logging.getLevelName(root.level)

    def setup_logging_alias():
        set_quiet_mode()
        setup_logging(verbose=True)
        assert is_verbose_mode(), f"setup_logging(verbose=True) left {get_verbosity_level()!r}"

    def string_level():
        set_quiet_mode()
        configure_logging(level="DEBUG")
        assert root.level == logging.DEBUG, logging.getLevelName(root.level)
        configure_logging(level="invalid_level")
        assert root.level == logging.INFO, f"an invalid level gave {logging.getLevelName(root.level)}, not the INFO default"

    def explicit_config_survives_get_logger():
        configure_logging(verbose=True)
        assert root.level == logging.DEBUG
        get_logger("marEx.some.module")
        assert root.level == logging.DEBUG, "a later get_logger reset an explicit configuration"

    def lazy_autoconfig_fires_for_submodule_logger():
        lc._logger_configured = False
        root.handlers.clear()
        get_logger("marEx.pipeline")
        assert lc._logger_configured is True, "get_logger('marEx.pipeline') did not configure the package logger"
        assert root.handlers, "the package logger has no handler after the lazy configuration"

    def get_logger_configures_once():
        lc._logger_configured = False
        first = get_logger("marEx")
        assert lc._logger_configured is True
        sentinel = logging.NullHandler()
        root.addHandler(sentinel)
        assert get_logger("marEx") is first
        assert sentinel in root.handlers, "a second get_logger reconfigured and cleared the handlers"

    def plot_config_verbose():
        set_quiet_mode()
        config = marEx.PlotConfig(verbose=True)
        assert config.verbose is True
        assert is_verbose_mode(), f"PlotConfig(verbose=True) left {get_verbosity_level()!r}"

    def plot_config_quiet():
        set_verbose_mode()
        config = marEx.PlotConfig(quiet=True)
        assert config.quiet is True
        assert is_quiet_mode(), f"PlotConfig(quiet=True) left {get_verbosity_level()!r}"

    _run_rows(
        failures,
        (
            modes,
            precedence,
            setup_logging_alias,
            string_level,
            explicit_config_survives_get_logger,
            lazy_autoconfig_fires_for_submodule_logger,
            get_logger_configures_once,
            plot_config_verbose,
            plot_config_quiet,
        ),
    )
    assert_no_failures(failures)


# Each spelling (comma-joined NAME=value pairs) and the verbosity configure_logging() then reports.
ENV_ROWS = [
    ("MAREX_VERBOSE=true", "verbose"),
    ("MAREX_VERBOSE=1", "verbose"),
    ("MAREX_VERBOSE=yes", "verbose"),
    ("MAREX_VERBOSE=True", "verbose"),
    ("MAREX_VERBOSE=YES", "verbose"),
    ("MAREX_VERBOSE=false", "normal"),
    ("MAREX_VERBOSE=0", "normal"),
    ("MAREX_VERBOSE=no", "normal"),
    ("MAREX_VERBOSE=False", "normal"),
    ("MAREX_VERBOSE=NO", "normal"),
    ("MAREX_VERBOSE=anything_else", "normal"),
    ("MAREX_QUIET=true", "quiet"),
    ("MAREX_QUIET=yes", "quiet"),
    ("MAREX_QUIET=no", "normal"),
    ("MAREX_VERBOSE=true,MAREX_QUIET=true", "quiet"),
]

# The environment is read when marEx.logging_config is imported, so the child re-imports it per spelling.
_ENV_CHILD = """
import importlib, os, sys
import marEx
import marEx.logging_config as lc
print("@@", marEx.__file__)
for spec in sys.argv[1:]:
    for name in ("MAREX_VERBOSE", "MAREX_QUIET"):
        os.environ.pop(name, None)
    for pair in spec.split(","):
        name, value = pair.split("=", 1)
        os.environ[name] = value
    lc = importlib.reload(lc)
    lc.configure_logging(console_output=False)
    print("@@", spec, lc.get_verbosity_level())
"""

_LOG_FILE_CHILD = """
import logging
import marEx
import marEx.logging_config as lc
print("@@", marEx.__file__)
lc.configure_logging(level="INFO", console_output=False)
lc.get_logger("marEx.env_check").info("message routed by MAREX_LOG_FILE")
logging.shutdown()
"""


def _child(code, args, tmp_path, **env):
    """Run ``code`` in a fresh interpreter from ``tmp_path``; return its stdout lines after checking it imported this marEx."""
    base = {k: v for k, v in os.environ.items() if k not in ("MAREX_VERBOSE", "MAREX_QUIET", "MAREX_LOG_FILE")}
    run = subprocess.run(
        [sys.executable, "-c", code, *args],
        cwd=tmp_path,
        env={**base, **env},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert run.returncode == 0, run.stderr[-2000:]
    lines = [line[3:] for line in run.stdout.splitlines() if line.startswith("@@ ")]
    assert lines and lines[0] == marEx.__file__, f"child imported {lines[:1]}, this process {marEx.__file__}"
    return lines[1:]


def test_log_destinations_and_environment(tmp_path):
    """The log file holds the message, console output can be switched off, a custom format reaches the console
    handler, and MAREX_VERBOSE / MAREX_QUIET / MAREX_LOG_FILE are honoured (in child processes)."""
    failures = []

    def log_file():
        path = tmp_path / "test.log"
        configure_logging(log_file=path, verbose=True)
        get_logger("marEx.file_check").info("Test message to the file")
        for handler in logging.getLogger("marEx").handlers:
            handler.flush()
        text = path.read_text()
        assert "Test message to the file" in text, f"the log file holds {text[-300:]!r}"

    def no_console():
        configure_logging()
        assert _stream_handlers(), "precondition: the default configuration has a console handler"
        configure_logging(console_output=False)
        assert not _stream_handlers(), f"console_output=False left {_stream_handlers()}"

    def custom_format():
        custom = "CUSTOM: %(levelname)s - %(message)s"
        configure_logging(format_str=custom)
        (console,) = _stream_handlers()
        fmt = getattr(console.formatter, "_fmt", None)
        assert fmt == custom, f"console handler format {fmt!r}"
        stream = io.StringIO()
        console.setStream(stream)
        get_logger("marEx.format_check").info("Test message")
        assert "CUSTOM: INFO - Test message" in stream.getvalue(), stream.getvalue()

    def env_verbosity():
        lines = _child(_ENV_CHILD, [spec for spec, _ in ENV_ROWS], tmp_path)
        got = dict(line.rsplit(" ", 1) for line in lines)
        wrong = {spec: (got.get(spec), want) for spec, want in ENV_ROWS if got.get(spec) != want}
        assert not wrong, f"(reported, expected) verbosity per spelling: {wrong}"

    def env_log_file():
        path = tmp_path / "env" / "env_test.log"
        _child(_LOG_FILE_CHILD, [], tmp_path, MAREX_LOG_FILE=str(path))
        assert path.exists(), "MAREX_LOG_FILE was not created"
        text = path.read_text()
        assert "message routed by MAREX_LOG_FILE" in text and f"Logging to file: {path}" in text, text[-300:]

    _run_rows(failures, (log_file, no_console, custom_format, env_verbosity, env_log_file))
    assert_no_failures(failures)


def test_log_timing():
    """Start and completion records, the failure record on an exception, memory records at DEBUG, the verbose init."""
    logger = get_logger(PROBE)
    failures = []

    def basic():
        with _recorded() as records:
            with log_timing(logger, "Test operation"):
                pass
        messages = _messages(records, logging.INFO)
        assert "Starting Test operation" in messages, messages
        assert any(m.startswith("Completed Test operation in ") for m in messages), messages

    def with_exception():
        with _recorded() as records:
            with pytest.raises(ValueError, match="Test error"):
                with log_timing(logger, "Failed operation"):
                    raise ValueError("Test error")
        errors = _messages(records, logging.ERROR)
        assert len(errors) == 1 and errors[0].startswith("Failed Failed operation after "), errors

    def with_memory():
        with _recorded() as records:
            with log_timing(logger, "Memory test", log_memory=True):
                pass
        memory = _messages(records, logging.DEBUG)
        for when in ("Before", "After"):
            assert any(m.startswith(f"{when} Memory test - Memory Usage - RSS:") for m in memory), (when, memory)

    def verbose_init():
        set_verbose_mode()
        with _recorded() as records:
            with log_timing(logger, "Verbose test", show_progress=True):
                pass
        assert "Initializing Verbose test" in _messages(records, logging.DEBUG)
        set_normal_logging()
        with _recorded() as records:
            with log_timing(logger, "Verbose test", show_progress=True):
                pass
        assert "Initializing Verbose test" not in _messages(records), "the init record is verbose-only"

    _run_rows(failures, (basic, with_exception, with_memory, verbose_init))
    assert_no_failures(failures)


def test_log_progress():
    """One record per crossed bucket (exactly 10 for 37 steps at 10 %), none when quiet or for a non-positive total,
    and the rate in verbose mode."""
    logger = get_logger(PROBE)
    failures = []

    def halfway():
        with _recorded() as records:
            log_progress(logger, current=50, total=100, operation="Test op")
        assert _messages(records) == ["Test op: 50% complete (50/100)"], _messages(records)

    def quiet():
        set_quiet_mode()
        with _recorded() as records:
            log_progress(logger, current=50, total=100, operation="Test op")
        assert records == [], _messages(records)

    def non_positive_total():
        with _recorded() as records:
            log_progress(logger, current=0, total=0, operation="Zero test")
            log_progress(logger, current=1, total=-1, operation="Negative test")
        assert records == [], _messages(records)

    def non_round_total_fires_once_per_bucket():
        with _recorded() as records:
            for current in range(1, 38):
                log_progress(logger, current=current, total=37, operation="Bucket test", frequency=10)
        fired = [m for m in _messages(records) if m.startswith("Bucket test:")]
        assert len(fired) == 10, f"{len(fired)} progress records for 37 steps at 10%: {fired}"

    def verbose_rate():
        set_verbose_mode()
        with _recorded() as records:
            start = time.perf_counter() - 1.0
            log_progress(logger, current=10, total=100, operation="Verbose test", frequency=10, start_time=start)
        messages = _messages(records, logging.DEBUG)
        assert len(messages) == 1 and messages[0].startswith("Verbose test: 10/100 (10.0%) - Rate: "), messages

    _run_rows(failures, (halfway, quiet, non_positive_total, non_round_total_fires_once_per_bucket, verbose_rate))
    assert_no_failures(failures)


def test_progress_bar():
    """A bar exactly when tqdm is importable; none when disabled or quiet; the logger fallback reports completion
    except in quiet mode; a bar the context manager made is closed on exit."""
    logger = get_logger(PROBE)
    failures = []

    def available():
        pbar = create_progress_bar(total=100, desc="Test")
        try:
            has_tqdm = importlib.util.find_spec("tqdm") is not None
            assert (pbar is not None) == has_tqdm, f"bar {pbar!r} with tqdm importable={has_tqdm}"
        finally:
            if pbar is not None:
                pbar.close()

    def disabled():
        assert create_progress_bar(total=100, desc="Test", disable=True) is None

    def quiet():
        set_quiet_mode()
        assert create_progress_bar(total=100, desc="Test") is None

    def fallback_normal():
        with mock.patch("marEx.logging_config.create_progress_bar", return_value=None), _recorded() as records:
            with progress_bar(total=10, desc="Fallback test", logger=logger) as pbar:
                assert pbar is None
        assert _messages(records, logging.INFO) == ["Completed Fallback test"], _messages(records)

    def fallback_quiet():
        set_quiet_mode()
        with _recorded() as records:
            with progress_bar(total=10, desc="Quiet test", logger=logger) as pbar:
                assert pbar is None
        assert records == [], _messages(records)

    def bar_closed_on_exit():
        bar = mock.MagicMock()
        with mock.patch("marEx.logging_config.create_progress_bar", return_value=bar), _recorded() as records:
            with progress_bar(total=10, desc="Closed test", logger=logger) as pbar:
                assert pbar is bar
                bar.close.assert_not_called()
        bar.close.assert_called_once_with()
        assert records == [], f"a bar was shown, yet the fallback logged: {_messages(records)}"

    _run_rows(failures, (available, disabled, quiet, fallback_normal, fallback_quiet, bar_closed_on_exit))
    assert_no_failures(failures)


def test_log_function_call():
    """Call, completion and failure records; the module logger by default; argument and length truncation."""
    logger = get_logger(PROBE)
    failures = []

    def call_and_complete():
        @log_function_call(logger=logger, level=logging.INFO)
        def add(x, y=10):
            return x + y

        with _recorded() as records:
            assert add(5, y=15) == 20
        messages = _messages(records, logging.INFO)
        assert messages[0] == f"Calling {__name__}.add(5, y=15)", messages
        assert messages[1].startswith(f"Completed {__name__}.add in "), messages

    def with_exception():
        @log_function_call(logger=logger, level=logging.INFO)
        def failing():
            raise ValueError("Test error")

        with _recorded() as records:
            with pytest.raises(ValueError, match="Test error"):
                failing()
        errors = _messages(records, logging.ERROR)
        assert len(errors) == 1 and errors[0].startswith(f"Failed {__name__}.failing after "), errors

    def default_logger():
        @log_function_call(level=logging.INFO)
        def plain():
            return "test"

        with _recorded(__name__) as records:
            assert plain() == "test"
        assert f"Calling {__name__}.plain()" in _messages(records), _messages(records)
        assert {r.name for r in records} == {__name__}

    def many_args():
        @log_function_call(logger=logger, level=logging.INFO)
        def many(a, b, c, d, e, f=1, g=2, h=3, i=4):
            return sum([a, b, c, d, e, f, g, h, i])

        with _recorded() as records:
            assert many(1, 2, 3, 4, 5, f=6, g=7, h=8, i=9) == 45
        expected = f"Calling {__name__}.many(1, 2, 3, ... +2 more, f=6, g=7, h=8, ... +1 more)"
        assert _messages(records)[0] == expected, _messages(records)

    def long_params():
        @log_function_call(logger=logger, level=logging.INFO)
        def long_param(value):
            return len(value)

        with _recorded() as records:
            assert long_param("x" * 300) == 300
        call = _messages(records)[0]
        params = call[len(f"Calling {__name__}.long_param(") : -1]
        assert params == repr("x" * 300)[:200] + "...", call

    _run_rows(failures, (call_and_complete, with_exception, default_logger, many_args, long_params))
    assert_no_failures(failures)


class _ChunksRaise:
    """An object whose ``.chunks`` raises, so ``log_dask_info`` reaches its error branch."""

    chunks = property(lambda self: 1 / 0)


def test_dask_and_memory_info():
    """Memory usage numbers and records; dask object records with and without a message; long chunk tuples
    truncated; an object that raises is logged, not re-raised."""
    logger = get_logger(PROBE)
    sample = xr.DataArray(da.ones((10, 5, 5), chunks=(5, 5, 5)), dims=["time", "lat", "lon"])
    failures = []

    def memory_usage():
        memory = get_memory_usage()
        assert set(memory) == {"rss_mb", "vms_mb", "percent", "available_mb"}, memory
        assert all(isinstance(v, (int, float)) for v in memory.values()), memory
        assert memory["rss_mb"] > 0 and memory["available_mb"] > 0, memory

    def log_memory_with_message():
        with _recorded() as records:
            log_memory_usage(logger, "Test operation")
        messages = _messages(records, logging.DEBUG)
        assert len(messages) == 1 and messages[0].startswith("Test operation - Memory Usage - RSS: "), messages

    def log_memory_without_message():
        with _recorded() as records:
            log_memory_usage(logger)
        messages = _messages(records, logging.DEBUG)
        assert len(messages) == 1 and messages[0].startswith("Memory Usage - RSS:"), messages

    def dask_info_with_message():
        with _recorded() as records:
            log_dask_info(logger, sample, "Test dask array")
        messages = _messages(records, logging.DEBUG)
        assert messages[0].startswith("Test dask array - Dask object - Shape: (10, 5, 5), Chunks: "), messages
        assert re.fullmatch(r"Dask graph size: \d+ tasks", messages[1]), messages

    def dask_info_without_message():
        with _recorded() as records:
            log_dask_info(logger, sample)
        assert _messages(records)[0].startswith("Dask object - Shape: (10, 5, 5)"), _messages(records)

    def long_chunks_truncated():
        many = xr.DataArray(da.ones((100, 100, 100), chunks=10), dims=["x", "y", "z"])
        with _recorded() as records:
            log_dask_info(logger, many, "Long chunks test")
        field = re.search(r"Chunks: (.*), Size: ", _messages(records)[0]).group(1)
        assert len(field) <= 103 and field.endswith("..."), f"Chunks field of {len(field)} chars: {field[-40:]!r}"

    def error_is_logged_not_raised():
        with _recorded() as records:
            log_dask_info(logger, _ChunksRaise(), "Bad object")
        errors = [m for m in _messages(records, logging.DEBUG) if m.startswith("Could not log Dask info")]
        assert len(errors) == 1, _messages(records)

    _run_rows(
        failures,
        (
            memory_usage,
            log_memory_with_message,
            log_memory_without_message,
            dask_info_with_message,
            dask_info_without_message,
            long_chunks_truncated,
            error_is_logged_not_raised,
        ),
    )
    assert_no_failures(failures)


def test_external_loggers_keep_memory_warnings_visible():
    """Library noise is set to ERROR; the worker and nanny memory warnings stay visible, also through
    configure_logging."""
    failures = []

    def reset():
        for name in EXTERNAL_LOGGERS:
            logging.getLogger(name).setLevel(logging.NOTSET)
        logging.getLogger("distributed").setLevel(logging.NOTSET)

    def levels(configure):
        saved_distributed = logging.getLogger("distributed").level
        try:
            reset()
            logging.getLogger("distributed").setLevel(logging.ERROR)  # a memory logger that inherits must not pass
            configure()
            for name in NOISY_LOGGERS:
                assert logging.getLogger(name).level == logging.ERROR, (
                    f"{name} at {logging.getLevelName(logging.getLogger(name).level)}"
                )
            for name in MEMORY_LOGGERS:
                effective = logging.getLogger(name).getEffectiveLevel()
                assert effective <= logging.WARNING, f"{name} effective level {logging.getLevelName(effective)}"
        finally:
            logging.getLogger("distributed").setLevel(saved_distributed)

    check(failures, "configure_external_loggers", lambda: levels(_configure_external_loggers))
    check(failures, "configure_logging", lambda: levels(configure_logging))
    assert_no_failures(failures)


def _sample_sst():
    times = pd.date_range("2020-01-01", periods=10, freq="D")
    data = np.random.default_rng(42).standard_normal((10, 5, 5)) + 20
    return xr.DataArray(
        da.from_array(data, chunks=(5, 5, 5)),
        coords={"time": times, "lat": np.linspace(-10, 10, 5), "lon": np.linspace(-10, 10, 5)},
        dims=["time", "lat", "lon"],
        name="sst",
    )


def _flag_takes_effect(marex_records, flag, run):
    """Force the opposite mode, run ``run`` (which passes ``flag``), and check the mode and the records it left."""
    configure_logging(**({"quiet": True} if flag == "verbose" else {"verbose": True}))
    root = logging.getLogger("marEx")
    forced = root.level
    records = marex_records(logging.DEBUG)
    root.setLevel(forced)  # only the flag under test may lower the level
    run()
    if flag == "verbose":
        assert is_verbose_mode(), f"verbose=True left the mode {get_verbosity_level()!r}"
        assert any(r.levelno == logging.DEBUG for r in records), "verbose=True emitted no DEBUG record from marEx"
    else:
        assert is_quiet_mode(), f"quiet=True left the mode {get_verbosity_level()!r}"
        info = _messages(records, logging.INFO)
        assert not info, f"quiet=True still emitted {len(info)} INFO record(s): {info[:3]}"


def test_stage_verbosity_flags_control_marEx_records(client, marex_records):
    """``verbose=`` / ``quiet=`` on each stage set the mode and decide which marEx records are emitted: the stages
    on threads, and preprocessing followed by a tracker run on the cluster."""
    saved = _logging_state()
    failures = []
    tracker_kw = dict(R_fill=2, area_filter_quartile=0.5, regional_mode=False, coordinate_units="degrees")
    try:
        with dask.config.set(THREADED):
            sst = _sample_sst()
            anomalies = marEx.anomaly.compute_normalised_anomaly(sst).dat_anomaly
            binary = (sst > sst.mean()).astype(bool)
            mask = xr.ones_like(sst.isel(time=0), dtype=bool)
            threaded_rows = {
                "preprocess_data": lambda f: marEx.preprocess_data(
                    sst, method_anomaly="detrend_harmonic", threshold_percentile=90, **{f: True}
                ),
                "compute_normalised_anomaly": lambda f: marEx.anomaly.compute_normalised_anomaly(
                    sst, method_anomaly="detrend_harmonic", **{f: True}
                ),
                "identify_extremes": lambda f: marEx.extremes.identify_extremes(
                    anomalies, method_extreme="global_percentile", threshold_percentile=90, **{f: True}
                ),
                "tracker_init": lambda f: marEx.tracker(binary, mask, **tracker_kw, **{f: True}),
            }
            for name, stage in threaded_rows.items():
                for flag in ("verbose", "quiet"):
                    check(
                        failures,
                        f"{name}[{flag}]",
                        lambda stage=stage, flag=flag: _flag_takes_effect(marex_records, flag, lambda: stage(flag)),
                    )

        def end_to_end(flag):
            assert on_client(client)
            held = {}

            def preprocess():
                held["ds"] = marEx.preprocess_data(
                    sst, method_anomaly="detrend_fixed_baseline", threshold_percentile=90, **{flag: True}
                )

            def track():
                tracker = marEx.tracker(held["ds"].extreme_events, held["ds"].mask, **tracker_kw, **{flag: True})
                assert getattr(tracker, flag) is True
                tracker.run()

            try:
                _flag_takes_effect(marex_records, flag, preprocess)
                _flag_takes_effect(marex_records, flag, track)
            finally:
                held.clear()

        for flag in ("verbose", "quiet"):
            check(failures, f"end_to_end_on_client[{flag}]", lambda flag=flag: end_to_end(flag))
    finally:
        _restore_logging(saved)
    assert _logging_state() == saved, "the logging state was not restored"
    assert_no_failures(failures)

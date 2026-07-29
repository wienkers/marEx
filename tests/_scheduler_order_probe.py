"""Run by ``test_cluster.py`` in a child pytest, in this order; not collected on its own."""

import dask
import dask.array
import dask.threaded


def test_starts_the_session_client(client):
    assert getattr(dask.base.get_scheduler(), "__self__", None) is client


def test_then_computes_on_threads():
    from distributed import default_client

    session_client = default_client()  # raises if the session client is not up
    assert dask.base.get_scheduler() is dask.threaded.get
    assert dask.config.get("array.rechunk.method") == "tasks"
    x = dask.array.ones((40, 40), chunks=(4, 40)).rechunk((40, 4))
    assert float(x.sum().compute()) == 1600.0
    assert not session_client.futures, "a threaded compute left futures on the session client"

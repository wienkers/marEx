"""Release check: does a green ``ci.yml`` push run on ``main`` exist for this commit?

The qualifying run is a ``push`` run of ``ci.yml`` on ``main`` whose ``head_sha`` is the commit
being released. The latest such run (highest ``run_number``, ties broken by ``id``) is read at its
CURRENT ``run_attempt``, and the check passes only if that attempt's ``CI OK`` job concluded
``success``. Nonterminal runs are polled until ``deadline_s``; when no qualifying run exists the
check waits ``grace_s`` (the push may not have registered yet) and then fails.

Standard library only, so the release job needs no install step. The HTTP getter, clock and
sleep are injectable so the unit tests run offline and instantly.

CLI (exits 0 when the check passes, 1 otherwise)::

    GITHUB_TOKEN=... python ci_check.py --repo owner/name --sha <40-hex> [--deadline 5400] [--grace 120]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

API = "https://api.github.com"
WORKFLOW = "ci.yml"
BRANCH = "main"
EVENT = "push"
REQUIRED_JOB = "CI OK"
PER_PAGE = 100
NONTERMINAL = frozenset({"queued", "waiting", "requested", "pending", "in_progress"})
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
MAX_BACKOFF_S = 120.0


def http_get_json(url: str, token: str) -> dict:
    """GET ``url`` from the GitHub REST API and return the decoded JSON body."""
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "marEx-release-check",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def _paginate(get, url: str, token: str, key: str) -> list:
    """Collect every item under ``key`` across pages of ``per_page=100``."""
    items: list = []
    page = 1
    while True:
        sep = "&" if "?" in url else "?"
        body = get(f"{url}{sep}per_page={PER_PAGE}&page={page}", token)
        batch = body.get(key, [])
        items.extend(batch)
        total = body.get("total_count", len(items))
        if not batch or len(items) >= total:
            return items
        page += 1


def _runs_url(repo: str, sha: str) -> str:
    query = urllib.parse.urlencode({"event": EVENT, "branch": BRANCH, "head_sha": sha})
    return f"{API}/repos/{repo}/actions/workflows/{WORKFLOW}/runs?{query}"


def _jobs_url(repo: str, run_id: int, attempt: int) -> str:
    return f"{API}/repos/{repo}/actions/runs/{run_id}/attempts/{attempt}/jobs"


def _qualifying(runs: list, sha: str) -> list:
    """Keep only push runs on main at ``sha`` (the query filters too; this guards the response)."""
    return [r for r in runs if r.get("event") == EVENT and r.get("head_branch") == BRANCH and r.get("head_sha") == sha]


class CheckError(Exception):
    """A request failed in a way the check does not retry, or kept failing past the deadline."""


def _transient(error: Exception) -> tuple[bool, float | None]:
    """Return ``(retry, retry_after_s)`` for a failed request."""
    if isinstance(error, urllib.error.HTTPError):
        headers = error.headers or {}
        retry_after = headers.get("retry-after")
        try:
            after = float(retry_after) if retry_after is not None else None
        except ValueError:
            after = None
        if error.code >= 500 or error.code == 429:
            return True, after
        if error.code == 403 and (headers.get("x-ratelimit-remaining") == "0" or retry_after is not None):
            return True, after
        return False, None
    if isinstance(error, (urllib.error.URLError, TimeoutError, ConnectionError)):
        return True, None
    return False, None


def ci_check(
    repo: str,
    sha: str,
    token: str,
    deadline_s: float,
    grace_s: float,
    *,
    get=http_get_json,
    clock=time.monotonic,
    sleep=time.sleep,
    poll_s: float = 30.0,
) -> tuple[bool, str, int, str]:
    """Return ``(ok, run_url, attempt, message)`` for the release check on commit ``sha``."""
    if not SHA_RE.match(sha or ""):
        return False, "", 0, f"sha {sha!r} is not a full 40-character lowercase commit SHA"
    start = clock()

    def wait(limit: float, seconds: float = poll_s) -> bool:
        remaining = limit - (clock() - start)
        if remaining <= 0:
            return False
        sleep(min(seconds, remaining))
        return True

    def fetch(url: str, tok: str) -> dict:
        """``get`` with retries on rate limits, 5xx and network errors, bounded by the deadline."""
        failures = 0
        while True:
            try:
                return get(url, tok)
            except Exception as error:  # noqa: BLE001 - classified below; anything unknown is re-raised
                retry, after = _transient(error)
                code = getattr(error, "code", None)
                what = f"HTTP {code} from {url}" if code is not None else f"{type(error).__name__} from {url}: {error}"
                if not retry:
                    if code is not None:
                        raise CheckError(what) from error
                    raise
                failures += 1
                backoff = after if after is not None else min(5.0 * 2 ** (failures - 1), MAX_BACKOFF_S)
                if not wait(deadline_s, backoff):
                    raise CheckError(
                        f"{what} (still failing after {failures} attempts at the {deadline_s:.0f} s deadline)"
                    ) from error

    run_url, attempt = "", 0
    try:
        while True:
            runs = _qualifying(_paginate(fetch, _runs_url(repo, sha), token, "workflow_runs"), sha)
            if not runs:
                if wait(grace_s):
                    continue
                return (
                    False,
                    "",
                    0,
                    f"no {EVENT} run of {WORKFLOW} on {BRANCH} for {sha} after {grace_s:.0f} s: it was never pushed to main, "
                    f"or was not the tip of its push, or that run was superseded; tag a commit whose own main run passed CI",
                )
            run = max(runs, key=lambda r: (r.get("run_number", 0), r.get("id", 0)))
            run_url = run.get("html_url", "")
            attempt = int(run.get("run_attempt", 1))
            status = run.get("status")
            if status in NONTERMINAL:
                if wait(deadline_s):
                    continue
                return False, run_url, attempt, f"run still {status} after the {deadline_s:.0f} s deadline"
            if status != "completed":
                return False, run_url, attempt, f"run has unexpected status {status!r}"
            jobs = _paginate(fetch, _jobs_url(repo, run["id"], attempt), token, "jobs")
            required = [j for j in jobs if j.get("name") == REQUIRED_JOB]
            if not required:
                return False, run_url, attempt, f"attempt {attempt} has no {REQUIRED_JOB!r} job"
            conclusion = required[-1].get("conclusion")
            if conclusion == "success":
                return True, run_url, attempt, f"{REQUIRED_JOB!r} succeeded on attempt {attempt}"
            return False, run_url, attempt, f"{REQUIRED_JOB!r} concluded {conclusion!r} on attempt {attempt}"
    except CheckError as error:
        return False, run_url, attempt, str(error)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: print the run URL, attempt and message; return the exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", required=True)
    parser.add_argument("--sha", required=True)
    parser.add_argument(
        "--deadline", type=float, default=5400.0, help="seconds to poll a nonterminal run or retry a failing request"
    )
    parser.add_argument("--grace", type=float, default=120.0, help="seconds to wait for a run to appear")
    parser.add_argument("--poll", type=float, default=30.0)
    args = parser.parse_args(argv)
    token = os.environ.get("GITHUB_TOKEN", "")
    ok, run_url, attempt, message = ci_check(args.repo, args.sha, token, args.deadline, args.grace, poll_s=args.poll)
    print(f"run: {run_url or '(none)'}")
    print(f"attempt: {attempt}")
    print(f"{'PASS' if ok else 'FAIL'}: {message}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

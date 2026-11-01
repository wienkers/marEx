# ruff: noqa: D102, D103, D107  (test module: the repo exempts tests/* from pydocstyle the same way)
"""Offline tests of the release check against recorded (and derived) GitHub API payloads.

Each fixture's ``_source`` key says whether it is a real recorded payload or derived from one.
Run with ``pytest .github/scripts``; the marEx suite (testpaths = ["tests"]) does not collect it.
"""

from __future__ import annotations

import copy
import email.message
import json
import re
import urllib.error
from pathlib import Path

import ci_check
import pytest

FIXTURES = Path(__file__).parent / "fixtures"
REPO = "wienkers/marEx"
SHA = "73ed4575a0c290eefe6202b02771fa4e25c32e86"
DEADLINE = 3600.0
GRACE = 120.0


def load(name: str) -> dict:
    body = json.loads((FIXTURES / f"{name}.json").read_text())
    body.pop("_source")
    return body


class FakeGitHub:
    """Routes requests by URL, like the API: runs list, or jobs of one run attempt, one page at a time.

    ``runs`` is a list of runs payloads served on successive runs requests (the last repeats).
    ``jobs`` maps ``(run_id, attempt)`` to a jobs payload; an unknown key is a test error.
    ``errors`` are raised, one per request, before anything is served; with ``sticky`` the first
    error is raised on every request.
    The clock advances only through ``sleep``; more than ``max_sleeps`` sleeps raises, so a check
    that never gives up fails the test instead of hanging it.
    """

    def __init__(self, runs, jobs, max_sleeps=1000, errors=()):
        self.runs = list(runs)
        self.jobs = jobs
        self.errors = list(errors)
        self.urls: list[str] = []
        self.now = 0.0
        self.sleeps = 0
        self.max_sleeps = max_sleeps
        self.sticky = False

    def get(self, url, token):
        self.urls.append(url)
        if self.errors:
            raise self.errors[0] if self.sticky else self.errors.pop(0)
        page = int(re.search(r"[?&]page=(\d+)", url).group(1))
        per_page = int(re.search(r"[?&]per_page=(\d+)", url).group(1))
        if "/actions/workflows/ci.yml/runs?" in url:
            body = self.runs.pop(0) if len(self.runs) > 1 else self.runs[0]
            key = "workflow_runs"
        else:
            m = re.search(r"/actions/runs/(\d+)/attempts/(\d+)/jobs", url)
            body = self.jobs[(int(m.group(1)), int(m.group(2)))]
            key = "jobs"
        items = body[key][(page - 1) * per_page : page * per_page]
        return {"total_count": body["total_count"], key: copy.deepcopy(items)}

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps += 1
        if self.sleeps > self.max_sleeps:
            raise RuntimeError(f"check slept more than {self.max_sleeps} times: it does not stop polling")
        assert seconds > 0, seconds
        self.now += seconds


def run_check(fake, sha=SHA):
    return ci_check.ci_check(REPO, sha, "token", DEADLINE, GRACE, get=fake.get, clock=fake.clock, sleep=fake.sleep)


RUN_ID = 41052242346
GREEN = load("jobs_real_green_attempt1")


def test_green_run_passes():
    fake = FakeGitHub([load("runs_green_on_main")], {(RUN_ID, 1): GREEN})
    ok, url, attempt, message = run_check(fake)
    assert ok, message
    assert url == f"https://github.com/{REPO}/actions/runs/{RUN_ID}"
    assert attempt == 1
    assert fake.sleeps == 0


def test_rerun_attempt2_failed_after_green_attempt1_fails():
    fake = FakeGitHub(
        [load("runs_rerun_attempt2")],
        {(RUN_ID, 1): GREEN, (RUN_ID, 2): load("jobs_rerun_attempt2_failed")},
    )
    ok, url, attempt, message = run_check(fake)
    assert not ok, message
    assert attempt == 2
    assert "'failure'" in message
    assert any("/attempts/2/jobs" in u for u in fake.urls), fake.urls


def test_in_progress_past_deadline_fails():
    fake = FakeGitHub([load("runs_in_progress")], {(RUN_ID, 1): GREEN})
    ok, url, attempt, message = run_check(fake)
    assert not ok, message
    assert "deadline" in message and "in_progress" in message
    assert DEADLINE <= fake.now <= DEADLINE + 1, fake.now


@pytest.mark.parametrize("status", sorted(ci_check.NONTERMINAL))
def test_nonterminal_run_is_polled_until_it_completes(status):
    pending = load("runs_in_progress")
    pending["workflow_runs"][0]["status"] = status
    fake = FakeGitHub([pending, pending, load("runs_green_on_main")], {(RUN_ID, 1): GREEN})
    ok, url, attempt, message = run_check(fake)
    assert ok, message
    assert fake.sleeps == 2


def test_no_run_for_sha_fails_after_grace_with_never_pushed_message():
    fake = FakeGitHub([load("runs_real_none_on_main")], {})
    ok, url, attempt, message = run_check(fake)
    assert not ok
    assert "never pushed to main" in message
    assert "or that run was superseded" in message
    assert url == "" and attempt == 0
    assert GRACE <= fake.now <= GRACE + 1, fake.now


def test_run_that_appears_within_grace_is_used():
    fake = FakeGitHub([load("runs_real_none_on_main"), load("runs_green_on_main")], {(RUN_ID, 1): GREEN})
    ok, url, attempt, message = run_check(fake)
    assert ok, message
    assert fake.sleeps == 1


def test_ci_ok_skipped_fails():
    fake = FakeGitHub([load("runs_green_on_main")], {(RUN_ID, 1): load("jobs_ci_ok_skipped")})
    ok, url, attempt, message = run_check(fake)
    assert not ok, message
    assert "'skipped'" in message


def test_missing_ci_ok_job_fails():
    fake = FakeGitHub([load("runs_green_on_main")], {(RUN_ID, 1): load("jobs_real_main_old_ci_no_ci_ok")})
    ok, url, attempt, message = run_check(fake)
    assert not ok, message
    assert "no 'CI OK' job" in message


def test_run_on_another_branch_is_not_a_main_run():
    # The real recorded run for this sha: green, but on ci/overhaul.
    fake = FakeGitHub([load("runs_real_ci_overhaul_branch")], {(RUN_ID, 1): GREEN})
    ok, url, attempt, message = run_check(fake)
    assert not ok, message
    assert "never pushed to main" in message


def test_newer_pull_request_and_branch_runs_do_not_mask_a_failed_main_run():
    failed = load("jobs_rerun_attempt2_failed")
    fake = FakeGitHub(
        [load("runs_decoys_newer_than_main")],
        {(RUN_ID, 1): failed, (RUN_ID + 1, 1): GREEN, (RUN_ID + 2, 1): GREEN},
    )
    ok, url, attempt, message = run_check(fake)
    assert not ok, message
    assert url.endswith(str(RUN_ID)), url


def test_runs_query_filters_push_main_and_sha():
    fake = FakeGitHub([load("runs_green_on_main")], {(RUN_ID, 1): GREEN})
    run_check(fake)
    runs_url = fake.urls[0]
    assert runs_url.startswith(f"https://api.github.com/repos/{REPO}/actions/workflows/ci.yml/runs?"), runs_url
    for param in ("event=push", "branch=main", f"head_sha={SHA}"):
        assert re.search(rf"[?&]{param}(&|$)", runs_url), (param, runs_url)


def test_latest_run_number_wins_and_ties_break_on_id():
    older = load("runs_green_on_main")["workflow_runs"][0]
    newer = dict(older, id=RUN_ID + 5, run_number=older["run_number"] + 1, html_url="newer")
    tie = dict(older, id=RUN_ID + 9, run_number=newer["run_number"], html_url="tie")
    fake = FakeGitHub(
        [{"total_count": 3, "workflow_runs": [older, tie, newer]}],
        {(RUN_ID, 1): GREEN, (RUN_ID + 5, 1): GREEN, (RUN_ID + 9, 1): GREEN},
    )
    ok, url, attempt, message = run_check(fake)
    assert ok and url == "tie", (url, message)


def test_ci_ok_on_the_second_jobs_page_is_found():
    jobs = copy.deepcopy(GREEN)
    ci_ok = [j for j in jobs["jobs"] if j["name"] == "CI OK"]
    others = [j for j in jobs["jobs"] if j["name"] != "CI OK"]
    padding = [dict(others[0], id=others[0]["id"] + 10_000 + i, name=f"Pad {i}") for i in range(100 - len(others))]
    jobs["jobs"] = others + padding + ci_ok
    jobs["total_count"] = len(jobs["jobs"])
    assert jobs["total_count"] == 101
    fake = FakeGitHub([load("runs_green_on_main")], {(RUN_ID, 1): jobs})
    ok, url, attempt, message = run_check(fake)
    assert ok, message
    assert any("page=2" in u and "/jobs" in u for u in fake.urls), fake.urls


@pytest.mark.parametrize("sha", ["73ed457", SHA.upper(), "", SHA + "0"])
def test_short_or_malformed_sha_is_refused(sha):
    fake = FakeGitHub([load("runs_green_on_main")], {(RUN_ID, 1): GREEN})
    ok, url, attempt, message = run_check(fake, sha=sha)
    assert not ok
    assert "40-character" in message
    assert fake.urls == []


def test_cli_exit_codes(monkeypatch, capsys):
    fake = FakeGitHub([load("runs_green_on_main")], {(RUN_ID, 1): load("jobs_ci_ok_skipped")})
    real = ci_check.ci_check

    def patched(*args, **kwargs):
        return real(*args, **{**kwargs, "get": fake.get, "clock": fake.clock, "sleep": fake.sleep})

    monkeypatch.setattr(ci_check, "ci_check", patched)
    assert ci_check.main(["--repo", REPO, "--sha", SHA]) == 1
    out = capsys.readouterr().out
    assert f"run: https://github.com/{REPO}/actions/runs/{RUN_ID}" in out
    assert "attempt: 1" in out and "FAIL:" in out
    fake.jobs[(RUN_ID, 1)] = GREEN
    assert ci_check.main(["--repo", REPO, "--sha", SHA]) == 0
    assert "PASS:" in capsys.readouterr().out


def http_error(code, **headers):
    message = email.message.Message()
    for name, value in headers.items():
        message[name.replace("_", "-")] = value
    return urllib.error.HTTPError("https://api.github.com/x", code, "error", message, None)


def test_dispatch_run_newer_than_a_failed_push_run_is_ignored():
    failed = load("jobs_rerun_attempt2_failed")
    fake = FakeGitHub([load("runs_dispatch_newer_than_failed_push")], {(RUN_ID, 1): failed, (RUN_ID + 3, 1): GREEN})
    ok, url, attempt, message = run_check(fake)
    assert not ok, message
    assert url.endswith(str(RUN_ID)), url


def test_green_run_at_another_sha_does_not_mask_a_failed_run_at_this_sha():
    failed = load("jobs_rerun_attempt2_failed")
    fake = FakeGitHub([load("runs_other_sha_newer_than_failed_push")], {(RUN_ID, 1): failed, (RUN_ID + 4, 1): GREEN})
    ok, url, attempt, message = run_check(fake)
    assert not ok, message
    assert url.endswith(str(RUN_ID)), url


def test_unknown_status_fails_closed():
    assert "action_required" not in ci_check.NONTERMINAL
    fake = FakeGitHub([load("runs_action_required")], {(RUN_ID, 1): GREEN})
    ok, url, attempt, message = run_check(fake)
    assert not ok, message
    assert "unexpected status 'action_required'" in message


@pytest.mark.parametrize(
    "error",
    [
        http_error(503),
        http_error(502),
        http_error(429),
        http_error(403, x_ratelimit_remaining="0"),
        http_error(403, retry_after="7"),
        urllib.error.URLError("connection reset"),
        TimeoutError("timed out"),
    ],
    ids=["503", "502", "429", "403-ratelimit", "403-retry-after", "urlerror", "timeout"],
)
def test_transient_error_then_green_passes(error):
    fake = FakeGitHub([load("runs_green_on_main")], {(RUN_ID, 1): GREEN}, errors=[error, error])
    ok, url, attempt, message = run_check(fake)
    assert ok, message
    assert fake.sleeps == 2


def test_retry_after_header_sets_the_wait():
    fake = FakeGitHub([load("runs_green_on_main")], {(RUN_ID, 1): GREEN}, errors=[http_error(429, retry_after="7")])
    ok, url, attempt, message = run_check(fake)
    assert ok, message
    assert fake.now == 7.0


@pytest.mark.parametrize("code", [401, 403, 404, 422])
def test_permanent_http_error_fails_with_its_code(code):
    fake = FakeGitHub([load("runs_green_on_main")], {(RUN_ID, 1): GREEN}, errors=[http_error(code)])
    fake.sticky = True
    ok, url, attempt, message = run_check(fake)
    assert not ok
    assert message.startswith(f"HTTP {code} from https://api.github.com/repos/{REPO}/actions/workflows/ci.yml/runs?"), message
    assert fake.sleeps == 0


def test_permanent_error_on_the_jobs_request_keeps_the_run_url():
    class JobsForbidden(FakeGitHub):
        def get(self, url, token):
            if "/jobs" in url:
                self.urls.append(url)
                raise http_error(403)
            return super().get(url, token)

    fake = JobsForbidden([load("runs_green_on_main")], {})
    ok, url, attempt, message = run_check(fake)
    assert not ok
    assert url.endswith(str(RUN_ID)) and attempt == 1
    assert message.startswith("HTTP 403 from ") and "/attempts/1/jobs" in message, message


def test_transient_errors_past_the_deadline_fail_without_raising():
    fake = FakeGitHub([load("runs_green_on_main")], {(RUN_ID, 1): GREEN}, errors=[http_error(503)])
    fake.sticky = True
    ok, url, attempt, message = run_check(fake)
    assert not ok
    assert message.startswith("HTTP 503 from ") and "deadline" in message, message
    assert DEADLINE <= fake.now <= DEADLINE + 1, fake.now

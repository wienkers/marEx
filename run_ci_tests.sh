#!/usr/bin/env bash
# Run each CI test step locally and report timing.
# Mirrors the steps in .github/workflows/ci.yml (test job).
#
# Usage:
#   ./run_ci_tests.sh              # 2 pytest-xdist workers (matches CI)
#   PYTEST_WORKERS=4 ./run_ci_tests.sh   # override worker count
#   ./run_ci_tests.sh -k slow     # pass extra pytest flags

set -euo pipefail

WORKERS=${PYTEST_WORKERS:-2}
BASE_FLAGS="-n ${WORKERS} -v --tb=short --no-header"
EXTRA_FLAGS="${*}"          # forward any extra args (e.g. -k, --timeout)

# ── colour helpers ────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
RED='\033[0;31m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

DIVIDER="────────────────────────────────────────────────────────────────"

# ── result accumulators ───────────────────────────────────────────────────────
declare -a STEP_NAMES=()
declare -a STEP_DURATIONS=()
declare -a STEP_STATUSES=()

# ── run_step <label> <pytest args…> ──────────────────────────────────────────
run_step() {
    local label="$1"; shift
    local cmd="pytest $* ${BASE_FLAGS} ${EXTRA_FLAGS}"

    echo ""
    echo -e "${CYAN}${DIVIDER}${RESET}"
    echo -e "${BOLD}▶  ${label}${RESET}"
    echo -e "   ${cmd}"
    echo -e "${CYAN}${DIVIDER}${RESET}"

    local t0=$SECONDS
    set +e
    eval "${cmd}"
    local exit_code=$?
    set -e
    local elapsed=$(( SECONDS - t0 ))

    STEP_NAMES+=("${label}")
    STEP_DURATIONS+=("${elapsed}")
    if [[ ${exit_code} -eq 0 ]]; then
        STEP_STATUSES+=("PASS")
        echo -e "${GREEN}✓  PASSED${RESET}  ($(fmt_time ${elapsed}))"
    else
        STEP_STATUSES+=("FAIL")
        echo -e "${RED}✗  FAILED${RESET}  ($(fmt_time ${elapsed}))"
    fi
}

fmt_time() {
    local s=$1
    printf "%dm %02ds" $(( s / 60 )) $(( s % 60 ))
}

# ── steps matching ci.yml ─────────────────────────────────────────────────────

run_step "Unit tests (helpers)" \
    tests/test_detect_helpers.py tests/test_track_helpers.py

run_step "Error handling tests" \
    tests/test_error_handling.py

run_step "Exception tests" \
    tests/test_exceptions.py

run_step "Logging system tests" \
    tests/test_logging_system.py

run_step "Preprocessing tests — gridded" \
    tests/test_gridded_preprocessing.py

run_step "Preprocessing tests — unstructured" \
    tests/test_unstructured_preprocessing.py

run_step "Tracking tests — gridded" \
    tests/test_gridded_tracking.py

run_step "Tracking tests — unstructured" \
    tests/test_unstructured_tracking.py

run_step "Plotting tests" \
    tests/test_plotx.py

run_step "Integration tests  (-m 'not slow')" \
    "tests/test_integration.py -m 'not slow'"

# ── extra test files present locally but not in ci.yml ───────────────────────

run_step "Detect detrending tests  [not in CI]" \
    tests/test_detect_detrending.py

run_step "Track edge-case tests  [not in CI]" \
    tests/test_track_edge_cases.py

# ── summary table ─────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${DIVIDER}${RESET}"
echo -e "${BOLD}  SUMMARY${RESET}"
echo -e "${BOLD}${DIVIDER}${RESET}"

total=0
any_failed=0
for i in "${!STEP_NAMES[@]}"; do
    d=${STEP_DURATIONS[$i]}
    total=$(( total + d ))
    if [[ ${STEP_STATUSES[$i]} == "PASS" ]]; then
        icon="${GREEN}✓${RESET}"
    else
        icon="${RED}✗${RESET}"
        any_failed=1
    fi
    printf "  %b  %-46s %s\n" "${icon}" "${STEP_NAMES[$i]}" "$(fmt_time ${d})"
done

echo -e "${BOLD}${DIVIDER}${RESET}"
printf "  ${BOLD}%-48s %s${RESET}\n" "TOTAL" "$(fmt_time ${total})"
echo ""

exit ${any_failed}

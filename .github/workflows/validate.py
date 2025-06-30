#!/usr/bin/env python3
"""
Validate GitHub Actions workflow files for marEx package.

This script performs basic validation of the workflow YAML files to catch
common issues before committing.
"""

import sys
from pathlib import Path

import yaml


def validate_yaml_file(filepath):
    """Validate a YAML file for syntax errors."""
    try:
        with open(filepath, "r") as f:
            yaml.safe_load(f)
        return True, None
    except yaml.YAMLError as e:
        return False, str(e)


def validate_workflow_structure(workflow_data, filename):
    """Validate the basic structure of a GitHub Actions workflow."""
    issues = []

    # Check required top-level keys
    # Note: YAML parser converts 'on:' to True (boolean), so we check for both
    required_keys = ["name", "jobs"]
    for key in required_keys:
        if key not in workflow_data:
            issues.append(f"Missing required key: {key}")

    # Check for 'on' trigger (can be parsed as True due to YAML quirk)
    if "on" not in workflow_data and True not in workflow_data:
        issues.append("Missing required key: on (workflow triggers)")

    # Check if there are any jobs
    if "jobs" in workflow_data and not workflow_data["jobs"]:
        issues.append("No jobs defined")

    # Check for common issues in job definitions
    if "jobs" in workflow_data:
        for job_name, job_data in workflow_data["jobs"].items():
            if "runs-on" not in job_data:
                issues.append(f"Job '{job_name}' missing 'runs-on'")

            if "steps" not in job_data:
                issues.append(f"Job '{job_name}' missing 'steps'")
            elif not job_data["steps"]:
                issues.append(f"Job '{job_name}' has no steps")

    return issues


def validate_ci_workflow(workflow_data):
    """Validate specific requirements for CI workflow."""
    issues = []

    # Check for required jobs
    required_jobs = ["code-quality", "test", "coverage"]
    if "jobs" in workflow_data:
        for job in required_jobs:
            if job not in workflow_data["jobs"]:
                issues.append(f"CI workflow missing required job: {job}")

    # Check for Python version matrix
    if "jobs" in workflow_data and "test" in workflow_data["jobs"]:
        test_job = workflow_data["jobs"]["test"]
        if "strategy" not in test_job or "matrix" not in test_job["strategy"]:
            issues.append("Test job should use matrix strategy for Python versions")

    return issues


def validate_release_workflow(workflow_data):
    """Validate specific requirements for release workflow."""
    issues = []

    # Check for tag trigger (handle YAML parser quirk where 'on' becomes True)
    trigger_data = workflow_data.get("on") or workflow_data.get(True)
    if trigger_data:
        if "push" not in trigger_data:
            issues.append("Release workflow should trigger on push")
        elif "tags" not in trigger_data["push"]:
            issues.append("Release workflow should trigger on tags")

    # Check for required jobs
    required_jobs = ["validate-tag", "build", "publish-pypi"]
    if "jobs" in workflow_data:
        for job in required_jobs:
            if job not in workflow_data["jobs"]:
                issues.append(f"Release workflow missing required job: {job}")

    return issues


def main():
    """Validate all workflow files in the directory."""
    workflows_dir = Path(__file__).parent
    workflow_files = list(workflows_dir.glob("*.yml"))

    if not workflow_files:
        print("No workflow files found!")
        return 1

    all_valid = True

    for filepath in workflow_files:
        print(f"\nValidating {filepath.name}...")

        # Skip backup files
        if filepath.name.endswith(".bak"):
            print(f"  Skipping backup file: {filepath.name}")
            continue

        # Validate YAML syntax
        is_valid, error = validate_yaml_file(filepath)
        if not is_valid:
            print(f"  ❌ YAML syntax error: {error}")
            all_valid = False
            continue

        # Load and validate workflow structure
        with open(filepath, "r") as f:
            workflow_data = yaml.safe_load(f)

        # Basic structure validation
        issues = validate_workflow_structure(workflow_data, filepath.name)

        # Specific workflow validation
        if filepath.name == "ci.yml":
            issues.extend(validate_ci_workflow(workflow_data))
        elif filepath.name == "release.yml":
            issues.extend(validate_release_workflow(workflow_data))

        if issues:
            print("  ❌ Issues found:")
            for issue in issues:
                print(f"    - {issue}")
            all_valid = False
        else:
            print("  ✅ Valid")

    if all_valid:
        print("\n✅ All workflow files are valid!")
        return 0
    else:
        print("\n❌ Some workflow files have issues that need to be fixed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

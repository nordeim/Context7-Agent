#!/usr/bin/env python3
"""Generate detailed test coverage report."""
import subprocess
import sys
from pathlib import Path

def run_coverage():
    """Run coverage analysis and generate reports."""

    # Run tests with coverage
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "--cov=src",
        "--cov-report=html",
        "--cov-report=term-missing",
        "--cov-fail-under=90"
    ]

    try:
        subprocess.run(cmd, check=True)
        print("\n📊 Coverage report generated at: htmlcov/index.html")
    except subprocess.CalledProcessError as e:
        print(f"❌ Coverage check failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_coverage()


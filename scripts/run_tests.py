"""
Run the whole test suite with one command, locally and in CI.

    venv\\Scripts\\python.exe scripts/run_tests.py
    venv\\Scripts\\python.exe scripts/run_tests.py --offline

Each file under tests/ is already self-running — it has its own `__main__` block and
exits non-zero on failure — so this runs them rather than replacing them with a framework.
That keeps one way of running a test: the same command works on a single file during
development and across the suite in CI, and the project takes on no test dependency.

Each file runs in its own SUBPROCESS. These tests monkeypatch shared modules (httpx, and
market_data's own imports), and while each one restores what it replaced, a process per
file means a test that ever forgets cannot silently corrupt the next file's run.

--offline refuses every outbound socket first. The suite needs no network today, and that
is a property worth enforcing rather than rediscovering: a test that quietly starts
calling a live API still passes locally and then makes CI fail on someone else's outage.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TESTS = ROOT / "tests"

# Refuses connections, then hands over to the test file. Kept as source text because it
# has to run inside the child process, before that file imports anything.
OFFLINE_PREAMBLE = """
import socket, sys, runpy
def _refuse(*a, **k):
    raise OSError("NETWORK BLOCKED: this suite must not need the network")
socket.socket.connect = _refuse
socket.create_connection = _refuse
sys.argv = [{path!r}]
runpy.run_path({path!r}, run_name="__main__")
"""

SUMMARY = re.compile(r"(\d+)\s*/\s*(\d+)\s+passed")


def run_one(path: Path, offline: bool) -> tuple[bool, str]:
    """Run one test file, returning (ok, summary line)."""
    if offline:
        cmd = [sys.executable, "-c", OFFLINE_PREAMBLE.format(path=str(path))]
    else:
        cmd = [sys.executable, str(path)]

    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    output = (proc.stdout or "") + (proc.stderr or "")
    match = None
    for match in SUMMARY.finditer(output):
        pass  # the last summary line is the file's own total

    ok = proc.returncode == 0
    if match:
        detail = f"{match.group(1)}/{match.group(2)} passed"
    elif ok:
        detail = "passed (no summary line)"
    else:
        # A non-assertion error kills the child before it can summarise. Show the last
        # real line so the failure is identifiable without re-running by hand.
        lines = [ln for ln in output.strip().splitlines() if ln.strip()]
        # ASCII only: this prints to a Windows console, where cp1252 has crashed this
        # project's scripts on non-ASCII output before.
        detail = f"FAILED -- {lines[-1][:120]}" if lines else "FAILED (no output)"
    return ok, detail


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true",
                        help="refuse all network access while running")
    args = parser.parse_args()

    files = sorted(TESTS.glob("test_*.py"))
    if not files:
        print(f"No test files found in {TESTS}")
        return 1

    if args.offline:
        print("Network blocked for this run.\n")

    failed, total_passed, total_run = [], 0, 0
    for path in files:
        ok, detail = run_one(path, args.offline)
        print(f"  {'pass' if ok else 'FAIL'}  {path.name:26s} {detail}")
        if not ok:
            failed.append(path.name)
        counts = SUMMARY.search(detail)
        if counts:
            total_passed += int(counts.group(1))
            total_run += int(counts.group(2))

    print()
    if failed:
        print(f"{len(failed)} file(s) failed: {', '.join(failed)}")
        # Re-run the first failure in full, so CI logs carry the reason rather than
        # only the verdict.
        print(f"\n--- {failed[0]} ---")
        subprocess.run([sys.executable, str(TESTS / failed[0])], cwd=ROOT)
        return 1

    print(f"{total_passed}/{total_run} tests passed across {len(files)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())

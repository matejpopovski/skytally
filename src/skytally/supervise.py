"""Restart training after a hang or crash.

Runs ``skytally.train`` as a child process and restarts it from the last checkpoint on any
failure, up to a limit. The stall watchdog can only stop a stuck process; this is the half
that gets the run going again.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

from skytally.train import EXIT_BUDGET_REACHED


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-restarts", type=int, default=5)
    args, passthrough = parser.parse_known_args()
    command = [sys.executable, "-u", "-m", "skytally.train", *passthrough]

    restarts = 0
    while True:
        started = time.time()
        code = subprocess.call(command)
        if code in (0, EXIT_BUDGET_REACHED):
            sys.exit(0)
        restarts += 1
        print(
            f"training exited with {code} after {time.time() - started:.0f} s, "
            f"restart {restarts}/{args.max_restarts}",
            flush=True,
        )
        if restarts > args.max_restarts:
            sys.exit(code)
        time.sleep(5)


if __name__ == "__main__":
    main()

import subprocess
import sys
import time
from unittest import mock

import pytest

from skytally import supervise
from skytally.train import EXIT_BUDGET_REACHED

HANGING_CHILD = """
import faulthandler, threading
faulthandler.dump_traceback_later(0.3, exit=True)
threading.Event().wait()
"""


def test_watchdog_ends_a_process_that_is_stuck_forever():
    started = time.monotonic()
    result = subprocess.run(
        [sys.executable, "-c", HANGING_CHILD], capture_output=True, text=True, timeout=30
    )
    assert time.monotonic() - started < 15
    assert result.returncode != 0
    assert "Timeout" in result.stderr


def run_supervisor(codes, max_restarts=2):
    argv = ["supervise", "--max-restarts", str(max_restarts), "--config", "x.json"]
    with (
        mock.patch.object(sys, "argv", argv),
        mock.patch.object(supervise.subprocess, "call", side_effect=codes) as call,
        mock.patch.object(supervise.time, "sleep"),
        pytest.raises(SystemExit) as stop,
    ):
        supervise.main()
    return stop.value.code, call.call_count, call.call_args.args[0]


def test_supervisor_restarts_after_failures_until_training_finishes():
    code, calls, command = run_supervisor([1, -9, 0])
    assert code == 0
    assert calls == 3
    assert command[-2:] == ["--config", "x.json"]


def test_supervisor_treats_a_budget_stop_as_success():
    code, calls, _ = run_supervisor([EXIT_BUDGET_REACHED])
    assert (code, calls) == (0, 1)


def test_supervisor_gives_up_after_the_restart_limit():
    code, calls, _ = run_supervisor([1, 1, 1, 1], max_restarts=2)
    assert code == 1
    assert calls == 3

"""qpi/utils/progress.py: COMPLETE only after real success; non-zero exit otherwise."""
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qpi.utils.progress import Progress  # noqa: E402


def test_complete_is_printed_only_when_every_unit_succeeded(capsys):
    p = Progress("t", 3, stream=sys.stdout)
    p.stage("work")
    p.start("a")
    p.done("a", 1.0)
    p.skipped("b", "up to date")
    p.done("c", 0.1, note="x")
    assert p.finish() == 0
    out = capsys.readouterr().out
    assert out.strip().splitlines()[-1] == "COMPLETE"
    assert "3/3" in out and "stage: work" in out


def test_a_failed_unit_gives_exit_1_and_no_complete(capsys):
    p = Progress("t", 2)
    p.done("a")
    p.fail("b", ValueError("boom"))
    assert p.finish() == 1
    cap = capsys.readouterr()
    assert "COMPLETE" not in cap.out and "COMPLETE" not in cap.err
    assert "FAILED: 1 of 2" in cap.err and "boom" in cap.err


def test_unprocessed_units_and_a_failed_extra_check_are_not_complete(capsys):
    p = Progress("t", 3)
    p.done("a")
    assert p.finish() == 1                                   # 1 of 3 processed
    p = Progress("t", 1)
    p.done("a")
    assert p.finish(extra_ok=False, extra_message="read-back failed") == 1
    assert "COMPLETE" not in capsys.readouterr().out

"""Progress and exit contract for every long script: foreground, every line flushed, COMPLETE only after real success.

    prog = Progress("cell_analysis", total=45)
    prog.stage("analyse")
    prog.start("edge_sam | r8 | full | s42")      # optional: not printed for parallel units
    prog.done("edge_sam | r8 | full | s42", seconds=5.1, note="1269 cells")
    prog.skipped("edge_sam | r8 | full | s43", "up to date")
    prog.fail("mobile_sam | r32 | full | s44", exc)
    return prog.finish()                           # prints COMPLETE (exit 0) or FAILED: k of N (exit 1)
"""
from __future__ import annotations

import sys
import time
from typing import List, Optional, TextIO, Tuple


class Progress:
    def __init__(self, label: str, total: int, stream: Optional[TextIO] = None):
        self.label, self.total = label, int(total)
        self.out = stream if stream is not None else sys.stdout
        self.n_done = self.n_skipped = 0
        self.failures: List[Tuple[str, str]] = []
        self.t0 = time.time()
        self._stage = ""

    # ------------------------------------------------------------------ printing
    def _p(self, msg: str, err: bool = False) -> None:
        print(msg, file=(sys.stderr if err else self.out), flush=True)

    @property
    def finished_units(self) -> int:
        return self.n_done + self.n_skipped + len(self.failures)

    def _prefix(self) -> str:
        stage = f" | {self._stage}" if self._stage else ""
        return f"[{self.label}{stage}] {self.finished_units}/{self.total}"

    def stage(self, name: str) -> None:
        self._stage = name
        self._p(f"[{self.label}] --- stage: {name} ({time.time() - self.t0:.0f} s elapsed)")

    def start(self, unit: str) -> None:
        self._p(f"[{self.label}{' | ' + self._stage if self._stage else ''}] {self.finished_units + 1}/{self.total} {unit} started")

    def done(self, unit: str, seconds: Optional[float] = None, note: str = "") -> None:
        self.n_done += 1
        t = f" in {seconds:.1f} s" if seconds is not None else ""
        self._p(f"{self._prefix()} {unit} done{t}{(' - ' + note) if note else ''}")

    def skipped(self, unit: str, reason: str = "up to date") -> None:
        self.n_skipped += 1
        self._p(f"{self._prefix()} {unit} skipped ({reason})")

    def fail(self, unit: str, exc) -> None:
        msg = f"{type(exc).__name__}: {exc}" if isinstance(exc, BaseException) else str(exc)
        self.failures.append((unit, msg))
        self._p(f"{self._prefix()} {unit} FAILED: {msg}", err=True)

    # ------------------------------------------------------------------ end
    def finish(self, extra_ok: bool = True, extra_message: str = "") -> int:
        """Return the exit status. ``COMPLETE`` is printed only when every unit succeeded or was up to date,
        ``extra_ok`` (for example: the compiled file was written and read back) is true, and the unit count matches."""
        secs = time.time() - self.t0
        ok = not self.failures and extra_ok and self.finished_units == self.total
        if ok:
            self._p(f"[{self.label}] {self.n_done} done, {self.n_skipped} up to date, 0 failed, {secs:.0f} s")
            self._p("COMPLETE")
            return 0
        reasons = [f"{u}: {m}" for u, m in self.failures[:10]]
        if not extra_ok and extra_message:
            reasons.append(extra_message)
        if self.finished_units != self.total:
            reasons.append(f"{self.finished_units} of {self.total} units were processed")
        self._p(f"FAILED: {len(self.failures)} of {self.total} units failed" + ("" if not reasons else "; " + "; ".join(reasons)), err=True)
        return 1

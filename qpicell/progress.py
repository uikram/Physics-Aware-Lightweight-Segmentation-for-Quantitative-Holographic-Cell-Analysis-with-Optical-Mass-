"""Progress of runs read from their folders (works while the queue runs, or after it stopped).
State: complete / diverged (training.json), running (running.json with a live pid on this host, or a log written in
the last 15 min), failed (a log exists but the run neither finished nor is alive), waiting (nothing yet)."""
from __future__ import annotations

import json
import os
import socket
import time
from pathlib import Path
from typing import Dict, Optional


def bar(done: int, total: int, width: int = 30) -> str:
    f = done / total if total else 1.0; n = int(round(width * f))
    return "[" + "#" * n + "." * (width - n) + f"] {100 * f:3.0f}% {done}/{total}"


def hms(s: Optional[float]) -> str:
    if s is None:
        return "   -   "
    s = int(s); return f"{s // 3600:d}h{(s % 3600) // 60:02d}m"


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0); return True
    except (ProcessLookupError, PermissionError, OSError):
        return False


def run_state(tdir: Path, epochs: int) -> Dict:
    st: Dict = {"state": "waiting", "epoch": 0, "epochs": epochs, "val_dice": None, "elapsed_s": None, "eta_s": None}
    tj = tdir / "training.json"
    if tj.exists():
        try:
            r = json.loads(tj.read_text())
            st.update(state=r.get("status", "complete"), epoch=r.get("epochs_completed", epochs), wall_s=r.get("wall_seconds")); return st
        except Exception:
            pass
    h = tdir / "history.csv"
    if h.exists():
        lines = h.read_text().strip().splitlines()
        if len(lines) > 1:
            last = dict(zip(lines[0].split(","), lines[-1].split(",")))
            st["epoch"] = len(lines) - 1
            try:
                st["val_dice"] = float(last.get("val_dice", "nan"))
            except ValueError:
                pass
    lock, log = tdir / "running.json", tdir / "train.log"
    live, started = False, None
    if lock.exists():
        try:
            L = json.loads(lock.read_text()); started = L.get("started_unix")
            live = L.get("host") == socket.gethostname() and _alive(int(L.get("pid", -1)))
        except Exception:
            pass
    recent = log.exists() and time.time() - log.stat().st_mtime < 900
    if live or (recent and not lock.exists() and st["epoch"] < epochs):
        st["state"] = "running"
    elif log.exists():
        st["state"] = "failed"
    if st["state"] == "running" and started:
        st["elapsed_s"] = time.time() - started
        if st["epoch"]:
            st["eta_s"] = st["elapsed_s"] / st["epoch"] * (epochs - st["epoch"])
    return st


def tail(path: Path, n: int = 25) -> str:
    try:
        return "\n".join(Path(path).read_text(errors="replace").splitlines()[-n:])
    except Exception as e:
        return f"(cannot read {path}: {e})"

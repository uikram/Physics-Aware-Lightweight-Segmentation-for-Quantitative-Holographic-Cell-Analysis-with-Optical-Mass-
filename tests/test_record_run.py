"""record_run: a start record written before the dataset exists must get its manifest hash at the end stage."""
import importlib.util
import json
import sys
from pathlib import Path

from qpi.utils.io import sha256_file

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("record_run", ROOT / "scripts" / "record_run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_end_stage_fills_missing_manifest_hash(tmp_path, monkeypatch):
    rr = _load()
    proc = tmp_path / "processed"
    proc.mkdir()
    out = tmp_path / "full"
    out.mkdir()
    (out / "run_metadata.json").write_text(json.dumps({"data_manifest_sha256": None}))
    cfg = {"paths_abs": {"processed_root": str(proc)}}
    monkeypatch.setattr(rr, "load_config", lambda overlay=None: cfg)
    monkeypatch.setattr(rr, "claim_output_root", lambda c: out)
    monkeypatch.setattr(sys, "argv", ["record_run.py", "--stage", "end"])

    rr.main()  # no manifest yet: stays None
    assert json.loads((out / "run_metadata.json").read_text())["data_manifest_sha256"] is None

    (proc / "manifest.csv").write_text("split,stem\ntrain,a\n")
    rr.main()
    meta = json.loads((out / "run_metadata.json").read_text())
    assert meta["data_manifest_sha256"] == sha256_file(proc / "manifest.csv")


def test_end_stage_keeps_existing_hash(tmp_path, monkeypatch):
    rr = _load()
    proc = tmp_path / "processed"
    proc.mkdir()
    (proc / "manifest.csv").write_text("x\n")
    out = tmp_path / "full"
    out.mkdir()
    (out / "run_metadata.json").write_text(json.dumps({"data_manifest_sha256": "abc"}))
    monkeypatch.setattr(rr, "load_config", lambda overlay=None: {"paths_abs": {"processed_root": str(proc)}})
    monkeypatch.setattr(rr, "claim_output_root", lambda c: out)
    monkeypatch.setattr(sys, "argv", ["record_run.py", "--stage", "end"])
    rr.main()
    assert json.loads((out / "run_metadata.json").read_text())["data_manifest_sha256"] == "abc"


def test_fill_hash_only_fills_a_missing_hash_and_checks_the_prefix(tmp_path, monkeypatch, capsys):
    rr = _load()
    proc = tmp_path / "processed"
    proc.mkdir()
    (proc / "manifest.csv").write_text("split,stem\ntrain,a\n")
    sha = sha256_file(proc / "manifest.csv")
    out = tmp_path / "full"
    out.mkdir()
    rec = {"data_manifest_sha256": None, "started_utc": "T0", "finished_utc": "T1"}
    (out / "run_metadata.json").write_text(json.dumps(rec))
    monkeypatch.setattr(rr, "load_config", lambda overlay=None: {"paths_abs": {"processed_root": str(proc)}})
    monkeypatch.setattr(rr, "claim_output_root", lambda c: out)

    def run(*extra):
        monkeypatch.setattr(sys, "argv", ["record_run.py", "--stage", "fill-hash", *extra])
        return rr.main()

    assert run("--expect-sha-prefix", "0000") == 1          # wrong prefix: nothing written
    assert json.loads((out / "run_metadata.json").read_text())["data_manifest_sha256"] is None
    assert run("--expect-sha-prefix", sha[:12].upper()) == 0
    meta = json.loads((out / "run_metadata.json").read_text())
    assert meta["data_manifest_sha256"] == sha and meta["started_utc"] == "T0" and meta["finished_utc"] == "T1"
    meta["data_manifest_sha256"] = "kept"
    (out / "run_metadata.json").write_text(json.dumps(meta))
    assert run() == 0                                         # already recorded: untouched
    assert json.loads((out / "run_metadata.json").read_text())["data_manifest_sha256"] == "kept"

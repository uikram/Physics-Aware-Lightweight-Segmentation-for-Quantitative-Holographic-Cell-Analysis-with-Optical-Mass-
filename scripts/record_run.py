#!/usr/bin/env python
"""Write run_metadata.json (command, config, seeds, data manifest hash, environment, GPU info, package list).

    python scripts/record_run.py --stage start|end|fill-hash [--config-overlay smoke.yaml] [--note TEXT]
                                 [--expect-sha-prefix HEX]

``fill-hash`` only fills a missing ``data_manifest_sha256`` of an existing record (no timestamps are touched). With
``--expect-sha-prefix`` it refuses to write unless the current manifest hash starts with that prefix.
"""
import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qpi.config import load_config
from qpi.utils.env import collect_environment
from qpi.utils.io import read_json, sha256_file, write_json
from qpi.utils.outputs import claim_output_root, mode_of


def _run(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["start", "end", "fill-hash"])
    ap.add_argument("--expect-sha-prefix", default=None)
    ap.add_argument("--config-overlay", default=None)
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    cfg = load_config(a.config_overlay)
    root = claim_output_root(cfg)
    path = root / "run_metadata.json"
    meta = read_json(path) if path.exists() else {}
    now = datetime.now(timezone.utc).isoformat()
    if a.stage == "fill-hash":
        manifest = Path(cfg["paths_abs"]["processed_root"]) / "manifest.csv"
        if not path.exists() or not manifest.exists():
            print(f"[record_run] fill-hash: need both {path} and {manifest}", file=sys.stderr)
            return 1
        cur = sha256_file(manifest)
        if a.expect_sha_prefix and not cur.startswith(a.expect_sha_prefix.lower()):
            print(f"[record_run] fill-hash: manifest hash {cur[:16]}... does not start with {a.expect_sha_prefix}; nothing written",
                  file=sys.stderr)
            return 1
        if meta.get("data_manifest_sha256") is not None:
            print(f"[record_run] fill-hash: hash already recorded ({str(meta['data_manifest_sha256'])[:16]}...); nothing to do")
            return 0
        meta["data_manifest_sha256"] = cur
        meta["data_manifest_sha256_note"] = "filled afterwards by record_run.py --stage fill-hash (the start record preceded the dataset build)"
        write_json(path, meta)
        print(f"[record_run] fill-hash -> {path} ({cur[:16]}...)")
        return 0
    if a.stage == "start":
        proc = Path(cfg["paths_abs"]["processed_root"])
        manifest = proc / "manifest.csv"
        cfg_files = {p.name: sha256_file(p) for p in sorted((Path(__file__).resolve().parents[1] / "configs").glob("*.yaml"))}
        meta.update({
            "mode": mode_of(cfg), "started_utc": now, "seed": cfg["seed"], "config_overlay": a.config_overlay,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "config_file_sha256": cfg_files,
            "data_manifest_sha256": sha256_file(manifest) if manifest.exists() else None,
            "environment": collect_environment(None),
            "nvidia_smi": _run(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv"]),
            "pip_freeze": (_run([sys.executable, "-m", "pip", "freeze"]) or "").splitlines(),
            "note": a.note,
        })
    else:
        meta["finished_utc"] = now
        # The start record can precede the dataset build (a fresh data/processed is built after it), in which case the
        # manifest hash was stored as null. The runs are trained on the manifest that exists now, so fill it in here.
        manifest = Path(cfg["paths_abs"]["processed_root"]) / "manifest.csv"
        if meta.get("data_manifest_sha256") is None and manifest.exists():
            meta["data_manifest_sha256"] = sha256_file(manifest)
            meta["data_manifest_sha256_note"] = "filled at the end of the run: the dataset was built after the start record"
    write_json(path, meta)
    print(f"[record_run] {a.stage} -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Regenerate every manuscript figure and table from the project's result files.

    python manuscript/make_all.py                      # everything that the result files allow
    python manuscript/make_all.py --only tables        # tables only (or: figures, images)
    python manuscript/make_all.py --images             # also the image figures (needs data/prepared and the
                                                       # predictions_last.npz files: run on the training server)

Reads only:   results/, additional_analysis/results/, training_results/, configs/config.yaml
              (and, with --images, data/prepared/).
Writes only:  manuscript/figures/, manuscript/tables/, manuscript/captions.md, manuscript/SOURCES.json,
              manuscript/BUILD_REPORT.md.
Nothing is trained, evaluated or changed; every number is read from a result file, and SOURCES.json records for
every output which files (with SHA-256) it was built from. A product whose inputs are missing is skipped and named
in BUILD_REPORT.md rather than silently left out.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import platform
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from figtab import style  # noqa: E402
from figtab.core import Inputs, MissingInput, Runs  # noqa: E402
from figtab.writers import save_figure, write_docx, write_table, write_xlsx  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(HERE.parent), help="project folder (default: the folder above manuscript/)")
    ap.add_argument("--out", default=str(HERE), help="where figures/ and tables/ are written (default: manuscript/)")
    ap.add_argument("--only", choices=["all", "tables", "figures"], default="all")
    ap.add_argument("--images", action="store_true", help="also build the image figures (needs data/prepared)")
    ap.add_argument("--formats", default="pdf,svg,png", help="figure formats, comma-separated (pdf,svg,png,tif)")
    ap.add_argument("--dpi", type=int, default=600, help="raster resolution for png/tif")
    ap.add_argument("--checkpoint", default="last", choices=["last", "best"])
    ap.add_argument("--allow-stale", action="store_true",
                    help="TESTING ONLY: accept per-run files written before the rescore (their dry-mass statistics differ "
                         "slightly from runs_last.json)")
    args = ap.parse_args()

    root, out = Path(args.root).resolve(), Path(args.out).resolve()
    inp = Inputs(root)
    inp.allow_stale = args.allow_stale
    inp.product("run index")
    R = Runs(inp, args.checkpoint)
    formats = tuple(f.strip() for f in args.formats.split(",") if f.strip())
    style.apply()
    made, skipped = [], []

    def record(kind, number, name, files, caption=None):
        made.append({"kind": kind, "number": number, "name": name, "caption": caption,
                     "files": [Path(f).relative_to(out).as_posix() for f in files],
                     "inputs": dict(inp.log.get(inp.current, {}))})

    # ------------------------------------------------------------------ tables
    if args.only in ("all", "tables"):
        from figtab import tables as T
        built = []
        for fn in T.MAIN + T.SUPP:
            inp.product(fn.__name__)
            try:
                t = fn(inp, R)
            except MissingInput as e:
                skipped.append((fn.__name__, f"missing input {e}"))
                continue
            except Exception as e:  # noqa: BLE001 - report and continue with the other products
                skipped.append((fn.__name__, f"{type(e).__name__}: {e}"))
                traceback.print_exc()
                continue
            t.sources = list(inp.log.get(inp.current, {}).keys())
            files = write_table(t, out)
            built.append(t)
            record("table", t.number, t.title, files)
            print(f"[table] {t.label}: {len(t.rows)} rows", flush=True)
        main_t = [t for t in built if t.section == "main"]
        supp_t = [t for t in built if t.section == "supplementary"]
        for name, ts, head in (("main", main_t, "Main-text tables"), ("supplementary", supp_t, "Supplementary tables")):
            if not ts:
                continue
            x = write_xlsx(ts, out / "tables" / f"all_tables_{name}.xlsx")
            d = write_docx(ts, out / "tables" / f"all_tables_{name}.docx", head)
            print(f"[table] {name}: xlsx {'written' if x else 'skipped (openpyxl missing)'}, "
                  f"docx {'written' if d else 'skipped (python-docx missing)'}", flush=True)

    # ------------------------------------------------------------------ figures
    if args.only in ("all", "figures"):
        from figtab import figures as F
        products = list(F.MAIN) + list(F.SUPP)
        if args.images:
            from figtab import image_figures as IF
            products += list(IF.PRODUCTS)
        for fn in products:
            inp.product(fn.__name__)
            try:
                res = fn(inp, R)
            except MissingInput as e:
                skipped.append((fn.__name__, f"missing input {e}"))
                continue
            except Exception as e:  # noqa: BLE001
                skipped.append((fn.__name__, f"{type(e).__name__}: {e}"))
                traceback.print_exc()
                continue
            for fig, section, stem, number, caption in (res if isinstance(res, list) else [res]):
                files = save_figure(fig, out, section, stem, formats, args.dpi)
                import matplotlib.pyplot as plt
                plt.close(fig)
                record("figure", number, stem, files, caption)
                print(f"[figure] {number}: {stem}", flush=True)

        # Fig. 2 is a drawing kept in assets/ (see assets/fig2_setup/README.md), copied so the set is complete
        inp.product("fig2_setup")
        src = HERE / "assets" / "fig2_setup"
        if src.is_dir():
            import shutil
            dst = out / "figures" / "main"
            dst.mkdir(parents=True, exist_ok=True)
            files = []
            for f in sorted(src.glob("fig2_setup.*")):
                if f.suffix.lower() in (".pdf", ".svg", ".png"):
                    shutil.copy2(f, dst / f.name)
                    files.append(dst / f.name)
            record("figure", "2", "fig2_setup", files,
                   "Off-axis transmission Mach–Zehnder digital holographic microscope (drawing; see "
                   "assets/fig2_setup/README.md).")
            print("[figure] 2: fig2_setup (asset copied)", flush=True)
        else:
            skipped.append(("fig2_setup", "assets/fig2_setup/ not found"))

    # ------------------------------------------------------------------ provenance and report
    meta = {"created": _dt.datetime.now().isoformat(timespec="seconds"), "python": platform.python_version(),
            "root": root.name, "checkpoint": args.checkpoint, "command": " ".join(sys.argv)}
    try:
        import matplotlib
        import numpy
        import pandas
        meta.update(matplotlib=matplotlib.__version__, numpy=numpy.__version__, pandas=pandas.__version__)
    except ImportError:
        pass
    old = {}
    sp = out / "SOURCES.json"
    if sp.exists() and args.only != "all":
        try:
            old = {(m["kind"], m["name"]): m for m in json.loads(sp.read_text(encoding="utf-8"))["outputs"]}
        except (ValueError, KeyError):
            old = {}
    for m in made:
        old[(m["kind"], m["name"])] = m
    outputs = sorted(old.values(), key=lambda m: (m["kind"], _numkey(m["number"])))
    sp.write_text(json.dumps({"metadata": meta, "outputs": outputs, "skipped": skipped}, indent=1), encoding="utf-8")
    _write_captions(out, [(m["kind"], m["number"], m.get("caption")) for m in outputs
                          if m["kind"] == "figure" and m.get("caption")])
    rep = [f"# Build report\n", f"{meta['created']} — `{meta['command']}`\n",
           f"Built {len(made)} products; skipped {len(skipped)}.\n"]
    if skipped:
        rep += ["## Skipped\n"] + [f"* `{n}`: {why}" for n, why in skipped]
    (out / "BUILD_REPORT.md").write_text("\n".join(rep) + "\n", encoding="utf-8")
    print(f"done: {len(made)} products, {len(skipped)} skipped", flush=True)
    for n, why in skipped:
        print(f"  skipped {n}: {why}", flush=True)
    return 0


def _numkey(n: str):
    s = str(n)
    return (s.startswith("S"), int("".join(ch for ch in s if ch.isdigit()) or 0), s)


def _write_captions(out: Path, captions) -> None:
    lines = ["# Figure captions", "",
             "Generated with the figures; every number in a caption is read from the same files as the figure.", ""]
    for _, number, cap in sorted(captions, key=lambda c: _numkey(c[1])):
        lines += [f"**Fig. {number}.** {cap}", ""]
    (out / "captions.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())

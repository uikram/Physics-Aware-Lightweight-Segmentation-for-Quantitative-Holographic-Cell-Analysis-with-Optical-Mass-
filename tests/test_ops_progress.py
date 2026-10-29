"""GPU opt-in, progress viewer and queue helpers (no GPU, no real data)."""
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import progress  # noqa: E402
import run_queue  # noqa: E402
from qpi.utils import env  # noqa: E402


# ------------------------------------------------------------------------------------------------ GPU policy
def test_gpu_policy_default_and_explicit_opt_in(monkeypatch):
    monkeypatch.setattr(env.torch.cuda, "is_available", lambda: True)
    monkeypatch.delenv("QPI_ALLOW_ALL_GPUS", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    env.enforce_gpu_policy([2, 3])                                  # default policy unchanged
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,4")
    with pytest.raises(RuntimeError, match="not allowed"):
        env.enforce_gpu_policy([2, 3])
    monkeypatch.setenv("QPI_ALLOW_ALL_GPUS", "0")
    with pytest.raises(RuntimeError, match="not allowed"):
        env.enforce_gpu_policy([2, 3])
    monkeypatch.setenv("QPI_ALLOW_ALL_GPUS", "1")
    env.enforce_gpu_policy([2, 3])                                  # opted in for this launch
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    with pytest.raises(RuntimeError, match="not set to explicit ids"):
        env.enforce_gpu_policy([2, 3])                              # an explicit list is still required


# ------------------------------------------------------------------------------------------------ queue helpers
def test_queue_bar_and_finished_detection(tmp_path):
    assert run_queue.bar(0, 58).startswith("[" + "." * 30 + "]") and run_queue.bar(58, 58).count("#") == 30
    assert run_queue.bar(29, 58).count("#") == 15 and run_queue.bar(1, 0).count("#") == 30
    (tmp_path / "runs/a").mkdir(parents=True)
    (tmp_path / "runs/b").mkdir(parents=True)
    (tmp_path / "runs/c").mkdir(parents=True)
    (tmp_path / "runs/a/metrics.json").write_text(json.dumps({"status": "complete"}))
    (tmp_path / "runs/b/metrics.json").write_text('{"status": "compl')       # half-written: not finished
    assert run_queue.is_finished(tmp_path, "a") and not run_queue.is_finished(tmp_path, "b")
    assert not run_queue.is_finished(tmp_path, "c") and not run_queue.is_finished(tmp_path, "zzz")


# ------------------------------------------------------------------------------------------------ progress viewer
def test_last_epoch_reads_the_end_of_a_log(tmp_path):
    p = tmp_path / "r.log"
    p.write_text("[r] epoch 3/50 loss=0.9 val_dice=0.100 val_ipe=0.02\ntrain 4: 10%|#\r"
                 "[r] epoch 4/50 loss=0.8 val_dice=0.250 val_ipe=0.02\ntrain 5:  80%|########")
    assert progress.last_epoch(str(p)) == (4, 50, 0.25)
    p.write_text("loading model\n")
    assert progress.last_epoch(str(p)) is None and progress.last_epoch(str(tmp_path / "none.log")) is None


def test_eta_simulation():
    assert progress.eta_seconds([], []) == 0.0
    assert progress.eta_seconds([100.0, 300.0], []) == 300.0
    # two slots: 100 s and 300 s left; pending 200, 200, 200 -> slot A: 100+200+200 = 500, slot B: 300+200 = 500
    assert progress.eta_seconds([100.0, 300.0], [200.0, 200.0, 200.0]) == 500.0
    assert progress.fmt_duration(30) == "1min" and progress.fmt_duration(3 * 3600 + 5 * 60) == "3h 05min"


def _tree(tmp_path):
    od = tmp_path / "outputs/multiseed"
    (od / "logs/train").mkdir(parents=True)
    runs = [dict(run_id="a__s43", stage=1, priority=1, arch="edge_sam"), dict(run_id="b__s43", stage=1, priority=1, arch="mobile_sam"),
            dict(run_id="c__s43", stage=1, priority=2, arch="mobile_sam"), dict(run_id="d__s43", stage=2, priority=1, arch="edge_sam")]
    (od / "grid.json").write_text(json.dumps({"runs": runs}))
    (od / "runs/a__s43").mkdir(parents=True)
    (od / "runs/a__s43/metrics.json").write_text(json.dumps({"status": "complete", "wall_seconds": 500}))
    (od / "logs/train/b__s43.log").write_text("[b__s43] epoch 12/50 loss=0.6 val_dice=0.512 val_ipe=0.02\n")
    return od


def test_progress_render(tmp_path):
    assert "No multi-seed grid yet" in progress.render(str(tmp_path))
    _tree(tmp_path)
    text = progress.render(str(tmp_path))
    assert "TOTAL" in text and " 1/4" in text and "stage 1" in text and "stage 2" in text
    assert "b__s43" in text and "epoch 12/50" in text and "val_dice 0.512" in text
    assert "waiting: 2    finished: 1" in text and "estimated time left" in text and "measured" in text
    # a stale log (not written for > 3 minutes) is not 'running'
    old = tmp_path / "outputs/multiseed/logs/train/b__s43.log"
    os.utime(old, (1, 1))
    assert "running now (0)" in progress.render(str(tmp_path))


# ------------------------------------------------------------------------------------------------ seed-42 mirror
def test_seed42_mirror_links_verifies_and_never_overwrites(tmp_path):
    import link_seed42
    main, out = tmp_path / "full/runs", tmp_path / "multiseed/runs"
    for rid, body in (("edge_sam_lora_r8", '{"status": "complete"}'), ("edge_sam_lora_r4", '{"status": "complete"}')):
        (main / rid / "checkpoints").mkdir(parents=True)
        (main / rid / "metrics.json").write_text(body)
        (main / rid / "checkpoints" / "last.pt").write_bytes(b"weights" + rid.encode())
    (main / "edge_sam_lora_r2").mkdir()                               # no metrics.json: unfinished
    out.mkdir(parents=True)
    ids = ["edge_sam_lora_r8", "edge_sam_lora_r4", "edge_sam_lora_r2", "edge_sam_lora_r16"]
    created, verified, skipped, problems = link_seed42.mirror(main, out, ids, dry_run=True)
    assert created == ["edge_sam_lora_r8", "edge_sam_lora_r4"] and not (out / "edge_sam_lora_r8__s42").exists()
    created, verified, skipped, problems = link_seed42.mirror(main, out, ids)
    assert created == ["edge_sam_lora_r8", "edge_sam_lora_r4"] and skipped == ["edge_sam_lora_r2", "edge_sam_lora_r16"] and not problems
    d = out / "edge_sam_lora_r8__s42"
    assert (d / "metrics.json").read_text() == '{"status": "complete"}' and (d / "checkpoints/last.pt").read_bytes().startswith(b"weights")
    assert (d / link_seed42.NOTE).exists() and not (main / "edge_sam_lora_r8" / link_seed42.NOTE).exists()
    assert os.stat(d / "checkpoints/last.pt").st_ino == os.stat(main / "edge_sam_lora_r8/checkpoints/last.pt").st_ino   # hard link, no extra disk
    # second call: verified, nothing rewritten; a changed mirror is reported, not overwritten
    created, verified, skipped, problems = link_seed42.mirror(main, out, ids)
    assert created == [] and verified == ["edge_sam_lora_r8", "edge_sam_lora_r4"] and not problems
    (out / "edge_sam_lora_r4__s42" / "metrics.json").unlink()
    (out / "edge_sam_lora_r4__s42" / "metrics.json").write_text("changed")
    created, verified, skipped, problems = link_seed42.mirror(main, out, ids)
    assert len(problems) == 1 and "differs" in problems[0]

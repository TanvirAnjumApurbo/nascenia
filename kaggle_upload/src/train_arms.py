"""Run one training arm per GPU, in parallel, inside a single Kaggle session.

Kaggle bills a session by wall-clock, not per-GPU, so on "GPU T4 x2" the second
T4 is free compute -- but only if something occupies it. One run cannot use both:
Unsloth is mandatory on fp16 cards (Gemma-2 overflows to inf in plain fp16, see
`train_sft.py`) and has no multi-GPU path, so DDP is off the table. Two
independent single-GPU runs are, which is what this launches.

    python src/train_arms.py --max-rows 16000 --max-hours 10.5 \
        --arm A_gemma2:unsloth/gemma-2-2b:32 \
        --arm B_titulm:hishab/titulm-gemma-2-2b-v1.1:32

Each arm gets its own SFT export, because the length policy has to run with the
tokenizer that arm actually trains with -- two bases with different vocabularies
would otherwise silently share one file and one of them would be wrong.

Progress for every arm is polled from its `progress.json` and printed on one
line, so a six-hour cell shows a live table instead of two interleaved logs.
Full output per arm lands in `runs/<name>/train.log`.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("NASCENIA_ROOT", Path(__file__).resolve().parents[1]))
PY = sys.executable


def parse_arm(spec: str) -> dict:
    """`name:base:lora_r`. The base is a HF id and contains a slash, never a colon."""
    parts = spec.split(":")
    if len(parts) != 3:
        raise SystemExit(f"--arm wants name:base:lora_r, got {spec!r}")
    name, base, rank = parts
    if not rank.isdigit():
        raise SystemExit(f"--arm lora_r must be an integer, got {rank!r} in {spec!r}")
    return {"name": name, "base": base, "lora_r": int(rank)}


def visible_gpus() -> int:
    """Counted in a subprocess: importing torch here would initialise CUDA in the
    parent, and every child inherits that context for no reason."""
    out = subprocess.run(
        [PY, "-c", "import torch;print(torch.cuda.device_count())"],
        capture_output=True, text=True,
    )
    try:
        return int(out.stdout.strip())
    except ValueError:
        raise SystemExit(f"could not count GPUs: {out.stderr.strip()[:400]}")


def export_for(arm: dict, max_seq_len: int) -> tuple[Path, Path]:
    """One SFT export per arm, reused if it is already there.

    Serial and on CPU, so it happens before any GPU is claimed -- two tokenizer
    passes over 90k rows racing each other for RAM is a bad way to start a run.
    """
    prefix = f"sft_{arm['name']}"
    train = ROOT / "data" / f"{prefix}_train.jsonl"
    val = ROOT / "data" / f"{prefix}_val.jsonl"
    if train.exists() and val.exists():
        print(f"[{arm['name']}] reusing {train.name}")
        return train, val
    cmd = [PY, "-u", str(ROOT / "src" / "export_sft.py"),
           "--tokenizer", arm["base"], "--max-seq-len", str(max_seq_len),
           "--out-prefix", prefix]
    print(f"[{arm['name']}] $ {' '.join(cmd)}", flush=True)
    if subprocess.run(cmd, cwd=ROOT).returncode:
        raise SystemExit(f"export failed for {arm['name']}")
    return train, val


def launch(arm: dict, gpu: int, files: tuple[Path, Path], args) -> tuple[subprocess.Popen, object]:
    out_dir = ROOT / "runs" / arm["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    train_file, val_file = files

    cmd = [PY, "-u", str(ROOT / "src" / "train_sft.py"),
           "--run-name", arm["name"], "--base-model", arm["base"],
           "--lora-r", str(arm["lora_r"]),
           "--train-file", str(train_file), "--val-file", str(val_file),
           "--max-rows", str(args.max_rows), "--skip-rows", str(args.skip_rows),
           "--eval-rows", str(args.eval_rows), "--max-seq-len", str(args.max_seq_len),
           "--lr", str(args.lr), "--batch-size", str(args.batch_size),
           "--grad-accum", str(args.grad_accum), "--max-hours", str(args.max_hours),
           "--save-fraction", str(args.save_fraction),
           "--eval-fraction", str(args.eval_fraction),
           "--use-liger", str(args.use_liger),
           "--progress-every", str(args.progress_every)]
    if args.init_adapter:
        cmd += ["--init-adapter", args.init_adapter]
    elif args.resume:
        cmd += ["--resume", args.resume]

    # One visible device per child. Without this HF Trainer sees two GPUs and
    # wraps the model in nn.DataParallel, which breaks a 4-bit model pinned to
    # cuda:0 -- and both children would fight over the same card anyway.
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), NASCENIA_ROOT=str(ROOT))
    log = open(out_dir / "train.log", "w", encoding="utf-8")
    print(f"[{arm['name']}] -> GPU {gpu}, log {out_dir / 'train.log'}", flush=True)
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log,
                            stderr=subprocess.STDOUT, text=True)
    return proc, log


def poll_line(arm: dict) -> str:
    p = ROOT / "runs" / arm["name"] / "progress.json"
    if not p.exists():
        return f"  {arm['name']:<12} starting..."
    try:
        d = json.loads(p.read_text("utf-8"))
    except (json.JSONDecodeError, OSError):
        return f"  {arm['name']:<12} (progress file mid-write)"
    loss = "  ....  " if d.get("loss") is None else f"{d['loss']:.4f}"
    return (f"  {arm['name']:<12} {d['fraction']*100:5.1f}%  "
            f"step {d['step']}/{d['max_steps']}  loss {loss}  "
            f"{d['elapsed_hours']:.2f}h elapsed  ETA {d['eta_hours']:.2f}h")


def tail(path: Path, n: int = 25) -> str:
    try:
        return "".join(path.read_text("utf-8", errors="replace").splitlines(True)[-n:])
    except OSError:
        return "(no log)"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", action="append", required=True, metavar="NAME:BASE:LORA_R")
    ap.add_argument("--max-rows", type=int, default=16000)
    ap.add_argument("--skip-rows", type=int, default=0)
    ap.add_argument("--eval-rows", type=int, default=400)
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-4)
    # 2x8, not 4x4: without a fused loss the logits tensor is batch*seq*256000*4
    # bytes, which is 3.9 GB at batch 4 seq 1024 and OOMs a 16 GB T4.
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-hours", type=float, default=10.5)
    ap.add_argument("--use-liger", type=int, default=0,
                    help="fused linear cross-entropy on the hf backend; see train_sft.py")
    ap.add_argument("--save-fraction", type=float, default=0.25)
    ap.add_argument("--eval-fraction", type=float, default=0.5)
    ap.add_argument("--progress-every", type=int, default=120)
    ap.add_argument("--resume", default="")
    ap.add_argument("--init-adapter", default="")
    ap.add_argument("--poll-every", type=int, default=120)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    arms = [parse_arm(s) for s in args.arm]
    names = [a["name"] for a in arms]
    if len(set(names)) != len(names):
        raise SystemExit(f"arm names must be unique: {names}")

    n_gpu = visible_gpus()
    print(f"{n_gpu} GPU(s) visible, {len(arms)} arm(s) requested")
    if len(arms) > n_gpu:
        raise SystemExit(
            f"{len(arms)} arms but {n_gpu} GPU(s). Two arms sharing one card are "
            f"slower than running them one after the other -- drop an arm.")
    print(f"effective batch {args.batch_size}x{args.grad_accum}="
          f"{args.batch_size * args.grad_accum} per arm")

    if args.dry_run:
        for i, a in enumerate(arms):
            print(f"  would run {a['name']} ({a['base']}, r{a['lora_r']}) on GPU {i}")
        return 0

    files = [export_for(a, args.max_seq_len) for a in arms]

    procs = []
    for i, (a, f) in enumerate(zip(arms, files)):
        procs.append(launch(a, i, f, args))
        # Stagger: two 5 GB base checkpoints materialising at the same instant
        # is the one moment this can OOM host RAM rather than VRAM.
        if i + 1 < len(arms):
            time.sleep(30)

    t0 = time.time()
    while any(p.poll() is None for p, _ in procs):
        time.sleep(args.poll_every)
        print(f"\n=== {time.strftime('%H:%M:%S')}  "
              f"{(time.time() - t0)/3600:.2f}h into the session ===", flush=True)
        for a, (p, _) in zip(arms, procs):
            state = "running" if p.poll() is None else f"exited {p.returncode}"
            print(f"{poll_line(a)}   [{state}]", flush=True)

    for _, log in procs:
        log.close()

    failed = []
    print("\n" + "=" * 70)
    for a, (p, _) in zip(arms, procs):
        ok = p.returncode == 0
        print(f"{a['name']:<12} exit {p.returncode}  {'OK' if ok else 'FAILED'}")
        if not ok:
            failed.append(a)
    for a in failed:
        print(f"\n--- last 25 lines of runs/{a['name']}/train.log ---")
        print(tail(ROOT / "runs" / a["name"] / "train.log"))

    for a in arms:
        sp = ROOT / "runs" / a["name"] / "chunk_state.json"
        if sp.exists():
            s = json.loads(sp.read_text("utf-8"))
            flag = "" if s["finished"] else "  <-- stopped early, re-run with --resume auto"
            print(f"{a['name']:<12} finished={s['finished']}  "
                  f"next_skip_rows={s['next_skip_rows']}{flag}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

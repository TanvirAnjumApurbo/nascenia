"""QLoRA supervised fine-tuning for the Bengali medical dialogue task.

Runs unchanged on Kaggle (T4, fp16, Unsloth required) and locally (RTX 5060 Ti,
bf16 native). Pick with `--backend`; `auto` uses Unsloth when it imports.

    # Kaggle A/B, 8k rows
    python src/train_sft.py --base-model unsloth/gemma-2-2b --max-rows 8000 \
        --lora-r 32 --run-name A_gemma2

    # local main run, full data
    python src/train_sft.py --base-model unsloth/gemma-2-2b \
        --lora-r 64 --lora-alpha 128 --run-name main --save-fraction 0.25

Long runs are built to be interrupted, in two different senses:

* `--max-hours 11 --resume auto` -- *same* run, continued. Restores optimizer,
  scheduler and data order, so the result is what an uninterrupted run would
  have produced. Re-running the identical command after a crash is always safe:
  `auto` starts fresh when there is no checkpoint. This is the one to use when a
  session died or Kaggle's 12 h cap is about to hit.
* `--init-adapter <dir> --skip-rows N` -- *new* run from finished weights, on
  data the previous chunk never saw. The schedule restarts, which is the point:
  it lets a 90 k-row epoch be spread across sessions, machines or accounts,
  carrying only the ~100 MB adapter instead of a full optimizer state. Each
  chunk leaves a `chunk_state.json` naming the next `--skip-rows`.

Two things here are deliberate and load-bearing:

* **Completion-only loss.** The patient question is ~30% of every sequence and
  the model is never asked to produce it. Labels are masked to -100 across the
  prompt. We build `input_ids`/`labels` by hand instead of going through TRL's
  collator: the split point is exact (verified as a token prefix, see
  `check_export`), and it removes a dependency whose API has churned hard.

* **Gemma-2 needs `attn_implementation="eager"`.** Its attention and final logits
  are soft-capped, and the SDPA/flash paths silently drop the capping. Turing
  (T4) has no FlashAttention-2 anyway.

On fp16 hardware Unsloth is not a speed preference, it is a correctness
requirement: Gemma-2 in plain fp16 autocast overflows to inf on T4.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(os.environ.get("NASCENIA_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Before torch is imported, or the allocator is already configured. Cross-entropy
# over Gemma's 256k vocab asks for one enormous contiguous block once per step
# (batch*seq*vocab*4 bytes), and the default allocator strands memory around it --
# a real OOM here reported 1.18 GB "reserved but unallocated". Expandable segments
# hand that back instead of fragmenting.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base-model", default="unsloth/gemma-2-2b")
    ap.add_argument("--train-file", default=str(ROOT / "data" / "sft_train.jsonl"))
    ap.add_argument("--val-file", default=str(ROOT / "data" / "sft_val.jsonl"))
    ap.add_argument("--output-dir", default=None, help="default: runs/<run-name>")
    ap.add_argument("--run-name", default="run")

    ap.add_argument("--max-rows", type=int, default=0, help="0 = all rows")
    ap.add_argument("--skip-rows", type=int, default=0,
                    help="start this far into the shuffled order -- chunk N uses "
                         "--skip-rows N*max_rows so chunks never overlap")
    ap.add_argument("--eval-rows", type=int, default=500)
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--epochs", type=float, default=1.0)

    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=0, help="0 -> 2*r")
    ap.add_argument("--lora-dropout", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--scheduler", default="cosine")
    ap.add_argument("--weight-decay", type=float, default=0.0)

    ap.add_argument("--batch-size", type=int, default=1, help="per device")
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--save-fraction", type=float, default=0.25,
                    help="checkpoint every this fraction of an epoch; 0 disables")
    ap.add_argument("--eval-fraction", type=float, default=0.25,
                    help="eval every this fraction of an epoch; 0 = final eval only")
    ap.add_argument("--logging-steps", type=int, default=10)

    ap.add_argument("--backend", choices=["auto", "unsloth", "hf"], default="auto")
    ap.add_argument("--use-liger", type=int, default=0,
                    help="hf backend: fused linear cross-entropy (pip install "
                         "liger-kernel). Gemma's 256k vocab makes the logits tensor "
                         "batch*seq*256000*4 bytes -- 3.9 GB at batch 4, seq 1024 -- "
                         "and Liger never materialises it. This is what lets the "
                         "batch size go up on a 16 GB card.")
    ap.add_argument("--load-in-4bit", type=int, default=1)
    ap.add_argument("--upcast-fp32", type=int, default=-1,
                    help="hf backend: fp32-upcast frozen weights. -1 = only on fp16 hardware")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", default=None,
                    help="'auto' = newest checkpoint-* in the output dir, or a path. "
                         "Restores optimizer, scheduler, step count and data order.")
    ap.add_argument("--init-adapter", default=None,
                    help="warm-start LoRA weights from a finished adapter, fresh "
                         "optimizer/schedule. Use with --skip-rows to continue on "
                         "new data across sessions or accounts.")
    ap.add_argument("--max-hours", type=float, default=0.0,
                    help="stop cleanly and save after this long. Set it below the "
                         "platform's session cap (Kaggle 12h) so a run always "
                         "produces a usable adapter.")
    ap.add_argument("--progress-every", type=int, default=60,
                    help="seconds between progress lines / progress.json writes")
    return ap.parse_args(argv)


def resolve_resume(spec: str | None, out_dir: Path) -> str | None:
    """`--resume auto` -> newest `checkpoint-N` in the output dir, else None.

    'auto' returning None on a clean directory is what makes one command safe to
    run twice: the first call trains from scratch, a re-run after a crash picks
    up where it stopped, and neither needs the operator to know which happened.
    """
    if spec is None:
        return None
    if spec != "auto":
        if not Path(spec).exists():
            raise SystemExit(f"--resume {spec} does not exist")
        return spec
    ckpts = [p for p in out_dir.glob("checkpoint-*") if (p / "trainer_state.json").exists()]
    if not ckpts:
        print("--resume auto: no checkpoint found, starting fresh")
        return None
    newest = max(ckpts, key=lambda p: int(p.name.split("-")[1]))
    print(f"--resume auto: resuming from {newest.name}")
    return str(newest)


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def pick_backend(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import unsloth  # noqa: F401
        return "unsloth"
    except Exception:
        return "hf"


def load_model(args, backend: str, dtype):
    """-> (model, tokenizer). Unsloth must be imported before transformers."""
    if backend == "unsloth":
        from unsloth import FastLanguageModel

        # Pointed at an adapter directory, Unsloth reads `base_model_name_or_path`
        # from adapter_config.json and rebuilds base+LoRA itself. That is its
        # supported continue-training path, so prefer it over attaching PEFT by hand.
        model, tok = FastLanguageModel.from_pretrained(
            model_name=args.init_adapter or args.base_model,
            max_seq_length=args.max_seq_len,
            dtype=dtype,
            load_in_4bit=bool(args.load_in_4bit),
        )
        if args.init_adapter:
            FastLanguageModel.for_training(model)
        else:
            model = FastLanguageModel.get_peft_model(
                model,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=TARGET_MODULES,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=args.seed,
            )
        return model, tok

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    quant = None
    if args.load_in_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quant,
        dtype=dtype,
        attn_implementation="eager",  # Gemma-2 soft-capping; see module docstring
        device_map={"": 0},
    )
    tok = AutoTokenizer.from_pretrained(args.base_model)

    # NOT `prepare_model_for_kbit_training`. It casts every non-4bit parameter to
    # fp32, and Gemma-2's tied 256k-vocab embedding is 589.8M parameters -- 2.36 GB
    # in fp32 against 1.18 GB in bf16. On an 8 GB card with ~6.2 GB free that
    # single upcast is the difference between fitting comfortably and thrashing at
    # the ceiling, and it buys nothing here: the embedding is frozen, and bf16 is
    # what the model was pretrained in.
    #
    # The upcast does earn its keep on fp16-only hardware (T4), where fp32 norms
    # are a real stability aid -- hence the flag rather than an outright removal.
    # On Kaggle that path is Unsloth's anyway.
    if args.upcast_fp32:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        for p in model.parameters():
            p.requires_grad = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()  # so checkpointing sees a grad-requiring input
    model.config.use_cache = False
    if args.init_adapter:
        from peft import PeftModel

        # `is_trainable=True` or PEFT loads the adapter frozen for inference and
        # the run burns its whole budget updating nothing.
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
        print(f"warm-started LoRA from {args.init_adapter}")
    else:
        model = get_peft_model(
            model,
            LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=TARGET_MODULES,
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
    return model, tok


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def build_dataset(path: str, tok, max_seq_len: int, max_rows: int, seed: int,
                  skip_rows: int = 0):
    """Tokenise into input_ids + completion-only labels.

    `render_prompt` output is an exact token prefix of `render_example` output, so
    the mask boundary is just a length -- no string search, no off-by-one.

    Row selection is a slice of one fixed shuffle, not a fresh shuffle per call.
    That is what makes chunked training honest: chunk 0 takes [0, N), chunk 1
    takes [N, 2N) of the *same* permutation, so a model trained across sessions
    sees each example once rather than re-reading a random third of the corpus.
    """
    import datasets

    from prompt import render_example, render_prompt

    ds = datasets.load_dataset("json", data_files=path, split="train")
    if skip_rows or (max_rows and max_rows < len(ds)):
        ds = ds.shuffle(seed=seed)
        if skip_rows >= len(ds):
            raise SystemExit(f"--skip-rows {skip_rows} >= {len(ds)} rows: nothing left to train on")
        end = min(len(ds), skip_rows + max_rows) if max_rows else len(ds)
        ds = ds.select(range(skip_rows, end))
        print(f"rows [{skip_rows}, {end}) of the seed-{seed} shuffle")

    def encode(batch):
        full = [render_example(i, o) for i, o in zip(batch["input"], batch["output"])]
        prompts = [render_prompt(i) for i in batch["input"]]
        full_ids = tok(full, add_special_tokens=False)["input_ids"]
        prompt_ids = tok(prompts, add_special_tokens=False)["input_ids"]

        out_ids, out_labels, lengths = [], [], []
        for ids, pids in zip(full_ids, prompt_ids):
            ids = ids[:max_seq_len]
            n_prompt = min(len(pids), len(ids))
            labels = [-100] * n_prompt + ids[n_prompt:]
            out_ids.append(ids)
            out_labels.append(labels)
            lengths.append(len(ids))
        return {"input_ids": out_ids, "labels": out_labels, "length": lengths}

    return ds.map(encode, batched=True, batch_size=1000, remove_columns=ds.column_names,
                  desc="tokenising")


def count_inference_params(model) -> int:
    """True parameter count of the model that would be deployed.

    Two corrections over a naive `sum(p.numel())`:

      * bitsandbytes packs two nf4 values into one uint8, so `numel()` on a 4-bit
        weight reports half the real size. The logical shape is on `quant_state`.
      * LoRA A/B are excluded: merging folds them into the existing weights and
        adds nothing at inference.

    This is the number Phase 2 checks against the 3B cap, so it is worth getting
    right here rather than discovering it during verification.
    """
    import numpy as np

    total = 0
    for name, p in model.named_parameters():  # dedupes tied weights by default
        if "lora_" in name:
            continue
        qs = getattr(p, "quant_state", None)
        shape = getattr(qs, "shape", None) if qs is not None else None
        total += int(np.prod(tuple(shape))) if shape is not None else p.numel()
    return total


def filter_training_args(kwargs: dict) -> dict:
    """Drop TrainingArguments keys this transformers version does not accept.

    We run on two very different stacks -- transformers 5.12 locally, whatever
    Unsloth pins on Kaggle -- and the argument list churns between them
    (`group_by_length` is gone in 5.12 while `length_column_name` survives).
    Failing the whole run over a renamed convenience flag is worse than losing
    the flag, but silently losing one is worse still, so it is reported.
    """
    import inspect

    from transformers import TrainingArguments

    allowed = set(inspect.signature(TrainingArguments.__init__).parameters)
    dropped = sorted(set(kwargs) - allowed)
    if dropped:
        print(f"NOTE: TrainingArguments does not accept {dropped} in this "
              f"transformers version -- ignoring.")
    return {k: v for k, v in kwargs.items() if k in allowed}


def make_callbacks(out_dir: Path, args, total_rows: int):
    """Progress reporting and a wall-clock stop, as TrainerCallbacks.

    Progress goes to a file as well as stdout because Kaggle only shows a
    notebook's output when the cell ends -- a 6 h cell is a blank screen
    otherwise -- and because a `progress.json` can be polled from another
    process without touching the training loop.

    The time limit exists because Kaggle kills a session at 12 h with no warning
    and no save. Stopping ourselves at 11 h turns that cliff into a checkpoint.
    """
    import time as _time

    from transformers import TrainerCallback

    class ProgressCallback(TrainerCallback):
        def on_train_begin(self, targs, state, control, **kw):
            self.t0 = _time.time()
            self.last = 0.0
            self.loss = float("nan")
            print(f"training {state.max_steps} steps over {total_rows} rows", flush=True)

        def on_log(self, targs, state, control, logs=None, **kw):
            if logs and "loss" in logs:
                self.loss = logs["loss"]

        def _loss_str(self):
            # Trainer only emits a loss every `logging_steps`, so the first
            # progress lines of a run have nothing to show. Say so rather than
            # printing a nan that reads like divergence.
            return "  ....  " if self.loss != self.loss else f"{self.loss:.4f}"

        def on_step_end(self, targs, state, control, **kw):
            now = _time.time()
            if now - self.last < args.progress_every and state.global_step != state.max_steps:
                return
            self.last = now
            done = state.global_step / max(state.max_steps, 1)
            elapsed = now - self.t0
            eta = elapsed / max(done, 1e-9) - elapsed
            rows = state.global_step * args.batch_size * args.grad_accum
            line = (f"[{done*100:5.1f}%] step {state.global_step}/{state.max_steps} | "
                    f"loss {self._loss_str()} | {rows}/{total_rows} rows | "
                    f"{elapsed/3600:.2f}h elapsed | ETA {eta/3600:.2f}h")
            print(line, flush=True)
            (out_dir / "progress.json").write_text(json.dumps({
                "run_name": args.run_name, "step": state.global_step,
                "max_steps": state.max_steps, "fraction": round(done, 4),
                "loss": None if self.loss != self.loss else self.loss,  # JSON has no NaN
                "rows_seen": rows, "total_rows": total_rows,
                "elapsed_hours": round(elapsed / 3600, 3),
                "eta_hours": round(eta / 3600, 3),
                "epoch": round(state.epoch or 0, 3),
                "updated": _time.strftime("%Y-%m-%d %H:%M:%S"),
            }, indent=2), "utf-8")

    class TimeLimitCallback(TrainerCallback):
        def on_train_begin(self, targs, state, control, **kw):
            self.deadline = _time.time() + args.max_hours * 3600

        def on_step_end(self, targs, state, control, **kw):
            if _time.time() > self.deadline:
                print(f"\n--max-hours {args.max_hours} reached at step "
                      f"{state.global_step}/{state.max_steps}: saving and stopping. "
                      f"Continue with --resume auto.", flush=True)
                control.should_save = True
                control.should_training_stop = True
            return control

    cbs = [ProgressCallback()]
    if args.max_hours > 0:
        cbs.append(TimeLimitCallback())
    return cbs


@dataclass
class PadCollator:
    pad_token_id: int
    pad_to_multiple_of: int = 8

    def __call__(self, features):
        import torch

        longest = max(len(f["input_ids"]) for f in features)
        m = self.pad_to_multiple_of
        width = ((longest + m - 1) // m) * m

        input_ids, labels, attn = [], [], []
        for f in features:
            ids, lab = list(f["input_ids"]), list(f["labels"])
            pad = width - len(ids)
            input_ids.append(ids + [self.pad_token_id] * pad)
            labels.append(lab + [-100] * pad)      # padding never contributes loss
            attn.append([1] * len(ids) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    args = parse_args(argv)
    if args.lora_alpha == 0:
        args.lora_alpha = 2 * args.lora_r
    out_dir = Path(args.output_dir or (ROOT / "runs" / args.run_name))
    out_dir.mkdir(parents=True, exist_ok=True)

    resume = resolve_resume(args.resume, out_dir)
    if resume and args.init_adapter:
        # A checkpoint already contains its adapter weights; loading a different
        # one underneath it would be silently overwritten on restore.
        raise SystemExit("--resume and --init-adapter are mutually exclusive: "
                         "resume continues one run, init-adapter starts a new one "
                         "from finished weights.")

    backend = pick_backend(args.backend)
    if backend == "unsloth":
        import unsloth  # noqa: F401  (must precede transformers for its patches)

    import torch
    from transformers import Trainer, TrainingArguments, set_seed

    bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16
    if not bf16 and backend != "unsloth":
        print(
            "WARNING: fp16 hardware without Unsloth. Gemma-2 overflows to inf in "
            "plain fp16 autocast on T4/V100. Use --backend unsloth.",
            file=sys.stderr,
        )

    if args.upcast_fp32 == -1:
        args.upcast_fp32 = int(not bf16)

    set_seed(args.seed)
    print(f"backend={backend}  dtype={dtype}  gpu={torch.cuda.get_device_name(0)}  "
          f"upcast_fp32={bool(args.upcast_fp32)}")

    model, tok = load_model(args, backend, dtype)
    n_params = count_inference_params(model)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params at inference: {n_params:,} = {n_params/1e9:.4f}B "
          f"({'OK, under the 3B cap' if n_params <= 3_000_000_000 else 'OVER THE 3B CAP'}); "
          f"{n_train/1e6:.1f}M trainable")
    if n_params > 3_000_000_000:
        raise SystemExit("base model exceeds the 3B cap -- would be disqualified at Phase 2")

    train_ds = build_dataset(args.train_file, tok, args.max_seq_len, args.max_rows,
                             args.seed, args.skip_rows)
    eval_ds = build_dataset(args.val_file, tok, args.max_seq_len, args.eval_rows, args.seed)
    tokens_per_epoch = sum(train_ds["length"])
    print(f"train rows={len(train_ds)}  eval rows={len(eval_ds)}  "
          f"tokens/epoch={tokens_per_epoch/1e6:.1f}M")

    eff_batch = args.batch_size * args.grad_accum
    steps_per_epoch = max(1, len(train_ds) // eff_batch)
    save_steps = max(1, int(steps_per_epoch * args.save_fraction)) if args.save_fraction else 0
    eval_steps = max(1, int(steps_per_epoch * args.eval_fraction)) if args.eval_fraction else 0

    targs = TrainingArguments(**filter_training_args(dict(
        output_dir=str(out_dir),
        run_name=args.run_name,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type=args.scheduler,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        optim="adamw_8bit",
        bf16=bf16,
        fp16=not bf16,
        logging_steps=args.logging_steps,
        save_strategy="steps" if save_steps else "no",
        save_steps=save_steps or 500,
        save_total_limit=6,
        eval_strategy="steps" if eval_steps else "no",
        eval_steps=eval_steps or 500,
        # Batches of similar length -> less padding waste when sequences run
        # 200..1024 tokens. Best-effort: transformers 5.12 dropped this argument,
        # and at per-device batch 1-2 the waste it saves is small anyway.
        group_by_length=True,
        length_column_name="length",
        # Dropped by filter_training_args on a transformers that predates it, and
        # only meaningful on the hf backend -- Unsloth already fuses the loss.
        use_liger_kernel=bool(args.use_liger) and backend == "hf",
        # Keep `length` on the dataset: the length-grouped sampler reads it, and
        # the collator ignores columns it does not use.
        remove_unused_columns=False,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        max_grad_norm=1.0,
    )))

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0  # Gemma's is 0, which is falsy
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=PadCollator(pad_id),
        callbacks=make_callbacks(out_dir, args, len(train_ds)),
    )

    config = {**vars(args), "backend": backend, "dtype": str(dtype),
              "total_params": n_params, "trainable_params": n_train,
              "tokens_per_epoch": tokens_per_epoch, "steps_per_epoch": steps_per_epoch,
              "save_steps": save_steps, "eval_steps": eval_steps,
              "gpu": torch.cuda.get_device_name(0)}
    (out_dir / "train_config.json").write_text(json.dumps(config, indent=2), "utf-8")

    t0 = time.time()
    result = trainer.train(resume_from_checkpoint=resume)
    elapsed = time.time() - t0
    finished = trainer.state.global_step >= trainer.state.max_steps

    # Tokenizer first: it is small, so an interrupted save is far less likely to
    # leave a partial directory. A half-written tokenizer loads without error as a
    # tiny broken vocabulary rather than raising -- see generate.check_tokenizer.
    adapter_dir = out_dir / "adapter"
    tok.save_pretrained(str(adapter_dir))
    model.save_pretrained(str(adapter_dir))

    from transformers import AutoTokenizer

    reloaded = AutoTokenizer.from_pretrained(str(adapter_dir))
    assert len(reloaded) == len(tok), (
        f"tokenizer did not round-trip: saved {len(tok)} tokens, reloaded {len(reloaded)}"
    )

    # Always land a final eval loss: it is the cheap, directly comparable signal
    # between two bases that share a tokenizer, and it costs a couple of minutes.
    final_eval = trainer.evaluate()

    seen = trainer.state.global_step / max(trainer.state.max_steps, 1) * len(train_ds)
    metrics = {**result.metrics, **final_eval, "wall_seconds": elapsed,
               "tokens_per_second": tokens_per_epoch * args.epochs / max(elapsed, 1),
               "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
               "finished": finished,
               "hours_per_full_epoch": round(
                   54.9e6 / max(tokens_per_epoch * args.epochs / max(elapsed, 1), 1e-9) / 3600, 2)}
    (out_dir / "train_metrics.json").write_text(json.dumps(metrics, indent=2), "utf-8")

    # What the *next* chunk needs to know, so continuing is a copy-paste rather
    # than arithmetic done at 2am against a half-remembered row count.
    next_skip = args.skip_rows + int(seen)
    (out_dir / "chunk_state.json").write_text(json.dumps({
        "base_model": args.base_model, "run_name": args.run_name,
        "skip_rows": args.skip_rows, "max_rows": args.max_rows,
        "rows_consumed": int(seen), "next_skip_rows": next_skip,
        "finished": finished, "seed": args.seed,
        "adapter": str(out_dir / "adapter"),
    }, indent=2), "utf-8")

    print(json.dumps(metrics, indent=2))
    print(f"adapter -> {out_dir / 'adapter'}")
    if not finished:
        print(f"\nSTOPPED EARLY at step {trainer.state.global_step}/{trainer.state.max_steps}. "
              f"Continue this run:  --resume auto")
    print(f"Next chunk on fresh data:  --init-adapter {out_dir / 'adapter'} "
          f"--skip-rows {next_skip}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose of this directory

This is the working directory for the **Nascenia AI Hackathon** (Kaggle): *Bengali Medical Dialogue Generation*. Given a Bengali patient prompt (`input`), generate the doctor's response (`output`). Everything here exists to produce one Kaggle `submission.csv` and, if the entry lands top-10, a reproducible Phase 2 package.

Not a git repo; no build or lint step, and the only test is `src/check_metric.py`. `rulebook_and_context.md` is the competition spec, but it contradicts itself in places — the live Kaggle page and anything measured empirically outrank it (see the settled items below).

`src/` is a pipeline of standalone CLI scripts, not a library. Each one reads files, asserts its own invariants, writes files, and prints what it did; they compose through `data/` and `runs/` rather than through imports. `metric.py` and `prompt.py` are the only real shared modules. Every script honours `NASCENIA_ROOT` (default: repo root) so the same code runs unmodified on Kaggle at `/kaggle/working/ns`.

## Competition rules that bind the work

- **Model ≤ 3B parameters at inference.** Hard cap, verified in Phase 2 by loading the weights. Ensembles count *combined* parameters. LoRA/adapters, quantization, and distillation are all allowed on top. Exceeding or misrepresenting the count is disqualification.
- **Submissions: 5 per day** (verified on the live Kaggle page, 2026-08-05). Rulebook §8 reads "5 during Phase 1"; the live page is authoritative when they conflict. Roughly 95 submissions over the run, so the leaderboard *is* usable — but choose locally on `val_dev`/`val_test` and use the leaderboard only to confirm. Two submissions scored on the *same* public rows can be compared exactly; a single absolute score cannot, since the public split is a fraction of 1,000 rows and private LB decides rank.
- **Metric (Phase 1), as published:** `0.5 × BERTScore F1 + 0.3 × Token F1 + 0.2 × ROUGE-L F1`, per example, averaged. **In practice the BERTScore term is a constant** — see the measurement below — so surface form, phrasing and length are what actually move rank. That is also the reasoning behind the "deliberately not normalized" section.
- **Phase 2 (top 10 only, 20% of final rank):** blind LLM-as-judge on *medical diagnostic accuracy* and *response quality*, on a separate judging set. Do not optimize so hard for n-gram overlap that clinical soundness suffers. Requires a reproducible inference script that regenerates the leaderboard outputs within tolerance — **pin seeds and generation config from the first submission onward**.
- **Using test inputs for training/fine-tuning is an explicit disqualification rule**, not just good hygiene.
- External data is permitted but must be disclosed in the submission notes.
- Dataset is CC BY-NC 4.0, competition use only — do not redistribute it or paste it into external services.

**Timeline** (GMT+6): data released Aug 4 · **Phase 1 deadline Aug 24, 00:00** · Phase 2 package due Aug 25 · results Aug 26.

**Submission column is settled: `id,output`.** Submission 001 was accepted and scored with it, so the Evaluation section's `id,doctor_response` is the wrong half of the rulebook's contradiction. One rulebook discrepancy remains: the Phase 2 deadline reads "August 25th, 12 PM" in §5.1 and "August 25th, 00:00" in the §7 timeline — build to 00:00.

**The organisers' BERTScore is saturated — measured, not guessed (2026-08-05).** Two constant-answer submissions differing only in lexical match moved the leaderboard by 0.01072 while our full composite moved 0.01975 and our lexical half moved 0.01268. The LB tracks the lexical drop. Solving `(LB − our lexical)/0.5` on each gives **0.9445 and 0.9484** — two independent estimates agreeing within 0.004, on identical public rows, so no split noise is involved.

> **Effective leaderboard metric: `≈ 0.473 + 0.3×TokenF1 + 0.2×ROUGE-L`.**

Consequences, and they are large:
- **Rank models on the lexical score (`metric.py --fast`), not the composite.** Ours weights BERTScore 0.5 where the LB weights it ~0, so the composite will mis-order any two models that trade fluency against overlap.
- Length matching to the 102-word reference mean is first-order — both surviving metrics are F1s and nothing else absorbs the error.
- MBR with the task's own lexical utility is the top decode move, not one option among several.
- Gains look small: +0.05 lexical is +0.05 LB against a 0.574 floor from a fixed string. The band is 0.574 → 0.973 and all of it is lexical.
- Saturation holds *in the fluent-Bengali-medical regime*; their BERTScore would still punish empty or English output. And Phase 2 judges clinical accuracy, so overlap-chasing must not gut medical content.

Full derivation in `submissions/submissions.md`.

## Hard constraints

- **Never touch `data/test.csv`.** Do not modify it, and do not derive any cleaning threshold, statistic, or decision from it. Reading its `id` column to build a submission is the only permitted use. This is both the user's standing instruction and a competition rule.
- **Never derive a cleaning decision from val.** The split was made *before* cleaning precisely so corpus-wide statistics (dedup, length cutoffs, frequency counts) cannot leak. Val gets the same transformations, but only `train` may drive them.
- `data/train.csv` is the untouched original; all work reads/writes Parquet.
- **The Kaggle dataset stays private.** The corpus is CC BY-NC 4.0 and licensed for this competition only — never pass `--public`. `make_kaggle_dataset.py` deliberately omits `test.csv`: it is one click away on Kaggle via *+ Add Input → Competitions*, and a second copy in a user dataset is redistribution risk for no gain.

## Status

Preprocessing is **complete and verified**, and so is the training pipeline — smoke-tested end to end locally, including resume, warm-start and chunk disjointness.

Submissions 001 and 002 (both constant-answer probes) are on the board at 0.57429 and 0.56357; between them they settled the submission column and the BERTScore question above. Every submission is archived in `submissions/` with its local *and* LB score and a row in `submissions/submissions.md` — Phase 2 has to regenerate leaderboard outputs within tolerance, which is only checkable against the bytes actually uploaded.

Not yet done: a trained model on the board, and the decode sweep.

Long runs are chunked and resumable — `--max-hours` stops and saves before a platform kill, `--resume auto` continues the same run, `--init-adapter` + `--skip-rows` continues on unseen data in a new session or account. Chunks are disjoint slices of one fixed seed-42 shuffle, so an epoch can be assembled across sessions and accounts without any example being seen twice.

## Two conda envs — pick the right one

The system Python has none of the dependencies. `test_research` is the analysis env; `nascenia_train` is the training env (never install Unsloth or training pins into `test_research`, it would break its transformers 5.12.1).

```bash
"/g/miniconda3/envs/test_research/python.exe"  src/metric.py      # Bash tool, MSYS paths
G:\miniconda3\envs\nascenia_train\python.exe -u src\train_sft.py  # PowerShell, training
```

Python 3.11 · pandas 2.3.3 · pyarrow 24.0 · torch 2.9.1+cu128 · transformers 5.12.1 · peft 0.20.0. Local GPU: RTX 5060 Ti, **8 GB** VRAM, bf16-native, 23.7 GB RAM. Always `-u` on long runs — Kaggle buffers a cell until it ends.

## Commands

```bash
P="/g/miniconda3/envs/test_research/python.exe"     # analysis
T="/g/miniconda3/envs/nascenia_train/python.exe"    # training

# --- the closest thing to a test suite: run after touching metric.py ---
$P src/check_metric.py            # micro-cases + 50-row identity check
$P src/check_metric.py --full     # identity on all 600 val_dev rows; rewrites data/baselines.json

# --- one-time setup (already done; regenerate only if data/ is rebuilt) ---
$P src/eval_sets.py                                   # -> val_dev.parquet (600), val_test.parquet (2500)
$P src/export_sft.py --tokenizer unsloth/gemma-2-2b --max-seq-len 1024   # -> data/sft_*.jsonl

# --- train (local, one GPU) ---
$T -u src/train_sft.py --run-name E_full --lora-r 64 --max-rows 0 --resume auto
$T -u src/train_sft.py --run-name c2 --init-adapter runs/E_full/adapter --skip-rows 20000

# --- train (Kaggle, one arm per GPU, in parallel) ---
$T -u src/train_arms.py --max-rows 5000 --max-hours 10.5 \
     --arm A_gemma2:unsloth/gemma-2-2b:32 --arm B_titulm:hishab/titulm-gemma-2-2b-v1.1:32
$T src/train_arms.py --dry-run --arm A:unsloth/gemma-2-2b:32     # validate specs, no GPU

# --- generate + score (score_run.py appends to runs/results.csv) ---
$T -u src/generate.py --adapter runs/E_full/adapter --data data/val_dev.parquet \
     --limit 300 --max-new-tokens 768 --out runs/E_full/val_dev.parquet
$P src/score_run.py --pred runs/E_full/val_dev.parquet --tag E_full@greedy --subset-ok
$P src/score_run.py --pred ... --tag ... --fast          # lexical only: the ranking signal, ~50x cheaper

# --- submit ---
$P src/make_submission.py --pred runs/E_full/test_preds.parquet --out submissions/003_E_full.csv
$P src/baseline_submission.py --rank 0        # constant-answer probe; --rank -1 = worst of shortlist

# --- ship code to Kaggle (rebuild after ANY src/ or notebook change) ---
$P src/make_kaggle_dataset.py --user <kaggle-username>   # -> kaggle_upload.zip, upload PRIVATE
```

Analysis scripts belong in the session scratchpad, **not** in the repo — the user asked that preprocessing work not be written into the notebook after the initial split cells.

## How the training pipeline fits together

Data flows `train_final.parquet → export_sft.py → sft_train.jsonl → train_sft.py → runs/<name>/adapter → generate.py → *.parquet → score_run.py → runs/results.csv → make_submission.py → submissions/NNN.csv`. Five things about that chain are not obvious from any single file:

- **The SFT export is model-neutral; the chat template is applied at train time.** `export_sft.py` stores `{id, input, output}` and the length policy runs with the tokenizer passed in, so an A/B between two bases needs no re-export — but two bases with *different* vocabularies need different files (`--out-prefix`, which is why `train_arms.py` exports per arm).
- **Completion-only loss, built by hand.** `input_ids`/`labels` are assembled directly and labels are `-100` across the whole prompt span, rather than going through TRL's collator: the split point is then exact and verifiable as a token prefix. The patient question is ~30% of every sequence and the model is never asked to produce it.
- **Chunking is a slice of one fixed shuffle, not a fresh shuffle per call.** `build_dataset(..., skip_rows=N)` shuffles with seed 42 *then* takes `[skip_rows, skip_rows+max_rows)`. This is the correctness crux of training across sessions and accounts: chunk 1 takes `[0, N)`, chunk 2 takes `[N, 2N)` of the same permutation, so each example is seen once instead of re-reading a random third of the corpus. Verified: 3 chunks × 2,000 rows share 0 ids and cover 6,000 distinct ones.
- **Two resume modes that must not be mixed** (`train_sft.py` raises if both are passed):
  - `--resume auto` — HF `resume_from_checkpoint`. Restores optimizer, scheduler, step count *and data order*. For continuing an interrupted run on the **same** data and the same schedule. Safe on a clean directory: it just starts fresh, which is what makes one command safe to run twice.
  - `--init-adapter DIR` + `--skip-rows N` — LoRA warm-start. Fresh optimizer and schedule, ~100 MB to carry instead of full optimizer state. For continuing on **new** data in a new session or account. PEFT needs `is_trainable=True` here or it loads the adapter frozen and the run burns its budget updating nothing.
- **Version drift is absorbed, not fatal.** `filter_training_args()` drops `TrainingArguments` kwargs the installed transformers does not know (`group_by_length` is gone in 5.12 while `length_column_name` survives) and *reports* each one. `train_sft.py` runs against both local transformers 5.12 and whatever Kaggle ships.

`train_arms.py` is a thin launcher over `train_sft.py`: one child process per GPU with `CUDA_VISIBLE_DEVICES` pinned, per-arm SFT export done serially on CPU first, and a status table polled from each arm's `runs/<name>/progress.json`. The pin is load-bearing — HF `Trainer` wraps the model in `nn.DataParallel` whenever it sees more than one device, which breaks a 4-bit model pinned to `cuda:0`.

Gemma-2 requires `attn_implementation="eager"` (attention and final-logit soft-capping); anything else silently changes the model.

## Data files

| file | rows | role |
|---|---|---|
| `train.csv` | 108,954 | original, untouched |
| `test.csv` | 1,000 | `id,input` only — **off limits**, public/private split |
| `train_raw.parquet` | 108,954 | lossless Parquet copy of `train.csv` |
| `train_final.parquet` | 90,940 | **training set** |
| `val_final.parquet` | 10,127 | **validation set** |
| `removed_rows.parquet` | 7,887 | audit trail: `+drop_reason, step, split` |
| `val_dev.parquet` | 600 | sweep against this freely (decode configs, checkpoints) |
| `val_test.parquet` | 2,500 | disjoint from `val_dev`; **touch at most once per model**, for go/no-go |
| `sft_train.jsonl` / `sft_val.jsonl` | 90,508 / 10,080 | rendered training text, regenerated per tokenizer |
| `sft_dropped.parquet` | 432 | outputs too long to fit `max_seq_len` |
| `baselines.json` | — | floor table from `check_metric.py`. Its `best_constant_answer` (0.4433) is the **old 60-candidate** search; the current floor is **0.4574** from a 1,500-candidate search |

Schema is `id, input, output` (`input` = patient question, `output` = doctor answer). Row counts reconcile exactly: `108,954 = 90,940 + 10,127 + 7,887`, and every surviving id traces back to `train_raw`.

`val_dev` and `val_test` are seed-0 slices of `val_final`, deterministic so every experiment scores identical rows. The split exists because the decode sweep evaluates dozens of configs on `val_dev` and whatever wins is partly fitted to it; `val_test` stays clean for the number that drives a submission decision.

Intermediates (`*_split`, `*_clean`, `*_norm`, `*_dedup`, `*_dropped`) were deleted at the user's request. Re-running a stage means regenerating its input from `train_raw`. The SFT JSONL is *not* shipped to Kaggle — the notebook regenerates it there so the length policy runs with the tokenizer that run actually uses.

## Preprocessing pipeline as executed

1. **Split** — 10% val, seed 42, before any cleaning (notebook cells).
2. **Incomplete / mismatched pairs** — 5,754 removed. Truncation is judged only on *open-ended* text (no terminal `।?!.…`), then classified: dangling connective, unfulfilled list promise (`নিম্নলিখিত`/`নিচে` + a promise verb), dangling enumerator, unclosed bracket, boilerplate-only, too short.
3. **Unicode** — NFC; ZWNJ/ZWSP/BOM stripped. **ZWJ (U+200D) is deliberately preserved** — it is linguistically meaningful in Bengali (`র` + ZWJ + `্`). Note NFC *decomposes* য়/ড়/ঢ় (U+09DF/09DC/09DD are composition exclusions), which is why frequency counts fragment if you don't account for it.
4. **Duplicates** — 2,133 removed: 2,075 templates that announce a list/link and deliver nothing, 50 near-dupes within a split, 8 train→val leaks.
5. **Final tidy** — invisible chars, curly→straight quotes, exotic spaces, space-before-punctuation, punctuation runs capped at 3. Asserted to have changed no letter or digit.

## Deliberately NOT normalized

50% of the metric is literal token/LCS overlap against references drawn from this same corpus, so "standardizing" an artifact the references also contain trains the model *away* from the target. Do not fix these without the user asking:

- **Mixed digits** — 110k texts use Bengali `০-৯`, 4,845 ASCII, 3,965 both.
- **`...` vs `…`, em-dash** — authorial punctuation, not noise.
- **Danda vs period** — periods also appear in abbreviations and decimals; no safe blanket rule.
- **Brand-substitution corruption** — the source was built by find/replacing "ChatDoctor" → `নাসেনিয়া ডক`, which damaged ~242 places where a common noun belonged (*"Metformin is a Nascenia Doc"* should read *"a drug"*). The original noun is unrecoverable and the test references almost certainly carry the same damage. The brand string appears in 49,197 texts and is overwhelmingly legitimate.
- **197 val rows (1.95%)** still share a template answer with train — kept so val stays predictive of the leaderboard.
- **Formulaic openers/closers** (`হেলো,`, thanks-for-asking, see-a-doctor) — they are in the references, so the model should emit them. Stripping them would cost lexical overlap.

## Conventions that matter here

- **Every stage asserts its own integrity** before writing: ids unique and disjoint across splits, `kept + dropped == input`, text unmodified where it shouldn't be, output is NFC, ZWJ count preserved, transform is idempotent. The strongest one — strip all whitespace/punctuation and assert equality — proves a "tidy" step changed no letter or digit. Keep this pattern for new stages.
- **Every removal is written to a `_dropped` file with a `drop_reason`**, then folded into `removed_rows.parquet`. Nothing is discarded silently.
- **Validate a signal before trusting it.** Two plausible approaches failed here and were discarded: embedding cosine similarity cannot detect Q/A mismatch (a shuffled-pair control scored 0.837 vs 0.875 for real pairs), and it cannot detect near-duplicates either (89% of the corpus sits above 0.92). Char-level `difflib` overlap ≥0.90 is the reliable duplicate signal; embeddings serve only as a ≥0.97 candidate generator.
- Bengali regexes need care: several natural closers (`পরামর্শ দিচ্ছি`, `হতে পারে`, `দেওয়া হলো`) both end real sentences and precede lists — context decides, not the phrase. Avoid nested quantifiers; one such pattern caused catastrophic backtracking that hung for minutes.
- **A guard that cannot fail is not a guard.** `any(d.glob(p) for d in dirs)` is always `True` — `Path.glob` returns a generator and a generator object is truthy regardless of what it yields. That silently skipped the pre-flight canary and let an OOM config book a 10-hour Kaggle session. Any check that gates an expensive run should print what it decided (`print(f"resuming: {RESUMING}")`), so a wrong answer is visible in the log rather than inferred from the wreckage.
- **Everything expensive gets a cheap canary first.** `train_arms.py` at 16 steps exercises the same GPUs, batch size, tokenizers and multi-process launch as the real run, for five minutes. Every failure so far — OOM, the skipped guard, the backend fallback — would have surfaced there.

## Modeling notes

**Base model settled: `gemma-2-2b`, measured at 2,614,341,888 params** (386M under the cap), confirmed both from `config.json` and from the loaded weights.

Two earlier candidates are struck — **Qwen2.5-3B is 3.09B and Llama-3.2-3B is 3.21B, both over the cap** and disqualifying at Phase 2 verification. "3B" in a model name is marketing, not a parameter count. Qwen3-1.7B remains legal but gives up capacity for nothing.

Count with `count_inference_params()` in `src/train_sft.py`, not `sum(p.numel())`: bitsandbytes packs two nf4 values per uint8, so a naive sum under-reports a 4-bit model by half (it printed 1.61B for this 2.61B model).

**The memory ceiling is the loss, not the model.** Without a fused cross-entropy, HF materialises a `batch × seq × 256000 × 4` byte logits tensor per forward — 3.9 GB at batch 4 / seq 1024, which OOMs a 16 GB T4 while the 4-bit model itself uses ~2 GB. Batch 2 (2.0 GB) fits. `--use-liger 1` never builds the tensor and is what lets batch go higher; `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set at the top of `train_sft.py` because one real OOM had 1.18 GB stranded as reserved-but-unallocated around that giant block.

**Unsloth is not required after all, and does not install on Kaggle's current image.** The claim that Gemma-2 overflows to inf in plain fp16 on T4 did not hold on this stack: `--backend auto` fell through to `hf`, and with `--upcast-fp32` (automatic on fp16 hardware) it trained two hours at a finite, falling loss. Treat Unsloth as an optional speed path, not a correctness requirement — and never assume `--backend auto` got it, the run prints `backend=` on startup.

Gemma's tokenizer is **~2× more efficient on Bengali** than the alternatives — 0.543 tok/char vs Qwen 1.060 and Llama 1.203 — which halves both training and inference cost and is the main reason to prefer it beyond the cap.

Length budget with the **Gemma** tokenizer: per example median 572, p95 907, p99 1,249 → `max_seq_len=1024` covers ~97%, one epoch ≈ **54.9M tokens**. (The older figures here — median 1,101, p99 2,395, `max_seq_len=2048`, 107M tokens/epoch — were measured with the Qwen2.5 tokenizer and do not transfer.)

Because Token F1 and ROUGE-L are recall/precision balanced, generation length matters — a model that stops early or rambles loses on both. Tune `max_new_tokens` and repetition settings on `val_dev` against the **lexical** score (`score_run.py --fast`), never against loss, and never against the full composite (see the BERTScore finding above).

## Compute reality — measured 2026-08-05/06, and it reshaped the plan

| | tok/s | note |
|---|---|---|
| local RTX 5060 Ti, batch 1 | **921** | bf16-native, no session cap |
| Kaggle T4, one arm | **~110** | fp16, eager attention, 4-bit dequant, grad checkpointing |

**One local GPU is worth roughly eight Kaggle T4 arms.** Kaggle is a screening rig — parallel A/Bs at small row counts — not a training rig. The epoch runs locally.

Two corrections to earlier assumptions, both measured the hard way:

- **Raising the batch size does not help on T4.** The theory was that batch 1 leaves a 16 GB card latency-bound. It does not: at batch 4 the per-step time was unchanged, so the card is compute-bound (eager attention is mandatory for Gemma-2's soft-capping and has no flash path). Batch size on T4 is a *memory* dial, not a speed dial.
- **The second T4 is not free compute.** Running two arms in one session bought **+20% aggregate**, not +100% — per-arm throughput collapsed by ~1.7×. Kaggle gives 4 vCPUs and two length-grouped samplers plus collation plus dequant contend for them. Weigh that against the extra complexity before using `train_arms.py` with two arms.

Corollary for planning: size `MAX_ROWS` so the **cosine schedule completes** inside `--max-hours`. A run stopped at 45% freezes the LoRA with the LR still near 0.7× peak, which is worse than a completed shorter schedule. The exception is an A/B — if both arms stop at the same fraction the *comparison* is still valid even though neither absolute score is.

## Open items

- Whether more than ~1 epoch helps; decide from the per-chunk `val_dev` **lexical** curve, not from loss.
- Whether `--upcast-fp32 0` and `--max-seq-len 768` recover meaningful T4 throughput. Both are canary-testable in five minutes and neither has been tried.
- Whether `titulm-gemma-2-2b` beats vanilla `gemma-2-2b` on Bengali. Round 1 is in flight; within 0.005 lexical, take vanilla for the better-tested tooling path.

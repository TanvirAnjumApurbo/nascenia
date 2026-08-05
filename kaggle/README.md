# Runbook

**Submission budget: 5 per day**, confirmed on the live Kaggle page. Rulebook §8
says "5 during Phase 1"; the live page wins, and CLAUDE.md already says to
resolve rulebook conflicts that way. Worth one forum post asking organizers to
confirm on the record.

~5/day × 19 days is roughly 95 submissions, so the leaderboard is a usable
signal — but a disciplined one:

- **Choose on local `val_dev`. Confirm on the leaderboard.** The public LB is a
  fraction of 1,000 test rows and moves ±0.01 on noise. Private LB decides rank,
  and chasing public-LB wiggles across 95 submissions is how you overfit a few
  hundred rows.
- Submit the best *new* candidate each day, not five variants of the same one.
- Keep the ledger in `runs/results.csv` so local score and LB score stay paired.

---

## One-time setup (~15 min)

1. `python src/make_kaggle_dataset.py --user <your-kaggle-username>`
   → writes `G:\nascenia\kaggle_upload.zip` (102 MB).
2. kaggle.com/datasets → **New Dataset** → drag the zip → title
   *Nascenia Bengali Medical Dialogue (cleaned)* → **keep it Private** → Create.
   The slug must end up `nascenia-bengali-med` (or edit `DATASET` in the notebook).
3. Share the dataset with teammates: dataset → Settings → Collaborators.
4. Each account: kaggle.com/code → **New Notebook** → File → Import → pick
   `kaggle_train.ipynb` (it is inside the zip, and at `kaggle/kaggle_train.ipynb`).
5. In the notebook: **+ Add Input** → your dataset, **and** → Competitions → this
   competition (needed for `test.csv`). Settings → Accelerator → **GPU**.
   Prefer **L4** if offered: it is bf16-native, so the notebook skips the Unsloth
   install entirely and one whole class of dependency failure with it.

## Round 1 — three arms in parallel, tonight

Same notebook, three different `CONFIG` blocks. All three finish overnight.

| Where | `RUN_NAME` | `BASE` | rows | `LORA_R` |
|---|---|---|---|---|
| Kaggle acct 1 | `A_gemma2` | `unsloth/gemma-2-2b` | 20000 | 32 |
| Kaggle acct 2 | `B_titulm` | `hishab/titulm-gemma-2-2b-v1.1` | 20000 | 32 |
| Your 5060 Ti | `C_keeper` | `unsloth/gemma-2-2b` | 20000 | 64 |

Everything else stays at the defaults. `SKIP_ROWS=0`, `MAX_HOURS=10.5`,
`MAKE_SUBMISSION=False`.

Local arm C:

```bash
G:\miniconda3\envs\nascenia_train\python.exe -u src/train_sft.py \
  --run-name C_keeper --lora-r 64 --max-rows 20000 --max-hours 9 --resume auto
```

**Gate:** composite on `val_dev` must beat **0.4574** (the best single constant
answer). Below that the run is broken, not merely weak — check the loss mask
before anything else. Within 0.005 between A and B, take vanilla `gemma-2-2b`.

## Round 2 — first trained submission

Winner config, `MAKE_SUBMISSION=True`. `SUB_COLUMN="output"` is already settled —
submission 001 was accepted with it. Before you submit:

- Read the three sample generations the notebook prints. Boilerplate, register
  and length should look like the references.
- Expect the LB to read roughly **local + 0.117** — that offset comes from the
  constant-answer baseline (local 0.4574 → LB 0.57429). A trained model that
  lands *below* that line is worse than a fixed string and something is wrong.
- Add a row to `submissions/submissions.md` and archive the CSV there.

From this submission onward, **pin the seed and generation config** — Phase 2
requires the script to regenerate these outputs.

## Round 3 — finish the epoch, chunked

Continue the winner on data it has not seen. Each run prints its own
`next_skip_rows`; carry the adapter forward.

| chunk | `SKIP_ROWS` | `INIT_ADAPTER` |
|---|---|---|
| 2 | 20000 | chunk 1's adapter |
| 3 | 40000 | chunk 2's adapter |
| 4 | 60000 | chunk 3's adapter |

To move an adapter between accounts: Save Version → the notebook's Output tab →
Add as input to the next notebook, or download `out/adapter` (~100 MB) and
upload it as a small dataset.

Score every chunk on `val_dev`. **Stop when the gain per chunk falls below
~0.003** — that is more data no longer paying, and the remaining time is better
spent on decode tuning.

## Round 4 — decode tuning (local, no Kaggle quota)

Cheapest points on the board. Sweep on `val_dev`, confirm the single winner once
on `val_test`:

```bash
P=G:\miniconda3\envs\nascenia_train\python.exe
$P src/generate.py --adapter runs/<run>/adapter --data data/val_dev.parquet \
    --max-new-tokens 600 --out runs/<run>/d600.parquet
$P src/score_run.py --pred runs/<run>/d600.parquet --tag <run>@600
```

Vary, in this order of expected value:

1. `--max-new-tokens` 600 / 768 / 900 — chase `pred_words_mean` toward the
   reference mean of **102 words**. Both lexical metrics are F1s, so overshooting
   costs precision and undershooting costs recall.
2. `--mbr 8 --temperature 0.8` — minimum Bayes risk, scored with the
   competition's own lexical utility. The most reliable known gain for overlap
   metrics.
3. `--num-beams 4 --length-penalty 0.8/1.0/1.2`.
4. `--repetition-penalty 1.0/1.05/1.1`, `--no-repeat-ngram-size`.

Use `--fast` on the metric for wide grids (lexical half only, ranks within a
sweep), then re-score finalists on the full composite — the two halves disagree
about length and only the full metric settles it.

---

## Resuming

Two different things, do not mix them:

- **Session died / hit `MAX_HOURS`** → re-run the *identical* notebook.
  `RESUME="auto"` restores optimizer, scheduler, step count and data order.
  Safe on a clean directory too: it just starts fresh.
- **Continue on new data, new session or account** → `INIT_ADAPTER` +
  `SKIP_ROWS`. Fresh schedule, ~100 MB to carry instead of a full optimizer state.

Chunks are slices of one fixed seed-42 shuffle, so they never overlap —
verified: 3 chunks of 2000 rows share 0 ids and cover 6000 distinct ones.

Progress: a line every 2 minutes, and `runs/<name>/progress.json` with the same
numbers for polling from elsewhere.

```
[ 37.5%] step 3/8 | loss 1.2841 | 12000/20000 rows | 1.42h elapsed | ETA 2.31h
```

## Reference numbers

| | composite on `val_dev` | public LB |
|---|---|---|
| oracle (reference vs itself) | 1.0000 | — |
| **best single constant answer** (1500-candidate search) | **0.4574** ← below this is broken | **0.57429** |
| best constant from a 60-candidate pool | 0.4433 | — |
| random train answer | 0.4146 | — |
| copy the question back | 0.3987 | — |
| TF-IDF 1-NN retrieval | 0.0805 lexical — worse than a constant; retrieval is a dead end | — |

BERTScore floors near 0.68 locally for any fluent Bengali, so ~0.34 of the
composite is free and the competition is decided in a narrow band above it.

The single LB point gives an offset of **+0.1169**. The likely cause is that the
organisers' BERTScore encoder scores much higher than our layer-9 unrescaled
mBERT — implying theirs sits near 0.94, i.e. *saturated*, i.e. rank is decided
almost entirely by the lexical half. Read the calibration section of
`submissions/submissions.md` before deciding what to optimise.

Measured: Gemma-2-2B = **2,614,341,888** params (386 M under the cap). Local
throughput 921 tok/s at batch 1; batch 2 spills to host RAM and runs 3× slower.

## Troubleshooting

| Symptom | Cause |
|---|---|
| `loss = nan` | fp16 overflow on T4/P100. The notebook installs Unsloth for this; if it persists, switch the accelerator to L4. |
| generations all empty | tokenizer loaded from a half-written adapter dir. `generate.py` now checks and refuses; it loads the tokenizer from the base model. |
| OOM during generation | drop `--batch-size` to 4. Eager attention's fp32 softmax spikes at prefill and is O(batch × heads × seq²). |
| `TrainingArguments got an unexpected keyword` | version drift; `filter_training_args` drops and reports it. Not fatal. |
| notebook output > 20 GB | the last cell deletes checkpoints once a run finished, and keeps them when it stopped early because resume needs them. |

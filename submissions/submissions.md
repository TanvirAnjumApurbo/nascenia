# Submission ledger

Every Kaggle submission gets a row here **and** a numbered CSV in this folder.
Keeping the exact file that was uploaded is not bookkeeping for its own sake:
Phase 2 requires the inference script to regenerate the leaderboard outputs
within tolerance, and that check is only possible against the bytes that were
actually scored.

**Budget: 5 per day** (live Kaggle page; the rulebook's "5 during Phase 1" is
wrong). Phase 1 closes **2026-08-24 00:00 GMT+6**.

Rules for spending them:

- **Choose on local `val_dev`. Confirm on the leaderboard.** The public split is
  a fraction of 1,000 rows and moves ±0.01 on noise. Private LB decides rank.
- One best *new* candidate per day, not five variants of one.
- Never leave a day's quota unspent while an unanswered question is cheap to
  settle — the column name cost one submission and was worth it.

---

## Ledger

| # | date | file | what it is | local `val_dev` | local lexical | **public LB** |
|---|---|---|---|---|---|---|
| 001 | 2026-08-05 | `001_constant_baseline.csv` | best constant answer, 1000× | 0.4574 | 0.10204 | **0.57429** |
| 002 | 2026-08-05 | `002_calibration_probe.csv` | worst-of-shortlist constant | 0.4377 | 0.08936 | **0.56357** |

Daily usage — 2026-08-05: **2 / 5**.

---

## 001 — constant-answer baseline

`python src/baseline_submission.py` · best of 1,500 train answers under
`0.3·TokenF1 + 0.2·ROUGE-L`, scored on 300 `val_dev` rows. Answer text and
selection config are in `001_constant_baseline.json`.

**Settled: the submission column is `id,output`.** The file was accepted and
scored, so the rulebook's `id,doctor_response` (Evaluation section) is the wrong
one and the Dataset Description is right. `SUB_COLUMN="output"` everywhere.

Local breakdown:

| component | value | × weight | contribution |
|---|---|---|---|
| BERTScore F1 | 0.7108 | 0.5 | 0.35540 |
| Token F1 | 0.2392 | 0.3 | 0.07176 |
| ROUGE-L F1 | 0.1514 | 0.2 | 0.03028 |
| | | | **0.45743** |

89 predicted words against a 102-word reference mean.

---

## 002 — calibration probe

`python src/baseline_submission.py --rank -1`. Same search, but the *worst* of
the 30-candidate shortlist: still fluent, still in register, still the same
boilerplate — a materially poorer lexical match. Fluency held fixed, lexical
moved by a known amount.

| | 001 best | 002 worst | drop |
|---|---|---|---|
| BERTScore contribution (0.5×) | 0.35540 | 0.34832 | 0.00708 |
| lexical contribution | 0.10204 | 0.08936 | **0.01268** |
| local composite | 0.45743 | 0.43768 | 0.01975 |
| **public LB** | 0.57429 | 0.56357 | **0.01072** |

---

## Result: the organisers' BERTScore is saturated

**The leaderboard moved 0.01072. Our lexical half moved 0.01268. Our full
composite moved 0.01975.** The LB tracks the lexical drop, not the composite
drop — their BERTScore barely registered a difference that ours scored at 0.014.

Solving for it directly, on each submission independently:

```
implied BERTScore = (LB − our lexical) / 0.5
    001:  (0.57429 − 0.10204) / 0.5 = 0.9445
    002:  (0.56357 − 0.08936) / 0.5 = 0.9484
```

Two independent points landing within **0.004** of each other. That is not a
coincidence, and it is not something a sampling artifact would produce. Both
submissions were scored on the *same* public rows, so the difference between
them carries no split noise at all.

> **The effective leaderboard metric is `≈ 0.473 + 0.3·TokenF1 + 0.2·ROUGE-L`.**

Their encoder is almost certainly last-layer and unrescaled, where raw cosine
between any two fluent same-language texts sits very high. Ours is
`bert-base-multilingual-cased` layer 9, chosen to spread scores out — good for
diagnosis, wrong as a proxy for their number.

### What this changes

1. **Rank models on the lexical score, not the composite.** Our composite weights
   BERTScore at 0.5 where the leaderboard weights it at ~0. Two models that trade
   fluency against overlap will be ordered *wrongly* by the composite.
   `metric.py --fast` computes exactly the right thing and is ~50× cheaper.
2. **Length matching is first-order.** Both surviving metrics are F1s against a
   102-word reference mean; overshooting costs precision, undershooting costs
   recall, and nothing else absorbs the error any more.
3. **MBR with the competition's own lexical utility is now clearly the top decode
   move**, not one option among several.
4. **Expect compressed gains.** A model that improves lexical by 0.05 moves the
   LB by 0.05 — real, but small-looking next to a 0.574 floor. Do not read a
   +0.03 jump as a weak result.
5. **Headroom:** the ceiling is `0.473 + 0.5 = 0.973`, and the floor a fixed
   string already earns is 0.574. The entire competition lives in that band, and
   all of it is lexical.

Caveat worth keeping: saturation is established *in the fluent-Bengali-medical
regime*. Their BERTScore would still punish empty, English or degenerate output.
It discriminates — just not between reasonable candidates. Phase 2 also judges
medical accuracy by LLM, so do not let "maximise n-gram overlap" degrade clinical
content.

---

## Adding a row

```bash
# after a training run, from a predictions parquet
python src/make_submission.py --pred runs/<run>/test_preds.parquet \
    --out submissions/00N_<name>.csv
```

Then fill in the table above **including the local `val_dev` score**, and update
`runs/results.csv` so the local and LB numbers stay paired. A submission whose
local score was never recorded is a data point that cannot be used.

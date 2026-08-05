"""Self-test for `metric.py` plus the baseline floor it has to beat.

Run after any change to metric.py. Writes `data/baselines.json`.

The baseline table is the useful part. It says what a submission scores *without
a model*, which is the only way to read a leaderboard number honestly: BERTScore
never falls below ~0.68 on fluent Bengali, so roughly 0.34 of the composite is
free and the interesting band is narrow.

    python src/check_metric.py            # 50-row identity check
    python src/check_metric.py --full     # identity on all 600 val_dev rows
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from metric import BERTScorer, composite, get_scorer, rouge_l_f1, token_f1

ROOT = Path(os.environ.get("NASCENIA_ROOT", Path(__file__).resolve().parents[1]))
DATA = ROOT / "data"


def micro_cases() -> None:
    """Cases where the right answer is checkable by hand."""
    assert token_f1("a b c", "a b c") == 1.0
    assert rouge_l_f1("a b c", "a b c") == 1.0
    assert token_f1("", "a b") == 0.0 and rouge_l_f1("", "a b") == 0.0
    assert token_f1("x y", "a b") == 0.0 and rouge_l_f1("x y", "a b") == 0.0
    # bag-of-tokens ignores order, LCS does not
    assert token_f1("c b a", "a b c") == 1.0
    assert abs(rouge_l_f1("c b a", "a b c") - 1 / 3) < 1e-9
    # a repeated token must not match a single occurrence twice
    assert abs(token_f1("a a b", "a b") - 0.8) < 1e-9
    # set semantics collapse the repeat
    assert token_f1("a a b", "a b", multiset=False) == 1.0
    # Bengali survives tokenisation intact (rouge_score's tokeniser would not)
    bn = "হেলো, নাসেনিয়া ডকে আপনাকে স্বাগতম।"
    assert token_f1(bn, bn) == 1.0 and rouge_l_f1(bn, bn) == 1.0
    print("micro cases OK")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="identity check on all rows")
    ap.add_argument("--out", default=str(DATA / "baselines.json"))
    args = ap.parse_args()

    micro_cases()

    dev = pd.read_parquet(DATA / "val_dev.parquet")
    refs = dev["output"].tolist()

    n_ident = len(refs) if args.full else 50
    ident = composite(refs[:n_ident], refs[:n_ident])
    print(f"identity({n_ident} rows): score={ident['score']:.6f}")
    assert abs(ident["score"] - 1.0) < 1e-4, f"identity must be 1.0, got {ident}"

    # How much of the corpus does mBERT's 512-token window actually see? The
    # organisers' scorer truncates the same way, but if it bites often we would
    # be blind to the tail of long answers.
    scorer: BERTScorer = get_scorer("bert-base-multilingual-cased", 9)
    lens = [len(scorer.tok(r)["input_ids"]) for r in refs]
    over = sum(1 for n in lens if n > 512)
    print(
        f"mBERT wordpieces per reference: median={int(np.median(lens))} "
        f"p95={int(np.percentile(lens, 95))} max={max(lens)}  "
        f">512 (truncated): {over}/{len(lens)} ({100 * over / len(lens):.1f}%)"
    )

    train = pd.read_parquet(DATA / "train_final.parquet")
    pool = train["output"].sample(60, random_state=7).tolist()
    best_const = max(
        pool,
        key=lambda c: np.mean(
            [0.3 * token_f1(c, r) + 0.2 * rouge_l_f1(c, r) for r in refs[:120]]
        ),
    )

    rows = {
        "oracle_exact_reference": refs,
        "best_constant_answer": [best_const] * len(refs),
        "random_train_answer": train["output"].sample(len(refs), random_state=99).tolist(),
        "copy_patient_question": dev["input"].tolist(),
    }

    print(f"\n{'baseline':26s} {'composite':>10s} {'BERTScore':>10s} {'TokenF1':>8s} {'ROUGE-L':>8s} {'words':>7s}")
    results = {}
    for name, preds in rows.items():
        r = composite(preds, refs)
        results[name] = r
        print(
            f"{name:26s} {r['score']:10.4f} {r['bertscore_f1']:10.4f} "
            f"{r['token_f1']:8.4f} {r['rouge_l_f1']:8.4f} {r['pred_words_mean']:7.1f}"
        )

    payload = {
        "eval_set": "val_dev",
        "n": len(refs),
        "bert_model": "bert-base-multilingual-cased@L9",
        "mbert_truncated_frac": over / len(lens),
        "baselines": {k: {m: v[m] for m in ("score", "bertscore_f1", "token_f1", "rouge_l_f1", "pred_words_mean")} for k, v in results.items()},
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False), "utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

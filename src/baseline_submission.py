"""Constant-answer submission: the cheapest possible leaderboard calibration.

Answers every test row with one fixed doctor response chosen from the training
set. Useless as a model, valuable as an instrument:

  * It settled the submission format. `id,output` was accepted and scored, so
    the rulebook's `id,doctor_response` is the wrong half of its contradiction.
  * It anchors the local-to-leaderboard offset at a point whose local score we
    know exactly, so every later model's LB number can be read as a delta rather
    than an absolute. Measured: local 0.4574 -> LB 0.57429, a gap of +0.1169.
  * It measures the free BERTScore floor on the *actual* test distribution.
    Roughly 0.34 of the composite is available to any fluent Bengali text, and
    knowing where that floor sits tells us how much of a trained model's score is
    real signal.

`--rank` exists to settle where that +0.1169 comes from. Rank 0 is the best
candidate; `--rank -1` is the worst of the shortlist -- still fluent, still in
register, but a much poorer lexical match. Submitting one of each holds fluency
fixed and moves only the lexical half, which separates "their BERTScore is
higher than ours" from "the test references match this answer better". See the
calibration section of `submissions/submissions.md`.

Selection uses `train_final.parquet` for candidates and `val_dev.parquet` for
scoring. `test.csv` is read only for its `id` column -- no statistic, threshold
or decision is derived from it.

    python src/baseline_submission.py
    python src/baseline_submission.py --column doctor_response
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("NASCENIA_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from metric import composite, rouge_l_f1, token_f1  # noqa: E402


def pick_constant(candidates: list[str], refs: list[str], shortlist: int = 30,
                  rank: int = 0) -> tuple[str, list]:
    """Rank-th best single answer under the task's lexical utility, in two passes.

    Token F1 is a couple of Counter operations; ROUGE-L is an O(n*m) DP per pair.
    Scoring every candidate on both would be hours, so Token F1 ranks the field
    and only the shortlist pays for the LCS.
    """
    coarse = np.array([np.mean([token_f1(c, r) for r in refs]) for c in candidates])
    top = np.argsort(-coarse)[:shortlist]

    scored = []
    for i in top:
        c = candidates[i]
        lex = np.mean([0.3 * token_f1(c, r) + 0.2 * rouge_l_f1(c, r) for r in refs])
        scored.append((float(lex), int(i)))
    scored.sort(reverse=True)
    return candidates[scored[rank][1]], scored


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pool", type=int, default=1500, help="train answers to search")
    ap.add_argument("--refs", type=int, default=300, help="val_dev rows to score against")
    ap.add_argument("--column", default="output", help="settled: 'output'")
    ap.add_argument("--rank", type=int, default=0,
                    help="0 = best candidate, -1 = worst of the shortlist "
                         "(the calibration probe -- see module docstring)")
    ap.add_argument("--out", default=str(ROOT / "submissions" / "constant_baseline.csv"))
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)

    train = pd.read_parquet(ROOT / "data" / "train_final.parquet")
    dev = pd.read_parquet(ROOT / "data" / "val_dev.parquet")
    refs = dev["output"].tolist()[: args.refs]
    pool = train["output"].sample(args.pool, random_state=args.seed).tolist()
    print(f"searching {len(pool)} candidates against {len(refs)} val_dev refs")

    best, scored = pick_constant(pool, refs, rank=args.rank)
    print(f"\nshortlist lexical utility: best {scored[0][0]:.4f}, "
          f"runner-up {scored[1][0]:.4f}, worst {scored[-1][0]:.4f}")
    print(f"taking rank {args.rank} -> {scored[args.rank][0]:.4f}")

    full = composite([best] * len(refs), refs)
    print(f"\nfull composite on {len(refs)} val_dev rows: {full['score']:.4f}")
    print(f"  BERTScore {full['bertscore_f1']:.4f} | Token F1 {full['token_f1']:.4f} | "
          f"ROUGE-L {full['rouge_l_f1']:.4f} | {full['pred_words_mean']:.0f} words")
    print(f"  (references average {full['ref_words_mean']:.0f} words)")

    test = pd.read_csv(ROOT / "data" / "test.csv")
    sub = pd.DataFrame({"id": test["id"], args.column: best})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out, index=False, encoding="utf-8")

    back = pd.read_csv(out)
    assert len(back) == len(test), f"round-trip {len(back)} != {len(test)}"
    assert (back["id"].values == test["id"].values).all(), "round-trip reordered ids"
    assert (back[args.column].astype(str) == best).all(), "round-trip altered text"

    (out.with_suffix(".json")).write_text(json.dumps({
        "constant_answer": best, "val_dev_composite": full["score"],
        "bertscore_f1": full["bertscore_f1"], "token_f1": full["token_f1"],
        "rouge_l_f1": full["rouge_l_f1"],
        "val_dev_rows": len(refs), "pool": args.pool, "seed": args.seed,
        "rank": args.rank, "column": args.column,
    }, indent=2, ensure_ascii=False), "utf-8")

    print(f"\nwrote {out}  ({len(sub)} rows, columns {list(sub.columns)}, round-trip verified)")
    print(f"\n--- the answer being submitted ---\n{best[:700]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Score a predictions file and append it to the running results table.

Every experiment -- checkpoint, decode config, base model -- lands one row in
`runs/results.csv`, so the comparison table builds itself instead of living in
scrollback. With 5 submissions and a leaderboard that is not a tuning signal,
this table is what decisions get made from.

    python src/score_run.py --pred runs/main/val_dev_greedy.parquet \
        --ref data/val_dev.parquet --tag main@greedy
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from metric import align, composite

ROOT = Path(os.environ.get("NASCENIA_ROOT", Path(__file__).resolve().parents[1]))
RESULTS = ROOT / "runs" / "results.csv"

FIELDS = ["utc", "tag", "eval_set", "n", "score", "bertscore_f1", "token_f1",
          "rouge_l_f1", "pred_words_mean", "ref_words_mean", "fast", "notes"]


def _load(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--ref", default=str(ROOT / "data" / "val_dev.parquet"))
    ap.add_argument("--tag", required=True, help="short label, e.g. main@ckpt2000@beam4")
    ap.add_argument("--fast", action="store_true", help="lexical only (sweeps)")
    ap.add_argument("--subset-ok", action="store_true",
                    help="score only the predicted ids (spot checks on a --limit run)")
    ap.add_argument("--notes", default="")
    ap.add_argument("--results", default=str(RESULTS))
    args = ap.parse_args(argv)

    ref_path = Path(args.ref)
    preds, refs = align(_load(Path(args.pred)), _load(ref_path), subset_ok=args.subset_ok)
    res = composite(preds, refs, fast=args.fast)

    row = {
        "utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "tag": args.tag,
        "eval_set": ref_path.stem,
        "n": res["n"],
        **{k: round(res[k], 5) for k in
           ("score", "bertscore_f1", "token_f1", "rouge_l_f1", "pred_words_mean", "ref_words_mean")},
        "fast": int(args.fast),
        "notes": args.notes,
    }

    out = Path(args.results)
    out.parent.mkdir(parents=True, exist_ok=True)
    new_file = not out.exists()
    with out.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        w.writerow(row)

    print(json.dumps(row, indent=2, ensure_ascii=False))
    if args.fast:
        print("NOTE: --fast -> `score` is the lexical half only (max 0.5), "
              "comparable within a sweep but not to a full score.")

    hist = pd.read_csv(out)
    same = hist[(hist["eval_set"] == row["eval_set"]) & (hist["fast"] == row["fast"])]
    print(f"\ntop 10 on {row['eval_set']}:")
    print(same.nlargest(10, "score")[["tag", "score", "bertscore_f1", "token_f1",
                                      "rouge_l_f1", "pred_words_mean"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

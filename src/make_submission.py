"""Turn a predictions file into a validated `submission.csv`.

    python src/make_submission.py --pred runs/main/test_preds.parquet
    python src/make_submission.py --pred runs/main/test_preds.parquet --column doctor_response

The column is **`id,output`**, settled empirically: submission 001 was accepted
and scored with it, so the rulebook's `id,doctor_response` (Evaluation section)
is the wrong half of its own contradiction. `--column` stays configurable only in
case the organisers change the sample file mid-competition.

Every file written here belongs in `submissions/` with a number, and its row in
`submissions/submissions.md`. Phase 2 has to regenerate the leaderboard outputs
within tolerance, which is only checkable against the bytes actually uploaded.

Nothing here touches `data/test.csv` beyond reading its `id` column to check
coverage -- no statistic, threshold, or decision is ever derived from it.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(os.environ.get("NASCENIA_ROOT", Path(__file__).resolve().parents[1]))


def _load(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def validate(sub: pd.DataFrame, test_ids: pd.Series, column: str) -> list[str]:
    """Every way this file can be silently wrong."""
    problems = []
    if list(sub.columns) != ["id", column]:
        problems.append(f"columns are {list(sub.columns)}, expected ['id', '{column}']")
    if len(sub) != len(test_ids):
        problems.append(f"{len(sub)} rows, expected {len(test_ids)}")
    if sub["id"].duplicated().any():
        problems.append(f"{int(sub['id'].duplicated().sum())} duplicate ids")

    missing = set(test_ids) - set(sub["id"])
    extra = set(sub["id"]) - set(test_ids)
    if missing:
        problems.append(f"{len(missing)} test ids missing, e.g. {sorted(missing)[:5]}")
    if extra:
        problems.append(f"{len(extra)} ids not in test.csv, e.g. {sorted(extra)[:5]}")

    text = sub[column]
    if text.isna().any():
        problems.append(f"{int(text.isna().sum())} NaN outputs")
    else:
        blank = int((text.astype(str).str.strip().str.len() == 0).sum())
        if blank:
            problems.append(f"{blank} empty outputs")
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="output of generate.py on test.csv")
    ap.add_argument("--test", default=str(ROOT / "data" / "test.csv"))
    ap.add_argument("--out", default=str(ROOT / "submissions" / "submission.csv"))
    ap.add_argument("--column", default="output",
                    help="settled: 'output'. Submission 001 was accepted with it.")
    ap.add_argument("--show", type=int, default=5, help="predictions to print for eyeballing")
    args = ap.parse_args(argv)

    preds = _load(Path(args.pred))
    test = pd.read_csv(args.test)

    src_col = next((c for c in ("output", "doctor_response", "pred") if c in preds), None)
    if src_col is None:
        raise SystemExit(f"no prediction column in {list(preds.columns)}")

    # Order rows as test.csv does; a left join also surfaces any missing id.
    sub = test[["id"]].merge(
        preds[["id", src_col]].rename(columns={src_col: args.column}), on="id", how="left"
    )

    problems = validate(sub, test["id"], args.column)
    if problems:
        print("VALIDATION FAILED:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out, index=False, encoding="utf-8")

    # Round-trip: what Kaggle parses must be what we wrote. Outputs contain commas
    # and newlines, so this catches a quoting failure before it costs a submission.
    back = pd.read_csv(out)
    assert len(back) == len(sub), f"round-trip row count {len(back)} != {len(sub)}"
    assert (back["id"].values == sub["id"].values).all(), "round-trip reordered ids"
    assert (back[args.column].astype(str).values == sub[args.column].astype(str).values).all(), \
        "round-trip altered text -- CSV quoting problem"

    words = sub[args.column].str.split().str.len()
    print(f"wrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    print(f"  {len(sub)} rows, columns {list(sub.columns)}")
    print(f"  words per response: mean={words.mean():.1f} median={words.median():.0f} "
          f"min={words.min()} max={words.max()}")
    print(f"  (train references average 102 words -- a large gap either way costs Token F1)")
    print("  round-trip verified")

    if args.show:
        print(f"\n--- {args.show} predictions, read these before submitting ---")
        for i in range(min(args.show, len(sub))):
            print(f"\n[{sub['id'][i]}] Q: {test['input'][i][:160]}")
            print(f"      A: {sub[args.column][i][:320]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

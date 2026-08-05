"""Carve two fixed evaluation slices out of `val_final.parquet`.

    val_dev   600 rows -- swept against freely (decode configs, checkpoints)
    val_test  2500 rows -- disjoint from val_dev, touched at most once per model
                           for a go/no-go decision

Splitting the two matters because Step 4 evaluates dozens of decode configs on the
same rows; whatever wins is partly fitted to them. val_test stays clean so the
number we carry into a submission decision is honest.

Both are deterministic (seed 0) so every experiment scores the same rows, and the
remaining ~7k val rows stay untouched as reserve.

    python src/eval_sets.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

ROOT = Path(os.environ.get("NASCENIA_ROOT", Path(__file__).resolve().parents[1]))
DATA = ROOT / "data"

SEED = 0
N_DEV = 600
N_TEST = 2500


def main() -> int:
    val = pd.read_parquet(DATA / "val_final.parquet")
    train = pd.read_parquet(DATA / "train_final.parquet")

    shuffled = val.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    dev = shuffled.iloc[:N_DEV].reset_index(drop=True)
    test = shuffled.iloc[N_DEV : N_DEV + N_TEST].reset_index(drop=True)

    dev_ids, test_ids, train_ids = set(dev["id"]), set(test["id"]), set(train["id"])
    assert len(dev) == N_DEV and len(test) == N_TEST
    assert not (dev_ids & test_ids), "val_dev and val_test overlap"
    assert not (dev_ids & train_ids) and not (test_ids & train_ids), "train leak"
    assert dev["output"].str.strip().str.len().gt(0).all()
    assert test["output"].str.strip().str.len().gt(0).all()

    dev.to_parquet(DATA / "val_dev.parquet", index=False)
    test.to_parquet(DATA / "val_test.parquet", index=False)

    for name, df in (("val_dev", dev), ("val_test", test)):
        w = df["output"].str.split().str.len()
        print(
            f"{name:9s} n={len(df):5d}  output words mean={w.mean():6.1f} "
            f"median={w.median():5.0f}  ids {df['id'].min()}..{df['id'].max()}"
        )
    print(f"reserve   n={len(val) - N_DEV - N_TEST:5d} val rows untouched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

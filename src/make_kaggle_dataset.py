"""Assemble the folder to upload to Kaggle as a PRIVATE dataset.

Ships the cleaned parquet splits plus `src/`, and lets the notebook regenerate
the SFT JSONL on Kaggle. That keeps the parquet as the single source of truth,
makes the upload ~105 MB instead of ~285 MB, and -- the real reason -- runs the
length policy with whichever tokenizer that run actually uses, so an A/B against
a model with a different tokenizer stays correct.

    python src/make_kaggle_dataset.py --user <your-kaggle-username>

Then either drag `kaggle_upload/` into kaggle.com/datasets/new, or:

    kaggle datasets create -p kaggle_upload --dir-mode zip

**Keep it private.** The dataset is CC BY-NC 4.0 and licensed for this
competition only; `kaggle datasets create` defaults to private, do not pass
`--public`.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

ROOT = Path(os.environ.get("NASCENIA_ROOT", Path(__file__).resolve().parents[1]))

DATA_FILES = [
    "train_final.parquet",
    "val_final.parquet",
    "val_dev.parquet",
    "val_test.parquet",
    # test.csv is deliberately NOT here. It is competition data, already one click
    # away on Kaggle via "+ Add Input -> Competitions"; a second copy living in a
    # user dataset is redistribution risk for no convenience gain.
]
SRC_FILES = [
    "prompt.py",
    "metric.py",
    "export_sft.py",
    "train_sft.py",
    "train_arms.py",
    "generate.py",
    "score_run.py",
    "make_submission.py",
]
NOTEBOOKS = ["kaggle_train.ipynb"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default="YOUR_KAGGLE_USERNAME")
    ap.add_argument("--slug", default="nascenia-bengali-med")
    ap.add_argument("--out", default=str(ROOT / "kaggle_upload"))
    ap.add_argument("--zip", default=None, help="default: <out>.zip")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    (out / "data").mkdir(parents=True)
    (out / "src").mkdir(parents=True)

    total = 0
    for name in DATA_FILES:
        src = ROOT / "data" / name
        if not src.exists():
            raise SystemExit(f"missing {src} -- run eval_sets.py first")
        shutil.copy2(src, out / "data" / name)
        total += src.stat().st_size
    for name in SRC_FILES:
        src = ROOT / "src" / name
        if not src.exists():
            raise SystemExit(f"missing {src}")
        shutil.copy2(src, out / "src" / name)
        total += src.stat().st_size
    for name in NOTEBOOKS:
        src = ROOT / "kaggle" / name
        if not src.exists():
            raise SystemExit(f"missing {src}")
        shutil.copy2(src, out / name)
        total += src.stat().st_size

    (out / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": "Nascenia Bengali Medical Dialogue (cleaned)",
                "id": f"{args.user}/{args.slug}",
                "licenses": [{"name": "other"}],
            },
            indent=2,
        ),
        "utf-8",
    )

    print(f"{out}  ({total / 1e6:.0f} MB)")
    for p in sorted(out.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(out).as_posix():40s} {p.stat().st_size / 1e6:8.1f} MB")

    # One zip, because kaggle.com/datasets/new takes a zip by drag-and-drop and
    # expands it server-side. Uploading 6 loose files through the browser is the
    # step most likely to be done half-way.
    zip_path = Path(args.zip) if args.zip else out.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    shutil.make_archive(str(zip_path.with_suffix("")), "zip", root_dir=out)
    print(f"\n>>> UPLOAD THIS: {zip_path}  ({zip_path.stat().st_size / 1e6:.0f} MB)")
    print(
        "    kaggle.com/datasets  ->  New Dataset  ->  drag the zip  ->  keep it PRIVATE\n"
        "    (competition licence forbids redistribution; private is the default)\n"
        f"    Notebook then reads /kaggle/input/{args.slug}/"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Export the cleaned splits to supervised-fine-tuning JSONL.

Records stay model-neutral -- `{id, input, output}`, no chat template baked in --
so switching base models means re-running this, not redesigning the format. The
chat template is applied at train time via `prompt.py`.

Length policy (the one thing here that does need a tokenizer):

  * `--max-seq-len 1024` covers ~96.6% of pairs untouched.
  * If the doctor answer alone exceeds `--max-output-tokens 900` the row is
    DROPPED (0.43% of train). Truncating a target teaches the model to stop
    mid-sentence, which costs far more than the row is worth.
  * Otherwise the patient question is middle-truncated to fit. **The doctor
    answer is never modified** -- asserted below.

    python src/export_sft.py
    python src/export_sft.py --tokenizer hishab/titulm-gemma-2-2b-v1.1
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from prompt import TEMPLATE_OVERHEAD_TOKENS, render_example, truncate_input_tokens

ROOT = Path(os.environ.get("NASCENIA_ROOT", Path(__file__).resolve().parents[1]))
DATA = ROOT / "data"


def encode_lengths(tok, texts: list[str], batch: int = 1000) -> np.ndarray:
    lens = []
    for i in range(0, len(texts), batch):
        lens.extend(len(x) for x in tok(texts[i : i + batch], add_special_tokens=False)["input_ids"])
    return np.array(lens)


def process(df: pd.DataFrame, tok, max_seq_len: int, max_out: int, split: str):
    n_in = len(df)
    len_out = encode_lengths(tok, df["output"].tolist())
    len_inp = encode_lengths(tok, df["input"].tolist())

    too_long = len_out > max_out
    dropped = df[too_long].copy()
    dropped["drop_reason"] = "output_too_long"
    dropped["step"] = "sft_export"
    dropped["split"] = split
    dropped["n_tok_output"] = len_out[too_long]

    keep = df[~too_long].reset_index(drop=True)
    k_out, k_inp = len_out[~too_long], len_inp[~too_long]

    budget = max_seq_len - TEMPLATE_OVERHEAD_TOKENS - k_out
    needs_trunc = k_inp > budget

    inputs = keep["input"].tolist()
    n_trunc = 0
    for i in np.flatnonzero(needs_trunc):
        inputs[i], did = truncate_input_tokens(inputs[i], tok, int(budget[i]))
        n_trunc += int(did)
    keep["input"] = inputs

    assert len(keep) + len(dropped) == n_in, "rows lost"
    assert (keep["output"].values == df[~too_long]["output"].values).all(), "output modified"

    final_len = encode_lengths(tok, [render_example(i, o) for i, o in zip(keep["input"], keep["output"])])
    over = final_len > max_seq_len
    assert not over.any(), f"{over.sum()} rows still over {max_seq_len} after truncation"

    print(
        f"{split:5s}: {n_in:6d} in -> {len(keep):6d} kept  "
        f"(dropped {len(dropped):4d} output_too_long, truncated input on {n_trunc:4d})\n"
        f"       rendered tokens: mean={final_len.mean():6.1f} median={np.median(final_len):5.0f} "
        f"p95={np.percentile(final_len, 95):5.0f} max={final_len.max():5.0f}  "
        f"total={final_len.sum() / 1e6:.1f}M tokens/epoch"
    )
    return keep[["id", "input", "output"]], dropped, final_len


def write_jsonl(df: pd.DataFrame, path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in df.itertuples(index=False):
            fh.write(json.dumps({"id": int(row.id), "input": row.input, "output": row.output}, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="unsloth/gemma-2-2b")
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--max-output-tokens", type=int, default=900)
    ap.add_argument("--out-prefix", default="sft")
    args = ap.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    print(f"tokenizer: {args.tokenizer} (vocab {len(tok)})  max_seq_len={args.max_seq_len}\n")

    train = pd.read_parquet(DATA / "train_final.parquet")
    val = pd.read_parquet(DATA / "val_final.parquet")

    tr_keep, tr_drop, tr_len = process(train, tok, args.max_seq_len, args.max_output_tokens, "train")
    va_keep, va_drop, _ = process(val, tok, args.max_seq_len, args.max_output_tokens, "val")

    assert not (set(tr_keep["id"]) & set(va_keep["id"])), "train/val id overlap"

    train_path = DATA / f"{args.out_prefix}_train.jsonl"
    val_path = DATA / f"{args.out_prefix}_val.jsonl"
    write_jsonl(tr_keep, train_path)
    write_jsonl(va_keep, val_path)

    dropped = pd.concat([tr_drop, va_drop], ignore_index=True)
    if len(dropped):
        dropped.to_parquet(DATA / f"{args.out_prefix}_dropped.parquet", index=False)

    # Round-trip: the file on disk must reproduce the frame we asserted on.
    back = pd.read_json(train_path, lines=True)
    assert len(back) == len(tr_keep) and (back["output"].values == tr_keep["output"].values).all()

    meta = {
        "tokenizer": args.tokenizer,
        "max_seq_len": args.max_seq_len,
        "max_output_tokens": args.max_output_tokens,
        "train_rows": len(tr_keep),
        "val_rows": len(va_keep),
        "dropped_rows": len(dropped),
        "train_tokens_per_epoch": int(tr_len.sum()),
    }
    (DATA / f"{args.out_prefix}_meta.json").write_text(json.dumps(meta, indent=2), "utf-8")
    print(f"\nwrote {train_path.name}, {val_path.name}, {args.out_prefix}_meta.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

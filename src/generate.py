"""Batched generation for evaluation and for the final submission.

One code path for both so what we tune on `val_dev` is exactly what produces
`submission.csv`.

    # score a checkpoint on val_dev
    python src/generate.py --adapter runs/main/adapter --data data/val_dev.parquet \
        --out runs/main/val_dev_greedy.parquet

    # beam search
    python src/generate.py --adapter runs/main/adapter --data data/val_dev.parquet \
        --num-beams 4 --length-penalty 1.0 --out preds.parquet

    # minimum Bayes risk over 8 samples
    python src/generate.py --adapter runs/main/adapter --data data/val_dev.parquet \
        --mbr 8 --temperature 0.8 --out preds.parquet

Details that matter:

* **Left padding.** Decoder-only batched generation with right padding puts pad
  tokens between the prompt and the first generated token, and the output is
  quietly garbage.
* **Stop on `<end_of_turn>` as well as `<eos>`.** Gemma's turn terminator is not
  the eos token; without it generation runs to `max_new_tokens` and the trailing
  text costs BERTScore precision.
* Prompts are sorted by length before batching and restored afterwards, which
  removes most padding waste. Order out always matches order in.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("NASCENIA_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prompt import STOP_TOKEN_IDS, render_prompt  # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base-model", default="unsloth/gemma-2-2b")
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir; omit for the raw base")
    ap.add_argument("--tokenizer", default=None, help="defaults to --base-model; see check_tokenizer")
    ap.add_argument("--data", required=True, help="parquet/csv with id + input")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--load-in-4bit", action="store_true")

    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--min-new-tokens", type=int, default=0)
    ap.add_argument("--num-beams", type=int, default=1)
    ap.add_argument("--length-penalty", type=float, default=1.0)
    ap.add_argument("--repetition-penalty", type=float, default=1.0)
    ap.add_argument("--no-repeat-ngram-size", type=int, default=0)
    ap.add_argument("--do-sample", action="store_true")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--mbr", type=int, default=0,
                    help="N>1: sample N candidates per input and keep the consensus one")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args(argv)


def check_tokenizer(tok) -> None:
    """Fail fast on a tokenizer that cannot represent Bengali.

    An interrupted `save_pretrained` can leave a directory with only
    `tokenizer_config.json`, and `AutoTokenizer` will happily load that as a
    vocab-size-7 tokenizer rather than raising. Every prompt then collapses to a
    handful of garbage tokens, the model answers with an immediate stop token,
    and the run produces 1000 empty rows that look like a modelling failure. Two
    seconds of checking here beats debugging that.
    """
    if len(tok) < 100_000:
        raise SystemExit(f"tokenizer has only {len(tok)} tokens -- wrong or truncated")
    probe = "হেলো, আমার মাথা ব্যথা করছে।"
    ids = tok(probe, add_special_tokens=False)["input_ids"]
    if len(ids) < 5 or tok.decode(ids) != probe:
        raise SystemExit(f"tokenizer fails to round-trip Bengali: {probe!r} -> {ids}")


def load_for_inference(base_model: str, adapter: str | None, load_in_4bit: bool,
                       tokenizer: str | None = None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quant = None
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype,
        )

    # From the base model, not the adapter: we never train embeddings, so the
    # vocabulary is unchanged, and the adapter's copy is the one that can be
    # half-written.
    tok = AutoTokenizer.from_pretrained(tokenizer or base_model)
    check_tokenizer(tok)
    tok.padding_side = "left"  # see module docstring
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=dtype,
        quantization_config=quant,
        attn_implementation="eager",  # Gemma-2 soft-capping
        device_map={"": 0},
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    model.generation_config.pad_token_id = tok.pad_token_id
    return model, tok, dtype


def build_gen_kwargs(args) -> dict:
    kw = {
        "max_new_tokens": args.max_new_tokens,
        "eos_token_id": STOP_TOKEN_IDS,
        "repetition_penalty": args.repetition_penalty,
    }
    if args.min_new_tokens:
        kw["min_new_tokens"] = args.min_new_tokens
    if args.no_repeat_ngram_size:
        kw["no_repeat_ngram_size"] = args.no_repeat_ngram_size

    if args.mbr > 1:
        kw.update(do_sample=True, temperature=args.temperature,
                  top_p=args.top_p, num_return_sequences=args.mbr)
        if args.top_k:
            kw["top_k"] = args.top_k
    elif args.do_sample:
        kw.update(do_sample=True, temperature=args.temperature, top_p=args.top_p)
        if args.top_k:
            kw["top_k"] = args.top_k
    elif args.num_beams > 1:
        kw.update(do_sample=False, num_beams=args.num_beams,
                  length_penalty=args.length_penalty, early_stopping=True)
    else:
        kw["do_sample"] = False
    return kw


def mbr_select(candidates: list[str]) -> str:
    """Pick the candidate most similar to the others under the task's own metric.

    Minimum Bayes risk: with the other samples standing in for the unknown
    reference, the consensus candidate is the one with the highest expected
    score. Uses the lexical half of the composite as the utility -- it is the
    half that actually separates submissions, and it is cheap enough to run
    O(n^2) per example.
    """
    from metric import rouge_l_f1, token_f1

    if len(candidates) == 1:
        return candidates[0]
    best, best_u = candidates[0], -1.0
    for i, cand in enumerate(candidates):
        u = np.mean([
            0.3 * token_f1(cand, other) + 0.2 * rouge_l_f1(cand, other)
            for j, other in enumerate(candidates) if j != i
        ])
        if u > best_u:
            best, best_u = cand, u
    return best


def main(argv=None) -> int:
    args = parse_args(argv)
    import torch
    from transformers import set_seed

    set_seed(args.seed)

    path = Path(args.data)
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    if args.limit:
        df = df.head(args.limit)
    print(f"{len(df)} prompts from {path.name}")

    model, tok, dtype = load_for_inference(args.base_model, args.adapter, args.load_in_4bit,
                                           args.tokenizer)
    gen_kwargs = build_gen_kwargs(args)
    print(f"dtype={dtype} gen={json.dumps({k: v for k, v in gen_kwargs.items() if k != 'eos_token_id'})}")

    prompts = [render_prompt(x) for x in df["input"]]
    order = np.argsort([-len(p) for p in prompts])  # longest first: OOM shows up immediately
    outputs: dict[int, str] = {}
    n_capped = 0

    t0 = time.time()
    for start in range(0, len(order), args.batch_size):
        idx = order[start : start + args.batch_size]
        batch = [prompts[i] for i in idx]
        enc = tok(batch, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)

        with torch.no_grad():
            out = model.generate(**enc, **gen_kwargs)

        new = out[:, enc["input_ids"].shape[1]:]
        texts = tok.batch_decode(new, skip_special_tokens=True)
        n_capped += int(sum(1 for row in new if STOP_TOKEN_IDS[0] not in row.tolist()
                            and STOP_TOKEN_IDS[1] not in row.tolist()))

        k = args.mbr if args.mbr > 1 else 1
        for j, i in enumerate(idx):
            group = [t.strip() for t in texts[j * k : (j + 1) * k]]
            outputs[int(i)] = mbr_select(group) if k > 1 else group[0]

        done = start + len(idx)
        if start % (args.batch_size * 10) == 0 or done == len(order):
            rate = done / max(time.time() - t0, 1e-9)
            print(f"  {done}/{len(order)}  {rate:.2f} prompt/s  "
                  f"eta {(len(order)-done)/max(rate,1e-9)/60:.1f} min", flush=True)

    preds = [outputs[i] for i in range(len(prompts))]
    res = pd.DataFrame({"id": df["id"].values, "output": preds})
    empty = int((res["output"].str.strip().str.len() == 0).sum())

    # A handful of empties is a model quirk; a wave of them is a broken run, and
    # writing the file anyway is how a silent failure reaches the leaderboard.
    if empty > max(2, 0.02 * len(res)):
        raise SystemExit(
            f"{empty}/{len(res)} generations are empty -- refusing to write {args.out}. "
            f"Check the tokenizer, the adapter, and that the prompt format matches training."
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix == ".parquet":
        res.to_parquet(out_path, index=False)
    else:
        res.to_csv(out_path, index=False)

    meta = {**vars(args), "wall_seconds": round(time.time() - t0, 1),
            "hit_token_cap": n_capped, "empty_outputs": empty,
            "pred_words_mean": float(res["output"].str.split().str.len().mean())}
    out_path.with_suffix(".gen.json").write_text(json.dumps(meta, indent=2), "utf-8")

    print(f"\nwrote {out_path}  ({time.time()-t0:.0f}s)")
    print(f"mean predicted words {meta['pred_words_mean']:.1f} | "
          f"hit max_new_tokens: {n_capped}/{len(res)} | empty: {empty}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Local implementation of the Phase 1 composite metric.

    Score = 0.5 * BERTScore_F1 + 0.3 * Token_F1 + 0.2 * ROUGE-L_F1

scored per example against the reference, then averaged (rulebook section 4).

The organisers' exact implementation is not published, so this is a faithful
reading of the spec plus the most likely library defaults:

  * BERTScore -> `bert-base-multilingual-cased`, layer 9, no IDF weighting, no
    baseline rescaling. That is what the `bert_score` package resolves to for a
    non-English `lang=`, and it matches the published formula. Reimplemented here
    rather than pulled in as a dependency so the environment stays untouched and
    the layer/model are explicit; `--bert-model` lets us re-score with BanglaBERT
    to confirm a ranking is not an artefact of one encoder.
  * Token F1 -> multiset (bag-of-tokens) overlap on whitespace tokens, the SQuAD
    convention. `--set-tokens` switches to set semantics to check sensitivity.
  * ROUGE-L -> LCS over whitespace tokens, beta = 1. Implemented directly because
    `rouge_score`'s default tokeniser strips every non-ASCII character, which
    would zero out Bengali entirely.

No lowercasing and no stemming: Bengali is caseless and the references are raw
corpus text, so any normalisation here would measure something the leaderboard
does not.

What matters is that this ranks configurations the same way the leaderboard does,
not that it reproduces the absolute number.

Usage
-----
    python src/metric.py --pred preds.csv --ref data/val_dev.parquet
    python src/metric.py --pred preds.csv --ref data/val_dev.parquet --fast
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("NASCENIA_ROOT", Path(__file__).resolve().parents[1]))

W_BERT, W_TOKEN, W_ROUGE = 0.5, 0.3, 0.2

DEFAULT_BERT_MODEL = "bert-base-multilingual-cased"
DEFAULT_BERT_LAYER = 9


# --------------------------------------------------------------------------- #
# Lexical components
# --------------------------------------------------------------------------- #
def tokenize(text: str) -> list[str]:
    """Whitespace tokenisation. See module docstring for why nothing else."""
    return text.split()


def token_f1(pred: str, ref: str, multiset: bool = True) -> float:
    pt, rt = tokenize(pred), tokenize(ref)
    if not pt or not rt:
        return 0.0
    if multiset:
        overlap = sum((Counter(pt) & Counter(rt)).values())
        n_p, n_r = len(pt), len(rt)
    else:
        sp, sr = set(pt), set(rt)
        overlap = len(sp & sr)
        n_p, n_r = len(sp), len(sr)
    if overlap == 0:
        return 0.0
    precision, recall = overlap / n_p, overlap / n_r
    return 2 * precision * recall / (precision + recall)


def _lcs_length(a: list[int], b: list[int]) -> int:
    """Row-rolled LCS DP. Inner loop is hot enough to be worth the local binds."""
    if not a or not b:
        return 0
    if len(a) < len(b):  # keep the inner (allocated) row short
        a, b = b, a
    nb = len(b)
    prev = [0] * (nb + 1)
    for ai in a:
        cur = [0] * (nb + 1)
        prev_j = 0  # prev[j], carried so we index the list once per cell
        for j in range(nb):
            if ai == b[j]:
                cur[j + 1] = prev_j + 1
            else:
                up, left = prev[j + 1], cur[j]
                cur[j + 1] = up if up >= left else left
            prev_j = prev[j + 1]
        prev = cur
    return prev[nb]


def rouge_l_f1(pred: str, ref: str) -> float:
    pt, rt = tokenize(pred), tokenize(ref)
    if not pt or not rt:
        return 0.0
    vocab: dict[str, int] = {}
    a = [vocab.setdefault(t, len(vocab)) for t in pt]
    b = [vocab.setdefault(t, len(vocab)) for t in rt]
    lcs = _lcs_length(a, b)
    if lcs == 0:
        return 0.0
    precision, recall = lcs / len(pt), lcs / len(rt)
    return 2 * precision * recall / (precision + recall)


def lexical_scores(
    preds: list[str], refs: list[str], multiset: bool = True
) -> dict[str, np.ndarray]:
    if len(preds) != len(refs):
        raise ValueError(f"length mismatch: {len(preds)} preds vs {len(refs)} refs")
    return {
        "token_f1": np.array([token_f1(p, r, multiset) for p, r in zip(preds, refs)]),
        "rouge_l": np.array([rouge_l_f1(p, r) for p, r in zip(preds, refs)]),
    }


# --------------------------------------------------------------------------- #
# BERTScore
# --------------------------------------------------------------------------- #
class BERTScorer:
    """Greedy-matching BERTScore F1, no IDF, no baseline rescaling.

    Equivalent to `bert_score.score(..., model_type=model, num_layers=layer,
    idf=False, rescale_with_baseline=False)`. Special tokens are excluded from
    the matching, as in the reference implementation.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_BERT_MODEL,
        layer: int = DEFAULT_BERT_LAYER,
        device: str | None = None,
        batch_size: int = 32,
        max_length: int = 512,
    ):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.max_length = max_length
        self.model_name = model_name
        self.layer = layer

        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        self.model.eval().to(self.device)
        self.n_truncated = 0

    def _embed(self, texts: list[str]):
        """-> list of (n_tokens, hidden) L2-normalised tensors, specials dropped."""
        torch = self.torch
        out = []
        for start in range(0, len(texts), self.batch_size):
            chunk = texts[start : start + self.batch_size]
            enc = self.tok(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            lengths = enc["attention_mask"].sum(dim=1)
            self.n_truncated += int((lengths >= self.max_length).sum())
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                hidden = self.model(**enc).hidden_states[self.layer]
            hidden = torch.nn.functional.normalize(hidden, dim=-1)
            for i, n in enumerate(lengths.tolist()):
                out.append(hidden[i, 1 : n - 1].cpu())  # drop [CLS] / [SEP]
        return out

    def score(self, preds: list[str], refs: list[str]) -> np.ndarray:
        p_emb = self._embed(preds)
        r_emb = self._embed(refs)
        f1 = np.zeros(len(preds))
        for i, (pe, re_) in enumerate(zip(p_emb, r_emb)):
            if pe.shape[0] == 0 or re_.shape[0] == 0:
                continue
            sim = pe @ re_.T                       # (n_pred, n_ref) cosine
            precision = sim.max(dim=1).values.mean().item()
            recall = sim.max(dim=0).values.mean().item()
            if precision + recall > 0:
                f1[i] = 2 * precision * recall / (precision + recall)
        return f1


_SCORER_CACHE: dict[tuple[str, int], BERTScorer] = {}


def get_scorer(model_name: str, layer: int, **kw) -> BERTScorer:
    """Cached so a sweep loading the encoder once does not pay for it per config."""
    key = (model_name, layer)
    if key not in _SCORER_CACHE:
        _SCORER_CACHE[key] = BERTScorer(model_name, layer, **kw)
    return _SCORER_CACHE[key]


# --------------------------------------------------------------------------- #
# Composite
# --------------------------------------------------------------------------- #
def composite(
    preds: list[str],
    refs: list[str],
    fast: bool = False,
    multiset: bool = True,
    bert_model: str = DEFAULT_BERT_MODEL,
    bert_layer: int = DEFAULT_BERT_LAYER,
    per_example: bool = False,
) -> dict:
    """Composite score. `fast=True` skips BERTScore entirely (lexical half only).

    In fast mode `score` is the lexical partial sum (max 0.5), NOT comparable to
    a full score -- it is for ranking inside a sweep only.
    """
    preds = ["" if p is None else str(p) for p in preds]
    refs = ["" if r is None else str(r) for r in refs]
    lex = lexical_scores(preds, refs, multiset)

    if fast:
        bert = np.zeros(len(preds))
    else:
        bert = get_scorer(bert_model, bert_layer).score(preds, refs)

    score = W_BERT * bert + W_TOKEN * lex["token_f1"] + W_ROUGE * lex["rouge_l"]
    result = {
        "score": float(score.mean()),
        "bertscore_f1": float(bert.mean()),
        "token_f1": float(lex["token_f1"].mean()),
        "rouge_l_f1": float(lex["rouge_l"].mean()),
        "n": len(preds),
        "fast": fast,
        "pred_words_mean": float(np.mean([len(tokenize(p)) for p in preds])),
        "ref_words_mean": float(np.mean([len(tokenize(r)) for r in refs])),
    }
    if per_example:
        result["per_example"] = {
            "score": score,
            "bertscore_f1": bert,
            "token_f1": lex["token_f1"],
            "rouge_l_f1": lex["rouge_l"],
        }
    return result


# --------------------------------------------------------------------------- #
# I/O helpers + CLI
# --------------------------------------------------------------------------- #
def _load(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix in (".jsonl", ".json"):
        return pd.read_json(path, lines=path.suffix == ".jsonl")
    return pd.read_csv(path)


def align(
    pred_df: pd.DataFrame, ref_df: pd.DataFrame, subset_ok: bool = False
) -> tuple[list[str], list[str]]:
    """Join predictions to references on `id`.

    Strict by default: a missing prediction is an error, because on a real
    evaluation it means rows were silently dropped and the mean is computed over
    the wrong denominator. `subset_ok=True` scores just the overlap, for spot
    checks on a `--limit`ed run.
    """
    pred_col = next(
        (c for c in ("output", "doctor_response", "pred", "prediction") if c in pred_df),
        None,
    )
    if pred_col is None:
        raise ValueError(f"no prediction column in {list(pred_df.columns)}")
    merged = ref_df[["id", "output"]].merge(
        pred_df[["id", pred_col]].rename(columns={pred_col: "_pred"}),
        on="id",
        how="inner" if subset_ok else "left",
    )
    missing = int(merged["_pred"].isna().sum())
    if missing:
        raise ValueError(f"{missing} reference ids have no prediction")
    if subset_ok and len(merged) < len(ref_df):
        print(f"NOTE: scoring {len(merged)}/{len(ref_df)} references (subset).")
    return merged["_pred"].tolist(), merged["output"].tolist()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pred", required=True, help="csv/parquet/jsonl with id + output")
    ap.add_argument("--ref", required=True, help="csv/parquet with id + output")
    ap.add_argument("--fast", action="store_true", help="lexical only, no BERTScore")
    ap.add_argument("--set-tokens", action="store_true", help="set (not multiset) Token F1")
    ap.add_argument("--bert-model", default=DEFAULT_BERT_MODEL)
    ap.add_argument("--bert-layer", type=int, default=DEFAULT_BERT_LAYER)
    ap.add_argument("--json", help="also write the result dict here")
    args = ap.parse_args(argv)

    preds, refs = align(_load(args.pred), _load(args.ref))
    res = composite(
        preds,
        refs,
        fast=args.fast,
        multiset=not args.set_tokens,
        bert_model=args.bert_model,
        bert_layer=args.bert_layer,
    )
    if not args.fast:
        res["bert_model"] = f"{args.bert_model}@L{args.bert_layer}"
    print(json.dumps(res, indent=2, ensure_ascii=False))
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=2, ensure_ascii=False), "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

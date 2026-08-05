"""The single definition of how a patient prompt becomes model input.

Everything -- export, training, evaluation, inference -- renders through here, so
training and inference cannot drift apart. Train/inference prompt skew is the
classic silent failure in this kind of project: the model is fine, the prompt is
one newline different, and the score quietly drops.

Format is the Gemma-2 chat format, written out by hand because the *base*
checkpoints ship no `chat_template` (only the `-it` ones do). The turn tokens
already exist in the base vocabulary:

    <bos><start_of_turn>user
    {patient question}<end_of_turn>
    <start_of_turn>model
    {doctor answer}<end_of_turn>

No instruction preamble. With ~90k supervised examples an instruction teaches the
model nothing it does not learn from the data, and it would cost tokens on every
training row and every generation.
"""

from __future__ import annotations

BOS = "<bos>"
START = "<start_of_turn>"
END = "<end_of_turn>"

# Everything before this marker is context, everything after it is what we score
# and what the loss is computed on.
RESPONSE_PREFIX = f"{START}model\n"

# Generation must stop on <end_of_turn>; <eos> is included because a base model
# that has not fully absorbed the turn format can still emit it. Missing this is
# a Gemma footgun -- generation runs to max_new_tokens and the extra text costs
# BERTScore precision.
END_OF_TURN_ID = 107
EOS_ID = 1
STOP_TOKEN_IDS = [END_OF_TURN_ID, EOS_ID]

# <bos><start_of_turn>user \n <end_of_turn> \n <start_of_turn>model \n <end_of_turn> \n
TEMPLATE_OVERHEAD_TOKENS = 11


def render_prompt(patient_input: str) -> str:
    """Inference-time prompt: ends exactly where the model must start writing."""
    return f"{BOS}{START}user\n{patient_input}{END}\n{RESPONSE_PREFIX}"


def render_example(patient_input: str, doctor_output: str) -> str:
    """Full training sequence, prompt + target + terminator."""
    return f"{render_prompt(patient_input)}{doctor_output}{END}\n"


def truncate_input_tokens(text: str, tok, budget: int) -> tuple[str, bool]:
    """Fit `text` into `budget` tokens, keeping the head and the tail.

    Returns (text, was_truncated).

    Middle truncation rather than head- or tail-only: a Bengali patient prompt
    opens with demographics ("my age is 22, height 5'9\"...") and closes with the
    actual question, and both matter clinically. Dropping either end is worse
    than dropping the narrative in between.
    """
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= budget:
        return text, False

    head = budget // 3
    tail = budget - head
    out = tok.decode(ids[:head] + ids[-tail:], skip_special_tokens=True)

    # Re-encoding can land a token or two over budget because the join creates
    # new merges. Shave until it actually fits rather than trusting the count.
    while len(tok(out, add_special_tokens=False)["input_ids"]) > budget and tail > 1:
        tail -= 8
        out = tok.decode(ids[:head] + ids[-tail:], skip_special_tokens=True)
    return out, True

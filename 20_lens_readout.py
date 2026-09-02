# %% [markdown]
# # Stage 2/3 — lens readout over the filler region (GPU box)
#
# For each example (2-fact addition, chat format, replaying the stage-1 sweep's
# own fixed test set — see `SOURCE`):
#
# 1. greedy-generate the model's answer → correct / wrong split (paper Fig. 3
#    separates the two);
# 2. apply the selected lens(es) at every (source layer, position) over the
#    filler region and the post-filler tail (`"Answer:"` + generation prompt) —
#    * `LENS = "logit"` — logit-lens baseline (stage 2, the paper's readout;
#      the jlens code path with `use_jacobian=False`)
#    * `LENS = "jlens"` — Jacobian lens (stage 3)
#    * `LENS = "both"` — both in one pass (one model load, cheapest overall);
# 3. record, per (lens, layer, position): the top NUMERIC token's value and the
#    full-vocab ranks of A1, A2 and A1+A2.
#
# The output is a compact tidy CSV that `21_analyze_readout.py` turns into
# Figure-3-style heatmaps and `30_compare_lenses.py` into the J-lens-vs-logit-lens
# comparison. No raw hidden states are stored (the paper's 50 MB/example cache is
# not needed for this readout).
#
# Positions are passed to jlens as NEGATIVE indices (relative to sequence end) so
# a leading BOS added by internal re-tokenization can't shift them. Truncation
# *would* shift them, which is why every lens call goes through
# `common.apply_lens` with `MAX_SEQ_LEN = None` — see `common.fit_seq_len`.
#
# ```bash
# python 20_lens_readout.py --model dev --n 40 --k 10                # pipe-clean
# python 20_lens_readout.py --model deepseek --n 300 --k 10 --lens both
# ```

# %% Config
from __future__ import annotations

import os
import sys
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import paper_tasks as pt

MODEL = "dev"          # registry key in config.py
LENS = "logit"         # "logit" (stage 2) | "jlens" (stage 3) | "both" (one pass)
FILLER = "dots"        # see paper_tasks.FILLER_KINDS
K = 10                 # filler length
N = 300                # examples
POS_CHUNK = 32         # positions per lens.apply call (memory knob; one forward pass per chunk)

# Where the examples come from. "fig2" REPLAYS stage 1's own test set and its
# exact rendered prompts from data/fig2_2fact.jsonl — same pairs, same idx, same
# five few-shot pairs in context — which is what makes local answers comparable
# to the API sweep example-for-example (see 22_agreement_check.py). "synthetic"
# regenerates a test set from paper_tasks.build_dataset for a machine that has
# no stage-1 dataset; those examples are NOT the stage-1 ones, so an agreement
# check against fig2_raw.jsonl is meaningless. Falls back with a warning.
SOURCE = "fig2"        # "fig2" | "synthetic"
FIG2_PATH = "data/fig2_2fact.jsonl"
SEED = 0               # synthetic source only
MAX_SEQ_LEN = None     # None = size the context window to each prompt; never truncate
REGEN_ANSWERS = False  # re-generate greedy answers even if answers_<tag>.csv exists
OUTDIR = "results"

LENS_CHOICES = {"logit": [("logit", False)],
                "jlens": [("jlens", True)],
                "both": [("jlens", True), ("logit", False)]}


def _running_as_script() -> bool:
    """True for `python 20_lens_readout.py ...`; False in a notebook cell."""
    return __name__ == "__main__" and "ipykernel" not in sys.modules


if _running_as_script() and len(sys.argv) > 1:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL, help="registry key (config.py)")
    ap.add_argument("--lens", default=LENS, choices=list(LENS_CHOICES),
                    help="logit = stage 2 baseline; jlens = stage 3; both = one pass")
    ap.add_argument("--filler", default=FILLER, choices=list(pt.FILLER_KINDS))
    ap.add_argument("--k", type=int, default=K)
    ap.add_argument("--n", type=int, default=N)
    ap.add_argument("--source", default=SOURCE, choices=["fig2", "synthetic"],
                    help="fig2 = replay stage 1's examples and prompts (default)")
    ap.add_argument("--fig2-path", default=FIG2_PATH)
    ap.add_argument("--seed", type=int, default=SEED, help="synthetic source only")
    ap.add_argument("--pos-chunk", type=int, default=POS_CHUNK,
                    help="positions per lens.apply call (memory knob)")
    ap.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN,
                    help="context window; omit to size it to each prompt")
    ap.add_argument("--regen-answers", action="store_true")
    ap.add_argument("--outdir", default=OUTDIR)
    _a = ap.parse_args()
    MODEL, LENS, FILLER, K, N, SEED = (_a.model, _a.lens, _a.filler,
                                       _a.k, _a.n, _a.seed)
    SOURCE, FIG2_PATH = _a.source, _a.fig2_path
    POS_CHUNK, MAX_SEQ_LEN = _a.pos_chunk, _a.max_seq_len
    REGEN_ANSWERS, OUTDIR = _a.regen_answers, _a.outdir

os.makedirs(OUTDIR, exist_ok=True)
TAG = f"{MODEL}_{FILLER}-{K}"
OUT_CSV = os.path.join(OUTDIR, f"lens_readout_{TAG}_{LENS}.csv")
ANS_CSV = os.path.join(OUTDIR, f"answers_{TAG}.csv")
LENSES = LENS_CHOICES[LENS]
print(f"{TAG}  lens={LENS}  n={N}  source={SOURCE}\n  -> {OUT_CSV}\n  -> {ANS_CSV}")

# %% [markdown]
# ## 1. Load the model, the lens, and the numeric decode criterion
#
# The slow cell — weights, then the lens (a few hundred MB for the dev model,
# 1.4 GB for DeepSeek V4 Flash). `check_provenance` warns if the configured
# `hf_id` is not the checkpoint the lens was fit on; a lens is only meaningful on
# its own model. `load_model` prints VRAM after loading — check that number now
# rather than discovering an OOM 200 examples into the run.
#
# The decode criterion adapts to the tokenizer: "exact" where digits are grouped
# into single tokens (DeepSeek), "prefix" (first-token match) where they are
# split (Qwen, Llama 3). Headline numbers should come from an exact-mode model.

# %%
import torch

import config
from common import load_model, load_lens, check_provenance, apply_lens

spec = config.get(MODEL)
check_provenance(spec)
model, hf, tok = load_model(spec)
lens = load_lens(spec, kind="j")
numeric = pt.build_numeric_readout(tok)

print(f"\nlens: {len(lens.source_layers)} source layers "
      f"(layers {lens.source_layers[0]}..{lens.source_layers[-1]}, "
      f"d_model={lens.d_model}, n_prompts={lens.n_prompts})")
print(numeric.describe())
if numeric.mode == "prefix":
    print("NOTE: prefix mode — this model can pipe-clean the pipeline, but the "
          "headline decode numbers should come from DeepSeek.")

# %% [markdown]
# ## 2. Prompt rendering + position bookkeeping
#
# Prompts are rendered by `pt.render_chat`, which turns the model's reasoning
# mode OFF (both Qwen3.5 and DeepSeek V4 Flash open a `<think>` block by
# default — see its docstring). `locate_positions` returns NEGATIVE token
# indices: the filler region first, in order, then every trailing token — the
# `"Answer:"` tail and the generation prompt, whose last position (-1) is the
# one the answer is actually predicted from. Negative indexing survives a BOS
# being added or not by whoever re-tokenizes the prompt.

# %%
def locate_positions(tok, text: str, ex: pt.Example, kind: str, k: int
                     ) -> tuple[list[int], int]:
    """Return (negative token positions, n_filler).

    The first n_filler entries are filler positions, in order; the rest are the
    post-filler tail ("Answer:" and the generation prompt), through position -1.

    k=0 has no filler region, so n_filler is 0 and only the tail is read. That
    tail matters: the operands are decoded at the tail tokens of a filler prompt
    too, so the k=0 tail is the control for "do the dots add computation, or
    only more positions doing what 'Answer:' already does?". It also gives
    22_agreement_check.py the no-filler answers it needs for the uplift.
    """
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    if k == 0:
        a0 = text.rfind("Answer:")          # tail starts at the final "Answer:"
        tail = pt.span_to_negative_positions(offsets, a0, len(text))
        return tail, 0
    filler = pt.make_filler(kind, k)
    c0, c1 = pt.final_filler_char_span(text, filler, ex.question)
    fill_neg = pt.span_to_negative_positions(offsets, c0, c1)
    post_neg = list(range(fill_neg[-1] + 1, 0))   # after the filler, up to -1
    return fill_neg + post_neg, len(fill_neg)


# %% [markdown]
# ## 3. Readout of one example
#
# One row per (lens, layer, position). The `match_*` booleans are computed HERE
# rather than downstream, because whether a quantity counts as decoded depends on
# the tokenizer's numeric mode — `top_num` is only meaningful in "exact" mode, so
# a downstream `top_num == a1` comparison would silently report zero decodes on a
# digit-splitting tokenizer.
#
# `ctrl_*` is the same test against a *different* example's A1/A2/sum (the
# previous one in the dataset): the chance level of the decode criterion. It
# matters because "decoded" is an argmax over a few hundred numeric tokens
# (exact mode) or ten digits (prefix mode), so any-layer-any-position
# aggregates saturate on noise alone; 21/30 report the two side by side.

# %%
def readout_example(
    ex: pt.Example,
    control: pt.Example,
    text: str,
    positions: Sequence[int],
    n_filler: int,
    correct: bool,
    apply_fn: Callable[[str, Sequence[int], bool], dict[int, np.ndarray]],
    numeric: pt.NumericReadout,
    lenses: Sequence[tuple[str, bool]],
    pos_chunk: int = 16,
) -> list[dict]:
    rows = []
    quantities = (("A1", ex.a1, control.a1), ("A2", ex.a2, control.a2),
                  ("sum", ex.target, control.target))
    for lens_name, use_j in lenses:
        for start in range(0, len(positions), pos_chunk):
            chunk = list(positions[start:start + pos_chunk])
            out = apply_fn(text, chunk, use_j)
            for layer, arr in out.items():
                for pi_local, pos in enumerate(chunk):
                    pi = start + pi_local
                    row_logits = np.asarray(arr[pi_local], dtype=np.float32)
                    rec = dict(
                        idx=ex.idx, correct=correct, lens=lens_name,
                        layer=int(layer), pos=pi,
                        pos_type="filler" if pi < n_filler else "post",
                        readout_mode=numeric.mode,
                        top_tok=numeric.top_token(row_logits),
                        top_num=numeric.top_value(row_logits),
                    )
                    for qname, qval, cval in quantities:
                        rec[f"match_{qname}"] = numeric.decodes(row_logits, qval)
                        rec[f"rank_{qname}"] = numeric.rank(row_logits, qval)
                        rec[f"ctrl_{qname}"] = numeric.decodes(row_logits, cval)
                        rec[f"ctrl_rank_{qname}"] = numeric.rank(row_logits, cval)
                    rows.append(rec)
    return rows


def apply_fn(text, positions, use_j):
    out = apply_lens(lens, model, text, positions, use_jacobian=use_j,
                     max_seq_len=MAX_SEQ_LEN)
    return {int(L): t.detach().float().cpu().numpy() for L, t in out.items()}


@torch.no_grad()
def answer(text: str) -> str:
    """Greedy reply, special tokens kept so a stray <think> stays visible to
    pt.parse_answer (skip_special_tokens=True would strip it and score the
    reasoning trace's first integer as the prediction)."""
    enc = tok(text, add_special_tokens=False, return_tensors="pt")
    enc = {k_: v.to(hf.device) for k_, v in enc.items()}
    gen = hf.generate(**enc, max_new_tokens=6, do_sample=False,
                      pad_token_id=tok.eos_token_id)
    return tok.decode(gen[0, enc["input_ids"].shape[1]:], skip_special_tokens=False)


# %% [markdown]
# ## 4. Dataset + cached greedy answers
#
# `SOURCE = "fig2"` replays stage 1's own examples **and its exact rendered
# prompts**, so `idx` joins straight onto `results/fig2_raw.jsonl` and
# `22_agreement_check.py` can pair local answers against API answers. Note that
# `pt.build_dataset` (the `"synthetic"` source) does *not* reproduce that test
# set at any seed — it holds out a different 10 elements for few-shot, so it
# draws from a different pool. Use it only where there is no stage-1 dataset.
#
# Greedy answers are identical for every lens choice, so a `LENS = "jlens"` pass
# reuses the generations from the `"logit"` pass via `answers_<tag>.csv`.

# %%
dataset: list[tuple[pt.Example, list[dict]]] = []
if SOURCE == "fig2":
    if FILLER != "dots":
        raise ValueError(f"the stage-1 dataset is dots-only; FILLER={FILLER!r} "
                         f"needs SOURCE='synthetic'")
    if os.path.exists(FIG2_PATH):
        dataset = pt.load_fig2_examples(FIG2_PATH, K, n=N)
        print(f"[data] replaying stage 1: {len(dataset)} examples from "
              f"{FIG2_PATH} at k={K} (idx joins onto results/fig2_raw.jsonl)")
    else:
        import warnings
        warnings.warn(f"{FIG2_PATH} not found — falling back to a SYNTHETIC test "
                      f"set. These are NOT the stage-1 examples, so do not run "
                      f"22_agreement_check.py against the result.")
        SOURCE = "synthetic"
if SOURCE == "synthetic":
    dataset = [(ex, pt.build_messages(ex, FILLER, K))
               for ex in pt.build_dataset(N, seed=SEED)]
    print(f"[data] synthetic test set: {len(dataset)} examples, seed={SEED} "
          f"(not comparable to the stage-1 API sweep)")

cached_answers: dict[int, dict] = {}
if os.path.exists(ANS_CSV) and not REGEN_ANSWERS:
    cached_answers = {int(r["idx"]): dict(r)
                      for _, r in pd.read_csv(ANS_CSV).iterrows()}
    print(f"[answers] reusing {len(cached_answers)} greedy answers from {ANS_CSV}")

# One prompt's worth of bookkeeping up front, so a surprise shows up here and
# not as a bad heatmap 200 examples later: how long the prompts are, which
# tokens the post-filler tail reads, and what the model actually replies —
# a reasoning trace or a cached answer from a broken run would show up right
# here as an unparsed reply or a cache/model disagreement.
_ex0, _msgs0 = dataset[0]
_probe = pt.render_chat(tok, _msgs0)
_probe_ids = tok(_probe, add_special_tokens=False).input_ids
_probe_pos, _probe_nf = locate_positions(tok, _probe, _ex0, FILLER, K)
print(f"[prompt] {len(_probe_ids)} tokens, {len(_probe_pos)} readout positions "
      f"({_probe_nf} filler) — "
      f"max_seq_len={MAX_SEQ_LEN if MAX_SEQ_LEN else 'fitted per prompt'}")
print(f"[prompt] post-filler tail read: "
      f"{[tok.decode([_probe_ids[p]]) for p in _probe_pos[_probe_nf:]]}")
_reply = answer(_probe)
_pred = pt.parse_answer(_reply)
print(f"[probe] {_ex0.elem_a}+{_ex0.elem_b}={_ex0.target}: reply {_reply!r} "
      f"-> pred {_pred}")
if _pred is None:
    raise RuntimeError("the model's reply has no answer in it (a reasoning trace, "
                       "most likely) — fix the prompt before spending GPU hours")
if _ex0.idx in cached_answers and cached_answers[_ex0.idx]["pred"] != _pred:
    raise RuntimeError(f"{ANS_CSV} is stale: it says {cached_answers[_ex0.idx]['pred']} "
                       f"for idx {_ex0.idx}, the model now says {_pred}. "
                       f"Set REGEN_ANSWERS = True (--regen-answers).")

# %% [markdown]
# ## 5. Readout loop
#
# Checkpoints both CSVs every 10 examples, so an interrupted run keeps what it
# had. Re-run this cell after changing `LENSES`/`POS_CHUNK` without reloading the
# model.

# %%
all_rows: list[dict] = []
answers: list[dict] = []
n_filler = n_positions = 0
max_prompt_tokens = 0

for i, (ex, msgs) in enumerate(tqdm(dataset, desc=f"readout ({LENS})", unit="ex")):
    text = pt.render_chat(tok, msgs)
    positions, n_filler = locate_positions(tok, text, ex, FILLER, K)
    n_positions = len(positions)
    max_prompt_tokens = max(max_prompt_tokens,
                            len(tok(text, add_special_tokens=False).input_ids))

    if ex.idx in cached_answers:
        a = cached_answers[ex.idx]
        pred, correct = a["pred"], bool(a["correct"])
        answers.append(a)
    else:
        reply = answer(text)
        pred = pt.parse_answer(reply)
        correct = pred == ex.target
        answers.append(dict(idx=ex.idx, elem_a=ex.elem_a, elem_b=ex.elem_b,
                            a1=ex.a1, a2=ex.a2, target=ex.target,
                            reply=str(reply).strip()[:32], pred=pred,
                            correct=correct))

    if positions:
        control = dataset[i - 1][0]   # another example's quantities = chance level
        all_rows += readout_example(ex, control, text, positions, n_filler, correct,
                                    apply_fn, numeric, LENSES, pos_chunk=POS_CHUNK)

    if (i + 1) % 10 == 0 or i == 0:
        if all_rows:
            pd.DataFrame(all_rows).to_csv(OUT_CSV, index=False)
        pd.DataFrame(answers).to_csv(ANS_CSV, index=False)

print(f"running accuracy: {np.mean([a['correct'] for a in answers]):.2%}  "
      f"(unparsed replies: {sum(pd.isna(a['pred']) for a in answers)})")

# %% [markdown]
# ## 6. Write + sanity peek
#
# The peek is the cheapest way to tell a working readout from a broken one before
# you spend time on figures: at the last filler position, the top numeric token
# should drift toward A1/A2/sum as the layer index rises.

# %%
answers_df = pd.DataFrame(answers)
answers_df.to_csv(ANS_CSV, index=False)
print(f"wrote {ANS_CSV}  (accuracy {answers_df.correct.mean():.2%})")
print(f"longest prompt seen: {max_prompt_tokens} tokens")

readout = pd.DataFrame(all_rows)
readout.to_csv(OUT_CSV, index=False)
print(f"wrote {OUT_CSV} ({len(readout)} rows; {n_positions} positions/example, "
      f"{n_filler} filler; readout mode={numeric.mode})")

if n_filler == 0:                     # k=0: baseline answers + the tail readout
    print(f"\nk=0: no filler region; the tail ('Answer:' onward) was read so 21 can "
          f"compare it with a filler prompt's tail. Next: 22_agreement_check.py "
          f"for the local-vs-API uplift.")
else:
    last = readout[(readout.pos == n_filler - 1) & readout.correct]
    print("\ndecode fraction at the last filler position (correct examples), "
          "with the shuffled-quantity control (ctrl_*) as chance level:")
    if last.empty:
        print("  (no correct examples in this run — nothing to peek at)")
    else:
        print(last.groupby(["lens", "layer"])[["match_A1", "ctrl_A1", "match_A2",
                                               "ctrl_A2", "match_sum", "ctrl_sum"]]
                  .mean().round(3).to_string())
    print(f"\nnext: 21_analyze_readout.py (TAG = {TAG!r})")
    if SOURCE == "fig2":
        print("      22_agreement_check.py to check local accuracy against the API")

# %%

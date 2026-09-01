# %% [markdown]
# # Stage 2/3 — lens readout over the filler region (GPU box)
#
# For each example (2-fact addition, chat format, fixed test set shared with the
# stage-1 sweep via `SEED`):
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
# Run cells top-to-bottom in VS Code (`# %%` = one Jupyter cell). Cell 1 is the
# slow one; re-run cell 5 freely without reloading the model. The same file still
# runs headless on a rented box:
#
# ```bash
# python 20_lens_readout.py --model dev --n 40 --k 10                # pipe-clean
# python 20_lens_readout.py --model deepseek --n 300 --k 10 --lens both
# ```

# %% Config
from __future__ import annotations

import os
import sys
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import paper_tasks as pt

MODEL = "dev"          # registry key in config.py
LENS = "logit"         # "logit" (stage 2) | "jlens" (stage 3) | "both" (one pass)
FILLER = "dots"        # see paper_tasks.FILLER_KINDS
K = 10                 # filler length
N = 300                # examples
SEED = 0               # SAME seed as stage 1, or the two stages describe different examples
POS_CHUNK = 16         # positions per lens.apply call (memory knob)
POST_TAIL_MAX = 8      # post-filler token positions to read (incl. "Answer:")
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
    ap.add_argument("--seed", type=int, default=SEED, help="same seed as stage 1!")
    ap.add_argument("--pos-chunk", type=int, default=POS_CHUNK,
                    help="positions per lens.apply call (memory knob)")
    ap.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN,
                    help="context window; omit to size it to each prompt")
    ap.add_argument("--regen-answers", action="store_true")
    ap.add_argument("--outdir", default=OUTDIR)
    _a = ap.parse_args()
    MODEL, LENS, FILLER, K, N, SEED = (_a.model, _a.lens, _a.filler,
                                       _a.k, _a.n, _a.seed)
    POS_CHUNK, MAX_SEQ_LEN = _a.pos_chunk, _a.max_seq_len
    REGEN_ANSWERS, OUTDIR = _a.regen_answers, _a.outdir

os.makedirs(OUTDIR, exist_ok=True)
TAG = f"{MODEL}_{FILLER}-{K}"
OUT_CSV = os.path.join(OUTDIR, f"lens_readout_{TAG}_{LENS}.csv")
ANS_CSV = os.path.join(OUTDIR, f"answers_{TAG}.csv")
LENSES = LENS_CHOICES[LENS]
print(f"{TAG}  lens={LENS}  n={N}  seed={SEED}\n  -> {OUT_CSV}\n  -> {ANS_CSV}")

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
# `locate_positions` returns NEGATIVE token indices: the filler region first, in
# order, then up to `POST_TAIL_MAX` trailing tokens (the `"Answer:"` tail and the
# generation prompt). Negative indexing survives a BOS that jlens's own
# re-tokenization may prepend.

# %%
def render_chat(tok, ex: pt.Example, kind: str, k: int) -> str:
    msgs = pt.build_messages(ex, kind, k)
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def locate_positions(tok, text: str, ex: pt.Example, kind: str, k: int
                     ) -> Tuple[List[int], int]:
    """Return (negative token positions, n_filler).

    The first n_filler entries are filler positions, in order; the rest are the
    post-filler tail.
    """
    filler = pt.make_filler(kind, k)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    c0, c1 = pt.final_filler_char_span(text, filler, ex.question)
    fill_neg = pt.span_to_negative_positions(offsets, c0, c1)
    # post tail: negative indices after the last filler token, up to -1
    post_neg = list(range(fill_neg[-1] + 1, 0))[:POST_TAIL_MAX]
    return fill_neg + post_neg, len(fill_neg)


# %% [markdown]
# ## 3. Readout of one example
#
# One row per (lens, layer, position). The `match_*` booleans are computed HERE
# rather than downstream, because whether a quantity counts as decoded depends on
# the tokenizer's numeric mode — `top_num` is only meaningful in "exact" mode, so
# a downstream `top_num == a1` comparison would silently report zero decodes on a
# digit-splitting tokenizer.

# %%
def readout_example(
    ex: pt.Example,
    text: str,
    positions: Sequence[int],
    n_filler: int,
    correct: bool,
    apply_fn: Callable[[str, Sequence[int], bool], Dict[int, np.ndarray]],
    numeric: pt.NumericReadout,
    lenses: Sequence[Tuple[str, bool]],
    pos_chunk: int = 16,
) -> List[dict]:
    rows = []
    quantities = (("A1", ex.a1), ("A2", ex.a2), ("sum", ex.target))
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
                    for qname, qval in quantities:
                        rec[f"match_{qname}"] = numeric.decodes(row_logits, qval)
                        rec[f"rank_{qname}"] = numeric.rank(row_logits, qval)
                    rows.append(rec)
    return rows


def apply_fn(text, positions, use_j):
    out = apply_lens(lens, model, text, positions, use_jacobian=use_j,
                     max_seq_len=MAX_SEQ_LEN)
    return {int(L): t.detach().float().cpu().numpy() for L, t in out.items()}


@torch.no_grad()
def answer(text: str) -> str:
    enc = tok(text, add_special_tokens=False, return_tensors="pt")
    enc = {k_: v.to(hf.device) for k_, v in enc.items()}
    gen = hf.generate(**enc, max_new_tokens=6, do_sample=False,
                      pad_token_id=tok.eos_token_id)
    return tok.decode(gen[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)


# %% [markdown]
# ## 4. Dataset + cached greedy answers
#
# Greedy answers are identical for every lens choice, so a `LENS = "jlens"` pass
# reuses the generations from the `"logit"` pass via `answers_<tag>.csv`.

# %%
dataset = pt.build_dataset(N, seed=SEED)

cached_answers: Dict[int, dict] = {}
if os.path.exists(ANS_CSV) and not REGEN_ANSWERS:
    cached_answers = {int(r["idx"]): dict(r)
                      for _, r in pd.read_csv(ANS_CSV).iterrows()}
    print(f"[answers] reusing {len(cached_answers)} greedy answers from {ANS_CSV}")

# One prompt's worth of bookkeeping up front: how long these prompts actually
# are, so a context-window surprise shows up here and not as a bad heatmap.
_probe = render_chat(tok, dataset[0], FILLER, K)
_probe_pos, _probe_nf = locate_positions(tok, _probe, dataset[0], FILLER, K)
print(f"[prompt] {len(tok(_probe, add_special_tokens=False).input_ids)} tokens, "
      f"{len(_probe_pos)} readout positions ({_probe_nf} filler) — "
      f"max_seq_len={MAX_SEQ_LEN if MAX_SEQ_LEN else 'fitted per prompt'}")

# %% [markdown]
# ## 5. Readout loop
#
# Checkpoints both CSVs every 10 examples, so an interrupted run keeps what it
# had. Re-run this cell after changing `LENSES`/`POS_CHUNK` without reloading the
# model.

# %%
all_rows: List[dict] = []
answers: List[dict] = []
n_filler = n_positions = 0
max_prompt_tokens = 0

for i, ex in enumerate(tqdm(dataset, desc=f"readout ({LENS})", unit="ex")):
    text = render_chat(tok, ex, FILLER, K)
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

    all_rows += readout_example(ex, text, positions, n_filler, correct,
                                apply_fn, numeric, LENSES, pos_chunk=POS_CHUNK)

    if (i + 1) % 10 == 0 or i == 0:
        pd.DataFrame(all_rows).to_csv(OUT_CSV, index=False)
        pd.DataFrame(answers).to_csv(ANS_CSV, index=False)

print(f"running accuracy: {np.mean([a['correct'] for a in answers]):.2%}")

# %% [markdown]
# ## 6. Write + sanity peek
#
# The peek is the cheapest way to tell a working readout from a broken one before
# you spend time on figures: at the last filler position, the top numeric token
# should drift toward A1/A2/sum as the layer index rises.

# %%
readout = pd.DataFrame(all_rows)
answers_df = pd.DataFrame(answers)
readout.to_csv(OUT_CSV, index=False)
answers_df.to_csv(ANS_CSV, index=False)

print(f"wrote {OUT_CSV} ({len(readout)} rows; {n_positions} positions/example, "
      f"{n_filler} filler; readout mode={numeric.mode})")
print(f"wrote {ANS_CSV}  (accuracy {answers_df.correct.mean():.2%})")
print(f"longest prompt seen: {max_prompt_tokens} tokens")

peek = (readout[(readout.pos == n_filler - 1) & readout.correct]
        .groupby(["lens", "layer"])[["match_A1", "match_A2", "match_sum"]]
        .mean().round(3))
print("\ndecode fraction at the last filler position (correct examples):")
print(peek.to_string())
print(f"\nnext: 21_analyze_readout.py (TAG = {TAG!r})")

# %%

"""
Lens extraction for the Figure 3 replication + the J-lens comparison.

For each example (2-fact addition, chat format, fixed test set shared with
03_accuracy_sweep.py):
  1. greedy-generate the model's answer -> correct / wrong split (paper
     Fig. 3 separates the two);
  2. apply BOTH the Jacobian lens and the logit-lens baseline
     (use_jacobian=True / False — same code path in jlens) at every
     (source layer, position) over the filler region and the post-filler
     tail ("Answer:" + generation prompt);
  3. record, per (lens, layer, position): the top NUMERIC token's value and
     the full-vocab ranks of A1, A2 and A1+A2.

The output is a compact tidy CSV that 05_analyze_lens.py turns into
Figure-3-style heatmaps and the J-lens-vs-logit-lens comparison. No raw
hidden states are stored (the paper's 50 MB/example cache is not needed
for this readout).

Positions are passed to jlens as NEGATIVE indices (relative to sequence
end) so a leading BOS added by internal re-tokenization can't shift them.

Run (on the GPU box; dev model first):
    python 04_lens_readout.py --model dev --n 40 --k 10          # pipe-clean
    python 04_lens_readout.py --model deepseek --n 300 --k 10    # Fig. 3 cond
    python 04_lens_readout.py --model deepseek --n 150 --k 50

Output (results/):
    lens_readout_<model>_<filler>-<k>.csv
    answers_<model>_<filler>-<k>.csv
"""
from __future__ import annotations

import argparse
import os
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

import paper_tasks as pt

POST_TAIL_MAX = 8  # how many post-filler token positions to read (incl. "Answer:")


# --------------------------------------------------------------------------- #
# Rendering + token bookkeeping (needs only a tokenizer, mock-testable)
# --------------------------------------------------------------------------- #
def render_chat(tok, ex: pt.Example, kind: str, k: int) -> str:
    msgs = pt.build_messages(ex, kind, k)
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def locate_positions(tok, text: str, ex: pt.Example, kind: str, k: int
                     ) -> Tuple[List[int], int]:
    """Return (negative token positions, n_filler).

    Positions cover the filler region plus up to POST_TAIL_MAX trailing
    tokens after it (the "Answer:" tail + generation prompt). The first
    n_filler entries are filler positions, in order.
    """
    filler = pt.make_filler(kind, k)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    c0, c1 = pt.final_filler_char_span(text, filler, ex.question)
    fill_neg = pt.span_to_negative_positions(offsets, c0, c1)
    # post tail: negative indices after the last filler token, up to -1
    post_neg = list(range(fill_neg[-1] + 1, 0))[:POST_TAIL_MAX]
    return fill_neg + post_neg, len(fill_neg)


# --------------------------------------------------------------------------- #
# Core per-example readout, with the lens call injected (mock-testable).
# apply_fn(text, positions, use_jacobian) -> {layer: np.ndarray[P, vocab]}
# --------------------------------------------------------------------------- #
def readout_example(
    ex: pt.Example,
    text: str,
    positions: Sequence[int],
    n_filler: int,
    correct: bool,
    apply_fn: Callable[[str, Sequence[int], bool], Dict[int, np.ndarray]],
    tok,
    numeric_ids: Dict[int, int],
    pos_chunk: int = 16,
) -> List[dict]:
    rows = []
    quantities = (("A1", ex.a1), ("A2", ex.a2), ("sum", ex.target))
    for lens_name, use_j in (("jlens", True), ("logit", False)):
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
                        top_num=pt.top_numeric_value(row_logits, numeric_ids),
                    )
                    for qname, qval in quantities:
                        rec[f"rank_{qname}"] = pt.value_rank(row_logits, tok, qval)
                    rows.append(rec)
    return rows


# --------------------------------------------------------------------------- #
# Real-model driver
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dev", help="registry key (config.py)")
    ap.add_argument("--filler", default="dots", choices=list(pt.FILLER_KINDS))
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0, help="same seed as 03!")
    ap.add_argument("--pos-chunk", type=int, default=16,
                    help="positions per lens.apply call (memory knob)")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    import torch
    import config
    from common import load_model, load_lens, check_provenance, apply_lens

    os.makedirs(args.outdir, exist_ok=True)
    tag = f"{args.model}_{args.filler}-{args.k}"
    out_csv = os.path.join(args.outdir, f"lens_readout_{tag}.csv")
    ans_csv = os.path.join(args.outdir, f"answers_{tag}.csv")

    spec = config.get(args.model)
    check_provenance(spec)
    model, hf, tok = load_model(spec)
    lens = load_lens(spec, kind="j")
    numeric_ids = pt.numeric_token_ids(tok)
    print(f"[numeric tokens] {len(numeric_ids)} single-token integer spellings")

    def apply_fn(text, positions, use_j):
        out = apply_lens(lens, model, text, positions, use_jacobian=use_j)
        return {int(L): t.detach().float().cpu().numpy() for L, t in out.items()}

    @torch.no_grad()
    def answer(text: str) -> str:
        enc = tok(text, add_special_tokens=False, return_tensors="pt")
        enc = {k_: v.to(hf.device) for k_, v in enc.items()}
        gen = hf.generate(**enc, max_new_tokens=6, do_sample=False,
                          pad_token_id=tok.eos_token_id)
        return tok.decode(gen[0, enc["input_ids"].shape[1]:],
                          skip_special_tokens=True)

    dataset = pt.build_dataset(args.n, seed=args.seed)
    all_rows, answers = [], []
    for i, ex in enumerate(dataset):
        text = render_chat(tok, ex, args.filler, args.k)
        positions, n_filler = locate_positions(tok, text, ex, args.filler, args.k)
        reply = answer(text)
        pred = pt.parse_answer(reply)
        correct = pred == ex.target
        answers.append(dict(idx=ex.idx, elem_a=ex.elem_a, elem_b=ex.elem_b,
                            a1=ex.a1, a2=ex.a2, target=ex.target,
                            reply=reply.strip()[:32], pred=pred, correct=correct))
        all_rows += readout_example(ex, text, positions, n_filler, correct,
                                    apply_fn, tok, numeric_ids,
                                    pos_chunk=args.pos_chunk)
        if (i + 1) % 10 == 0 or i == 0:
            acc = np.mean([a["correct"] for a in answers])
            print(f"[{i+1}/{len(dataset)}] running acc={acc:.2%}  "
                  f"({ex.elem_a}+{ex.elem_b}={ex.target}, got {pred})")
            # checkpoint so long runs are resumable-ish
            pd.DataFrame(all_rows).to_csv(out_csv, index=False)
            pd.DataFrame(answers).to_csv(ans_csv, index=False)

    pd.DataFrame(all_rows).to_csv(out_csv, index=False)
    pd.DataFrame(answers).to_csv(ans_csv, index=False)
    n_pos = len(positions)
    print(f"\nwrote {out_csv} ({len(all_rows)} rows; {n_pos} positions/example, "
          f"{n_filler} filler)")
    print(f"wrote {ans_csv}  (accuracy {np.mean([a['correct'] for a in answers]):.2%})")
    print("next: python 05_analyze_lens.py --readout", out_csv)


if __name__ == "__main__":
    main()

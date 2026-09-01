"""
Stage 4 (optional) — attention-mechanism study, as in the paper (Sec. 4.1).

Ready-to-run code for the attention analysis; do stages 2-3 first. The paper's
question: what do filler positions ATTEND TO while the hidden computation runs?
Its answer: filler tokens attend back to the question (the fact entities), and
the answer position attends to the filler region — consistent with retrieval
happening at filler positions and being collected at the answer slot.

For each example this script runs one forward pass with attention weights and
records, per (layer, head, query position), the attention mass aggregated over
key spans:
    entity_a | entity_b | question_rest | filler | fewshot | other
for query positions covering the filler region + the post-filler tail.

Outputs (results/):
    attn_<model>_<filler>-<k>.csv        tidy per (idx, layer, head, qpos, span)
    attn_summary_<model>_<filler>-<k>.png  layer x qpos maps of entity/filler mass
                                           (head-averaged, example-averaged)

Run (dev model first — attention capture is memory-hungry):
    python 40_attention_study.py --model dev --n 40 --k 10
    python 40_attention_study.py --model deepseek --n 100 --k 10

Memory note: attention weights need eager attention (flash/sdpa kernels never
materialize them), which this script forces at load time. All layers' weights
for one sequence are held at once — on the big model keep k and the few-shot
prompt modest, or thin --layers-every.
"""
from __future__ import annotations

import argparse
import os
from typing import List, Tuple

import numpy as np
import pandas as pd

import paper_tasks as pt

POST_TAIL_MAX = 8
SPANS = ["entity_a", "entity_b", "question_rest", "filler", "fewshot", "other"]


# --------------------------------------------------------------------------- #
# Span bookkeeping: char spans -> absolute token index sets
# --------------------------------------------------------------------------- #
def char_span_tokens(offsets, c0: int, c1: int) -> List[int]:
    """Absolute token indices overlapping [c0, c1) (zero-length specials skipped)."""
    return [i for i, (s, e) in enumerate(offsets) if e > s and s < c1 and e > c0]


def build_spans(tok, text: str, ex: pt.Example, kind: str, k: int):
    """Return (query positions, n_filler, {span_name: set of key token indices})."""
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    n_tok = len(offsets)

    filler = pt.make_filler(kind, k)
    fc0, fc1 = pt.final_filler_char_span(text, filler, ex.question)
    filler_toks = char_span_tokens(offsets, fc0, fc1)

    q_at = text.rfind(ex.question)
    q_toks = set(char_span_tokens(offsets, q_at, q_at + len(ex.question)))
    ea_at = text.find(ex.elem_a, q_at)
    eb_at = text.find(ex.elem_b, ea_at + len(ex.elem_a))
    ea_toks = set(char_span_tokens(offsets, ea_at, ea_at + len(ex.elem_a)))
    eb_toks = set(char_span_tokens(offsets, eb_at, eb_at + len(ex.elem_b)))

    spans = {
        "entity_a": ea_toks,
        "entity_b": eb_toks,
        "question_rest": q_toks - ea_toks - eb_toks,
        "filler": set(filler_toks),
        "fewshot": set(range(0, min(q_toks) if q_toks else 0)),
    }
    covered = set().union(*spans.values())
    spans["other"] = set(range(n_tok)) - covered

    # query positions: filler region + post tail (absolute indices)
    post = list(range(filler_toks[-1] + 1, n_tok))[:POST_TAIL_MAX]
    qpos = filler_toks + post
    return qpos, len(filler_toks), spans


# --------------------------------------------------------------------------- #
# Per-example attention readout
# --------------------------------------------------------------------------- #
def attention_rows(attns, qpos, n_filler, spans, idx: int, correct: bool,
                   layers_every: int = 1) -> List[dict]:
    """attns: tuple over layers of [heads, seq, seq] numpy arrays (batch removed)."""
    rows = []
    span_idx = {name: np.fromiter(toks, dtype=np.int64)
                for name, toks in spans.items() if toks}
    for L, att in enumerate(attns):
        if L % layers_every:
            continue
        for qi, qp in enumerate(qpos):
            att_q = att[:, qp, :]                      # [heads, seq]
            for name, kidx in span_idx.items():
                mass = att_q[:, kidx].sum(axis=1)      # [heads]
                for h, m in enumerate(mass):
                    rows.append(dict(
                        idx=idx, correct=correct, layer=L, head=h, qpos=qi,
                        pos_type="filler" if qi < n_filler else "post",
                        span=name, mass=float(m)))
    return rows


def plot_summary(df: pd.DataFrame, n_filler: int, tag: str, outdir: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    show = ["entity_a", "entity_b", "question_rest", "filler"]
    fig, axes = plt.subplots(1, len(show), figsize=(4 * len(show), 4),
                             sharey=True)
    for ax, span in zip(axes, show):
        sub = df[(df.span == span) & df.correct]
        p = sub.pivot_table(index="layer", columns="qpos", values="mass",
                            aggfunc="mean")  # head- and example-averaged
        im = ax.imshow(p.to_numpy(), aspect="auto", origin="lower",
                       cmap="viridis",
                       extent=[p.columns[0] - .5, p.columns[-1] + .5,
                               p.index[0] - .5, p.index[-1] + .5])
        ax.axvline(n_filler - 0.5, color="w", ls="--", lw=1)
        ax.set_title(f"mass on {span}", fontsize=10)
        ax.set_xlabel("query position (filler → | answer)")
        fig.colorbar(im, ax=ax, fraction=0.046)
    axes[0].set_ylabel("layer")
    fig.suptitle(f"Attention from filler/answer positions — {tag} "
                 "(correct examples, head-averaged)")
    fig.tight_layout()
    path = os.path.join(outdir, f"attn_summary_{tag}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")

    # top heads: most entity-directed attention from inside the filler
    ent = df[df.correct & (df.pos_type == "filler")
             & df.span.isin(["entity_a", "entity_b"])]
    top = (ent.groupby(["layer", "head"])["mass"].mean()
              .sort_values(ascending=False).head(15))
    print("\ntop (layer, head) by mean entity-directed mass from filler positions:")
    print(top.to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dev", help="registry key (config.py)")
    ap.add_argument("--filler", default="dots", choices=list(pt.FILLER_KINDS))
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0, help="same seed as stage 1!")
    ap.add_argument("--layers-every", type=int, default=1,
                    help="keep every Nth layer (memory/size knob)")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    import torch
    import transformers
    import config
    from common import load_model

    os.makedirs(args.outdir, exist_ok=True)
    tag = f"{args.model}_{args.filler}-{args.k}"
    out_csv = os.path.join(args.outdir, f"attn_{tag}.csv")

    # eager attention so weights are materialized (flash/sdpa never build them)
    spec = config.get(args.model)
    orig = transformers.AutoModelForCausalLM.from_pretrained
    transformers.AutoModelForCausalLM.from_pretrained = (
        lambda *a, **kw: orig(*a, attn_implementation="eager", **kw))
    try:
        model, hf, tok = load_model(spec)
    finally:
        transformers.AutoModelForCausalLM.from_pretrained = orig

    @torch.no_grad()
    def forward_attn(text: str):
        enc = tok(text, add_special_tokens=False, return_tensors="pt")
        enc = {k_: v.to(hf.device) for k_, v in enc.items()}
        out = hf(**enc, output_attentions=True, use_cache=False)
        gen_logits = out.logits[0, -1]
        attns = tuple(a[0].float().cpu().numpy() for a in out.attentions)
        return attns, gen_logits

    dataset = pt.build_dataset(args.n, seed=args.seed)
    all_rows = []
    for i, ex in enumerate(dataset):
        msgs = pt.build_messages(ex, args.filler, args.k)
        text = tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True)
        qpos, n_filler, spans = build_spans(tok, text, ex, args.filler, args.k)
        attns, gen_logits = forward_attn(text)
        # cheap correctness proxy: argmax of the next-token logits
        pred = pt.parse_answer(tok.decode([int(gen_logits.argmax())]))
        correct = pred == ex.target
        all_rows += attention_rows(attns, qpos, n_filler, spans,
                                   ex.idx, correct, args.layers_every)
        if (i + 1) % 10 == 0 or i == 0:
            print(f"[{i+1}/{len(dataset)}] {ex.elem_a}+{ex.elem_b}={ex.target} "
                  f"(next-token pred {pred})")
            pd.DataFrame(all_rows).to_csv(out_csv, index=False)

    df = pd.DataFrame(all_rows)
    df.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv} ({len(df)} rows)")
    plot_summary(df, n_filler, tag, args.outdir)


if __name__ == "__main__":
    main()

"""
Stage 5 — causal tests: does the answer position READ the dots, and is what it
reads the lens-readable operand?

The lens readouts (20/21/30) are correlational: the operands are decodable on
the dot positions at layers 34-40, but so are they at every other
post-question token, and nothing there shows that the answer position uses
what sits on the dots. Two interventions, on the k=10 prompts of the readouts,
each ending in a greedy answer:

A. Position patching (parameter-free, in-distribution). A DONOR example with
   disjoint elements and a different sum is chosen for each example.
     clean      no patch — must reproduce results/answers_<tag>.csv;
     self@L     the dot positions' residual at the output of layer L is
                replaced by the example's OWN recorded residual — plumbing
                check, must reproduce `clean`;
     donor@L    the same with the DONOR's residual at its own dots (all
                hyper-connection streams), for L in LAYERS. Only the prefill
                is patched; layers L+1.. and every decode step run on it.

B. J-lens coordinate edits (paper Sec. 2.5 of the workspace paper), at the dot
   positions only, at every layer of BAND. A token's lens vector at layer l is
   v_t = (W_U[t] * g) J_l (g = the final RMSNorm gain); the swap s->t reflects
   the residual across the bisector of the unit vectors, h -= 2a(h.u)u with
   u ~ v_s/|v_s| - v_t/|v_t| (a=1 is the exact coordinate exchange, a=2 the
   paper's double strength); the ablation projects the unit vector(s) out.
     jswap_A1     A1 -> donor's A1 (a=1), and jswap_A1x2 (a=2)
     jswap_A2     A2 -> donor's A2 (a=1)
     jswap_ctrl   two numbers absent from both prompts (a=1): must do nothing
     jabl_A1 / jabl_A2 / jabl_A1A2   project the operand direction(s) out

Each prediction is classified: the true sum, the donor's sum, a MIXED sum
(donor_a1+a2 = A1 replaced, a1+donor_a2 = A2 replaced), a bare operand, other.

Predictions, written before the run. If the answer position reads operands
off the dots: donor patching BEFORE the operands appear there (L <= 30) is
harmless (retrieval after L still attends to the recipient's own question);
AFTER (L >= 37) accuracy falls and predictions move to donor-derived values.
The k=0 tail readout says the dots' contribution is the FIRST operand (A2 is
retrieved at the answer position itself), so: donor patches should yield
donor_a1+a2 rather than a1+donor_a2; jswap_A1 should turn answers into
donor_a1+a2 far more often than jswap_A2 turns them into a1+donor_a2;
jswap_ctrl should leave answers at `clean`; jabl_A1 should cost accuracy
(toward the k=0 level) where jabl_A2 should not. If the dots are epiphenomenal
every condition stays at `clean`. Chance for "moved toward the donor" is how
often the CLEAN prediction already equals a donor-derived value.

Outputs (results/):
    patching_<tag>.csv        one row per (idx, condition, layer)
    patching_<tag>.png        A: accuracy vs patch layer + class mix;
                              B: accuracy and class mix per J-edit
    patching_summary_<tag>.json

    python 50_filler_patching.py --model deepseek --n 300 --k 10
"""
from __future__ import annotations

import argparse
import json
import math
import os
from contextlib import contextmanager
from typing import Sequence

import numpy as np
import pandas as pd

import paper_tasks as pt

LAYERS = (30, 35, 39)                # donor-patch layers (outputs of these blocks)
SELF_LAYER = 35                      # where the plumbing check patches
BAND = tuple(range(33, 41))          # J-edit band: where the operands sit on the dots
J_EDITS = ("jswap_A1", "jswap_A1x2", "jswap_A2", "jswap_ctrl",
           "jabl_A1", "jabl_A2", "jabl_A1A2")
CLASSES = ["target", "donor_target", "donor_a1+a2", "a1+donor_a2",
           "a1", "a2", "donor_a1", "donor_a2", "unparsed", "other"]


# --------------------------------------------------------------------------- #
# Patching mechanics
# --------------------------------------------------------------------------- #
def block_output(output):
    """The residual tensor in a decoder block's output (tensor, or tuple[0])."""
    return output if hasattr(output, "shape") else output[0]


def with_output(output, tensor):
    """Put a modified residual back in the block's output structure."""
    if hasattr(output, "shape"):
        return tensor
    return (tensor,) + tuple(output[1:])


def capture_residuals(hf, layers, input_ids, positions: Sequence[int]) -> dict:
    """{layer: residual[positions] at the output of `layer`} for one prompt,
    recorded during a prefill exactly like the one `generate` runs (use_cache
    on), so a self-patch is a faithful no-op. `positions` are negative."""
    import torch

    S = input_ids.shape[1]
    abs_pos = [p + S for p in positions]
    store, handles = {}, []
    for L in layers:
        def hook(module, inputs, output, L=L):
            h = block_output(output)
            store[L] = h[:, abs_pos].detach().clone()
        handles.append(hf.model.layers[L].register_forward_hook(hook))
    try:
        with torch.no_grad():
            hf(input_ids=input_ids, use_cache=True)
    finally:
        for h in handles:
            h.remove()
    return store


@contextmanager
def patched(hf, layer: int, positions: Sequence[int], replacement, seq_len: int):
    """While active, the residual at `positions` (negative, relative to a
    prompt of `seq_len` tokens) at the output of block `layer` is overwritten
    with `replacement` — during the prefill only: decode steps have a
    different sequence length and pass through untouched."""
    abs_pos = [p + seq_len for p in positions]

    def hook(module, inputs, output):
        h = block_output(output)
        if h.shape[1] != seq_len:          # a decode step, not the prefill
            return None
        h = h.clone()
        h[:, abs_pos] = replacement.to(h.device, h.dtype)
        return with_output(output, h)

    handle = hf.model.layers[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def greedy_answer(hf, tok, input_ids, max_new_tokens: int = 6) -> str:
    import torch
    with torch.no_grad():
        gen = hf.generate(input_ids=input_ids, max_new_tokens=max_new_tokens,
                          do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(gen[0, input_ids.shape[1]:], skip_special_tokens=False)


# --------------------------------------------------------------------------- #
# J-lens coordinate edits (swap / ablate a token's lens direction on the dots)
# --------------------------------------------------------------------------- #
def lens_vectors(hf, lens, layer: int, token_ids: Sequence[int]):
    """Rows of W_U diag(g) J_layer for `token_ids`: the residual-space
    direction at `layer` along which the J-lens reads each token. [n, d] fp32."""
    import torch

    W = hf.lm_head.weight
    if W.element_size() == 1:
        raise RuntimeError("lm_head is stored in an 8-bit format; lens_vectors needs "
                           "its rows in bf16/fp32 — dequantize it first")
    g = hf.model.norm.weight.float().to(W.device)
    rows = W[list(token_ids)].float() * g                   # [n, d]
    J = lens.jacobians[layer].to(W.device)                  # [d, d] fp32
    return rows @ J


def swap_fn(v_s, v_t, alpha: float = 1.0):
    """h -> h - 2a (h.u) u with u the unit bisector normal of v_s, v_t:
    exchanges the two tokens' lens coordinates, leaves the rest of h alone."""
    vs, vt = v_s / v_s.norm(), v_t / v_t.norm()
    u = vs - vt
    u = u / u.norm()

    def fn(h):
        uu = u.to(h.device)
        return (h.float() - 2 * alpha * (h.float() @ uu)[..., None] * uu).to(h.dtype)
    return fn


def ablate_fn(vectors):
    """h -> h with the span of `vectors` projected out (orthonormalized)."""
    import torch

    Q, _ = torch.linalg.qr(vectors.T)                        # [d, n] orthonormal

    def fn(h):
        q = Q.to(h.device)
        hf_ = h.float()
        return (hf_ - (hf_ @ q) @ q.T).to(h.dtype)
    return fn


@contextmanager
def edited(hf, band: Sequence[int], positions: Sequence[int], fn, seq_len: int):
    """Apply `fn` to the residual at `positions` (negative) at the output of
    every block in `band`, during the prefill only. On a hyper-connection
    model the residual is [B, S, streams, D]; `fn` acts on the last axis, so
    each stream is edited and the collapse (a weighted sum) inherits the edit."""
    abs_pos = [p + seq_len for p in positions]
    handles = []
    for layer in band:
        def hook(module, inputs, output):
            h = block_output(output)
            if h.shape[1] != seq_len:
                return None
            h = h.clone()
            h[:, abs_pos] = fn(h[:, abs_pos])
            return with_output(output, h)
        handles.append(hf.model.layers[layer].register_forward_hook(hook))
    try:
        yield
    finally:
        for hd in handles:
            hd.remove()


def numeral_id(tok, value: int) -> int:
    """The single token the model writes `value` with (exact decode mode)."""
    ids = tok.encode(str(value), add_special_tokens=False)
    assert len(ids) == 1, f"{value} is not a single token ({ids}); J edits need exact mode"
    return ids[0]


def control_pair(ex, donor, lo: int = 1, hi: int = 199):
    """Two numbers that appear nowhere in either prompt (deterministic)."""
    taken = {ex.a1, ex.a2, ex.target, donor.a1, donor.a2, donor.target}
    c1 = next(v for v in range(lo + (ex.idx * 7) % 50, hi) if v not in taken)
    c2 = next(v for v in range(c1 + 23, hi) if v not in taken)
    return c1, c2


def j_edit_fns(hf, lens, tok, ex, donor, band):
    """{condition: {layer: fn}} for every J edit, on the bare numeral tokens."""
    ids = {k: numeral_id(tok, v) for k, v in dict(
        a1=ex.a1, a2=ex.a2, d1=donor.a1, d2=donor.a2).items()}
    c1, c2 = control_pair(ex, donor)
    ids["c1"], ids["c2"] = numeral_id(tok, c1), numeral_id(tok, c2)
    out = {name: {} for name in J_EDITS}
    for L in band:
        V = lens_vectors(hf, lens, L, [ids[k] for k in ("a1", "a2", "d1", "d2", "c1", "c2")])
        v_a1, v_a2, v_d1, v_d2, v_c1, v_c2 = V
        out["jswap_A1"][L] = swap_fn(v_a1, v_d1, 1.0)
        out["jswap_A1x2"][L] = swap_fn(v_a1, v_d1, 2.0)
        out["jswap_A2"][L] = swap_fn(v_a2, v_d2, 1.0)
        out["jswap_ctrl"][L] = swap_fn(v_c1, v_c2, 1.0)
        out["jabl_A1"][L] = ablate_fn(V[[0]])
        out["jabl_A2"][L] = ablate_fn(V[[1]])
        out["jabl_A1A2"][L] = ablate_fn(V[[0, 1]])
    return out


@contextmanager
def edited_per_layer(hf, fns: dict, positions, seq_len):
    """`edited` with a different fn per layer ({layer: fn})."""
    from contextlib import ExitStack
    with ExitStack() as stack:
        for L, fn in fns.items():
            stack.enter_context(edited(hf, [L], positions, fn, seq_len))
        yield


# --------------------------------------------------------------------------- #
# Bookkeeping
# --------------------------------------------------------------------------- #
def pick_donor(dataset, i: int):
    """Another example with no element in common and a different sum."""
    ex = dataset[i][0]
    for step in range(1, len(dataset)):
        cand = dataset[(i + step) % len(dataset)][0]
        if ({cand.elem_a, cand.elem_b} & {ex.elem_a, ex.elem_b}) or cand.target == ex.target:
            continue
        return cand
    raise RuntimeError(f"no usable donor for example {ex.idx}")


def classify(pred, ex, donor) -> str:
    """Which quantity the prediction equals (first match in CLASSES order)."""
    if pred is None or (isinstance(pred, float) and math.isnan(pred)):
        return "unparsed"
    table = {
        "target": ex.target, "donor_target": donor.target,
        "donor_a1+a2": donor.a1 + ex.a2, "a1+donor_a2": ex.a1 + donor.a2,
        "a1": ex.a1, "a2": ex.a2, "donor_a1": donor.a1, "donor_a2": donor.a2,
    }
    for name, value in table.items():
        if pred == value:
            return name
    return "other"


def wilson(k: int, n: int, z: float = 1.96):
    """Wilson 95% interval for a proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


# --------------------------------------------------------------------------- #
# Analysis + figure
# --------------------------------------------------------------------------- #
def summarize(df: pd.DataFrame, clean_acc_k0=None) -> dict:
    """Per condition: accuracy with a Wilson CI, agreement with clean, and the
    prediction-class mix; plus the chance level for donor-derived classes."""
    out = {}
    clean = df[df.condition == "clean"].set_index("idx")
    for (cond, layer), g in df.groupby(["condition", "layer"], dropna=False):
        g = g.set_index("idx")
        n = len(g)
        acc = int(g.correct.sum())
        same = float((g.pred == clean.loc[g.index].pred).mean())
        mix = {c: round(float((g.cls == c).mean()), 3) for c in CLASSES}
        key = cond if (cond == "clean" or pd.isna(layer)) else f"{cond}@{int(layer)}"
        out[key] = dict(n=n, accuracy=round(acc / n, 3), ci=[round(x, 3) for x in wilson(acc, n)],
                        same_as_clean=round(same, 3), classes=mix)
    # chance for "moved toward the donor": how often the CLEAN prediction
    # already equals a donor-derived value
    donor_cls = ["donor_target", "donor_a1+a2", "a1+donor_a2", "donor_a1", "donor_a2"]
    out["chance_donor_derived"] = round(float(clean.cls.isin(donor_cls).mean()), 3)
    if clean_acc_k0 is not None:
        out["accuracy_k0_no_filler"] = round(float(clean_acc_k0), 3)
    return out


def plot(df: pd.DataFrame, summary: dict, tag: str, outdir: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = sorted(int(l) for l in df[df.condition == "donor"].layer.unique())
    acc = [summary[f"donor@{L}"]["accuracy"] for L in layers]
    lo = [summary[f"donor@{L}"]["ci"][0] for L in layers]
    hi = [summary[f"donor@{L}"]["ci"][1] for L in layers]

    jkeys = [k for k in J_EDITS if k in summary]
    fig, axes = plt.subplots(1, 3 if jkeys else 2, figsize=(18 if jkeys else 12, 4.8))
    ax = axes[0]
    ax.errorbar(layers, acc, yerr=[np.subtract(acc, lo), np.subtract(hi, acc)],
                marker="o", capsize=3, color="C3", label="dots patched from donor at layer L")
    ax.axhline(summary["clean"]["accuracy"], color="k", ls="-", lw=1.2, label="clean (no patch)")
    self_key = next(k for k in summary if k.startswith("self@"))
    ax.axhline(summary[self_key]["accuracy"], color="0.5", ls="--", lw=1,
               label=f"self-patch @{self_key.split('@')[1]} (plumbing check)")
    if "accuracy_k0_no_filler" in summary:
        ax.axhline(summary["accuracy_k0_no_filler"], color="C0", ls=":", lw=1.2,
                   label="k=0 (no filler at all)")
    ax.set_xlabel("patch layer L (output of block L)")
    ax.set_ylabel("greedy accuracy")
    ax.set_title("Accuracy when the dots carry another question's residual", fontsize=10)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    show = ["donor_a1+a2", "a1+donor_a2", "donor_target"]
    colors = {"donor_a1+a2": "C3", "a1+donor_a2": "C1", "donor_target": "C4"}
    for c in show:
        ax.plot(layers, [summary[f"donor@{L}"]["classes"][c] for L in layers],
                marker="o", color=colors[c], label=c)
    ax.plot(layers, [summary[f"donor@{L}"]["same_as_clean"] for L in layers],
            marker="s", color="k", label="prediction unchanged from clean")
    ax.axhline(summary["chance_donor_derived"], color="k", ls=":", lw=1,
               label="chance: clean pred already donor-derived")
    ax.set_xlabel("patch layer L")
    ax.set_ylabel("fraction of predictions")
    ax.set_title("Where the answers go (donor_a1+a2 = A1 read from the dots)", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    if jkeys:
        ax = axes[2]
        x = np.arange(len(jkeys))
        accs = [summary[k]["accuracy"] for k in jkeys]
        errs = [[summary[k]["accuracy"] - summary[k]["ci"][0] for k in jkeys],
                [summary[k]["ci"][1] - summary[k]["accuracy"] for k in jkeys]]
        ax.bar(x - 0.2, accs, 0.4, yerr=errs, capsize=3, color="0.4", label="accuracy")
        ax.bar(x + 0.2, [summary[k]["classes"]["donor_a1+a2"] for k in jkeys], 0.4,
               color="C3", label="pred = donor_a1+a2 (A1 replaced)")
        ax.bar(x + 0.2, [summary[k]["classes"]["a1+donor_a2"] for k in jkeys], 0.4,
               bottom=[summary[k]["classes"]["donor_a1+a2"] for k in jkeys],
               color="C1", label="pred = a1+donor_a2 (A2 replaced)")
        ax.axhline(summary["clean"]["accuracy"], color="k", lw=1.2, label="clean accuracy")
        if "accuracy_k0_no_filler" in summary:
            ax.axhline(summary["accuracy_k0_no_filler"], color="C0", ls=":", lw=1.2,
                       label="k=0 accuracy")
        ax.axhline(summary["chance_donor_derived"], color="k", ls=":", lw=1,
                   label="chance donor-derived")
        ax.set_xticks(x)
        ax.set_xticklabels(jkeys, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("fraction of examples")
        ax.set_title(f"J-lens edits on the dots, layers {min(BAND)}–{max(BAND)}", fontsize=10)
        ax.legend(frameon=False, fontsize=7)
    n = summary["clean"]["n"]
    fig.suptitle(f"Causal tests on the {tag} filler positions (n={n}, Wilson 95% CIs)")
    fig.tight_layout()
    path = os.path.join(outdir, f"patching_{tag}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek", help="registry key (config.py)")
    ap.add_argument("--filler", default="dots", choices=list(pt.FILLER_KINDS))
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--layers", type=int, nargs="+", default=list(LAYERS),
                    help="donor-patch layers")
    ap.add_argument("--band", type=int, nargs="+", default=list(BAND),
                    help="layers the J-lens swaps/ablations are applied at")
    ap.add_argument("--no-j-edits", action="store_true", help="position patching only")
    ap.add_argument("--fig2-path", default="data/fig2_2fact.jsonl")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    import torch
    import config
    from common import load_model, load_lens

    tag = f"{args.model}_{args.filler}-{args.k}"
    out_csv = os.path.join(args.outdir, f"patching_{tag}.csv")
    ans_csv = os.path.join(args.outdir, f"answers_{tag}.csv")
    ans_k0 = os.path.join(args.outdir, f"answers_{args.model}_{args.filler}-0.csv")

    dataset = pt.load_fig2_examples(args.fig2_path, k=args.k, n=args.n)
    print(f"{tag}: {len(dataset)} examples, donor-patch layers {args.layers}, "
          f"self-check @{SELF_LAYER}, J-edit band {args.band}")
    cached = pd.read_csv(ans_csv).set_index("idx") if os.path.exists(ans_csv) else None

    spec = config.get(args.model)
    model, hf, tok = load_model(spec)
    lens = load_lens(spec, kind="j")
    n_layers = len(hf.model.layers)
    assert all(0 <= L < n_layers - 1 for L in args.layers + [SELF_LAYER]), \
        f"patch layers must be below the last block ({n_layers - 1})"
    band = [L for L in args.band if L in lens.source_layers]
    assert band, f"none of the band layers {args.band} are in the lens ({lens.source_layers})"

    # resume: rows already on disk are kept
    rows = pd.read_csv(out_csv).to_dict("records") if os.path.exists(out_csv) else []
    done = {r["idx"] for r in rows}
    if done:
        print(f"resuming: {len(done)} examples already in {out_csv}")

    capture_layers = sorted(set(args.layers) | {SELF_LAYER})
    mismatch_with_cache = 0
    for i, (ex, msgs) in enumerate(dataset):
        if ex.idx in done:
            continue
        donor = pick_donor(dataset, i)
        d_msgs = next(m for e, m in dataset if e.idx == donor.idx)

        text = pt.render_chat(tok, msgs)
        d_text = pt.render_chat(tok, d_msgs)
        pos, nf = pt.readout_positions(tok, text, ex, args.filler, args.k)
        d_pos, d_nf = pt.readout_positions(tok, d_text, donor, args.filler, args.k)
        dots, d_dots = pos[:nf], d_pos[:d_nf]
        assert len(dots) == len(d_dots) == args.k, (
            f"idx {ex.idx}: {len(dots)} dot tokens vs donor {len(d_dots)} — the "
            f"filler must tokenize to one token per dot for a position-wise patch")

        ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids.to(hf.device)
        d_ids = tok(d_text, add_special_tokens=False, return_tensors="pt").input_ids.to(hf.device)
        S = ids.shape[1]

        own = capture_residuals(hf, [SELF_LAYER], ids, dots)
        theirs = capture_residuals(hf, capture_layers, d_ids, d_dots)

        def record(condition, layer, reply):
            pred = pt.parse_answer(reply)
            rows.append(dict(idx=ex.idx, condition=condition, layer=layer,
                             a1=ex.a1, a2=ex.a2, target=ex.target,
                             donor_idx=donor.idx, donor_a1=donor.a1, donor_a2=donor.a2,
                             donor_target=donor.target, reply=reply.strip()[:32],
                             pred=pred, correct=pred == ex.target,
                             cls=classify(pred, ex, donor)))
            return pred

        clean_pred = record("clean", np.nan, greedy_answer(hf, tok, ids))
        if cached is not None and ex.idx in cached.index and cached.pred[ex.idx] != clean_pred:
            mismatch_with_cache += 1
        with patched(hf, SELF_LAYER, dots, own[SELF_LAYER], S):
            record("self", SELF_LAYER, greedy_answer(hf, tok, ids))
        for L in args.layers:
            with patched(hf, L, dots, theirs[L], S):
                record("donor", L, greedy_answer(hf, tok, ids))
        if not args.no_j_edits:
            fns = j_edit_fns(hf, lens, tok, ex, donor, band)
            for name in J_EDITS:
                with edited_per_layer(hf, fns[name], dots, S):
                    record(name, np.nan, greedy_answer(hf, tok, ids))

        if (i + 1) % 10 == 0 or i == 0:
            df = pd.DataFrame(rows)
            df.to_csv(out_csv, index=False)
            done_n = df.idx.nunique()
            acc = df[df.condition == "clean"].correct.mean()
            self_ok = (df[df.condition == "self"].set_index("idx").pred
                       == df[df.condition == "clean"].set_index("idx").pred).mean()
            late = df[(df.condition == "donor") & (df.layer == max(args.layers))]
            sw = df[df.condition == "jswap_A1"]
            ct = df[df.condition == "jswap_ctrl"]
            print(f"[{done_n}/{len(dataset)}] clean acc {acc:.2%} | self-patch = clean on "
                  f"{self_ok:.0%} | donor@{max(args.layers)} acc {late.correct.mean():.2%}, "
                  f"donor_a1+a2 {(late.cls == 'donor_a1+a2').mean():.0%} | "
                  + (f"jswap_A1 -> donor_a1+a2 {(sw.cls == 'donor_a1+a2').mean():.0%}, "
                     f"ctrl swap acc {ct.correct.mean():.2%} | " if len(sw) else "")
                  + f"clean≠cached answers: {mismatch_with_cache}")

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv} ({len(df)} rows, {df.idx.nunique()} examples)")

    k0_acc = pd.read_csv(ans_k0).correct.mean() if os.path.exists(ans_k0) else None
    summary = summarize(df, k0_acc)
    js = os.path.join(args.outdir, f"patching_summary_{tag}.json")
    with open(js, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {js}\n")
    for key, s in summary.items():
        if isinstance(s, dict):
            top = {c: v for c, v in s["classes"].items() if v >= 0.05}
            print(f"{key:>10}: acc {s['accuracy']:.3f} {s['ci']} | same as clean "
                  f"{s['same_as_clean']:.2f} | {top}")
    print(f"chance (clean pred already donor-derived): {summary['chance_donor_derived']}")
    plot(df, summary, tag, args.outdir)


if __name__ == "__main__":
    main()

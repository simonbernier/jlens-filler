"""
Mini experiment: does the model compute the answer *in the filler region*, and
does the Jacobian lens surface it differently from the logit-lens baseline?

Task: 2-fact addition ("atomic number of X plus atomic number of Y") with dot
filler between the question and the answer slot. For every filler position and
every lens source-layer we ask: what is the rank of the correct sum (A1+A2)
among the vocabulary? We do this for the Jacobian lens and the logit-lens
baseline and compare.

Outputs (in ./results/):
  - filler_ranks.csv      : pair, filler_pos, layer, lens, quantity, rank
  - heatmap_<pair>.png    : layers x filler-positions decode strength for the SUM,
                            J-lens vs logit-lens side by side (paper Fig. 3/4 style)
  - a printed summary comparing the two lenses

Run:
    python 02_filler_experiment.py                 # dev model, dots, k=10
    python 02_filler_experiment.py --k 25 --filler counting
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from common import load_model, load_lens, check_provenance, apply_lens, numeric_token_rank
from filler_tasks import default_prompts


def filler_positions(tok, fp):
    """Negative token indices of the filler region (robust to a leading BOS)."""
    n_post = len(tok.encode(fp.post, add_special_tokens=False))
    n_fill = len(tok.encode(fp.filler, add_special_tokens=False))
    # filler sits just before the post tail; index from the end.
    return list(range(-(n_post + n_fill), -n_post)), n_fill


def decode_strength(rank):
    """Map a vocab rank to a 0..1 strength for the heatmap (1 = top token)."""
    return 1.0 / (1.0 + rank)


def run_pair(lens, model, tok, fp):
    """Return a tidy DataFrame of ranks for A1, A2, target across positions/layers/lenses."""
    pos, n_fill = filler_positions(tok, fp)
    quantities = {"A1": fp.a1, "A2": fp.a2, "sum": fp.target}

    rows = []
    grids = {}  # (lens_name) -> [n_layers, n_positions] strength for the SUM
    for lens_name, use_j in [("jlens", True), ("logit", False)]:
        out = apply_lens(lens, model, fp.text, pos, use_jacobian=use_j)
        layers = sorted(out.keys())
        grid = np.zeros((len(layers), len(pos)))
        for li, L in enumerate(layers):
            logits = out[L]  # [num_positions, vocab]
            for pi in range(len(pos)):
                row = logits[pi]
                for qname, qval in quantities.items():
                    r = numeric_token_rank(row, tok, qval)
                    rows.append(dict(pair=f"{fp.meta['elem_a']}+{fp.meta['elem_b']}",
                                     filler_pos=pi, layer=int(L), lens=lens_name,
                                     quantity=qname, rank=r))
                    if qname == "sum":
                        grid[li, pi] = decode_strength(r)
        grids[lens_name] = (layers, grid)
    return pd.DataFrame(rows), grids, n_fill


def plot_pair(fp, grids, out_png):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, name in zip(axes, ["jlens", "logit"]):
        layers, grid = grids[name]
        im = ax.imshow(grid, aspect="auto", origin="lower", vmin=0, vmax=1, cmap="magma")
        ax.set_title(f"{name}  —  P(sum={fp.target} is top token)")
        ax.set_xlabel("filler position")
        ax.set_yticks(range(0, len(layers), max(1, len(layers) // 8)))
        ax.set_yticklabels([layers[i] for i in range(0, len(layers), max(1, len(layers) // 8))])
    axes[0].set_ylabel("source layer")
    fig.colorbar(im, ax=axes, fraction=0.046, pad=0.02, label="decode strength")
    fig.suptitle(f"2-fact addition: {fp.meta['elem_a']} + {fp.meta['elem_b']} "
                 f"= {fp.a1}+{fp.a2}={fp.target}   (filler={fp.meta['filler_kind']}, k={fp.meta['k']})")
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)


def summarize(df):
    """Compare, per lens, how strongly the SUM is decoded anywhere in the filler."""
    print("\n================ SUMMARY (lower rank = better) ================")
    sums = df[df.quantity == "sum"]
    for lens_name, g in sums.groupby("lens"):
        best = g.groupby("pair")["rank"].min()          # best (position,layer) per pair
        top1 = (best == 0).mean()
        print(f"  {lens_name:6}: sum is TOP token in filler for "
              f"{top1*100:4.0f}% of pairs | median best-rank {best.median():.0f}")
    # earliest layer where the sum becomes top-1, per lens (a proxy for 'crystallizes')
    print("  ---- where the sum first becomes the top token (median layer over pairs) ----")
    hits = sums[sums["rank"] == 0]
    if len(hits):
        for lens_name, g in hits.groupby("lens"):
            earliest = g.groupby("pair")["layer"].min()
            print(f"  {lens_name:6}: median earliest layer {earliest.median():.0f}")
    else:
        print("  (sum never reached top-1 on this small model — try --k 25 or --model dev-9b)")
    print("===============================================================\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dev")
    ap.add_argument("--filler", default="dots", choices=["dots", "counting", "alphabet"])
    ap.add_argument("--k", type=int, default=10, help="number of filler tokens")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    spec = config.get(args.model)
    check_provenance(spec)
    model, hf, tok = load_model(spec)
    lens = load_lens(spec, kind="j")

    prompts = default_prompts(filler_kind=args.filler, k=args.k)
    all_df = []
    for i, fp in enumerate(prompts):
        print(f"[{i+1}/{len(prompts)}] {fp.meta['elem_a']}+{fp.meta['elem_b']} "
              f"= {fp.target}   ({fp.text!r})")
        df, grids, n_fill = run_pair(lens, model, tok, fp)
        all_df.append(df)
        png = os.path.join(args.outdir, f"heatmap_{fp.meta['elem_a']}_{fp.meta['elem_b']}.png")
        plot_pair(fp, grids, png)
        print(f"      wrote {png}  (filler tokens located: {n_fill})")

    df = pd.concat(all_df, ignore_index=True)
    csv = os.path.join(args.outdir, "filler_ranks.csv")
    df.to_csv(csv, index=False)
    print(f"\nwrote {csv}  ({len(df)} rows)")
    summarize(df)


if __name__ == "__main__":
    main()

"""
Analysis of 04_lens_readout.py output: Figure-3-style heatmaps, the
J-lens vs logit-lens comparison, and the "what algorithm is this?" summary.

Decode criterion (paper Sec. 4.2): a quantity counts as decoded at
(layer, position) when the TOP NUMERIC token equals its ground-truth value.
Heatmap color = fraction of examples decoded, aggregated separately over
correct and wrong examples.

Outputs (results/):
    fig3_<tag>_<lens>.png        heatmaps: correct/wrong x A1/A2/sum (per lens)
    jlens_vs_logit_<tag>.png     difference maps + per-layer decode curves
    algorithm_summary_<tag>.json + printed report

Run:
    python 05_analyze_lens.py --readout results/lens_readout_deepseek_dots-10.csv
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

QUANTITIES = ["A1", "A2", "sum"]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load(readout_csv: str) -> pd.DataFrame:
    df = pd.read_csv(readout_csv)
    ans_csv = readout_csv.replace("lens_readout_", "answers_")
    if os.path.exists(ans_csv):
        ans = pd.read_csv(ans_csv)[["idx", "a1", "a2", "target"]]
        df = df.merge(ans, on="idx", how="left")
    else:
        raise SystemExit(f"companion answers file not found: {ans_csv}")
    df["match_A1"] = df.top_num == df.a1
    df["match_A2"] = df.top_num == df.a2
    df["match_sum"] = df.top_num == df.target
    return df


def grid(df: pd.DataFrame, col: str):
    """(layers, positions, matrix[layer, pos]) of mean(col)."""
    p = df.pivot_table(index="layer", columns="pos", values=col, aggfunc="mean")
    p = p.sort_index().sort_index(axis=1)
    return p.index.to_numpy(), p.columns.to_numpy(), p.to_numpy()


# --------------------------------------------------------------------------- #
# Figure 3 replication (one figure per lens)
# --------------------------------------------------------------------------- #
def fig3(df: pd.DataFrame, lens: str, n_filler: int, tag: str, outdir: str):
    sub = df[df.lens == lens]
    n_cor = sub[sub.correct].idx.nunique()
    n_wrg = sub[~sub.correct].idx.nunique()
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), sharex=True, sharey=True)
    im = None
    for r, (label, mask, n_ex) in enumerate(
            [("correct", sub.correct, n_cor), ("wrong", ~sub.correct, n_wrg)]):
        for c, q in enumerate(QUANTITIES):
            ax = axes[r, c]
            if n_ex == 0:
                ax.set_axis_off()
                ax.set_title(f"{q} — no {label} examples")
                continue
            layers, poss, g = grid(sub[mask], f"match_{q}")
            im = ax.imshow(g, aspect="auto", origin="lower", vmin=0, vmax=1,
                           cmap="magma",
                           extent=[poss[0] - .5, poss[-1] + .5,
                                   layers[0] - .5, layers[-1] + .5])
            ax.axvline(n_filler - 0.5, color="w", ls="--", lw=1)
            ax.set_title(f"{q}  ({label}, n={n_ex})", fontsize=10)
            if r == 1:
                ax.set_xlabel("position (filler → | answer)")
            if c == 0:
                ax.set_ylabel("source layer")
    if im is not None:
        fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02,
                     label="fraction of examples decoded (top numeric token)")
    fig.suptitle(f"{lens} — 2-fact addition, {tag}")
    path = os.path.join(outdir, f"fig3_{tag}_{lens}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


# --------------------------------------------------------------------------- #
# J-lens vs logit-lens
# --------------------------------------------------------------------------- #
def compare_lenses(df: pd.DataFrame, n_filler: int, tag: str, outdir: str):
    cor = df[df.correct]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    for c, q in enumerate(QUANTITIES):
        layers_j, poss, gj = grid(cor[cor.lens == "jlens"], f"match_{q}")
        layers_l, _, gl = grid(cor[cor.lens == "logit"], f"match_{q}")
        ax = axes[0, c]
        vmax = max(np.abs(gj - gl).max(), 1e-9)
        im = ax.imshow(gj - gl, aspect="auto", origin="lower", cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax,
                       extent=[poss[0] - .5, poss[-1] + .5,
                               layers_j[0] - .5, layers_j[-1] + .5])
        ax.axvline(n_filler - 0.5, color="k", ls="--", lw=1)
        ax.set_title(f"{q}: J-lens − logit-lens", fontsize=10)
        ax.set_xlabel("position")
        if c == 0:
            ax.set_ylabel("source layer")
        fig.colorbar(im, ax=ax, fraction=0.046)
        # per-layer curves: best-over-positions decode fraction
        ax2 = axes[1, c]
        for lens, g, layers in (("jlens", gj, layers_j), ("logit", gl, layers_l)):
            ax2.plot(layers, np.nanmax(g, axis=1), marker=".", label=lens)
        ax2.set_xlabel("source layer")
        ax2.set_title(f"{q}: max decode fraction over positions", fontsize=10)
        if c == 0:
            ax2.set_ylabel("decode fraction")
            ax2.legend(frameon=False)
    fig.suptitle(f"J-lens vs logit-lens (correct examples) — {tag}")
    fig.tight_layout()
    path = os.path.join(outdir, f"jlens_vs_logit_{tag}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


# --------------------------------------------------------------------------- #
# "What algorithm?" summary
# --------------------------------------------------------------------------- #
def first_true(series_by_key):
    """min key where value True, else None."""
    hit = [k for k, v in series_by_key.items() if v]
    return min(hit) if hit else None


def algorithm_summary(df: pd.DataFrame, n_filler: int) -> dict:
    out = {}
    fill = df[(df.pos < n_filler) & df.correct]
    post = df[(df.pos >= n_filler) & df.correct]
    for lens in ("jlens", "logit"):
        s = {}
        fl = fill[fill.lens == lens]
        po = post[post.lens == lens]
        # 1. Does each quantity ever decode in the filler region?
        for q in QUANTITIES:
            per_ex = fl.groupby("idx")[f"match_{q}"].any()
            s[f"{q}_decoded_in_filler_frac"] = round(float(per_ex.mean()), 3)
            # earliest layer at which it first decodes (median over examples)
            firsts = (fl[fl[f"match_{q}"]].groupby("idx")["layer"].min())
            s[f"{q}_first_layer_median"] = (
                float(firsts.median()) if len(firsts) else None)
            # position center-of-mass (where in the filler it lives)
            m = fl[fl[f"match_{q}"]]
            s[f"{q}_mean_position"] = (
                round(float(m.pos.mean()), 2) if len(m) else None)
        # 2. sum in filler vs at the answer tail
        s["sum_decoded_in_post_frac"] = round(
            float(po.groupby("idx")["match_sum"].any().mean()), 3) if len(po) else None
        # 3. parallel retrieval: same example, same layer, A1 and A2
        #    simultaneously decoded at DIFFERENT positions
        both = (fl.groupby(["idx", "layer"])
                  .agg(a1_any=("match_A1", "any"), a2_any=("match_A2", "any")))
        par_ex = both[both.a1_any & both.a2_any].reset_index().idx.nunique()
        n_ex = fl.idx.nunique()
        s["parallel_A1_A2_same_layer_frac"] = (
            round(par_ex / n_ex, 3) if n_ex else None)
        out[lens] = s
    # 4. wrong-example signature: retrieval without composition
    wf = df[(df.pos < n_filler) & (~df.correct) & (df.lens == "logit")]
    if wf.idx.nunique():
        out["wrong_examples_logit"] = {
            q: round(float(wf.groupby("idx")[f"match_{q}"].any().mean()), 3)
            for q in QUANTITIES}
    return out


def print_report(s: dict):
    print("\n=================== WHAT ALGORITHM? ===================")
    for lens in ("jlens", "logit"):
        d = s[lens]
        print(f"\n--- {lens} (correct examples) ---")
        for q in QUANTITIES:
            print(f"  {q:>4}: decoded-in-filler {d[f'{q}_decoded_in_filler_frac']:>6} | "
                  f"first layer (median) {d[f'{q}_first_layer_median']} | "
                  f"mean position {d[f'{q}_mean_position']}")
        print(f"  sum decoded in post/answer tail: {d['sum_decoded_in_post_frac']}")
        print(f"  A1 & A2 co-decoded at one layer (parallel retrieval): "
              f"{d['parallel_A1_A2_same_layer_frac']}")
    if "wrong_examples_logit" in s:
        print("\n--- wrong examples (logit lens) ---")
        print("  ", s["wrong_examples_logit"],
              "  <- paper's signature: A1/A2 present, sum absent")
    print("\nReading guide: retrieval-then-composition = A1/A2 first-layer << "
          "sum first-layer; position specialization = mean_position(A1) < "
          "mean_position(A2); J-lens 'sees more' = higher decoded-in-filler "
          "fractions and/or smaller first-layer at matched positions.")
    print("=======================================================\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--readout", required=True,
                    help="results/lens_readout_<tag>.csv from 04")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    outdir = args.outdir or os.path.dirname(args.readout) or "."
    tag = (os.path.basename(args.readout)
           .replace("lens_readout_", "").replace(".csv", ""))
    df = load(args.readout)
    n_filler = int(df[df.pos_type == "filler"].pos.max()) + 1
    print(f"{df.idx.nunique()} examples | {df.layer.nunique()} layers | "
          f"{df.pos.nunique()} positions ({n_filler} filler) | "
          f"accuracy {df.groupby('idx')['correct'].first().mean():.2%}")

    for lens in ("jlens", "logit"):
        fig3(df, lens, n_filler, tag, outdir)
    compare_lenses(df, n_filler, tag, outdir)

    summary = algorithm_summary(df, n_filler)
    js = os.path.join(outdir, f"algorithm_summary_{tag}.json")
    with open(js, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {js}")
    print_report(summary)


if __name__ == "__main__":
    main()

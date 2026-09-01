"""
Stage 2 analysis — Figure-3-style heatmaps + the "what algorithm?" summary,
for whichever lens(es) a readout CSV contains.

Stage 2 is the paper replication proper: run 20 with --lens logit, then this
script reproduces the paper's logit-lens picture on DeepSeek V4 Flash
(decode-fraction heatmaps over layer x position, correct vs wrong examples,
A1/A2/sum). It works identically on a jlens or both readout — the J-lens vs
logit-lens COMPARISON lives in 30_compare_lenses.py.

Outputs (results/):
    fig3_<tag>_<lens>.png        heatmaps: correct/wrong x A1/A2/sum (per lens)
    algorithm_summary_<tag>_<lens...>.json + printed report

Run:
    python 21_analyze_readout.py --readout results/lens_readout_deepseek_dots-10_logit.csv
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lens_analysis import (QUANTITIES, algorithm_summary, grid, load,
                           n_filler_of, print_report, tag_of)


def fig3(df, lens: str, n_filler: int, tag: str, outdir: str):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--readout", required=True, nargs="+",
                    help="results/lens_readout_<tag>_<lens>.csv from 20 "
                         "(several files of the same tag are merged)")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    outdir = args.outdir or os.path.dirname(args.readout[0]) or "."
    tag = tag_of(args.readout[0])
    df = load(args.readout)
    n_filler = n_filler_of(df)
    lenses = sorted(df.lens.unique())
    print(f"{df.idx.nunique()} examples | {df.layer.nunique()} layers | "
          f"{df.pos.nunique()} positions ({n_filler} filler) | lenses: {lenses} | "
          f"accuracy {df.groupby('idx')['correct'].first().mean():.2%}")

    for lens in lenses:
        fig3(df, lens, n_filler, tag, outdir)

    summary = algorithm_summary(df, n_filler)
    js = os.path.join(outdir, f"algorithm_summary_{tag}_{'-'.join(lenses)}.json")
    with open(js, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {js}")
    print_report(summary)


if __name__ == "__main__":
    main()

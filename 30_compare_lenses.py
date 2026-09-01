"""
Stage 3 — the J-lens vs logit-lens comparison (the new result over the paper).

Takes readout CSVs from 20 that together cover BOTH lenses for one condition
(either one `--lens both` file, or the stage-2 logit file plus a jlens file),
and asks where the Jacobian lens sees more than the paper's logit-lens readout:

  * difference maps (J-lens − logit-lens decode fraction) per quantity;
  * per-layer curves of the best-over-positions decode fraction, per lens —
    a leftward J-lens shift = earlier crystallization than the logit lens
    can read (see ANALYSIS.md for what each outcome would mean);
  * a side-by-side algorithm summary (first-layer medians, in-filler fractions).

Outputs (results/):
    jlens_vs_logit_<tag>.png
    compare_summary_<tag>.json + printed report

Run:
    python 30_compare_lenses.py --readout results/lens_readout_deepseek_dots-10_logit.csv \
                                          results/lens_readout_deepseek_dots-10_jlens.csv
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lens_analysis import (QUANTITIES, algorithm_summary, grid, load,
                           n_filler_of, print_report, tag_of)


def compare_lenses(df, n_filler: int, tag: str, outdir: str):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--readout", required=True, nargs="+",
                    help="readout CSV(s) from 20 covering both lenses for one "
                         "condition (one --lens both file, or logit + jlens files)")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    outdir = args.outdir or os.path.dirname(args.readout[0]) or "."
    tag = tag_of(args.readout[0])
    df = load(args.readout)
    lenses = set(df.lens.unique())
    assert {"jlens", "logit"} <= lenses, (
        f"need both lenses, found {sorted(lenses)} — run 20 with --lens jlens "
        f"(or --lens both) and pass both CSVs")
    n_filler = n_filler_of(df)
    print(f"{df.idx.nunique()} examples | {df.layer.nunique()} layers | "
          f"{df.pos.nunique()} positions ({n_filler} filler) | "
          f"accuracy {df.groupby('idx')['correct'].first().mean():.2%}")

    compare_lenses(df, n_filler, tag, outdir)

    summary = algorithm_summary(df, n_filler)
    js = os.path.join(outdir, f"compare_summary_{tag}.json")
    with open(js, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {js}")
    print_report(summary)


if __name__ == "__main__":
    main()
